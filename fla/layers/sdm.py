# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedLayerNormGated
from fla.ops.sparse_delta_rule import product_key_topk
from fla.ops.sparse_delta_rule.chunk import _chunk_sparse_delta_rule
from fla.ops.sparse_delta_rule.fused_recurrent import _fused_recurrent_sparse_delta_rule

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


class SparseDeltaMemory(nn.Module):
    """
    Sparse Delta Memory (SDM) layer.

    Each head owns a bank of `prod(product_key_dims)` slots of `head_dim` values, starting from a learned bank.
    Per token and head, product keys select `num_writes` slots to write and `num_reads` slots to read, weighted
    by a softmax over their scores. The written slots decay and take a delta-rule update
    (see :func:`fla.ops.sparse_delta_rule.chunk_sparse_delta_rule`); the readings are layer-normalized
    and gated per head. Training and long inputs run the chunked kernel, which keeps the bank in its dtype and rounds
    each written row once per chunk; short inputs and decoding run the recurrent kernel on a float32 bank. As with any
    top-k routing, cached decoding and a full forward can also select different slots where scores nearly tie in half
    precision.

    Each sequence of the batch carries one bank, never reset inside it: the layer ignores `cu_seqlens` (documents
    packed into one sequence), so the bank carries across document boundaries, since a reset per document would need
    a copy of the whole bank per document. Document masking can still be applied by the model's other layers
    (sliding-window attention, GDN, ...), which receive `cu_seqlens` unchanged. Padding (`attention_mask`) is
    supported: padded tokens are removed before the op, and their outputs are zero.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        product_key_dims (tuple[int, ...], Optional):
            The widths of the product-key factors, whose product is the number of slots per head.
            Default: `(16, 16, 16, 16)`.
        num_writes (int, Optional):
            The number of slots each token writes per head. Default: 64.
        num_reads (int, Optional):
            The number of slots each token reads per head. Default: 64.
        mode (str, Optional):
            Which kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        norm_eps (float, Optional):
            The epsilon value for the normalization of the readings. Default: 1e-6.
        layer_idx (int, Optional):
            The index of the layer. Default: `None`.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        product_key_dims: tuple[int, ...] = (16, 16, 16, 16),
        num_writes: int = 64,
        num_reads: int = 64,
        mode: str = 'chunk',
        norm_eps: float = 1e-6,
        layer_idx: int = None,
        **kwargs,
    ) -> SparseDeltaMemory:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.product_key_dims = tuple(product_key_dims)
        self.num_slots = math.prod(self.product_key_dims)
        self.num_writes = num_writes
        self.num_reads = num_reads
        self.mode = mode
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}.")
        if not 1 <= min(num_writes, num_reads) <= max(num_writes, num_reads) <= min(self.num_slots, 2**14):
            raise ValueError(f"num_writes and num_reads must be in [1, {min(self.num_slots, 2**14)}].")

        key_dim = num_heads * sum(self.product_key_dims)
        self.q_proj = nn.Linear(hidden_size, key_dim)
        self.k_proj = nn.Linear(hidden_size, key_dim)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.a_proj = nn.Linear(hidden_size, num_heads)
        self.b_proj = nn.Linear(hidden_size, num_heads)

        A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16).clamp_(min=1e-4)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # the learned initial bank of every sequence
        self.memory = nn.Parameter(torch.randn(num_heads, self.num_slots, self.head_dim) * hidden_size ** -0.5)

        self.g_proj = nn.Linear(hidden_size, hidden_size)
        self.o_norm = FusedLayerNormGated(self.head_dim, bias=True, activation='sigmoid', eps=norm_eps)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        if torch.is_grad_enabled():
            mode = 'chunk'
        elif q_len <= 64 and not self.training:
            mode = 'fused_recurrent'
        else:
            mode = self.mode

        last_state = get_layer_cache(self, past_key_values)

        # padding only: the rows of the batch, one bank each. `cu_seqlens` in kwargs (packed documents) is left to the
        # other layers: the bank carries across documents
        if attention_mask is not None:
            if attention_mask.shape[0] != batch_size or attention_mask.shape[1] < q_len:
                raise ValueError(
                    f"`attention_mask` is expected to be of shape [{batch_size}, >= {q_len}] "
                    f"rather than {list(attention_mask.shape)}.",
                )
            attention_mask = attention_mask[:, attention_mask.shape[1] - q_len:]
        hidden_states, indices, cu_seqlens = unpad_hidden_states(hidden_states, None, attention_mask, q_len)

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
        q_val, q_idx = product_key_topk(s=q, dims=self.product_key_dims, k=self.num_reads)
        k_val, k_idx = product_key_topk(s=k, dims=self.product_key_dims, k=self.num_writes)
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)
        beta = self.b_proj(hidden_states).sigmoid()

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        # decoding advances a cached float32 bank in place unless autograd holds it; any other bank is copied first
        inplace = (mode == 'fused_recurrent' and bool(use_cache) and recurrent_state is not None
                   and recurrent_state.dtype == torch.float32 and not recurrent_state.requires_grad)
        if recurrent_state is None:
            recurrent_state = self.memory.expand(batch_size, -1, -1, -1)
        # the private entry points take the unpadded rows of a padded batch, one bank each (no per-document resets)
        sparse_delta_rule = _chunk_sparse_delta_rule
        if mode == 'fused_recurrent':
            sparse_delta_rule = partial(_fused_recurrent_sparse_delta_rule, inplace_final_state=inplace)
        o, recurrent_state = sparse_delta_rule(
            q_idx=q_idx,
            q_val=q_val.softmax(-1),
            k_idx=k_idx,
            k_val=k_val.softmax(-1),
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        update_layer_cache(self, past_key_values, recurrent_state=recurrent_state, offset=q_len)

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim))
        o = self.o_proj(rearrange(o, 'b t h d -> b t (h d)'))
        o = repad_hidden_states(o, indices, batch_size, q_len)

        return o, None, past_key_values
