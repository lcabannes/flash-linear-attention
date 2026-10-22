# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.sparse_delta_rule.chunk import check_initial_state
from fla.ops.utils.op import exp
from fla.utils import input_guard


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_sparse_delta_rule_fwd_kernel(
    q_idx,
    q_val,
    k_idx,
    k_val,
    v,
    g,
    beta,
    o,
    h,
    cu_seqlens,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    V: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
    BR: tl.constexpr,
    BW: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0).to(tl.int64) % NV, tl.program_id(0).to(tl.int64) // NV
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T

    o_r = tl.arange(0, BR)
    o_w = tl.arange(0, BW)
    o_v = i_v * BV + tl.arange(0, BV)
    m_r = o_r < R
    m_w = o_w < W
    m_v = o_v < V
    # this program owns columns `o_v` of the bank of sequence `i_n` and head `i_h`
    p_h = h + i_nh * S * V
    for i_t in range(bos, bos + T):
        b_kidx = tl.load(k_idx + (i_t * H + i_h) * W + o_w, mask=m_w, other=0).to(tl.int64)
        b_kval = tl.load(k_val + (i_t * H + i_h) * W + o_w, mask=m_w, other=0).to(tl.float32)
        b_v = tl.load(v + (i_t * H + i_h) * V + o_v, mask=m_v, other=0).to(tl.float32)
        b_g = tl.load(g + i_t * H + i_h).to(tl.float32)
        b_beta = tl.load(beta + i_t * H + i_h).to(tl.float32)

        # [BW, BV]
        p_hk = p_h + b_kidx[:, None] * V + o_v[None, :]
        m_hk = m_w[:, None] & m_v[None, :]
        b_h = tl.load(p_hk, mask=m_hk, other=0).to(tl.float32) * exp(b_g)
        b_v = b_beta * (b_v - tl.sum(b_kval[:, None] * b_h, 0))
        b_h += b_kval[:, None] * b_v[None, :]
        tl.store(p_hk, b_h.to(p_hk.dtype.element_ty), mask=m_hk)
        # the read below and the next token's write touch rows stored by other threads
        tl.debug_barrier()

        b_qidx = tl.load(q_idx + (i_t * H + i_h) * R + o_r, mask=m_r, other=0).to(tl.int64)
        b_qval = tl.load(q_val + (i_t * H + i_h) * R + o_r, mask=m_r, other=0).to(tl.float32)
        # [BR, BV]
        b_hq = tl.load(p_h + b_qidx[:, None] * V + o_v[None, :], mask=m_r[:, None] & m_v[None, :], other=0).to(tl.float32)
        b_o = tl.sum(b_qval[:, None] * b_hq, 0)
        tl.store(o + (i_t * H + i_h) * V + o_v, b_o.to(o.dtype.element_ty), mask=m_v)
        tl.debug_barrier()


def fused_recurrent_sparse_delta_rule_fwd(
    q_idx: torch.Tensor,
    q_val: torch.Tensor,
    k_idx: torch.Tensor,
    k_val: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    inplace_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, V = v.shape
    N, _, S, _ = initial_state.shape
    R, W = q_idx.shape[-1], k_idx.shape[-1]
    BV = min(32, triton.next_power_of_2(V))

    o = torch.empty_like(v)
    # the kernel advances the bank in place: `initial_state` itself, or a float32 copy
    h = initial_state
    if not inplace_final_state:
        h = initial_state.to(torch.float32, memory_format=torch.contiguous_format, copy=True)
    grid = (triton.cdiv(V, BV) * N * H,)
    fused_recurrent_sparse_delta_rule_fwd_kernel[grid](
        q_idx=q_idx,
        q_val=q_val,
        k_idx=k_idx,
        k_val=k_val,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h=h,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        S=S,
        V=V,
        R=R,
        W=W,
        BR=triton.next_power_of_2(R),
        BW=triton.next_power_of_2(W),
        BV=BV,
        NV=triton.cdiv(V, BV),
        num_warps=1,
    )
    return o, h


def fused_recurrent_sparse_delta_rule(
    q_idx: torch.Tensor,
    q_val: torch.Tensor,
    k_idx: torch.Tensor,
    k_val: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool = False,
    inplace_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Token-by-token sparse delta rule (forward only), for decoding and short inputs.
    See :func:`chunk_sparse_delta_rule` for the other arguments and the recurrence.

    Args:
        inplace_final_state (bool, Optional):
            Whether to advance `initial_state` itself, in its dtype, rather than a float32 copy: decoding then never
            copies the bank. `initial_state` must then be contiguous, and not a tensor that autograd saved for a
            backward: the update bypasses autograd's version checks. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final bank of shape `[B, H, S, V]` if `output_final_state=True` else `None`: float32, or `initial_state`
            itself with `inplace_final_state=True`.
    """
    return _fused_recurrent_sparse_delta_rule(q_idx, q_val, k_idx, k_val, v, g, beta, initial_state, output_final_state,
                                              inplace_final_state)


@input_guard(no_guard_contiguous=['initial_state'])
def _fused_recurrent_sparse_delta_rule(q_idx, q_val, k_idx, k_val, v, g, beta, initial_state, output_final_state=False,
                                       inplace_final_state=False, cu_seqlens=None):
    # `cu_seqlens` is internal: the rows of a padded batch with their padding removed, one bank each
    if cu_seqlens is not None and v.shape[0] != 1:
        raise ValueError(f"The batch size is expected to be 1 rather than {v.shape[0]} when using `cu_seqlens`.")
    check_initial_state(initial_state, v, cu_seqlens)
    if inplace_final_state and not initial_state.is_contiguous():
        raise ValueError("`inplace_final_state=True` requires a contiguous `initial_state`.")
    o, final_state = fused_recurrent_sparse_delta_rule_fwd(
        q_idx=q_idx,
        q_val=q_val,
        k_idx=k_idx,
        k_val=k_val,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
    )
    return o, (final_state if output_final_state else None)
