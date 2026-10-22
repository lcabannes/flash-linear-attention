# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy

import pytest
import torch

from fla.layers import SparseDeltaMemory, sdm
from fla.models.cache_utils import Cache
from fla.ops.sparse_delta_rule.naive import naive_recurrent_sparse_delta_rule
from fla.utils import assert_close, device


def _reference(q_idx, q_val, k_idx, k_val, v, g, beta, initial_state, output_final_state=False, cu_seqlens=None):
    if cu_seqlens is None:
        return naive_recurrent_sparse_delta_rule(q_idx, q_val, k_idx, k_val, v, g, beta, initial_state, output_final_state)
    outputs, states = [], []
    for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True)):
        o, h = naive_recurrent_sparse_delta_rule(
            *(x[:, bos:eos] for x in (q_idx, q_val, k_idx, k_val, v, g, beta)),
            initial_state=initial_state[i:i + 1],
            output_final_state=output_final_state,
        )
        outputs.append(o)
        states.append(h)
    return torch.cat(outputs, dim=1), (torch.cat(states) if output_final_state else None)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16], ids=['fp32', 'bf16'])
@pytest.mark.parametrize('layout', ['dense', 'packed', 'padded'])
def test_forward_backward(dtype, layout, monkeypatch):
    torch.manual_seed(42)
    layer = SparseDeltaMemory(hidden_size=128, num_heads=2, product_key_dims=(4, 4, 4, 4), num_writes=8, num_reads=8)
    layer = layer.to(device=device, dtype=dtype)
    reference = copy.deepcopy(layer)
    B, T = (1, 150) if layout == 'packed' else (2, 90)
    x = torch.randn(B, T, 128, dtype=dtype, device=device)
    do = torch.randn_like(x)
    kwargs = {}
    if layout == 'packed':
        kwargs['cu_seqlens'] = torch.tensor([0, 37, 150], dtype=torch.int32, device=device)
    elif layout == 'padded':
        mask = torch.ones(B, T, dtype=torch.long, device=device)
        mask[0, :13], mask[1, 70:] = 0, 0
        kwargs['attention_mask'] = mask

    actual_x = x.clone().requires_grad_()
    actual, _, _ = layer(hidden_states=actual_x, **kwargs)
    if layout == 'packed':
        # one bank carried across both documents: the document boundaries change nothing
        assert torch.equal(actual, layer(hidden_states=x)[0])
    elif layout == 'padded':
        assert (actual[mask == 0] == 0).all()
    actual.backward(do)
    ref_x = x.clone().requires_grad_()
    with monkeypatch.context() as patch:
        patch.setattr(sdm, '_chunk_sparse_delta_rule', _reference)
        expected, _, _ = reference(hidden_states=ref_x, **kwargs)
        expected.backward(do)
    tol = 0.005 if dtype == torch.float32 else 0.05
    assert_close('output', expected, actual, tol)
    assert_close('dx', ref_x.grad, actual_x.grad, tol)
    for (name, param), (_, ref_param) in zip(layer.named_parameters(), reference.named_parameters(), strict=True):
        assert_close(name, ref_param.grad, param.grad, tol)


def test_padding_isolated():
    # padded tokens never reach the bank: NaN there changes nothing, an all-padding row outputs zeros, and packed-document
    # boundaries given with the mask are ignored; a mask of another batch size raises
    torch.manual_seed(42)
    layer = SparseDeltaMemory(hidden_size=128, num_heads=2, product_key_dims=(4, 4, 4, 4), num_writes=8, num_reads=8)
    layer = layer.to(device)
    x = torch.randn(3, 90, 128, device=device)
    mask = torch.ones(3, 90, dtype=torch.long, device=device)
    mask[0, :13], mask[1, 70:], mask[2] = 0, 0, 0
    clean, _, _ = layer(hidden_states=x, attention_mask=mask)
    poisoned = x.masked_fill(mask[..., None] == 0, float('nan')).requires_grad_()
    cu_seqlens = torch.tensor([0, 50, 270], dtype=torch.int32, device=device)
    out, _, _ = layer(hidden_states=poisoned, attention_mask=mask, cu_seqlens=cu_seqlens)
    assert torch.equal(out, clean) and (out[mask == 0] == 0).all()
    out.sum().backward()
    assert all(param.grad.isfinite().all() for param in layer.parameters()) and poisoned.grad.isfinite().all()
    with pytest.raises(ValueError):
        layer(hidden_states=x, attention_mask=mask[:2])


def test_decoding():
    # float32: in half precision, decoding and the full forward round the product-key scores differently,
    # which can flip near-tied selections
    torch.manual_seed(42)
    layer = SparseDeltaMemory(hidden_size=256, num_heads=2, layer_idx=0).to(device).eval()
    x = torch.randn(2, 100, 256, device=device)
    with torch.no_grad():
        expected, _, _ = layer(hidden_states=x)
        cache = Cache()
        prefill, _, cache = layer(hidden_states=x[:, :90], past_key_values=cache, use_cache=True)
        steps = [layer(hidden_states=x[:, i:i + 1], past_key_values=cache, use_cache=True)[0] for i in range(90, 100)]
    assert_close('output', expected, torch.cat([prefill, *steps], dim=1), 1e-3)


def test_decoding_padded():
    # a left-padded row decodes as the same row without its padding: one bank per row, padding never reaches it
    torch.manual_seed(42)
    layer = SparseDeltaMemory(hidden_size=256, num_heads=2, layer_idx=0).to(device).eval()
    x = torch.randn(2, 100, 256, device=device)
    mask = torch.ones(2, 100, dtype=torch.long, device=device)
    mask[0, :7] = 0
    with torch.no_grad():
        cache = Cache()
        prefill, _, cache = layer(hidden_states=x[:, :90], attention_mask=mask[:, :90], past_key_values=cache, use_cache=True)
        steps = [layer(hidden_states=x[:, i:i + 1], attention_mask=mask[:, :i + 1], past_key_values=cache, use_cache=True)[0]
                 for i in range(90, 100)]
        alone = Cache()
        row, _, alone = layer(hidden_states=x[:1, 7:90], past_key_values=alone, use_cache=True)
        row_steps = [layer(hidden_states=x[:1, i:i + 1], past_key_values=alone, use_cache=True)[0] for i in range(90, 100)]
    assert (prefill[0, :7] == 0).all()
    assert_close('prefill', row, prefill[:1, 7:], 1e-3)
    assert_close('steps', torch.cat(row_steps, 1), torch.cat(steps, 1)[:1], 1e-3)


def test_decoding_cache():
    torch.manual_seed(42)
    layer = SparseDeltaMemory(
        hidden_size=128,
        num_heads=2,
        product_key_dims=(4, 4, 4, 4),
        num_writes=8,
        num_reads=8,
        layer_idx=0,
    ).to(device).eval()
    x = torch.randn(1, 80, 128, device=device)
    # a decode step between a grad-enabled prefill and its backward leaves the bank saved for the backward intact
    reference = copy.deepcopy(layer)
    reference(hidden_states=x[:, :70])[0].sum().backward()
    cache = Cache()
    out, _, cache = layer(hidden_states=x[:, :70], past_key_values=cache, use_cache=True)
    with torch.no_grad():
        layer(hidden_states=x[:, 70:71], past_key_values=cache, use_cache=True)
    out.sum().backward()
    for (name, param), (_, ref_param) in zip(layer.named_parameters(), reference.named_parameters(), strict=True):
        assert torch.equal(param.grad, ref_param.grad), name
    # a cache record without a bank starts from the learned bank; a bfloat16 bank moves to float32 once, then is
    # advanced in place
    layer = layer.bfloat16()
    with torch.no_grad():
        cache = Cache()
        layer(hidden_states=x[:, :70].bfloat16(), past_key_values=cache, use_cache=False)
        layer(hidden_states=x[:, 70:71].bfloat16(), past_key_values=cache, use_cache=True)
        cache = Cache()
        layer(hidden_states=x[:, :70].bfloat16(), past_key_values=cache, use_cache=True)
        assert cache[0]['recurrent_state'].dtype == torch.bfloat16
        layer(hidden_states=x[:, 70:71].bfloat16(), past_key_values=cache, use_cache=True)
        bank = cache[0]['recurrent_state']
        layer(hidden_states=x[:, 71:72].bfloat16(), past_key_values=cache, use_cache=True)
    assert bank.dtype == torch.float32 and cache[0]['recurrent_state'] is bank
