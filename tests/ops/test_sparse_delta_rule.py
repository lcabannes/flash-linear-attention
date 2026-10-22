# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from fla.ops.sparse_delta_rule import chunk_sparse_delta_rule, fused_recurrent_sparse_delta_rule, product_key_topk
from fla.ops.sparse_delta_rule.chunk import _chunk_sparse_delta_rule
from fla.ops.sparse_delta_rule.fused_recurrent import _fused_recurrent_sparse_delta_rule
from fla.ops.sparse_delta_rule.naive import naive_product_key_topk, naive_recurrent_sparse_delta_rule
from fla.utils import assert_close, device


def make_inputs(B: int, T: int, H: int, S: int, W: int, R: int, V: int, dtype: torch.dtype):
    # distinct slots per token, drawn from a small bank so that tokens collide on slots
    q_idx = torch.rand(B, T, H, S, device=device).argsort(-1)[..., :R].sort(-1).values
    k_idx = torch.rand(B, T, H, S, device=device).argsort(-1)[..., :W].sort(-1).values
    q_val = torch.randn(B, T, H, R, device=device).softmax(-1).to(dtype)
    k_val = torch.randn(B, T, H, W, device=device).softmax(-1).to(dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float))
    beta = torch.rand(B, T, H, device=device, dtype=dtype)
    h0 = torch.randn(B, H, S, V, device=device, dtype=dtype)
    return q_idx, q_val, k_idx, k_val, v, g, beta, h0


def check_chunk(q_idx, q_val, k_idx, k_val, v, g, beta, h0):
    inputs = [x.requires_grad_(True) for x in (q_val, k_val, v, g, beta, h0)]
    do = torch.randn_like(v)
    dht = torch.randn_like(h0, dtype=torch.float)

    def run(fn):
        o, ht = fn(q_idx, q_val, k_idx, k_val, v, g, beta, h0, output_final_state=True)
        ((o * do).sum() + (ht * dht).sum()).backward()
        grads = [x.grad for x in inputs]
        for x in inputs:
            x.grad = None
        return o, ht, grads

    ref, ref_ht, ref_grads = run(naive_recurrent_sparse_delta_rule)
    tri, tri_ht, tri_grads = run(chunk_sparse_delta_rule)
    # deterministic by construction: a rerun gives the same bits
    tri2, tri2_ht, tri2_grads = run(chunk_sparse_delta_rule)
    assert all(torch.equal(a, b) for a, b in zip([tri, tri_ht, *tri_grads], [tri2, tri2_ht, *tri2_grads]))
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    for name, ref_grad, tri_grad in zip(('dq_val', 'dk_val', 'dv', 'dg', 'dbeta', 'dh0'), ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, 0.008)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'S', 'W', 'R', 'V', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-S{}-W{}-R{}-V{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 8, 8, 64, torch.float),
            (2, 100, 4, 256, 16, 32, 60, torch.bfloat16),
            (2, 1000, 2, 4096, 64, 64, 120, torch.bfloat16),
            (1, 300, 2, 65536, 64, 64, 128, torch.bfloat16),
        ]
    ],
)
def test_fused_recurrent(B: int, T: int, H: int, S: int, W: int, R: int, V: int, dtype: torch.dtype):
    torch.manual_seed(42)
    q_idx, q_val, k_idx, k_val, v, g, beta, h0 = make_inputs(B, T, H, S, W, R, V, dtype)

    ref, ref_ht = naive_recurrent_sparse_delta_rule(q_idx, q_val, k_idx, k_val, v, g, beta, h0, output_final_state=True)
    tri, tri_ht = fused_recurrent_sparse_delta_rule(q_idx, q_val, k_idx, k_val, v, g, beta, h0, output_final_state=True)
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    # in place: the same computation on the given float32 bank, which is returned
    bank = h0.to(torch.float, copy=True)
    o, ht = fused_recurrent_sparse_delta_rule(
        q_idx=q_idx,
        q_val=q_val,
        k_idx=k_idx,
        k_val=k_val,
        v=v,
        g=g,
        beta=beta,
        initial_state=bank,
        output_final_state=True,
        inplace_final_state=True,
    )
    assert ht is bank and torch.equal(o, tri) and torch.equal(ht, tri_ht)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'S', 'W', 'R', 'V', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-S{}-W{}-R{}-V{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 8, 8, 64, torch.float),
            (1, 100, 2, 8, 5, 3, 16, torch.float),
            (1, 70, 1, 32, 4, 4, 1, torch.float),
            (1, 70, 1, 32, 4, 4, 2, torch.bfloat16),
            (1, 70, 1, 32, 4, 4, 4, torch.float16),
            (1, 70, 1, 32, 4, 4, 8, torch.float16),
            (2, 200, 4, 256, 16, 32, 60, torch.bfloat16),
            (2, 300, 2, 4096, 64, 64, 120, torch.bfloat16),
            # every slot shared by about half of the tokens: long slot groups
            (1, 150, 2, 64, 32, 32, 32, torch.bfloat16),
            # every token writes and reads every slot: the longest slot groups
            (1, 130, 2, 64, 64, 64, 32, torch.bfloat16),
            # enough heads for the wider blocks of value columns (32, then 64), the last one partial
            (1, 100, 32, 256, 16, 16, 120, torch.bfloat16),
            (1, 100, 64, 64, 32, 16, 200, torch.bfloat16),
        ]
    ],
)
def test_chunk(B: int, T: int, H: int, S: int, W: int, R: int, V: int, dtype: torch.dtype):
    torch.manual_seed(42)
    check_chunk(*make_inputs(B, T, H, S, W, R, V, dtype))


class _CountLaunches:
    # wraps a kernel: counts the launches and runs them unchanged
    def __init__(self, kernel):
        self.kernel, self.launches = kernel, 0

    def __getitem__(self, grid):
        self.launches += 1
        return self.kernel[grid]


@pytest.mark.skipif(
    device != 'cuda' or not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason='the Gluon state kernels need NVIDIA SM90+',
)
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'S', 'W', 'R', 'V', 'gluon_fwd'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-S{}-W{}-R{}-V{}-gluon{}".format(*test))
        for test in [
            (1, 200, 2, 256, 16, 32, 64, True),
            # every token writes and reads every slot: the longest slot groups
            (1, 130, 2, 64, 64, 64, 32, True),
            # enough programs for the Gluon backward pass on SM100+, a partial last chunk
            (4, 260, 16, 4096, 64, 64, 128, True),
            # W % 8 != 0 and V % 16 != 0: both passes stay in Triton
            (1, 100, 2, 256, 4, 4, 64, False),
            (1, 100, 2, 256, 16, 16, 60, False),
        ]
    ],
)
def test_chunk_gluon(monkeypatch, B: int, T: int, H: int, S: int, W: int, R: int, V: int, gluon_fwd: bool):
    from fla.backends import GluonBackend
    if not GluonBackend.is_available():
        pytest.skip('Gluon is unavailable')
    from fla.ops.sparse_delta_rule import gluon
    fwd = _CountLaunches(gluon.chunk_sparse_delta_rule_fwd_kernel_h_gluon)
    bwd = _CountLaunches(gluon.chunk_sparse_delta_rule_bwd_kernel_h_gluon)
    monkeypatch.setattr(gluon, 'chunk_sparse_delta_rule_fwd_kernel_h_gluon', fwd)
    monkeypatch.setattr(gluon, 'chunk_sparse_delta_rule_bwd_kernel_h_gluon', bwd)
    monkeypatch.setenv('FLA_SDM_GLUON', '1')
    torch.manual_seed(42)
    check_chunk(*make_inputs(B, T, H, S, W, R, V, torch.bfloat16))
    # the same tolerances as the Triton kernels; the forward pass always runs in Gluon here, the backward pass where
    # its verifier accepts the call (SM100+ and at least 2.5 programs per SM)
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    # two runs of the chunk op in check_chunk
    assert fwd.launches == (2 if gluon_fwd else 0)
    gluon_bwd = gluon_fwd and torch.cuda.get_device_capability()[0] >= 10 and 2.5 * sm <= B * H * (V // 16)
    assert bwd.launches == (2 if gluon_bwd else 0)


@pytest.mark.parametrize('H', [1, 64])
def test_chunk_one_slot(H: int):
    # every token writes the only slot and reads it 64 times: one slot group per chunk, of 4160 entries
    torch.manual_seed(42)
    B, T, R, V = 1, 100, 64, 72
    q_idx = torch.zeros(B, T, H, R, device=device, dtype=torch.long)
    k_idx = torch.zeros(B, T, H, 1, device=device, dtype=torch.long)
    q_val = torch.randn(B, T, H, R, device=device).softmax(-1)
    k_val = torch.ones(B, T, H, 1, device=device)
    v = torch.randn(B, T, H, V, device=device)
    g = F.logsigmoid(torch.randn(B, T, H, device=device))
    beta = torch.rand(B, T, H, device=device)
    check_chunk(q_idx, q_val, k_idx, k_val, v, g, beta, torch.randn(B, H, 1, V, device=device))


def test_chunk_scan_handoff():
    # per chunk, 1021 slots with one write each sort first, then a slot that every token writes and reads: its group
    # starts 3 entries before the end of the first block of 1024 entries and runs deep into the next one
    torch.manual_seed(42)
    B, T, H, S, V = 1, 100, 1, 1026, 32
    u = torch.arange(T, device=device) % 64
    k_idx = (u[:, None] * 16 + torch.arange(16, device=device)).view(B, T, H, 16)
    k_idx[..., 15] = torch.where((u < 3).view(B, T, H), S - 1, k_idx[..., 15])
    k_idx = torch.cat([k_idx, torch.full_like(k_idx[..., :1], S - 2)], -1)
    q_idx = torch.full((B, T, H, 1), S - 2, device=device)
    k_val = torch.randn(B, T, H, 17, device=device).softmax(-1)
    q_val = torch.ones(B, T, H, 1, device=device)
    v = torch.randn(B, T, H, V, device=device)
    g = F.logsigmoid(torch.randn(B, T, H, device=device))
    beta = torch.rand(B, T, H, device=device)
    check_chunk(q_idx, q_val, k_idx, k_val, v, g, beta, torch.randn(B, H, S, V, device=device))


@pytest.mark.parametrize('cu_seqlens', [None, [0, 0, 0]])
def test_chunk_empty(cu_seqlens):
    torch.manual_seed(42)
    B = 2 if cu_seqlens is None else 1
    q_idx, q_val, k_idx, k_val, v, g, beta, _ = make_inputs(B, 0, 2, 64, 4, 4, 16, torch.float)
    h0 = torch.randn(2, 2, 64, 16, device=device)
    inputs = [x.requires_grad_(True) for x in (q_val, k_val, v, g, beta, h0)]
    if cu_seqlens is not None:
        cu_seqlens = torch.tensor(cu_seqlens, device=device)
    o, ht = _chunk_sparse_delta_rule(q_idx, q_val, k_idx, k_val, v, g, beta, h0, output_final_state=True,
                                     cu_seqlens=cu_seqlens)
    assert o.shape == v.shape and torch.equal(ht, h0)
    dht = torch.randn_like(h0)
    (ht * dht).sum().backward()
    assert torch.equal(h0.grad, dht) and all(x.grad.numel() == 0 for x in inputs[:-1])


def test_initial_state_shape():
    # one bank of the heads and value width of `v` per input sequence: anything else raises before a kernel runs
    torch.manual_seed(42)
    q_idx, q_val, k_idx, k_val, v, g, beta, h0 = make_inputs(2, 10, 2, 64, 4, 4, 16, torch.float)
    cu_seqlens = torch.tensor([0, 4, 20], device=device)
    flat = [x.flatten(0, 1)[None] for x in (q_idx, q_val, k_idx, k_val, v, g, beta)]
    for fn in (chunk_sparse_delta_rule, fused_recurrent_sparse_delta_rule):
        # no per-document resets: the public ops take no cu_seqlens
        assert 'cu_seqlens' not in inspect.signature(fn).parameters
        for bad in (h0[:1], h0[:, :1], h0[..., :8], h0[:, :, :0], h0[0]):
            with pytest.raises(ValueError):
                fn(q_idx, q_val, k_idx, k_val, v, g, beta, bad)
    for fn in (_chunk_sparse_delta_rule, _fused_recurrent_sparse_delta_rule):
        with pytest.raises(ValueError):
            fn(*flat, torch.cat([h0, h0]), cu_seqlens=cu_seqlens)
        fn(*flat, h0, cu_seqlens=cu_seqlens)


def test_chunk_padding_reads_no_row():
    # the padding token at t = 63 must not read a bank row: a non-finite row it would touch must not leak
    torch.manual_seed(42)
    q_idx, q_val, k_idx, k_val, v, g, beta, _ = make_inputs(1, 63, 2, 63, 4, 4, 16, torch.float)
    q_idx, k_idx = q_idx + 1, k_idx + 1
    h0 = torch.randn(1, 2, 64, 16, device=device)
    h0[:, :, 0] = float('nan')
    inputs = [x.requires_grad_(True) for x in (q_val, k_val, v, g, beta, h0)]
    o, ht = chunk_sparse_delta_rule(q_idx, q_val, k_idx, k_val, v, g, beta, h0, output_final_state=True)
    assert o.isfinite().all() and ht[:, :, 1:].isfinite().all()
    (o.sum() + ht[:, :, 1:].sum()).backward()
    assert all(x.grad.isfinite().all() for x in inputs)


@pytest.mark.parametrize(('g0', 'q0', 'h0', 'v0'), [(-128., 1., 2.**127, 0.), (-32., 1., 2.**127, 0.), (0., 0., 0., 2.**127)])
def test_chunk_extreme_values(g0, q0, h0, v0):
    # one token and slot: a zero weight, or a decay applied to a huge row, must not turn an overflowing contraction
    # into Inf or NaN gradients that the reference does not have
    idx = torch.zeros(1, 1, 1, 1, dtype=torch.long, device=device)
    q_val, k_val = torch.full((1, 1, 1, 1), q0, device=device), torch.ones(1, 1, 1, 1, device=device)
    v, h0 = torch.full((1, 1, 1, 16), v0, device=device), torch.full((1, 1, 1, 16), h0, device=device)
    g, beta = torch.full((1, 1, 1), g0, device=device), torch.ones(1, 1, 1, device=device)
    inputs = [x.requires_grad_(True) for x in (q_val, k_val, v, g, beta, h0)]

    def grads(fn):
        o, _ = fn(idx, q_val, idx, k_val, v, g, beta, h0)
        return torch.autograd.grad((o * 8).sum(), inputs)

    for name, ref, tri in zip(('dq_val', 'dk_val', 'dv', 'dg', 'dbeta', 'dh0'), grads(naive_recurrent_sparse_delta_rule),
                              grads(chunk_sparse_delta_rule)):
        if ref.isfinite().all():
            assert tri.isfinite().all(), name
            if ref.abs().max() > 0:
                assert_close(name, ref.double(), tri.double(), 0.005)


def test_chunk_disjoint_overflow():
    # two tokens on different slots: an overflowed contraction times a zero entry of A must not reach dbeta
    idx = torch.tensor([0, 1], device=device).view(1, 2, 1, 1)
    q_val, k_val = torch.ones(1, 2, 1, 1, device=device), torch.ones(1, 2, 1, 1, device=device)
    v = torch.tensor([2.**127, 0.], device=device).view(1, 2, 1, 1).expand(1, 2, 1, 16).contiguous()
    g, beta = torch.zeros(1, 2, 1, device=device), torch.ones(1, 2, 1, device=device)
    h0 = torch.zeros(1, 1, 2, 16, device=device)
    inputs = [x.requires_grad_(True) for x in (q_val, k_val, v, g, beta, h0)]
    do = torch.tensor([0., 8.], device=device).view(1, 2, 1, 1)

    def grads(fn):
        o, _ = fn(idx, q_val, idx, k_val, v, g, beta, h0)
        return torch.autograd.grad((o * do).sum(), inputs)

    for name, ref, tri in zip(('dq_val', 'dk_val', 'dv', 'dg', 'dbeta', 'dh0'), grads(naive_recurrent_sparse_delta_rule),
                              grads(chunk_sparse_delta_rule)):
        assert not ref.isfinite().all() or tri.isfinite().all(), name


def test_chunk_input_range():
    # values, bank and gradients at the edge of the documented range (2**40) keep every product finite
    torch.manual_seed(42)
    q_idx, q_val, k_idx, k_val, v, g, beta, h0 = make_inputs(1, 100, 2, 256, 16, 16, 32, torch.float)
    v, h0 = v * 2.**40, h0 * 2.**40
    inputs = [x.requires_grad_(True) for x in (q_val, k_val, v, g, beta, h0)]
    do, dht = torch.randn_like(v) * 2.**40, torch.randn_like(h0) * 2.**40

    def run(fn):
        o, ht = fn(q_idx, q_val, k_idx, k_val, v, g, beta, h0, output_final_state=True)
        return [o, ht, *torch.autograd.grad((o * do).sum() + (ht * dht).sum(), inputs)]

    for name, ref, tri in zip(('o', 'ht', 'dq_val', 'dk_val', 'dv', 'dg', 'dbeta', 'dh0'),
                              run(naive_recurrent_sparse_delta_rule), run(chunk_sparse_delta_rule)):
        assert tri.isfinite().all(), name
        assert_close(name, ref.double(), tri.double(), 0.008)
    # 2**25 written and read slots per token would overflow the int32 entry counts of a chunk
    with pytest.raises(ValueError):
        chunk_sparse_delta_rule(q_idx.new_zeros(1, 0, 2, 2**25), q_val, k_idx, k_val, v, g, beta, h0)


def test_chunk_gate_gradient_precision():
    # a huge gradient on the first slot of a chunk must not wash out the gate gradients of the other slots
    torch.manual_seed(42)
    B, T, H, S, W, R, V = 1, 128, 1, 64, 2, 2, 16
    q_idx, q_val, k_idx, k_val, v, g, beta, _ = make_inputs(B, T, H, S - 2, W, R, V, torch.float)
    q_idx, k_idx = q_idx + 2, k_idx + 2
    # the last token is the only one to touch slots 0 and 1
    k_idx[:, -1] = torch.tensor([0, 1], device=device)
    h0 = torch.randn(B, H, S, V, device=device)
    g.requires_grad_(True)
    dht = torch.zeros(B, H, S, V, device=device)
    dht[:, :, 0] = 1e15

    def dg(fn):
        o, ht = fn(q_idx, q_val, k_idx, k_val, v, g, beta, h0, output_final_state=True)
        return torch.autograd.grad((ht * dht).sum() + o.sum(), g)[0]

    assert_close('dg', dg(naive_recurrent_sparse_delta_rule)[:, :-1], dg(chunk_sparse_delta_rule)[:, :-1], 0.008)


def test_chunk_bank_rounding():
    # a bfloat16 bank is rounded once per chunk of 64 tokens: the reference rounds it at the same points
    torch.manual_seed(42)
    q_idx, q_val, k_idx, k_val, v, g, beta, h0 = make_inputs(1, 200, 2, 256, 16, 16, 32, torch.float)
    h0 = h0.bfloat16()
    ref, ht = [], h0
    for i in range(0, 200, 64):
        o, ht = naive_recurrent_sparse_delta_rule(
            *(x[:, i:i + 64] for x in (q_idx, q_val, k_idx, k_val, v, g, beta)),
            initial_state=ht.float(),
            output_final_state=True,
        )
        ref.append(o)
        ht = ht.bfloat16()
    tri, tri_ht = chunk_sparse_delta_rule(q_idx, q_val, k_idx, k_val, v, g, beta, h0, output_final_state=True)
    assert_close('o', torch.cat(ref, 1), tri, 0.002)
    assert_close('ht', ht, tri_ht, 0.002)


def test_chunk_autocast():
    torch.manual_seed(42)
    q_idx, q_val, k_idx, k_val, v, g, beta, h0 = make_inputs(1, 100, 2, 256, 16, 16, 32, torch.float)
    inputs = [x.requires_grad_(True) for x in (q_val, k_val, v, g, beta, h0)]
    ref, ref_ht = naive_recurrent_sparse_delta_rule(q_idx, q_val, k_idx, k_val, v, g, beta, h0, output_final_state=True)
    ref_grads = torch.autograd.grad(ref.sum() + ref_ht.sum(), inputs)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        tri, tri_ht = chunk_sparse_delta_rule(q_idx, q_val, k_idx, k_val, v, g, beta, h0, output_final_state=True)
        tri_grads = torch.autograd.grad(tri.sum() + tri_ht.sum(), inputs)
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    for name, ref_grad, tri_grad in zip(('dq_val', 'dk_val', 'dv', 'dg', 'dbeta', 'dh0'), ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, 0.008)


@pytest.mark.parametrize(
    ('H', 'S', 'W', 'R', 'V', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-S{}-W{}-R{}-V{}-ns{}-{}".format(*test[:5], len(test[5]) - 1, test[6]))
        for test in [
            (2, 256, 16, 16, 64, [0, 15, 100, 100, 300, 1200, 1500], torch.bfloat16),
            (2, 4096, 64, 64, 120, [0, 1, 63, 64, 1000], torch.bfloat16),
        ]
    ],
)
def test_varlen(H: int, S: int, W: int, R: int, V: int, cu_seqlens: list[int], dtype: torch.dtype):
    # the private entry point of the layer's padded batches: rows of different lengths, padding removed, one bank each
    torch.manual_seed(42)
    N, T = len(cu_seqlens) - 1, cu_seqlens[-1]
    q_idx, q_val, k_idx, k_val, v, g, beta, _ = make_inputs(1, T, H, S, W, R, V, dtype)
    h0 = torch.randn(N, H, S, V, device=device, dtype=dtype)
    inputs = [x.requires_grad_(True) for x in (q_val, k_val, v, g, beta, h0)]
    do = torch.randn_like(v)
    dht = torch.randn(N, H, S, V, device=device, dtype=torch.float)
    offsets = list(zip(cu_seqlens[:-1], cu_seqlens[1:]))
    cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.long)

    def reference(q_idx, q_val, k_idx, k_val, v, g, beta, h0, output_final_state, cu_seqlens):
        refs = [
            naive_recurrent_sparse_delta_rule(
                *(x[:, bos:eos] for x in (q_idx, q_val, k_idx, k_val, v, g, beta)),
                initial_state=h0[i:i + 1],
                output_final_state=output_final_state,
            )
            for i, (bos, eos) in enumerate(offsets)
        ]
        return torch.cat([o for o, _ in refs], 1), torch.cat([ht for _, ht in refs])

    def run(fn):
        o, ht = fn(q_idx, q_val, k_idx, k_val, v, g, beta, h0, output_final_state=True, cu_seqlens=cu_seqlens)
        ((o * do).sum() + (ht * dht).sum()).backward()
        grads = [x.grad for x in inputs]
        for x in inputs:
            x.grad = None
        return o, ht, grads

    ref, ref_ht, ref_grads = run(reference)
    tri, tri_ht, tri_grads = run(_chunk_sparse_delta_rule)
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    for name, ref_grad, tri_grad in zip(('dq_val', 'dk_val', 'dv', 'dg', 'dbeta', 'dh0'), ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, 0.008)
    with torch.no_grad():
        tri, tri_ht = _fused_recurrent_sparse_delta_rule(
            q_idx=q_idx,
            q_val=q_val,
            k_idx=k_idx,
            k_val=k_val,
            v=v,
            g=g,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)


def _count_gluon(monkeypatch):
    from fla.backends import GluonBackend
    if not GluonBackend.is_available():
        pytest.skip('Gluon is unavailable')
    from fla.ops.sparse_delta_rule import gluon
    fwd = _CountLaunches(gluon.chunk_sparse_delta_rule_fwd_kernel_h_gluon)
    bwd = _CountLaunches(gluon.chunk_sparse_delta_rule_bwd_kernel_h_gluon)
    monkeypatch.setattr(gluon, 'chunk_sparse_delta_rule_fwd_kernel_h_gluon', fwd)
    monkeypatch.setattr(gluon, 'chunk_sparse_delta_rule_bwd_kernel_h_gluon', bwd)
    monkeypatch.setenv('FLA_SDM_GLUON', '1')
    return fwd, bwd


def _gluon_bwd_expected(N: int, H: int, V: int) -> bool:
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.cuda.get_device_capability()[0] >= 10 and 2.5 * sm <= N * H * (V // 16)


@pytest.mark.skipif(
    device != 'cuda' or not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason='the Gluon state kernels need NVIDIA SM90+',
)
def test_chunk_gluon_fp32_values(monkeypatch):
    # fp32 values and output gradients with a bf16 bank: the backward reads do back as fp32, whatever its dtype
    fwd, bwd = _count_gluon(monkeypatch)
    torch.manual_seed(42)
    q_idx, q_val, k_idx, k_val, v, g, beta, h0 = make_inputs(4, 200, 16, 4096, 64, 64, 128, torch.float)
    check_chunk(q_idx, q_val, k_idx, k_val, v, g, beta, h0.bfloat16())
    assert fwd.launches == 2
    assert bwd.launches == (2 if _gluon_bwd_expected(4, 16, 128) else 0)


@pytest.mark.skipif(
    device != 'cuda' or not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason='the Gluon state kernels need NVIDIA SM90+',
)
def test_chunk_gluon_padding_reads_no_row(monkeypatch):
    # as test_chunk_padding_reads_no_row, on shapes the Gluon kernels accept: the padding tokens of the last chunk
    # (t >= 63) must not read the non-finite bank row 0
    fwd, bwd = _count_gluon(monkeypatch)
    torch.manual_seed(42)
    q_idx, q_val, k_idx, k_val, v, g, beta, _ = make_inputs(4, 63, 16, 255, 8, 8, 128, torch.bfloat16)
    q_idx, k_idx = q_idx + 1, k_idx + 1
    h0 = torch.randn(4, 16, 256, 128, device=device, dtype=torch.bfloat16)
    h0[:, :, 0] = float('nan')
    inputs = [x.requires_grad_(True) for x in (q_val, k_val, v, g, beta, h0)]
    o, ht = chunk_sparse_delta_rule(q_idx, q_val, k_idx, k_val, v, g, beta, h0, output_final_state=True)
    assert o.isfinite().all() and ht[:, :, 1:].isfinite().all()
    (o.float().sum() + ht[:, :, 1:].float().sum()).backward()
    assert all(x.grad.isfinite().all() for x in inputs)
    assert fwd.launches == 1
    assert bwd.launches == (1 if _gluon_bwd_expected(4, 16, 128) else 0)


@pytest.mark.skipif(
    device != 'cuda' or not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason='the Gluon state kernels need NVIDIA SM90+',
)
def test_varlen_gluon(monkeypatch):
    from fla.backends import GluonBackend
    if not GluonBackend.is_available():
        pytest.skip('Gluon is unavailable')
    from fla.ops.sparse_delta_rule import gluon
    fwd = _CountLaunches(gluon.chunk_sparse_delta_rule_fwd_kernel_h_gluon)
    bwd = _CountLaunches(gluon.chunk_sparse_delta_rule_bwd_kernel_h_gluon)
    monkeypatch.setattr(gluon, 'chunk_sparse_delta_rule_fwd_kernel_h_gluon', fwd)
    monkeypatch.setattr(gluon, 'chunk_sparse_delta_rule_bwd_kernel_h_gluon', bwd)
    monkeypatch.setenv('FLA_SDM_GLUON', '1')
    # sequences of 1, 62, 1 and 236 tokens, an empty one, then 700 and 500: enough programs for the Gluon backward pass
    cu_seqlens = [0, 1, 63, 64, 300, 300, 1000, 1500]
    test_varlen(16, 4096, 64, 64, 128, cu_seqlens, torch.bfloat16)
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    assert fwd.launches == 1
    gluon_bwd = torch.cuda.get_device_capability()[0] >= 10 and 2.5 * sm <= (len(cu_seqlens) - 1) * 16 * (128 // 16)
    assert bwd.launches == (1 if gluon_bwd else 0)


@pytest.mark.parametrize(
    ('dims', 'k', 'dtype'),
    [
        pytest.param(*test, id="dims{}-k{}-{}".format(*test))
        for test in [
            ((7,), 5, torch.float),
            ((7,), 1, torch.float),
            ((4, 4), 16, torch.bfloat16),
            ((16, 16), 32, torch.bfloat16),
            ((3, 5, 7, 2), 40, torch.float),
            ((15, 16, 16, 15), 64, torch.bfloat16),
            ((16, 16, 16, 16), 64, torch.bfloat16),
        ]
    ],
)
def test_product_key_topk(dims: tuple[int, ...], k: int, dtype: torch.dtype):
    torch.manual_seed(42)
    # rounding the scores to a coarse grid makes exact ties common
    s = (torch.randn(4, 100, sum(dims), device=device) * 4).round().div(4).to(dtype).requires_grad_(True)
    val, idx = product_key_topk(s, dims, k)
    assert torch.equal(idx, product_key_topk(s, dims, k)[1])
    assert (idx.diff(dim=-1) > 0).all()
    # the selection rule, ties included, against plain sorts on a few rows
    assert torch.equal(idx[0, :16], naive_product_key_topk(s[0, :16], dims, k))

    # the score of every slot, rounded to `s.dtype` after each factor from the second on
    full = None
    for x in s.detach().float().split(dims, dim=-1):
        full = x if full is None else (full.to(dtype).float().unsqueeze(-1) + x.unsqueeze(-2)).flatten(-2)
    assert torch.equal(val.detach(), full.gather(-1, idx).to(dtype))
    assert torch.equal(val.detach().float().sort(-1).values, full.to(dtype).float().topk(k, dim=-1).values.sort(-1).values)

    dval = torch.randn_like(val)
    val.backward(dval)
    cols = [idx // math.prod(dims[f + 1:]) % dim for f, dim in enumerate(dims)]
    ref = torch.cat([
        torch.zeros(*idx.shape[:-1], dim, device=device, dtype=torch.double).scatter_add_(-1, col, dval.double())
        for col, dim in zip(cols, dims)
    ], dim=-1)
    assert_close('ds', ref.to(dtype), s.grad, 1e-3)


def test_product_key_topk_edges():
    torch.manual_seed(42)
    s = torch.randn(2, 8, 12, device=device)
    s[0, 0, :3] = torch.tensor([float('nan'), float('-inf'), float('inf')])
    s[0, 1, :2] = torch.tensor([0., -0.])
    assert torch.equal(product_key_topk(s, (6, 6), 10)[1][0], naive_product_key_topk(s[0], (6, 6), 10))
    for dims, k in [((6, 6), 0), ((6, 6), 37), ((0, 6), 1), ((256, 256), 2**14 + 1)]:
        with pytest.raises(ValueError):
            product_key_topk(torch.randn(1, sum(dims), device=device), dims, k)
