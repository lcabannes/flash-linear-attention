# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Gluon backend for the two sequential state passes of ``chunk_sparse_delta_rule`` (see ``chunk.py`` for the Triton
reference: ``chunk_sparse_delta_rule_fwd_h`` and ``chunk_sparse_delta_rule_bwd_h``).

Same algorithm as the Triton kernels; the speed comes from explicit layouts: bank rows move as 8-byte words
(int64 quads of bf16 values, int64 pairs of fp32 values), the per-chunk solves use FMA dots in fp32, the per-entry
metadata of the row updates is packed into one int32, the entries of a token stay in one thread, and the row updates
run as segmented scans over large blocks of the slot-sorted entries. No shared-memory descriptors are used, so the
kernels need no API beyond Triton 3.6. Sums run in float32 in a different fixed order from the Triton kernels,
so results need not match theirs bitwise: a float32 difference can change the bf16 rounding of a bank element.
Results are bitwise reproducible for a fixed GPU type, software and settings (no atomics, one writer per bank element).

Opt-in like the other Gluon backends: ``FLA_GLUON=1`` or ``FLA_SDM_GLUON=1``. The verifiers accept CUDA calls with a
bf16 bank, ``chunk_size == 64``, ``V % 16 == 0``, ``W % 8 == 0`` and ``R % 4 == 0`` (whole blocks, no tails), and route
by device:

    | device             | forward pass | backward pass                                          |
    | SM90 (H100)        | Gluon        | Triton                                                 |
    | SM100+ (B200, B300)| Gluon        | Gluon if N * H * V / 16 >= 2.5 * SMs, else Triton      |

(the backward needs at least ~2.5 programs per SM; below that, and on SM90, it is slower than the Triton kernel).
"""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from fla.backends import GluonBackend, register
from fla.ops.sparse_delta_rule.chunk import chunk_sparse_delta_rule_bwd_h, chunk_sparse_delta_rule_fwd_h
from fla.ops.utils.op import barrier


@gluon.jit
def _bf16_part(b_p, k: gl.constexpr):
    # bf16 number k (0 = lowest 16 bits) of each int64, as float32
    return ((b_p >> (16 * k)) & 0xFFFF).to(gl.int16).to(gl.bfloat16, bitcast=True).to(gl.float32)


@gluon.jit
def _gather_sum(h64, idx, wt, i_ch, i_nh, i_v, N: gl.constexpr, C: gl.constexpr, S: gl.constexpr, V: gl.constexpr,
                BW: gl.constexpr, BV: gl.constexpr, LQ: gl.constexpr, L2: gl.constexpr):
    # [C, BV]: sum_n wt[c, n] * h[idx[c, n]], the bank rows loaded as int64 quads of bf16 columns
    o_e = gl.arange(0, C * BW, layout=gl.SliceLayout(1, LQ))
    o_q = i_v * (BV // 4) + gl.arange(0, BV // 4, layout=gl.SliceLayout(0, LQ))
    m_q = (o_q < V // 4)[None, :]
    b_0 = gl.zeros([C * BW, BV // 4], gl.float32, layout=LQ)
    b_1 = gl.zeros([C * BW, BV // 4], gl.float32, layout=LQ)
    b_2 = gl.zeros([C * BW, BV // 4], gl.float32, layout=LQ)
    b_3 = gl.zeros([C * BW, BV // 4], gl.float32, layout=LQ)
    for n0 in gl.static_range(0, N, BW):
        p_m = i_ch * C * N + (o_e // BW) * N + n0 + o_e % BW
        b_w = gl.load(wt + p_m)
        b_m = gl.load(idx + p_m)
        m = (b_m >= 0)[:, None]
        b_p = gl.load(h64 + i_nh * S * (V // 4) + gl.maximum(b_m, 0)[:, None] * (V // 4) + o_q[None, :],
                      mask=m & m_q, other=0)
        b_ww = gl.where(m, b_w[:, None], 0.)
        b_0 += _bf16_part(b_p, 0) * b_ww
        b_1 += _bf16_part(b_p, 1) * b_ww
        b_2 += _bf16_part(b_p, 2) * b_ww
        b_3 += _bf16_part(b_p, 3) * b_ww
    b_0 = gl.sum(gl.reshape(b_0, [C, BW, BV // 4]), axis=1)
    b_1 = gl.sum(gl.reshape(b_1, [C, BW, BV // 4]), axis=1)
    b_2 = gl.sum(gl.reshape(b_2, [C, BW, BV // 4]), axis=1)
    b_3 = gl.sum(gl.reshape(b_3, [C, BW, BV // 4]), axis=1)
    # [C, BV // 4, 2, 2] -> [C, BV]: column 4k + 2a + b is part 2a + b
    return gl.convert_layout(gl.reshape(gl.join(gl.join(b_0, b_2), gl.join(b_1, b_3)), [C, BV]), L2)


@gluon.jit
def _segsum2_combine(a0, a1, fa, b0, b1, fb):
    return gl.where(fb, b0, a0 + b0), gl.where(fb, b1, a1 + b1), fa | fb


@gluon.jit
def _add2(a0, a1, b0, b1):
    return a0 + b0, a1 + b1


@gluon.jit
def _bf16_lo(b):
    return (b & 0xFFFF).to(gl.int16).to(gl.bfloat16, bitcast=True).to(gl.float32)


@gluon.jit
def _bf16_hi(b):
    return (b >> 16).to(gl.int16).to(gl.bfloat16, bitcast=True).to(gl.float32)


@gluon.jit
def _bf16_pack(lo, hi):
    return (lo.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF) | \
        (hi.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) << 16)


@gluon.jit
def _row_updates(u64, h32, snap32, wpk, coef_w, decay_w, slot32, i_ch, i_nh, i_v, SAVE: gl.constexpr, S: gl.constexpr,
                 V: gl.constexpr, C: gl.constexpr, E: gl.constexpr, BE: gl.constexpr, BV: gl.constexpr,
                 LS: gl.constexpr):
    # h[s] = decay[s] * h[s] + sum_e coef[e] * u[tok[e]] over the slot-sorted write entries e of slot s: fp32 pairs of u
    # from global memory, segmented inclusive scan in a fixed order with the carry across blocks; the last entry of each
    # slot stores the new row (bf16 pairs) and, with SAVE, the old row as the snapshot at the slot's first write rank
    # e - depth[e]; metadata per entry: wpk = token | start << 6 | last << 7 | depth << 8, then coef, slot and decay
    o_k = i_v * (BV // 2) + gl.arange(0, BV // 2, layout=gl.SliceLayout(0, LS))
    m_k = (o_k < V // 2)[None, :]
    c_lo = gl.zeros([BV // 2], gl.float32, layout=gl.SliceLayout(0, LS))
    c_hi = gl.zeros([BV // 2], gl.float32, layout=gl.SliceLayout(0, LS))
    z = gl.zeros([BE, BV // 2], gl.int32, layout=LS)
    for e0 in range(0, E, BE):
        o_e = e0 + gl.arange(0, BE, layout=gl.SliceLayout(1, LS))
        p_e = i_ch * E + o_e
        b_pk = gl.load(wpk + p_e)
        b_cf = gl.load(coef_w + p_e)
        b_tok = b_pk & 63
        b_start = (b_pk & 64) != 0
        b_last = (b_pk & 128) != 0
        b_slot = gl.load(slot32 + p_e, mask=b_last, other=0).to(gl.int64)
        b_dec = gl.load(decay_w + p_e, mask=b_last, other=0.)[:, None]
        p_row = h32 + i_nh * S * (V // 2) + b_slot[:, None] * (V // 2) + o_k[None, :]
        m_row = b_last[:, None] & m_k
        b_op = gl.load(p_row, mask=m_row, other=0)
        b_xp = gl.load(u64 + (i_ch * C + b_tok)[:, None].to(gl.int64) * (V // 2) + o_k[None, :], mask=m_k, other=0)
        b_lo = (b_xp & 0xFFFFFFFF).to(gl.int32).to(gl.float32, bitcast=True) * b_cf[:, None]
        b_hi = (b_xp >> 32).to(gl.int32).to(gl.float32, bitcast=True) * b_cf[:, None]
        b_f = (b_start.to(gl.int32)[:, None] + z) != 0
        m_c = ((o_e == e0) & ~b_start)[:, None]
        b_lo = gl.where(m_c, b_lo + c_lo[None, :], b_lo)
        b_hi = gl.where(m_c, b_hi + c_hi[None, :], b_hi)
        b_lo, b_hi, _ = gl.associative_scan((b_lo, b_hi, b_f), 0, _segsum2_combine)
        m_end = (o_e == e0 + BE - 1)[:, None]
        c_lo, c_hi = gl.reduce((gl.where(m_end, b_lo, 0.), gl.where(m_end, b_hi, 0.)), 0, _add2)
        if SAVE:
            p_s = snap32 + (i_ch * E + o_e - (b_pk >> 8))[:, None].to(gl.int64) * (V // 2) + o_k[None, :]
            gl.store(p_s, b_op, mask=m_row)
        gl.store(p_row, _bf16_pack(_bf16_lo(b_op) * b_dec + b_lo, _bf16_hi(b_op) * b_dec + b_hi), mask=m_row)


@gluon.jit
def chunk_sparse_delta_rule_fwd_kernel_h_gluon(
    k_idx,
    q_idx,
    k_wt,
    q_wt,
    v,
    beta,
    Tm,
    Aqk,
    wpk,
    coef_w,
    decay_w,
    slot32,
    o,
    u,
    u64,
    snap32,
    h32,
    h64,
    chunk_offsets,
    H: gl.constexpr,
    S: gl.constexpr,
    V: gl.constexpr,
    W: gl.constexpr,
    R: gl.constexpr,
    C: gl.constexpr,
    BW: gl.constexpr,
    BE: gl.constexpr,
    BV: gl.constexpr,
    NV: gl.constexpr,
    NW: gl.constexpr,
    SAVE: gl.constexpr,
):
    pid = gl.program_id(0).to(gl.int64)
    i_v, i_nh = pid % NV, pid // NV
    i_n, i_h = i_nh // H, i_nh % H
    # [rows, BV]: 8 elements per thread along V; [entries, BV // 4] int64 quads for the gathers; [write entries, BV // 2]
    # fp32 or bf16 pairs for the row updates, 4 pairs and BE // (32 // (BV // 8)) // NW entries per thread
    L2: gl.constexpr = gl.BlockedLayout([1, 8], [32 // (BV // 8), BV // 8], [NW, 1], [1, 0])
    LQ: gl.constexpr = gl.BlockedLayout([1, 2], [32 // (BV // 8), BV // 8], [NW, 1], [1, 0])
    LS: gl.constexpr = gl.BlockedLayout([BE // (32 // (BV // 8)) // NW, 4], [32 // (BV // 8), BV // 8], [NW, 1], [1, 0])
    DA: gl.constexpr = gl.DotOperandLayout(0, L2, 0)
    DB: gl.constexpr = gl.DotOperandLayout(1, L2, 0)
    o_v = i_v * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, L2))
    m_v = o_v < V
    o_c = gl.arange(0, C, layout=gl.SliceLayout(1, L2))
    o_di = gl.arange(0, C, layout=gl.SliceLayout(1, DA))
    o_dj = gl.arange(0, C, layout=gl.SliceLayout(0, DA))
    for i_c in range(gl.load(chunk_offsets + i_n), gl.load(chunk_offsets + i_n + 1)):
        i_ch = i_c * H + i_h
        b_r = _gather_sum(h64, k_idx, k_wt, i_ch, i_nh, i_v, W, C, S, V, BW, BV, LQ, L2)
        b_q = _gather_sum(h64, q_idx, q_wt, i_ch, i_nh, i_v, R, C, S, V, BW, BV, LQ, L2)
        p_cv = i_ch * C * V + o_c[:, None] * V + o_v[None, :]
        b_beta = gl.load(beta + i_ch * C + o_c)
        b_x = b_beta[:, None] * (gl.load(v + p_cv, mask=m_v[None, :], other=0).to(gl.float32) - b_r)
        # u = Tm @ x and o = q + Aqk @ u with FMA dots (fp32, as tl.dot(input_precision='ieee') in the Triton kernel)
        b_tm = gl.load(Tm + i_ch * C * C + o_di[:, None] * C + o_dj[None, :])
        b_u = gl.dot_fma(b_tm, gl.convert_layout(b_x, DB), gl.zeros([C, BV], gl.float32, layout=L2))
        b_aq = gl.load(Aqk + i_ch * C * C + o_di[:, None] * C + o_dj[None, :])
        b_o = gl.dot_fma(b_aq, gl.convert_layout(b_u, DB), b_q)
        gl.store(o + p_cv, b_o.to(o.dtype.element_ty), mask=m_v[None, :])
        gl.store(u + p_cv, b_u, mask=m_v[None, :])
        # the rows of u this program just stored are read back by other threads of the program
        barrier()
        _row_updates(u64, h32, snap32, wpk, coef_w, decay_w, slot32, i_ch, i_nh, i_v, SAVE, S, V, C, C * W, BE, BV, LS)
        barrier()


@gluon.jit
def _f32_lo(b):
    return (b & 0xFFFFFFFF).to(gl.int32).to(gl.float32, bitcast=True)


@gluon.jit
def _f32_hi(b):
    return (b >> 32).to(gl.int32).to(gl.float32, bitcast=True)


@gluon.jit
def _f32_pack(lo, hi):
    return (lo.to(gl.int32, bitcast=True).to(gl.int64) & 0xFFFFFFFF) | (hi.to(gl.int32, bitcast=True).to(gl.int64) << 32)


@gluon.jit
def _join4(b_0, b_1, b_2, b_3, C: gl.constexpr, BV: gl.constexpr, L2: gl.constexpr):
    # 4 x [C, BV // 4] (part k = columns 4q + k) -> [C, BV]
    return gl.convert_layout(gl.reshape(gl.join(gl.join(b_0, b_2), gl.join(b_1, b_3)), [C, BV]), L2)


@gluon.jit
def _dh_updates(dx64, dh64, bpk, wt, decay, slot32, i_ch, i_nh, i_v, S: gl.constexpr, V: gl.constexpr,
                C: gl.constexpr, E: gl.constexpr, BE: gl.constexpr, BV: gl.constexpr, LS: gl.constexpr):
    # dh[s] = decay[s] * dh[s] + sum_e wt[e] * dx[row[e]] over the slot-sorted entries e of slot s (fp32 pairs), segmented
    # scan in a fixed order with the carry across blocks, stored once by the last entry of each slot
    o_k = i_v * (BV // 2) + gl.arange(0, BV // 2, layout=gl.SliceLayout(0, LS))
    m_k = (o_k < V // 2)[None, :]
    c_lo = gl.zeros([BV // 2], gl.float32, layout=gl.SliceLayout(0, LS))
    c_hi = gl.zeros([BV // 2], gl.float32, layout=gl.SliceLayout(0, LS))
    z = gl.zeros([BE, BV // 2], gl.int32, layout=LS)
    for e0 in range(0, E, BE):
        o_e = e0 + gl.arange(0, BE, layout=gl.SliceLayout(1, LS))
        p_e = i_ch * E + o_e
        b_pk = gl.load(bpk + p_e)
        b_wt = gl.load(wt + p_e)
        b_row = b_pk & 127
        b_start = (b_pk & 128) != 0
        b_last = (b_pk & 256) != 0
        b_slot = gl.load(slot32 + p_e, mask=b_last, other=0).to(gl.int64)
        b_dec = gl.load(decay + p_e, mask=b_last, other=0.)[:, None]
        p_row = dh64 + i_nh * S * (V // 2) + b_slot[:, None] * (V // 2) + o_k[None, :]
        m_row = b_last[:, None] & m_k
        b_op = gl.load(p_row, mask=m_row, other=0)
        b_xp = gl.load(dx64 + (i_ch * 2 * C + b_row)[:, None].to(gl.int64) * (V // 2) + o_k[None, :], mask=m_k, other=0)
        b_lo = _f32_lo(b_xp) * b_wt[:, None]
        b_hi = _f32_hi(b_xp) * b_wt[:, None]
        b_f = (b_start.to(gl.int32)[:, None] + z) != 0
        m_c = ((o_e == e0) & ~b_start)[:, None]
        b_lo = gl.where(m_c, b_lo + c_lo[None, :], b_lo)
        b_hi = gl.where(m_c, b_hi + c_hi[None, :], b_hi)
        b_lo, b_hi, _ = gl.associative_scan((b_lo, b_hi, b_f), 0, _segsum2_combine)
        m_end = (o_e == e0 + BE - 1)[:, None]
        c_lo, c_hi = gl.reduce((gl.where(m_end, b_lo, 0.), gl.where(m_end, b_hi, 0.)), 0, _add2)
        gl.store(p_row, _f32_pack(_f32_lo(b_op) * b_dec + b_lo, _f32_hi(b_op) * b_dec + b_hi), mask=m_row)


@gluon.jit
def chunk_sparse_delta_rule_bwd_kernel_h_gluon(
    kidx32,
    spk,
    k_wt,
    k_coef,
    qidx32,
    src_q,
    q_wt,
    v,
    beta,
    Tm,
    Aqk,
    u,
    u64,
    snap64,
    bpk,
    wt,
    decay,
    slot32,
    e_k,
    e_q,
    e_c,
    e_end,
    do,
    h64,
    dh64,
    dv,
    dx,
    dx64,
    dAqk,
    dL,
    dbeta,
    dkw,
    dqw,
    dc,
    dgend,
    chunk_offsets,
    NTH,
    H: gl.constexpr,
    S: gl.constexpr,
    V: gl.constexpr,
    W: gl.constexpr,
    R: gl.constexpr,
    C: gl.constexpr,
    BW: gl.constexpr,
    BE: gl.constexpr,
    BV: gl.constexpr,
    NV: gl.constexpr,
    NW: gl.constexpr,
):
    pid = gl.program_id(0).to(gl.int64)
    i_v, i_nh = pid % NV, pid // NV
    i_n, i_h = i_nh // H, i_nh % H
    L2: gl.constexpr = gl.BlockedLayout([1, 8], [32 // (BV // 8), BV // 8], [NW, 1], [1, 0])
    # [C, C] accumulators of dAqk and dL
    LC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    # [C, BW, BV // 4]: entries of BW consecutive writes (or reads) of every token, int64 quads / pairs along V
    L3: gl.constexpr = gl.BlockedLayout([1, BW, 2], [32 // (BV // 8), 1, BV // 8], [NW, 1, 1], [2, 1, 0])
    LT: gl.constexpr = gl.SliceLayout(1, L3)
    LE: gl.constexpr = gl.SliceLayout(2, L3)
    LS: gl.constexpr = gl.BlockedLayout([BE // (32 // (BV // 8)) // NW, 4], [32 // (BV // 8), BV // 8], [NW, 1], [1, 0])
    DA: gl.constexpr = gl.DotOperandLayout(0, L2, 0)
    DB: gl.constexpr = gl.DotOperandLayout(1, L2, 0)
    DCA: gl.constexpr = gl.DotOperandLayout(0, LC, 0)
    DCB: gl.constexpr = gl.DotOperandLayout(1, LC, 0)
    o_v = i_v * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, L2))
    m_v = o_v < V
    o_c = gl.arange(0, C, layout=gl.SliceLayout(1, L2))
    o_di = gl.arange(0, C, layout=gl.SliceLayout(1, DA))
    o_dj = gl.arange(0, C, layout=gl.SliceLayout(0, DA))
    o_ci = gl.arange(0, C, layout=gl.SliceLayout(1, LC))
    o_cj = gl.arange(0, C, layout=gl.SliceLayout(0, LC))
    o_bk = i_v * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, DCB))
    o_bj = gl.arange(0, C, layout=gl.SliceLayout(0, DCB))
    # entry indices [C, BW] and quad indices [BV // 4] of the 3D tiles; token tiles [C, BV // 4]
    o_et = gl.arange(0, C, layout=gl.SliceLayout(1, LE))
    o_ej = gl.arange(0, BW, layout=gl.SliceLayout(0, LE))
    o_q3 = i_v * (BV // 4) + gl.arange(0, BV // 4, layout=gl.SliceLayout(0, LT))
    m_q3 = (o_q3 < V // 4)[None, None, :]
    o_tt = gl.arange(0, C, layout=gl.SliceLayout(1, LT))
    o_tq = i_v * (BV // 4) + gl.arange(0, BV // 4, layout=gl.SliceLayout(0, LT))
    m_tq = (o_tq < V // 4)[None, :]
    c0, c1 = gl.load(chunk_offsets + i_n), gl.load(chunk_offsets + i_n + 1)
    for i_r in range(c0, c1):
        i_c = c1 - 1 - (i_r - c0)
        i_ch = i_c * H + i_h
        p_cv = i_ch * C * V + o_c[:, None] * V + o_v[None, :]
        # (A) the write entries: u rows per token as 4 parts of fp32 pairs
        p_u = u64 + (i_ch * C + o_tt)[:, None] * (V // 2) + 2 * o_tq[None, :]
        b_ua = gl.load(p_u, mask=m_tq, other=0)
        b_ub = gl.load(p_u + 1, mask=m_tq, other=0)
        u_0 = gl.expand_dims(_f32_lo(b_ua), 1)
        u_1 = gl.expand_dims(_f32_hi(b_ua), 1)
        u_2 = gl.expand_dims(_f32_lo(b_ub), 1)
        u_3 = gl.expand_dims(_f32_hi(b_ub), 1)
        a_r0 = gl.zeros([C, BV // 4], gl.float32, layout=LT)
        a_r1 = gl.zeros([C, BV // 4], gl.float32, layout=LT)
        a_r2 = gl.zeros([C, BV // 4], gl.float32, layout=LT)
        a_r3 = gl.zeros([C, BV // 4], gl.float32, layout=LT)
        a_d0 = gl.zeros([C, BV // 4], gl.float32, layout=LT)
        a_d1 = gl.zeros([C, BV // 4], gl.float32, layout=LT)
        a_d2 = gl.zeros([C, BV // 4], gl.float32, layout=LT)
        a_d3 = gl.zeros([C, BV // 4], gl.float32, layout=LT)
        for w0 in gl.static_range(0, W, BW):
            p_w = i_ch * C * W + o_et[:, None] * W + w0 + o_ej[None, :]
            b_i = gl.load(kidx32 + p_w)
            b_sp = gl.load(spk + p_w)
            m = b_i >= 0
            m3 = m[:, :, None] & m_q3
            b_src = (b_sp & 0xFFFF).to(gl.int64)
            b_first = ((b_sp >> 16) & 1) != 0
            b_rows = gl.load(snap64 + (i_ch * C * W + b_src)[:, :, None] * (V // 4) + o_q3[None, None, :], mask=m3, other=0)
            p_bank = i_nh * S + gl.maximum(b_i, 0).to(gl.int64)
            gl.store(h64 + p_bank[:, :, None] * (V // 4) + o_q3[None, None, :], b_rows, mask=m3 & b_first[:, :, None])
            p_d = dh64 + p_bank[:, :, None] * (V // 2) + 2 * o_q3[None, None, :]
            b_da = gl.load(p_d, mask=m3, other=0)
            b_db = gl.load(p_d + 1, mask=m3, other=0)
            r_0, r_1, r_2, r_3 = _bf16_part(b_rows, 0), _bf16_part(b_rows, 1), _bf16_part(b_rows, 2), _bf16_part(b_rows, 3)
            d_0, d_1, d_2, d_3 = _f32_lo(b_da), _f32_hi(b_da), _f32_lo(b_db), _f32_hi(b_db)
            b_kw = gl.expand_dims(gl.load(k_wt + p_w), 2)
            b_kc = gl.expand_dims(gl.load(k_coef + p_w), 2)
            a_r0 += gl.sum(b_kw * r_0, 1)
            a_r1 += gl.sum(b_kw * r_1, 1)
            a_r2 += gl.sum(b_kw * r_2, 1)
            a_r3 += gl.sum(b_kw * r_3, 1)
            a_d0 += gl.sum(b_kc * d_0, 1)
            a_d1 += gl.sum(b_kc * d_1, 1)
            a_d2 += gl.sum(b_kc * d_2, 1)
            a_d3 += gl.sum(b_kc * d_3, 1)
            # the decays scale a factor before each contraction, so a contraction overflows only if its result does
            b_ec = gl.expand_dims(gl.load(e_c + p_w), 2)
            b_dc = gl.sum(d_0 * b_ec * u_0 + d_1 * b_ec * u_1 + d_2 * b_ec * u_2 + d_3 * b_ec * u_3, 2)
            gl.store(dc + i_v * NTH * C * W + p_w, b_dc)
            b_end = gl.expand_dims(gl.load(e_end + p_w), 2)
            b_dg = gl.sum(d_0 * (r_0 * b_end) + d_1 * (r_1 * b_end) + d_2 * (r_2 * b_end) + d_3 * (r_3 * b_end), 2)
            gl.store(dgend + i_v * NTH * C * W + p_w, b_dg)
        b_r = _join4(a_r0, a_r1, a_r2, a_r3, C, BV, L2)
        b_du = _join4(a_d0, a_d1, a_d2, a_d3, C, BV, L2)
        # (B) the [C, BV] part
        b_vr = gl.load(v + p_cv, mask=m_v[None, :], other=0).to(gl.float32) - b_r
        b_beta = gl.load(beta + i_ch * C + o_c)
        b_do = gl.load(do + p_cv, mask=m_v[None, :], other=0).to(gl.float32)
        b_aqt = gl.load(Aqk + i_ch * C * C + o_dj[None, :] * C + o_di[:, None])
        b_du = gl.dot_fma(b_aqt, gl.convert_layout(b_do, DB), b_du)
        b_tmt = gl.load(Tm + i_ch * C * C + o_dj[None, :] * C + o_di[:, None])
        b_dx = gl.dot_fma(b_tmt, gl.convert_layout(b_du, DB), gl.zeros([C, BV], gl.float32, layout=L2))
        gl.store(dv + p_cv, (b_beta[:, None] * b_dx).to(dv.dtype.element_ty), mask=m_v[None, :])
        p_x = dx + i_ch * 2 * C * V + o_c[:, None] * V + o_v[None, :]
        gl.store(p_x, -b_beta[:, None] * b_dx, mask=m_v[None, :])
        gl.store(p_x + C * V, b_do, mask=m_v[None, :])
        b_ut = gl.load(u + i_ch * C * V + o_bj[None, :] * V + o_bk[:, None], mask=(o_bk < V)[:, None], other=0)
        p_vcc = i_v * NTH * C * C + i_ch * C * C + o_ci[:, None] * C + o_cj[None, :]
        gl.store(dAqk + p_vcc, gl.dot_fma(gl.convert_layout(b_do, DCA), b_ut, gl.zeros([C, C], gl.float32, layout=LC)))
        # through the solve u = (I + diag(beta) A)^-1 x: the gradient of (I + diag(beta) A) is -dx u^T
        gl.store(dL + p_vcc, -gl.dot_fma(gl.convert_layout(b_dx, DCA), b_ut, gl.zeros([C, C], gl.float32, layout=LC)))
        gl.store(dbeta + i_v * NTH * C + i_ch * C + o_c, gl.sum(b_dx * b_vr, 1))
        # the dx rows and the restored bank rows are read back by other threads of the program
        barrier()
        # (C) dkw per write entry: -(beta dx)[t] . (snapshot row * e_k)
        p_n = dx64 + (i_ch * 2 * C + o_tt)[:, None] * (V // 2) + 2 * o_tq[None, :]
        b_na = gl.load(p_n, mask=m_tq, other=0)
        b_nb = gl.load(p_n + 1, mask=m_tq, other=0)
        n_0 = gl.expand_dims(_f32_lo(b_na), 1)
        n_1 = gl.expand_dims(_f32_hi(b_na), 1)
        n_2 = gl.expand_dims(_f32_lo(b_nb), 1)
        n_3 = gl.expand_dims(_f32_hi(b_nb), 1)
        for w0 in gl.static_range(0, W, BW):
            p_w = i_ch * C * W + o_et[:, None] * W + w0 + o_ej[None, :]
            b_i = gl.load(kidx32 + p_w)
            b_src = (gl.load(spk + p_w) & 0xFFFF).to(gl.int64)
            m3 = (b_i >= 0)[:, :, None] & m_q3
            b_rows = gl.load(snap64 + (i_ch * C * W + b_src)[:, :, None] * (V // 4) + o_q3[None, None, :], mask=m3, other=0)
            b_ek = gl.expand_dims(gl.load(e_k + p_w), 2)
            b_kw = gl.sum(n_0 * (_bf16_part(b_rows, 0) * b_ek) + n_1 * (_bf16_part(b_rows, 1) * b_ek) +
                          n_2 * (_bf16_part(b_rows, 2) * b_ek) + n_3 * (_bf16_part(b_rows, 3) * b_ek), 2)
            gl.store(dkw + i_v * NTH * C * W + p_w, b_kw)
        # (D) dqw per read entry: do[t] . (start row * e_q), the start row being the slot's snapshot if the chunk writes
        # the slot, else the bank row
        # do as the dx rows C..2C-1 hold it, in fp32 whatever the dtype of do
        p_o = p_n + C * (V // 2)
        b_oa = gl.load(p_o, mask=m_tq, other=0)
        b_ob = gl.load(p_o + 1, mask=m_tq, other=0)
        g_0 = gl.expand_dims(_f32_lo(b_oa), 1)
        g_1 = gl.expand_dims(_f32_hi(b_oa), 1)
        g_2 = gl.expand_dims(_f32_lo(b_ob), 1)
        g_3 = gl.expand_dims(_f32_hi(b_ob), 1)
        for w0 in gl.static_range(0, R, BW):
            p_w = i_ch * C * R + o_et[:, None] * R + w0 + o_ej[None, :]
            b_i = gl.load(qidx32 + p_w)
            b_src = gl.load(src_q + p_w)
            m3 = (b_i >= 0)[:, :, None] & m_q3
            # one gather per entry: the snapshot of a slot the chunk writes, else the bank row
            p_r = gl.where(b_src >= 0, snap64 + (i_ch * C * W + b_src.to(gl.int64)) * (V // 4),
                           h64 + (i_nh * S + gl.maximum(b_i, 0).to(gl.int64)) * (V // 4))
            b_rows = gl.load(p_r[:, :, None] + o_q3[None, None, :], mask=m3, other=0)
            b_eq = gl.expand_dims(gl.load(e_q + p_w), 2)
            b_qw = gl.sum(g_0 * (_bf16_part(b_rows, 0) * b_eq) + g_1 * (_bf16_part(b_rows, 1) * b_eq) +
                          g_2 * (_bf16_part(b_rows, 2) * b_eq) + g_3 * (_bf16_part(b_rows, 3) * b_eq), 2)
            gl.store(dqw + i_v * NTH * C * R + p_w, b_qw)
        # (E) dh_start[s] = decay[s] * dh_end[s] + the gradients of the reads of row s, written once per touched row
        _dh_updates(dx64, dh64, bpk, wt, decay, slot32, i_ch, i_nh, i_v, S, V, C, C * (W + R), BE, BV, LS)
        barrier()


def _check(p, h, bwd):
    if not h.is_cuda:
        return False, 'the Gluon sparse delta rule requires CUDA tensors'
    major = torch.cuda.get_device_capability(h.device)[0]
    if major < 9:
        return False, 'the Gluon sparse delta rule requires NVIDIA SM90+'
    if h.dtype != torch.bfloat16:
        return False, 'the Gluon sparse delta rule requires a bf16 bank'
    N, H, S, V = h.shape
    C, W, R = p['k_idx'].shape[-2], p['k_idx'].shape[-1], p['q_idx'].shape[-1]
    # whole blocks only: 16 value columns per program, 4 writes or reads per token per step, 512 (forward) and 256
    # (backward) slot-sorted entries per scan block; 32-bit column offsets
    if C != 64 or not 0 < V < 2**24 or V % 16 or W % 8 or R % 4:
        return False, 'the Gluon sparse delta rule requires chunk_size 64, 0 < V < 2**24, V % 16 == 0, W % 8 == 0, R % 4 == 0'
    # packed fields: write ranks below 2**16, slots (padding slots go past S) below 2**31
    if C * W > 2**16 or S + C * (W + R) >= 2**31:
        return False, 'the Gluon sparse delta rule requires C * W <= 2**16 and S + C * (W + R) < 2**31'
    if bwd:
        if major < 10:
            return False, 'the Gluon backward pass is slower than the Triton kernel before SM100'
        if 2.5 * torch.cuda.get_device_properties(h.device).multi_processor_count > N * H * (V // 16):
            return False, 'the Gluon backward pass needs at least 2.5 programs per SM'
    return True, None


def _buffers(h, *specs):
    # the kernels address every buffer as contiguous rows on the bank's device and reinterpret some as int64 words
    for x, dtype, shape in specs:
        if x.device != h.device or not x.is_contiguous() or tuple(x.shape) != shape or dtype not in (None, x.dtype):
            return False, 'the Gluon sparse delta rule requires contiguous buffers of the documented shapes and dtypes'
    return True, None


def chunk_sparse_delta_rule_fwd_h_verifier(p, Tm, Aqk, o, u, snap, h):
    ok, reason = _check(p, h, bwd=False)
    if not ok:
        return ok, reason
    NT, H, C, W = p['k_idx'].shape
    V = h.shape[-1]
    specs = [(h, None, tuple(h.shape)), (o, None, (NT, H, C, V)), (u, torch.float, (NT, H, C, V)),
             (Tm, torch.float, (NT, H, C, C)), (Aqk, torch.float, (NT, H, C, C))]
    return _buffers(h, *specs, *([] if snap is None else [(snap, torch.bfloat16, (NT, H, C, W, V))]))


def chunk_sparse_delta_rule_bwd_h_verifier(p, Tm, Aqk, u, snap, do, h, dh):
    ok, reason = _check(p, h, bwd=True)
    if not ok:
        return ok, reason
    NT, H, C, W = p['k_idx'].shape
    V = h.shape[-1]
    return _buffers(h, (h, None, tuple(h.shape)), (dh, torch.float, tuple(h.shape)), (u, torch.float, (NT, H, C, V)),
                    (snap, torch.bfloat16, (NT, H, C, W, V)), (do, None, (NT, H, C, V)),
                    (Tm, torch.float, (NT, H, C, C)), (Aqk, torch.float, (NT, H, C, C)))


@register(chunk_sparse_delta_rule_fwd_h, backend=GluonBackend, verifier=chunk_sparse_delta_rule_fwd_h_verifier,
          env_var='FLA_SDM_GLUON')
def chunk_sparse_delta_rule_fwd_h_gluon(p, Tm, Aqk, o, u, snap, h):
    N, H, S, V = h.shape
    NT, _, C, W = p['k_idx'].shape
    R = p['q_idx'].shape[-1]
    BV, NW = 16, 4
    depth = p['depth_w']
    wpk = p['tok_w'] | ((depth == 0).int() << 6) | (p['last_w'].int() << 7) | (depth << 8)
    chunk_sparse_delta_rule_fwd_kernel_h_gluon[(V // BV * N * H,)](
        p['k_idx'], p['q_idx'], p['k_wt'], p['q_wt'], p['v'], p['beta'], Tm, Aqk, wpk, p['coef_w'], p['decay_w'],
        p['slot_w'].int(), o, u, u.view(torch.int64), (h if snap is None else snap).view(torch.int32),
        h.view(torch.int32), h.view(torch.int64), p['chunk_offsets'], H=H, S=S, V=V, W=W, R=R, C=C, BW=4, BE=512,
        BV=BV, NV=V // BV, NW=NW, SAVE=snap is not None, num_warps=NW)


@register(chunk_sparse_delta_rule_bwd_h, backend=GluonBackend, verifier=chunk_sparse_delta_rule_bwd_h_verifier,
          env_var='FLA_SDM_GLUON')
def chunk_sparse_delta_rule_bwd_h_gluon(p, Tm, Aqk, u, snap, do, h, dh):
    N, H, S, V = h.shape
    NT, _, C, W = p['k_idx'].shape
    R = p['q_idx'].shape[-1]
    BV, NW = 16, 4
    NV = V // BV
    dv = torch.empty_like(p['v'])
    dx = h.new_empty(NT, H, 2 * C, V, dtype=torch.float)
    dAqk = h.new_empty(NV, NT, H, C, C, dtype=torch.float)
    dL = torch.empty_like(dAqk)
    dbeta = h.new_empty(NV, NT, H, C, dtype=torch.float)
    dkw = h.new_empty(NV, NT, H, C, W, dtype=torch.float)
    dc, dgend = torch.empty_like(dkw), torch.empty_like(dkw)
    dqw = h.new_empty(NV, NT, H, C, R, dtype=torch.float)
    spk = p['src_k'] | (p['first'].int() << 16)
    bpk = p['row'] | ((p['depth'] == 0).int() << 7) | (p['last'].int() << 8)
    # 128 registers per thread keep four programs per SM; the spills land in L1 and cost less than a second wave
    chunk_sparse_delta_rule_bwd_kernel_h_gluon[(NV * N * H,)](
        p['k_idx'].int(), spk, p['k_wt'], p['k_coef'], p['q_idx'].int(), p['src_q'], p['q_wt'], p['v'], p['beta'], Tm,
        Aqk, u, u.view(torch.int64), snap.view(torch.int64), bpk, p['wt'], p['decay'], p['slot'].int(), p['e_k'],
        p['e_q'], p['e_c'], p['e_end'], do, h.view(torch.int64), dh.view(torch.int64), dv, dx,
        dx.view(torch.int64), dAqk, dL, dbeta, dkw, dqw, dc, dgend, p['chunk_offsets'], NT * H, H=H, S=S, V=V, W=W, R=R,
        C=C, BW=4, BE=256, BV=BV, NV=NV, NW=NW, num_warps=NW, maxnreg=128)
    return (dv, *(x.sum(0) for x in (dAqk, dL, dbeta, dkw, dqw, dc, dgend)))
