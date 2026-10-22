# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.backends import dispatch
from fla.ops.utils.op import exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


# data layout and kernel map: README.md in this directory
@triton.jit
def chunk_sparse_delta_rule_fwd_kernel_intra(
    row,
    val,
    G,
    pair_i,
    pair_j,
    pair_offsets,
    AA,
    row0,
    C: tl.constexpr,
    BP: tl.constexpr,
):
    # one program per (chunk, head) of the batch, over its pairs (i, j) of an entry and an earlier write of the same slot;
    # a row whose pairs span two batches accumulates over both
    i_r = tl.program_id(0).to(tl.int64)
    i_ch = row0 + i_r
    o_c = tl.arange(0, C).to(tl.int64)
    o_r = tl.arange(0, 2 * C).to(tl.int64)
    # [C, 2C]: the transposes of A (pairs of writes) and Aqk (a read after a write), side by side
    b_AT = tl.zeros([C, 2 * C], dtype=tl.float32)
    p0, p1 = tl.load(pair_offsets + i_r), tl.load(pair_offsets + i_r + 1)
    for i_p in range(p0, p1, BP):
        o_p = i_p + tl.arange(0, BP).to(tl.int64)
        m = o_p < p1
        b_i = tl.load(pair_i + o_p, mask=m, other=0)
        b_j = tl.load(pair_j + o_p, mask=m, other=0)
        b_w = exp(tl.load(G + b_i, mask=m, other=0) - tl.load(G + b_j, mask=m, other=0))
        b_w = tl.where(m, tl.load(val + b_i, mask=m, other=0) * tl.load(val + b_j, mask=m, other=0) * b_w, 0)
        b_row = tl.load(row + b_i, mask=m, other=0)
        # one-hot columns and rows, and the pair weights as three bfloat16 parts:
        # the products are exact for pair weights of magnitude in [2**-100, 2**127) (smaller ones lose bits or flush to zero),
        # and every pair lands in its cell in a fixed order, in float32
        b_oj = (o_c[:, None] == tl.load(row + b_j, mask=m, other=0)[None, :]).to(tl.bfloat16)
        b_oi = (b_row[:, None] == o_r[None, :]).to(tl.bfloat16)
        for _ in tl.static_range(3):
            b_part = b_w.to(tl.bfloat16)
            b_AT += tl.dot(b_oj, b_oi * b_part[:, None])
            b_w -= b_part.to(tl.float32)
    p_AA = AA + i_ch * 2 * C * C + o_r[None, :] * C + o_c[:, None]
    tl.store(p_AA, tl.load(p_AA) + b_AT)


@triton.jit
def chunk_sparse_delta_rule_fwd_kernel_intra_dense(
    slot,
    row,
    val,
    G,
    depth,
    rows,
    AA,
    E: tl.constexpr,
    C: tl.constexpr,
    BT: tl.constexpr,
    BJ: tl.constexpr,
):
    # the same sums for the (chunk, head) rows whose slots many entries share: over tiles of BT sorted entries i and the
    # windows of earlier entries j back to the start of their slots' groups, in blocks of BJ, the pair weights are formed
    # densely in tiles instead of being listed
    i_ch = tl.load(rows + tl.program_id(0)).to(tl.int64)
    o_c = tl.arange(0, C)
    o_r = tl.arange(0, 2 * C)
    b_AT = tl.zeros([C, 2 * C], dtype=tl.float32)
    for i0 in range(0, E, BT):
        o_i = i0 + tl.arange(0, BT)
        p_i = i_ch * E + o_i
        b_d = tl.load(depth + p_i)
        if tl.max(b_d, 0) > 0:
            b_si, b_ri, b_vi, b_Gi = tl.load(slot + p_i), tl.load(row + p_i), tl.load(val + p_i), tl.load(G + p_i)
            # [BT, C]: the weight of each entry with the write of each token on its slot, one term per sum
            b_Y = tl.zeros([BT, C], dtype=tl.float32)
            for j0 in range(tl.min(o_i - b_d, 0) // BJ * BJ, i0 + BT, BJ):
                p_j = i_ch * E + j0 + tl.arange(0, BJ)
                b_rj = tl.load(row + p_j)
                m = (b_si[:, None] == tl.load(slot + p_j)[None, :]) & (p_j[None, :] < p_i[:, None]) & (b_rj[None, :] < C)
                if tl.max(m.to(tl.int32)) > 0:
                    b_w = exp(b_Gi[:, None] - tl.load(G + p_j)[None, :])
                    b_w = tl.where(m, b_vi[:, None] * tl.load(val + p_j)[None, :] * b_w, 0)
                    b_oj = (b_rj[:, None] == o_c[None, :]).to(tl.bfloat16)
                    for _ in tl.static_range(3):
                        b_part = b_w.to(tl.bfloat16)
                        b_Y += tl.dot(b_part, b_oj)
                        b_w -= b_part.to(tl.float32)
            b_oi = (b_ri[:, None] == o_r[None, :]).to(tl.bfloat16)
            b_Y = tl.trans(b_Y)
            for _ in tl.static_range(3):
                b_part = b_Y.to(tl.bfloat16)
                b_AT += tl.dot(b_part, b_oi)
                b_Y -= b_part.to(tl.float32)
    tl.store(AA + i_ch * 2 * C * C + o_r[None, :].to(tl.int64) * C + o_c[:, None], b_AT)


@triton.jit
def _segsum1_combine(a, fa, b, fb):
    return tl.where(fb, b, a + b), fa | fb


@triton.jit
def _segsum_combine(a0, a1, a2, fa, b0, b1, b2, fb):
    return tl.where(fb, b0, a0 + b0), tl.where(fb, b1, a1 + b1), tl.where(fb, b2, a2 + b2), fa | fb


@triton.jit
def _segsum(x0, x1, x2, f, c0, c1, c2, BE: tl.constexpr):
    # inclusive sums over runs that start at the flagged entries, in a fixed order, continued from the previous block's
    # last sums c; returns the sums and the carries for the next block
    o = tl.arange(0, BE)
    m = (o == 0) & ~f
    x0, x1, x2 = tl.where(m, x0 + c0, x0), tl.where(m, x1 + c1, x1), tl.where(m, x2 + c2, x2)
    x0, x1, x2, _ = tl.associative_scan((x0, x1, x2, f), 0, _segsum_combine)
    m = o == BE - 1
    return x0, x1, x2, tl.sum(tl.where(m, x0, 0), 0), tl.sum(tl.where(m, x1, 0), 0), tl.sum(tl.where(m, x2, 0), 0)


@triton.jit
def chunk_sparse_delta_rule_prepare_kernel(
    key,
    perm,
    k_val,
    q_val,
    g,
    slot,
    row,
    val,
    G,
    depth,
    nd,
    wt,
    decay,
    last,
    rep,
    slot_w,
    tok_w,
    coef_w,
    decay_w,
    depth_w,
    last_w,
    wpos,
    npair,
    r0,
    k_wt,
    q_wt,
    k_coef,
    src_k,
    src_q,
    first,
    e_k,
    e_q,
    e_c,
    e_end,
    S: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
):
    # one program per (chunk, head), over its entries sorted by (slot, token, kind): forward, then backward over the slots
    i_ch = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BE)
    c0, c1, c2 = tl.zeros([], dtype=tl.float64), tl.zeros([], dtype=tl.int32), tl.zeros([], dtype=tl.int32)
    n_w = tl.zeros([], dtype=tl.int32)
    for i_e in range(0, E, BE):
        o_e = i_e + o
        m = o_e < E
        p_e = i_ch * E + o_e
        b_key = tl.load(key + p_e, mask=m, other=0)
        b_s, b_t, b_k = b_key // (2 * C), b_key % (2 * C) // 2, b_key % 2
        b_f = (o_e == 0) | (tl.load(key + p_e - 1, mask=m & (o_e > 0), other=0) // (2 * C) != b_s)
        b_p = tl.load(perm + p_e, mask=m, other=0)
        b_pt, b_pj = i_ch * C + b_p // (W + R), b_p % (W + R)
        b_v = tl.load(k_val + b_pt * W + b_pj, mask=m & (b_pj < W), other=0)
        b_v += tl.load(q_val + b_pt * R + b_pj - W, mask=m & (b_pj >= W), other=0)
        b_w = m & (b_k == 0)
        # G: the log decays of the writes of the slot up to the entry, summed in float64, kept in float32
        b_g = tl.load(g + i_ch * C + b_t, mask=b_w, other=0).to(tl.float64)
        b_G, b_nw, b_n, c0, c1, c2 = _segsum(b_g, b_w.to(tl.int32), m.to(tl.int32), b_f, c0, c1, c2, BE)
        b_eG = exp(b_G.to(tl.float32))
        b_wt = b_v * b_eG
        # the first write of each written slot saves its row at the chunk start, at the slot's first write rank
        b_rep = b_w & (b_nw == 1) & (b_s < S)
        b_cw = tl.cumsum(b_w.to(tl.int32), 0)
        b_r0 = n_w + b_cw - b_nw
        tl.store(slot + p_e, b_s, mask=m)
        tl.store(row + p_e, (b_t + C * b_k).to(tl.int32), mask=m)
        tl.store(val + p_e, b_v, mask=m)
        tl.store(G + p_e, b_G, mask=m)
        tl.store(depth + p_e, b_n - 1, mask=m)
        tl.store(wt + p_e, b_wt, mask=m)
        tl.store(rep + p_e, b_rep, mask=m)
        tl.store(npair + p_e, b_nw - b_w.to(tl.int32), mask=m)
        tl.store(r0 + p_e, b_r0, mask=m)
        tl.store(k_wt + b_pt * W + b_pj, b_wt, mask=m & (b_pj < W))
        tl.store(q_wt + b_pt * R + b_pj - W, b_wt, mask=m & (b_pj >= W))
        tl.store(e_k + b_pt * W + b_pj, b_eG, mask=m & (b_pj < W))
        tl.store(e_q + b_pt * R + b_pj - W, b_eG, mask=m & (b_pj >= W))
        tl.store(src_k + b_pt * W + b_pj, b_r0, mask=m & (b_pj < W))
        tl.store(src_q + b_pt * R + b_pj - W, b_r0, mask=m & (b_pj >= W))
        tl.store(first + b_pt * W + b_pj, b_rep, mask=m & (b_pj < W))
        # padding entries count as writes past the bank, after the real ones: the first C * W writes are kept
        b_r = n_w + b_cw - 1
        m_r = b_w & (b_r < C * W)
        p_r = i_ch * C * W + b_r
        tl.store(slot_w + p_r, b_s, mask=m_r)
        tl.store(tok_w + p_r, b_t.to(tl.int32), mask=m_r)
        tl.store(depth_w + p_r, b_nw - 1, mask=m_r)
        tl.store(wpos + p_r, o_e.to(tl.int32), mask=m_r)
        n_w += tl.sum(b_w.to(tl.int32), 0)
    tl.debug_barrier()
    b_z = tl.zeros([BE], dtype=tl.float64)
    c0, c1, c2 = tl.zeros([], dtype=tl.float64), tl.zeros([], dtype=tl.int32), tl.zeros([], dtype=tl.int32)
    n_a = tl.zeros([], dtype=tl.int32)
    for i_e in range(0, E, BE):
        o_e = E - 1 - i_e - o
        m = o_e >= 0
        p_e = i_ch * E + o_e
        b_s = tl.load(slot + p_e, mask=m, other=0)
        b_f = (o_e == E - 1) | (tl.load(slot + p_e + 1, mask=m & (o_e < E - 1), other=0) != b_s)
        b_w = m & (tl.load(row + p_e, mask=m, other=C) < C)
        # the writes and the entries from each entry to the last of its slot
        _, b_na, b_nr, c0, c1, c2 = _segsum(b_z, b_w.to(tl.int32), m.to(tl.int32), b_f, c0, c1, c2, BE)
        b_G = tl.load(G + p_e, mask=m, other=0)
        b_gend = tl.load(G + p_e + b_nr - 1, mask=m, other=0)
        b_ec = exp(b_gend - b_G)
        b_coef = tl.load(val + p_e, mask=m, other=0) * b_ec
        b_decay = exp(b_gend)
        tl.store(decay + p_e, b_decay, mask=m)
        tl.store(last + p_e, b_f & (b_s < S), mask=m)
        # per block of 64 sorted entries: the farthest partner of any entry, earlier or later (backward)
        b_far = tl.reshape(tl.maximum(tl.load(depth + p_e, mask=m, other=0), b_nr - 1), [BE // 64, 64])
        o_nd = (E - i_e) // 64 - 1 - tl.arange(0, BE // 64)
        tl.store(nd + i_ch * (E // 64) + o_nd, tl.max(b_far, 1), mask=o_nd >= 0)
        b_p = tl.load(perm + p_e, mask=m, other=0)
        b_pt, b_pj = i_ch * C + b_p // (W + R), b_p % (W + R)
        tl.store(k_coef + b_pt * W + b_pj, b_coef, mask=m & (b_pj < W))
        tl.store(e_c + b_pt * W + b_pj, b_ec, mask=m & (b_pj < W))
        tl.store(e_end + b_pt * W + b_pj, b_decay, mask=m & (b_pj < W))
        # a read of a slot without writes in the chunk reads the bank (-1), any other the slot's snapshot
        m_q = m & (b_pj >= W)
        b_r0 = tl.load(src_q + b_pt * R + b_pj - W, mask=m_q, other=0)
        m_q = m_q & ((b_r0 >= C * W) | (tl.load(slot_w + i_ch * C * W + b_r0, mask=m_q & (b_r0 < C * W), other=-1) != b_s))
        tl.store(src_q + b_pt * R + b_pj - W, -1, mask=m_q)
        b_r = n_w - n_a - tl.cumsum(b_w.to(tl.int32), 0)
        m_r = b_w & (b_r < C * W)
        p_r = i_ch * C * W + b_r
        tl.store(coef_w + p_r, b_coef, mask=m_r)
        tl.store(decay_w + p_r, b_decay, mask=m_r)
        tl.store(last_w + p_r, b_w & (b_na == 1) & (b_s < S), mask=m_r)
        n_a += tl.sum(b_w.to(tl.int32), 0)


@triton.jit
def chunk_sparse_delta_rule_bwd_kernel_intra(
    slot,
    row,
    perm,
    val,
    G,
    rep,
    nd,
    dA,
    dAqk,
    dkw,
    dqw,
    dc,
    dgend,
    dk_val,
    dq_val,
    dG,
    W: tl.constexpr,
    R: tl.constexpr,
    E: tl.constexpr,
    C: tl.constexpr,
    BE: tl.constexpr,
    BS: tl.constexpr,
):
    # one program per (chunk, head), over the sorted entries: the gradients of each weight and of G, then in reverse
    # those of g (G sums g over the earlier writes of the slot), written over dkw
    i_ch = tl.program_id(0).to(tl.int64)
    b_z = tl.zeros([BE], dtype=tl.int32)
    c0, c1 = tl.zeros([], dtype=tl.float64), tl.zeros([], dtype=tl.int32)
    for i_e in range(0, E, BE):
        o_e = i_e + tl.arange(0, BE).to(tl.int64)
        p_e = i_ch * E + o_e
        b_s, b_row = tl.load(slot + p_e), tl.load(row + p_e).to(tl.int64)
        b_k, b_t = b_row // C, b_row % C
        b_v, b_G = tl.load(val + p_e), tl.load(G + p_e)
        b_dv = tl.zeros([BE], dtype=tl.float32)
        b_dG = tl.zeros([BE], dtype=tl.float32)
        for d in range(1, tl.load(nd + i_ch * (E // BE) + i_e // BE) + 1):
            # as the later entry, paired with the write at distance d before it
            m = (o_e >= d) & (tl.load(slot + p_e - d, mask=o_e >= d, other=-1) == b_s)
            b_tj = tl.load(row + p_e - d, mask=m, other=C)
            m = m & (b_tj < C)
            b_w = exp(b_G - tl.load(G + p_e - d, mask=m, other=0))
            b_w = tl.where(m, tl.load(val + p_e - d, mask=m, other=0) * b_w, 0)
            b_g = tl.load(dA + i_ch * C * C + b_t * C + b_tj, mask=m & (b_k == 0), other=0)
            b_g += tl.load(dAqk + i_ch * C * C + b_t * C + b_tj, mask=m & (b_k == 1), other=0)
            # a zero weight passes no gradient, even from an overflowed contraction
            b_g = tl.where(b_w != 0, b_g * b_w, 0)
            b_dv += b_g
            b_dG += tl.where(b_v != 0, b_g * b_v, 0)
            # as the earlier write, paired with the entry at distance d after it
            m = (o_e + d < E) & (b_k == 0)
            m = m & (tl.load(slot + p_e + d, mask=m, other=-1) == b_s)
            b_rj = tl.load(row + p_e + d, mask=m, other=0).to(tl.int64)
            b_w = exp(tl.load(G + p_e + d, mask=m, other=0) - b_G)
            b_w = tl.where(m, tl.load(val + p_e + d, mask=m, other=0) * b_w, 0)
            b_g = tl.load(dA + i_ch * C * C + (b_rj % C) * C + b_t, mask=m & (b_rj < C), other=0)
            b_g += tl.load(dAqk + i_ch * C * C + (b_rj % C) * C + b_t, mask=m & (b_rj >= C), other=0)
            b_g = tl.where(b_w != 0, b_g * b_w, 0)
            b_dv += b_g
            b_dG -= tl.where(b_v != 0, b_g * b_v, 0)
        # the reads take val * exp(G) and the update val * exp(gend - G), gend being G at the last entry of the slot; the
        # row contractions dkw, dqw, dc and dgend come already scaled by exp(G), exp(G), exp(gend - G) and exp(gend)
        b_p = tl.load(perm + p_e)
        b_pt, b_pj = i_ch * C + b_p // (W + R), b_p % (W + R)
        m_k, m_q = b_pj < W, b_pj >= W
        b_dkw = tl.load(dkw + b_pt * W + b_pj, mask=m_k, other=0)
        b_dc = tl.load(dc + b_pt * W + b_pj, mask=m_k, other=0)
        b_dqw = tl.load(dqw + b_pt * R + b_pj - W, mask=m_q, other=0)
        tl.store(dk_val + b_pt * W + b_pj, b_dv + b_dkw + b_dc, mask=m_k)
        tl.store(dq_val + b_pt * R + b_pj - W, b_dv + b_dqw, mask=m_q)
        b_dG += tl.where(b_v != 0, (b_dkw + b_dqw - b_dc) * b_v, 0)
        # the end decay also scales the start row of each written slot
        b_dgend = tl.where(b_v != 0, b_dc * b_v, 0) + tl.load(dgend + b_pt * W + b_pj, mask=m_k & tl.load(rep + p_e), other=0)
        b_f = (o_e == 0) | (tl.load(slot + p_e - 1, mask=o_e > 0, other=-1) != b_s)
        b_l = (o_e == E - 1) | (tl.load(slot + p_e + 1, mask=o_e < E - 1, other=-1) != b_s)
        b_sum, _, _, c0, c1, _ = _segsum(b_dgend.to(tl.float64), b_z, b_z, b_f, c0, c1, c1, BE)
        tl.store(dG + p_e, b_dG + tl.where(b_l, b_sum, 0))
    tl.debug_barrier()
    b_z = tl.zeros([BS], dtype=tl.int32)
    c0, c1 = tl.zeros([], dtype=tl.float64), tl.zeros([], dtype=tl.int32)
    for i_e in range(0, E, BS):
        o_e = E - 1 - i_e - tl.arange(0, BS).to(tl.int64)
        m = o_e >= 0
        p_e = i_ch * E + o_e
        b_s = tl.load(slot + p_e, mask=m, other=0)
        b_l = (o_e == E - 1) | (tl.load(slot + p_e + 1, mask=m & (o_e < E - 1), other=-1) != b_s)
        b_sum, _, _, c0, c1, _ = _segsum(tl.load(dG + p_e, mask=m, other=0), b_z, b_z, b_l, c0, c1, c1, BS)
        b_p = tl.load(perm + p_e, mask=m, other=0)
        b_pj = b_p % (W + R)
        tl.store(dkw + (i_ch * C + b_p // (W + R)) * W + b_pj, b_sum.to(tl.float32), mask=m & (b_pj < W))


@triton.jit
def _stack(b_x, b_y, C: tl.constexpr, N: tl.constexpr):
    # [2C, N]: the rows of x, then those of y
    return tl.reshape(tl.permute(tl.join(b_x, b_y), (2, 0, 1)), [2 * C, N])


@triton.jit
def _scatter_rows(
    p_h,
    p_x,
    b_xt,
    slot,
    tok,
    wt,
    depth,
    last,
    decay,
    i_ch,
    o_v,
    m_v,
    V: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
    BV: tl.constexpr,
):
    # h[s] = decay[s] * h[s] + sum_e wt[e] * x[tok[e]] over the entries e of slot s, stored once by the last entry of s:
    # short groups by a look-back from their last entry over the rows of x in memory,
    # long ones by a segmented scan over rows gathered from the chunk's tile b_xt of x in registers,
    # both in a fixed order; c carries the partial sum of the group that runs into the next block
    b_c = tl.zeros([BV], dtype=tl.float32)
    for i_e in range(0, E, BE):
        o_e = i_e + tl.arange(0, BE).to(tl.int64)
        p_e = i_ch * E + o_e
        m_e = o_e < E
        b_last = tl.load(last + p_e, mask=m_e, other=0) != 0
        b_depth = tl.load(depth + p_e, mask=m_e, other=0)
        b_acc = b_last | (o_e == i_e + BE - 1)
        n_d = tl.max(tl.where(b_acc, b_depth, 0), 0)
        if n_d < 8:
            b_x = tl.zeros([BE, BV], dtype=tl.float32)
            for d in range(0, n_d + 1, 2):
                # the distances n_d - d and n_d - d - 1 together, the farther first
                m1 = b_acc & m_e & (b_depth >= n_d - d)
                m2 = b_acc & m_e & (b_depth >= n_d - d - 1) & (d < n_d)
                p1 = p_e - (n_d - d)
                p_x1 = p_x + tl.load(tok + p1, mask=m1, other=0).to(tl.int64)[:, None] * V + o_v[None, :]
                p_x2 = p_x + tl.load(tok + p1 + 1, mask=m2, other=0).to(tl.int64)[:, None] * V + o_v[None, :]
                b_x1 = tl.load(p_x1, mask=m1[:, None] & m_v[None, :], other=0)
                b_x2 = tl.load(p_x2, mask=m2[:, None] & m_v[None, :], other=0)
                b_x += tl.load(wt + p1, mask=m1, other=0)[:, None] * b_x1
                b_x += tl.load(wt + p1 + 1, mask=m2, other=0)[:, None] * b_x2
        else:
            b_t = tl.broadcast_to(tl.load(tok + p_e, mask=m_e, other=0)[:, None], [BE, BV])
            b_x = tl.where(m_e[:, None], tl.gather(b_xt, b_t, 0), 0)
            b_x *= tl.load(wt + p_e, mask=m_e, other=0)[:, None]
            b_f = (b_depth == 0) | ~m_e
            b_x = tl.where(((o_e == i_e) & ~b_f)[:, None], b_x + b_c[None, :], b_x)
            b_x, _ = tl.associative_scan((b_x, tl.broadcast_to(b_f[:, None], [BE, BV])), 0, _segsum1_combine)
        b_c = tl.sum(tl.where((o_e == i_e + BE - 1)[:, None], b_x, 0), 0)
        p_row = p_h + tl.load(slot + p_e, mask=b_last, other=0)[:, None] * V + o_v[None, :]
        m_row = b_last[:, None] & m_v[None, :]
        b_row = tl.load(p_row, mask=m_row, other=0).to(tl.float32) * tl.load(decay + p_e, mask=b_last, other=0)[:, None]
        tl.store(p_row, (b_row + b_x).to(p_row.dtype.element_ty), mask=m_row)


@triton.jit
def _gather_rows(p_h, idx, wt, snap, src, first, i_ch, o_c, o_v, m_v, N: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
                 BW: tl.constexpr, BV: tl.constexpr, SAVE: tl.constexpr):
    # [C, BV]: sum_n wt[t, n] * h[idx[t, n]]; with SAVE, the first write of each slot also saves the row as its snapshot
    b_o = tl.zeros([C, BV], dtype=tl.float32)
    for i_n in range(0, N, BW):
        o_n = i_n + tl.arange(0, BW).to(tl.int64)
        p_n = i_ch * C * N + o_c[:, None] * N + o_n[None, :]
        m_n = o_n[None, :] < N
        b_w = tl.load(wt + p_n, mask=m_n, other=0)
        b_i = tl.load(idx + p_n, mask=m_n, other=0).to(tl.int64)
        m = (m_n & (b_i >= 0))[:, :, None] & m_v[None, None, :]
        b_rows = tl.load(p_h + b_i[:, :, None] * V + o_v[None, None, :], mask=m, other=0)
        if SAVE:
            p_s = snap + (i_ch * C * N + tl.load(src + p_n, mask=m_n, other=0))[:, :, None] * V + o_v[None, None, :]
            tl.store(p_s, b_rows, mask=m & tl.load(first + p_n, mask=m_n, other=0)[:, :, None])
        b_o += tl.sum(b_w[:, :, None] * b_rows.to(tl.float32), 1)
    return b_o


@triton.jit
def chunk_sparse_delta_rule_fwd_kernel_h(
    k_idx,
    k_wt,
    q_idx,
    q_wt,
    v,
    beta,
    Tm,
    Aqk,
    slot,
    tok,
    coef,
    depth,
    last,
    decay,
    src_k,
    first,
    o,
    u,
    snap,
    h,
    chunk_offsets,
    H: tl.constexpr,
    S: tl.constexpr,
    V: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    E: tl.constexpr,
    BW: tl.constexpr,
    BE: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    SAVE_SNAP: tl.constexpr,
):
    # one program per (sequence, head, V block): the chunks in order, the bank advanced in place
    i_v, i_nh = tl.program_id(0).to(tl.int64) % NV, tl.program_id(0).to(tl.int64) // NV
    i_n, i_h = i_nh // H, i_nh % H
    o_c = tl.arange(0, C).to(tl.int64)
    o_v = i_v * BV + tl.arange(0, BV).to(tl.int64)
    m_v = o_v < V
    p_h = h + i_nh * S * V
    for i_c in range(tl.load(chunk_offsets + i_n), tl.load(chunk_offsets + i_n + 1)):
        i_ch = i_c * H + i_h
        b_r = _gather_rows(p_h, k_idx, k_wt, snap, src_k, first, i_ch, o_c, o_v, m_v, W, V, C, BW, BV, SAVE_SNAP)
        b_q = _gather_rows(p_h, q_idx, q_wt, snap, src_k, first, i_ch, o_c, o_v, m_v, R, V, C, BW, BV, False)
        p_cc = i_ch * C * C + o_c[:, None] * C + o_c[None, :]
        p_cv = i_ch * C * V + o_c[:, None] * V + o_v[None, :]
        b_x = tl.load(beta + i_ch * C + o_c)[:, None] * (tl.load(v + p_cv, mask=m_v[None, :], other=0).to(tl.float32) - b_r)
        b_u = tl.dot(tl.load(Tm + p_cc), b_x, input_precision='ieee')
        b_o = b_q + tl.dot(tl.load(Aqk + p_cc), b_u, input_precision='ieee')
        tl.store(o + p_cv, b_o.to(o.dtype.element_ty), mask=m_v[None, :])
        tl.store(u + p_cv, b_u, mask=m_v[None, :])
        tl.debug_barrier()
        _scatter_rows(p_h, u + i_ch * C * V, b_u, slot, tok, coef, depth, last, decay, i_ch, o_v, m_v, V, E, BE, BV)
        tl.debug_barrier()


@triton.jit
def chunk_sparse_delta_rule_bwd_kernel_h(
    k_idx,
    k_wt,
    k_coef,
    q_idx,
    q_wt,
    v,
    beta,
    Tm,
    Aqk,
    u,
    snap,
    slot,
    tok,
    wt,
    depth,
    last,
    decay,
    src_k,
    src_q,
    first,
    e_k,
    e_q,
    e_c,
    e_end,
    do,
    h,
    dh,
    dv,
    dx,
    dAqk,
    dL,
    dbeta,
    dkw,
    dqw,
    dc,
    dgend,
    chunk_offsets,
    NTH,
    H: tl.constexpr,
    S: tl.constexpr,
    V: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    E: tl.constexpr,
    BW: tl.constexpr,
    BE: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
):
    # one program per (sequence, head, V block): the chunks in reverse, the bank restored and its gradient carried in place;
    # contractions over V are written per V block and summed afterwards in a fixed order
    i_v, i_nh = tl.program_id(0).to(tl.int64) % NV, tl.program_id(0).to(tl.int64) // NV
    i_n, i_h = i_nh // H, i_nh % H
    o_c = tl.arange(0, C).to(tl.int64)
    o_v = i_v * BV + tl.arange(0, BV).to(tl.int64)
    m_v = o_v < V
    p_h = h + i_nh * S * V
    p_dh = dh + i_nh * S * V
    c0, c1 = tl.load(chunk_offsets + i_n), tl.load(chunk_offsets + i_n + 1)
    for i_r in range(c0, c1):
        i_c = c1 - 1 - (i_r - c0)
        i_ch = i_c * H + i_h
        p_cc = i_ch * C * C + o_c[:, None] * C + o_c[None, :]
        p_cv = i_ch * C * V + o_c[:, None] * V + o_v[None, :]
        # the written rows at the chunk start come from their snapshots, which their first writes restore in the bank
        b_u = tl.load(u + p_cv, mask=m_v[None, :], other=0)
        b_r = tl.zeros([C, BV], dtype=tl.float32)
        b_du = tl.zeros([C, BV], dtype=tl.float32)
        for i_w in range(0, W, BW):
            o_w = i_w + tl.arange(0, BW).to(tl.int64)
            p_w = i_ch * C * W + o_c[:, None] * W + o_w[None, :]
            m_w = o_w[None, :] < W
            b_i = tl.load(k_idx + p_w, mask=m_w, other=0)
            m = (m_w & (b_i >= 0))[:, :, None] & m_v[None, None, :]
            p_s = snap + (i_ch * C * W + tl.load(src_k + p_w, mask=m_w, other=0))[:, :, None] * V + o_v[None, None, :]
            b_rows = tl.load(p_s, mask=m, other=0)
            m_f = m & tl.load(first + p_w, mask=m_w, other=0)[:, :, None]
            tl.store(p_h + b_i[:, :, None] * V + o_v[None, None, :], b_rows, mask=m_f)
            b_rows = b_rows.to(tl.float32)
            b_drows = tl.load(p_dh + b_i[:, :, None] * V + o_v[None, None, :], mask=m, other=0)
            b_r += tl.sum(tl.load(k_wt + p_w, mask=m_w, other=0)[:, :, None] * b_rows, 1)
            b_du += tl.sum(tl.load(k_coef + p_w, mask=m_w, other=0)[:, :, None] * b_drows, 1)
            # the decays scale a factor before each contraction, so a contraction overflows only if its result does
            b_ec = tl.load(e_c + p_w, mask=m_w, other=0)[:, :, None]
            tl.store(dc + i_v * NTH * C * W + p_w, tl.sum(b_drows * b_ec * b_u[:, None, :], 2), mask=m_w)
            b_end = tl.load(e_end + p_w, mask=m_w, other=0)[:, :, None]
            tl.store(dgend + i_v * NTH * C * W + p_w, tl.sum(b_drows * (b_rows * b_end), 2), mask=m_w)
        b_vr = tl.load(v + p_cv, mask=m_v[None, :], other=0).to(tl.float32) - b_r
        b_beta = tl.load(beta + i_ch * C + o_c)
        b_do = tl.load(do + p_cv, mask=m_v[None, :], other=0).to(tl.float32)
        b_du += tl.dot(tl.trans(tl.load(Aqk + p_cc)), b_do, input_precision='ieee')
        b_dx = tl.dot(tl.trans(tl.load(Tm + p_cc)), b_du, input_precision='ieee')
        tl.store(dv + p_cv, (b_beta[:, None] * b_dx).to(dv.dtype.element_ty), mask=m_v[None, :])
        # [2C, BV], in memory and as a tile: the gradients of the written-row reads (r) then of the read-row reads (o)
        b_xt = _stack(-b_beta[:, None] * b_dx, b_do, C, BV)
        p_x = dx + i_ch * 2 * C * V + o_c[:, None] * V + o_v[None, :]
        tl.store(p_x, -b_beta[:, None] * b_dx, mask=m_v[None, :])
        tl.store(p_x + C * V, b_do, mask=m_v[None, :])

        p_vcc = i_v * NTH * C * C + p_cc
        tl.store(dAqk + p_vcc, tl.dot(b_do, tl.trans(b_u), input_precision='ieee'))
        # through the solve u = (I + diag(beta) A)^-1 x: the gradient of (I + diag(beta) A) is -dx u^T
        tl.store(dL + p_vcc, -tl.dot(b_dx, tl.trans(b_u), input_precision='ieee'))
        tl.store(dbeta + i_v * NTH * C + i_ch * C + o_c, tl.sum(b_dx * b_vr, 1))
        for i_w in range(0, W, BW):
            o_w = i_w + tl.arange(0, BW).to(tl.int64)
            p_w = i_ch * C * W + o_c[:, None] * W + o_w[None, :]
            m_w = o_w[None, :] < W
            b_i = tl.load(k_idx + p_w, mask=m_w, other=0)
            m = (m_w & (b_i >= 0))[:, :, None] & m_v[None, None, :]
            p_s = snap + (i_ch * C * W + tl.load(src_k + p_w, mask=m_w, other=0))[:, :, None] * V + o_v[None, None, :]
            b_rows = tl.load(p_s, mask=m, other=0).to(tl.float32) * tl.load(e_k + p_w, mask=m_w, other=0)[:, :, None]
            tl.store(dkw + i_v * NTH * C * W + p_w, tl.sum(-(b_beta[:, None] * b_dx)[:, None, :] * b_rows, 2), mask=m_w)
        for i_w in range(0, R, BW):
            o_w = i_w + tl.arange(0, BW).to(tl.int64)
            p_w = i_ch * C * R + o_c[:, None] * R + o_w[None, :]
            b_i = tl.load(q_idx + p_w, mask=o_w[None, :] < R, other=0).to(tl.int64)
            b_src = tl.load(src_q + p_w, mask=o_w[None, :] < R, other=-1)
            m = ((o_w[None, :] < R) & (b_i >= 0))[:, :, None] & m_v[None, None, :]
            # the start rows: the snapshot of a slot written in the chunk, the bank for any other
            p_s = snap + (i_ch * C * W + b_src)[:, :, None] * V + o_v[None, None, :]
            b_rows = tl.load(p_s, mask=m & (b_src >= 0)[:, :, None], other=0)
            b_rows += tl.load(p_h + b_i[:, :, None] * V + o_v[None, None, :], mask=m & (b_src < 0)[:, :, None], other=0)
            b_rows = b_rows.to(tl.float32) * tl.load(e_q + p_w, mask=o_w[None, :] < R, other=0)[:, :, None]
            tl.store(dqw + i_v * NTH * C * R + p_w, tl.sum(b_do[:, None, :] * b_rows, 2), mask=o_w[None, :] < R)
        tl.debug_barrier()
        # dh_start[s] = decay[s] * dh_end[s] + the gradients of the reads of row s, written once per touched row
        _scatter_rows(p_dh, dx + i_ch * 2 * C * V, b_xt, slot, tok, wt, depth, last, decay, i_ch, o_v, m_v, V, E, BE, BV)
        tl.debug_barrier()


def chunk_sparse_delta_rule_prepare(q_idx, q_val, k_idx, k_val, v, g, beta, num_slots, cu_seqlens, chunk_size):
    # chunk-major [NT, H, C, ...] inputs and the slot-sorted entries of every (chunk, head)
    S, C = num_slots, chunk_size
    W, R, H = k_idx.shape[-1], q_idx.shape[-1], v.shape[2]
    E = C * (W + R)
    N = len(cu_seqlens) - 1
    nc = (cu_seqlens.diff().long() + C - 1) // C
    chunk_offsets = torch.cat([nc.new_zeros(1), nc.cumsum(0)])
    NT = int(chunk_offsets[-1])
    i_n = torch.repeat_interleave(torch.arange(N, device=v.device), nc, output_size=NT)
    pos = cu_seqlens[i_n, None] + (torch.arange(NT, device=v.device) - chunk_offsets[i_n])[:, None] * C
    pos = pos + torch.arange(C, device=v.device)
    valid = pos < cu_seqlens[i_n + 1, None]
    pos = pos.clamp(max=v.shape[1] - 1)

    def chunks(x, fill=0):
        x = x[0][pos]
        x = torch.where(valid.view(NT, C, *([1] * (x.dim() - 2))), x, fill)
        return x.transpose(1, 2).contiguous()

    # padding tokens read no bank row: a 0 weight on a non-finite row would still give NaN
    k_idx, q_idx, v = chunks(k_idx.long(), -1), chunks(q_idx.long(), -1), chunks(v)
    k_val, q_val, g, beta = (chunks(x.float()) for x in (k_val, q_val, g, beta))
    # exp(g) is 0 in float32 below -104: clamping there keeps every float32 result and bounds the float64 sums of g
    g_live = g > -128
    g = g.clamp(min=-128)
    t = torch.arange(C, device=v.device).view(1, 1, C, 1)
    key = torch.cat([k_idx * (2 * C) + 2 * t, q_idx * (2 * C) + 2 * t + 1], -1)
    # padding tokens get distinct slots past the bank, so they pair with nothing and sort last
    pad = (S + t * (W + R) + torch.arange(W + R, device=v.device)) * (2 * C)
    key = torch.where(valid.view(NT, 1, C, 1), key, pad).flatten(-2)
    # 32-bit keys sort faster
    key, perm = (key.int() if (S + E) * 2 * C < 2**31 else key).sort(-1)
    p = dict(NT=NT, E=E, chunk_offsets=chunk_offsets, pos=pos, valid=valid, perm=perm, k_idx=k_idx, q_idx=q_idx, v=v,
             beta=beta, g_live=g_live)
    long, i32, f32, b1 = torch.long, torch.int32, torch.float, torch.bool
    # [NT, H, n] outputs: in sorted order (n = E), in write order (n = C * W), in entry order (n = C * W or C * R)
    outputs = dict(
        slot=(E, long), row=(E, i32), val=(E, f32), G=(E, f32), depth=(E, i32), nd=(E // 64, i32),
        wt=(E, f32), decay=(E, f32), last=(E, b1), rep=(E, b1), npair=(E, i32), r0=(E, i32),
        slot_w=(C * W, long), tok_w=(C * W, i32), coef_w=(C * W, f32), decay_w=(C * W, f32), depth_w=(C * W, i32),
        last_w=(C * W, b1), wpos=(C * W, i32),
        k_wt=(C * W, f32), q_wt=(C * R, f32), k_coef=(C * W, f32), src_k=(C * W, i32), src_q=(C * R, i32),
        first=(C * W, b1), e_k=(C * W, f32), e_q=(C * R, f32), e_c=(C * W, f32), e_end=(C * W, f32),
    )
    p.update({name: v.new_empty(NT, H, n, dtype=dtype) for name, (n, dtype) in outputs.items()})
    chunk_sparse_delta_rule_prepare_kernel[(NT * H,)](
        key=key,
        perm=perm,
        k_val=k_val,
        q_val=q_val,
        g=g,
        **{name: p[name] for name in outputs},
        S=S,
        W=W,
        R=R,
        C=C,
        E=E,
        BE=1024,
    )
    return p


def _pair_list(npair: torch.Tensor, r0: torch.Tensor, wpos: torch.Tensor, E: int, e0: int, e1: int):
    # the pairs (i, j) of a flat sorted entry i in [e0, e1) and an earlier write j of its slot, ordered by (i, j), as
    # flat entry indices, with the offsets of the pairs of each row met; the earlier writes of entry i are the writes
    # r0[i] + k, k < npair[i], of the write order, at the sorted positions wpos
    i = npair[e0:e1].nonzero().squeeze(1) + e0
    n = npair[i].long()
    total = int(n.sum())
    pair_i = torch.repeat_interleave(i, n, output_size=total)
    k = torch.arange(total, device=n.device) - torch.repeat_interleave(n.cumsum(0) - n, n, output_size=total)
    rows = pair_i // E
    pair_j = rows * E + wpos.flatten()[rows * wpos.shape[-1] + r0[pair_i] + k]
    return pair_i, pair_j, torch.searchsorted(pair_i, torch.arange(e0 // E, (e1 - 1) // E + 2, device=n.device) * E)


def _bv(V, n):
    # the widest block of value columns (16, 32 or 64; 16 is the narrowest output tile of tl.dot) that still gives the
    # state kernels 128 programs over the n (sequence, head) pairs; wider blocks gather longer row slices
    BV = next(b for b in (64, 32, 16) if b == 16 or n * triton.cdiv(V, b) >= 128)
    return BV, triton.cdiv(V, BV)


@dispatch
def chunk_sparse_delta_rule_fwd_h(p, Tm, Aqk, o, u, snap, h):
    # the sequential pass over the chunks: o, u and (if snap is given) the snapshots, the bank h advanced in place
    N, H, S, V = h.shape
    NT, _, C, W = p['k_idx'].shape
    R = p['q_idx'].shape[-1]
    BV, NV = _bv(V, N * H)
    chunk_sparse_delta_rule_fwd_kernel_h[(NV * N * H,)](
        k_idx=p['k_idx'],
        k_wt=p['k_wt'],
        q_idx=p['q_idx'],
        q_wt=p['q_wt'],
        v=p['v'],
        beta=p['beta'],
        Tm=Tm,
        Aqk=Aqk,
        slot=p['slot_w'],
        tok=p['tok_w'],
        coef=p['coef_w'],
        depth=p['depth_w'],
        last=p['last_w'],
        decay=p['decay_w'],
        src_k=p['src_k'],
        first=p['first'],
        o=o,
        u=u,
        snap=snap,
        h=h,
        chunk_offsets=p['chunk_offsets'],
        H=H,
        S=S,
        V=V,
        W=W,
        R=R,
        C=C,
        E=C * W,
        BW=256 // BV,
        BE=16384 // BV,
        BV=BV,
        NV=NV,
        SAVE_SNAP=snap is not None,
        num_warps=8,
    )


def chunk_sparse_delta_rule_fwd(q_idx, q_val, k_idx, k_val, v, g, beta, initial_state, cu_seqlens, chunk_size=64,
                                save_for_backward=False):
    N, H, S, V = initial_state.shape
    W, C = k_idx.shape[-1], chunk_size
    p = chunk_sparse_delta_rule_prepare(
        q_idx=q_idx,
        q_val=q_val,
        k_idx=k_idx,
        k_val=k_val,
        v=v,
        g=g,
        beta=beta,
        num_slots=S,
        cu_seqlens=cu_seqlens,
        chunk_size=C,
    )
    NT, E = p['NT'], p['E']
    AA = v.new_zeros(NT, H, 2 * C, C, dtype=torch.float)
    # rows with few pairs take a pair list, in ranges of entries of about 2**26 pairs at most; rows whose slots many
    # entries share take the dense kernel, which forms the pair weights in tiles instead of listing them
    npair, r0 = p['npair'].view(NT * H, E), p['r0'].flatten()
    dense = npair.sum(-1) > 4096
    npair = torch.where(dense[:, None], 0, npair).flatten()
    count = npair.cumsum(0)
    total = int(count[-1]) if count.numel() else 0
    bounds = torch.searchsorted(count, torch.arange(1, total // 2**26 + 1, device=v.device) * 2**26, right=True).tolist()
    for e0, e1 in zip([0, *bounds], [*bounds, NT * H * E]):
        if e0 == e1:
            continue
        pair_i, pair_j, pair_offsets = _pair_list(npair, r0, p['wpos'], E, e0, e1)
        chunk_sparse_delta_rule_fwd_kernel_intra[(pair_offsets.numel() - 1,)](
            row=p['row'],
            val=p['val'],
            G=p['G'],
            pair_i=pair_i,
            pair_j=pair_j,
            pair_offsets=pair_offsets,
            AA=AA,
            row0=e0 // E,
            C=C,
            BP=64,
        )
    rows = dense.nonzero().squeeze(1)
    if rows.numel():
        chunk_sparse_delta_rule_fwd_kernel_intra_dense[(rows.numel(),)](
            slot=p['slot'],
            row=p['row'],
            val=p['val'],
            G=p['G'],
            depth=p['depth'],
            rows=rows,
            AA=AA,
            E=E,
            C=C,
            BT=64,
            BJ=32,
        )
    A, Aqk = AA[:, :, :C].contiguous(), AA[:, :, C:].contiguous()
    eye = torch.eye(C, device=v.device, dtype=A.dtype)
    Tm = torch.linalg.solve_triangular(
        eye + p['beta'][..., None] * A,
        eye.expand_as(A),
        upper=False,
        unitriangular=True,
    ).contiguous()

    o = v.new_empty(NT, H, C, V)
    u = v.new_empty(NT, H, C, V, dtype=torch.float)
    snap = initial_state.new_empty(NT, H, C, W, V) if save_for_backward else None
    h = initial_state.clone()
    chunk_sparse_delta_rule_fwd_h(p, Tm, Aqk, o, u, snap, h)
    o = o.transpose(1, 2)[p['valid']].unsqueeze(0)
    return o, h, ((p, A, Aqk, Tm, u, snap) if save_for_backward else None)


# the prepared tensors the backward reads
SAVED = ('chunk_offsets', 'pos', 'valid', 'perm', 'k_idx', 'q_idx', 'v', 'beta', 'g_live', 'slot', 'row', 'val', 'G',
         'depth', 'nd', 'wt', 'decay', 'last', 'rep', 'k_wt', 'q_wt', 'k_coef', 'src_k', 'src_q', 'first', 'e_k', 'e_q', 'e_c',
         'e_end')


@dispatch
def chunk_sparse_delta_rule_bwd_h(p, Tm, Aqk, u, snap, do, h, dh):
    # the sequential pass over the chunks in reverse: the bank h restored and its gradient dh carried in place; returns
    # dv and the contractions over V, summed over the value blocks
    N, H, S, V = h.shape
    NT, _, C, W = p['k_idx'].shape
    R, E = p['q_idx'].shape[-1], p['E']
    BV, NV = _bv(V, N * H)
    dv = torch.empty_like(p['v'])
    dx = h.new_empty(NT, H, 2 * C, V, dtype=torch.float)
    dAqk = h.new_empty(NV, NT, H, C, C, dtype=torch.float)
    dL = torch.empty_like(dAqk)
    dbeta = h.new_empty(NV, NT, H, C, dtype=torch.float)
    dkw = h.new_empty(NV, NT, H, C, W, dtype=torch.float)
    dc, dgend = torch.empty_like(dkw), torch.empty_like(dkw)
    dqw = h.new_empty(NV, NT, H, C, R, dtype=torch.float)
    chunk_sparse_delta_rule_bwd_kernel_h[(NV * N * H,)](
        k_idx=p['k_idx'],
        k_wt=p['k_wt'],
        k_coef=p['k_coef'],
        q_idx=p['q_idx'],
        q_wt=p['q_wt'],
        v=p['v'],
        beta=p['beta'],
        Tm=Tm,
        Aqk=Aqk,
        u=u,
        snap=snap,
        slot=p['slot'],
        tok=p['row'],
        wt=p['wt'],
        depth=p['depth'],
        last=p['last'],
        decay=p['decay'],
        src_k=p['src_k'],
        src_q=p['src_q'],
        first=p['first'],
        e_k=p['e_k'],
        e_q=p['e_q'],
        e_c=p['e_c'],
        e_end=p['e_end'],
        do=do,
        h=h,
        dh=dh,
        dv=dv,
        dx=dx,
        dAqk=dAqk,
        dL=dL,
        dbeta=dbeta,
        dkw=dkw,
        dqw=dqw,
        dc=dc,
        dgend=dgend,
        chunk_offsets=p['chunk_offsets'],
        NTH=NT * H,
        H=H,
        S=S,
        V=V,
        W=W,
        R=R,
        C=C,
        E=E,
        BW=256 // BV,
        BE=16384 // BV,
        BV=BV,
        NV=NV,
        num_warps=8,
    )
    return (dv, *(x.sum(0) for x in (dAqk, dL, dbeta, dkw, dqw, dc, dgend)))


def chunk_sparse_delta_rule_bwd(do, dht, h, saved, chunk_size=64):
    p, A, Aqk, Tm, u, snap = saved
    N, H, S, V = h.shape
    NT, E, C = p['NT'], p['E'], chunk_size
    W, R = p['k_idx'].shape[-1], p['q_idx'].shape[-1]
    do = torch.where(p['valid'].view(NT, C, 1, 1), do[0][p['pos']], 0).transpose(1, 2).contiguous()
    h = h.clone()
    dh = torch.zeros(N, H, S, V, device=h.device, dtype=torch.float) if dht is None else dht.float().clone()
    dv, dAqk, dL, dbeta, dkw, dqw, dc, dgend = chunk_sparse_delta_rule_bwd_h(p, Tm, Aqk, u, snap, do, h, dh)
    dL = dL.tril(-1)
    dbeta = dbeta + torch.where(A != 0, dL * A, 0).sum(-1)
    dk_val, dq_val = torch.empty_like(dkw), torch.empty_like(dqw)
    chunk_sparse_delta_rule_bwd_kernel_intra[(NT * H,)](
        slot=p['slot'],
        row=p['row'],
        perm=p['perm'],
        val=p['val'],
        G=p['G'],
        rep=p['rep'],
        nd=p['nd'],
        dA=p['beta'][..., None] * dL,
        dAqk=dAqk,
        dkw=dkw,
        dqw=dqw,
        dc=dc,
        dgend=dgend,
        dk_val=dk_val,
        dq_val=dq_val,
        dG=torch.empty_like(p['G'], dtype=torch.double),
        W=W,
        R=R,
        E=E,
        C=C,
        BE=64,
        BS=1024,
    )
    # dkw now holds the gradient of the log decay of each write
    dg = torch.where(p['g_live'], dkw.sum(-1), 0)

    def unchunk(x):
        return x.transpose(1, 2)[p['valid']].unsqueeze(0)

    return (unchunk(dq_val), unchunk(dk_val), unchunk(dv), unchunk(dg), unchunk(dbeta), dh)


def check_initial_state(initial_state, v, cu_seqlens):
    N = v.shape[0] if cu_seqlens is None else len(cu_seqlens) - 1
    H, V = v.shape[2], v.shape[3]
    if initial_state.ndim != 4 or initial_state.shape[2] < 1 or tuple(initial_state.shape[i] for i in (0, 1, 3)) != (N, H, V):
        raise ValueError(
            f"One bank per input sequence is expected: `initial_state` of shape [{N}, {H}, S, {V}] with S >= 1 "
            f"rather than {list(initial_state.shape)}.",
        )


class ChunkSparseDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q_idx, q_val, k_idx, k_val, v, g, beta, initial_state, output_final_state, cu_seqlens):
        B, T = v.shape[:2]
        dense = cu_seqlens is None
        if dense:
            cu_seqlens = torch.arange(B + 1, device=v.device) * T
            q_idx, q_val, k_idx, k_val, v, g, beta = (x.flatten(0, 1)[None] for x in (q_idx, q_val, k_idx, k_val, v, g, beta))
        o, ht, saved = chunk_sparse_delta_rule_fwd(
            q_idx=q_idx,
            q_val=q_val,
            k_idx=k_idx,
            k_val=k_val,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            save_for_backward=any(ctx.needs_input_grad),
        )
        if saved is not None:
            p, *tensors = saved
            ctx.save_for_backward(ht, *(p[k] for k in SAVED), *tensors)
            ctx.NT, ctx.E = p['NT'], p['E']
        ctx.shape, ctx.dense = (B, T), dense
        ctx.dtypes = (q_val.dtype, k_val.dtype, v.dtype, g.dtype, beta.dtype, initial_state.dtype)
        if dense:
            o = o.view(B, T, *o.shape[2:])
        return o, (ht if output_final_state else None)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        ht, *tensors = ctx.saved_tensors
        p = dict(zip(SAVED, tensors), NT=ctx.NT, E=ctx.E)
        if ctx.dense:
            do = do.flatten(0, 1)[None]
        grads = chunk_sparse_delta_rule_bwd(do=do, dht=dht, h=ht, saved=(p, *tensors[len(SAVED):]))
        grads = [x.view(*ctx.shape, *x.shape[2:]) if ctx.dense else x for x in grads[:5]] + [grads[5]]
        dq_val, dk_val, dv, dg, dbeta, dh0 = (x.to(dtype) for x, dtype in zip(grads, ctx.dtypes))
        return None, dq_val, None, dk_val, dv, dg, dbeta, dh0, None, None


def chunk_sparse_delta_rule(
    q_idx: torch.Tensor,
    q_val: torch.Tensor,
    k_idx: torch.Tensor,
    k_val: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    The gated delta rule over a bank of `S` slots per head, with sparse keys and queries: each token writes
    `W` slots and reads `R` slots, with weights. Per token and head, only the written slots decay:

        h[k_idx] = exp(g) * h[k_idx]
        h[k_idx] += k_val * beta * (v - sum_w k_val[w] * h[k_idx[w]])
        o = sum_r q_val[r] * h[q_idx[r]]

    This is the gated delta rule of KDA with one-hot-weighted keys and queries over `K = S` channels,
    whose per-channel decay is `g` on the written slots and zero elsewhere.
    The bank is kept in the dtype of `initial_state`: each written row is rounded to it once per chunk of 64 tokens
    (pass a float32 bank for float32 state). Every sum runs in a fixed order and every bank element has one writer, so
    outputs, final bank and gradients are bitwise reproducible for fixed inputs, shapes included, and GPU type.
    Each chunk inverts `I + diag(beta) A` in float32, where `A` sums decayed products of write weights over shared slots:
    with write weights of L2 norm at most 1 per token (softmax weights qualify) and `beta` in `[0, 1]`,
    every entry of `diag(beta) A` is at most 1 in magnitude (up to rounding) and the inverse cannot overflow;
    read weights of L1 norm at most 1 per token bound the read-write products the same way.
    Products of two weights below `2**-100` in magnitude may be flushed to zero there.
    The backward forms float32 products of gradients with values and bank rows before applying the weights:
    keep values, bank and gradients below about `2**40` in magnitude.
    As in the other chunked delta-rule kernels, the intra-chunk solve works to float32 precision:
    values of a chunk that differ by more than about `2**20` in magnitude, and results that cancel exactly,
    are reproduced to float32 rounding rather than exactly.
    The log decay of each slot since the chunk start is summed in float64 and kept in float32,
    like the cumulative gates of KDA.
    There is no `cu_seqlens`: each sequence of the batch carries its own bank, which is never reset inside it, so
    documents packed into one sequence share its bank (a reset per document would need a copy of the whole bank per
    document).

    Args:
        q_idx (torch.Tensor):
            Read slots of shape `[B, T, H, R]`, in `[0, S)`.
        q_val (torch.Tensor):
            Read weights of shape `[B, T, H, R]`, of L1 norm at most 1 per token (see above).
        k_idx (torch.Tensor):
            Written slots of shape `[B, T, H, W]`, in `[0, S)` and distinct within a token; `W + R` below `2**25`.
        k_val (torch.Tensor):
            Write weights of shape `[B, T, H, W]`, of L2 norm at most 1 per token (see above).
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Log decays (at most 0) of the written slots, of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`, in `[0, 1]`.
        initial_state (torch.Tensor):
            Initial bank of shape `[B, H, S, V]`, one per sequence; it receives a gradient.
        output_final_state (bool, Optional):
            Whether to output the final bank of shape `[B, H, S, V]`. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final bank of shape `[B, H, S, V]` if `output_final_state=True` else `None`.
    """
    return _chunk_sparse_delta_rule(q_idx, q_val, k_idx, k_val, v, g, beta, initial_state, output_final_state)


def _chunk_sparse_delta_rule(q_idx, q_val, k_idx, k_val, v, g, beta, initial_state, output_final_state=False,
                             cu_seqlens=None):
    # `cu_seqlens` is internal: the rows of a padded batch with their padding removed, one bank each (the SDM layer
    # passes exactly one sequence per row), never documents packed into a row
    if k_idx.shape[-1] + q_idx.shape[-1] >= 2**25:
        raise ValueError(
            f"`W + R` is expected to be below 2**25 rather than {k_idx.shape[-1] + q_idx.shape[-1]}: "
            f"a chunk counts its 64 * (W + R) entries in int32.",
        )
    if cu_seqlens is not None and v.shape[0] != 1:
        raise ValueError(f"The batch size is expected to be 1 rather than {v.shape[0]} when using `cu_seqlens`.")
    check_initial_state(initial_state, v, cu_seqlens)
    return ChunkSparseDeltaRuleFunction.apply(
        q_idx,
        q_val,
        k_idx,
        k_val,
        v,
        g,
        beta,
        initial_state,
        output_final_state,
        cu_seqlens,
    )
