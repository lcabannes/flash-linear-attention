# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch


def naive_recurrent_sparse_delta_rule(
    q_idx: torch.Tensor,
    q_val: torch.Tensor,
    k_idx: torch.Tensor,
    k_val: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool = False,
):
    r"""
    Token-by-token reference of the sparse delta rule: the gated delta rule over a bank of `S` slots per head,
    with the keys and queries replaced by `W` written and `R` read slots and their weights.
    Per token and head, only the written slots decay:

        h[k_idx] = exp(g) * h[k_idx]
        h[k_idx] += k_val * beta * (v - sum_w k_val[w] * h[k_idx[w]])
        o = sum_r q_val[r] * h[q_idx[r]]

    Args:
        q_idx (torch.Tensor):
            Read slots of shape `[B, T, H, R]`.
        q_val (torch.Tensor):
            Read weights of shape `[B, T, H, R]`.
        k_idx (torch.Tensor):
            Written slots of shape `[B, T, H, W]`, distinct within a token.
        k_val (torch.Tensor):
            Write weights of shape `[B, T, H, W]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Log decays of the written slots, of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        initial_state (torch.Tensor):
            Initial bank of shape `[B, H, S, V]`.
        output_final_state (bool):
            Whether to return the final bank. Default: `False`.

    Returns:
        A tuple `(o, h)` where `o` has shape `[B, T, H, V]` and `h` has shape `[B, H, S, V]` (float32)
        if `output_final_state` else `None`.
    """
    dtype = v.dtype
    B, T, H, V = v.shape
    q_val, k_val, v, g, beta = (x.float() for x in (q_val, k_val, v, g, beta))
    i_b = torch.arange(B, device=v.device)[:, None, None]
    i_h = torch.arange(H, device=v.device)[None, :, None]

    h = initial_state.float().clone()
    o = torch.zeros_like(v)
    for i in range(T):
        # [B, H, W, V]
        b_h = h[i_b, i_h, k_idx[:, i]] * g[:, i, :, None, None].exp()
        b_v = beta[:, i, :, None] * (v[:, i] - (k_val[:, i, ..., None] * b_h).sum(-2))
        h[i_b, i_h, k_idx[:, i]] = b_h + k_val[:, i, ..., None] * b_v[:, :, None]
        o[:, i] = (q_val[:, i, ..., None] * h[i_b, i_h, q_idx[:, i]]).sum(-2)
    return o.to(dtype), (h if output_final_state else None)


def naive_product_key_topk(s: torch.Tensor, dims: tuple[int, ...], k: int) -> torch.Tensor:
    r"""
    Row-by-row reference of the slot selection of `product_key_topk`, with plain sorts. Slot `(c_0, ..., c_{F-1})`
    scores the float32 sum of its factor values, added in factor order, each partial sum rounded to `s.dtype`
    before the next factor is added. Scores compare as float32 numbers, with -0 below +0 and NaN counted as -inf.
    Each factor is ranked by value, larger column first on ties. Stage `f = 1, ..., F-1` keeps the best
    `c = min(k, n * m)` of the pairs `(i, j)` of a rank `i < n` of the previous stage and a rank `j < m` of factor
    `f` with `(i + 1) * (j + 1) <= c`, exact ties going to the larger `i + j`, then the larger `i`.

    Returns:
        The slot ids of shape `[..., k]`, in ascending order.
    """
    def key(x):
        return (-math.inf, -1.) if math.isnan(x) else (x, math.copysign(1., x))

    def rounded(x):
        return torch.tensor(x, dtype=torch.float).to(s.dtype).item()

    offsets = [sum(dims[:f]) for f in range(len(dims))]
    out = []
    for row in s.detach().float().reshape(-1, s.shape[-1]).tolist():
        val, idx = None, None
        for f, (o, d) in enumerate(zip(offsets, dims)):
            x = row[o:o + d]
            ranks = sorted(range(d), key=lambda c: (key(x[c]), c), reverse=True)[:min(k, d)]
            if f == 0:
                val, idx = [x[c] for c in ranks], ranks
                continue
            n, m = len(val), len(ranks)
            c = min(k, n * m)
            pairs = [(i, j) for i in range(n) for j in range(m) if (i + 1) * (j + 1) <= c]
            score = {(i, j): torch.tensor(val[i] + x[ranks[j]], dtype=torch.float).item() for i, j in pairs}
            pairs = sorted(pairs, key=lambda p: (key(score[p]), p[0] + p[1], p[0]), reverse=True)[:c]
            val, idx = [rounded(score[p]) for p in pairs], [idx[i] * d + ranks[j] for i, j in pairs]
        out.append(sorted(idx))
    return torch.tensor(out, device=s.device).view(*s.shape[:-1], -1)
