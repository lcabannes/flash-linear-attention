# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def _sort_key(x: torch.Tensor) -> torch.Tensor:
    # int64 keys ordering float32 values, with -0 below +0 and NaN as -inf
    bits = x.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    bits = torch.where(x.isnan(), 0xFF800000, bits)
    return torch.where(bits >= 0x80000000, bits ^ 0xFFFFFFFF, bits | 0x80000000)


@torch.no_grad()
def _product_key_select(s: torch.Tensor, dims: tuple[int, ...], k: int) -> tuple[torch.Tensor, torch.Tensor]:
    val, idx = None, None
    for f, x in enumerate(s.float().split(dims, dim=-1)):
        # rank the columns of each factor by value, the larger column first on ties
        col = torch.arange(dims[f], device=s.device)
        x_idx = ((_sort_key(x) << 31) | col).topk(min(k, dims[f]), dim=-1).indices
        x_val = x.gather(-1, x_idx)
        if f == 0:
            val, idx = x_val, x_idx
            continue
        # pairs of ranks (i, j) with (i + 1) * (j + 1) <= c hold an exact top-c of all pairs
        n, m = val.shape[-1], x_val.shape[-1]
        c = min(k, n * m)
        i = torch.arange(n, device=s.device).repeat_interleave(m)
        j = torch.arange(m, device=s.device).repeat(n)
        keep = (i + 1) * (j + 1) <= c
        i, j = i[keep], j[keep]
        score = val[..., i] + x_val[..., j]
        # exact ties go to the larger i + j, then to the larger i
        nb = max(n, m).bit_length()
        order = ((_sort_key(score) << (2 * nb + 1)) | ((i + j) << nb) | i).topk(c, dim=-1).indices
        val = score.gather(-1, order).to(s.dtype).float()
        idx = idx.gather(-1, i[order]) * dims[f] + x_idx.gather(-1, j[order])
    return idx.sort(dim=-1)


class ProductKeyTopKFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, s, dims, k):
        idx, _ = _product_key_select(s, dims, k)
        strides = [math.prod(dims[f + 1:]) for f in range(len(dims))]
        # [..., k, F], the column of each selected slot in every factor
        cols = torch.stack([idx // stride % dim for stride, dim in zip(strides, dims)], dim=-1)
        val = None
        for f, x in enumerate(s.float().split(dims, dim=-1)):
            val = x.gather(-1, cols[..., f]) if val is None else val.to(s.dtype).float() + x.gather(-1, cols[..., f])
        ctx.save_for_backward(cols)
        ctx.dims = dims
        ctx.mark_non_differentiable(idx)
        return val.to(s.dtype), idx

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dval, didx):
        cols, = ctx.saved_tensors
        # a fixed-order sum over the selected slots, deterministic unlike a scatter-add
        ds = [
            (dval.float().unsqueeze(-1) * (cols[..., f, None] == torch.arange(dim, device=dval.device))).sum(-2)
            for f, dim in enumerate(ctx.dims)
        ]
        return torch.cat(ds, dim=-1).to(dval.dtype), None, None


def product_key_topk(s: torch.Tensor, dims: tuple[int, ...], k: int) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Exact top-`k` over the `prod(dims)` slots of a product key: slot `(c_0, ..., c_{F-1})` scores the float32 sum
    of the `c_f`-th column of every factor `f`, added in factor order; each partial sum is rounded to `s.dtype`
    before the next factor is added.

    Selection rule: scores compare as float32 numbers, with -0 below +0 and NaN counted as -inf. Each factor is
    ranked by value, larger column first on ties. Stage `f = 1, ..., F-1` keeps the best `c = min(k, n * m)` of
    the pairs `(i, j)` of a rank `i < n` of the previous stage and a rank `j < m = min(k, dims[f])` of factor `f`
    with `(i + 1) * (j + 1) <= c`, exact ties going to the larger `i + j`, then the larger `i`. Each pair left out
    has `c` pairs of lower or equal ranks scoring at least as much, so the selection is an exact top-`k`.

    Args:
        s (torch.Tensor):
            Factor scores of shape `[..., sum(dims)]`, the `F` factors side by side.
        dims (tuple[int, ...]):
            Factor widths.
        k (int):
            Number of slots to select, at most `prod(dims)` and `2**14`.

    Returns:
        A tuple `(val, idx)` of shape `[..., k]`: the scores (differentiable) and the slot ids
        `(..(c_0 * dims[1] + c_1) * ..) * dims[-1] + c_{F-1}` in ascending order.
    """
    if not dims or min(dims) < 1 or max(dims) >= 2**31 or math.prod(dims) >= 2**62:
        raise ValueError(f"dims = {tuple(dims)} are expected to be positive, below 2**31, with a product below 2**62.")
    if not 1 <= k <= min(math.prod(dims), 2**14):
        raise ValueError(f"k = {k} is expected to be in [1, min(prod(dims), 2**14)] for dims = {tuple(dims)}.")
    return ProductKeyTopKFunction.apply(s, tuple(dims), k)
