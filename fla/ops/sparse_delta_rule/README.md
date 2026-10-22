# Sparse Delta Rule

`chunk_sparse_delta_rule` runs the gated delta rule over a bank of `S` slots per head: each token writes `W` slots and reads `R` slots, with weights, and only the written slots decay.
Its docstring gives the recurrence, the supported inputs and the numerics, and `naive.py` holds the token-by-token reference.
This note describes how `chunk.py` lays out a chunk and which kernel computes what.

## Chunk form

The bank advances by chunks of `C = 64` tokens.
In a chunk, each write and each read of a token is an entry `i`, of token `t_i`, slot `s_i` and weight `val_i`.
`G_i` is the sum of the log decays `g`, in float32 and clamped at -128, over the writes of slot `s_i` by the tokens `t <= t_i`, accumulated in float64 and stored in float32; `gend_s` is `G` at the last entry of slot `s`.
With `h` the bank at the chunk start, the kernels compute

```math
\begin{aligned}
r_t &= \sum_{\text{writes } i \text{ of } t} \mathrm{val}_i\, e^{G_i} h_{s_i}, &
q_t &= \sum_{\text{reads } i \text{ of } t} \mathrm{val}_i\, e^{G_i} h_{s_i}, \\
A_{t t'} &= \sum_{(i, j) \in P^{w}_{t t'}} \mathrm{val}_i\, \mathrm{val}_j\, e^{G_i - G_j}, &
A^{qk}_{t t'} &= \sum_{(i, j) \in P^{r}_{t t'}} \mathrm{val}_i\, \mathrm{val}_j\, e^{G_i - G_j}, \\
u &= (I + \mathrm{diag}(\beta) A)^{-1} \mathrm{diag}(\beta) (v - r), &
o &= q + A^{qk} u, \\
h_s &\leftarrow e^{\mathrm{gend}_s} h_s + \sum_{\text{writes } j \text{ of } s} \mathrm{val}_j\, e^{\mathrm{gend}_s - G_j} u_{t_j},
\end{aligned}
```

where `P^w_{t t'}` holds the pairs of a write `i` of token `t` and a write `j` of token `t' < t` on the same slot, and `P^r_{t t'}` the pairs of a read `i` of token `t` and a write `j` of token `t' <= t` on the same slot.
These are the pairs of an entry and an earlier write of its slot in the sorted order below; `A^{qk}` is `Aqk` in the code.

## Data layout

`chunk_sparse_delta_rule_prepare` gathers the inputs chunk-major, as `[NT, H, C, ...]`: the `NT` chunks of all sequences, each sequence padded to whole chunks.
`chunk_offsets` delimits the chunks of each sequence, `pos` and `valid` map the chunks back to tokens, and the kernels index a (chunk, head) as `i_ch = i_c * H + i_h`.
Padding tokens get weight 0 and slot `-1` in `k_idx` and `q_idx`, so they gather no bank row.

Each (chunk, head) has `E = C * (W + R)` entries.
In input order, entry `t * (W + R) + j` is write `j` of token `t` for `j < W`, and read `j - W` for `j >= W`.
The entries are sorted by the key `slot * 2C + 2t + kind`, with `kind` 0 for a write and 1 for a read: by slot, then token, then kind.
The entries of a slot are thus contiguous, and the write of a slot by a token precedes its read by the same token; `perm` gives the input position of each sorted entry.
The entries of padding tokens get the keys `(S + t * (W + R) + j) * 2C`: distinct slots past the bank, so they pair with nothing and sort last.
These keys carry no token or kind bits, so padding entries decode as writes of token 0, with weight 0; the token and gate meanings below hold for real entries.

The `row` of an entry, `t` for a write and `C + t` for a read, is its row in the `[2C, C]` tile of the forward pass that stacks `A` over `Aqk`.
It is also its row in the `[2C, V]` buffer `dx` of the backward state pass, which holds the gradients of `r`, then those of `q`: `-beta * b_dx` and `do` (see below).
The snapshots `snap`, of shape `[NT, H, C, W, V]` in the dtype of the bank, keep the chunk-start rows of the slots each chunk writes, at the write rank of the first write of each slot.
The forward pass saves them when a gradient is needed, and the backward pass restores the bank from them.

## Prepared arrays

`chunk_sparse_delta_rule_prepare` also returns the `[NT, H, n]` arrays of its `outputs` dict, in three orders.

Sorted order, `n = E` (`E // 64` for `nd`):

| Array   | dtype   | Meaning                                                                                     |
| ------- | ------- | ------------------------------------------------------------------------------------------- |
| `slot`  | int64   | slot of the entry, at least `S` for padding                                                 |
| `row`   | int32   | `t` for a write, `C + t` for a read                                                         |
| `val`   | float32 | weight of the entry, from `k_val` or `q_val`                                                |
| `G`     | float32 | `G_i` of the chunk form                                                                     |
| `depth` | int32   | number of earlier entries of the slot                                                       |
| `nd`    | int32   | per block of 64 entries, the largest distance from one of them to another entry of its slot |
| `wt`    | float32 | `val * exp(G)`, the weight of the entry on the chunk-start row of its slot                  |
| `decay` | float32 | `exp(gend)`                                                                                 |
| `last`  | bool    | the last entry of a slot below `S`                                                          |
| `rep`   | bool    | the first write of a slot below `S`                                                         |
| `npair` | int32   | number of earlier writes of the slot, that is, of pairs `(i, j)` with the entry as `i`      |
| `r0`    | int32   | write rank of the first write of the slot, for a slot the chunk writes                      |

Write order, `n = C * W`: the first `C * W` writes in sorted order, padding entries counting as writes after the real ones.

| Array     | dtype   | Meaning                                                                             |
| --------- | ------- | ----------------------------------------------------------------------------------- |
| `slot_w`  | int64   | slot                                                                                |
| `tok_w`   | int32   | token `t`                                                                           |
| `coef_w`  | float32 | `val * exp(gend - G)`, the weight of `u[t]` in the row of the slot at the chunk end |
| `decay_w` | float32 | `exp(gend)`                                                                         |
| `depth_w` | int32   | number of earlier writes of the slot                                                |
| `last_w`  | bool    | the last write of a slot below `S`                                                  |
| `wpos`    | int32   | sorted position of the write                                                        |

Entry order, `n = C * W` for the writes (laid out as `k_idx`) or `C * R` for the reads (as `q_idx`):

| Array          | dtype   | Meaning                                                                                                    |
| -------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `k_wt`, `q_wt` | float32 | `wt`                                                                                                       |
| `k_coef`       | float32 | `coef_w`                                                                                                   |
| `src_k`        | int32   | `r0`, the index of the snapshot of the slot                                                                |
| `src_q`        | int32   | `r0` if the chunk writes the slot, else `-1`: the read then takes the bank row                             |
| `first`        | bool    | `rep`: the write saves the snapshot of its slot in the Triton forward pass and restores it in the backward |
| `e_k`, `e_q`   | float32 | `exp(G)`                                                                                                   |
| `e_c`          | float32 | `exp(gend - G)`                                                                                            |
| `e_end`        | float32 | `exp(gend)`                                                                                                |

## Backward arrays

The backward state pass writes `dAqk`, `dL`, `dbeta` and contractions over `V`, already scaled by the decays in the forward weights, once per value block; the host sums them over the blocks, and `chunk_sparse_delta_rule_bwd_kernel_intra` turns them into the gradients of the weights and of `g`.
Below, `h[s]` is the row of slot `s` at the chunk start and `dh[s]` its gradient at the chunk end, `du` is the gradient of `u`, and `b_dx = Tm^T du`, as the kernels name it, is the gradient of the right-hand side `diag(beta) (v - r)` of the solve, with `Tm = (I + diag(beta) A)^-1`.
The table gives the arrays after the sum over the value blocks; all are float32 but `dG`.

| Array    | Layout                       | Meaning                                                                                                                             |
| -------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `dA`     | `[NT, H, C, C]`              | gradient of `A`: `diag(beta) tril(dL, -1)`, where `dL = -b_dx u^T` is the gradient of `I + diag(beta) A` through the solve          |
| `dAqk`   | `[NT, H, C, C]`              | gradient of `Aqk`: `do u^T`                                                                                                         |
| `dkw`    | `[NT, H, C, W]`, entry order | `-(beta b_dx)[t] . (exp(G) h[s])`, the gradient of `val` through `r`; on exit, the gradient of the `g` of the write                 |
| `dqw`    | `[NT, H, C, R]`, entry order | `do[t] . (exp(G) h[s])`, the gradient of `val` through `q`                                                                          |
| `dc`     | `[NT, H, C, W]`, entry order | `dh[s] . (exp(gend - G) u[t])`, the gradient of `val` through the row update                                                        |
| `dgend`  | `[NT, H, C, W]`, entry order | `dh[s] . (exp(gend) h[s])`, the gradient of `gend` through the decay of the start row, stored at every write and used at `rep` only |
| `dk_val` | `[NT, H, C, W]`, entry order | gradient of `k_val`: the pair terms plus `dkw + dc`                                                                                 |
| `dq_val` | `[NT, H, C, R]`, entry order | gradient of `q_val`: the pair terms plus `dqw`                                                                                      |
| `dG`     | `[NT, H, E]`, sorted order   | gradient of the `G` of each entry, in float64                                                                                       |

## Kernels

In call order, a launch runs one program per (chunk, head) it covers, except in the state passes: there, one program per (sequence, head, block of `BV` value columns) runs over the chunks of the sequence, with `BV` chosen by `_bv` (16 in the Gluon passes).

1. `chunk_sparse_delta_rule_prepare_kernel`, after the sort: a pass over the sorted entries with segmented sums, in float64 for `G`, then a pass in reverse for the arrays that depend on the later entries of each slot, and for `src_q`.
2. `chunk_sparse_delta_rule_fwd_kernel_intra` and `chunk_sparse_delta_rule_fwd_kernel_intra_dense`: the pair sums `A` and `Aqk`, the sparse inner products of the chunk form, as one tile per (chunk, head) indexed by `row` and `t`.
   The first takes the pairs that `_pair_list` lists from `npair`, `r0` and `wpos`, in launches of about `2**26` pairs at most; a (chunk, head) whose pairs span two launches accumulates over both, in order.
   The second takes the (chunk, head) with more than 4096 pairs and forms their pair weights in tiles.
   Both add the weights through one-hot `tl.dot`s, each weight split into three bfloat16 parts.
   The host then inverts `I + diag(beta) A` into `Tm` with `torch.linalg.solve_triangular`.
3. `chunk_sparse_delta_rule_fwd_kernel_h`, the forward state pass (`chunk_sparse_delta_rule_fwd_h`), over the chunks in order: it gathers `r` and `q` from the bank, the first write of each slot saving the snapshot when `snap` is given, computes `u` and `o`, then updates the written rows in place (`_scatter_rows` over the write order).
   `gluon.py` registers an opt-in Gluon version, `chunk_sparse_delta_rule_fwd_kernel_h_gluon`, enabled by setting `FLA_GLUON` or `FLA_SDM_GLUON` to any value but `0` (see `fla/backends.py`); its verifiers, built on `_check` and `_buffers`, choose the calls it serves, and its module docstring summarizes them.
4. `chunk_sparse_delta_rule_bwd_kernel_h`, the backward state pass (`chunk_sparse_delta_rule_bwd_h`), over the chunks in reverse: it restores the written rows from their snapshots, computes `dv`, `dAqk`, `dL`, `dbeta`, `dkw`, `dqw`, `dc` and `dgend`, then carries `dh` back over the chunk (`_scatter_rows` over all the entries in sorted order).
   The opt-in Gluon version is `chunk_sparse_delta_rule_bwd_kernel_h_gluon`.
   After the sum over the value blocks, the host keeps `tril(dL, -1)`, adds the row sums of `dL * A` (where `A != 0`) to `dbeta`, and passes `dA = diag(beta) dL` on.
5. `chunk_sparse_delta_rule_bwd_kernel_intra`: for each sorted entry, the pairs within distance `nd` give the gradients of `val` and `G` from `dA` and `dAqk`.
   It adds the contractions into `dk_val`, `dq_val` and `dG`, the gradient of `gend` going to the `G` of the last entry of the slot, then a reverse segmented sum over each slot turns `dG` into the gradient of the `g` of each write, written over `dkw`.
   The host sums it over the writes of each token, and zeroes it where `g` is at most -128, the value `g` is clamped to.

`fused_recurrent_sparse_delta_rule_fwd_kernel` in `fused_recurrent.py` runs the recurrence token by token, forward only, for decoding and short inputs.
`product_key_topk` in `topk.py` selects slots by an exact top-`k` over product keys, in PyTorch.

## Determinism

The docstring of `chunk_sparse_delta_rule` states that outputs, final bank and gradients are bitwise reproducible for fixed inputs, shapes and GPU type.
With the same software and settings, such as the Gluon opt-in and `FLA_USE_FAST_OPS`, this rests on the following.

- Sorted order: the entries of a slot are contiguous, so a row update sums one run of entries; per block of entries, by a look-back from the last entry of each run when the runs are short, or else by a segmented scan with a carry between blocks.
- One writer per bank element: a state-pass program owns a block of value columns of the bank of one (sequence, head).
  Per chunk, the forward pass stores each written row once, from the last write of its slot, and each snapshot once; the backward pass restores each written row once, at the first write of its slot, and stores each touched row of `dh` once, from the last entry of its slot.
- No atomics: each program of the other kernels writes only its own (chunk, head), and the backward state pass writes what it reduces over `V` to one buffer per value block, which the host sums.
- Fixed-order sums: the pair weights reach their cells of the tile in a fixed order, in float32, the segmented sums of `G` and `dG` run in float64, and the backward visits the pairs of each entry by increasing distance.

The Gluon passes sum in another fixed order: they are reproducible too, but need not match the Triton passes bitwise.
`product_key_topk` sums its backward over the selected slots in a fixed order instead of a scatter-add.
