# Why the least-squares MoS2 ladder ran out of memory, and on which axis

## The measurement that started it

The LS rungs would not run at ranks that the ARDR rungs of the *same block
sizes* had run at: `lsM4` OOM'd at 82.5 GB/rank on 4 nodes where `cvM4e` had
been fine on 2, and `lsM6` OOM'd on 8 nodes and again on 16 where `cvM6b` had
been fine on 8.

The two probes on `lsM6` separate the per-rank budget into its two halves,
because only the frequency-split part responds to node count:

| probe | nodes | ranks | peak allocated |
|---|---|---|---|
| `lsM6probe` (4491151) | 8 | 32 | 77.4 GB |
| `lsM6probe16` (4491167) | 16 | 64 | 68.9 GB |

Doubling the ranks removed 8.5 GB. So the split part is 16.8 GB at 32 ranks and
the **replicated part is 60.5 GB**, which no number of nodes touches. Adding
nodes was scaling the wrong axis.

## Where the replicated part is

`perm_cache` (`sse_phonon_phonon.py`), the dict of pre-permuted vertex pairs.
`phonon/studies/_memory_model.py` puts it at 75.6 % of the SSE phase at `lsM4`
and 90.3 % at `lsM6`. Its size is

    32 * b^3 * (nq^2 / P_q) * Q     bytes

and all three factors grew at once when the ladder moved to the least-squares
vertex:

* `b^3` with thickness -- 18, 36, 54 is 1 : 8 : 27;
* `Q`, the ring-quad count, as (vertex blocks)^2. The engine prints it:
  `qtasks=2500` on `cvM2b` against `qtasks=40000` on `lsM2`. ARDR pruned the
  cross-gap third-order parameters to exact zero, so its vertex is
  block-diagonal and its quad count 16x smaller. The LS vertex keeps that
  channel -- which is the whole reason for the rebuild -- and pays for it here;
* `nq^2 = 625`, replicated on every rank.

`P_stack` does not appear. That is the structural fact.

## `q_comm_size` divides it, and helps exactly one of the two rungs

`q_comm_size > 1` with replicated G buffers has been implemented all along and
was set to 1 in **all 143** recorded runs. It splits the external-q loop over
`comm.q` and sums the partial `Sigma(q_ext)` back, and the external-q
restriction is what selects which `(iqp, iq2)` pairs a rank caches -- so
`perm_cache` divides by `P_q`. It costs a factor `P_q` on the frequency-split
terms, since `P_stack = world / (P_block * P_q)`.

Both probed at `QX_QCS=2`, 20 min each:

| run | nodes | `q_local` | `n_tau` | peak |
|---|---|---|---|---|
| `lsM4probe` (4491162) | 8 | 25 | 376 | 86.3 % |
| `lsM4q2` (4492139) | 8 | 12 | 751 | **78.1 %** |
| `lsM6probe16` (4491167) | 16 | 25 | -- | 68.9 GB (OOM) |
| `lsM6q2` (4492140) | 16 | 12 | -- | **77.4 GB (OOM, worse)** |

The asymmetry is the model, not an anomaly. Halving `q_local` halves
`perm_cache`; halving `P_stack` doubles `n_tau` and with it every tau buffer.
Net memory is `A/P_q + B*P_q`, and which term wins depends on the rung: at
`lsM4` the cache dominates and `P_q = 2` saves 8 points of device memory, at
`lsM6` the tau buffers dominate and the same switch costs 8.5 GB.

So the replicated-G form of the q axis is a real lever for `lsM4` and the wrong
lever for `lsM6`.

## The form that would work for `lsM6`, and what it still needs

Sectioning the G buffers themselves (`q_distributed`) divides the legs AND
`perm_cache` by `P_q` with no compensating growth. That path existed but was
refused; making it work turned up three defects, all dormant because it had
never run (commit `0c04a1a1`):

1. the Sigma accumulators were sectioned along with the legs, which they cannot
   be -- the bubble is a convolution over q, so one slice pair contributes to
   every external momentum. `_SIGMA_TAU_SLOTS` keeps slots 2 and 3 whole and the
   section is taken after the `comm.q` sum;
2. `q -> -q` on the reversed legs indexed the LOCAL axis with GLOBAL indices.
   The negation maps a rank's slice onto another rank's, so `_negate_q_across_comm`
   rebuilds the whole axis from zero-padded disjoint sections;
3. the batched contraction stacked only the A family and read leg B from it, so
   a rank contracted its own slice against itself, and indexed out of bounds
   once the widths differed.

`test_internal_q_rotation_reproduces_the_replicated_result` now passes: a
q-sectioned G reproduces the replicated Sigma to 1e-10.

What is left is geometric. Production MoS2 has `global_stack_shape = (ne, 5, 5)`
and `q_distributed` requires exactly one transverse axis, so the 5x5 mesh has to
be flattened to 25 and the three `enumerate(nk)` q -> -q sites replaced by a flat
permutation before `lsM6` can use it.

## The verdict for this budget

Even with the memory solved, `lsM6` does not fit. `lsM4` runs at 5.2-5.6 min per
iteration on 8 nodes and has taken 65 iterations without reaching 1e-3; `lsM6`
is 3.4x the b^3 at the same block count, so ~9.5 min per iteration on 16 nodes,
and 65 iterations of that is ~165 node-hours against the ~96 left under the 400
cap. The three-point LS ladder is out of reach on cost, not only on memory, and
the flattening above is infrastructure for a later budget rather than the thing
standing between us and the number.

## What not to do

Do not bound or evict `perm_cache`: it has no eviction policy, and adding one
trades memory for recomputing `phi_perms` -- but the ring is already FLOP-bound
(3.6 PFLOP/pass at `lsM4`), so that trade runs the wrong way.

Do not re-block `lsM6` to 6 blocks of 1 cell to shrink `b^3`. It works, but it
reintroduces the Sigma output pin and the indefinite block-band mask, and it
stops being the same experiment as the ARDR ladder.
