# The ring's spatial truncations, derived

Three separate spatial approximations sit in the phonon-phonon self-energy, and
they are easy to confuse with one another. This document does the index algebra
once so that the question "does a spatial truncation cost anything here" has an
answer that does not depend on which of them was being thought about.

It exists because reasoning about it in prose produced two opposite wrong
answers in a row; the error trail is at the end.

## Setup

The cubic self-energy is

    Sigma(I,J) = sum_{K1,K2,K1',K2'} Phi_{I,K1,K2} G(K1,K1') G(K2,K2')
                 Phi_{J,K2',K1'}

over transport-cell block indices, convolved in frequency (the frequency
structure plays no part in anything below).

Two supports enter.

**Vertex reach `p`.** `Phi_{I,K,L}` is nonzero only for
`|I-K|, |I-L|, |K-L| <= p`. Production has `p = 1`: `fc3_loader.py` keeps one
nearest-neighbour shell and drops the rest.

**Leg band `b`.** `G(K,K')` is retained only for `|K-K'| <= b`. This is
`sse_g_band`, default 3, `Field(default=3, ge=1, le=3)`.

## The support law

`Phi_{I,K1,K2}` forces `K1, K2` within `p` of `I`. `Phi_{J,K2',K1'}` forces
`K1', K2'` within `p` of `J`. The legs contribute only for `|K-K'| <= b`.
Chaining the three,

    |I - J| <= |I - K1| + |K1 - K1'| + |K1' - J| <= p + b + p,

so

    supp(Sigma) = { |I - J| <= 2p + b }.                          (*)

Measured on a 1-DOF chain with `p = 1`, taking the largest distance at which
`Sigma` exceeds 1e-13 of its peak:

| leg band `b` | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| measured reach | 2 | 3 | 4 | 5 |
| `2p + b` | 2 | 3 | 4 | 5 |

Test: `test_the_sigma_support_law_is_two_p_plus_band`.

The consequence is the one that matters below: **`Sigma` is not tridiagonal**,
and its reach grows with the leg band.

## The three truncations

| # | truncation | where | status |
|---|---|---|---|
| 1 | `Phi` to `p = 1` | `fc3_loader.py:5-9` | **exact** on every bed here -- the FC3 cutoff is shorter than the transport cell, so nothing beyond the shell exists to drop (see below) |
| 2 | legs to `\|K-K'\| <= b` | `sse_g_band` | **exact** for the retained output once `b >= 3` |
| 3 | output to `\|I-J\| <= 1` | `sse_phonon_phonon.py:474`, `for J in range(max(0, I-1), min(n, I+2))` | approximate; discards `2 <= \|I-J\| <= 2p+b` |

Truncation 3 is hard-coded. It does not follow from `p`, it does not follow
from `b`, and by (*) it is not a property of `Sigma`.

### Why 2 is exact and 3 is not

They truncate the same object and are usually discussed together, which is the
trap. Truncation 2 is exact **given** truncation 3: with the output pinned at
`|I-J| <= 1`, the reachable leg distance is `|K-K'| <= 2p + 1 = 3`, so `b = 3`
discards only links that could not have contributed. That is what the config
docstring means by "the first off-diagonal Sigma blocks become exact and
causal", and why the field is capped at 3.

Split by output distance, on the same bed:

| `b` | rel. err on `\|I-J\| <= 1` | rel. err on `\|I-J\| > 1` |
|---|---|---|
| 1 | 3.18e-01 | large |
| 2 | 1.04e-01 | large |
| 3 | **0.000e+00** | large |

Test: `test_band_three_is_exact_on_the_output_band_and_lossy_off_it`.

Reporting a whole-array error therefore overstates the leg band's cost;
reporting only the retained band hides the output pin's. Both mistakes were
made here, in that order.

### What the output pin costs

With `p = 1` and an untruncated `G`, the weight of `Sigma` by output distance:

| `\|I-J\|` | 0 | 1 | 2 | 3 | 4 | >=5 |
|---|---|---|---|---|---|---|
| share | 46.7 % | 42.5 % | 8.1 % | 2.0 % | 0.7 % | 0.02 % |

The pin discards **10.8 %** of the self-energy weight.

### The pin does not care how far `G` reaches

Varying the Green-function range over a factor 3.5:

| range `xi` [cells] | 2.06 | 2.65 | 3.35 | 4.60 | 7.15 |
|---|---|---|---|---|---|
| discarded | 10.5 % | 10.4 % | 10.5 % | 10.9 % | 11.4 % |

Flat, where a long-range effect would grow. The reason is index algebra again:
for `|I-J| = 2` one may take `K = I+1` and `K' = J-1 = I+1`, giving
`|K-K'| = 0`. **The near tail of `Sigma` is fed by the diagonal of `G` through
the vertex's reach**, not by long-range `G`. Long-range `G` appears only in
`Sigma` blocks the pin has already thrown away.

Test: `test_the_discarded_output_weight_does_not_track_the_green_range`.

### Why the pin is there

The Dyson operator `omega^2 I - D - Sigma` is solved by RGF, which needs
block-tridiagonal structure. A `Sigma` with reach 5 is not block-tridiagonal.
The pin is what keeps the solver applicable, so removing it is a change of
solver structure and not a change of parameter.

## What this means for the modal/spatial route

A low-rank modal representation of `G` answers "how do I carry `G(K,K')` at
large `|K-K'|` cheaply" (method proposal Secs. 33-34). Nothing in the shipped
kernel asks that: truncation 2 makes long-range `G` unreachable, and truncation
3 discards the blocks where it would have shown up. The representation is
correct -- `G(n) = V diag(lambda^n) C` reproduces real CNT and Si cells to
roundoff (`spatial_band_range.md`) -- and it is aimed at a question this ring
does not pose.

The live approximation is truncation 3, and repairing it means carrying a
`Sigma` that is not block-tridiagonal. That is the proposal's Sec. 32
modal/regular **Schur complement**, a different construction: it keeps the
tridiagonal part in the BTD solver and couples a small non-local sector to it,
rather than compressing distant `G`.

Truncation 1 turns out to cost nothing on these beds: the FC3 cutoff is
shorter than the transport-cell length, so the nearest-neighbour shell holds
every triplet the force constants contain.

## Caveats on the numbers

The 10.8 % is a 1-DOF chain with a random dense nearest-neighbour vertex. A
real `Phi` may be considerably more diagonal-dominant, which would shrink it,
or not. The support law (*) and the flatness argument are structural and do not
depend on the bed; the percentage does. Evaluating the same split on the stored
FC3 blocks is offline and is what would turn it into a production number.

## Truncation 1 is exact on every bed in the tree (2026-08-15)

Measured, then corrected once. Across every stored `fc3_blocks.hdf5` -- 45 beds
spanning CNT, Si and MoS2 -- the maximum block-index offset is 1 and the
dropped fraction at load is **0.000 %** without exception.

The first reading of that was that the builder discards distant triplets before
writing, leaving the loader's warning unable to fire. That is wrong.
`build_device_fc3_blocks` takes `vertex_cutoff=None` by default -- "imposes no
extra truncation; the result contains every triplet the supercell FC3 can
resolve" -- and `build_inputs.py:153` passes no cutoff. **The builder does not
truncate.** The stored files contain no triplet beyond the nearest-neighbour
shell because the force constants contain none.

Why they contain none is a two-line argument. Three atoms with all pairwise
distances within the FC3 cutoff `r_c` span at most `r_c` along transport, so
they occupy at most `ceil(r_c / L) + 1` consecutive cells of length `L`. For
`r_c < L` that is two, i.e.

    r_c < L  =>  max block offset 1  =>  the nearest-neighbour shell is EXACT.

MoS2: `r_c = 4.0 Ang` against a primitive `c = 12.294 Ang`
(`cluster/mos2_*_reap/hiphive_meta.json`, cutoffs `[6.0, 4.0]`). Four is
comfortably under twelve, so no triplet can reach a second-neighbour cell and
there is nothing for the projection to drop.

Two honest limits on that. The fitting supercell is `[4, 4, 3]` for the film,
and `build_device_fc3_blocks` resolves only `|delta| <= N_super_z // 2 = 1`, so
the supercell independently bounds the offset at 1 -- the two explanations
cannot be separated from the stored data, though the cutoff argument alone is
sufficient and does not depend on the supercell. And the force-constant
metadata for CNT, SiNW and SrTiO3 is not in this repository
(`phonon/configs/*/fc3_hiphive_*` are absent), so `r_c` and `L` have not been
compared for those beds; the stored offset of 1 is consistent with the same
conclusion but is not by itself proof of it, since the supercell bound would
also produce it.

So of the three spatial approximations, truncation 1 is exact on MoS2 by a
cutoff argument, and exact-or-supercell-limited elsewhere pending those inputs.
It is not the unmeasured hazard the first version of this section made it out
to be.

## Error trail

Recorded because the document exists on account of it.

1. Measured a boxcar leg band as costing 32 % at `b = 1` and 7 % at `b = 3`,
   and reported it as evidence that the shipped kernel truncates modes that
   carry heat. The 7 % was entirely in `|I-J| >= 2` blocks production never
   outputs.
2. Correcting that, concluded `b = 3` is exact and therefore no spatial
   truncation is live. Exact for the legs -- but the output pin is a separate
   hard-coded truncation, and stopping at the leg band missed it.
3. The pin costs ~11 % and is insensitive to the range of `G`, so the
   long-range modal machinery does not repair it either. The conclusion in (2)
   was right; the reason given for it was not.

4. Reported that truncation 1's size is unrecorded and its guard structurally
   dead, on the strength of every stored file showing 0.000 % dropped. The
   builder does not truncate -- `vertex_cutoff` defaults to `None` -- so the
   zero means the force constants have no triplet beyond the shell, and the
   guard reads zero because zero is the right answer.

Each step was a claim about which blocks matter, argued from a number without
its mechanism. The algebra is four lines and settles the first three; the
fourth needed one line of the builder's signature. Both are written down here
so the next reader does not re-derive them from percentages.
