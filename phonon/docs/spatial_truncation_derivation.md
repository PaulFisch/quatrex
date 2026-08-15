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
| 1 | `Phi` to `p = 1` | `fc3_loader.py:5-9`; drops longer triplets, warns above a dropped-Frobenius threshold | approximate; **never measured on a real bed** |
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

Truncation 1 is untouched by either and is the only one whose size is still
unknown.

## Caveats on the numbers

The 10.8 % is a 1-DOF chain with a random dense nearest-neighbour vertex. A
real `Phi` may be considerably more diagonal-dominant, which would shrink it,
or not. The support law (*) and the flatness argument are structural and do not
depend on the bed; the percentage does. Evaluating the same split on the stored
FC3 blocks is offline and is what would turn it into a production number.

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

Each step was a claim about which blocks matter, argued in prose. The algebra
is four lines and settles all of it, which is why it is written down here.
