# Auxiliary SCBA in the production phonon solver: implementation and Si gates

**Status:** private production algebra implemented; default solver unchanged

**Date:** 2026-08-28

**Decision:** exact auxiliary states are viable algebraically, but promoting the
complete present Si pole census is neither accurate with one constant source
nor cheaper than two-cell reblocking.  Retain the implementation as a
selective-cluster research backend; use exact reblocking plus the existing
tensor FC3 path for the current maximum-fidelity device calculations.

## 1. What is now implemented

`src/quatrex/phonon/auxiliary_scba.py` carries a passive rational Keldysh
channel as

\[
 \Sigma^x(\omega)=+iL(\omega I-K)^{-1}Q^x
 (\omega I-K)^{-H}L^\dagger,\qquad Q^x\succeq0,
 \quad x\in\{<,>\}.
\]

This is Quatrex's sign convention, `-i Sigma^{<,>} >= 0`.  The retarded part
is reconstructed from `Q^>-Q^<`; it is not fitted independently.  Therefore
`Sigma^R-Sigma^A=Sigma^>-Sigma^<` holds between grid points as a rational
identity.

The module provides:

* a coherent `PassiveClusterState` and causal `RationalKeldyshChannel`;
* the exact cluster--cluster bubble closure.  Its poles are the Kronecker sum
  `K_a (x) I + I (x) K_b`, and its source is a sum of PSD Kronecker products;
* a local augmented-RGF representation.  A state owned by block `I` may couple
  to physical blocks `I-1:I+1`; eliminating it can generate the missing
  physical distance-two shell while the augmented operator stays BTD;
* an exact global Woodbury solve around the production BTD factorisation for a
  propagating state whose coupling spans the device;
* dense augmented and Schur-complement oracles.

`PhononSolver.set_auxiliary_channel(...)` selects either auxiliary solve.
There is deliberately no public configuration field.  With the default
`None`, the established RGF path is unchanged.  The adapter is suitable for a
frozen-state experiment or for a future fixed-basis SCBA epoch; it does not
silently infer a rational sidecar from sampled Sigma.

The focused tests pin the rational Gramian against infinite-line quadrature,
the Kronecker bubble against independent real-axis quadrature, the spectral
identity and Keldysh symmetry, and both production-side solves against dense
augmented oracles.  The errors are at `2e-10` for the numerical quadrature
oracle and approximately `1e-12` or below for the algebraic/DSDB solves.

## 2. Why this is not yet a self-consistent public SCBA mode

A conserving implementation cannot update a changing pole list outside the
existing mixer.  It needs all of the following:

1. Hold a coherent q-resolved basis fixed over an SCBA epoch.  Mix its PSD
   source coordinates and the smooth cell coefficients in the same fixed-point
   vector.  Basis changes happen only between epochs, with an adjoint
   projection of the old state.
2. Use one enriched bilinear bubble: cell--cell by the existing FFT,
   cluster--cell by Toeplitz product integration, and cluster--cluster by the
   state-space closure.  Narrow sum poles remain auxiliary states.
3. Carry a positive frequency-dependent source model when one constant `Q`
   fails.  Convex local source anchors preserve passivity; using them in an
   analytic bubble requires polynomial/rational product-integration weights or
   an additional passive realization.  Complex continuation of the source to
   a pole is not a covariance and must not be used as `Q`.
4. Direct-sum the channels over q and FC3 pairs, then reduce them with a
   symmetry-constrained, moment-enriched model reduction.  Positivity and the
   collision moment are acceptance constraints, not post-processing repairs.
5. Implement the cut-aware internal current for global Woodbury channels and
   the distributed augmented recurrence.  The current global adapter returns
   exact lead currents but intentionally does not invent an internal cut
   decomposition.

Without these steps, a sidecar may be causal and PSD yet still define a
different, nonconserving SCBA map.

## 3. Real Si source and rank gate

Pole states were extracted from frozen eta-zero production states with an
undistributed frequency axis.  L3 uses 141 points over 0--35 THz; L8 uses 121
points over 0--15 THz.  Both use the 9x9 transverse mesh and the production
rank-128 tensor-decomposed FC3 vertex.

The initial analysis incorrectly tested `-i source_at_poles(...)` for
passivity.  That quantity is a complex analytic continuation used by the old
partial-fraction formula; it need not be Hermitian or PSD.  The corrected gate
tests the complete projected source on the real axis, then constructs a
positive response-weighted constant source and measures its error after the
full `U D [.] D^H U^H` congruence.

| quantity | Si L3 | Si L8 |
|---|---:|---:|
| q points | 81 | 81 |
| coherent clusters | 195 | 365 |
| promoted poles | 460 | 1184 |
| poles/q, median / max | 6 / 12 | 14 / 30 |
| real-axis source PSD floor | `-4.04e-9` | `-9.16e-9` |
| source anti-Hermiticity error | `2.67e-15` | `3.08e-15` |
| constant-`Q` rational-leg error, median | 10.53 % | 15.88 % |
| constant-`Q` error, 90th percentile / max | 43.06 / 76.61 % | 51.05 / 133.87 % |
| modal effective cells, median | 2.86 of 3 | 7.16 of 8 |

Thus Si does not have a passivity problem.  It has a source-resolution and
state-count problem: the propagating pole subspaces span nearly the complete
finite device, and a constant source is far outside the `2e-3` target.

The production FC3 q fold then gives:

| quantity | Si L3, all q | Si L8, worst input-rank q=38 |
|---|---:|---:|
| raw output sum states/q | median 2608, max 2872 | 17036 |
| linewidth-merged sum poles/q | median 104, max 121 | 219 |
| integrated physical rank at 1 % | median 15, max 16 of 18 | 40 of 48 |
| integrated physical rank at 0.1 % | 18 of 18 | 48 of 48 |
| conservative Woodbury / two-cell-reblock cost at 0.1 % | 9.5 | 44.5 |

The L8 row is explicitly a conservative one-q screen, not a full-q statistic.
That limitation does not weaken the rejection: one required q already
saturates every physical direction.  A positive multi-anchor source would add
states, not recover the factor of 44.5.  Promoting the complete detector output
is therefore a no-go for both short and long Si.

This does **not** reject selective spectral enrichment.  A small number of
FC3-important, genuinely sub-cell clusters can still remove inverse-linewidth
grid cost.  Such a path must rank candidates by their contribution to Sigma or
current and stop promotion when the reduced output state ceases to be much
narrower than a two-cell block.

## 4. The long-film spatial reference

The frozen primitive L8 archive has shape `(121,9,9,2304)`, but all physical
shells beyond primitive distance one are exactly zero at the off-resonant,
largest-weight and narrow-pole samples.  This is not successful compression:
the current SSE output pin never generated the shells.  Post-hoc HODLR on this
archive consequently has far rank zero and cannot answer whether the missing
SCBA interaction is compressible.

To obtain the correct reference without a multi-gigabyte dense q-fold, the
primitive tensor factors are lifted exactly into two-cell superblocks.  For
external subcell `u`, internal subcell `v` and supercell offset `Delta`, the
primitive offset is

\[
 \delta=2\Delta+v-u.
\]

Replicating each CP component once per external subcell gives an exact factor
lift relative to the original rank-128 approximation: block DOF `6 -> 12`,
rank `128 -> 256`, offsets `[-2,-1,0,1,2] -> [-1,0,1]`.  Dense subblock parity
is tested for every subcell/offset combination.  The serialized L8x2 factor
archive is 6.8 MB.

The warm-start mapper decodes the full primitive DSDB block order into the
physical 48x48 matrix and gathers it in the reblocked DSDB order.  It refuses a
banded source archive.  This lets the production solver begin from the
converged primitive L8 Sigma and generate the newly admitted shells itself.

The warm reblocked trajectory was continued to the same `1e-3` retarded
self-energy gate.  A forced two-iteration primitive run on the same code,
four-GPU/q distribution and rank-128 factor path supplies the matched timing
and memory control.  The second iteration is used for steady iteration time;
the primitive physical row is the already-converged warm state.

| metric | primitive L8 | converged L8x2 |
|---|---:|---:|
| lead current | 990.0002 | 989.8948 |
| lead balance | `7.0525e-5` | `5.8497e-6` |
| internal current spread | `4.3714e-3` | `2.8288e-3` |
| retarded Sigma residual | `7.5967e-4` | `9.5112e-4` |
| steady iteration time | 34.655 s | about 32.35 s |
| FC3 ring time | 23.973 s | about 21.3 s |
| GPU mempool peak | 66.82 GB | 37.58 GB |

The exact two-cell result is therefore not only more faithful but 6.7 % faster
per steady iteration and 44 % smaller in measured GPU-pool memory.  The lead
current changes by only `1.06e-4` relative, lead balance improves by a factor
12.1, and internal spread improves by 35 %.  Bubble energy balance is
`1.31e-7`.  However, the internal-spread target `1e-3` still fails.  Unlike the
CNT 8x2 result, two-cell Si reblocking is a better reference, not the final
spatial answer.

Decoding the converged 4x12 snapshot back to eight primitive 6-DOF cells makes
the newly admitted shells explicit:

| quantity | off-resonant | broad maximum | narrow pole |
|---|---:|---:|---:|
| primitive band-1 discarded norm | 3.459 % | 1.813 % | 3.073 % |
| distance-2 shell norm | 3.020 % | 1.728 % | 2.698 % |
| distance-3 shell norm | 1.686 % | 0.548 % | 1.470 % |
| global far rank at `1e-3` | 46 / 48 | 41 / 48 | 45 / 48 |
| HODLR symmetry storage / exact reblock at `1e-3` | 1.092 | 0.958 | 1.092 |

At `1e-3`, the off-resonant and narrow cases require rank 12 in every nonzero
HODLR sibling block and use more storage than the exact reblock.  At `1e-2`
they still use 1.058 and 1.025 times reblock storage.  No material negativity
is added at `1e-3`.  Thus the shells lost by the primitive output pin are
high-rank near field, not a compressible tail.  Post-hoc HODLR is a no-go on
this real Si gate.

The two-cell state is still pinned to zero at primitive distances four and
beyond.  The support law permits distance five for `p=1,b=3`, and the residual
internal-current error makes those shells the next reference question.  Use a
three-cell/variable-block reference or an explicit band-five solver to generate
them exactly before testing compression only outside that band.

For films longer than L8, the lifted FC3 archive remains 6.8 MB because the
vertex is local and length-independent.  At fixed q/frequency distribution,
storage and contractions are linear in cell count.  A deliberately
conservative linear projection from L8x2 gives about 75.2 GB/GPU and 65 s per
iteration for L16x2; these are planning estimates, not an L16 measurement.
They fit a 96-GB GH200 narrowly, so L16 should stream/cache the factor action
and avoid wholesale spectral promotion.

## 5. Decision for production work

### Use now

* For CNT, retain the measured conserving 8x2 support.  Keep `g_band=1` as a
  no-go.  Speed the exact result with four-ring reuse and the measured
  atom-sparse FC3 actions.
* For Si, use the exact factor lift for the cheaper L8x2 baseline and future
  wider-block references.  It adds no FC3 fit error beyond the existing
  production factors and avoids the dense q-fold memory; L8x2 alone is not yet
  internally conserved to `1e-3`.
* Correct the production SSE output reach before interpreting any post-hoc
  far-field rank.  A wider Green input band alone cannot create discarded
  output shells.

### Continue as bounded research

* Promote only a small FC3/current-weighted subset of spectral clusters.
  Require a passive positive-anchor source fit and merge output sum poles
  before the Dyson solve.
* Generate spatial auxiliary states directly inside the half-diagram/FC3
  contraction with an exact near band.  Do not fit one-sided propagating modes
  or a completed zero-pinned Sigma.
* Enable self-consistent auxiliary SCBA only if its q-resolved reduced rank,
  full current profile and wall time beat the matched reblocked reference.

### Stop conditions

Reject the auxiliary route for a case if the source basis needs many anchors,
the 0.1 % physical output rank approaches the device DOF, internal current
cannot be decomposed exactly, or the combined smooth-plus-state iteration does
not beat reblocking.  Real L3 and the worst-rank L8 q already trigger the
physical-rank/cost stop for wholesale pole promotion.

## 6. Reproduction

All local commands use the required environment and `eta=0` production states:

```bash
QTX_ARRAY_MODULE=numpy PYTHONPATH=src:phonon \
  /home/paul/miniconda3/envs/quatrex-dev/bin/python \
  phonon/studies/_si_auxiliary_scba_review.py \
  --case L3=cluster/si-aux-l3c/poles.npz \
  --vertices cluster/si-aux-inputs/decomposed_vertices.npz \
  --output phonon/studies/out/si_auxiliary_scba_L3_full.json

QTX_ARRAY_MODULE=numpy PYTHONPATH=src:phonon \
  /home/paul/miniconda3/envs/quatrex-dev/bin/python \
  phonon/studies/_si_auxiliary_scba_review.py \
  --case L8=cluster/si-aux-l8b/poles.npz \
  --vertices cluster/si-aux-inputs/decomposed_vertices.npz \
  --external-q-count 1 \
  --output phonon/studies/out/si_auxiliary_scba_L8_worstq.json

# Run on a memory-capable node holding the converged per-rank snapshot.
QTX_ARRAY_MODULE=numpy PYTHONPATH=src:phonon \
  /home/paul/miniconda3/envs/quatrex-dev/bin/python \
  phonon/studies/_si_long_film_spatial_gate.py \
  --sigma cluster/si-l8x2-final/sigma_last.rank0.npz \
  --pole-states cluster/si-aux-l8b/poles.npz \
  --stored-n-cells 4 --stored-cell-dof 12 \
  --n-cells 8 --cell-dof 6 \
  --output cluster/si-l8x2-spatial/spatial_gate.json

QTX_ARRAY_MODULE=numpy PYTHONPATH=src:phonon \
  /home/paul/miniconda3/envs/quatrex-dev/bin/python -m pytest -q \
  tests/quatrex/phonon/test_auxiliary_scba.py \
  tests/quatrex/phonon/test_si_auxiliary_scba_review.py \
  tests/quatrex/phonon/test_si_long_film_spatial_gate.py \
  tests/quatrex/phonon/test_vertex_factors.py \
  tests/quatrex/phonon/test_btd_linalg.py \
  tests/quatrex/phonon/test_reblock_device.py
```

The complete focused review suite contains 69 passing tests in the
`quatrex-dev` environment (`QTX_ARRAY_MODULE=numpy`).

Alps provenance:

* jobs 4552938/4552943: L8/L3 real pole-state extraction;
* job 4552964: primitive L8 spatial snapshot gate;
* job 4552982: exact L8x2 device and FC3-factor lift;
* job 4552987: six-iteration warm L8x2 production SCBA;
* job 4553002: 24-iteration L8x2 convergence continuation;
* job 4553056: final 17-iteration L8x2 continuation to `9.511e-4`;
* job 4553064: forced matched primitive L8 timing/memory control;
* job 4553127: converged L8x2 far-shell/HODLR gate.

The broader method transferability review, reduced frequency sweep, CNT FC3
compression retry and hierarchical spatial studies remain in
[`phph_acceleration_review.md`](phph_acceleration_review.md).
