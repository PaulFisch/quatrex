# Nonuniform frequency grids in phonon--phonon SCBA

**Status:** reduced direct/auxiliary oracles and matched CNT/long-Si production
A/B complete, 2026-08-28

**Production changes:** none.  The study and preparation tools are private to
`phonon/studies/`.

**Reproduction:**

```bash
PYTHONPATH=phonon python phonon/studies/_nonuniform_grid_review.py \
  --json phonon/studies/out/nonuniform_grid_review.json
PYTHONPATH=phonon python phonon/studies/_nonuniform_production_review.py \
  --json phonon/studies/out/nonuniform_production_review.json
PYTHONPATH=phonon pytest -q \
  tests/quatrex/phonon/test_nonuniform_grid_review.py \
  tests/quatrex/phonon/test_prepare_nonuniform_production.py \
  tests/quatrex/phonon/test_nonuniform_production_review.py
```

The scripts are
[`_nonuniform_grid_review.py`](../studies/_nonuniform_grid_review.py) and
[`_prepare_nonuniform_production.py`](../studies/_prepare_nonuniform_production.py),
and the pulled-run normaliser is
[`_nonuniform_production_review.py`](../studies/_nonuniform_production_review.py).
The full tables are
[`nonuniform_grid_review.json`](../studies/out/nonuniform_grid_review.json) and
[`nonuniform_production_review.json`](../studies/out/nonuniform_production_review.json).

## Verdict

A nonuniform frequency basis **can work accurately**.  Exact product
integration of the reduced nonuniform P1 basis keeps the bubble L2 error below
`6.1e-3` across a `1000x` linewidth sweep while its input size grows only from
263 to 814 points.  However, that direct reference costs quadratically and is
not what production computes.

The existing Quatrex transfer operators implement their claimed weighted
adjoint identity exactly.  They can reduce the number of Dyson/OBC solves and
primary `G/Sigma` storage, but that identity alone does not make the nonlinear
SCBA map conserving.  Nor is the bridge, by itself, a solution to
under-resolved narrow lines in the bubble.

The reason is structural.  Production does not convolve on the nonuniform
grid.  It interpolates both Green-function legs to a uniform auxiliary grid,
runs the ordinary FFT/FC3 ring there, and restricts the result back.  A narrow
line therefore still requires the auxiliary spacing to track its linewidth.
The inverse-linewidth cost has moved out of Dyson, not disappeared.

The reduced test makes this distinction decisive:

* while the linewidth shrinks by `1000x`, an accurate graded primary grid grows
  only from 263 to 814 points;
* direct nonuniform P1 product integration remains bounded at
  `2.28e-3` bubble L2 error in the narrow-line limit, without a fine uniform
  grid, but its reference implementation costs `O(N_p N_out)`;
* keeping the auxiliary grid at the background spacing makes the bubble
  relative error grow from `9.995e-3` to `1.194e2`;
* retaining eight auxiliary points per HWHM bounds the bubble error at
  `5.94e-3` and, below `gamma/h=0.2`, about `1.7e-3`, but grows the auxiliary
  grid from 257 to 256001 points;
* the energy-weighted adjoint restriction preserves its defining pairing to
  `2.44e-16`.  Point sampling has no such identity (a deliberately generic
  random probe gives a `3.72` relative defect).
* the longer, reblocked Si film converges with 113 primary / 121 auxiliary
  points: current changes by `-3.946e-4`, spread is unchanged, and iteration
  time falls only `2.82%`;
* an aggressive 67 / 161 CNT grid reaches residual `7.98e-4` and current error
  `8.40e-4`, but fails the lead-conservation gate with `8.68%` imbalance.
  A safer 108 / 161 grid also fails: at residual `5.80e-4`, its current error
  is `-3.90e-3` and lead imbalance is `1.86%`.

Thus the answers to the two questions are different:

| question | answer |
|---|---|
| Can a sparse nonuniform primary grid reproduce a converged spectrum/current? | **Yes, conditionally**, if its interpolation error is controlled and the basis is frozen during an SCBA epoch. |
| Can a genuinely nonuniform collision discretisation resolve narrow bubbles? | **Yes in the reduced oracle.** Exact P1 product integration stays accurate with logarithmic point growth, but the direct algorithm is quadratic and still needs conserving moments. |
| Can the present nonuniform-primary/uniform-auxiliary path remove the fine frequency convolution? | **No.** The auxiliary grid still needs the narrow-line resolution. |
| Can it still be cheaper? | **Sometimes**, when Dyson/OBC or primary memory dominates. It offers little wall-time gain when the FC3 ring dominates. |
| What can remove both costs? | A mixed collision basis: adaptive polynomial cells with fast sparse/multiresolution or NUFFT-assisted connection coefficients, plus selective passive rational clusters for subcell poles and their combination outputs. |

## What production actually computes

Let primary coefficients be `g_m` at nonuniform frequencies `omega_m`, and
let the auxiliary frequencies be `nu_n=n h_a`.  The production interpolation
is the nonnegative P1 operator

\[
  (P g)_n=(1-t_n)g_{m(n)}+t_n g_{m(n)+1}.
\]

Both Keldysh legs are transferred with `P`, the existing folded FFT bubble is
formed on `nu`, and `Sigma_aux` is restricted to the primary grid.  The
default restriction is

\[
 R=(W_p\Omega_p)^{-1}P^T(h_a\Omega_a),
\]

where `W_p` contains primary cell widths.  Consequently

\[
 \sum_m w_m\omega_m(R\Sigma)_m g_m
 =h_a\sum_n\nu_n\Sigma_n(Pg)_n
\]

to rounding.  This is the correct transfer for the energy balance used by the
phonon current.  The alternative `sample` restriction can look sharper at an
individual output node, but it is not the adjoint of the Green-function
reconstruction and does not transfer the discrete energy identity.

### Why the adjoint identity is necessary but not sufficient

The CNT A/B exposes a limitation that the earlier bridge unit test could not.
Exact transfer of one pairing does **not** make the entire interpolated bubble
a conserving nonuniform collision discretisation.  Frequency multiplication
must also be represented consistently.  For nodal interpolation that would
require

\[
 \Omega_a P=P\Omega_p.
\]

This is false whenever an auxiliary node lies between primary nodes.  The
dimensionless Frobenius defect

\[
 \epsilon_\Omega=
 \frac{\|\Omega_aP-P\Omega_p\|_F}{\|\Omega_aP\|_F}
\]

is `0.0451` for aggressive CNT, `0.00494` for the continuum-safe CNT grid,
and `0.00312` for the successful near-uniform Si grid.  The corresponding
maximum row defects are 3.52, 0.243 and 0.118 THz.  This is a useful screening
metric, not a universal error bound, but it explains why large gaps can retain
the adjoint pairing yet lose end-to-end lead conservation.

In a true nonuniform Galerkin method, frequency is not represented only by the
diagonal nodal values.  One carries a mass matrix and first-moment matrix,

\[
 (M_0)_{ij}=\int\phi_i\phi_j\,d\omega,
 \qquad
 (M_1)_{ij}=\int\omega\phi_i\phi_j\,d\omega,
\]

and constructs the collision connection coefficients so the zeroth/first
moment identities hold in that same basis.  Alternatively, a low-rank
conservative correction can impose the moments after an approximate fast
convolution.  Merely changing `R` cannot make a two-node interpolant commute
with frequency multiplication for arbitrary gaps.

The implementation lives in
[`sse_phonon_phonon.py`](../../src/quatrex/phonon/sse_phonon_phonon.py):
`_aux_grid_plan`, `_interp_axis0`, and `_restrict_from_aux`.  Existing tests in
[`test_sse_aux_grid.py`](../../tests/quatrex/phonon/test_sse_aux_grid.py)
already establish:

* identity with the legacy path when primary and auxiliary grids coincide;
* agreement with a fine uniform Lorentzian-comb reference for both restriction
  choices;
* exact weighted-adjoint pairing;
* correct DC masking and auxiliary-support extension.

Those are implementation tests.  They did not answer the cost/asymptotic
question or establish a converged device A/B.

## Reduced decisive experiment

### Oracle and grid construction

The positive test leg is a two-line Lorentzian mixture

\[
 A(\omega)=\sum_j a_j
 \frac{\gamma_j}{\pi[(\omega-c_j)^2+\gamma_j^2]},
 \qquad a_j\ge0.
\]

Its bubble has an analytic infinite-line oracle because Cauchy densities are
closed under convolution:

\[
 L(c_i,\gamma_i)*L(c_j,\gamma_j)
 =L(c_i+c_j,\gamma_i+\gamma_j).
\]

The primary grid contains a coarse background and graded
`gamma sinh(t)` nodes around every input and combination line.  This detail is
important.  A fixed narrow patch is insufficient: its Lorentzian value is
still large at the patch edge, so linear interpolation to a distant coarse
node creates a broad artificial triangular wing.  Geometric grading controls
the tails with only logarithmic growth in the inverse linewidth.  The CDF
construction in [`make_grid.py`](../studies/make_grid.py) is another valid way
to obtain this grading, provided the acoustic continuum receives its own
spacing floor.

The sweep uses `gamma/h={1,0.2,0.04,0.008,0.001}` and pole offsets
`{0,0.25h,0.49h}`.  The auxiliary choices are `h`, `h/2`, and `gamma/8`.

### Results

Each entry is the worst result over the three offsets.

| auxiliary rule | `gamma/h` | primary pts | auxiliary pts | leg L2 error | bubble L2 error | peak-area error |
|---|---:|---:|---:|---:|---:|---:|
| `h` | 1 | 263 | 33 | 0 | `9.995e-3` | `2.758e-2` |
| `h` | 0.2 | 391 | 33 | 0 | `1.158` | `0.764` |
| `h` | 0.04 | 522 | 33 | 0 | `2.011` | `7.143` |
| `h` | 0.008 | 650 | 33 | 0 | `14.06` | `202.6` |
| `h` | 0.001 | 814 | 33 | 0 | `119.4` | `1.303e4` |
| `h/2` | 1 | 263 | 65 | `1.038e-3` | `5.847e-3` | `2.619e-2` |
| `h/2` | 0.2 | 391 | 65 | `2.473e-3` | `0.388` | `0.306` |
| `h/2` | 0.04 | 522 | 65 | `1.068e-3` | `0.975` | `0.964` |
| `gamma/8` | 1 | 263 | 257 | `1.514e-3` | `5.940e-3` | `2.696e-2` |
| `gamma/8` | 0.2 | 391 | 1281 | `2.011e-3` | `1.481e-3` | `1.498e-3` |
| `gamma/8` | 0.04 | 522 | 6401 | `1.934e-3` | `1.655e-3` | `2.046e-3` |
| `gamma/8` | 0.008 | 650 | 32001 | `1.945e-3` | `1.671e-3` | `2.081e-3` |
| `gamma/8` | 0.001 | 814 | 256001 | `1.945e-3` | `1.675e-3` | `2.092e-3` |

The zero leg error on the `h` rows is not evidence of accuracy: those
auxiliary nodes are also background primary nodes, so interpolation is exactly
the identity *at the wrong quadrature nodes*.  The convolution misses or
overweights the subcell line depending on registration.  This is precisely
why inspecting only reconstructed `G` at auxiliary nodes can give a false
pass.

The broadest line has a roughly 2.7% peak-window error even on the fine rule
because the oracle is infinite-line while the numerical leg is truncated at
the finite test boundary.  It is independent of the subcell asymptotic result;
all narrower cases settle near 0.2%.

### Direct nonuniform collision arm

The second arm removes the uniform auxiliary grid completely.  For every
output frequency it merges the primary P1 breakpoints with their reflected
breakpoints.  The two factors are linear on each resulting interval, so their
quadratic product is integrated exactly.  The only approximation is therefore
the graded P1 representation itself.

| `gamma/h` | input points | output points | bubble L2 error | peak-area error |
|---:|---:|---:|---:|---:|
| 1 | 263 | 228 | `6.023e-3` | `2.689e-2` |
| 0.2 | 391 | 304 | `1.964e-3` | `1.922e-3` |
| 0.04 | 522 | 381 | `2.254e-3` | `2.573e-3` |
| 0.008 | 650 | 458 | `2.277e-3` | `2.605e-3` |
| 0.001 | 814 | 557 | `2.281e-3` | `2.611e-3` |

This is the missing positive result: nonuniformity itself is not the no-go.
The direct implementation is `O(N_p N_out)` and therefore only an oracle.
The production research problem is to retain its local accuracy and moment
identities while replacing the all-pairs evaluation by sparse
multiresolution connection coefficients, low-rank separated kernels, or an
oversampled NUFFT where appropriate.

There is a real combinatorial constraint: arbitrary P1 knots generate up to
`O(N_p^2)` pairwise-sum breakpoints in the convolution output.  A fast
production version should therefore use a nested dyadic/multiresolution cell
family, where most cross-level connection patterns repeat and small
coefficients can be certified away, rather than an unconstrained list of file
frequencies.  NUFFT gridding is another possible accelerator, but it remains
an approximate transform and needs an explicit moment correction.  It also
does not eliminate the bandwidth requirement for a narrow feature unless that
feature is represented separately, which is the role of rational clusters in
the mixed proposal.

The closest transferable numerical literature supports this distinction.
[NUFFT convolutional gridding](https://doi.org/10.1137/13092160X) provides a
convergent fast reconstruction from irregular samples, but it does not by
itself make a nonlinear collision functional conservative or resolve data
that were never represented.  Conservative kinetic schemes instead include
the collision invariants in the test space, as in a
[fully conservative Galerkin--Petrov method](https://doi.org/10.3934/krm.2019021),
or impose moment constraints while retaining fast spectral algorithms, as in
[moment-preserving Fourier--Galerkin methods](https://doi.org/10.1137/21M1423452).
For Quatrex the analogous invariant is the phonon energy moment of the full
bilinear Keldysh bubble, not merely one reconstruction/restriction pairing.

## Production evidence and matched A/B

### What was known before this test

The stored MoS2 grid ladder is useful but not decisive because it was stopped
after eight SCBA iterations with residuals around 0.34--0.46.  Relative to its
15001-point primary transient:

* 4001 and 2001 uniform primary points changed the integrated current by
  `+0.31%` and `-0.02%`;
* auxiliary spacings 0.02, 0.01 and 0.005 THz shifted comparable transients by
  about `+10.2%`, `+7.7%`, and `+4.4%`;
* `sample` restriction at 0.01 THz happened to give `+0.99%`, but lacks the
  adjoint energy identity.

This already suggested that primary coarsening can be cheap while auxiliary
coarsening is the dangerous operation.  It could not establish the fixed
point ordering.

The cross-structure reconstruction pilot in
[`mixed_representation_strategy.md`](mixed_representation_strategy.md) found:

| saved case | reference points | adaptive points at `1e-3` | uniform points at `1e-3` |
|---|---:|---:|---:|
| conserving CNT L16x2 | 161 | 67 | 154 |
| conserving Si L8x2 | 121 | 113 | 120 |
| MoS2 L3 ballistic pilot | 262 | 66 | 257 |

That study reconstructed saved observables only.  The following matched runs
put the selected grid inside the production SCBA map while retaining the
reference auxiliary spacing.

### Matched production cases

| case | spatial/reference state | uniform primary | nonuniform primary | auxiliary | outcome |
|---|---|---:|---:|---:|---|
| CNT 8x2 aggressive, `g_band=3` | converged uniform reference, spread `5.04e-4` | 161 | 67 | 161, `h=0.34375 THz` | job `4553266`: stopped after decisive failure, residual `7.98e-4`, current error `+8.40e-4`, imbalance `8.68e-2` |
| CNT 8x2 continuum-safe, `g_band=3` | same | 161 | 108 | 161, `h=0.34375 THz` | job `4553300`: stopped after failure, residual `5.80e-4`, current error `-3.90e-3`, imbalance `1.86e-2` |
| Si L8x2, `g_band=3` | converged, current spread `2.83e-3` | 121 | 113 | 121, `h=0.125 THz` | job `4553267`: converged in 25 iterations; current error `-3.946e-4`, spread `2.8294e-3` |

The Si case is intentionally the longer, reblocked film requested for the
spatial investigation.  It does not use the known-invalid primitive
`g_band=1` reference.

### Cost ceiling before measurement

For the uniform conserving CNT run, median per-iteration times are:

| component | time |
|---|---:|
| Dyson/OBC/selected solve | 0.829 s |
| FC3 ring | 11.936 s |
| total iteration | 12.901 s |

Even eliminating all primary solve cost would save only 6.4%.  Scaling that
component by `67/161` predicts roughly 3.8% total savings before interpolation
overhead.  The nonuniform path cannot accelerate the dominant ring because it
still has 161 auxiliary points.

The measurement is consistent with that ceiling.  Aggressive CNT reduces the
median solver time from 0.829 to 0.083 s, but the ring changes from 11.936 to
12.019 s.  Total iteration time is 12.227 rather than 12.901 s, a `5.23%`
speedup, and the run fails conservation.

The continuum-safe CNT grid reduces the maximum omitted interval from 9.97 to
0.6875 THz and the frequency-moment intertwining defect from `4.51e-2` to
`4.94e-3`.  It nevertheless reaches `1.86%` imbalance at residual `5.80e-4`.
Its median iteration is 12.245 s (`5.09%` faster); the FC3 ring is 12.001 s,
slightly slower than the 11.936 s reference.  The job was deliberately
cancelled after this predeclared rejection gate rather than spending the rest
of its allocation.

For Si the primary reduction is only `6.6%`.  It reduces median solver time
from 10.119 to 9.330 s, leaves the ring at 21.175 versus 21.295 s, and reduces
the total iteration from 32.232 to 31.324 s (`2.82%`).  Peak GPU-pool memory is
37.35 versus 37.58 GB (`0.61%` lower), because the unchanged auxiliary and
spatial workspaces dominate.  Its value as an A/B is the passed accuracy gate,
not the small speedup.

## When nonuniform primary sampling is useful

With cell DOF `d`, length `L`, primary size `N_p`, auxiliary size `N_a`, and
FC3 representation cost `C_V`, a useful planning model is

\[
 T_{iter}\approx N_p C_{Dyson}(L,d)
 +N_a\log N_a\,C_{FFT}
 +N_a C_V
 +T_{transfer}(N_p,N_a).
\]

Primary `G/Sigma` memory is proportional to `N_p`; FFT/tau workspaces and the
ring are proportional to `N_a`.  Therefore nonuniform primary sampling is a
good production option when at least one of the following holds:

1. the device solve, contact solve, or selected inversion dominates the ring;
2. primary Green-function storage is the memory bottleneck;
3. a fine auxiliary grid is needed only transiently or in a small selected
   sector, while most Dyson outputs are smooth;
4. a subsequent mixed-basis implementation removes narrow clusters from the
   uniform auxiliary grid.

It is not a good acceleration when the tensor/dense/sparse FC3 action dominates
and `N_a` is unchanged, which is the present CNT situation.

## How it fits the mixed approach

The nonuniform work should be kept, but used as one component rather than the
whole frequency solution.  The cross-structure representation should be

\[
 G(\omega)=G_C(\omega)+G_A(\omega)+
 \sum_s U_s(\omega I-K_s)^{-1}Q_s
 (\omega I-K_s)^{-H}U_s^\dagger.
\]

Here `C` is a coarse uniform P1 grid, `A` is a nested collection of adaptive
positive hats/cells, and the last term contains only coherent subcell clusters
that pass passivity, source-fit, FC3-importance and cost gates.

The collision engine then uses:

* coarse cell--cell: existing FFT ring;
* adaptive-cell sectors: local product-integration or sparse multiresolution
  connection coefficients;
* cluster--cell: analytic cell moments applied as Toeplitz kernels;
* cluster--cluster: analytic rational/state-space closure;
* narrow combination outputs: retained as rational auxiliary states, not
  resampled immediately;
* one mass/energy adjoint projection for every reconstruction operator.

This removes the need for `N_a~1/gamma_min` while retaining a non-rational
fallback for clusters whose Keldysh source cannot be fitted passively.  The
existing nonuniform path is then valuable twice: as the adaptive-cell
prototype and as the sparse set of physical frequencies where the augmented
Dyson solve is requested.

## Structure-independent decision rule

At a fixed-basis SCBA epoch:

1. Estimate interpolation residuals in spectral weight, current integrand and
   FC3-weighted bubble probes, including a low-frequency acoustic/continuum
   spacing floor.  Endpoint current/LDOS reconstruction alone is insufficient:
   it selected 67 CNT nodes while omitting internal-current/bubble content
   which later failed conservation.
2. If P1 refinement meets the target and the predicted Dyson/memory saving
   exceeds transfer overhead, keep a nonuniform primary grid.
3. If the auxiliary grid required by a feature is much finer than the coarse
   background, try passive rational promotion.
4. Accept promotion only when a coherent cluster source fit and its
   combination-output rank meet the physical and cost gates; otherwise keep
   adaptive local cells.
5. Freeze grid, cluster membership, and transfer operators during convergence.
   Change them only between epochs and restart mixer history.
6. Compare current, bubble energy balance, positivity and residual at the
   converged fixed point—not only reconstructed spectra or the first iterate.

This rule naturally predicts different representations without testing a
material name:

* **CNT:** the unchanged FC3 ring limits speed.  The aggressive production
  grid reproduces average current but fails lead conservation, and the
  continuum-safe grid still reaches `1.86%` imbalance.  The current bridge is
  therefore a no-go for CNT without a conservative collision projection and a
  stronger bubble/moment estimator.  No rational promotion is currently
  justified because its measured lines are resolved.
* **long Si:** the saved observables need almost every coarse node, while
  narrow clusters exist.  Mild nonuniformity passes current/conservation but
  saves only `2.82%`; use the coarse grid plus selective rational/local
  enrichment rather than expecting primary sampling alone to accelerate it.
* **MoS2:** primary compression appears large, but a converged interacting
  spatially valid reference is still required.  Its existing 0.02 THz
  auxiliary grid remains a cost, not evidence that subcell poles are resolved.

## Recommendation

Keep `frequency_grid="file"` and the energy-adjoint bridge as private research
tools.  They may reduce memory or an expensive Dyson solve, but each structure
requires a converged conservation A/B and a frequency-moment defect gate; the
CNT results show that the adjoint identity alone is not a physical guarantee.
Do not advertise a nonuniform primary grid as the mechanism that resolves
narrow SCBA poles, and do not coarsen the auxiliary grid based on a successful
primary-spectrum reconstruction.

The next production implementation should expose one internal mixed-frequency
state, with adaptive cells and selected passive rational clusters sharing the
same bilinear bubble and adjoint projection.  That is the route which can
simultaneously avoid a huge frequency grid, retain the conserving 8x2 spatial
physics, and reduce the dominant FC3 work.
