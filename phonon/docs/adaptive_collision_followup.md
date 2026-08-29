# Adaptive collision basis and production follow-up

**Status:** exact reduced implementation, real-Si spectral oracle and matched
CNT production A/B, 2026-08-29

**Default production behaviour:** unchanged.  The existing exact
`sse_greater_from_lesser` production path was exercised through a study-only
environment override.  The adaptive collision implementation remains private
to `phonon/studies/` because the real-Si cost gate below does not pass.

**Reproduction:**

```bash
PYTHONPATH=phonon python phonon/studies/_adaptive_p1_fct_review.py \
  --tolerance 1e-4 \
  --json phonon/studies/out/adaptive_p1_fct_review.json

PYTHONPATH=phonon python phonon/studies/_si_adaptive_fct_oracle.py \
  --run cluster/sichk_res/run.npz --tolerances 0.02 \
  --normalisation global --detector oracle --input-scope shared \
  --json phonon/studies/out/si_adaptive_fct_oracle.json

PYTHONPATH=phonon pytest -q \
  tests/quatrex/phonon/test_adaptive_p1_fct.py \
  tests/quatrex/phonon/test_adaptive_p1_fct_review.py
```

The implementation is
[`_adaptive_p1_fct.py`](../studies/_adaptive_p1_fct.py), the analytic sweep is
[`_adaptive_p1_fct_review.py`](../studies/_adaptive_p1_fct_review.py), and the
15,001-point Si oracle is
[`_si_adaptive_fct_oracle.py`](../studies/_si_adaptive_fct_oracle.py).

## Decision

There are two separate results.

1. **A genuinely nonuniform collision can be made exact and conserving at the
   frequency-discretisation level.**  The implemented Hackbusch fast
   convolution transform computes the exact L2 projection of the convolution
   of discontinuous P1 functions on locally refined dyadic meshes.  Scalar
   multiplication is replaced by an arbitrary bilinear callback, including
   Quatrex's noncommutative FC3 ring.  A final tridiagonal mass solve produces a
   continuous P1 result.  Zeroth and first collision moments, noncommutative
   ordering, diagonal Keldysh anti-Hermiticity and the FC3 index contraction
   pass to roundoff.
2. **One shared adaptive grid is not yet a cheaper Si production backend.**
   The union of moving q-resolved resonances fragments the multilevel support.
   On the available 15,001-point Si oracle, a reliable globally weighted mesh
   has 373 cells, but the current exact recurrence executes about 4.03 million
   transformed modes for a representative channel.  The fine uniform scalar
   oracle needs 60,001 modes.  A blind five-point detector is worse: it uses
   1,501 cells and still gives 9.1% bubble error on a typical channel because a
   narrow line can fall between all probes.  This rejects direct production
   integration of the current shared-grid FCT.

The result is not that spatial and spectral approximation cannot work.  It
defines a narrower mixed architecture:

* retain the exact, conserving reblocked spatial near field;
* execute CNT FC3 through exact primitive/atom-sparse actions and exact
  four-ring Keldysh reuse;
* retain the existing tensor-factor/q-FFT Si contraction;
* use pole locations only to create **q-local auxiliary cells**, without
  subtracting an additive Keldysh pole remainder;
* represent those q-local cells as a block-sparse joint `(q, frequency)`
  multiresolution object and carry combination-frequency output cells through
  the SCBA map;
* promote a rational cluster only if a passive multi-anchor source fit passes.

That q-local auxiliary representation is a substantial DSDBSparse/SCBA
refactor.  It is the next research implementation, not a flag that can be
safely added to the current rectangular frequency-by-q arrays.

## Implemented algorithm

Let level `l` have cell width `h0/2**l` and the orthonormal local basis

\[
 \phi_{i0}=h_l^{-1/2},\qquad
 \phi_{i1}=\sqrt{12/h_l}\,(\omega-\omega_i)/h_l.
\]

For native-level pieces `f_l`, `g_m`, the code evaluates

\[
 P_n(f_l*g_m)
\]

by the three exact recurrences in Wolfgang Hackbusch's
[projected-convolution algorithm](https://doi.org/10.1007/s00607-007-0229-z):

| sector | level ordering | operation |
|---|---|---|
| A | `n <= l <= m` | projected Gamma coefficients, discrete FFT convolution, exact coarsening |
| B | `l < n <= m` | exact P1 refinement followed by projected discrete convolution |
| C | `l <= m < n` | value/left-derivative/right-derivative convolutions and exact cubic refinement |

The combined implementation intertwines all native levels before the FFTs,
as in sections 5.1.4, 5.2.3 and 5.3.4 of the paper.  Disconnected supports are
split before padding, independent FFT products with equal padded length are
batched, and the callback sees one concatenated Fourier-mode axis.  The
transcription is independently checked against piecewise Gauss integration of
all A/B/C orderings.  It also supports two different input meshes.

The key production observation is bilinearity.  If `B(A,B)` is Quatrex's
frequency-independent FC3 ring, every scalar discrete convolution

\[
 \widehat c_k=\widehat a_k\widehat b_k
\]

is replaced exactly by

\[
 \widehat C_k=B(\widehat A_k,\widehat B_k).
\]

No Kronecker-sized frequency-space tensor is formed.  The existing dense,
atom-sparse, or tensor-factorised spatial action can implement `B`.

The discontinuous projection is converted to continuous nodal P1 through the
adaptive tridiagonal mass matrix.  Since `1` and `omega` belong to that space,
the two identities

\[
 M_0(f*g)=M_0(f)M_0(g),\qquad
 M_1(f*g)=M_1(f)M_0(g)+M_0(f)M_1(g)
\]

are retained to roundoff.  This is the missing frequency-multiplication
consistency of the old nonuniform-primary/uniform-auxiliary bridge.

## Reduced narrow-line result

The test is the same offset sweep used in the earlier nonuniform review: two
Lorentzians, three offsets relative to the background cell, and
`gamma/h={1,0.2,0.04,0.008,0.001}`.  At mesh tolerance `1e-4`, the worst
offset at each linewidth is:

| `gamma/h` | input/output cells | equivalent uniform input cells | bubble L2 error | ring calls | transformed modes |
|---:|---:|---:|---:|---:|---:|
| 1 | 294 / 266 | 2,048 | `5.96e-3` | 457 | 128,310 |
| 0.2 | 325 / 303 | 8,192 | `8.56e-4` | 647 | 137,164 |
| 0.04 | 353 / 318 | 65,536 | `8.74e-4` | 960 | 155,566 |
| 0.008 | 361 / 331 | 262,144 | `1.25e-3` | 1,315 | 199,680 |
| 0.001 | 480 / 336 | 2,097,152 | `8.67e-4` | 1,370 | 236,986 |

The broad-line `5.96e-3` floor is dominated by comparing a finite-window input
with an infinite-line analytic Cauchy oracle.  The subcell cases pass the
`2e-3` target and their cost is independent of inverse linewidth apart from
logarithmic mesh depth.  This is a positive mathematical result and a useful
oracle for future auxiliary representations.

It is not automatically a production speedup.  The existing 161-point CNT
grid needs a 321-point FFT and its narrowest recorded linewidth is 1.57 cells;
the adaptive transform has no crossover there.  It is only relevant when the
accurate uniform grid would contain many tens of thousands of points.

## Real-Si test

The oracle is the saved `sichk_res` three-cell film run: 15,001 positive
frequencies, 81 transverse q points and 18 device DOF.  The source run diverged
after four SCBA iterations, so it is used only as a real spectral-complexity
bed, not as a physical current reference.  Diagonal greater data are recovered
from the exact diagonal identity

\[
 \operatorname{Im}G^>=\operatorname{Im}G^< -2\operatorname{Im}G^R,
\]

then mirrored to the negative axis.  A fine uniform FFT supplies the scalar
convolution oracle.

### Mesh detection

| detector / norm | shared input cells | result |
|---|---:|---|
| five fixed probes per cell, global norm | 1,501 | sharp/large channels have `0.35--0.39%` continuous bubble error; a typical lower-weight channel has `9.12%` |
| every fine oracle point, global norm | 373 | no hidden-line alias for globally material features; typical low-weight channel is deliberately not resolved relatively |
| certified, individual-channel norm | 344 / 344 / 613 | sharpest / largest-weight / typical channel local meshes |

The five-point failure is important: an adaptive grid chosen only from current
nodal residuals is not safe for poles.  A production detector must use the
existing nonlinear eigenmode/pole locator, a nested verification set, or both.
Unlike the failed pole sector, location is used only to refine cells; no
nonphysical additive Keldysh remainder is created.

### Cost

On the 373-cell shared mesh, all three representative scalar runs execute
about 4.03 million transformed modes and have a maximum concatenated batch of
314,928 modes.  On q-local certified meshes the two dominant channels fall to
99,796 modes, while the spectrally complicated typical channel still requires
29.9 million at the same nominal tolerance.  The uniform fine oracle is one
60,001-mode FFT.

The reason is not linewidth but **support fragmentation**.  Different q/mode
branches refine different frequency cells.  A rectangular shared mesh forces
one channel through levels introduced for another, and Case C repeatedly sees
many disconnected pieces.  This is also why merely porting the NumPy control
flow to CuPy would not fix the crossover.

The remaining fast option is a block-sparse joint q-frequency hierarchy.  It
must retain q-dependent native-level pieces instead of taking their union.
Doing them independently loses the existing q FFT and tends toward `Nq^2`
pair work; doing them jointly requires low-rank or sparse q support at each
frequency level.  Relevant comparisons are multilevel composite-grid
convolution
([Khoromskij et al.](https://doi.org/10.1016/j.cam.2010.02.004)), adaptive
multiwavelet operator application
([Beylkin et al.](https://arxiv.org/abs/0706.0747)), and quantics tensor-cross
representations of many-body frequency objects
([Murray et al.](https://arxiv.org/abs/2312.03809)).  NUFFT alone is not a
solution: arbitrary nonuniform convolution samples are not closed under
frequency addition, and the q-dependent output still needs a conserving
basis.

## Production result that does pass now: conserving CNT four-ring reuse

The CNT frequency grid does not need adaptive treatment.  Its production
bottleneck is the dense merged 72-DOF ring on the conserving 8x2 layout.  The
existing exact identity reconstructs the two mixed greater terms from the
lesser pass and changes six dense rings per vertex quad to four.

The matched Daint run uses one archived `c16x2h` four-rank checkpoint,
unchanged `g_band=3`, unchanged 8x2 spatial support, the same FC3 and ten
forced SCBA iterations.  The only A/B difference is the exact greater-sector
reconstruction.  Excluding the first-call setup/warm-up, it measures:

* four-ring continuation: job `4554438`, `cluster/c16x2-4ring-final`;
* six-ring control: job `4554440`, `cluster/c16x2-6ring-control`.

| quantity | six-ring control | four-ring |
|---|---:|---:|
| rings per quad | 6 | 4 |
| model work/pass | 319.6 TFLOP | 213.1 TFLOP |
| steady ring median | 11.9631 s | 7.9904 s |
| steady iteration median | 12.2132 s | 8.2379 s |
| speedup | 1 | **1.50x ring / 1.48x iteration** |
| final lead current | `33.8245021252427` | `33.8245021252464` |
| final internal spread | `4.6526e-4` | `4.6526e-4` |
| final retarded residual | `1.6716e-4` | `1.6716e-4` |

The full saved A/B states also pass the algebraic gate.  The final lead
currents differ by `1.08e-13` relatively; the maximum relative differences are
`3.68e-19` for diagonal `Im G^R`, `4.48e-13` for diagonal `Im G^<`, and
`1.29e-11` for the current spectrum.  NaN masks are identical.  The earlier
three-iteration probe's `-0.49%` current shift relative to the old checkpoint
was therefore ordinary continuation of a loosely converged state, not a
four-ring approximation error.

This particular gain does not transfer unchanged to the production Si tensor
factor kernel.  That kernel already factors the quad sum and merges the six
bosonic fold terms into four q-convolutions.  Greater-from-lesser
reconstruction would still need the lesser decay term, its two cross terms,
and the greater decay term separately (four convolutions), plus a third
external `D_t H D_t^T` sandwich for the cross accumulator.  It can remove one
Gram variant but does not reduce the dominant q-convolution count and can add
work.  Si should retain the tensor-factor/q-FFT contraction unless a measured
kernel A/B overturns this cost count.

The next CNT implementation remains the exact primitive/atom-sparse ring, not
an approximate FC3 fit.  On the finite CNT vertex, global INDSCAL is still
4.52% wrong at rank 256, whereas the exact atom-triplet representation is
13.56%/4.52% of dense storage for onsite/cross blocks and the GH200 kernel is
1.92x faster after task weighting.  Combining that measured kernel ratio with
four-ring reuse projects about 3.60 s per conserving iteration.  The exact
dual-layout production contraction is still to be wired.

## Longer Si

The longer reblocked L8 film already gives the correct priority ordering:

* 121-point uniform run: converged in 17 iterations, internal spread
  `2.8288e-3`, lead imbalance `5.85e-6`, steady iteration about 32.25 s;
* 113-primary/121-auxiliary nonuniform run: normalized current changes by
  `-3.95e-4`, spread is unchanged, and the iteration improves only 2.82%;
* the present reblock generates primitive distance-2 and partial distance-3
  Sigma shells that were absent before, but still does not reach the `1e-3`
  current-spread target;
* those shells are high-rank near field, not an HSS tail.

Frequency nonuniformity therefore does not repair the remaining L8
conservation defect.  For L16 and longer, first generate the exact FC3/G graph
closure through the current-certified spatial band with a block-banded
selected solve.  At fixed band this remains linear in film length.  Only then
test a hierarchical residual.  Spectrally, retain a common coarse grid for the
smooth sector and add q-local auxiliary cells only when a pole-informed
fine-grid oracle shows that the 121-point result is inaccurate; conservation
alone is not evidence of spectral convergence.

## Concrete next implementation

The smallest credible production refactor has four internal objects:

```text
CoarseFrequencyField
    common cells, mass M0, first moment M1, adjoint reconstruction/projection

SparseQFrequencyPatch
    (q, dyadic level, cell run, selected G blocks)
    locations frozen for one SCBA epoch

RationalCluster (optional)
    coherent K, U, PSD multi-anchor source Q(omega)
    accepted only after passive bubble and combination-rank gates

CollisionOutput
    coarse coefficients + q-local combination cells + optional rational states
```

The Dyson/RGF batcher must accept a ragged list of `(q, omega)` evaluations.
The collision callback remains the existing tensor-factor or exact sparse FC3
action.  Mixer history is held in the fixed enriched coefficient basis and is
restarted only when the basis changes.  Retarded data are reconstructed once
from the same compressed spectral difference; no separately fitted
`Sigma^R` is allowed.

Acceptance requires a frozen-G equality test first, then a closed SCBA A/B:

* normalized `Sigma^{<,>}` and current-spectrum error below `2e-3`;
* zeroth/first collision moment defect below ten times the fine reference;
* Keldysh symmetry defect below `1e-10` and no new positivity defect;
* current within 1% and conservation no worse than the exact reblocked
  reference;
* measured FC3 work below both the accurate uniform grid and reblocking.

Until a block-sparse q-frequency prototype passes those gates, the production
recommendation is unambiguous: use four-ring/exact-sparse CNT execution,
tensor-factor/q-FFT Si execution, certified spatial range, and the existing
uniform frequency FFT.  Do not enable the current adaptive FCT as a solver
backend merely because its scalar linewidth asymptotics are favorable.
