# A conserving auxiliary-state treatment of long spatial and spectral tails

**Status:** derivation and private reference prototype, 2026-08-28  
**Production changes:** none  
**Prototype:**
[`_conserving_spatial_tail_review.py`](../studies/_conserving_spatial_tail_review.py)

## 1. Decision

Both the spatial and frequency reductions can be used in one SCBA solver.  The
promising construction is not a fitted one-sided continuation and not an
independent compression of the completed self-energy.  It is a reduced
representation of the entire SCBA functional:

* retain the first spatial shells exactly, including all pieces gained by the
  successful CNT 8 x 2 calculation;
* represent only the remaining two-sided finite-device tail by sequentially
  semiseparable auxiliary states;
* evaluate the far-field bubble directly on those states using Kronecker
  transition matrices;
* use smooth frequency cells plus passive rational pole clusters, with narrow
  output combinations kept as rational states;
* obtain the reduced self-energy as the adjoint derivative of one projected
  Phi functional and hold the basis fixed during each converged SCBA epoch.

This is systematically convergent.  Increasing the exact band, spatial state
rank, cell order and promoted rational rank approaches the dense discretised
SCBA.  It is not guaranteed to remain cheap: the spatial state rank can grow
to the full quasiseparable rank, and the uncompressed pole-pair rank grows as
the product of input ranks.  Those measured ranks, rather than the existence
of the representation, decide whether it beats reblocking.

For the current CNT the first implementation priority remains the exact
atom-sparse FC3 ring on the conserving 8 x 2 reference.  Its measured
four-ring projection is 3.60 s per iteration.  A new tail solver must beat
that result, not the original 12.90 s dense merged-ring run.

## 2. What the successful reblock requires

The 16 x 1 and 8 x 2 CNT calculations do not differ only in their linear
solver.  With `g_band=3`, reblocking changes the primitive support as follows:

| representation | selected Green blocks | produced self-energy blocks |
|---|---|---|
| 16 x 1 | every primitive distance 0--3 | distance 0--1 |
| 8 x 2 | distance 0--6 and part of 7 | distance 0--2 and part of 3 |

The added self-energy blocks have 24.8 % relative Frobenius weight and their
individual 36 x 36 matrices are nearly full rank.  They are exact near field.
No long-range compression should be asked to reproduce them from a low-rank
fit.  The auxiliary construction must retain at least this physical support
exactly and start beyond it.

The potential saving is beyond that support.  A primitive 16 x 1 solver can
carry the same exact near blocks and describe the remaining finite-device
response through auxiliary recurrences.  This avoids merging every primitive
FC3 block into a dense 72-DOF tensor.  It also avoids enumerating every far
Green-function and self-energy block if the bubble acts on the recurrences
directly.

## 3. Bidirectional spatial state algebra

For cells `i > j`, write a finite-device Green function or self-energy tail as

\[
 M_{ij}=C_i^+ A_{i-1}^+\cdots A_{j+1}^+B_j^+ .
\]

The opposite triangle has an independent recurrence

\[
 M_{ij}=C_i^- A_{i+1}^-\cdots A_{j-1}^-B_j^- ,\qquad i<j.
\]

The generators are cell dependent.  Reflections from both contacts and local
inhomogeneity are therefore included.  The failed continuation used one
homogeneous outgoing branch and cannot represent this class.

The state equations for applying the lower tail are

\[
 s_{i+1}=A_i^+s_i+B_i^+x_i,\qquad
 y_i\mathrel{+}=C_i^+s_i,
\]

with the mirrored backward recurrence for the upper tail.  They require
`O(N(r^2+rd))` work and do not materialise any far block.  This is the
time-varying-system form of a sequentially semiseparable matrix.  Linear-time
SSS solvers and model reduction are established for integral equations,
scattering and other wave problems
([Chandrasekaran et al.](https://doi.org/10.1137/S0895479802405884)).

### 3.1 Closure of the SCBA bubble

Outside the exact FC3 support, the direction of both internal Green-function
legs is fixed by the output-cell ordering.  For one internal frequency and one
pair of local FC3 offsets,

\[
 G^{x}_{K_1K'_1}=C_1 A_1^{R}B_1,
 \qquad
 G^{x}_{K_2K'_2}=C_2 A_2^{R}B_2 .
\]

Their product has transition

\[
 A_\Sigma=A_1\otimes A_2 .
\]

The left FC3 block contracts `C1 tensor C2` into `C_Sigma`; the right FC3
block contracts `B1 tensor B2` into `B_Sigma`.  Local FC3 offsets change these
endpoint maps and add a finite number of transition steps.  They do not change
the far-field closure.  Every q, offset and frequency term is accumulated as
a direct sum and then recompressed.

The prototype checks the scalar identity over both triangles.  Ranks 2 and 3
produce rank 6, and the represented bubble agrees with the dense entrywise
product to `1.75e-16`.  Thus the analytical spatial closure is exact before
the explicit recompression.

The raw rank is multiplicative and then additive across the frequency, q and
FC3 sums.  A production implementation needs streaming SSS model reduction,
not a dense conversion.  Balanced truncation or the QR/SVD compression used
by SSS solvers is the appropriate operation.  Failure of that recompression
to keep `r/d` small is a no-go result for this route.

### 3.2 Obtaining Keldysh Green functions without forming them

The bubble needs `G^<` and `G^>`, not only the retarded inverse.  They satisfy

\[
 G^x=G^R\Sigma^x_{\rm tot}G^A,\qquad x\in\{<,>\}.
\]

SSS matrices are closed under the solve and multiplication operations needed
here, with rank growth followed by model reduction.  Starting from the exact
banded harmonic/contact operator and an SSS scattering self-energy, an SSS/ULV
Dyson solve therefore supplies retarded generators rather than a dense
inverse.  Applying the same algebra to the Keldysh product supplies the two
Keldysh generator sets used by the next bubble.

For a local positive source this can be written more explicitly.  In a
homogeneous directional sector the accumulated source covariance obeys a
discrete Lyapunov recurrence,

\[
 Q_{i+1}=A_iQ_iA_i^\dagger+B_iS_iB_i^\dagger,
 \qquad S_i\succeq0,
\]

with mirrored recurrences for sources entering from the right.  Banded and
SSS scattering sources add finite cross terms and source states.  The
recurrence produces the endpoint factors of `G^x` and keeps positivity
structural before reduction.  It also includes both contacts, unlike a fit
continued from one interior anchor.

This closes the proposed SCBA loop entirely in the exact-band plus auxiliary-
state class:

```text
Sigma SSS states -> SSS/ULV retarded solve -> Keldysh source recurrences
       ^                                                |
       |                                                v
       `------ Kronecker-state FC3 bubble <------ G<,> SSS states
```

The loop is exact at the retained state dimensions.  Its unresolved question
is rank after each solve, Keldysh product and frequency-integrated bubble.

## 4. Sparse extended Dyson system

Let `B` contain the exact spatial band, harmonic operator and contact terms.
The unknowns at cell `i` are the physical vector `x_i` and the two directional
states.  The extended equations are

\[
 Bx+C^+s+C^-t=b,
\]

\[
 s_i-A_{i-1}^+s_{i-1}-B_{i-1}^+x_{i-1}=0,
 \qquad s_0=0,
\]

\[
 t_i-A_{i+1}^-t_{i+1}-B_{i+1}^-x_{i+1}=0,
 \qquad t_{N-1}=0.
\]

Eliminating the states gives exactly the band-plus-SSS Dyson matrix.  Kept as
written, the system has only local recurrence couplings.  The earlier cost
estimate `N(d+2r)^3` treated every local auxiliary block as dense and is an
upper bound, not the structure of this matrix.  The relevant solver literature
uses SSS or HSS ULV factorisations
([SSS algorithms](https://doi.org/10.1137/S0895479802405884),
[HSS ULV](https://doi.org/10.1137/S0895479803436652)) and sparse extensions
([Chandrasekaran et al.](https://doi.org/10.1137/050639028)).

The private SciPy implementation is a correctness oracle, not an efficient
ULV solver.  It gives the following results:

| case | `N,d` | directional rank | operator / solve error | storage / dense | extended nnz / dense |
|---|---:|---:|---:|---:|---:|
| extracted planted tail | 24,2 | 8,8 | `6.89e-16` / `1.81e-15` | 2.163 | 1.865 |
| analytic planted tail | 64,8 | 4,4 | exact near / `2.92e-16` | **0.101** | **0.0813** |
| frozen chain | 12,1 | 6,6 | `4.14e-17` / `6.58e-16` | 8.319 | 4.153 |
| frozen multi-mode | 20,2 | 14,14 | `1.53e-5` / `1.19e-15` | 6.495 | 3.873 |
| frozen multi-mode | 16,4 | 22,22 | `6.90e-6` / `1.64e-15` | 5.398 | 3.093 |

The two planted rows distinguish two problems.  Reconstructing a tail from a
dense matrix after zeroing its near band incurs an interface-rank penalty.
Direct analytical bubble generators do not: their low state dimension is
carried into the next SCBA step and the exact near band locally replaces their
near contribution.  This is the regime that can become cheaper at larger
cell DOF.  The real frozen rows show that obtaining such generators by a
post-hoc dense fit is not competitive.

The generic SciPy sparse solve remains 2.3 times slower than dense inversion
even in the 64 x 8 planted case, despite only 8.13 % as many structural
nonzeros.  A specialised SSS/ULV factorisation is mandatory.  Sparse storage
alone is not a speed result.

The previously missing `d=8,N=16` Arm S bed was also completed on Alps.  It
closes the post-hoc fitting question:

| arm | nominal width | current error vs full Sigma | lead imbalance |
|---|---:|---:|---:|
| 2-cell reblock | 16 | **`1.31e-4`** | **`4.70e-7`** |
| S2 | 12 | `2.19e-3` | `6.62e-3` |
| S4 | 16 | `2.29e-3` | `3.13e-3` |
| S6 | 20 | `4.11e-3` | `3.24e-4` |
| S8 | 24 | `4.31e-3` | `6.06e-5` |

The corrected bed converged in 98 iterations at `eta=0`; the generic
high-coupling first attempt diverged and was stopped.  At matched current
accuracy the 2-cell reblock wins.  The physical full-output correction on this
toy is itself below its pre-registered numerical floor, so it is not a bed on
which a costly tail should be enabled.  These results reject fitting a global
SSS representation after forming dense Sigma.  They do not test direct
Kronecker-state generation inside the bubble, which avoids the near-band
interface rank and is the proposed next experiment.

An ideal per-cell storage count with equal directional rank `r` is

\[
 (2R+1)d^2+2r^2+4rd,
\]

apart from boundary and nesting terms.  The two-cell RGF reference has a
leading arithmetic factor of four relative to primitive-cell RGF.  A useful
tail should therefore have bounded `r/d`, preferably below about one half,
and an actual ULV factorisation must beat the measured reblock, including
selected Green-block recovery.  A norm rank alone cannot decide that gate.

## 5. Combined frequency representation

The spatial generators still depend on internal frequency.  Sampling one
generator set for every point of a fine uniform grid would only move the
frequency cost.  Use the enriched frequency basis from the companion review:

\[
 G(\omega)=G_{\mathrm{cell}}(\omega)
 +U(\omega I-K)^{-1}Q(\omega I-K)^{-H}U^\dagger .
\]

The combined bubble has four sectors:

1. smooth cell--cell coefficients use the existing FFT convolution;
2. cluster--cell and cell--cluster terms use singular product-integration
   weights as FFT Toeplitz kernels;
3. cluster--cluster temporal states close under a Kronecker-sum generator;
4. the spatial part of every sector closes under the Kronecker-product
   transition above.

Narrow combination-frequency outputs remain auxiliary rational states.  They
are not sampled back onto the coarse grid.  The result is a two-axis state
network: Kronecker sums in temporal state space and Kronecker products in
spatial state space.  This is closely related to pseudomode/auxiliary-
Hamiltonian methods in open quantum systems and to the state-space/MPO
correspondence used for long-range interactions
([Fröwis et al.](https://arxiv.org/abs/1909.06341)).

The smooth sector needs a shared spatial basis over a frequency window so the
Hilbert transform acts on coefficient functions rather than on separately
fitted matrices.  Existing frozen tests found a compact common basis on the
sharp-line proxy but not on a generic dispersive chain.  If the common rank
grows with the number of frequency cells, use adaptive windows and measure the
sum of their ranks.  If that sum grows linearly with the fine-grid size, the
combined method has failed its frequency gate.

## 6. Conservation and positivity

A post-hoc approximation `Sigma -> Sigma_tilde` is generally not conserving,
even if its matrix error and anti-Hermiticity error are small.  The strict
route is to approximate the generating functional.

Let `R` reconstruct the fixed enriched spatial-frequency basis and let
`R*_W` be its adjoint under the same quadrature weights used by the bubble.
Define `P=R R*_W` and

\[
 \widetilde\Phi[G]=\Phi[P G].
\]

Then

\[
 \widetilde\Sigma[G]
 =\frac{\delta\widetilde\Phi}{\delta G}
 =P^*_W\Sigma_{\rm SCBA}[P G].
\]

This is Phi derivable for the projected model if the projection commutes with
the relevant time/Keldysh symmetries and the same cubic potential is used at
both vertices.  Phi derivability is the established sufficient condition for
self-consistent conservation
([Baym](https://doi.org/10.1103/PhysRev.127.1391)).  The basis must be fixed
during an inner SCBA convergence.  Updating it while omitting its derivative
adds an unaccounted term.  Basis adaptation can occur between fully converged
epochs, followed by reconvergence.

Three lower-level invariants remain useful:

* construct upper and lower SSS generators as adjoints.  The prototype keeps
  Keldysh anti-Hermiticity to `4.7e-16` or better on the frozen cases;
* enrich a compressed basis by the discretised energy-collision covector.  It
  costs at most one state and reduces the planted moment defect from 0.642 to
  `3.47e-16`;
* build `-i Sigma^<` and `+i Sigma^>` from compressed half-diagram factors
  `Z Z^dagger`.  Congruence retains positivity to roundoff.  Positive half
  diagrams are also the route used to reconcile analyticity and positivity in
  many-body approximations
  ([Karlsson--van Leeuwen](https://doi.org/10.1103/PhysRevB.94.125124)).

Moment enrichment alone proves only the chosen frozen collision moment, not a
closed nonlinear SCBA.  Congruence alone proves positivity, not conservation.
The projected functional is the conservation mechanism; the two structural
checks constrain its implementation.  The retarded self-energy must be built
from the same compressed `Sigma^>-Sigma^<`, with smooth Hilbert coefficients
and passive rational states.  Independent retarded fitting is excluded.

## 7. Implementation sequence and decision gates

1. Keep the converged CNT 8 x 2 result as the physics oracle.  Implement the
   exact four-ring identity and exact atom-sparse FC3 actions first.
2. Add an SSS/ULV solver for an exact primitive near band plus supplied
   bidirectional generators.  Verify selected Green blocks and contact current
   against a dense solve; do not obtain the generators by fitting `G` from one
   contact.
3. On a frozen state, generate the far `Sigma^{<,>}` directly from Green SSS
   generators with Kronecker transitions.  Stream direct sums through balanced
   reduction and compare every shell with the dense bubble.
4. Introduce the cell plus rational frequency basis and keep output sum poles
   auxiliary.  Use weighted-adjoint projection at every sector boundary.
5. Run a closed fixed-basis SCBA.  Require current spread no worse than the
   8 x 2 reference, collision-energy defect within ten times dense, Keldysh
   structure below `1e-10`, and selected current error below `1e-3`.
6. Measure end-to-end time against the 3.60 s projected exact-sparse/four-ring
   reblock.  Include basis construction, recompression, q/MPI distribution,
   selected inversion and auxiliary fill.

The route is rejected for CNT if any of the following holds at matched current
accuracy:

* the recompressed directional state rank is not clearly below the primitive
  cell DOF;
* summed spatial-frequency rank grows linearly with frequency count or device
  length;
* a specialised ULV solve does not beat exact reblocking;
* the fixed-basis projected SCBA loses conservation or positivity;
* basis rebuilds are needed so frequently that reconvergence dominates.

If the rank gates fail, retain exact reblocking/adaptive finite range and the
exact atom-sparse FC3 kernel.  That fallback is already accurate and has a
measured cost path.  Failure of low-rank compression would not imply that the
spatial and spectral mathematics is invalid; it would mean that the physical
state count of this finite device is not smaller than the explicit one.

## 8. Reproduction

```bash
QTX_ARRAY_MODULE=numpy PYTHONPATH=src:phonon \
  python phonon/studies/_conserving_spatial_tail_review.py

QTX_ARRAY_MODULE=numpy PYTHONPATH=src:phonon \
  python phonon/studies/_conserving_spatial_tail_review.py \
  --bed cluster/srk-d2/multi2_L20.npz --out /tmp/sss-d2.json

QTX_ARRAY_MODULE=numpy PYTHONPATH=src:phonon \
  python phonon/studies/_conserving_spatial_tail_review.py \
  --bed cluster/arm-dof/d4_bed.npz --out /tmp/sss-d4.json

QTX_ARRAY_MODULE=numpy PYTHONPATH=src:phonon \
  python -m pytest -q \
  tests/quatrex/phonon/test_conserving_spatial_tail_review.py
```

The focused module has seven passing tests.  They cover exact sparse Schur
elimination, exact near-band replacement, adjoint Keldysh generators, exact
Kronecker bubble closure, collision-moment enrichment, positive congruence and
the direct-generator crossover case.

The `d=8` device/rank run is reproduced by the pulled Alps artifacts
`cluster/spatial-d8-l16b/multi8_L16.npz` and
`cluster/spatial-d8-l16b/multi8_L16_tails.npz`; its log is
`cluster/spatial-d8-l16b/slurm-4552671.out`.
