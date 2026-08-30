# Phonon--phonon NEGF without a huge frequency grid or a dense spatial tail

**Status:** code, reduced numerical review and Si production certification,
2026-08-29

**Production changes:** private passive auxiliary RGF/Woodbury hooks, exact
FC3-factor reblocking and an opt-in primitive-microblock contraction.  The two
microblock configuration fields are disabled by default, so legacy solver
behaviour is unchanged.

**Reference studies:**
[`_hybrid_frequency_review.py`](../studies/_hybrid_frequency_review.py) and
[`_spatial_hierarchy_review.py`](../studies/_spatial_hierarchy_review.py), plus
the independent Si retarded-assembly audit
[`_si_kk_audit.py`](../studies/_si_kk_audit.py), plus
the real-Si/production follow-up
[`auxiliary_scba_production.md`](auxiliary_scba_production.md), plus
the conserving auxiliary-state follow-up
[`conserving_long_range_tail.md`](conserving_long_range_tail.md), plus
the measured CNT follow-up
[`cnt_reblock_acceleration.md`](cnt_reblock_acceleration.md) and its GH200
kernel benchmark
[`_cnt_sparse_ring_gpu_bench.py`](../studies/_cnt_sparse_ring_gpu_bench.py),
and the cross-structure mixed-basis synthesis
[`mixed_representation_strategy.md`](mixed_representation_strategy.md), plus
the decisive dual-grid follow-up
[`nonuniform_grid_review.md`](nonuniform_grid_review.md), and the exact
adaptive-convolution/real-Si follow-up
[`adaptive_collision_followup.md`](adaptive_collision_followup.md)
and the support-complete Si certification
[`si_film_microblock_campaign.md`](si_film_microblock_campaign.md).

## 1. Executive verdict

There is a credible way to remove the inverse-linewidth frequency-grid cost,
but it is not the additive pole sector that was tried.  The frequency variable
has to be discretised once in an **enriched basis**: smooth frequency cells plus
coherent passive rational clusters.  All four bubble sectors then belong to one
bilinear discretisation, and a narrow output line remains rational instead of
being sampled back onto the coarse grid.

The separate nonuniform-grid follow-up shows that a genuinely nonuniform P1
collision basis can resolve the reduced analytic bubble with logarithmic point
growth, but the direct reference contraction is quadratic.  The existing
file-grid bridge is only a **conditional primary-solve/memory compression**:
its unchanged uniform auxiliary bubble still grows as the inverse linewidth,
and two CNT grids fail converged conservation even though the weighted-adjoint
transfer identity is exact.  Mild nonuniformity passes on long Si but saves
only 2.82%.  The two mechanisms are therefore complementary: conservative
adaptive cells decide where Dyson and smooth collision sectors are sampled;
passive rational clusters remove subcell lines from the uniform convolution.

The exact fast nonuniform arm has now also been implemented.  Hackbusch's
dyadic P1 projected convolution, with Quatrex's FC3 ring as its bilinear
product, passes independent A/B/C projection, noncommutative ordering,
zeroth/first collision moments and Keldysh symmetry.  On the two-line proxy it
keeps subcell bubble error below `1.25e-3` while the equivalent uniform grid
grows to 2.1 million cells.  A real 15,001-point, 81-q Si oracle nevertheless
rejects a shared-grid production port: a reliable 373-cell mesh still causes
about 4.03 million transformed modes because moving q-resolved resonances
fragment the multilevel support; a blind five-point detector can miss a line
and gives 9.1% error on a typical channel.  The remaining spectral path is a
pole-informed, block-sparse joint q-frequency auxiliary basis, not one global
nonuniform grid.

The reduced study passes the quadrature gates:

* a coherent two-mode rational bubble agrees with independent real-axis
  quadrature to `4.5e-16`;
* the mixed cluster--cell error remains below `3.671e-4` from
  `gamma/h = 1` down to `0.001`, including every requested pole offset,
  cluster separation and 0/2 %/THz source slope;
* the proxy spectral matrices stay positive and the Keldysh wrapper stays
  anti-Hermitian to `2.27e-16`;
* the mixed algorithm takes 1.81--1.93 ms on the fixed 145-point test,
  independent of `gamma/h`.

This is a **go for selective device-level research**, not yet a go for a public
self-consistent production mode.  The private auxiliary Dyson algebra is now
implemented.  A production gate still needs FC3-important cluster selection,
a positive multi-anchor source model, a fixed-basis SCBA mixer, and a measured
conserving fixed point.  Those are precisely the pieces the reduced algebra
does not pretend to validate.

That production/Si gate has now been run.  The exact passive bubble closure,
local augmented RGF and global Woodbury solve are implemented and agree with
dense oracles.  The complete real-Si pole census is nevertheless a no-go for
wholesale auxiliary promotion.  Real-axis projected sources are passive to
about `1e-8`, but one constant PSD source has median rational-leg errors of
10.5 % (L3) and 15.9 % (L8).  At 0.1 % integrated accuracy the FC3-folded
physical output rank saturates all 18 L3 directions over the full q mesh and
all 48 L8 directions at the worst input-rank q.  Conservative Woodbury costs
are 9.5 and 44.5 times a two-cell reblock.  Auxiliary frequency enrichment
therefore remains credible only for a small FC3/current-important subset of
sub-cell clusters after positive source fitting and sum-pole reduction.

For the present converged CNT this is a future **coarsening** lever, not the
immediate bottleneck: its 144-mode census has a narrowest linewidth of 1.57
grid spacings and a median of 6.4.  There is no measured sub-bin CNT population
on the current 161-point grid.  The enriched frequency basis should therefore
be validated first on the recorded narrow Si clusters, then brought back to
CNT only with an explicit coarse-grid A/B.  The exact sparse spatial/FC3 path
below is the near-term CNT speed result.

Spatially, the answer is more conservative.  Direct hierarchical compression
of `Sigma` is accurate and avoids the invalid one-sided propagation assumption:
the largest frozen-bed current error at tolerance `1e-3` is `4.283e-5`.
However, the present HODLR reference stores 1.8--2.0 times a dense matrix on the
three available 1-DOF beds.  It becomes smaller than dense only in the larger
synthetic cases (`0.986` at `N=12,d=8`, `0.733` at `N=20,d=8`).  It has not yet
beaten reblocking at matched production cost.

The real L8 Si snapshot also exposes why post-hoc spatial fitting must not be
the first production test: every primitive shell beyond distance one is
exactly zero at broad, off-resonant and narrow-pole samples.  That is the
current SSE output pin, not a low-rank far field.  An exact lift of the rank-128
primitive FC3 factors into two-cell blocks (rank 256, 6.8 MB) now makes a true
L8x2 reference possible without a GB-scale q-fold.  The converged production
run reaches a `9.511e-4` retarded residual, lowers internal current spread from
`4.371e-3` to `2.829e-3`, and lowers lead imbalance from `7.053e-5` to
`5.850e-6` at a `1.06e-4` relative current change.  Its steady iteration is
6.7 % faster and its GPU mempool peak 44 % smaller than the matched primitive
control.  It is a better reference but still fails the `1e-3` internal-current
gate.

The newly generated primitive distance-2/3 shells carry 1.8--3.5 % combined
discarded norm.  They are high-rank near field: at `1e-3`, global residual rank
is 41--46 of 48 and HODLR uses 0.958--1.092 times exact-reblock storage, losing
at the off-resonant and narrow-pole samples.  Post-hoc hierarchy is therefore
a no-go on real L8 Si; distances four/five must first be generated exactly,
then only the residual beyond that certified band may be tested.  Full values,
the L16 planning projection, and job provenance are reported in
[`auxiliary_scba_production.md`](auxiliary_scba_production.md).

A subsequent algebraic and numerical follow-up identifies a narrower path that
is more promising than post-hoc HODLR.  A bidirectional sequentially
semiseparable tail has an exact sparse state-space extension, and the far SCBA
bubble closes under Kronecker-product transitions.  The earlier
`N(d+2r)^3` augmented-block estimate discards that sparsity.  The new prototype
reproduces its represented Dyson solve to `1.8e-15` or better, its scalar
bubble closure to `1.75e-16`, preserves a collision moment to `3.47e-16` by
one-vector basis enrichment, and preserves Keldysh anti-Hermiticity to
roundoff.  A planted `N=64,d=8,r=4` analytical tail uses 10.1 % of dense
storage and its extended system 8.13 % of dense nonzeros.  Generic SciPy sparse
LU is still 2.3 times slower than dense on that proxy, while post-hoc fits to
the frozen `d=1,2,4` beds still store more than dense.  This is therefore a go
for a direct-generator SSS/ULV experiment, not evidence that a dense fitted
tail is ready.

The previously missing `d=8,N=16` frozen bed was then completed.  It rejects
the old post-hoc global Arm S crossover: a 2-cell reblock has current error
`1.31e-4` and lead imbalance `4.70e-7`; S2 is narrower but gives `2.19e-3` and
`6.62e-3`, while S4 already matches the reblock width and gives `2.29e-3` and
`3.13e-3`.  S6/S8 exceed the reblock width.  The direct full-output tail on
this toy changes current by only `8.12e-4`, below its `9.31e-4` numerical
floor.  Thus ordinary fitted SSS is a no-go, while the direct-generator
functional route remains open and must be tested on CNT-relevant physics.

The subsequent CNT 16-cell campaign strengthens that decision.  Exact
two-cell reblocking reduces the interior current spread from 6.27 % to 0.050 %
and changes the current by 11 %, while costing 12.90 rather than 4.39 s per
iteration.  The added primitive distance-2/partial-distance-3 Sigma pieces
carry 24.8 % relative Frobenius weight and need median rank 63/70 at 1/0.1 %
error.  They are exact near field, not an HSS tail.  The cost penalty is instead
the merged dense FC3 ring (11.93 of 12.90 s).  A new CNT-specific audit also
shows why the old flat FC3 fit is not yet the answer: global INDSCAL errors are
67, 38, 29, 20, 16, 13, 11 and 8.3 % at ranks 16--256.  The exact vertex is,
however, strongly structured: only 228/1728 onsite and 76/1728 in each directed
cross-cell atom-triplet block are nonzero.  The focused follow-up therefore
prioritises exact four-ring Keldysh reuse, an exact primitive/atomic sparse
ring, and an orbit-structured retry of the tensor-factorised kernel.  It does
not assume that one global CP/INDSCAL fit will pass.  A restart-rich Tortin
retry improves global INDSCAL to 22.2, 13.2 and 4.52 % error at ranks 64, 128
and 256; a support-block Tucker diagnostic is still at 9.99 % at rank 32 and
breaks the global ASR by 12.0 %.  In contrast, the exact indexed atom-triplet
format stores only 13.56 % of the onsite block and 4.52 % of either cross
block, including index overhead.  The strongest current CNT result is thus an
**exact sparse vertex**, not an approximate low-rank vertex.

The task-weighted arithmetic model makes that path concrete.  Sparsifying the
two FC3 actions while retaining the final dense `T @ U` contraction uses
37.13 % of the primitive dense-ring MACs.  At equal useful-MAC throughput the
conserving 8 x 2 iteration projects to 3.78 s, or 2.85 s with four-ring reuse,
versus 12.90 s measured.  The corresponding fused CuPy/CUDA prototype was then
run on an Alps GH200 using the real 36-DOF CNT vertices, 161 complex128 Green
matrices and the unchanged final dense contraction.  It is exact to the
reported floating-point precision and speeds individual rings by 1.60 x for
onsite--onsite, 1.68--1.89 x for mixed layouts, and 1.86--2.06 x for the four
cross-layout pairs.  Weighting all nine 30-row/52-row layout combinations by
the exact slow-rank task counts gives a 0.5207 time ratio, or 1.92 x ring
speedup.  Applied conservatively to the archived timing, this predicts 4.92 s
per conserving iteration with six rings and **3.60 s with four rings**.  This
is now a go for a frozen production-contraction implementation; MPI/task-launch
and full-SCBA parity remain unmeasured.

The verdict is therefore:

1. **Now:** retain the conserving 8 x 2 spatial support.  The known CNT
   `g_band=1` path is a no-go and is not a recommendation.  First enable the
   exact four-ring identity, then keep the grouped Dyson layout while evaluating
   the bubble on primitive/atomic sparse blocks.  A same-checkpoint,
   ten-iteration Daint A/B measures 11.9631 -> 7.9904 s for the ring and
   12.2132 -> 8.2379 s for the steady iteration (1.50 x / 1.48 x), while the
   final lead current agrees to `1.08e-13` relatively and the
   current-conservation spread is identical.  This is an exact production
   speedup, not an approximate SCBA path.
2. **FC3 execution:** first implement the now-measured exact atom-sparse
   primitive vertex actions.  If further reduction is needed, factor the
   primitive cluster/orbit representation, preserve full S3 and the global ASR
   by construction, and lift it exactly into the two-cell layout.  Fit every
   candidate rank independently; do not fit the merged 72-DOF tensor or slice
   one high-rank CP file and call that a rank sweep.
3. **Next frequency research:** combine pole-informed q-local dyadic patches
   with the implemented enriched rational bubble/Dyson algebra for only a
   small FC3/current-weighted cluster subset.  Store patches as a block-sparse
   joint q-frequency hierarchy, add a positive multi-anchor source realization
   and use a fixed-basis SCBA mixer; do not take the union as one shared Si grid
   and do not promote the complete detector output.
4. **Next spatial research:** generate a bidirectional SSS self-energy directly
   from Green-function states, retain the near band exactly, and solve its
   sparse extension with a specialised SSS/ULV factorisation.  HSS remains the
   fallback if the single-level state rank grows.  Do not productionise the
   current dense-fit or generic-SciPy prototype.
5. **Do not revive:** one-sided modal continuation, global additive Keldysh
   pole subtraction, a global shared Krylov space, or a fixed hard taper.

## 2. What the code actually computes

The production and dense implementations use the same SCBA diagram, but their
stored objects and truncations are different.

| Stage | Production `src/quatrex/phonon` | Dense `phonon/solver` | Consequence |
|---|---|---|---|
| Green function | RGF selected inversion returns the configured `sse_g_band` | dense inverse, every block | production never possesses a full spatial `G` |
| Primary frequency grid | may be nonuniform | symmetric uniform grid | a nonuniform production grid is still bridged through a uniform FFT grid |
| Auxiliary grid | interpolate primary to uniform auxiliary grid; restrict with the weighted adjoint | not needed | helps output sampling and memory placement, but does not remove the fine uniform convolution grid |
| Bubble | q-resolved FFT in the conjugate time variable; dense or tensor-factorised FC3 contraction | the same FFT convolution with explicit dense blocks | the space--time idea used by GW is already present |
| Pole path | subtracts a pole leg before the bridge, adds mixed terms before restriction, pole--pole terms after restriction, and a separate retarded pole term | local product-integration tools are reference-only | three quadratures/representations are combined and are not automatically one discrete Phi functional |
| Retarded part | `Delta=Sigma^>-Sigma^<`, then half or Hilbert reconstruction; analytic pole retarded term may be added separately | audited FFT/PV reconstruction | independent compression of `Sigma^R` is unnecessary and risks causality |
| Spatial output | generated pair index is restricted to the production output pattern | `sigma_cutoff=None` produces all pairs | the output pin can discard terms produced by already-retained `G` blocks |

For an FC3 vertex of reach `p` and Green blocks retained through distance `b`,
the exact support implication is

\[
       |I-J| \le 2p+b.
\]

With the studied production values `p=1`, `b=3`, the bubble can generate
self-energy blocks out to five cells while the usual output pattern retains
only the first off-diagonal.  This is why extending only `G` does not repair the
dominant spatial error.  The dense four-arm experiment in
[`spatial_representation.md`](spatial_representation.md) already separates
`g_cutoff` from `sigma_cutoff`; its result is retained here rather than
reinterpreting the old ChatGPT conversation as evidence.

The existing tensor factorisation must remain in every cost comparison.  On
the documented Si film it is 91x faster than the dense ring at rank 32; on the
Gamma nanowire shapes it is 557--3553x faster, at near-roundoff parity.  The
frequency and spatial proposals below change the representation, not this FC3
contraction win.

## 3. Why the earlier frequency route failed

The failure is not that residues or analytic convolutions are unavailable.
The repository has verified QNM continuation, coherent pole clusters,
partial-fraction Keldysh legs, pole--pole residues, mixed finite-cell moments,
and local exact replacement.  The structural problems are:

| Failure | Evidence in the repository | Implication |
|---|---|---|
| Additive Keldysh remainder is nonphysical | subtracting a globally fitted pole contribution can make the remainder indefinite | positivity of the sum does not make each separately discretised sector physical |
| Same cluster represented in incompatible forms | congruence is PSD; frozen partial fractions are exact only for a constant projected source | the representation used for subtraction and restoration must be literally the same basis function |
| Different quadratures in one diagram | pole--pole residues, mixed cell moments and FFT background are inserted at different stages | continuum diagram identity alone does not guarantee a discrete conserving approximation |
| Input integration is not output representation | an exact pole-cell pair produces a line at the sum frequency with width `gamma_p+gamma_q` | evaluating it only at coarse output nodes reintroduces the grid error |
| Isolated-pole assumption is not the Si result | the frozen Si census broadens the bulk but leaves roughly 1--4 narrow, increasingly overlapping modes per q | the useful unit is a coherent invariant subspace/cluster, not scalar independent Lorentzians |
| Device form is missing | `pole_local` accepts dense residues and is not wired to projected rank-one device factors | its scalar/local accuracy is real but is not a production cost result |

The local finite-cell work was nevertheless the right clue.  It is a special
case of product integration: retain the difficult rational weight analytically
and approximate only the smooth factor.  The change proposed here is to make
that the definition of the whole enriched discretisation, not a correction
added to a separately defined ring.

## 4. Frequency methods from other fields

| Method | Where it works | Transfer to real-axis phonon SCBA |
|---|---|---|
| Product integration / locally corrected quadrature | singular integral equations; known singular weight times smooth data ([Rabinowitz--Sloan](https://doi.org/10.1137/0721010), [Strain](https://doi.org/10.1137/0916058)) | **Direct.** This is the mathematical class of `pole_local` and the mixed Toeplitz moments below. |
| Multipole self-energies and sum-of-poles inversion | GW and dynamical potentials ([multipole approximation](https://doi.org/10.1103/PhysRevB.104.115157), [algorithmic inversion](https://doi.org/10.1103/PhysRevResearch.4.013242)) | **Direct for the rational output.** Pole sums become a small auxiliary eigenproblem rather than a fine real grid. |
| Auxiliary Hamiltonians / pseudomodes | nonequilibrium memory kernels and open systems ([auxiliary Hamiltonian](https://doi.org/10.1103/PhysRevB.89.035148), [generalised pseudomodes](https://arxiv.org/abs/2002.09739)) | **Direct representation analogy.** A rational frequency memory becomes local dynamics in extra states. |
| Matrix Herglotz / Nevanlinna approximation | causal matrix Green functions ([Nevanlinna continuation](https://doi.org/10.1103/PhysRevLett.126.056402), [matrix-valued PES](https://doi.org/10.1103/PhysRevB.107.075151)) | **Direct constraint.** Fit poles and PSD sources/passive residues, not independent matrix entries. |
| Quantics tensor trains / cross interpolation | compressed two-time many-body diagrams ([Murray--Shinaoka--Werner](https://arxiv.org/abs/2312.03809)) | **Promising fallback, not yet direct.** A logically huge uniform real grid and its convolutions may have low QTT rank, including narrow rational peaks.  Rank growth for many moving Keldysh peaks and conservation under truncation must be measured; it is a useful comparison to explicit rational clusters. |
| DLR/IR and sparse Matsubara sampling | equilibrium imaginary-time/Matsubara Green functions ([DLR](https://arxiv.org/abs/2107.13094)) | **Not direct.** The compression is excellent before analytic continuation; steady-state real-axis lesser/greater functions would require the unstable continuation the method avoids. |
| GW space--time method | imaginary-axis convolution becomes a time-domain product | **Already used in substance.** Quatrex FFTs the real-axis Keldysh bubble; moving it to imaginary time would lose the nonequilibrium distribution. |
| Contour deformation / tetrahedron integration | retarded GW integrals and k/energy singularities | **Conditional.** Useful for a retarded or equilibrium component, but not a replacement for real-axis lesser/greater spectral measures. |
| Baym--Kadanoff / positive half diagrams | conserving and positive many-body approximations ([Baym](https://doi.org/10.1103/PhysRev.127.1391)) | **Mandatory validation principle.** Conservation and positivity are separate gates; neither follows from a small matrix norm. |

## 5. The hybrid frequency prototype

### 5.1 One enriched basis

For one coherent cluster the positive spectral/Keldysh weight is stored as

\[
 A_P(\omega)=U(\omega I-K)^{-1}Q(\omega I-K)^{-H}U^\dagger,
 \qquad Q\succeq0,\quad \operatorname{Im}\lambda(K)<0.
\]

The full leg is `G_h = G_cell + A_P`.  `G_cell` is a coarse cell polynomial;
there is no globally subtracted remainder.  A 2 %/THz scalar source slope is
included exactly: multiplying a `1/omega^2` congruence by an affine function
only evaluates that factor at each partial-fraction pole because the apparent
constant residues sum to zero.

Writing

\[
 A_P(\omega)=\sum_a \frac{C_a}{\omega-z_a}
             +\sum_b \frac{\bar C_b}{\omega-z_b^*},
\]

closes the pole--pole bubble analytically:

\[
 \int\frac{du}{2\pi}
 \frac{C_a}{u-z_a}\frac{C_b}{\Omega-u-z_b}
 =\frac{-iC_aC_b}{\Omega-z_a-z_b}
\]

for two lower-half-plane poles, with the conjugate sign for two upper poles and
zero for mixed half planes.  The output is a `RationalBubble` whose poles are
pairwise sums.  Its width is the sum of input widths, but it is never sampled;
the future Dyson interface consumes the rational object.

### 5.2 Toeplitz product integration for mixed sectors

On a uniform coarse grid, reconstruct a smooth partner in cell `j` as
`B_j+B'_j t`.  Precompute cluster moments for every integer lag:

\[
 W_n^{(0)}=\int_{-h/2}^{h/2} A_P(nh+t)dt,
 \qquad
 W_n^{(1)}=\int_{-h/2}^{h/2} tA_P(nh+t)dt.
\]

They are closed-form logarithmic resolvent moments.  The mixed sector at output
node `m` is

\[
 \sum_j\left[W_{m-j}^{(0)}B_j-W_{m-j}^{(1)}B'_j
             +B_jW_{m-j}^{(0)}-B'_jW_{m-j}^{(1)}\right]/(2\pi).
\]

Every term is Toeplitz in `m-j`; the prototype evaluates the matrix products by
scalar FFT convolutions.  Its work depends on the coarse grid, matrix size and
cluster rank, not on `1/gamma`.

### 5.3 Measured frequency results

This reduced sweep is explicitly not a frozen Si state.  It is a two-mode
coherent PSD cluster with the widths, overlap regime and source variation from
the earlier Si census.  The subsequently recovered real L3/L8 pole states are
tested separately in
[`auxiliary_scba_production.md`](auxiliary_scba_production.md); they reject
wholesale promotion even though this quadrature proxy passes.

| `gamma/h` | maximum mixed relative error | median mixed relative error |
|---:|---:|---:|
| 1 | 1.227e-4 | 1.212e-4 |
| 0.2 | 1.734e-4 | 1.161e-4 |
| 0.04 | 2.754e-4 | 9.550e-5 |
| 0.008 | 3.444e-4 | 9.513e-5 |
| 0.001 | 3.671e-4 | 9.626e-5 |

This is the full 120-case sweep:

* offsets `0`, `0.25h`, `0.49h`;
* separation/(sum of widths) `0.5`, `1`, `2`, `5`;
* source slopes `0`, `0.02 / THz`;
* 145 coarse points at `h=0.25 THz`.

The minimum spectral eigenvalue over the diagnostic samples is positive
(`5.245e-3`) and the largest normalised anti-Hermiticity residual is
`2.266e-16`.  The pure rational sector error is `4.5e-16`.  The mixed runtime
varies only from 1.81 to 1.93 ms over the five widths.

**What is not established:** this scalar-source proxy is not a discrete
device SCBA functional.  It cannot pass the planned energy-conservation gate.
A device implementation must construct projection and reconstruction as an
adjoint pair under the frequency weights, use the same enriched leg in both
internal lines, and measure the pre-mixing energy balance.  Until then the
frequency result is an accuracy/structure pass and a conservation gate left
open.

### 5.4 Retarded-assembly audit on the Si checkpoint

The causal reconstruction was checked separately from the pole-enrichment
proposal.  Four stack-distributed self-energy slices from the converged L5
(s=0.953125) state were concatenated and passed through an independent
NumPy implementation of the exact cell-integrated transform.  The raw
textbook spectral difference is

\[
 \Delta_{\rm raw}=\Sigma^<_{\rm stored}-\Sigma^>_{\rm stored},
 \qquad
 \Sigma^R={1\over2}\Delta_{\rm raw}
       +{i\over2}H[\Delta_{\rm raw}].
\]

After applying the production zero-frequency output mask, the reconstructed
array agrees with the saved retarded array to (4.80\times10^{-16}) in the
maximum relative norm.  The check includes both transverse axes and the
(q\mapsto-q) bosonic mirror.  The configured transverse shift (4/9) is not
an offset from this index convention.  In the repository's Monkhorst--Pack
formula it gives (q_i=i/9), which is the Gamma-centred mesh assumed by the
fold.  The SSE constructor validates this ordered equivalence and refuses a
different shifted mesh.

The largest spectral-difference entry at 40 THz is
(1.262\times10^{-3}) of the global peak.  The 40 THz window is therefore
not grossly truncated for this checkpoint, although an auxiliary bubble grid
to 80 THz remains part of the extent convergence test.  The dispersive and
instantaneous terms are both material: their maximum norms are respectively
0.598 and 0.917 times the maximum norm of the complete retarded array.  The
half rule is consequently not a small perturbation.

The same audit separates an implementation error from a representation
error.  For analytic bosonic pole pairs sampled at (h=0.25) THz, the
cell-constant transform has relative (L^2) errors of 0.991, 0.513 and 0.109
at (gamma/h=0.008,0.4,2), for a pole at 23 THz on the 0--40 THz window.
An exact piecewise-linear hat-function convolution gives 0.990, 0.467 and
0.092.  Extending the last case to 80 THz lowers these errors to 0.074 and
0.043.  Linear interpolation improves a resolved line but cannot recover a
pole much narrower than a cell.  This supports selective rational enrichment
rather than a replacement of the audited KK sign or q fold.

## 6. Why the spatial modal idea failed

At a fixed frequency the inverse of a block-tridiagonal operator is
semiseparable; this is a standard result for block tridiagonal inverses
([Meurant](https://doi.org/10.1137/0613045)).  That fact did not justify the
specific attempted representation:

* a finite two-terminal device contains left- and right-injected components;
  relative to one anchor, the far-contact branch appears to grow;
* discarding it loses 26--30 % of fitted residue weight on the recorded chain;
* continuing it in the wrong direction diverges;
* even a correct per-frequency spatial exponential remains inside the
  frequency convolution, so frequency integration does not leave one spatial
  exponential;
* the shared-Krylov action reached the full device basis on the measured bed;
* direct bidirectional semiseparable `Sigma` is accurate, but its global rank
  gives an augmented block width that loses to reblocking at the measured DOF.

These findings reject one-sided extrapolation, not low-rank operator algebra.
The correct next question is whether recursive off-diagonal blocks have small
**per-level** ranks even when one global corner rank grows with length.

## 7. Spatial methods from other fields

| Method | Transfer |
|---|---|
| HSS/HODLR and sparse extensions ([HSS sparse solver](https://doi.org/10.1137/050639028)) | Directly compress the self-energy or Dyson operator. A nested basis can turn many global far blocks into local tree states. |
| Positivity-preserving HSS ([Xia et al.](https://doi.org/10.1137/17M1137073)) | Relevant to `+i Sigma^< >= 0` in this tree's convention; independent upper/lower SVDs are not enough. |
| Hierarchical nonequilibrium Dyson ([Kaye--Golez](https://arxiv.org/abs/2010.06511)) | Evidence that hierarchical Green-function algebra can reduce nonequilibrium Dyson cost; its time-domain HODLR ranks are not automatically the present spatial ranks. |
| Directional H2 / butterfly ([directional H2](https://doi.org/10.1137/19M124280X)) | Conditional for many propagating phases. Directional partitioning can help analytic/interpolation bases, but multiplying rows/columns by one phase cannot change SVD rank. |
| Matrix-product operators / state-space sums of exponentials ([MPO/control correspondence](https://arxiv.org/abs/1909.06341)) | A useful implementation language: far interactions become local auxiliary recurrences. It is equivalent in spirit to a semiseparable/HSS sparse extension. |

## 8. The HODLR reference and results

The study stores an exact block band and recursively SVD-compresses the
off-diagonal residual.  A low-rank sibling block can spill numerically into the
masked near corner, so the explicit band includes the cancelling correction;
the represented near blocks equal the input to roundoff.

For anti-Hermitian self-energies it compresses `H=+i Sigma` as one Hermitian
object and returns `-i H_h`.  Thus anti-Hermiticity is structural, not repaired
after two independent fits.  A general low-rank hierarchy is converted to a
sparse extended system by the Schur construction shown in the study docstring.

### 8.1 Multi-DOF propagating proxy

At tolerance `1e-3`, the proxy is a finite sum of damped coherent propagating
modes.  It is therefore an optimistic rank-control bed, useful for DOF and
length scaling but not evidence about a real FC3 bubble.

| `N` | `d` | global quasiseparable rank | maximum HODLR rank | stored/dense |
|---:|---:|---:|---:|---:|
| 8 | 1 / 4 / 8 | 2 / 5 / 6 | 3 / 9 / 12 | 1.844 / 1.656 / 1.344 |
| 12 | 1 / 4 / 8 | 2 / 5 / 6 | 3 / 9 / 12 | 1.542 / 1.278 / 0.986 |
| 16 | 1 / 4 / 8 | 2 / 5 / 6 | 3 / 9 / 12 | 1.305 / 1.117 / 0.867 |
| 20 | 1 / 4 / 8 | 2 / 5 / 6 | 3 / 9 / 14 | 1.095 / 0.932 / 0.733 |

Operator and sparse-extended solve errors are at roundoff on this planted
finite-mode kernel.  That is a construction validation, not a realistic error
claim.  The useful result is the crossover: hierarchy overhead only amortises
at larger `N*d`.

A dense PSD congruence control remains full rank and stores exactly one dense
matrix (`stored/dense=1`).  It preserves positivity but removes the apparent
saving.  A production route therefore needs a genuinely positivity-preserving
**nested** construction, not dense eigenvalue clipping after compression.

Simple phase demodulation changes the measured numerical ranks by exactly zero
in all 48 cases.  This is inevitable: block-diagonal unitary left/right scaling
preserves singular values of every corner.  Directional H2 remains potentially
useful only through mode/direction partitioning or special interpolation, not
as a preconditioner for SVD rank.

### 8.2 Frozen 1-DOF beds

Three stored eta-zero chain beds were measured at three representative live
frequencies, three exact near bands and the requested tolerances.

At tolerance `1e-3`:

| Bed | HODLR max operator error | typical HODLR/global rank | typical stored/dense | max current error |
|---|---:|---:|---:|---:|
| `chain_L10` | 1.31e-15 | 5 / 4 | 1.96--2.02 | 9.39e-7 |
| `chain_L12` | 3.41e-4 | 5--6 / 4--5 | 1.88--1.97 | 4.28e-5 |
| `chain16_c2e+16` | 2.89e-4 | 6--8 / 5--6 | 1.81--1.95 | 4.21e-7 |

The current comparison rebuilds `Sigma^R` from the compressed
`Sigma^>-Sigma^<` with the audited bosonic FFT Hilbert transform and then runs
the dense Dyson/Keldysh equations.  The physical constants cancel in the
reported relative current.

The best tested reblock (`m=4`) has current errors `1.41e-2`, `4.77e-2`, and
`1.28e-3` on the same three beds.  HODLR is much more accurate, but its
reference storage is larger than dense here.  Hard bands are non-monotone in
current error (up to order unity on `chain_L12`), demonstrating why a shell norm
alone is not a safe current tolerance.

The frozen spectral carriers are themselves often indefinite.  Symmetric
HODLR adds at most `1.37e-4` absolute negativity on these samples, but an
absolute positivity pass would be meaningless when the reference minimum
eigenvalue reaches `-7.59`.  The correct metric is added negativity relative to
a physically valid reference; a new converged device bed is needed.

**Spatial decision:** accuracy passes; cost and positivity do not.  Do not wire
this HODLR reference into production.  A real nested-basis HSS/ULV test is
justified only at the observed crossover dimensions.

## 9. What a serious finite-CNT FC3 retry should do

The prior finite-device result is real but narrower than it was easy to imply.
It is a d11a Si-nanowire sweep, not a CNT factor run, and it stops at rank 16.
At that rank INDSCAL and Waring both miss the anharmonic conductance by about
15 %, HOSVD/CP by 23--27 %, while mSVD is unphysical.  Those fits minimise the
global FC3 Frobenius norm, and some enforce only the internal-leg S2 symmetry.
They do not test the current production factor kernel on the conserving CNT
8 x 2 fixed point.

[`_cnt_fc3_compression_review.py`](../studies/_cnt_fc3_compression_review.py)
now asks the flat-fit question on the committed CNT(3,3) vertex.  The exact
primitive target is `36 x 108 x 108`, has seven occupied offset pairs, S2 error
zero and contracted-leg ASR residual `1.25e-16`.  Independent global INDSCAL
screens (algebraic initialisation plus ALS, no restart campaign) give:

| rank | 16 | 32 | 64 | 96 | 128 | 160 | 192 | 256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| relative FC3 error | .669 | .376 | .285 | .202 | .165 | .131 | .107 | .0827 |

That is a negative screen for **plain global INDSCAL at a tensor-norm accuracy
target**, not a final optimiser no-go.  The requested restart-rich retry was
run independently on three Tortin nodes (algebraic start plus two random
starts, 250 ALS and 150 L-BFGS iterations):

| rank | 64 | 128 | 256 |
|---:|---:|---:|---:|
| relative FC3 error | .2220 | .1324 | .04520 |
| best pre-L-BFGS restart error | .2221 | .1324 | .04524 |
| wall time on Tortin CPU | 20.0 s | 122.0 s | 230.6 s |

Restarts materially help, but rank 256 still misses a 1 % vertex gate and
already incurs the `R^2 b` Gram cost in the factorised ring.  A frozen-bubble
observable could converge sooner than the vertex norm, so this does not make
the contraction format impossible; it does make a further flat tensor-norm
campaign lower priority than the exact sparse route.

The same audit shows exact locality that the flat tensor hides:

| orbit block | active atom triplets / 1728 | mode ranks for 99 / 99.9 % squared norm |
|---|---:|---:|
| onsite | 228 | 31 / 35 on every leg |
| directed cross `001` | 76 | 26, 26, 18 / 32, 32, 24 |
| directed cross `011` | 76 | 18, 26, 26 / 24, 32, 32 |

Storing the active 3 x 3 x 3 Cartesian tensors plus three 16-bit atom indices
per active triplet costs 13.56 % of the dense onsite block and 4.52 % of each
cross block.  Across the three orbit representatives that is about 7.5 % of
their dense scalar storage, with zero approximation and exact structural
zeros.

An intermediate support-block Tucker test retained the seven offset pairs and
internal-leg S2 exactly, but deliberately did not project away its ASR defect:

| local Tucker rank | 8 | 16 | 24 | 32 | 36 (exact) |
|---:|---:|---:|---:|---:|---:|
| global FC3 error | .602 | .423 | .247 | .0999 | `2.2e-15` |
| contracted-leg ASR residual | .372 | .335 | .172 | .120 | `1.7e-15` |

This rules out a naive independently truncated block Tucker just as clearly as
the flat fit.  It also validates the proposed design constraint: any
approximate orbit factors must enforce the **global** ASR across onsite and
neighbouring-cell terms, where the cancellation actually occurs.

The retry should therefore work on the **cubic-potential/cluster orbits**, not
on the merged 72-DOF tensor and not on Sigma after contraction:

1. Retain the hiPhive/symfc orbit basis (or reconstruct it from FC3) so
   translation, exact support, S3 and ASR are one shared parameterisation.
   hiPhive already constructs symmetry-related cluster orbits
   ([Eriksson--Fransson--Erhart](https://arxiv.org/abs/1811.09267)); expanding
   them to a dense device tensor and then rediscovering that structure is
   avoidable.
2. First implement an **exact atom-sparse/microblock ring**.  It is a bounded
   engineering baseline and will say how much of the 11.93 s ring is recoverable
   without a vertex approximation.
3. Fit the onsite and directed cross-cell orbits with a symmetry-coupled
   block-term/Tucker/PCP form, enforcing global ASR across orbits.  Block-term
   models are designed for sums of higher-rank components
   ([Rontogiannis--Kofidis--Giampouras](https://arxiv.org/abs/2002.09759)); the
   2025 phonon-specific tensor-learning result demonstrates large IFC and
   thermal-conductivity compression in bulk materials
   ([Luo et al.](https://doi.org/10.1103/nmgj-yq1g)), but does not establish
   finite-device NEGF conservation.
4. Try an interpolative/THC form before increasing a flat CP rank indefinitely.
   Exact THC for local finite-range many-body potentials and interpolative
   separable density fitting show how locality can yield separability without a
   single global CP objective
   ([Parrish et al.](https://arxiv.org/abs/1301.5064),
   [Lu--Ying](https://arxiv.org/abs/1410.7757)).
5. Optimise a fixed vertex against a shell-balanced and frozen-bubble metric,
   not current alone and not global Frobenius norm alone.  The same approximate
   cubic potential must be used at both vertices in every diagram.  Fit each
   rank independently; CP ranks are non-nested, so slicing a rank-256 result is
   not a valid rank-32 fit.

The gates are stricter than a tensor residual: exact S3/support, global ASR,
frozen `Sigma^{<,>}` and energy-balance parity, then current within 1 % and
interior spread no worse than `1e-3` on the 8 x 2 reference.  If the exact
atom-sparse kernel plus an orbit factorisation cannot beat the dense primitive
bubble, FC3 compression is a no-go for this CNT and the exact microblocked
fallback remains useful.

## 10. Recommended representation and interfaces

The production solver now exposes only the two bounded spatial controls
`sse_microblock_dof` and `sse_microblock_g_band`.  Both default to zero.  The
frequency and hierarchical representations remain research interfaces.  If
their next gates pass, the production design should expose concepts equivalent
to:

```text
HybridSpectrum
    cell_coefficients[frequency_cell, polynomial_order, stored_pair]
    clusters[q] -> RationalCluster(K, U, Q_lesser, Q_greater)
    reconstruct(grid_or_complex_frequency)
    project_adjoint(weighted_samples)

HybridSelfEnergy
    smooth_grid_lesser/greater
    rational_outputs -> RationalBubble(sum_poles, residues/source factors)
    retarded := causal reconstruction of the same spectral difference

HierarchicalSelfEnergy
    exact_near_band
    nested_far_factors_by_tree_level
    apply / sparse_extended_or_ULV_solve

StateSpaceTail
    exact_near_band
    forward/backward cell-dependent SSS generators
    direct_sum / balanced_reduce / adjoint_pair
    sparse_extended_or_SSS_ULV_solve

VertexOrbitFactors
    exact_support_and_S3_orbits
    globally_ASR_constrained local Tucker/CP/THC factors
    lift_to_solver_blocks_without_refitting
```

The intended data flow is:

```text
selected G / cluster extraction
        |
        v
one enriched frequency leg (cells + passive clusters)
        |
        +-- cell x cell ---------- existing FFT + tensor FC3
        +-- cluster x cell ------- Toeplitz product integration + tensor FC3
        +-- cluster x cluster ---- rational sum-pole output + tensor FC3
        |
        v
smooth Sigma grid + passive rational Sigma states
        |
        v
Dyson solve with frequency auxiliary states
        |
        +-- immediate: reblocked/exact certified spatial band
        `-- conditional: direct SSS states, then nested HSS if needed
```

The pole detector must promote an invariant subspace and its FC3-weighted
importance, not merely a small linewidth.  The output realization should merge
near-degenerate sum poles before its auxiliary dimension grows quadratically.

## 11. Cost model

Let `Nw` be coarse frequency cells, `c` the number of promoted clusters, `r`
their typical modal size, `N` the device cells, `d` cell DOF and `rho_l` the
hierarchical rank at level `l`.

| Component | Leading behaviour | Width dependence |
|---|---|---|
| smooth FFT bubble | existing `O(Nw log Nw)` per spatial/FC3 task | none once smooth |
| exact atom-sparse FC3 ring | exact sparse first/second vertex actions plus the remaining dense internal contraction | ideal 0.3713 MAC ratio; measured task-weighted GH200 time ratio 0.5207; indexed storage 13.56/4.52 % |
| orbit-factor FC3 ring | sum of per-orbit Tucker/CP/THC costs; rank must be measured after exact two-cell lifting | approximation belongs to one fixed cubic potential, not Sigma |
| mixed product integration | `O(c r^2 d^3 Nw log Nw)` in the unoptimised matrix prototype; tensor projection should reduce the `d` contraction | **none in `1/gamma`** |
| rational pole--pole output | `O(c^2 r^2)` pole pairs before merging, with small matrix residue contractions | output linewidth is stored, not sampled |
| exact spatial band | `O(N R d^2)` storage and application | chosen by support/current error, not a universal taper |
| bidirectional SSS tail | `O(N(r^2+rd))` storage/apply before FC3/q/frequency direct sums | useful only after streaming model reduction keeps `r/d` small |
| sparse SSS Dyson extension | about `(2R+1)d^2+2r^2+4rd` structural entries per cell | generic sparse LU is not sufficient; use SSS/ULV |
| diagnostic HODLR | approximately `O(N rho log N)` storage for bounded ranks, but with non-nested duplicated bases | measured overhead dominates at small `N*d` |
| nested HSS/ULV target | `O(N rho)` storage and near-linear/quadratic factorisation depending on implementation | only credible if per-level ranks remain bounded |

This model must be multiplied by the actual q/quad/task distribution and
measured alongside the production tensor decomposition.  It is not acceptable
to quote a compression ratio while ignoring the replicated permutation cache,
auxiliary FFT padding, q axes, or auxiliary sparse-factor fill.

## 12. Ranked implementation path

### Short term: production-safe

1. Make `sigma_cutoff`/output reach consistent with the Green band and FC3
   support law on a small reference, then use reblocking or the smallest
   current-certified block band the BTS solver can accept.  For the measured
   CNT bed, keep the successful 8 x 2, band-3 support; `g_band=1` is already a
   measured no-go.  Follow the exact ladder in
   [`cnt_reblock_acceleration.md`](cnt_reblock_acceleration.md): four-ring reuse,
   primitive/atomic microblocking, then the orbit-factor gate.
2. Extend the factorised SSE path to sums of support/orbit components and
   finish its memory/q-distribution tuning before adding another representation.
3. Extend the Si census to per-pole rows: linewidth, separation, continuation
   residual, projected source coherence, and FC3-weighted importance must be
   joined per mode.  Marginal percentiles cannot decide promotion.
4. Use `pole_local` as a diagnostic oracle, not as an automatically enabled
   production correction.

### Medium term: frequency auxiliary prototype on a real device slice

1. Keep the new production passive-channel/RGF/Woodbury algebra and the real
   q-resolved pole export as the reference implementation.  The complete Si
   pole census fails the rank/cost gate, so promotion must be selective.
2. Define the cell and rational reconstruction/projection pair with one weighted
   inner product and derive the bubble from one discrete Phi functional.
3. Fit a positive local source basis, merge cluster sum poles into a passive
   realization, and mix its fixed-basis coefficients together with the smooth
   grid component.  Stop whenever the output rank approaches the physical DOF.
4. Validate the selected subset against locally adaptive dense quadrature;
   then run a stable closed SCBA bed and require current/heat conservation and
   wall time better than the exact reblock.

### Medium term: spatial decision experiment

1. Generate SSS Sigma states directly from SSS Green states using the exact
   Kronecker transition, rather than fitting the completed dense matrix.  The
   frozen post-hoc `d=1,2,4` fits are already noncompetitive.
2. The converged L8x2 Si gate now rejects post-hoc HODLR for the generated
   distance-2/3 near field.  Generate the support-law distance-4/5 shells with
   a five-cell reblock, a coverage-proven overlapping/variable-band reference,
   or an explicit block-band-five selected solve before asking whether anything
   beyond that exact band is compressible.  A three-cell non-overlapping BTD
   reblock reaches distance five but does not contain the complete band.
3. Measure streaming direct-sum reduction, a specialised SSS/ULV solve and
   selected blocks.  Compare against the exact-sparse, four-ring 8 x 2 target
   at the same current tolerance.
4. If one global state grows, test nested-basis HSS reuse and sparse/ULV fill.
5. Enforce anti-Hermiticity by adjoint generators, positivity by half-diagram
   congruence, and conservation through one fixed projected Phi functional.
6. Stop if measured end-to-end memory and solve time do not beat reblocking.

### Long term: combine after auditable component gates

The spatial and spectral ideas are compatible because they act on different
indices of the same diagram.  Use the exact 8 x 2 or microblocked spatial
layout, one fixed symmetry-preserving vertex, and the enriched frequency basis
inside one discrete Phi functional.  Only after that path reproduces the dense
fixed point should an HSS spatial far field or a QTT frequency fallback be
added.  This ordering is a validation rule, not a claim that combined
compression is mathematically impossible.

## 13. Rejected or deferred paths

| Path | Decision |
|---|---|
| Finer global uniform grid | rejected as the scaling solution; it remains an oracle where affordable |
| Nonuniform primary grid with fine auxiliary FFT grid | useful interface/memory control, but does not remove fine convolution work |
| One-sided Bloch/modal continuation | rejected for finite two-contact devices |
| Independent exponential fit of each `G^<`/`G^>` block | rejected; does not preserve Keldysh positivity or the far-contact branch |
| Global additive pole subtraction/restoration | rejected; remainder can be indefinite and sector quadratures are inconsistent |
| Local correction sampled onto the same coarse output | incomplete; fixes input integration but not output representation |
| Shared real-action Krylov basis | rejected on measured basis saturation and conjugate-linear bosonic mirror complications |
| Fixed spatial boxcar/taper | rejected as a fidelity method; only a controlled baseline with current error certification |
| Plain phase demodulation before SVD | rejected as a rank reducer; unitary scaling leaves singular values unchanged |
| Direct DLR/imaginary-time replacement | deferred to equilibrium/benchmark work; not a real-axis nonequilibrium solution |
| Plain global CNT CP/INDSCAL fit | restart-rich rank 256 still has 4.52 % vertex error; allow a frozen-bubble observable gate only if its measured factor-kernel cost is competitive |
| Independent support-block Tucker | rank 32 has 9.99 % vertex error and 12.0 % global-ASR defect; rejected unless orbit coupling enforces the shared ASR |

## 14. Reproduction and acceptance ledger

Commands:

```bash
PYTHONPATH=src:phonon python phonon/studies/_hybrid_frequency_review.py \
    --full --json /tmp/frequency_review_full.json

PYTHONPATH=src:phonon python phonon/studies/_spatial_hierarchy_review.py \
    --json /tmp/spatial_review.json

PYTHONPATH=src:phonon python phonon/studies/_cnt_reblock_acceleration.py \
    --json /tmp/cnt_reblock_acceleration.json

# CUDA/GH200 study; requires CuPy and the CNT HDF5 input.
PYTHONPATH=src:phonon python phonon/studies/_cnt_sparse_ring_gpu_bench.py \
    --json /tmp/cnt_sparse_ring_gpu.json

PYTHONPATH=src:phonon python phonon/studies/_cnt_fc3_compression_review.py \
    --fit-ranks 16,32,64,96,128,160,192,256 \
    --json /tmp/cnt_fc3_review.json

# Restart-rich finite-CNT check used for the rank-64/128/256 table.
PYTHONPATH=src:phonon python phonon/studies/_cnt_fc3_compression_review.py \
    --fit-ranks 64,128,256 --restarts 2 --max-iter 250 \
    --lbfgs-iters 150 --seed-base 1729 \
    --json /tmp/cnt_fc3_restart_review.json

QTX_ARRAY_MODULE=numpy PYTHONPATH=src:phonon \
    python phonon/studies/_conserving_spatial_tail_review.py

python phonon/studies/_si_kk_audit.py \
    --checkpoint \
    cluster/si-l5-b3-v0953125save-from9375-q9-w40-dw025-t1e4 \
    --ne 161 --wmax 40 --q-shape 9 9 \
    --output phonon/studies/out/si_kk_audit.json

python phonon/studies/_si_kk_audit.py \
    --checkpoint cluster/si-l5-equilibrium-kms-map-q9-aux025f80 \
    --run cluster/si-l5-equilibrium-kms-map-q9-aux025f80/run.npz \
    --ne 161 --wmax 40 --q-shape 9 9 --temperature 300 \
    --current-reference 120.20623282509771 --skip-analytic \
    --output phonon/studies/out/si_equilibrium_invariants.json

python phonon/studies/_si_ballistic_validation.py \
    --run cluster/si-l5-ballistic-q17-w20-dw003125-caroli/run.npz \
    --output phonon/studies/out/si_ballistic_q17_w20_dw003125.json

python phonon/studies/_si_ballistic_validation.py \
    --run cluster/si-l5-ballistic-q13-w20-dw003125-dt5-caroli/run.npz \
    --output phonon/studies/out/si_ballistic_q13_w20_dw003125_dt5.json

OPENBLAS_NUM_THREADS=8 python phonon/studies/_si_ballistic_mode_count.py \
    --matrix cluster/si-l5-q9-r128-in/dynamical_matrix.mat \
    --run cluster/si-l5-ballistic-q9-w20-dw003125-caroli/run.npz \
    --nk 1025,2049,4097 \
    --output phonon/studies/out/si_ballistic_q9_mode_count.json

python phonon/studies/_si_conventional_fcp.py \
    --fc-hdf5 cluster/tortin-si-big-reap/fit/fc3.hdf5 \
    --source-repeat 5 --lattice-constant 5.468 \
    --output cluster/si-big-conventional-100-reap \
    --repeat 3 --orders 2,3 --validate-folding

OPENBLAS_NUM_THREADS=8 python phonon/studies/_si_ballistic_mode_count.py \
    --matrix cluster/si-conventional-100-l5-q13-ballistic-in/dynamical_matrix.mat \
    --q-mesh 17 --area 2.9899024e-19 --temperatures 305,295 \
    --nk 1025,2049,4097 \
    --output phonon/studies/out/si_conventional_100_ballistic_q17_mode_count.json

OPENBLAS_NUM_THREADS=8 python phonon/studies/_si_ballistic_mode_count.py \
    --matrix cluster/si-conventional-100-l5-q13-ballistic-in/dynamical_matrix.mat \
    --run cluster/si-conventional-100-l5-q13-ballistic-s4/run.npz \
    --nk 1025,2049,4097 --area 2.9899024e-19 \
    --output phonon/studies/out/si_conventional_100_ballistic_q13_production.json

QTX_ARRAY_MODULE=numpy PYTHONPATH=src:phonon \
    python phonon/studies/_si_auxiliary_scba_review.py \
    --case L3=cluster/si-aux-l3c/poles.npz \
    --vertices cluster/si-aux-inputs/decomposed_vertices.npz \
    --output phonon/studies/out/si_auxiliary_scba_L3_full.json

QTX_ARRAY_MODULE=numpy PYTHONPATH=src:phonon \
    python phonon/studies/_si_auxiliary_scba_review.py \
    --case L8=cluster/si-aux-l8b/poles.npz \
    --vertices cluster/si-aux-inputs/decomposed_vertices.npz \
    --external-q-count 1 \
    --output phonon/studies/out/si_auxiliary_scba_L8_worstq.json

PYTHONPATH=src:phonon python -m pytest -q \
    tests/quatrex/phonon/test_hybrid_frequency_review.py \
    tests/quatrex/phonon/test_si_kk_audit.py \
    tests/quatrex/phonon/test_spatial_hierarchy_review.py \
    tests/quatrex/phonon/test_conserving_spatial_tail_review.py \
    tests/quatrex/phonon/test_cnt_reblock_acceleration.py \
    tests/quatrex/phonon/test_cnt_fc3_compression_review.py \
    tests/quatrex/phonon/test_auxiliary_scba.py \
    tests/quatrex/phonon/test_si_auxiliary_scba_review.py \
    tests/quatrex/phonon/test_si_long_film_spatial_gate.py \
    tests/quatrex/phonon/test_vertex_factors.py \
    tests/quatrex/phonon/test_btd_linalg.py \
    tests/quatrex/phonon/test_reblock_device.py
```

The canonical test command was run in the `quatrex-dev` environment with
`QTX_ARRAY_MODULE=numpy`: all 75 tests passed.  The reduced CPU studies
reproduced the tables above.  The
focused ballistic audit and q-resampling tests passed four of four on a GH200
as Alps job 4556216.  The
restart-rich fits used the same code and seeds on Tortin; their pulled logs are
under the gitignored `cluster/cntfc3-r*-r2/` directories.  The CUDA study ran
on a GH200 as Alps job 4552618; its pulled log is
`cluster/cnt-sparse-ring-v3/slurm-4552618.out`.  The `d=8` spatial bed ran as
Alps job 4552671; its pulled log and arrays are under
`cluster/spatial-d8-l16b/`.

| Gate | Result | Status |
|---|---:|---|
| rational bubble vs independent oracle <= `1e-10` | `4.5e-16` | pass |
| mixed bubble <= `2e-3`, bounded as `gamma/h -> 0` | max `3.671e-4` | pass |
| new positivity / anti-Hermiticity defect <= `1e-10` | PSD proxy; `2.27e-16` anti-Hermitian residual | pass on proxy |
| frequency cost independent of inverse linewidth | 1.81--1.93 ms | pass |
| exact adaptive P1 bubble bounded as `gamma/h -> 0` | max subcell error `1.25e-3`; equivalent uniform grid up to 2.1 million cells | pass on isolated-line proxy |
| shared adaptive real-Si cost beats fine uniform FFT | 373 cells but 4.03 million transformed modes versus 60,001 | **fail; q-local joint hierarchy required** |
| passive auxiliary RGF/Woodbury vs dense oracle | approximately `1e-12` or below | pass |
| real-Si source passivity | floors `-4.04e-9` (L3), `-9.16e-9` (L8) | pass within frozen-reference defect |
| real-Si constant-source error <= `2e-3` | medians 10.5 % / 15.9 % | **fail** |
| real-Si auxiliary cost beats two-cell reblock | 9.5 x (L3), 44.5 x (worst-q L8) | **fail for wholesale promotion** |
| conserving closed auxiliary-state SCBA | fixed-basis state mixer not implemented | **open; selective path only** |
| real-Si L8x2 retarded residual <= `1e-3` | `9.511e-4` after the final continuation | pass |
| equilibrium Si KMS, global max / L2 | `6.21e-12` / `4.06e-11` | pass |
| equilibrium Si independent retarded assembly | `3.74e-16` | pass |
| equilibrium Si zero current / 305--295 K current scale | `6.19e-16` | pass |
| equilibrium Si bubble energy balance | `4.01e-16` | pass |
| ballistic Si Meir--Wingreen vs independent Caroli, spectral / integrated | `2.20e-12` / `0` at q=17 | pass |
| ballistic Si frequency refinement, last two changes | `0.118 %`, `0.010 %` | pass |
| ballistic Si transverse q refinement, q9--q13 / q13--q17 | `0.125 %`, `0.005 %` | pass |
| ballistic Si temperature linearity, 10 K / 5 K bias | `1000.7980` / `1000.8231` MW m\(^{-2}\) K\(^{-1}\), `0.00251 %` | pass |
| independent Bloch mode count vs production Caroli | `999.6319` / `999.5474` MW m\(^{-2}\) K\(^{-1}\), `8.45e-5` relative | pass |
| production FC2 vs recovered original fit | maximum sampled relative difference `4.52e-15` | pass |
| primitive-to-conventional hiPhive Gamma folding | maximum frequency difference `5.57e-7` THz | pass |
| conventional [100] q13--q17 / q17--q21 mode-integral refinement | `0.0846 %` / `0.0103 %` | pass at q=17 |
| conventional [100] production MW vs independent q13 mode integral | `1028.3548` / `1028.0337` MW m\(^{-2}\) K\(^{-1}\), `3.12e-4` relative | pass |
| conventional q17 production memory | 101.25 GB before a failed 0.43 GB spectral-OBC batch; q13 peak 79.25 GB | q13 accepted; tiled q-frequency storage needed for q17 |
| conventional [100] vs primitive-orientation conductance | `1027.1643` / `1000.8480` MW m\(^{-2}\) K\(^{-1}\), `+2.63 %` | orientation isolated |
| conventional [100] vs Guo et al. different-FC-input conductance | `1027.1643` / `1065.81` MW m\(^{-2}\) K\(^{-1}\), `-3.63 %` | same-orientation scale check |
| Guo-close joint-refit rank-256 FC3 frozen self-energy | INDSCAL `6.39 / 3.35 / 2.66 %`; symmetric CP `6.88 / 3.78 / 3.05 %` for lesser / greater / retarded | **fail; exact dense vertex required** |
| coupled-q analytic JVP vs central difference | `6.34e-10`--`9.81e-10` at step `1e-5` | pass |
| exact dense joint-refit q3, 1 THz SCBA root | residual `2.763e-10`, spread `6.948e-11`, bubble defect at most `1.84e-16` | pass as nonlinear gate; grid deliberately coarse |
| exact dense joint-refit q3, 0.5 THz linear continuation | factor 0.10 reaches residual `0.24612` at map 14, then rebounds to `0.42319`; factor 0.05 reaches `0.24556` at map 27 and then rises | **bounded Picard instability; no convergence** |
| exact Newton on the q3, 0.5 THz continuation | one 8-vector correction costs 16 bubbles; residual `0.50094 -> 0.42423` but accepted state emits into both leads with spread 2.0 | **fail without a passivity and heat-flow branch constraint** |
| ballistic Si conductance vs Guo et al. different-FC-input and orientation scale | `1000.85` vs `1065.81` MW m\(^{-2}\) K\(^{-1}\), `-6.10 %` | scale check only |
| real-Si L8x2 internal current spread <= `1e-3` | `2.829e-3` versus primitive `4.371e-3` | **fail; wider exact near field needed** |
| real-Si L8x2 cost beats matched primitive | 32.35 versus 34.66 s; 37.58 versus 66.82 GB mempool | pass |
| spatial operator/current error <= `1e-3` | max HODLR current `4.283e-5` | pass |
| spatial anti-Hermiticity | structural to roundoff | pass |
| no added negativity on real L8x2 at `1e-3` | additional floor no worse than `-6.7e-17` | pass on the three snapshots |
| real L8x2 HODLR cost beats exact reblock at `1e-3` | storage ratio 0.958--1.092; global rank 41--46 / 48 | **fail across required cases** |
| hierarchical/post-hoc SSS cost beats reblocking at large real `d,N` | `d=8`: reblock error `1.31e-4`; narrower S2 error `2.19e-3` and imbalance `6.62e-3` | **fail** |
| exact SSS spatial bubble closure | rank `2 x 3 -> 6`, error `1.75e-16` | pass |
| exact sparse SSS Schur solve | max focused error `1.81e-15` | pass |
| SSS collision-moment enrichment | rank `8 -> 9`, defect `3.47e-16` | pass on frozen functional |
| direct-generator SSS cost beats reblocking | 8.13 % dense nnz, but generic LU 2.3 x dense and no CNT closed SCBA | **open; blocks production** |
| finite-CNT flat vertex factor <= 1 % | restart-rich rank 256: 4.52 % | fail |
| exact atom-sparse vertex storage | 13.56 % onsite, 4.52 % cross; zero approximation | pass |
| fused atom-sparse ring beats dense primitive ring | task-weighted GH200 ratio `0.5207`, exact parity | pass on isolated kernel; frozen production contraction open |
| exact four-ring production A/B on conserving CNT | 1.48 x iteration; final current differs by `1.08e-13` relatively | **pass** |
| exact FC3 factor lift under 2-cell reblocking | dense subblock parity, rank `128 -> 256`, 6.8 MB | pass |
| long-Si primitive archive contains a measurable far field | shells `|I-J|>1` exactly zero in all three samples | **fail: output pin** |

The focused test modules additionally cover exact rational reconstruction,
cell moments, Toeplitz/direct cell parity, subgrid output poles, exact near-band
retention, dense/apply parity, sparse-extended solves, anti-Hermiticity,
congruence positivity and phase-rank invariance.

## 15. Promising analytical long-range architecture

The detailed derivation and measurements are in
[`conserving_long_range_tail.md`](conserving_long_range_tail.md).  The central
distinction is between fitting a completed dense tail and generating its
auxiliary states inside the diagram.  The first remains a no-go on the frozen
`d=1,2,4` beds: directional ranks 6, 14 and 22 at the selected high-self-energy
samples give extended representations larger than dense.  The second has a
valid exact algebra:

\[
 A_\Sigma(\omega,\Omega)
 =A_1(\omega)\otimes A_2(\Omega-\omega),
\]

with the two FC3 vertices folded into the endpoint maps.  Frequency, q and
offset contributions are direct sums followed by symmetry-constrained model
reduction.  The exact near band locally replaces the near part of this state
model, so its rank does not pay the interface penalty of a zeroed-band fit.

The corresponding Dyson equation is a sparse extended recurrence in the
physical, forward-state and backward-state variables.  Eliminating the states
recovers the represented long-range matrix exactly.  This construction is the
same time-varying-system structure used by fast SSS solvers
([Chandrasekaran et al.](https://doi.org/10.1137/S0895479802405884)); a
specialised factorisation is required because generic sparse LU does not turn
the low nonzero count into a speed-up.

The Keldysh step also remains in this class.  SSS solve/multiplication gives
`G^x=G^R Sigma^x G^A` as generators, or positive local sources can be
accumulated by forward/backward Lyapunov covariance recurrences.  Thus the
bubble need not be materialised merely to construct the next `G^{<,>}`.  The
rank after this Keldysh product is an additional decision gate.

The completed `d=8` bed rules out the old dense-fit Arm S route: S2 is
nominally narrower than a two-cell reblock but is 17 times worse in current
and four orders worse in lead balance; S4 already matches the reblock width.
This strengthens the distinction between post-hoc fitting and producing the
states inside the projected diagram.

Spatial and spectral auxiliary states can be combined.  Smooth frequency
cells use the existing FFT, rational input clusters close in temporal
state-space, and spatial products close in SSS state-space.  Narrow output sum
poles remain rational.  Exactness is recovered as both state ranks and the
near band grow; cheapness is conditional on recompressed ranks remaining
bounded.

Conservation requires more than adjoint upper/lower generators.  With a fixed
weighted reconstruction `R`, define `P=R R*_W` and approximate the functional
as `Phi_tilde[G]=Phi[P G]`.  Its derivative is
`Sigma_tilde=P*_W Sigma_SCBA[P G]`.  This is the object that must enter the
Dyson solve.  Collision-moment enrichment and positive half-diagram
congruence are implementation gates, not substitutes for the projected
functional.  The basis can be adapted only between converged epochs unless
its derivative is included.

For CNT the pass condition is end-to-end current and conservation parity with
the 8 x 2 reference while beating the 3.60 s projected exact-sparse/four-ring
iteration.  If `r/d` is not small, the shared spatial-frequency state count
grows with grid size, or a specialised ULV solve loses at matched current
accuracy, retain exact reblocking/adaptive range.  This makes the proposal a
bounded experiment rather than an assumption that every propagating tail is
compressible.
