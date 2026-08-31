# Reproducing the conserving CNT 8 x 2 result at lower cost

**Status:** repository and saved-run investigation, 2026-08-28  
**Production changes:** none  
**Reproduction:**
[`_cnt_reblock_acceleration.py`](../studies/_cnt_reblock_acceleration.py)

## 1. Decision

The `c16x2h` result is reproducible more cheaply, but the evidence does **not**
support replacing its newly retained spatial blocks by an ordinary low-rank
tail.  Those blocks are near field, materially large and almost full rank.

The ranked path is:

1. **Enable the exact four-ring Keldysh reuse on the unchanged 8 x 2, band-3
   calculation.**  `sse_greater_from_lesser` evaluates four rather than six
   ring contractions per vertex quad.  The current run did not enable it.
   Applying the measured 4/6 ring ratio predicts `8.92 s/iteration`, down from
   `12.90 s`, with the same Green-function and self-energy masks.
2. **Retry FC3 acceleration, but do not repeat the old flat fit.**  The
   factored kernel is algebraically exact for the vertex it is given.  A new
   audit of the actual CNT primitive vertex finds that one global INDSCAL fit
   is at 4.52 % error at rank 256 even after a restart-rich retry, while the
   exact onsite/cross-cell blocks contain only 228/76 active atom triplets out
   of 1728.  Including compact indices, those exact formats use 13.56/4.52 %
   of dense block storage.  The promising target is therefore an exact
   atom-sparse ring followed, conditionally, by a symmetry-coupled shell/orbit
   factorisation lifted into the two-cell layout without refitting.  Its
   task-weighted arithmetic is 37.13 % of the primitive dense ring; at equal
   useful-MAC throughput it projects to 3.78 s/iteration, or 2.85 s with
   four-ring reuse.  The real-shape GH200 prototype measures a task-weighted
   0.5207 time ratio, giving more conservative projections of 4.92/3.60 s for
   six/four rings.
3. **If an exact dense-vertex path is still required, reblock only the Dyson
   solver and contract the bubble on 36-DOF primitive subblocks.**  The
   grouped approximation is unchanged.  The archived task topology gives
   `85.4e3` rather than `319.6e3 GFLOP/pass` on the slow rank.  Using the
   measured 36-DOF throughput, conservatively, predicts `8.55 s/iteration`, or
   `6.02 s` when combined with four-ring reuse.
4. **Do not return to `g_band=1`.**  The existing CNT campaign already shows
   that it does not reproduce the conserving result.  Isolating Green-input
   from Sigma-output range now requires private explicit primitive masks, not
   another `g_band=1` production arm.

The ordinary per-frequency Woodbury/HSS proposal is a no-go on this snapshot.
At 1 % relative Frobenius error its required rank is already 63, whereas an
optimistic cubic solve model requires rank below 10.6 to beat exact two-cell
reblocking at 36 DOF/cell.

## 2. What the reblock actually changed

The reblocker verifies the dense FC2 operator, dense FC3 vertex and
translational FC3 blocks before writing the two-cell device.  The harmonic and
anharmonic inputs are therefore the same physical operators.  The production
approximation is not the same, because `sse_g_band` and the hard self-energy
output pin are expressed in **solver blocks**.

`sse_phonon_phonon.py` creates output pairs only for `|I-J| <= 1`, independent
of `sse_g_band`.  The selected solve returns G through the requested block
band.  Consequently:

| run | selected G in primitive cells | generated Sigma stored in primitive cells |
|---|---|---|
| 16 x 1, block `g_band=3` | every distance 0--3 | every distance 0--1 |
| 8 x 2, block `g_band=3` | every distance 0--6 and 5/9 distance-7 pairs | every distance 0--2 and 7/13 distance-3 pairs |

For adjacent two-cell blocks, the 72 x 72 self-energy block contains four
36 x 36 primitive blocks.  Only one was available to the one-cell output
pattern:

```text
                       destination primitive cell
                         2B+2          2B+3
source 2B       [ distance 2     distance 3 ]
       2B+1     [ distance 1     distance 2 ]
```

Thus `c16x2h` changes two supports at once: Sigma gains two distance-2 blocks
and one alternating distance-3 block at each grouped boundary, while the G
legs become much wider.  The present run alone cannot say which change caused
the 125-fold conservation improvement.  The already failed `g_band=1` arm
cannot answer that question; the explicit support-isolation arms in Sec. 7 can.

The larger `interaction_cutoff=40 A` in `c16x2h` is not the explanation.  The
CNT campaign found 10 and 40 A numerically identical at one- and two-cell
blocking.  Forty A merely made the archived sparse pattern block dense.  A
production rerun should restore 10 A and let the explicit G-band extension add
only the required stored blocks.

## 3. Measured physics and cost

Both runs use the half retarded rule, `eta=0`, 161 frequencies and the same
16-cell physical CNT.  Both met the self-energy convergence tolerance.

| quantity | 16 x 1 | 8 x 2 |
|---|---:|---:|
| iterations | 98 | 101 |
| relative Sigma-R residual | 9.63e-4 | 9.95e-4 |
| lead balance | 9.51e-5 | 5.04e-4 |
| interior heat-current spread | 6.27 % | **0.050 %** |
| reported interior current | 13.187 | **11.741** |
| `G/G_ball` | 0.557 | **0.496** |

Post-warm-up medians parsed directly from the profiler logs are:

| stage | 16 x 1 | 8 x 2 | ratio |
|---|---:|---:|---:|
| whole iteration | 4.389 s | 12.900 s | 2.94 x |
| ring contraction | 3.662 s | 11.933 s | 3.26 x |
| phonon solver | 0.638 s | 0.829 s | 1.30 x |

The result is important: the practical reblocked solver is **not** four times
slower.  Its measured penalty is only 0.19 s.  The cost problem is the dense
72-DOF ring, which occupies 92.5 % of the reblocked iteration.  A method that
optimises the Dyson recurrence but leaves the merged ring unchanged is aimed at
the wrong stage.

The production flop model independently reproduces the log headers:

| dense-ring representation on the slow rank | quads | rings/quad | model |
|---|---:|---:|---:|
| 16 x 1, 36 DOF, G band 3 | 2104 | 6 | 41.220e3 GF/pass |
| 8 x 2, 72 DOF, G band 3 | 513 | 6 | 319.621e3 GF/pass |
| solver 8 x 2, primitive 36-DOF bubble | 2193 | 6 | 85.396e3 GF/pass |

The first two counts are not fitted: the stored CNT FC3 topology has 106
primitive blocks, merges to 50 two-cell blocks, and enumerates exactly the
2104/513 quads printed by the two production runs.

## 4. The newly retained Sigma is not a compressible far tail

The saved `c16x2h/sigma_best.rank*.npz` files hold the converged block-band-1
self-energy over all 161 frequencies.  Splitting every adjacent 72 x 72 block
back into primitive pieces gives:

| primitive part | fraction of squared Frobenius weight | Frobenius ratio |
|---|---:|---:|
| old distance 0--1 support | 93.849 % | 96.88 % |
| newly admitted distance 2 | 5.003 % | 22.37 % |
| newly admitted alternating distance 3 | 1.148 % | 10.72 % |
| all newly admitted pieces | **6.151 %** | **24.80 %** |

The current changes by 11 % when these blocks and the wider G legs are
admitted, so a 24.8 % Sigma norm is entirely capable of being causal.  Calling
it a negligible tail would not be defensible.

Numerical ranks were measured on 22 evenly spaced frequencies and all seven
upper grouped links.  `Delta=Sigma^<-Sigma^>` has the same ranks as the saved
full `Sigma^R` under the half rule.

| relative Frobenius tail | assembled 72 x 72 farther correction, median / p90 / max | individual 36 x 36 far block, median / p90 / max |
|---:|---:|---:|
| 1e-2 | 63 / 66 / 67 | 32 / 34 / 35 |
| 1e-3 | 70 / 71 / 72 | 35 / 36 / 36 |
| 1e-4 | 71 / 72 / 72 | 36 / 36 / 36 |

For an optimistic local auxiliary solve with effective primitive block width
`d+2r`, beating exact pair reblocking requires

\[
 N(d+2r)^3 < \frac{N}{2}(2d)^3,
 \qquad r < \frac{d}{2}(2^{2/3}-1)=10.57
 \quad(d=36).
\]

The measured ranks miss this bound by factors of six to seven before sparse
extension overhead is counted.  Hierarchical far-field compression also has no
place to amortise: these are the first two omitted shells, not large separated
clusters.  They should remain exact near field.

## 5. Cheapest exact change: four instead of six rings

The dense Gamma kernel already implements
`sse_greater_from_lesser`.  It reconstructs the two cross terms of
`Sigma^>` from the `Sigma^<` pass using the exact bosonic tau-domain identity.
The direct diagonal term remains evaluated.  This changes the per-quad work
from six ring calls to four and does not change the spatial or frequency
representation.

The archived `c16x2h` line says `rings/quad=6`, so the option was off.  Keeping
the measured non-ring time and scaling only the ring gives

```text
12.900 - 11.933 + (4/6)*11.933 = 8.923 s/iteration.
```

Before a long campaign, run a short single-rank or small-bed verification with
`sse_fold_verify_iterations > 0`, then a multi-rank frozen-state A/B.  Required
gates are `Sigma^{<,>}` relative error below `1e-11`, bubble energy-balance
parity, identical residual trajectory to summation-order tolerance, and final
current parity.  This is the lowest-risk immediate win.

`sse_hermitian_pairs` could remove roughly half the output-pair work, but it is
currently single-rank only and is exact only to the input G anti-Hermiticity
error.  It is therefore secondary to the exact four-ring identity.

## 6. Exact dual layout: grouped Dyson, primitive bubble

The dense merged ring wastes work because a 72-DOF block is treated as one
dense tensor even though it was formed by merging sparse primitive FC3 blocks.
Multilinearity gives an exact alternative:

1. Keep SCBA/Dyson/mixing storage in eight 72-DOF solver blocks.  This preserves
   the exact block-tridiagonal operator and the successful output mask.
2. Expose every selected 72 x 72 Green block as four 36 x 36 primitive views.
   Grouped G band 3 then gives exactly the primitive mask through distance 6
   plus alternating distance 7.
3. Keep the 106 primitive FC3 blocks and build the ring task index using the
   grouped predicates for G and output ownership.
4. Accumulate primitive 36 x 36 Sigma blocks for every pair whose grouped
   indices differ by at most one.
5. Pack those views into the 72 x 72 block-tridiagonal Sigma before retarded
   reconstruction/mixing and the next Dyson solve.

This is not a new approximation.  Expanding a merged FC3/G contraction over
its primitive subindices gives exactly these primitive tasks; the only change
is that structural zero subblocks are never passed through dense GEMMs.

The slow-rank model falls by 3.74 x.  The 36-DOF baseline sustains lower GPU
throughput than the 72-DOF run, so applying the measured 36-DOF throughput is a
conservative estimate: `7.59 s` for the ring and `8.55 s` for the iteration.
Four-ring reuse reduces the same estimate to `6.02 s`.  The implementation
gate is one frozen-G contraction with max-relative parity below `1e-11` for
`Sigma^<`, `Sigma^>` and the assembled retarded buffer, followed by identical
closed-SCBA current and conservation.

The measured atom-triplet fills refine this projection.  In the existing
three-GEMM ring, sparsity reduces the first and second FC3 actions while the
final `T @ U` remains dense.  Task-weighting the onsite and cross-cell fills
over the 2193 slow-rank primitive quads gives

```text
ideal MAC ratio = sum_quads(1 + fill_left + fill_right) / (3 * nquads)
                = 0.371318.
```

At the measured 36-DOF useful-MAC rate, the ideal six-/four-ring iterations are
3.784/2.845 s.  Because the final dense contraction keeps normal GEMM
throughput, the two sparse vertex actions themselves need only 3.1 % of dense
useful-MAC throughput to beat the unchanged-layout four-ring result, and
11.1 % to beat the nonconserving one-cell iteration when four-ring reuse is
included.

The private CUDA benchmark
[`_cnt_sparse_ring_gpu_bench.py`](../studies/_cnt_sparse_ring_gpu_bench.py)
then evaluated the real CNT blocks on an Alps GH200 with 161 complex128 Green
matrices.  It retains the final dense contraction and reports zero relative
difference at the printed precision:

| vertex layout | sparse/dense ring time | speed-up |
|---|---:|---:|
| onsite--onsite | 0.6248 | 1.60 x |
| onsite--cross, 30/52 output rows | 0.5280 / 0.5660 | 1.89 / 1.77 x |
| cross, 30/52 rows--onsite | 0.5808 / 0.5950 | 1.72 / 1.68 x |
| four cross30/cross52 pairs | 0.4846--0.5379 | 1.86--2.06 x |

Using `(onsite,cross30,cross52)` for both left and right vertices, the exact
nine-category slow-rank counts are
48/184/92/184/708/354/92/354/177 in row-major order.  Their
weighted time ratio is 0.5207 (1.92 x), projecting the archived conserving
iteration to 4.917 s with six rings and **3.601 s with four rings**.  This is
strong enough to implement the sparse actions in a frozen production
contraction.  It is not yet an end-to-end result: q/task launch overhead, MPI
distribution, packing, and the closed-SCBA fixed point still need parity and
timing gates.

The production interface needs two layouts rather than a wider public physics
option:

```text
SolverBlockLayout     8 x 72, block-tridiagonal Dyson and mixed Sigma
InteractionMicroLayout
    16 x 36 primitive FC3/G/Sigma views
    solver_block_of_micro = floor(i/2)
    G predicate     abs(B(i)-B(j)) <= 3
    Sigma predicate abs(B(i)-B(j)) <= 1
pack/unpack          view-based where contiguous, tested adjoints otherwise
```

## 7. Experiments needed to identify the minimum faithful support

The present 16 x 1 versus 8 x 2 comparison is decisive about reblocking but
does not isolate its cause.  The smallest useful ladder is:

| arm | G mask in primitive cells | Sigma output mask | purpose |
|---|---|---|---|
| A | distance <=3 | distance <=1 | existing 16 x 1 reference |
| D | grouped band 3 | grouped band 1 | existing conserving `c16x2h` reference |
| E | distance <=3 | grouped band 1 | dual-layout isolation of the Sigma-output change |
| F | grouped band 3 | distance <=1 | dual-layout isolation of the longer-G change |
| G | grouped band 3 | every distance <=2, no distance 3 | tests whether the alternating third shell is needed |

Arm D is the existing reference.  E--G should remain private study switches
until the mechanism is known.  For a stable nonzero-current run, accept a
cheaper arm only with:

* interior spread and lead balance no worse than `1e-3`;
* current within 1 % of arm D;
* no material increase in the saved reference's positivity defect;
* converged residual below `1e-3`, not merely an early low-conservation
  iterate;
* bubble energy balance at the dense-kernel reference level.

The winning L16 arm must then be repeated at L24 and L32.  The one-cell error
grows with length, so L16 parity alone cannot certify a new spatial
approximation.

## 8. What the finite-device FC3 retry actually requires

The current `c16x2h` run used the dense Gamma ring.  Quatrex already has a
Gamma-capable tensor-decomposed kernel whose contraction is exact to roundoff
for a supplied factorized vertex.  For a rank-R vertex its leading work is
`O(R b^2 + R^2 b)` rather than the dense `O(b^4)` ring.  This is especially
relevant at the reblocked `b=72` where the dense ring dominates.  The missing
piece is not the contraction identity; it is a compact, conserving CNT vertex.

### 8.1 What failed before

The previous finite-device transport sweep was d11a SiNW, not CNT, and stopped
at rank 16.  At that rank INDSCAL and Waring each had about 15 % conductance
error, CP 27 %, HOSVD 23 %, and mSVD was unphysical.  It demonstrated that a
global Frobenius target is poorly aligned with transport and that ansatze which
do not retain the full potential symmetry can lose conservation.

The new CNT audit
[`_cnt_fc3_compression_review.py`](../studies/_cnt_fc3_compression_review.py)
builds the exact `36 x 108 x 108` primitive offset tensor from the committed L4
device.  Its seven support pairs have S2 error zero and ASR residual
`1.25e-16`.  Separate deterministic global INDSCAL screens (algebraic
initialisation plus ALS, without a restart campaign) give:

| rank | 16 | 32 | 64 | 96 | 128 | 160 | 192 | 256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| relative FC3 error | .669 | .376 | .285 | .202 | .165 | .131 | .107 | .0827 |

This makes another flat global fit unlikely to reach a 1 % current gate at a
rank where the `R^2 b` Gram remains cheap.  It is not a definitive optimiser
comparison, so the requested independent restart-rich fits were run on
Tortin.  With an algebraic start plus two random starts, 250 ALS iterations
and 150 L-BFGS iterations, ranks 64/128/256 reach 22.20/13.24/4.520 % error in
20.0/122.0/230.6 seconds.  Observable convergence can be much faster than
tensor convergence, so rank 256 may still enter a frozen-bubble gate if its
factor-kernel cost is competitive.  It should not trigger a long closed-SCBA
campaign merely because it is the best flat fit.

### 8.2 The structure the flat fit discards

At an exact `1e-12` relative atom-block threshold:

| orbit block | active 3 x 3 x 3 atom triplets | fill |
|---|---:|---:|
| onsite | 228 / 1728 | 13.19 % |
| directed cross `001` | 76 / 1728 | 4.40 % |
| directed cross `011` | 76 / 1728 | 4.40 % |

The onsite tensor is fully S3 symmetric; each cross tensor is symmetric in the
two legs that occupy the same cell, and the translated/permuted orbit supplies
the other device blocks.  These are the natural block terms.  Flattening the
cell offset, atom and Cartesian indices into one CP fit spends rank both on
seven different orbit roles and on reproducing their exact structural zeros.

The exact indexed atom format stores the full 27 Cartesian coefficients plus
three 16-bit atom indices per active triplet.  Its resulting storage is 13.56 %
of dense onsite and 4.52 % of dense cross-cell storage, or about 7.5 % over the
three orbit representatives.  This is exact compression and keeps the only
format property that the low-rank trials consistently lose: literal support.

A support-block Tucker diagnostic tested whether simply separating the seven
offset roles was enough.  It preserves the support and S2 exactly, but at rank
32 still has 9.99 % global FC3 error and a 12.0 % contracted-leg ASR residual;
rank 36 is required for an exact reconstruction.  The failure is informative:
independent block truncation removes cancellations between onsite and cross
orbits, so the global-ASR coupling is not optional.

### 8.3 Ranked retry

1. Implement an exact atom-sparse/microblock ring over the existing FC3
   support.  It is the baseline against which every approximate factor format
   must win.
2. Recover the cubic-potential orbit parameterisation from hiPhive/symfc (and
   retain the upstream FCP in future reaps).  Fit onsite and directed cross-cell
   orbits jointly with a block-term/Tucker/PCP or interpolative-THC form.
3. Enforce S3, exact offset support and the **global** ASR in that shared
   parameterisation.  Projecting each orbit independently is wrong because its
   ASR cancellations occur between onsite and neighbouring-cell terms.
4. Lift primitive factors exactly into the two-cell solver layout.  Do not fit
   the merged 72-DOF tensor: grouping introduces no new physics and an exact
   lift costs at most a small multiple of the primitive orbit ranks.
5. Fit `R=...` candidates independently.  CP/INDSCAL solutions are non-nested;
   truncating a high-rank file by column norm is not a controlled lower-rank
   optimum.
6. Use a shell-balanced plus frozen-bubble loss.  On representative converged
   G slices require `Sigma^{<,>}`, anti-Hermiticity and bubble-energy parity
   before spending a full SCBA run.  Then require current within 1 % and
   interior spread below `1e-3` against `c16x2h`.

Using one fixed approximate cubic potential at both vertices is compatible
with a conserving diagram.  An arbitrary post-hoc low-rank fit of Sigma is
not.  Existing bulk/SiNW speed-ups establish that the kernel can pay when a
compact factorisation exists; they do not establish that CNT has the required
rank.

## 9. Recommended order of work

1. Run `c16x2h-g3-gfl`, with the successful band and spatial mask unchanged and
   cutoff restored to 10 A.  This obtains the exact four-ring speed win.
2. Implement the exact solver/microblock and atom-sparse bubble layout.  It is
   both the dense-vertex fallback and the reference cost for factorisation.
3. Add orbit-component factors and do independently fitted frozen-G plus
   closed-SCBA rank gates.  Stop if they reduce the ring below the solver/FFT
   floor; reject flat global INDSCAL if its passing rank is too expensive.
4. Do not implement a Woodbury/HSS representation for these first omitted
   shells.  Revisit hierarchy only for interactions beyond the exact grouped
   near band and only after their actual rank is measured.
5. Validate the selected method on the 24- and 32-cell two-cell-blocked ladder.

This order preserves the physical content which fixed conservation, attacks
the stage that actually costs time, and supplies an unambiguous fallback: the
existing 8 x 2 result remains the fidelity reference.

## 10. Reproduction

```bash
PYTHONPATH=src:phonon python phonon/studies/_cnt_reblock_acceleration.py \
    --json /tmp/cnt_reblock_acceleration.json

PYTHONPATH=src:phonon python phonon/studies/_cnt_fc3_compression_review.py \
    --fit-ranks 16,32,64,96,128,160,192,256 \
    --json /tmp/cnt_fc3_compression_review.json

PYTHONPATH=src:phonon python phonon/studies/_cnt_fc3_compression_review.py \
    --fit-ranks 64,128,256 --restarts 2 --max-iter 250 \
    --lbfgs-iters 150 --seed-base 1729 \
    --json /tmp/cnt_fc3_restart_review.json

# Run on a CUDA host with CuPy (measured on one Alps GH200).
PYTHONPATH=src:phonon python phonon/studies/_cnt_sparse_ring_gpu_bench.py \
    --json /tmp/cnt_sparse_ring_gpu.json

PYTHONPATH=src:phonon python -m pytest -q \
    tests/quatrex/phonon/test_cnt_reblock_acceleration.py \
    tests/quatrex/phonon/test_cnt_fc3_compression_review.py
```

The study reads the archived profiler logs and, unless `--skip-sigma` is
given, the four saved `c16x2h/sigma_best.rank*.npz` slices.  It does not write
to the runs or alter production configuration.
