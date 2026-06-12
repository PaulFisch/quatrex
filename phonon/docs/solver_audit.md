# Audit of the production phonon solver (`src/quatrex`): physical correctness & performance

*2026-06-10. Spot-check scripts: `phonon/scripts/verify/audit_hilbert.py`,
`audit_bubble_fold.py`, `audit_qfold_trs.py` (single-core, seconds).*
***Update (same day): all flagged items FIXED — see the per-item status lines.***

## Executive summary

| # | Item | Verdict | Severity |
|---|------|---------|----------|
| 1 | `retarded_method="fft"` Hilbert reconstruction of Σ^R | **BUG (was): PV part overscaled by π/dω (43–160×).** → **FIXED**: `hilbert_transform` is now the normalized (1/π) PV transform with exact log-PV cell weights; validated to 1.3% vs exact (was 4300%+). Production unaffected throughout (uses `"half"`) | **HIGH → fixed** |
| 2 | DC handling (G^≷(ω=0)=0, Bose regularisation) | Correct-by-construction regularisation; ≤3% effect bound | OK |
| 3 | Bubble FFT fold (3-term negative-ω folding) | **Verified machine-exact** vs direct full-axis convolution | OK |
| 3b | Coupled-q vertex pairing | **REAL BUG (was): left vertex must be conjugated.** Proven by a real-space supercell ground truth (74% shape error before, 3e-15 after; TRS Σ(−q)=Σ(q)^T restored). → **FIXED** in `sse_phonon_phonon.py` AND dense `se_finite.py`. Γ-only results unaffected (real vertices). Likely explains part of the film-vs-Guo gap | **HIGH → fixed** |
| 4 | Meir-Wingreen heat current & lead signs | Correct structure; sign conventions internally consistent and empirically anchored | OK |
| 5 | Zero-mode projector floor (0.1 THz) | Documented, intended (includes d5a twist); spectrally unresolvable on all production grids | OK (note) |
| 6 | Dyson `(ω+iη)² − D − Σ` + OBC | Dyson standard. **Sancho-Rubio diverged on the d5a soft mode** (every d5a contact NaN'd the run) → **FIXED**: production configs default to the spectral NEVP (full eigensolver) OBC; d5a study rerunning | **HIGH → fixed** |
| 7 | SCP cubic tadpole | Textbook tadpole chain; pseudo-inverse floor is deliberate soft-mode regularisation | OK (note) |
| 8 | Causality / Im Σ^R | `"half"` path: Im part machine-exact (spot-check #1) | OK |
| 9 | **Σ grid-spacing mismatch** (found post-audit by the variants smoke test) | **REAL BUG**: the SSE took its dω from `phonon_energies.npy` while G lives on the configured window — Σ misscaled by dω_npy/dω_solver in every study where the geometry-build nfreq ≠ config nfreq (cnt33 ×0.75, cnt80 ×0.55, d11a/d5a ×0.40; srtio3 ×1.0 by luck). → **FIXED** three ways: scba passes the solver grid to the SSE; the SSE refuses mismatched grids (hard error); write_config regenerates the npy. **All affected anharmonic runs relaunched** (`run_master2.sh`); ballistics (Σ=0) kept | **HIGH → fixed** |
| 10 | **Conservation gate unreachable / "bad ballistic conservation"** | **Diagnosis**: the max−min heat spread over ALL interfaces contains the **η-absorption dip** — finite (ω+iη)² broadening soaks up flux at internal interfaces (leads stay exactly symmetric, balance ≤ 1e-9 even ballistically). Grows with device length (cnt33 3.2→6.1% for L2→L4) and T; d11a worst (11–17%); srtio3 0.2%. η-scan on d11a L2 (41 pts): dip 8.6/11.5/13.6% at η = 0.055/0.11/0.22 — sublinear with a resolution-related η-independent component. **The lead G itself is strongly η-regularized for the slow d11a cell: 9.23/6.10/3.24 (raw, same grid) at the same ηs** — absolute G needs η-awareness; the anh/ball *ratio* is the robust observable (same-η bias cancels). Grid check at fixed η=0.11: 41→161 pts leaves the dω-normalized ballistic G unchanged to 1% (2.75 vs 2.72) — the ballistic integral was grid-converged; the refinement matters for SCBA *stability*, not the ballistic quadrature. The dense code never showed the dip because its `conservation_err` is **leads-only** `\|J_L−J_R\|` and its ballistic G is Caroli transmission (exactly conserving by construction). Gating on the spread with tol 1e-2 made convergence *unreachable*. → **FIXED**: gate on **lead balance**; internal spread saved as `internal_spread`/`eta_dip` diagnostic | **HIGH → fixed** |
| 11 | **SiNW SCBA divergence at corrected Σ** | Localized by the E-matrix probes: the instability lives in the **resolved near-DC bins at small η** — 41 pts/η0.11 converges (bins start at 0.45 THz), 161 pts/η0.11 diverges at ANY coupling (even λ=0.3), 161 pts/η0.225 converges for d5a. Validated d5a recipe: **F10 coarse grid + linear 0.05** (E5: T100 λ=1 monotone). d11a at λ=1 converges under NO tested production combination (coarse/fine × linear/Anderson × η up to 0.45) — the dense d11a reference (0.997) was λ=0.3; λ=0.3 is the comparable production setting (probe interrupted, re-run cheap). **Dense ns=2 discriminator (partial, 36 iters before user stop): dense holds multi-cell d5a at λ=1 stable** (J flat to 0.5%, cons 1.1e-2) — its full symmetric axis (±130 THz incl. H modes) + DC interpolation avoid the near-DC feedback that production must mask. The SSE low-frequency cutoff (`sse_low_freq_cutoff_thz`, 2026-06-11) is the principled production-side handle: sub-cutoff modes excluded from the bubble and Σ not applied there (ballistic below) | **HIGH → understood + recipes validated** |

Performance: the solver is **memory-bandwidth-bound in `ring_contract`** (~99% of
iteration time, ~635 GFLOP/s aggregate vs ~1885 GFLOP/s node zgemm ceiling — ~34% of
compute peak, which is the roofline limit for these small block GEMMs). The largest
remaining optimization gap is that the **coupled-q (`nq>1`) path does not use the ring
thread pool**. Details and a deferred benchmark plan in Part 2.

---

## Part 1 — Physical correctness

### 1. `retarded_method="fft"`: PV part of Σ^R overscaled by π/dω — **BUG (confirmed)**

**Where.** `src/quatrex/core/fft_utils.py:15-47` (`hilbert_transform`) used at
`src/quatrex/phonon/sse_phonon_phonon.py:663`:

```python
sigma_retarded.data + 0.5j * hilbert_transform(delta, full_freqs)
```

with the anti-Hermitian half added in `core/scba.py:834`
(`sigma_retarded += 0.5*(sigma_greater - sigma_lesser)`).

**Exact relation.** With Δ = Σ^> − Σ^<,

    Σ^R(ω) = (i/2π) ∫ dω′ Δ(ω′)/(ω − ω′ + i0)
           = ½ Δ(ω) + (i/2π) PV ∫ dω′ Δ(ω′)/(ω − ω′).

`hilbert_transform` returns the **raw kernel sum** Σ_k Δ_k/(ω_n − ω_k ± dω/2) —
no dω quadrature weight — so `0.5j * H_raw` is used where `(i/2π)·dω·H_raw` is
required: an overscale of **π/dω**, and dimensionally `[Δ]/THz` instead of `[Δ]`.
The three-convolution structure itself (antisymmetric PV kernel with exact diagonal
cancellation + bosonic mirror term) is *correct in shape*.

**Evidence** (`audit_hilbert.py`; ground truth = θ(t)·Δ(t) in the time domain on a
dense symmetric axis):

```
ne=121, dω=0.0758 THz : π/dω = 41.5   measured |Re fft-path|/|Re exact| = 43.5
ne=121, dω=0.0207 THz : π/dω = 152.1  measured                          = 159.6
rescaled by dω/π: rel.err(Re) = 0.102  (shape correct; residual = half-step
                                        kernel bias + ω0 offset in mirror term)
Im part (the ½Δ half): rel.err ≤ 1e-14 (machine-exact)
|Re/Im| of the exact Σ^R ≈ 1.0 → the PV part is NOT small; with "fft" the real
part of Σ^R is ~40–160× too large on production-like grids.
```

**Impact.** All production runs set `retarded_method="half"`
(`phonon/scripts/prod/write_config.py:210`, default `"half"`; verified in every
generated `quatrex_config.toml`) → **no production result is affected**.
But `src/quatrex/core/config.py:739` defaults to `"fft"`, so any user running with
defaults gets a wildly overscaled Re Σ^R (enormous spurious frequency
renormalisation). This plausibly explains historical instabilities attributed to
the "fft" mode.

Note the same `hilbert_transform` is shared with
`coulomb_screening/polarization.py:158` — there the result *is* multiplied by the
quadrature-carrying `prefactor/2`, i.e. the electron call sites supply the missing
factor externally. The phonon call site does not.

**FIX (applied).** `core/fft_utils.hilbert_transform` is now the *standard*
(1/π)-normalized PV transform including the dω quadrature weight, with
exact cell-integrated log PV weights ln((j+½)/(j−½)) (removes the ~10%
half-step bias as well), the correct 2ω₀ mirror denominator, and an ω=0
double-count guard. Re-validation (`audit_hilbert.py`): |Re|/|Re exact| =
0.990, rel.err 1.3% (edge truncation of the test function), Im machine-exact,
grid-independent. The phonon call site `0.5j * hilbert_transform(...)` is now
exactly the textbook relation and needed no change; the polarization call
site was simplified to the same `0.5j` form (its old
`-(prefactor/2)*kvol == i·dE/2π` external weight was correct and is preserved
by construction).

**Literature.** The `"half"` approximation (neglecting Re Σ^R) is exactly what
Luisier uses: PRB **86**, 245407 (2012) neglects the principal-value part of the
retarded scattering self-energy and conserves current; our production results
therefore follow the established convention.

### 2. DC handling — OK

The production grids include ω=0 (verified: `phonon_energies.npy` starts at 0).
At `sse_phonon_phonon.py:442-444` G^≷(0) is zeroed before the bubble; lead Bose
factors are `where`-regularised at ω=0 (`phonon/solver.py:133-136`). Spot-check #2
bounds the bubble effect of zeroing the single DC point at **≤2.9%** even when the
DC value is artificially set to neighbour magnitude (in reality G(0) carries no
spectral weight after OBC). Cleaner alternative for the future: offset grid
(start at dω/2), which also removes the ω₀-related half-step bias in the Hilbert
mirror term.

### 3. Bubble FFT fold — verified exact; one open question for coupled-q

`audit_bubble_fold.py` replicates the solver's pad→FFT→index-reverse→3-term
product→IFFT pipeline (`sse_phonon_phonon.py:445-452, 502-518, 655`) on scalars and
compares against a direct O(N²) full-axis convolution with the bosonic continuation
G^<(−ω)=G^>(ω):

```
rel.err Σ^< fold vs direct = 6.1e-16     rel.err Σ^> = 7.2e-16   (machine-exact)
```

The 4th product (rev·rev) is correctly omitted (it lives entirely at negative
output frequencies; its hypothetical inclusion would inject ~0.3% wrap-around
aliasing). The prefactor `0.5j·ħ·dω/2π` (`phonon/units.py:55`) matches the dense
reference bit-exactly and is anchored empirically by the phono3py linewidth
validation (10–15% mode-averaged agreement; ÷4 and ×4 alternatives excluded by an
order of magnitude — results.tex F28).

**3b. Coupled-q vertex pairing — RESOLVED: real bug, fixed.**
`audit_qfold_trs.py` settles this in three steps:

1. *Continuation*: in the ω² Dyson representation, G^R(q,−ω) = G^A(q,ω) per q
   for **any Hermitian D**, so the same-q/no-transpose fold of the equilibrium
   G^≷ is exact (1.5e-16) — the fold itself was never the problem, and the
   3-term FFT fold equals the direct full-axis convolution to 4e-16.
2. *Ground truth*: the same toy built as one periodic real-space **supercell**
   (no q anywhere) and Bloch-projected disagrees with the q-space pairing by
   **74%** (and the output violates the physical TRS property
   Σ(−q) = Σ(q)^T).
3. *Variant scan*: conjugating the **left** vertex —
   Φ̃(q′,q−q′)\* paired with Φ̃(q−q′,q′) — matches the supercell to **3e-15**
   with the exact mesh-normalization scale (α = nq) and restores
   Σ(−q)^T = Σ(q) (6e-15). No other variant comes close.

**FIX (applied)** in both `src/quatrex/phonon/sse_phonon_phonon.py` (coupled-q
task list, `xp.conj(pl)`) and the dense `phonon/solver/se_finite.py`
(`np.conj(pl)` in the task builder). At Γ the vertices are real ⇒ all Γ-only
results (CNT, SiNW, SrTiO₃) are byte-unaffected. **All previous coupled-q film
numbers (incl. the dense results in results.tex and the 45%-vs-Guo gap) were
computed with the wrong pairing and are superseded by the corrected film
sweep now queued in the production pipeline.**

### 4. Meir-Wingreen heat current — OK

`qttools/greens_function_solver/rgf.py:383-431`: interface current
Tr[Σ̃^> G^< − G^> Σ̃^<] with re-injected tilde self-energies; lead currents from the
OBC Σ^≷ with the documented sign giving positive left→right flow (the
multi-batch lead-current bug was fixed earlier this cycle and is covered by the
inline comment). `core/scba.py:594-622` weights by |ω| (≡ ω on the positive grid)
and sums transverse q **before** testing conservation — correct, since 3-phonon
scattering exchanges energy between q channels and only the q-summed energy
current is conserved. Lead Σ^< = iΓn, Σ^> = iΓ(n+1) (`phonon/solver.py:172-178`)
together with the bilinear current trace forms a self-consistent sign set; the
realized observables are anchored by (i) ballistic G matching the dense reference
and literature values, (ii) heat conservation at 1e-4–1e-3 at the SCBA fixed
point. (Standard reference for the phonon MW form: Wang, Wang & Lü, Eur. Phys. J. B
62, 381 (2008).)

### 5. Zero-mode projector floor — OK with a documented physics choice

`sse_phonon_phonon.py:49-74`: Q = I − VV† over modes with ω < 0.1 THz of the cell
dynamical matrix, applied two-sided to Σ^≷ (Hermiticity-preserving). The docstring
*explicitly intends* to catch the d5a axial twist (0.0075 THz) along with the 3
rigid translations. Defensible: the projected modes lie far below both the energy
grid resolution (dω ≥ 0.34 THz) and the broadening η (≥0.11 THz), i.e. they are
spectrally unresolvable; keeping their (numerically garbage) contribution to Σ is
what destabilized the raw d5a SCBA. Caveat to keep in mind: this removes the
twist channel's scattering *by construction*, so d5a ratios should be quoted as
"twist-projected".

### 6. Dyson + OBC — Dyson OK; Sancho-Rubio replaced (soft-mode divergence)

`phonon/solver.py:236-251`: [(ω+iη)² − D − Σ^R]G = I in THz². (ω+iη)² = ω² + 2iηω
− η² gives the standard frequency-proportional phonon broadening (cf. Mingo, PRB
74, 125402 (2006) and Luisier 2012, who use ω²+iδ / (ω+iη)² variants). Contact
Σ_00 = M_10 g_00 M_01, Γ = i(Σ−Σ†) — canonical.

**OBC finding (post-audit): the Sancho-Rubio fixed point diverged on the d5a
soft twist mode** ("Surface Green's function did not converge" in every d5a
log; both contacts) and **NaN'd the entire d5a study** (all ballistic
final_heat = NaN, anharmonic NaN/garbage). **FIX (applied):**
`write_config.py` now defaults `[phonon.obc] algorithm = "spectral"` with
`nevp_solver = "full"` (the robust dense linearized NEVP — OBC is <0.1% of
runtime, so no contour tuning is worth it). Validated on d5a T30 ballistic:
zero warnings, clean heat [0.886, 0.860, 0.886]. The d5a study is rerunning
with the spectral OBC (broken outputs archived in `sinw_d5a/sr_broken/`).
All other studies show zero SR warnings (unaffected).

### 7. SCP cubic tadpole — OK

`static_self_energy.py:48-95`: ⟨uu⟩ = (dω/2π)Σ_ω iG^< (equal-time correlator from
the lesser GF), source s_a = ½Φ_acd⟨u_c u_d⟩, mean displacement ⟨w⟩ = −Φ⁺_eff s
(pseudo-inverse, eigenvalue floor rel. 1e-3·max ω²), Σ_T = Φ:⟨w⟩, Hermitized.
This is the standard static cubic tadpole (first-order ⟨u⟩ shift; cf. the tadpole
extension discussed by Tadano & Tsuneyuki, JPSJ 87, 041015 (2018)). The eigenvalue
floor excludes modes below ~0.03·ω_max from *responding* to the source — again a
deliberate soft-mode regularisation; document when quoting SrTiO₃ numbers.

### 8. Causality — OK (via #1)

With `"half"`, Σ^R is purely anti-Hermitian with Im tied to Σ^≷ (machine-exact in
spot-check #1) — linewidths are positive wherever Σ^> − Σ^< has the physical sign,
which the phono3py comparison confirms. The `"fft"` bug affects only Re Σ^R
(spurious renormalisation), not the sign of the broadening.

---

## Part 2 — Performance

### Measured profile (this cycle, EPYC 7742 2×64c node)

* `ring_contract` = **~99% of SCBA iteration time**; three zgemms per
  (ω,τ)-batched fold term; transpose-free kernel (verified ≤1e-15 vs einsum).
* Aggregate throughput **~635 GFLOP/s vs ~1885 GFLOP/s zgemm node ceiling (~34%)**:
  memory-bandwidth-bound at production block sizes (BS 36–135); this is the
  roofline limit, not an inefficiency — larger BS or multi-node scaling is the
  way up, not more threads.
* Thread-pool over the ω/τ batch (`QUATREX_PHPH_RING_THREADS`, single-thread BLAS):
  15–42× at 16–32 threads. Outer-loop pooling (one dispatch per bubble) was
  essential — per-call pooling lost ~40% to the GIL.
* FLOP model: bubble ∝ n_quads · 6 folds · 3 gemms · n_fft · BS⁴ → d11a (BS=135)
  ≈ 197× cnt33 (BS=36) per iteration, matching observed wall-times.

### Remaining headroom (ranked)

1. **`nq>1` coupled-q path was serial — now pooled** (same τ-chunk pattern as
   the nq==1 restructure; **verified bit-identical**, 0.0 max diff on
   heat/spectrum). Caveat discovered while timing: for the *film's tiny
   per-slab blocks* (6×6 gemms) the per-call Python overhead dominates and
   the GIL makes the pool *slower* (89→135 s/iter measured on a loaded
   node) — the pool pays off only in the large-block regime (Γ-only 36–135
   DOF, 15–42×). The film study therefore scales on **MPI q×stack ranks**
   instead (qcs 27 × stack 4 = 108 ranks; validated: all-reduced heat
   matches the stack=1 run to 1e-16). For the tiny-block regime the real
   future win is batching the (q-pair × quad) tasks into strided-batched
   gemms — noted, not implemented.
2. **τ-buffer reuse**: `_ensure_tau_buffers` caches the six DSDB buffers — good;
   but each iteration re-pads and re-FFTs the full G^≷ (unavoidable) and allocates
   per-(I,J) accumulators in Python (minor).
3. **Anderson memory**: depth m keeps 2m full nnz-vector copies
   (≈ 2m · ne_local · nnz · 16 B). For d11a L3 (nnz ≈ 1.1e6·BS² scale) this is the
   dominant SCBA-side allocation — keep depth ≤5 (current production setting) and
   prefer linear mixing on memory-tight multi-node runs.
4. **OBC per iteration**: recomputed each SCBA iteration because the system matrix
   carries Σ^R — necessary, and only on the two edge block-ranks; cost is noise
   (<0.1%). No redundant recompute found.
5. **q load imbalance**: external-q split `rank*nq//size` handles nq % size ≠ 0
   with ±1 imbalance — fine; the *internal* q′ loop is replicated on every q-rank
   (by design, memory-for-communication trade).
6. **GPU path**: the ring pool activates only for `xp is np`; a CuPy port should
   instead batch the fold over ω·τ in single large gemms (cublas strided-batched),
   which removes the need for the thread pool entirely.

### Multi-node expectations (not testable now)

* **stack (energy) axis**: embarrassingly parallel in the Dyson/ring stage; the two
  `dtranspose` (nnz↔stack) calls per iteration are the all-to-all cost, volume
  ≈ 2 · n_fft · nnz · 16 B per direction. Scales until all-to-all dominates
  (~16-32 ranks/node observed fine; inter-node depends on fabric).
* **block axis**: halo width-1 neighbour exchange only — cheap, but SerinV needs
  ≥2 blocks/rank (hang at 1 block/rank, known) and num_blocks limits parallelism.
* **q axis**: one all-reduce of Σ per iteration — scales to nq ranks trivially;
  best multi-node axis for the film.

### Deferred benchmark plan (run after `FINAL_STATUS.txt` appears)

```bash
# 1. nq>1 pool prototype A/B (single node, idle):
OMP_NUM_THREADS=1 QUATREX_PHPH_RING_THREADS={1,16,32} \
  mpirun -np 4 python phonon/scripts/prod/run.py --work /tmp/prod_work/sifilm_L3_nk5 ... QX_MAXIT=3
# 2. stack-axis strong scaling sweep (np = 8,16,32,64) on cnt80 L2, 3 iters
# 3. ring kernel roofline re-measurement at BS=135 (d11a) with likwid/perf stat
#    (FLOPS_DP + MEM groups) to confirm the bandwidth-bound verdict at large BS
```

---

## References

* M. Luisier, *Atomistic modeling of anharmonic phonon-phonon scattering in
  nanowires*, Phys. Rev. B **86**, 245407 (2012). doi:10.1103/PhysRevB.86.245407
  — SCBA formulation; neglects Re Σ^R (our `"half"`); current-conservation as
  convergence check.
* Y. Guo, M. Bescond, Z. Zhang, et al., *Quantum mechanical modeling of anharmonic
  phonon-phonon scattering in nanostructures*, Phys. Rev. B **102**, 195412 (2020).
  doi:10.1103/PhysRevB.102.195412 — film benchmark; prefactor-convention dispute.
* J.-S. Wang, J. Wang, J. T. Lü, *Quantum thermal transport in nanostructures*,
  Eur. Phys. J. B **62**, 381 (2008) — phonon NEGF/Meir-Wingreen conventions.
* N. Mingo, *Anharmonic phonon flow through molecular-sized junctions*, Phys.
  Rev. B **74**, 125402 (2006) — ω² Dyson form, phonon OBC.
* T. Tadano, S. Tsuneyuki, J. Phys. Soc. Jpn. **87**, 041015 (2018) — SCP theory,
  tadpole extension, SrTiO₃.
* B. Latour, N. Shulumba, A. J. Minnich, Phys. Rev. B **96**, 104310 (2017);
  Z. Tian et al., Phys. Rev. B **86**, 235304 (2012) — Si/Ge interface AGF
  references (validation anchors).

## Addendum (2026-06-11): bubble energy-balance verification

Per-iteration diagnostic implemented (`SCBA._phonon_bubble_energy_balance`,
printed + saved as `iter_bubble_balance`): P_in = Σ ħω Tr[Σ^< G^>] vs
P_out = Σ ħω Tr[Σ^> G^<] with the transpose pairing — the Φ-derivable
conserving identity (Lü & Wang arXiv:0704.0723 App. B). Findings on cnt33 L2:

1. **Baseline imbalance ~1e-5 relative** (absolute steady ~4e-3). Excluded:
   alignment step, zero-mode projector, DC handling, contact symmetry.
2. **The nonequilibrium G asymmetry (1.8% at ΔT=10 K) is physical** (the
   current lives in G^<_ij − G^<_ji) — and exposed that the bosonic fold
   used the no-transpose shortcut, exact only for the equilibrium part.
   **FIXED**: the fold now uses the exact `G^<_ij(q,−ω) = G^>_ji(−q,ω)`
   (nnz transpose permutation + q-negation; falls back with a warning when
   the nnz slice is stack-split). Physical shift in J ≈ 1.5%.
   (The dense reference avoids the question entirely: full symmetric axis.)
3. **The remaining 1e-5 floor is the device-vertex class**: stored
   Φ(I,K1,K2) is exact under contracted-leg exchange (1e-16) but violates
   external↔contracted exchange structurally (41% on interior blocks —
   the Γ-image-folding pins the external leg). S3-symmetrizing the vertex
   (`QX_SYMMETRIZE_FC3=1` experiment hook in run.py) drops the imbalance
   ~100× to ~1e-7, confirming the chain. Whether to adopt the symmetrized
   (exactly conserving) vertex is a physics decision: it changes Σ slightly
   and would need linewidth re-validation.

### Balance-floor closure (2026-06-11)

The remaining finite P_in/P_out difference was hunted to ground with five
controlled experiments (cnt33 L2/T300, linear broadening):

| hypothesis | experiment | verdict |
|---|---|---|
| window truncation (η-tails beyond f_max) | `baltest_wide` 110 THz/241pt vs 55 THz | residual identical to 4 digits — **excluded** |
| skew-Hermitian Σ projection before the check | `QX_BALANCE_PRE_SYM=1` raw-vs-projected | projection *improves* (6.0→3.3e-6) — **excluded** |
| zero-mode projector (Σ = P·F[G]·P vs full outer G) | `baltest_noproj` | same floor (4.0e-6 it0) — **excluded** |
| ω=0 inner-only masking of the SSE inputs | toy replica, inner-zero-only | exact 2.9e-16 (the two removed relabeling partners cancel pairwise) — **excluded** |
| floating-point accumulation | toy replica float64 vs float128, FFT path, growing size | 1e-15/1e-19, scales with eps — **excluded** |

The standalone replica (`phonon/scripts/verify/balance_roundoff_test.py`)
reproduces the production discrete pipeline exactly (zero-based grid,
n_fft=2ne−1, negative frequencies at the TOP of the FFT buffer,
transposed bosonic fold, transpose-paired trace) and proves the identity
holds to machine precision **iff the vertex carries full S3 symmetry**:
random Φ without S3 violates at O(0.25); symmetrizing only single leg
2-cycles (k↔l, i↔k, i↔l) still violates at O(0.2–0.6); only the full
S3 average conserves (1e-16). Left/right factor consistency alone is NOT
sufficient.

Conclusion: the production floor (~constant absolute |P_in−P_out| ≈ 1e-2
on P≈7e2, i.e. 1e-6–1e-5 relative) is the **structural S3 violation of the
image-folded device vertex** — Γ-image folding pins the external leg, so
the *effective* per-block-pair vertex is not S3 even when the raw FC3
blocks are S3-averaged first. This is why the full-run `sym_full_anh`
(QX_SYMMETRIZE_FC3=1, 24 iters) still floors at 4.5e-6: the in-place
block average cannot symmetrize the post-fold object. The hook's early-
iteration ~100× gain reflects the removable raw-block part only.

Status: NOT a code bug — the convolution/fold/trace machinery is exactly
conserving. The residual measures the weight of the asymmetric image
tails of the open-device embedding. An exactly-conserving variant would
need a leg-symmetric vertex construction (e.g. plain truncation of the
supercell FC3 to the device region — a sub-tensor restricted to the same
index set on all three legs is automatically S3 — instead of folding
images onto contracted legs only); that changes Σ physically and needs
linewidth re-validation before adoption.

### Balance floor FULLY closed: exact conservation achieved (2026-06-11, supersedes part of the section above)

The "structural, unfixable in place" conclusion above was wrong in one
detail: the QX_SYMMETRIZE_FC3=1 hook was itself buggy — the two 3-cycle
key permutations must pair with their INVERSE as the numpy axis transpose
(key-perm (1,2,0) ↔ transpose (2,0,1)), and the average must be /6 with
missing orbit members as zero, not /n_present. With the corrected exact
projection (hook mode =2) the device vertex is S3 to 2.3e-16 (it changes
the vertex by ~53% on cnt33_L2 — strong supercell aliasing).

Production decomposition of the floor (cnt33 L2/T300, 121pt, linear):
  raw vertex + projector            : 6.0e-6   (baltest_presym)
  exact-S3 vertex + projector       : 2.7e-6   (baltest_s3fix)
  exact-S3 vertex + NO projector    : **1.5e-15**  (baltest_conserving)
i.e. THREE stacked causes: (1) vertex S3 violation from Γ-image folding,
(2) the zero-mode projector applied two-sided to the Σ OUTPUT while the
trace pairs with the full G (toy: a 1-of-6-direction projector violates
at 0.46), (3) the post-SSE skew-Hermitian Σ projection, whose own
contribution (~3e-6..1e-5 per iterate) is visible in the conserving run
as the pre-sym (1e-15) vs post-sym (3e-6) gap.

Literature verification (Scite): Lü & Wang, PRB 76, 165418 (2007),
App. B: SCBA conserves the energy current; the proof manipulates the
cubic coefficients via their permutation symmetry. Wang, Wang & Lü,
EPJ B 62, 381 (2008): the cubic vertex "is symmetric with respect to
simultaneous permutations of the triplet". Luisier, PRB 86, 245407
(2012): the device vertex is dV(3)/du^3 of the explicitly-described
device region — i.e. PLAIN TRUNCATION of a derivative tensor, totally
symmetric by construction. An image-folded vertex is not the third
derivative of any device potential -> not Phi-derivable -> non-conserving.
Plain truncation is the literature-consistent construction.

Adoption caveats: the S3 projection (or a plain-truncation input rebuild)
changes Sigma physically (53% vertex change on the L2 build) — linewidth
validation vs phono3py must be re-run before making it the production
default. The zero-mode projector must then stay OFF (the SSE low-freq
cutoff is the conserving replacement: omega-diagonal masking is
inner/outer consistent and verified harmless in the replica). The
skew-Hermitian Sigma symmetrization should be re-examined — on a
conserving vertex the raw SSE output is already the better object.

### Root cause found: the ASR projection, not the folding (2026-06-12 — FINAL, supersedes both sections above)

Systematic reconstruction from the saved hiphive FCP (cnt33, (1,1,5)
supercell, `verify/plain_truncation_vertex.py`) shows:

1. The minimum-image device folding is EXACT and S3-preserving for cnt33:
   folded (1,1,3) blocks at `enforce_asr=False` have S3 violation 0.0, and
   the plain-truncation rebuild from the (1,1,5) supercell (no wrap
   possible, fc3 cutoff 2.5 A = 1.02 cells < half-window 2) matches them
   to 7.7e-16. The "image-folding pins the external leg" hypothesis was
   WRONG — there is no aliasing at this cutoff/supercell combination.
2. The S3 violation (worst 1.276) is caused ENTIRELY by
   `enforce_asr_fc3_matrices` (separable.py): it projects (I-PP^T) on the
   two SUPERCELL legs but never the compact external leg (leg-asymmetric),
   AND PP^T spans the full per-atom-class q=0 subspace (36 dims) instead
   of the 3 global translation vectors the ASR actually requires.
3. The projection was unnecessary for cnt33: the raw hiphive FC3
   satisfies the leg sum rule to 5.7e-14 (machine precision; hiphive
   enforces translational invariance in the fit). Yet the projection
   changes the tensor by 136% (rel L1).

Physics validation (`verify/cnt33_linewidth_vertex_check.py`, single-shot
NEGF bubble vs phono3py on the same FC2/FC3, q=(0,0,1/4), mesh 8, eta 0.2):
the RAW (= plain-truncation) vertex agrees with phono3py at order unity
across all bands with real phase space (R ~ 0.3-1.5; high-omega flat
bands where gamma_p3p < 1e-3 show the expected eta-floor artifact). The
ASR-PROJECTED vertex (production default until today) suppresses
linewidths UNIFORMLY ~4-5x (median R = 0.18): **all previous cnt33/SiNW
production anharmonic numbers underestimated 3-phonon scattering by
roughly 4-5x.**

Conservation with the plain vertex (`baltest_plain`, no symmetrization
hook, no zero-mode projector): pre-sym bubble balance 2.5e-16..1.6e-15
every iteration — exact, from the natural vertex.

ADOPTION: (a) cnt33 (and any hiphive-fit FC3 with exact translational
invariance): set `enforce_asr=False` — the raw vertex is S3, conserving,
and phono3py-consistent. The shipped folded blocks at asr=False are
identical to plain truncation. (b) d5a/SiNW: the raw VASP-fit FC3 has a
genuine leg-ASR residual (~0.8, F9/F16) and needs SOME regularization for
SCBA stability — the current projector must be replaced by a MINIMAL,
LEG-SYMMETRIC one: project out only the 3 global-translation vectors, on
all three legs (S3-preserving by construction), optionally alternating
with S3 projection to convergence. (c) The zero-mode projector stays off
for conserving runs (use the SSE low-frequency cutoff instead).
