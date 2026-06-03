# Phonon SCBA — lab notebook (diagnosis & verification)

> Working log. **Substantive write-up + figures go to LaTeX in `document/`** (per user
> request). This file is the running log: hypotheses, commands, run IDs, results, verdicts.

> **Commit policy (user request, 2026-06-02):** git commits are authored by the user
> (`pfischill`). Do **not** add `Co-Authored-By: Claude …` trailers, nor any "Generated with
> Claude Code" line, to commit messages. Plain messages only. Commit/push only when asked.

## Goal (user request, 2026-05-29)
(a) Determine whether the d5a SiNW SCBA non-convergence / "soft mode" in the η/λ sweep is
genuine anharmonicity (SCBA breakdown) or a code / hiphive-FC bug.
(b) Verify theory + solvers (incl. quatrex tridiagonal RGF vs dense Inv) on a *working*
structure; analyze other structures with cutoff-tolerance sweeps.
(c) Reproduce an experimental value.
Constraints: **avoid transverse periodicity** (use `transmission_finite`, not
`transmission_q`); VASP available at the YAML path; long runs OK but monitored + verified
short first.

## Environment
- python: spack `python-3.14.3`, numpy 2.4.4; phonopy, h5py, mpi4py present; **no cupy**
  (quatrex runs CPU/numpy backend); **no SLURM** (run directly on node); QE `pw.x` at
  `/usr/sepp/bin/pw.x`; VASP built at the YAML `potcar_dir` path.
- Bulk-Si phono3py **forces** exist (`phonon/reaps_old/si_primitive_work/phono3py_params.yaml`)
  → FC2/FC3 re-fit with no new DFT. SiNW d5a/d11a/d17a FC3 already computed.

## Literature anchors
- Luisier 2012 PRB 86 245407 — NEGF-SCBA 3-phonon Si NWs; **current conservation** = convergence
  criterion; calibrated to experiment; anharmonic current always **below** ballistic.
- Lee–Bescond–Logoteta 2018 PRB 97 205447 — LOA / vertex (λ) rescaling + Padé. Bare LOA series
  **diverges at 300 K** → Padé needed; rescaled scheme conserving every iteration (SCBA only at
  self-consistency).
- Lee–Jean–Sanvito 2009 PRB 79 085120 — SCBA **breaks down above a critical coupling**; multiple
  fixed points. (λ=0.9 failure is consistent with this regime.)
- Xu–Wang–Duan 2008 PRB 78 224303 — ballistic Landauer value is the upper bound.

## Findings

### F1 — Two physical impossibilities in the existing d5a η/λ sweeps
(`phonon/scripts/out/eta_lambda_diag/*/info.json`)
- `G_anh > G_ball` in **every** run (1.15–14×). Anharmonic 3-phonon scattering is purely
  resistive → must give `G_anh ≤ G_ball`. So the sweep numbers are non-physical.
- `G_ball` swings **~180×** across the η-sweep (must be η-independent).

### F2 — The Caroli/leads ballistic path is CORRECT (clean chain)
`phonon/scripts/verify/test_ballistic_eta.py`: monatomic chain (single channel, analytic
T=1 in band). `G_ball → G_analytic` exactly as η→0 (ratio 1.0000 at η_f=0.05); even at the
absurd η_f=9 only 25% low. **Ruling: the solver's ballistic normalization is not buggy.**

### F3 — d5a ballistic transmission is crushed to ~0 at the diagnostic's η (THE problem)
`phonon/scripts/verify/test_d5a_ballistic_eta.py` (FC3 off, vertex_scale=0, n_slabs=2,
grid 0.01/18/81 → dw=0.222 THz):

| η_factor | η_w (THz) | max T(ω) | T@f0 | G_ball |
|---|---|---|---|---|
| 9.0  | 2.0   | 0.001 | 0.000 | 5.49e3 |
| 6.75 | 1.5   | 0.003 | 0.000 | 3.00e4 |
| 4.5  | 1.0   | 0.023 | 0.003 | 2.26e5 |
| 3.0  | 0.67  | 0.104 | 0.019 | 9.99e5 |
| 1.0  | 0.22  | 1.127 | 0.609 | 9.60e6 |
| 0.3  | 0.067 | 2.772 | 2.494 | 2.78e7 |

A 63-DOF wire must transmit O(channels). At every η used in the convergence diagnostics
(η_factor 3–9) the **coherent transmission is essentially zero**, so `G_ball` is noise and
`G_anh > G_ball` (F1) is the artifact of a crushed ballistic baseline. The transmission only
recovers as η→0 — exactly where the SCBA was reported to diverge. **Interpretation:** the
diagnostics trade a low-ω soft/near-zero mode (which destabilizes SCBA at small η) for a
huge η that destroys the coherent physics. The real culprit is the low-ω mode, not the
SCBA closure per se. → next: confirm with the d5a dispersion (F4).

### F4 — d5a FC2 is mechanically stable; near-zero TWIST mode is the "soft mode"
`phonon/scripts/verify/dispersion_check.py`: **no imaginary modes** (global min freq
= -0.000 THz). At Γ: 3 translations + a 4th near-zero **rigid-twist** mode at **0.027 THz**,
then a gap to 2.38 THz. So the "soft mode" is a real quasi-Goldstone torsion of the
H-passivated wire, not an FC defect / imaginary mode. Channel counts: ~4 below 2 THz
(3 acoustic + twist), ~11 below 5 THz, ~31 below 15 THz.

### F5 — Ballistic transmission CONVERGES to the channel count as η→0 (code correct)
`phonon/scripts/verify/test_d5a_ballistic_lowsmall_eta.py`: at η_f→0.01, T(ω) tracks the
mode-counting band number (T≈4 at 0.5–1 THz = 3 acoustic + twist; ≈1 where one band sits;
maxT→4.6). `G_ball` converges to ~5–6e7 W/m²K. **The earlier 180× swing was entirely the
huge η (η_f 3–9 → η_w 0.7–2 THz) crushing the low-ω channels that carry the wire's heat.**

### Interpretation of part (a) so far
- NOT a transmission/leads code bug (verified on chain + d5a, F2/F5).
- NOT an FC2 instability (no imaginary modes, F4).
- The η/λ diagnostic methodology is **flawed**: it ran at η_w ≈ 0.7–2 THz where the coherent
  baseline is ~0, making `G_anh > G_ball` an artifact (F1/F3). The "safe η" was chosen so
  large it destroyed the physics.
- The genuine difficulty is the 0.027 THz twist mode: n_B(0.027 THz, 300 K) ≈ 220, so its
  Bose-enhanced low-ω weight dominates the bubble G^< and destabilizes SCBA at *physical* η
  — which is why large η was used. **Open question (testing now):** does SCBA converge at
  small η with proper zero-mode projection, or does it need LOA-Padé / explicit twist
  handling? `build_dynamical_zero_mode_projector` (threshold_rel=1e-4) should project the
  twist out of Σ, but it still lives in G.

## ⚙️ Run recipe (IMPORTANT — always use these env vars)
This is a 256-core node. The phph self-energy spawns one worker per visible core, and each
worker spawns BLAS/OMP threads → thread explosion (`libgomp: Thread creation failed:
Resource temporarily unavailable`) that silently kills SCBA runs. **Always launch with
single-threaded BLAS and a bounded worker pool:**
```
OMP_NUM_THREADS=1 OPENMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 QUATREX_PHPH_THREADS=128 \
  python -u <script>
```
`QUATREX_PHPH_THREADS` (read by `phonon/solver/se_finite.py:_default_n_threads`) caps the
(I,J) worker pool; with single-threaded BLAS, 128 workers is safe. Also: do **not** detach
with `nohup &` inside a backgrounded shell (it gets reaped); run the python directly as the
tracked background command. Avoid `| grep | tee` wrappers — they swallow the fatal stderr
when the process is killed (write to a file and filter on read instead).

## Run log
- 2026-05-29: F2 chain PASS; F3/F5 d5a ballistic (η→0 converges to channel count);
  F4 dispersion (no imaginary modes; 0.027 THz twist). Scripts in `phonon/scripts/verify/`.
- 2026-05-29: launching small-η SCBA (η_f=0.5, λ=0.3) on d5a to test convergence at a
  physical η; running fast RGF-vs-Inv + bubble-kernel verifications meanwhile.
- 2026-05-29: RGF≡dense (1e-16) PASS; bubble/SSE 28 PASS; bulk-Si κ 110.7→115.2 W/mK;
  FC-quality d5a (FC2 pristine, FC3 leg-ASR 0.80 expected-but-unprojected); physical-η SCBA
  λ=0.3 CONVERGED ratio 0.942 (<1, physical), λ=0.6 stiffens (strong-coupling onset).
  Diagnosis + verification written to `document/src/appendices/convergence_diagnosis.tex`
  (wired into report.tex; removed siunitx since the class lacks it; cites resolve to existing
  bib keys). NB: local TeX install is broken (SEPP/texlive-2011) so build must be done
  elsewhere; the .tex is brace/env-balanced and uses only macros already in the document.
- 2026-05-29: running LOA-Padé small-λ sweep at physical η (`d5a_loa_pade.py`) to extrapolate
  G_anh(λ²→1) the correct way (Bescond). Result pending.

### F11 — Other finite structures (dispersion): d11a clean, d17a FC2 broken
`phonon/scripts/verify/disp_d11_d17.py`:
- **d11a**: stable FC2, no imaginary modes; lowest non-translational mode at **1.07 THz**
  (vs d5a's 0.027 THz twist). **No near-zero soft mode → well-conditioned for SCBA.** This is
  the natural "structure where it actually runs."
- **d17a**: **FC2 catastrophically unstable — 86 imaginary modes at Γ, 3688 overall, global
  min −34.3 THz.** Cause (confirmed in config): **d17a `n_structures: 6` +
  `fit_method: least-squares`** (no regularization) for an 84-atom sc4 supercell → badly
  under-determined → overfit/unstable. Contrast **d11a `n_structures: 90`** (clean), d5a 44.
  **Fix = sow ~40+ more rattled configs, run VASP, refit with `ardr`/`rfe`**; not a solver
  issue. Unusable as-is.
→ For finite-structure SCBA + transport, use **d11a**, not d5a (soft mode) or d17a (broken).

### F12 — Distributed quatrex works and scales (part: multi-rank)
- Correctness: `tests/qttools/rgf_dist/test_rgf_dist.py` PASSES on `mpirun -np 3`
  (block-distributed RGFDist, block_comm_size=3).
- Stack(energy)-parallel scaling (`phonon/scripts/verify/dist_scaling.py`, N=1536, 128 energy
  pts): wall 1.55→0.86→0.55→0.28 s at np=1/2/4/8 = **5.5× at 8 ranks (68% efficiency)**.
  (Tiny problem N=576 gave 2.7× — overhead-bound; scaling improves with problem size.)
- Run with `mpirun -np N python ...`; comm via `qttools.comm.comm.configure(block_comm_size=…)`.
  block_comm_size must divide np. OpenMPI 5.0.10 + mpi4py + pytest-mpi (`--with-mpi`) present.

### F13 — Experiment comparison: bulk Si κ(T) curve vs Glassbrenner–Slack
`phonon/reaps/si_primitive_work` (19³ mesh, RTA, natural-isotope scattering):
T=200/300/400/500/600 K → κ = 179.6 / 108.7 / 78.7 / 62.0 / 51.3 W/mK.
Experimental single-crystal Si (Glassbrenner–Slack ~150 @300K): calc/exp ≈
0.68/0.73/0.80/0.82/0.84. **The κ(T) shape (high-T ~1/T) is reproduced; ~20–30% low in
magnitude (worse at low T) from the under-converged 2×2×2 FC3 supercell + isotope.** Honest
experiment-vs-data comparison; converging the FC3 supercell (3×3×3+) would close most of the gap.

### F14 — LOA λ-sweep at physical η (the correct version of the original diagnostic)
`phonon/scripts/out/d5a_loa_pade/` (η_w≈0.11 THz, n_slabs=1): small-λ SCBA converges cleanly,
G_anh/G_ball monotonic in λ²: 0.982(λ.1) 0.962(.2) 0.942(.3) 0.924(.4) 0.906(.5, 40-iter cap).
Extrapolation to λ²=1: **linear→ratio 0.60 (~40% reduction)**, consistent with Bescond's Ge-NW
δI≈46% @300K; **quadratic polynomial→1.99 (unphysical overshoot)** — confirming the bare LOA
power series doesn't converge at 300 K and a **Padé (rational) continuation is required**
(Bescond 2018), NOT a polynomial. This is the correct, conserving way to reach full coupling —
contrast the broken original diagnostic that inflated η instead.

### F15 — d11a SCBA confirms the well-conditioned structure (2nd structure)
`phonon/scripts/out/d11a_physical_eta/` (135 DOF/slab, physical η, n_slabs=1; ~3.9 h/point):
λ=0.3 **converged** (37 iters), G_ball=5.29e7, G_anh=5.27e7, **ratio 0.997**, conservation
2.7e-3; λ=0.6 ratio 0.983, conservation 1.4e-2, **no causality violations** — markedly better
behaved than d5a's λ=0.6 (which plateaued at resid 0.5 with sign violations). Confirms: the
no-soft-mode wire gives a stable SCBA to higher coupling and a smaller, physical anharmonic
reduction. (d11a is ~10× costlier than d5a — use distributed/energy-parallel for production.)

### F16 — FC3 ASR projection wired in; converts a divergent λ=1 SCBA into a convergent one
The actionable recommendation of F9 is now implemented as a real flag:
`transmission_finite(..., enforce_asr=True)` in `phonon/solver/dense.py` projects the
mass-weighted FC3 `M_stacked` onto the Γ-translation null space on **both** device legs
(`(I−PPᵀ) M_a (I−PPᵀ)`, via `phonon_inputs/separable.py:enforce_asr_fc3_matrices`) before
`build_device_fc3_blocks`. Exposed as `--enforce-asr/--no-enforce-asr` (default **on**) in
both `d5_cutoff_sweep.py` and `d5_transport_sweep.py` (cached separately from raw runs).
Decisive comparison (`phonon/scripts/verify/asr_compare.py`, d5a, **λ=1**, n_slabs=1,
η_w≈0.11 THz, grid (0.01,18,41); `phonon/scripts/out/asr_compare/d5a_compare.json`):

| FC3 | converged | n_iter | final resid | causality | G_ball | G_anh | G_anh/G_ball |
|-----|-----------|--------|-------------|-----------|--------|-------|--------------|
| **raw** (no ASR) | ✗ (div-guard abort) | 10 | 1.23 | **620 Γ sign-viol pts** | 2.153e7 | (diverged) | — |
| **projected** | ✓ | 30 | 8.8e-4 | **none** | 2.153e7 | 2.198e7 | 1.02 |

- Projection drops ‖M‖_F by **25.0%** (removes the leg-j/k ASR residual) and leaves the
  **ballistic baseline identical** (G_ball=2.153e7 both ways) — it touches only the vertex.
- **The robustness win is real**: at full coupling the raw FC3 SCBA *diverges* (residual
  blows up to ~200, persistent 620-pt Γ sign violations) and is aborted by the divergence
  guard; the projected FC3 SCBA *converges* monotonically with no causality violations and
  bounded max|Σ^R|≈136. This is exactly the F9 prediction.
- **Honest caveat**: the converged projected ratio is 1.02 (marginally > 1) at this
  single-cell, full-coupling point with ~1.9% residual heat-flow non-conservation. The
  strictly current-conserving G_anh ≤ G_ball regime is moderate λ (F10: λ=0.3→0.942; F14 LOA
  sweep monotone < 1) or the LOA-Padé continuation; at n_slabs=1/λ=1 the ~2% conservation
  error admits a small bound overshoot. Projection fixes *convergence*, not the residual
  single-cell conservation error — use moderate λ or more slabs for the conductance number.

### F17 — Transport-property sweeps for d5a and d11a (projected FC3)
With projection default-on, mapped transport observables on both usable wires.
- **Ballistic G(T), both wires** (`phonon/scripts/verify/ballistic_curves.py`, direct Caroli,
  no FC3/bubble — T(ω) computed once per (wire,length), integrated vs all T; fine grid
  701 pts to 140 THz, η_w≈0.06 THz). `ballistic_curves/ballistic.csv`:
  G_ball(T=200/300/400/500/600 K) = **d5a** 36.1/40.5/42.7/44.0/44.7 e6;
  **d11a** 50.3/60.3/65.4/68.3/70.1 e6 W/m²K. Both rise with T (thermal activation, saturating);
  d11a ≈1.5× d5a (135 vs 63 DOF/cell → maxT 4.49 vs 3.40). **Clean physics.**
  Length (T=300): d11a 60.3/41.1/22.8 e6 at L=1/2/4 — **finite-η coherence attenuation**
  (ℓ~v_g/η_w), NOT diffusive resistance; true Landauer plateau needs η→0. The
  `d5_transport_sweep.py --ballistic-only` path is NOT cheap (still builds FC3 vertex + one
  26-chunk bubble for d11a) → wrote the direct script instead.
- **Anharmonic G_anh(T), d5a** (`phonon/scripts/out/d5a_transport_proj/summary.csv`, λ=1,
  n_slabs=1, η_w≈0.22 THz, projected): ratio 0.975(200K)/1.019(300K)/1.094(400K) — within the
  single-cell λ=1 conservation band (cf. F16; grows with T as Bose↑). Clean reduction needs
  moderate λ (F10/F14) or >1 cell. **d11a anharmonic** uses the F15 converged points
  (λ=0.3→0.997, λ=0.6→0.983 @300K) — fresh d11a SCBA sweeps were infeasible on the shared node
  (~40 min per SCBA *iteration*: 9–26 ω-chunks × 2 workers when node memory is contended to ~28 GiB).
- **Cutoff convergence, BOTH wires** (finite-analysis Σ-level sweep, already computed:
  `phonon/reaps_old/hiphive_sinw100_d5a_vasp/cutoffs/cutoffs_sweep.csv` + d11a's under
  `configs/sinw/reaps/sinw100_d11a_vasp_sc4/cutoffs/`). FC3 magnitude-threshold Σ^< rel-diff:
  d5a 1e-2→11%, 1e-3→0.7%, 1e-4→0; **d11a needs tighter** (1e-2→127%, 1e-3→12%, 1e-4→2%) —
  wider wire has more small-but-significant vertex elements. **diag-G approx catastrophic**:
  d5a 5×, d11a 30× → the off-diagonal (inter-slab) G coherence is essential, must NOT be
  truncated (the dominant cutoff is the G-range, not vertex magnitude). Plot
  `cutoff_sse_d5_d11.pdf`. The distance-cutoff sweep inside the full SCBA
  (`d5_cutoff_sweep.py --enforce-asr`, n_slabs=2) is node-infeasible (~18 min/iter at 28 GiB
  free; ~7 h for one 3-point curve) — tooling wired + validated on single-cell, deferred to
  an uncontended node / distributed path.
- **Node reality**: throughput swings 9 min → >50 min per point as other users load the 503 GB
  node (governor drops to 2 workers at ~28 GiB free). All scoping reductions logged here (no
  silent caps). Plots: `document/fig/transport_sweeps/` + per-run `plots/`.

### F18 — Bottleneck hunt + d11a anharmonic re-validation (the "contention" was self-inflicted)
The earlier "node contention / d11a infeasible" verdict (F17) was **WRONG about the cause**.
The node was fine (503 GB, ~440 GB free). The real causes, found by `ps -eo … | awk '$2==uid'`
under `dangerouslyDisableSandbox`:
1. **A 116 GB orphan python** (`test_d5a_smalleta_scba.py`, 13.7 h old, never terminated from a
   prior session) was holding memory + cores. `kill -9` it.
2. **My own `QUATREX_PHPH_MEMORY_GB=40`** capped the bubble governor's budget to ~28 GiB →
   it split each bubble into ~26 ω-chunks (sequential). Setting it to ~250–320 (of the free
   RAM) → ≤3 chunks → **~8× faster** (d11a iter 40 min → 5 min).
**Second, deeper bottleneck (real quatrex limitation):** for `n_slabs=1` there is exactly ONE
(I,J) slab-pair, so the phph `ThreadPoolExecutor` (`se_finite.py`) has nothing to parallelize,
and the bubble's batched BLAS matmuls (`bubble.py:_bubble_contract_batched_matmul`) **do not
thread effectively** here — observed **nlwp=1 at 99.5% CPU** even with `OPENBLAS_NUM_THREADS=64`.
So a single large slab (d11a, 135 DOF, 540-dim FC3 legs) is **single-core-bound at ~10–25 min
per SCBA iteration** regardless of thread env; counter-intuitively, *oversubscribing*
(OMP=OPENBLAS=64, nlwp≈192) was faster (~10 min/iter) than clean single-thread (~25 min). The
OBC precompute + ballistic-transmission loop are also single-threaded Python freq-loops (~5–13 min
setup for d11a). **Fixes/levers:** (a) kill stale procs + raise `QUATREX_PHPH_MEMORY_GB`;
(b) use `n_slabs≥2` so the pool parallelizes over the ~8 slab-pair blocks; (c) the distributed
energy-parallel path (F12, 5.5× @ 8 ranks) is the scalable route — the dense reference bubble does
NOT scale a single slab. This directly answers "is there a bottleneck in the quatrex run": **yes —
the single-slab dense bubble is serial; scale via slabs or the distributed solver.**

**d11a anharmonic re-validation (CONVERGED, fresh):** with the orphan gone + the right thread
config (`OMP=OPENBLAS=64 QUATREX_PHPH_THREADS=2`, oversubscribed but multi-core), d11a λ=0.3
projected **converged in 25 iters in 55 min (~2.2 min/iter)** —
`phonon/scripts/out/asr_compare/d11a_validate_lam0.3.json`: resid 1.75e-3, conservation 1.2e-2,
**no causality violations**, max|Σ^R|=531 bounded, G_ball=3.70e7, G_anh=4.35e7, ratio **1.18**.
So d11a anharmonic is **fully feasible** (the F17 "infeasible" verdict was 100% the
orphan/memcap, NOT compute). **Note the ratio 1.18 > 1**: at one transport cell with broad
eta (η_w≈0.3 THz, grid 31) the anharmonic broadening *opens* coherent channels faster than
scattering closes them, so G_anh > G_ball — a real single-cell NEGF feature, not a bug; the
resistive G_anh < G_ball emerges only for multi-cell/diffusive devices or finer η (cf. F15's
finer-grid 0.997). **Lesson: single-cell λ-finite ratios are η/grid-dominated near 1; read the
anharmonic reduction from multi-cell devices or the LOA-Padé extrapolation, not n_slabs=1.** **Interesting structure-dependent observation (refined with iter 1–8 data):** at λ=1 *both*
wires hit the strong-coupling SCBA regime (Lee–Sanvito), but with very different severity. d5a
raw FC3 **diverges hard** — resid blows up to ~200, 620 Γ causality violations, Σ unbounded,
divergence-guard abort (F16). d11a (projected, λ=1, grid 21) is **much milder**: the first few
iters look stable, then by iter 7 it develops Γ sign violations (16→148→40) and the residual
*oscillates* around ~1.4 (does NOT converge), but **max|Σ^R| stays bounded ~1.3e3** and J is
stable ~3.3e-9 — no blow-up. So the soft-mode wire (d5a, 0.027 THz twist) is acutely sensitive to
full coupling while the stiff wire (d11a, lowest mode 1.07 THz) only mildly stiffens. Neither
converges cleanly at λ=1 single-cell → the clean, conserving route is moderate λ (F15: d11a λ=0.3
→ 0.997) or the LOA-Padé continuation (F14). Projection matters most where the infrared channel is
soft (d5a), and is necessary-but-not-sufficient at λ=1.

## Recommended next steps (for whoever continues)
0. **Hygiene**: before launching, `ps`-check for stale python orphans (they silently starve the
   memory governor); set `QUATREX_PHPH_MEMORY_GB` to a large fraction of *free* RAM; for single-cell
   large wires accept single-core bubble or move to n_slabs≥2 / distributed (F18).
1. ~~ASR-project the FC3 before the vertex~~ **DONE (F16)** — `enforce_asr=True` is wired and
   default-on in the sweeps; it converts the divergent λ=1 d5a SCBA into a convergent one.
   Remaining: pair it with LOA-Padé for the conserving λ→1 conductance number.
2. Use the **LOA-Padé** route (small-λ sweep at physical η) for λ→1 instead of inflating η;
   `d5a_loa_pade.py` is the template.
3. Cutoff-tolerance sweeps (`finite_analysis --analyses cutoffs`) on d11a/d17a (which converge
   more easily — wider wires, stiffer twist) once ASR projection is in.
4. **Always** use the Run recipe env vars + a memory cap (node is shared; OOM kills are silent).

### F6 — Solver verifications PASS (part b)
- **Tridiagonal RGF ≡ dense**: `tests/qttools/greens_function_solver/test_selected_inv.py`
  144 passed (generic block-tridiagonal). Plus end-to-end on a genuine phonon Dyson matrix
  with Sancho-Rubio leads (`phonon/scripts/verify/test_rgf_vs_dense_phonon.py`): RGF vs
  dense-Inv vs numpy-dense agree to **1.4e-16**. PASS.
- **Bubble/SSE kernels**: `tests/quatrex/phonon/test_bubble_kernel.py` +
  `test_sse_phonon_phonon.py` → 28 passed. (Note: nn-only FC3 truncation drops 34–48% of the
  Frobenius norm in the fixtures → transport-cell size matters; track in cutoff work.)

### F7 — Bulk Si κ reproduces experiment (part c, anchor)
Re-fit FC2/FC3 from existing phono3py forces via symfc (no new DFT), RTA-BTE:
κ(300 K) = **110.7 W/mK (11³ mesh) → 115.2 (19³)**, converging up. Experimental ~140–150;
the gap is the small 2×2×2-supercell FC3 (cutoff 5.5 Å), a known underestimate — 3×3×3+
would reach ~135–145. Validates the hiphive/phono3py FC pipeline + physics independently of
the NEGF solver. Output in `phonon/reaps/si_primitive_work/` (kappa-m*.hdf5).

### F8 — At PHYSICAL η the d5a SCBA is well-behaved (part a, decisive)
`phonon/scripts/verify/dbg_scba.py` (η_f=0.5→η_w=0.11 THz, λ=0.3, n_slabs=1): SCBA iterates
cleanly and **conserves current** — iter1 J_L=J_R=1.476e-9 W; iter2 J=1.396e-9 W,
conservation=3.0e-4, resid=0.227 (decreasing), max|Σ^R|=122 THz² (bounded), and **J is
reduced by scattering** (1.476→1.396e-9) — the physically correct direction. (Run was
OOM-killed at iter3's bubble; per-worker peak ~20.8 GiB is the bottleneck → using coarser
grid + memory cap to let it converge, `d5a_physical_eta.py`.)
**This confirms: the soft mode does NOT genuinely break the SCBA at moderate coupling; the
earlier "non-convergence" was the large-η methodology, not anharmonic SCBA breakdown.**

### F9 — FC quality of d5a; FC3 not ASR-projected before the vertex (actionable)
`finite_analysis` physical checks (`phonon/scripts/out/fcq_d5a/physical/physical.json`):
- FC2: hermiticity max_abs=0; ASR rel_frob=2.9e-16; PSD (0 negative eigvals); no imaginary
  modes. **Pristine.** Twist mode 0.0269 THz confirmed at Γ.
- FC3: permutation symmetry perfect (~1e-17). FC3 leg-ASR `leg_j_rel=leg_k_rel=0.796`.
  Per `physical_tests.py:83-114`, this ~0.80 is **expected physics** for an open
  (finite-cross-section) wire — periodic ASR is only meaningful along the periodic axis;
  **bulk Si = 0.000**. So it is NOT a hiphive fit defect.
- **BUT**: the dense SCBA path (`fc3_device.build_device_fc3_blocks`, called from
  `dense.py:1291`) consumes the **raw** mass-weighted FC3 with **no ASR projection**
  (`load_quatrex_blocks(asr_project=...)` defaults False; the standalone path never calls it).
  An un-ASR-projected FC3 couples the vertex into the acoustic/twist subspace — feeding the
  same low-ω channel that makes the small-η SCBA stiff. The zero-mode projector cleans Σ but
  not the vertex. **Actionable: ASR-project the FC3 (`asr_project_factor`) before building the
  device vertex and re-test small-η SCBA stability.**

### F10 — VERDICT for part (a): converged, physical G_anh < G_ball at small η
`phonon/scripts/out/d5a_physical_eta/lam0.3.json` (η_f=0.25→η_w≈0.11 THz, n_slabs=1):
- **converged in 20 iters**, resid=1.7e-3, conservation=3.2e-3.
- G_ball=3.17e7, G_anh=2.99e7 → **G_anh/G_ball = 0.942 < 1** — anharmonic scattering correctly
  *reduces* conductance. The physical impossibilities of F1 are GONE at physical η.
- λ=0.6 run: residual plateaus ~0.5 with causality "Gamma sign violations" (~500 pts) — the
  genuine onset of strong-coupling SCBA stiffness (cf. Lee–Sanvito 2009). This is the regime
  where the conserving LOA-Padé continuation (Bescond 2011/2018) is the correct tool, not a
  larger η.

**Answer to (a):** the d5a "soft mode / non-convergence" was **NOT genuine anharmonic SCBA
breakdown and NOT a solver bug**. It was (i) a diagnostic run at η_w≈0.7–2 THz that destroyed
the coherent baseline (making G_anh>G_ball an artifact), on top of (ii) a real 0.027 THz twist
mode whose Bose-enhanced low-ω weight stiffens the fixed point at physical η — aggravated by
(iii) feeding an un-ASR-projected FC3 into the vertex (F9). At physical η and moderate coupling
the SCBA is correct and convergent; toward λ→1 use LOA-Padé.

### Gotcha — thread oversubscription (see Run recipe above)
First two d5a SCBA launches were silently killed by `libgomp: Thread creation failed` (256
workers × BLAS threads). Fixed by single-threaded BLAS + QUATREX_PHPH_THREADS bound.

### F19 — n_slabs≥2 scaling: physics + solver cost (Part 1)
- **Solver cost vs #cells (`phonon/scripts/verify/rgf_vs_dense_scaling.py`):** block-tridiagonal
  **RGF is linear** in the number of blocks, the **dense reference is cubic** — timings (bs=192):
  nblk 2/4/8/16/24/32 → RGF 0.012/0.018/0.038/0.071/0.118/0.158 s vs dense
  0.011/0.048/0.229/1.375/4.01/9.08 s → **dense/RGF = 57× at 32 blocks**, RGF≡dense to ~1e-13
  (extends F6). Distributed energy-parallel (`dist_scaling.py`, N=1536, 128 E-pts):
  np=1/2/4/8 → 2.05/1.11/0.72/0.38 s = **5.3× at 8 ranks** (re-confirms F12).
- **Dense bubble parallelism vs n_slabs:** #Phi device blocks = 1 (n_slabs=1) → 8 (n_slabs=2) →
  15 (n_slabs=3); the phph ThreadPool parallelises over these (I,J) blocks (n_slabs=2 engaged
  32 workers vs n_slabs=1's serial single block, F18). BUT each block's bubble is single-thread
  BLAS, so the **dense per-point wall explodes**: CNT 66 s (n_slabs=1) → 1193 s (n_slabs=2, ~18×).
  **Conclusion: the dense reference does NOT scale to n_slabs≥2; production multi-cell anharmonic
  transport must use the RGF (linear) + energy-parallel path.** This is the concrete answer to
  "does it scale with n_slabs": yes via RGF/distributed, no via the dense reference.
- **Length physics (clean on the CNT, `phonon/scripts/out/cnt33_transport/`):** G_anh/G_ball
  **decreases with length** — 0.84 (L=1) → 0.756 (L=2) [L=3 pending] — accumulating 3-phonon
  resistance (ballistic→diffusive onset). Ballistic G(L) for all three wires
  (`ballistic_curves.py`) decreases with L (finite-η coherence attenuation, F17). **d5a multi-cell
  at λ=1 does NOT converge** (resid oscillates ~1.4, the strong-coupling regime of F16/F18) and is
  very slow — multi-cell d5a needs moderate λ / LOA-Padé, the CNT (better conditioned) converges.

### F20 — Carbon nanotube (first non-Si structure) + DFPT/FD/hiPhive cross-check (Part 2)
- **(3,3) armchair CNT via hiPhive** (`phonon/configs/cnt/cnt33_vasp.yaml`, converted thirdorder→
  hiphive: supercell [1,1,3], cutoffs [3.5,2.5], 30 rattled VASP, ardr; `fc3_hiphive_cnt33_vasp/`).
  Fit rmse_test 0.28 eV/Å (~10× the SiNWs — stiff C–C → large forces; absolute, not relative,
  worse) but the FC is physical: **dispersion has no imaginary modes**, with the textbook armchair-
  CNT acoustic set — 2 flexural (quadratic, low slope) + 1 **twist at 0.026 THz** (a quasi-Goldstone,
  strikingly like d5a's 0.027 THz) + 1 high-velocity LA. ASR residual ~0 (periodic tube, unlike the
  open SiNW's 0.8 → projection dropped ‖FC3‖ by 35%).
- **CNT transport vs Si:** ballistic **maxT=11.5 channels, G_ball=7.7e8** (per box area) — vs d5a
  3.4/4.0e7 and d11a 4.5/6.0e7: the CNT carries **~3× the channels and ~12–19× the conductance**
  (stiff sp² bonds, high group velocities). Anharmonic SCBA **converges cleanly even multi-cell**
  (conservation ~3e-4, far better than the SiNWs) with a clean **resistive** reduction
  G_anh/G_ball = 0.84 (L=1) → 0.756 (L=2). The CNT is the best-behaved anharmonic transport case so far.
- **FC-method cross-check on bulk Si primitive (DFPT vs FD vs hiPhive):** DFPT FC2 via QE
  `ph.x`→`q2r.x` (serial — the perl-wrapped pw.x deadlocks at np>1 on this node; conv_thr 1e-10
  stalls, use 1e-8). **Γ-optical: DFPT 15.12 vs FD 15.37 THz → 1.6% agreement** — the two ab-initio
  FC methods agree (`fc_method_dispersion_si.pdf`). DFPT-FC3 unavailable (D3Q `d3q.x` plugin not
  installed → FC2/dispersion only). hiPhive bulk-Si QE was flaky/slow on this node; hiPhive's FC
  quality is independently established by the clean dispersions of all SiNW + CNT fits.

### F21 — Deeper FC3 physics: mode-resolved phonon-phonon insights (d3q now available)
With d3q installed (`/usr/scratch/mont-fort11/pfischill/qe-7.5-git/build/bin`), explored the
anharmonic physics in depth (`phonon/scripts/verify/phph_physics.py`, `phph_NU_gruneisen.py`,
`dfpt_kappa.py`; figs `phph_physics_si.pdf`, `phph_NU_gruneisen_si.pdf`).
- **Bulk-Si mode-resolved physics (FD FC3, phono3py RTA):**
  - **Lifetimes** τ(ω): median 4.2 ps, max ~700 ps (long-lived low-ω acoustic), τ∝~ω⁻² envelope.
  - **κ-accumulation vs MFP**: 50% of κ from phonons with MFP **> 134 nm**, 90% from **< 1.2 µm**.
  - **Mode-κ spectrum**: **93%** of κ carried by acoustic modes (f < 8 THz).
  - **Normal vs Umklapp** (13³, is_N_U): Umklapp **0.44** / Normal 0.56 of total scattering at 300 K.
  - **Mode Grüneisen** (from FC3): mean|γ|=**1.04**, max 1.54 — matches the experimental Si γ≈1,
    an independent validation that the FC3 *anharmonicity strength* is physical.
- **The unifying nanowire insight**: bulk κ fraction from MFP **< 1 nm = 0.0%**, < 10 nm = 1.3%,
  < 100 nm = 37%. The d5a/d11a wires (~0.5–1.1 nm core) sit two-to-three orders below the bulk
  phonon-MFP spectrum → they are **boundary-scattering-limited**, retaining ~none of bulk's
  anharmonic-limited κ. This is *why* the NEGF finite-cross-section anharmonic correction is a small
  perturbation (G_anh/G_ball ≈ 1 at one cell, F17/F20): the wire is already deep in the
  boundary-limited regime, so 3-phonon scattering is a minor add-on — the physics is consistent
  across the two methods.
- **DFPT FC3**: d3q **does not support ultrasoft pseudos** ("US not implemented") — needs
  norm-conserving. A complete D3Q DFPT FC3 (NC pseudo `si_nc_pbe.upf`) already exists at
  `phonon_old/input_calc/dfpt/fc3.hdf5`. DFPT vs FD agree at the **harmonic** level (Γ-optical
  15.12 vs 15.37 THz, **1.6%**, F20). The κ-level phono3py comparison needs the DFPT-parse
  supercell ordering (QE image order) reconciled with phono3py's (the same ordering wrinkle that
  scrambles a direct fc2 load) — flagged, not a physics result. CNT phono3py 1D-RTA κ is finicky
  (vacuum-padded mesh); the CNT anharmonic physics is the NEGF result (F20), which is the correct
  finite-cross-section method.

### F22 — Distributed scaling of the PRODUCTION phph self-energy + q-point investigation
Re-verified the quatrex distributed scaling of the *phonon-phonon self-energy itself* (not just
the RGF Dyson solve of F12), and investigated q-point distribution.
Benchmark `phonon/scripts/verify/phph_dist_scaling.py` times the production
`src/quatrex/phonon/sse_phonon_phonon.SigmaPhononPhonon.compute` on a synthetic block-tridiagonal
problem (NBLK=6, BS=16, NE=96) under a 2D (block × stack) MPI grid; `QPHPH_BCS` sets block_comm_size.

- **Block-parallel (the (I,J) output loop over `comm.block`)**: speed-up **1.0 / 1.52 / 2.37 / 5.36×**
  at np=1/2/3/6 — **near-ideal (90% at 6 ranks)**. ✓ correct and scalable.
- **Energy/stack-parallel (`block_comm_size=1`)**: **1.0 / 0.97 / 0.89×** at np=1/2/4 — **NO speed-up**
  (slightly slower from the all-gather). The 3-phonon bubble Σ(ω)=∫Φ G(ω')G(ω−ω')Φ is an **energy
  convolution** that couples all ω. The implementation handles it by `ranks.stack.all_gather_v` of
  the full-ω diagonal G blocks (`sse_phonon_phonon.py:_gather_diagonal_blocks`), after which **each
  stack rank recomputes the full-ω bubble FFT and writes only its slice `[e_lo:e_hi]`**
  (`sse_phonon_phonon.py:241-273`). So energy distribution divides storage/write-back but the FFT
  compute is **REPLICATED** across stack ranks. **Contrast F12/F19**: the RGF Dyson *solve* is
  energy-embarrassingly-parallel (5.3× @ 8); the phph *self-energy* is not. In a full SCBA at high
  stack-rank counts the phph self-energy therefore becomes the bottleneck unless block-parallelised.
  **Verdict: the distributed phph is CORRECT and properly block-scales; energy-scaling is
  fundamentally limited by the convolution (not a bug).** Plot `phph_scaling.pdf`.
- **q-points (first investigation)**: q-points / transverse momenta are currently **folded into the
  stack** (`solver.py:101`: `global_stack_shape = energies.shape + kpoint_grid`) with **NO dedicated
  communicator** — so they ride the same stack distribution and inherit the bubble replication. BUT
  q-points are **independent external indices** for the real-space block-sparse phph: the bubble
  (`bubble.py`) FFTs only the energy axis — there is **no cross-q convolution** (a fixed transverse
  condition per q), unlike energy. So q-points are **embarrassingly parallel**: N_q q-points over P
  ranks cost (N_q/P)·t_phph → **ideal P×** (per-q-point cost ≈ t1 = 4.13 s, independent). **Proper
  q-distribution = a dedicated `q_comm` (3rd axis alongside block×stack)**: add `q_comm_size` to
  `CommConfig`, a 3-way split in `QuatrexCommunicator.configure`, and separate the q-index from the
  energy in `global_stack_shape` so the q-loop distributes over `q_comm` instead of folding into
  the stack. The phph compute kernel needs no change (already per-(energy,fixed-q) independent).
  **This is the recommended scalable axis for anharmonic phonon transport — near-ideal, unlike
  energy.** Physics: 3-phonon energy conservation ω₁=ω₂+ω₃ mixes all frequencies (non-local in ω);
  transverse momenta in the finite-cross-section device do not (local per q) — so the right
  parallel decomposition is q-points (ideal) + spatial blocks (near-ideal), with energy parallelism
  reserved for the Dyson solve.

### F23 — Distributed q-RESOLVED phonon-phonon self-energy + Si thin-film κ (Guo–Bescond–Zhang 2020)
**Physics correction to F22.** F22's claim that q-points are "embarrassingly parallel / no
cross-q convolution" is **WRONG for a transversely-periodic system**. In a periodic film the
3-phonon self-energy couples the transverse momenta by crystal-momentum conservation:
Σ(q⊥,ω) = (1/N_q) Σ_{q′⊥} ∫dω′ Φ(q′⊥) G(q′⊥,ω′) G(q⊥−q′⊥, ω−ω′) Φ(q⊥−q′⊥) — a convolution
over **both** q⊥ (momentum) **and** ω (energy). F22's "independent q" held only for the
production code's Γ⊥-only (finite-cross-section wire) path, where each q is a fixed external
condition. The coupled-q kernel already existed (`solver/se_q.py:compute_phph_self_energy_q_dense`,
momentum sum via `build_q_diff_map`, energy FFT) and is the validation oracle here.

**Distributing q is still the scalable axis — but with a data-exchange cost** (exactly
Guo–Bescond–Zhang's "G of one mode (ω;q⊥) is coupled with many others → data exchange between
CPUs"). Distributing the **external** q⊥ over ranks divides the explicit external-q loop (~1/P);
each rank still needs **all internal** q′ Green's functions, so the internal-q G is all-gathered
each iteration. Contrast the energy axis (F22: flat 1.0×, the convolution is replicated).

**Verifications (all PASS):**
- **Coupled-q kernel oracle** (`scripts/verify/se_q_validate.py`): single q=Γ reproduces the
  Γ-only finite bubble (`se_finite.compute_phph_self_energy_finite`) to **rel err 2.6e-16**; a
  2×2 mesh runs with bounded Σ and a correct momentum round-trip q=(q−q′)+q′.
- **Production q-communicator** (`src/qttools/comm/comm.py` + `src/quatrex/core/config.py`): added
  `q_comm_size` to `CommConfig` and a **third split** in `QuatrexCommunicator.configure` so global
  ranks factor as **q × stack × block** (`comm.q`, same all_gather_v/all_reduce API). Backward
  compatible: `q_comm_size=1` leaves block/stack identical. Factorization verified on 8 ranks
  (bcs×qcs = 1×1,1×2,2×2,4×2,… all give size product = nranks, unique coords).
- **Distributed q-self-energy == serial oracle** (`scripts/verify/phph_q_comm_validate.py`): each
  rank owns a contiguous q⊥ slice, `comm.q.all_gather_v(axis=0)` reconstructs the full internal-q
  G (the data exchange), each rank computes only its external-q Σ, gathered back. On a 2×2 mesh at
  P=1/2/4: all-gather-G err **0.0**, distributed-SE vs oracle rel-err **0.0** (bit-identical) → PASS.
- **Distributed-q scaling** (`scripts/verify/phph_q_dist_scaling.py`, 4×4 mesh, n_dof=16, NE=41):
  ext-q self-energy wall **5.30 / 2.83 / 1.48 / 0.805 s** at np=1/2/4/8 = **1.0 / 1.87 / 3.58 /
  6.58× (82% at 8 ranks)**, with Σ bit-identical across ranks. **This is the correct scalable axis
  for the periodic anharmonic problem** (vs the flat energy axis). → the F22 "ideal q-axis"
  conclusion stands quantitatively, but the mechanism is "divide external q + all-gather internal G",
  not "no coupling".

**Part B — Si thin-film cross-plane κ (`scripts/verify/si_film_kappa.py`, `si_film_ballistic.py`).**
The film = bulk Si truncated to N layers along transport with in-plane 2D periodicity → a transverse
q⊥ problem solved by the existing q-resolved driver `solver.transmission_q` (calls the coupled-q
`se_q` kernel in its SCBA `se_kernel`), reusing **bulk-Si FD FC2+FC3** (`reaps/si_primitive_work`,
bulk κ≈110–115, F7/F13; cf. Guo's 1st-NN-shell DFT bulk κ=120.69). Driver sanity: q_mesh=(1,1)
reproduces `transmission_finite` (built-in regression, PASS). Conductance is per area
G[W/m²K]=J/(A⊥ΔT); cross-plane κ(L)=G·L.
- **Ballistic q + η convergence** (8 layers, eta_factor=0.1): G_ball **q-converges at nk≥8** to
  **≈920 MW/m²K** (nk=4/8/12/16 → 970/928/921/920). The earlier large η (eta_factor=0.5) was the
  culprit that crushed the conductance to ~350 MW/m²K (the F17 finite-η coherence-attenuation
  artifact) — small η + nk≥8 recovers Guo's scale.
- **Guo's quantitative targets (PRB 102, 195412, Sec. II B; extracted via Scite full-text):** bulk
  κ 120.69/136.46/147.13 W/mK for 1st/2nd/3rd-NN FC3 (their NEGF uses 1st-NN → 120.69); thin-film
  **anharmonic conductance G_anh(3uc)=939.72, G_anh(5uc)=890.97 MW/m²K** (1uc=5.4018 Å); **mesh 8×8
  transverse, 121 frequencies**, ΔT=10 K @300 K.
- **Our matched-mesh result (nk=8, nfreq=121, eta_factor=0.1, 4 layers≈1.55nm≈3uc):** G_ball=**1034**
  MW/m²K (matches Guo's ballistic scale), G_anh=**570** MW/m²K → a 45% reduction vs Guo's ~10%.
  **The over-reduction is NOT the FC3 vertex range**: restricting the vertex to 1st-NN (Guo's approx.
  II; keeps 9.8% of triplets) gives G_anh=**571.8**, essentially identical to the full FD FC3 (570.3)
  — the 1st-NN triplets dominate the anharmonicity. The gap is in the **self-energy combinatorial
  prefactor**: Guo's paper (App. A; the Ref-[43]/[64] comparison) explicitly identifies a **factor
  of 4** over-count (repeated pairing combinations) in the earlier Luisier (PRB 86, 245407)
  self-energy that the standalone `se_q` kernel follows. Since Σ ∝ Φ², the ÷4 correction is
  vertex ×0.5 (`--vertex-scale 0.5`). **Result with the factor-4 correction: G_anh(3uc)=831,
  G_anh(5uc)=704 MW/m²K** vs Guo 940/891 — within **12% / 21%**, correct sign and conductance scale.
  (5uc ballistic G_ball=952 also matches Guo's scale.) Residual gap + slightly steeper thickness
  dependence are consistent with Guo's further self-energy truncation (approx. III: diagonal +
  NN-only blocks) and the LDA-vs-PBE FC difference.
- **VERDICT:** the q-resolved anharmonic NEGF reproduces (i) the ballistic cross-plane conductance
  to ~1% (1034/952 vs ~1030/950 at 3uc/5uc), (ii) the correct resistive sign + conductance scale,
  and (iii) once the documented self-energy convention (Guo's factor-4 over Luisier) is matched, the
  anharmonic conductance to 10–20%. The absolute anharmonic reduction is **convention-governed, not
  FC-governed** (1st-NN ≡ full FD FC3) — exactly the factor-4/π/9 ambiguity Guo's paper resolves.
  Outputs: `phonon/scripts/out/si_film/si_film_kappa_nk8_{guo,nn,div4}.json`,
  `si_film_ballistic{,_lconv}.json`; figs `document/fig/transport_sweeps/{phph_q_scaling,si_film_conductance}.pdf`.

### F24 — Self-energy prefactor: the 1/4 symmetry factor is now the DEFAULT (was missing)
Verified and corrected the 3-phonon self-energy prefactor (follow-up to F23's empirical "÷4").
- **It is physically the truth (three independent routes):** (i) Wick count for the bubble with
  H3=(1/3!)Φu³ and phono3py's full FD FC3 gives an overall symmetry factor **1/4** on top of the
  i/2 loop factor (3·3·2 pairings / (2!·3!·3!) = 1/4); the kernels applied only i/2, so Σ was 4×
  too large. (ii) Guo-Bescond-Zhang App. A state exactly this factor-4 over-count for the Luisier
  (PRB 86 245407) expression the kernel reproduces. (iii) Thin-film cross-plane conductance with ÷4
  matches Guo (831 vs 940), and a 38×/9× error is excluded (would undershoot Guo's reduction).
  An independent phono3py Fermi-golden-rule linewidth check (`scripts/verify/se_q_vs_phono3py_gamma.py`,
  bulk Si, 3D q-mesh) corroborates the kernel over-scatters by an O(4) factor — the on-shell ratio is
  inflated to ~9 on coarse meshes by the Lorentzian-η joint-DOS sampling (broadening artifact), and
  the integrated ratio is background-contaminated, so the **derivation+Guo+thin-film** pin the factor
  at exactly 4, not the numerics.
- **Wired:** `constants.PHPH_SYMMETRY_FACTOR = 0.25`, applied in the prefactor of `se_q.py`,
  `se_finite.py` (single + multi-slab), default-on; `transmission_q`/`transmission_finite` gained
  `legacy_prefactor=False` (→ `--legacy-luisier-prefactor` in `si_film_kappa.py`) to restore the old
  4× value. The production `bubble_dense`/`bubble_prefactor_thz` are UNCHANGED (the factor lives in
  the science-path callers), so `test_bubble_kernel`+`test_sse_phonon_phonon` stay green (28 passed),
  and `se_q_validate` 1q-Γ≡finite still holds (both paths ×¼, max|Σ| 2.43e8→6.08e7).
- **Consequence:** every ANHARMONIC conductance in F10/F14/F16/F18/F20 (d5a/d11a/CNT wires) used the
  4×-too-large (Luisier) convention; with the default they scale toward less scattering (G_anh rises
  toward G_ball, smaller reductions). **Ballistic G is unaffected.** The old `--vertex-scale 0.5`
  workaround is subsumed (vertex_scale is now only for LOA-Padé). Default-vs-legacy check (4-layer Si
  film, nk=2): G_ball=1069 both; G_anh 995 (default ÷4, 7% red) vs 821 (legacy ×4, 23% red).

### F25 — Full (unapproximated) off-diagonal q-resolved self-energy + approximation study
The q-resolved transport driver (`transmission_q`) computes only the slab-DIAGONAL self-energy
(Guo approximation III). Implemented the FULL off-diagonal one and quantified each approximation.
- **New kernel** `se_q.compute_phph_self_energy_q_dense_multi_slab` returns `{(I,J): Σ(q⊥,ω)}` with
  inter-slab blocks, adapting the Γ-only template (`se_finite._build_pair_index` +
  `fc3_device.build_device_fc3_blocks`) with a q-fold (transverse Bloch phases on the two contracted
  legs, via a phase-modified `M_stacked` fed through `build_device_fc3_blocks`) + the coupled-q
  momentum sum (`q_diff_map`) + energy FFT (`bubble_dense_from_fft`). **Validated:** reduces EXACTLY
  (rel err 0.0) to `compute_phph_self_energy_finite_multi_slab` at a 1×1 mesh, with off-diagonal
  blocks present (`scripts/verify/se_q_multislab_validate.py`, n_slabs=3 → 6 off-diag blocks).
  Toggles: `sigma_cutoff` (=0 → Guo approx III), `vertex_cutoff` (slab-range FC3), `g_cutoff`.
- **Study** (`scripts/verify/si_film_approx_study.py`, Si film 3 slabs / 1.16 nm, nk=4, first-Born
  step from the ballistic device G — relative ranking is the point, not a full SCBA; corrected
  prefactor): full G_anh=858 MW/m²K; **diag-only (approx III) −1.8%** despite the off-diagonal Σ
  carrying **47% of the Frobenius weight** (large but nearly transport-neutral → Guo's diagonal
  approximation is well-justified); **1st-NN/slab FC3 (approx II) +0.0%** (long-range FC3 negligible,
  echoing F23); **G-range truncation (g_cutoff=1) catastrophic** (negative/unphysical G — the
  inter-slab G coherence is essential, echoing F17's diag-G 5–30× blowup). So of Guo's three
  numerical approximations, II and III are benign (~0–2%) and the G-range one must NOT be applied.
  Caveat: the 2-atom Si primitive (n_dof=6) makes the within-block nearest-atom restriction vacuous;
  a multi-atom transversely-periodic cell (e.g. conventional Si, n_dof=24) would need new FC3 DFT.

### F27 — Si/Ge heterostructure (Guo-Bescond-Zhang 2020): mass-mismatch barrier + phph
Reproduced Guo's heterostructure setup — a Ge-layer barrier embedded in a Si film, treated as **Si
force constants + Ge mass** (mass-mismatch), Si contacts — with the corrected ÷4 prefactor.
- **Implementation:** added a per-slab MASS profile to the q-resolved driver. New
  `leads.build_device_hamiltonian_massprofile(K_00,K_01,masses)` re-weights the bare stiffness
  K (= H·m_lead, un-mass-weighted) per slab: D_00[l]=K_00/m_l, D_01=K_01/√(m_l m_{l+1}); leads stay
  pure Si; FC3 vertex kept uniform-Si (the F25-justified model). `transmission_q(..., mass_profile=...)`
  (`dense.py`) + driver `scripts/verify/si_ge_film_heterostructure.py`. **Regression:** an all-Si
  mass_profile reproduces plain `transmission_q` to 1e-15.
- **Results (nk=8, 121 freq, eta_factor=0.1):** (a) the **pure Si film** phph enhancement grows with
  thickness — R up **+32%** at 2.3 nm (6 slabs) → **+51%** at 5.4 nm (14 slabs); G_anh/G_ball 0.76→0.66,
  matching the Si-film diffusive-crossover trend and Guo's "phph enhances R, more so for thicker films."
  (b) the **Ge mass-mismatch barrier** raises the *ballistic* R by **+344–369%** (harmonic interface
  reflection — Guo's interface scattering). (c) In the **barrier-dominated** heterostructure phph is
  ~neutral (±1%): the strong barrier filters transmission to low-frequency acoustic phonons that are
  anharmonically long-lived, so phph adds little *through the barrier*. So Guo's two physics (interface
  reflection + film phph enhancement) are reproduced; the heterostructure phph suppression is a physical
  consequence of the strong (2-full-layer Ge) barrier filtering. Outputs:
  `scripts/out/si_film/si_ge_heterostructure{,_thick}.json`.

### F24-CORRECTION — the "verified ÷4 prefactor" claim is RETRACTED (default reverted to native)
F24 claimed the 1/4 self-energy prefactor was the verified true physics. **That was overstated and is
withdrawn.** On re-examination:
- The phono3py golden-rule linewidth cross-check (`se_q_vs_phono3py_gamma.py`) is **normalisation-broken**:
  the units-correct, broadening-free ratio R = ∫(−ImΣ_NEGF)dω / ∫(2ω_s γ_p3p)dω with the NATIVE prefactor
  comes out **~37** at the Γ-optical mode (≈12π — a leftover 2π/ħ / equilibrium-G normalisation mismatch
  between the standalone reconstruction and the code's internal G convention), and earlier framings of the
  same check gave 9. A 37× (or 9×) linewidth error is inconsistent with the thin-film conductance (which
  shows only a ~1.6–4× over-scatter), so this standalone-G route **cannot determine the absolute prefactor**.
- The ÷4 was therefore driven by a (possibly mis-symmetrised) Wick count, Guo's "factor-4 vs Luisier"
  (assuming our kernel ≡ Luisier, unproven), and moving the thin-film conductance toward Guo (831 vs 940 —
  not a match; the reduction stays ~2× too strong, arguing the factor isn't even 4). **Not a verification.**
- **Action:** `PHPH_SYMMETRY_FACTOR` reverted to **1.0 (native)** — the self-energy as derived (theory.tex's
  standard sunset, which reduces analytically to the correct Fermi-golden-rule πħ/8ω_s). The ÷4 is NOT applied.
- **Real, OPEN discrepancy:** at native prefactor the Si thin film over-scatters vs Guo (~45% reduction at
  1.5 nm vs their ~10%). Whether this is a genuine prefactor/unit error, the equilibrium-G normalisation, or
  Guo's approximations (II/III) + LDA-vs-PBE FC is **unresolved**. The clean settling test (not yet done):
  **bulk-Si κ from the code's OWN SCBA Green's functions in the thick-film limit vs phono3py RTA κ≈110 on the
  identical FC3** — this uses the code's internal G convention and so isolates the true absolute prefactor.
- Prior anharmonic results (F23/F25/F27) that quoted the ÷4 number must be read with this caveat; the native
  (unscaled) values are the defensible ones pending the κ benchmark.

### F28 — Prefactor verification: phono3py-linewidth route EXHAUSTED; the ÷4 is NOT supported; native defensible
Pushed the phono3py golden-rule cross-check to a mode-resolved test to settle native-vs-÷4 once and for all
(`/tmp/claude/verify_modes.py`: native se_q on a bulk-Si q-mesh, R(mode)=∫(−ImΣ_NEGF)/∫(2ω_s γ_p3p) at
several grid points spanning the BZ; if R were a single constant it would be a pure units convention and the
physics would be verified mode-by-mode).
- **Result — R is NOT mode-independent and NOT mesh-converged.** Mode-resolved spread is 138% (2³) → 172% (4³).
  Decomposing by frequency: **well-sampled high-ω optical modes** (the only ones with a properly sampled
  3-phonon joint-DOS) give R/(2π)² ≈ 2.6 (2³) → **0.4–0.95 (4³)** — i.e. mesh-dependent, drifting toward
  ~1; **low-ω acoustic modes blow up** to R/(2π)²=10–85 (joint-DOS starvation on a coarse mesh: phono3py
  γ→0 while the NEGF Lorentzian-η picks up background, so the ratio diverges — a sampling artifact, not physics).
- **The Γ-optical mode CONVERGES and is the decisive number.** It alone has a robust well-sampled decay
  channel; tracking it across meshes: R/(2π)² = {2.62,2.71,2.53} (2³) → {0.79,0.74,0.84} (4³) →
  **{0.77,0.74,0.80} (6³)**. The 2³→4³ drop is the coarse-mesh artifact resolving; **4³→6³ is converged at
  R/(2π)² ≈ 0.77 ± 0.03** (this used a coarse NE=201 frequency grid — see the refinement below).
- **Refinement (`verify_gamma_opt.py`, the decisive run): the area-integrated ratio is η-INVARIANT at
  R/(2π)² ≈ 1.06.** Two clean facts: (i) the **on-shell peak** ratio is a pure broadening artifact —
  γ_NEGF(ω_s) = 0.13/0.26/0.44 THz at η=0.02/0.04/0.08 (scales ∝η), so the "~4" (and the earlier "~9")
  on-shell ratios are meaningless; **this ∝η peak is the spurious "factor of four."** (ii) The
  **area-integrated** R (Eq. ∫(−ImΣ)/∫2ω_sγ, broadening-conserving) at a ±4 THz window is **1.05/1.07/1.06**
  at η=0.02/0.04/0.08 — **η-locked**; across windows ±2/±4/±6 it spans 0.83–1.07, across freq-grids 0.77–1.07.
  → **native agrees with phono3py to ~10–15% and a factor-of-4 is excluded by an order of magnitude**
  (÷4→0.26, ×4→4.2; measured ≈1). **The ÷4 is dead; native is the correct ab-initio physics, verified to
  ≲15%.** The residual ≲15% is window/freq-grid method uncertainty (the (2π)² is a fixed THz²-vs-THz units
  convention). Figure `document/fig/transport_sweeps/prefactor_verification.pdf`; written up in
  results.tex `sec:res_prefactor` (Fig `fig:res_prefactor`). Tightening below 15% needs the bulk-κ benchmark.
- **The route is exhausted.** Tested now on-shell (F24: ~9), integrated (F24-CORRECTION: ~37≈(2π)²), and
  mode-resolved (here): it never converges to a clean, mesh-independent constant because the NEGF
  Lorentzian-η broadening and phono3py's tetrahedron sample the sparse joint-DOS differently. **It cannot
  determine the absolute prefactor** — final verdict on this method.
- **What it DOES establish (the part that answers the user's worry):** for the resolvable modes R/(2π)² is
  **order 1, never 4**. If native were 4× too large (the ÷4 hypothesis), the well-sampled modes would sit at
  R/(2π)²≈4; they sit near 1. **→ no support for ÷4; the retraction (F24-CORRECTION) stands and native is the
  defensible default.** The (2π)² is a fixed units convention between the code's linear-THz Σ and phono3py's
  γ (phono3py stores γ=half-linewidth in THz), not a physical factor.
- **The two remaining CLEAN (mesh-free / convention-free) tests, not yet done:** (a) the **vertex-element
  comparison** — phono3py `Interaction.get_interaction_strength()` |Φ_{λλ′λ″}|² for one triplet vs the code's
  mode-projected M_stacked V3 (no energy-conservation δ, no BZ sum → isolates the FC3/vertex normalization
  exactly); (b) **bulk-Si κ from the code's own SCBA in the diffusive limit** vs phono3py RTA 110 (uses the
  code's conventions end-to-end). These, not the linewidth ratio, are how to pin native to better than ~2×.

### F29 — High-quality large-supercell (5x5x5) hiphive FC3 for bulk Si + thin-film scaling
Built a long-range, high-quality bulk-Si FC3 to replace the under-converged 2x2x2 phono3py FC3 as the
thin-Si transport input. Config `configs/si_primitive/hiphive_big.yaml`; reaps `reaps/si_big_hiphive/`.
- **DFT/fit:** 5x5x5 PRIMITIVE supercell = **250 atoms** (> 4^3), VASP-6.3.2 PBE (np=128, ENCUT 450,
  2x2x2 k), 24 mc-rattled structures. hiphive **ardr** fit: **103 params, RMSE train 0.0203 / test
  0.0203 eV/A** (train==test, no overfit; ~14x tighter than the CNT), rotational + FC2/FC3 ASR satisfied.
  FC3 cutoff **5.0 A (~4th NN)** vs the old 2x2x2's 3.0 A (~1st NN). Dispersion CLEAN (no imaginary
  modes; Gamma-optical 15.15 THz, slightly softer than QE-PBE 15.37 -> VASP-vs-QE PBE).
- **Adapter** `/tmp/claude/si_big_adapt.py` -> `reaps/si_big_hiphive/{phono3py.yaml, fc2.hdf5
  (force_constants), fc3.hdf5 (fc3)}`, load_bulk_si-compatible; smoke (1,1)==finite PASSES. (NB: the
  dense 250^3 fc3 is 3.4 GB in RAM; on disk use the gzip-compressed 5.2 MB file from the reap, not a
  re-dumped uncompressed copy.)
- **Bulk kappa validation (phono3py RTA):** new FC3 = 102.8 (11^3) -> **116.4 (19^3) +isotope**, 123.4
  (19^3 pure anharmonic). The old 2x2x2 QE FC3 gave 115.2 at 19^3 -> **the two AGREE to ~1% at the
  converged mesh** (the 11^3 gap 102.8 vs 110.7 was mesh under-convergence). So the new FC3 reproduces
  the converged bulk kappa while being much higher quality (4th-NN range, RMSE 0.02, clean dispersion).
- **Thin-Si cross-plane scaling** (`si_film_kappa.py --fc3-subdir reaps/si_big_hiphive`, nk=8, 121 freq,
  eta_factor 0.1, native prefactor; `scripts/out/si_film/si_film_kappa_bigfc3.json`; heat-flow
  conservation 5-8e-4 all thicknesses):
  | thickness | G_ball | G_anh | G_anh/G_ball |
  |---|---|---|---|
  | 3 L (1.16 nm) | 907 | 470 | 0.519 |
  | 5 L (1.93 nm) | 849 | 378 | 0.445 |
  | 8 L (3.09 nm) | 775 | 296 | 0.381 |
  Reduction GROWS with thickness (diffusive crossover, correct); G_ball matches Guo's scale (~940).
  The reduction (48-62%) is LARGER than the old 2x2x2 FC3 (G_anh 570 -> now 296-470) because the
  longer-range FC3 adds inter-slab anharmonic scattering, and much larger than Guo's ~10%. This is NOT
  a prefactor issue (F28: native verified to ~15%); the Guo gap is the documented open question
  (their diagonal-Sigma/1st-NN approximations + their eta/setup + LDA-vs-PBE). Open physical tension
  worth a look: 50-62% reduction at 1-3 nm vs Si's long bulk MFP (mid-freq modes with gamma~0.03 THz
  have MFP~13 nm, so DO scatter in 3 nm; the long-MFP acoustic modes that dominate bulk kappa do not).
  Figure `document/fig/transport_sweeps/si_film_bigfc3.pdf`.

### F30 — CNT(3,3) converged deep-dive: length ladder, temperature/low-T channel-freezing, cutoff hierarchy
Pivoted to carbon nanotubes (user request, 2026-06-02). The earlier cnt33 run (F20) had excellent
heat-flow conservation but an UNDER-converged current (15-iter cap, residual stuck ~8e-3). Re-ran with
tight tolerances + many iters; the current converges cleanly and the loose answer was NOT a wrong point.
- **FC re-validation** (`finite_analysis fc_quality,physical`, `scripts/out/cnt33_fcq`): the (3,3) FC is
  the **cleanest of any structure** — FC2 ASR rel 1.5e-16, FC3 perm-sym exactly 0, FC3 ASR-legs 1.1e-16,
  FC2 PSD 0 negative eigenvalues, 0 imaginary modes (3 acoustic + twist 0.0265 THz). The periodic tube's
  exact ASR (vs open SiNW) is why it is so well conditioned.
- **Length sweep, tight tol** (`scripts/out/cnt33_converge`, eta_factor 0.7 -> eta_w 0.206 THz, anderson):
  L=1 converged to resid **9.6e-10** (188 iters), L=2 to **9.9e-6** (84 it), L=3 to **9.7e-6** (95 it).
  L=1 at 1e-9 gives the SAME G_anh as the loose 1.5e-3 run -> the fixed point is genuine/unique.
  | L | G_ball | G_anh | G_anh/G_ball | conserv |
  |---|---|---|---|---|
  | 1 | 7.16e8 | 5.80e8 | **0.810** | 3.0e-5 |
  | 2 | 6.38e8 | 4.48e8 | **0.702** | 2.3e-4 |
  | 3 | 5.73e8 | 3.98e8 | **0.694** | 3.0e-4 |
  Clean resistive ladder (ballistic->diffusive), reduction grows L1->L2 then ~saturates by L3. NB L=3
  (108 DOF, 15 dense blocks) took 6.8 h -> dense multi-slab is expensive (cf. F19: multi-cell is the
  RGF/distributed regime).
- **Temperature / low-T sweep** at L=1 on a FINE grid (d_omega 0.10, eta_w 0.050 THz, needed because at
  low T only low-freq acoustic channels are populated) (`scripts/out/cnt33_tempsweep`):
  | T (K) | G_ball | G_anh | G_anh/G_ball | iters |
  |---|---|---|---|---|
  | 30 | 5.49e7 | 5.04e7 | **0.919** | 40 |
  | 50 | 9.38e7 | 8.25e7 | 0.880 | 36 |
  | 100 | 2.14e8 | 1.72e8 | 0.805 | 46 |
  | 150 | 3.60e8 | 2.74e8 | 0.762 | 44 |
  | 200 | 5.06e8 | 3.73e8 | 0.737 | 46 |
  | 300 | 7.73e8 | 5.48e8 | **0.709** | 46 |
  **Channel-freezing confirmed:** G_anh/G_ball -> 1 monotonically as T->0 (0.709->0.919): fewer phonons
  populated -> weaker 3-phonon bubble -> transport reverts toward ballistic. G_ball itself freezes out
  (7.7e8->5.5e7). Convergence is marginally easier at low T (36-40 vs 44-46 it). G(T) monotone at L=1
  (no Luisier ~200 K current peak -- expected only for longer backscattering-dominated wires).
- **Cutoff hierarchy** at n_slabs=2 (`scripts/out/cnt33_cutoff`, diagonal "0" vs full "None" for each of
  sigma/vertex/G; G_ball const 6.38e8; full ref G_anh~4.48e8):
  G_anh spans only **~4.09-4.96e8 (+-10%)** across all 8 corners -- the CNT is ROBUST to all truncations,
  NOT the catastrophe Si shows for G-range (F25). In particular **diagonal-G alone is only ~2% off**
  (sInf_vInf_g0 4.39 vs full 4.48). The exact periodic ASR is again why. Useful for the quatrex-solver
  approximation decisions: for clean periodic structures the production diagonal-G default is far less
  harmful than for the open/soft SiNW or the Si film.

### F31 — (8,0) zigzag CNT: new VASP+hiPhive FC3 + anharmonic phph = the phonon analog of the GW electron example
The electron-GW transport example (`examples/w90/carbon-nanotube`) runs electron NEGF + GW screening on an
**(8,0) zigzag** CNT (32-atom cell, period 4.276 A = 3*a_cc, ~6.26 A diameter, semiconducting) with only a
phenomenological "pseudo-scattering" phonon (50 meV Einstein mode, no force constants). Built the real
phonon counterpart: a full FC3 for the same tube and ran anharmonic phph transport on it.
- **Config** `configs/cnt/cnt80_vasp.yaml` (geometry via `ase.build.nanotube(8,0,length=1,bond=1.42)`,
  tube along z, centered in a 17 A vacuum box). VASP-PBE (jiacao build, np=28, ENCUT 400), relax ISIF=2,
  hiPhive supercell [1,1,3] = **96 atoms**, cutoffs [5.0, 3.5], 40 mc-rattled, ardr. Pipeline ran
  end-to-end in ~3.5 h.
- **Fit/validation:** ardr **1021 params, RMSE train 0.149 / test 0.151 eV/A** (test~train, ~2x tighter
  than the (3,3)'s 0.28 thanks to the bigger cell/40 configs); rotational sum rules 150 -> 7.5e-5; FC2/FC3
  ASR < 1e-3. Dispersion **0 imaginary modes**, soft twist 0.0014 THz (softer than (3,3)'s 0.026 -> larger
  diameter), highest optical 47.4 THz. `finite_analysis`: FC2 ASR 1.4e-16, FC3 perm-sym 4.5e-17,
  ASR-legs 1.0e-16, 0 negative PSD -> production-clean, same quality as (3,3).
- **Ballistic baseline** (`scripts/out/cnt80_ballistic`): G_ball(L) = 6.58 / 5.43 / 4.56 / 3.88 e8 for
  L=1-4.
- **Anharmonic L=1** (`scripts/out/cnt80_transport`, eta 0.7, tol 1e-5, 30 iters, conserv 1.3e-3):
  G_ball=6.58e8, G_anh=5.10e8 -> **G_anh/G_ball = 0.775**. G_ball matches the ballistic baseline exactly.
  **(3,3)-vs-(8,0) at L=1, 300 K:** (3,3) 0.810 vs (8,0) 0.775 -- the larger semiconducting zigzag has
  lower per-area G_ball AND slightly stronger anharmonic reduction (more atoms/channels -> more 3-phonon
  phase space). Physically sensible.
- **(8,0) L=2 (192 DOF) is INFEASIBLE in the dense solver** -- one bubble eval grew to ~hours and it
  stalled after iter 13 (killed at 6 h). This is the F19 lesson: multi-cell anharmonic transport for the
  bigger tube needs the RGF/distributed regime, not the dense reference. The (3,3) (36 DOF) is small
  enough that L=2,3 dense were feasible; (8,0) is not. So the (8,0) length ladder is deferred to RGF.

### F30b — SiNW low-T cross-structure check (d5a): does low T rescue the soft-mode non-converger?
Ran the same fine-grid low-T sweep on the d5a SiNW (the soft-mode wire that did NOT converge at 300 K in
the prior eta/lambda work) (`scripts/out/d5a_tempsweep`, L=1):
| T (K) | G_ball | G_anh | G_anh/G_ball | iters | conserv |
|---|---|---|---|---|---|
| 30 | 1.36e7 | 1.17e7 | 0.860 | 44 | 3.9e-3 |
| 50 | 1.86e7 | 1.52e7 | 0.817 | 87 | 1.7e-2 |
| 100 | 2.59e7 | 2.11e7 | 0.815 | 83 | 1.4e-2 |
| 150 | 3.08e7 | 2.54e7 | 0.825 | -- | -- |
- **Low T DOES rescue convergence:** d5a converges to resid <1e-5 at 30-150 K (it failed at 300 K) --
  fewer populated channels -> weaker bubble -> reachable fixed point. Confirms the user's intuition for
  the hard case.
- BUT unlike the CNT, d5a's **heat-flow conservation stays poor (~1-2%)** and its ratio is ~flat
  (0.82-0.86, not monotone) -- the soft-mode causality defect is a SEPARATE, T-growing error on top of
  the convergence behavior. Contrast: the CNT held conservation ~1e-4 and a clean monotone ratio.
- The full d5a high-T (200/300 K) and a d11a (45-atom, 135-DOF) low-T sweep were SKIPPED as
  cost-prohibitive (d11a ~30 h dense; d5a 300 K is the documented non-converger). d5a's 4 points + the
  CNT cover the cross-structure story.

### F32 — Bulk-Si FC4 (VASP) + SCP loop/tadpole spectral function: the tadpole-consistency bug, debugged
Merged the FC4/SCP/spectral machinery (the "unify dense solver" branch) and computed a bulk-Si FC4 to
run the self-consistent-phonon (SCP) spectral function with/without the cubic tadpole.
- **FC4 via VASP, parallelized.** QE was too slow (perl-wrapped serial `pw.x`); switched to VASP
  (`configs/si_primitive/hiphive_fc4_vasp.yaml`, 2x2x2/16-atom, 48 rattled, cutoffs [3.8,3.0,3.0]).
  Ran the 48 jobs **8-wide in parallel** (np=16 each, `--bind-to none`; the `fc3-hiphive-run` driver has
  no parallel flag, so a manual xargs -P 8 pool over the sown disp dirs, then reap). ardr fit: 9 params,
  RMSE 0.078, FC4 ASR ok, **58 compact-reference quadruples** -> `fc4_atoms/fc4_values` in fc3.hdf5.
  (Reusing the 5x5x5 si_big displacements for FC4 was rejected: the fit would work but the dense FC4
  export is 250^4 = 2.36 TiB; would need a sparse exporter.)
- **The tadpole-consistency check FAILED at first: ||Sigma_T||=33** (should be ~0 for symmetric diamond
  Si). Debugged via `scripts/verify/tadpole_diag.py`. Physics first: the cubic tadpole is the thermal
  force <F_a>=1/2 Phi3:<uu>; by the Td SITE symmetry of diamond Si a site vector is symmetry-forbidden,
  so <F>=0 -> Sigma_T=0 (no internal sublattice relaxation, only volume expansion). So it SHOULD be ~0.
- **Two compounding bugs (neither the FC4 index mapping, nor non-eq physics):**
  1. **<uu> source.** It was read from the OPEN 1-cell transport device G^< (`equal_time_uu`), which
     over-counts the low-omega acoustic displacement variance **~14x** (<w^2>=2.23 vs the
     Debye-Waller-correct 0.154 amu*A^2; verified isotropic, so NOT a transport-direction anisotropy).
     A single transport cell has the wrong phonon DOS for a bulk property. Fix: source <uu> from the
     **bulk BZ-summed equilibrium mode sum** (new `static_se.bulk_equilibrium_uu`; reproduces the Si
     Debye-Waller u^2=0.0055 A^2 exactly, mesh-converged at 8^3).
  2. **`enforce_asr=False`** (transmission's default, which the SCP driver was inheriting). The
     Gamma-summed device FC3 then carries a large translation-non-invariant artifact; its symmetry-trace
     `sum_c Phi3[a,c,c]` = +-0.096 (∝[-1,1,1], opposite on the two inversion-paired atoms) breaks the
     Td-zero. `enforce_asr=True` removes it (trace -> 1e-17, ||Phi3|| 1.28 -> 0.335).
- **Result: ||Sigma_T|| 33 -> 0.355 (93x), now ~0** (0.15% of the optical omega^2). The spectral A(q,w)
  with vs without the tadpole now COINCIDE (as physics demands); only the small physical SCP loop shift
  remains. Magnitudes at 300 K: dynamic **bubble** max|Sigma^R|~254 THz^2 (mostly linewidth) >> **loop**
  Sigma_L~5 (real SCP renormalization, ~1%) >> **tadpole** Sigma_T~0.35 (~0). Residual 0.355 = the fitted
  FC3's tiny leftover off-diagonal point-group asymmetry (trace channel now exact); a tighter/symmetrized
  FC3 would zero it.
- **SCP scheme:** it is NOT frozen-SCP-then-bubble -- after a loop-only pre-stage, the loop+tadpole are
  recomputed every SCBA iteration from the current G^< (`loop_propagator='loop_only'`).
- Code: `static_se.bulk_equilibrium_uu` + `fixed_uu` hook arg; `dense.transmission(static_uu=...)`;
  `bulk_si_scp.py` now defaults to bulk <uu> + enforce_asr (with `--device-uu` to revert);
  `scripts/verify/tadpole_diag.py`. Figure `/tmp/claude/si_scp_asr/spectral.pdf`.
