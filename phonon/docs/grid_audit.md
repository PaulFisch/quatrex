# Grid resolution & extent: what the thesis claims, and what measurement says

2026-08-07, in progress. Scripts: `phonon/studies/_grid_sideband.py`,
`_grid_stability_law.py`; data under
`phonon/studies/out/grid_audit/`. 48 claims were extracted from
`theory/40_scba.tex` §`sub:grid_resolution`, `theory/50_computation.tex`
§`sub:phph_freq`/§`sub:phph_nugrid`, and the grid passages of
`results/62,63,64,60,75,80` + the `phonon_solver`/`conservation`
appendices.

## Provenance finding (before any measurement)

**Neither theory file contains a single `% REVIEW` comment.** All 89 in
the thesis are in `results/` and `appendices/`. The theoretical grid
apparatus carries no inline evidence trail. Two `\todo`s sit inside the
passages: `50_computation:75` ("Check this agian", attached to an
asserted FFT-accuracy resolution requirement that is **never stated
anywhere in the thesis**) and `50_computation:106` ("Measure the
sideband weight above 2ω_max"), i.e. the extent truncation error was
admittedly never measured. Four of the sharpest quantitative
confirmations (#1-accuracy, #2, #4, #15) come from one synthetic
two-site chain and were never reproduced on a device.

## Verdicts so far

### EXTENT — the rule is right, and now has numbers

| claim | verdict | measurement |
|---|---|---|
| #11 ballistic bubble supported on [−2ω_max, 2ω_max] (`eq:conv_support`) | **TRUE, machine-exact** | first-Born Σ carries **3.8e-10** of its weight beyond 2ω_max (legs 2.8e-10) |
| #12/#16 self-consistency breaks the bound; each extra sideband order costs one power of \|Φ³\|² | **TRUE, quantified** | converged fixed point leaks **1.5e-4 / 1.5e-3 / 3.9e-3** beyond 2ω_max at g = 0.75/2.4/4.0e18; relative weight scales as **g^1.95** — one extra \|Φ³\|² per order, as claimed |
| #15 the grid must reach 2ω_max to keep the two-phonon shoulder | **TRUE, understated** | the shoulder (ω_max, 2ω_max] carries **15–35 %** of the scattering weight at the fixed point vs 0.3 % in first Born — it is *grown* by self-consistency |
| #13 truncation error of cutting at the top | **measured for the first time** | see the ladder below |

Truncation error at **identical dw** (reference = 4ω_max):

| top/ω_max | 3.0 | 2.0 | 1.75 | 1.5 | 1.35 | 1.2 | 1.1 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| ΔΓ/Γ | 1.2e-7 | **5.7e-4** | 5.7e-3 | 2.7e-2 | 6.7e-2 | 1.9e-1 | 3.7e-1 | n/a |
| ΔJ/J | 1.2e-8 | **6.3e-5** | 6.4e-4 | 2.5e-3 | 4.1e-3 | 7.3e-3 | 1.3e-2 | 4.7e-2 |

So: cutting at exactly 2ω_max costs 6e-4 in linewidth and 6e-5 in
current; **going beyond 2ω_max buys nothing** (1e-7 at 3×); the error
only becomes serious below ~1.75×. The terminal current is an order of
magnitude less sensitive than the spectral width — consistent with the
shoulder entering through Kramers–Kronig rather than through terminal
heat.

### RESOLUTION — three regimes: blind → partially registered → resolved

**Paul's hypothesis (2026-08-07), CONFIRMED.** Coarse-grid convergence
is not evidence that coarse grids work: the grid does not *register* the
resonance, so the map is nearly ballistic and trivially contracting
while the answer is wrong. The registration fraction is the exact
per-orbital spectral sum rule `S_i = ∫2ω(−1/π)Im G^R_ii dω = 1`,
restricted to the flat band whose entire width is anharmonic
(`_grid_regimes.py`, 8 resolutions × 5 sub-cell offsets, medians):

| nfreq | Δω | **S_B** | Γ_FB(own grid) | Γ_em | **ρ_raw** | ρ_damped | \|ΔΓ\|/Γ_ref | regime |
|---|---|---|---|---|---|---|---|---|
| 30 | 1.173 | **0.047** | 0.0071 | 0.683 | **0.13** | 0.826 | 34.8 | blind |
| 60 | 0.587 | 0.130 | 0.0139 | 0.325 | 0.39 | 0.863 | 16.1 | blind |
| 120 | 0.293 | 0.863 | 0.0198 | 0.153 | 1.12 | 0.950 | 7.03 | transition |
| 240 | 0.147 | 0.814 | 0.0158 | 0.082 | 1.40 | 0.954 | 3.32 | transition |
| 480 | 0.073 | 0.916 | 0.0220 | 0.049 | **1.50** | 0.935 | 1.58 | transition |
| 960 | 0.037 | 0.987 | 0.0200 | 0.035 | 1.22 | 0.933 | 0.84 | resolved |
| 1920 | 0.018 | 1.000 | 0.0188 | 0.024 | **0.79** | 0.932 | 0.27 | resolved |
| 3840 | 0.009 | 1.000 | 0.0184 | 0.019 | 0.82 | 0.929 | 0.00 | resolved |

All three predicted regimes appear:

1. **Blind.** At Δω/Γ ≈ 59 the grid registers **4.7 %** of the
   resonance, the raw loop gain is **0.13** (trivially contracting), and
   the answer is wrong by a factor 35. The a-priori check agrees
   independently: the first-Born linewidth computed with that rung's own
   quadrature is 0.0071 against a grid-converged 0.0184 — 60 % of the
   physics is invisible to the grid. Ballistic control: S_B = 0.000 at
   every rung, so the *difference* between the anharmonic and ballistic
   states at nfreq = 30 is 4.7 % of one orbital.
2. **Transition.** As the grid starts to sample the line, ρ_raw rises
   through 1 and **peaks at 1.50** (nfreq = 480, S_B ≈ 0.92).
3. **Resolved.** Once S_B → 1.000, ρ_raw falls **back below 1**
   (0.79–0.82) and the linewidth converges to the reference.

So the non-monotonicity is not a puzzle: it is the signature of passing
through partial registration. **Refining does not monotonically
destabilise, and coarse stability is worthless.**

Bonus finding: **the measured linewidth is a pure grid artefact until
resolution** — Γ_em ≈ Δω/2 at every under-resolved rung
(1.173→0.683, 0.587→0.325, 0.293→0.153, 0.147→0.082, 0.073→0.049) and
only detaches once Δω < 2Γ_true ≈ 0.038. A "converged" coarse run
reports its own grid spacing as the physics.

#### How fine is actually enough (the ladder pushed to Δω/Γ = 0.06)

S_B saturating at 1.000 does **not** mean the observable has converged.
Judged by the successive-rung change `|Γ(n) − Γ(n/2)|/Γ(n)` (the
distance to the finest rung is 0 by construction and must not be used):

| nfreq | Δω/Γ | Γ_spectral | d(spec) | Γ_from-Σ | d(Σ) | ρ_raw |
|---|---|---|---|---|---|---|
| 1920 | 0.9 | 0.02423 | 44 % | 0.02666 | 0.8 % | 0.79 |
| 3840 | 0.5 | 0.01906 | **27 %** | 0.02671 | 0.2 % | 0.82 |
| 7680 | 0.24 | 0.01824 | 4.5 % | 0.02674 | 0.1 % | 0.82 |
| 15360 | 0.12 | 0.01796 | 1.5 % | 0.02674 | 0.0 % | 0.82 |
| 30720 | 0.06 | 0.01791 | **0.3 %** | 0.02675 | 0.0 % | 0.82 |

The successive changes fall 27 → 4.5 → 1.5 → 0.3 %, i.e. roughly
second order in Δω. **1 % accuracy on the spectral width needs
Δω/Γ ≈ 0.12 — about 8 points per half-width** — a factor ~10 stricter
than `eq:grid_resolution`'s Δω ≲ Γ_anh (1 point per linewidth). ρ_raw
is flat at 0.82 across the whole resolved regime, so regime 3 is
genuinely stable, not marginally so.

The two width readers converge to **different** values, 0.0179
(spectral FWHM) vs 0.0267 (−Im Σ^R/2ω), and both are right: their ratio
0.67 is the quasiparticle residue `Z = (1 − ∂ReΣ/∂ω)^-1`. The Σ-based
reader converges ten times sooner (0.1 % by nf = 7680) because it
samples one bin; the spectral width is the demanding observable. Any
"resolved" claim must say which width it means.

#### A latent code trap this exposed

At nf = 30720 the cold-started run reported **converged at iteration 1**
with S_B = 0.006 and Γ = 6e-4 — it had never iterated and sat on the
ballistic branch. Cause: the SCBA residual `‖f‖/‖sol‖`
(`phonon/solver/dense.py:900`) is **grid-dependent at the first
iteration** — 5.6e-5 at nf = 15360 but 8.8e-8 at nf = 30720 — so against
a fixed `tol` the finest grid trips the convergence test immediately.
Warm-starting the same grid from the nf = 15360 fixed point runs 24
iterations to S_B = 1.0000, Γ = 0.01791, exactly on the convergence
trend. **`scba_tol` is not grid-invariant**, and the same signature
(“converged in 1 iteration”) is already sitting in the committed
`out/toy_grid/results_ref.json`. Production tolerances (1e-3) are far
from this threshold, but any fine-grid or tight-tolerance study must
guard on the iteration count — `_grid_regimes.py` now re-runs any rung
that stops in ≤ 2 iterations from the previous rung's solution.

Control: S_A (dispersive band) stays 0.70→0.93 across the whole ladder,
saturating below 1 only from fmax truncation — the deficit is
specifically the flat band, not global.

### RESOLUTION — the accuracy half holds, the stability half does not

| claim | verdict | measurement |
|---|---|---|
| #2 discrete pole weight `W ≈ (Δω/π)Γ/(d²+Γ²)` | **TRUE (exact)**; amplitude of the *swing* overstated ~4× | closed form reproduces the peak-bin weight to **1.0000** at dw/Γ = 0.25…64; sub-cell swing 1.02/1.25/5.0/65/862 — quadratic in dw/Γ as claimed, prefactor ≈ 1/4 |
| #1 **accuracy** half (`Δω ≲ Γ_anh` for the integrals) | **TRUE, and worse than reported** | at dw/Γ = 2.9 the measured linewidth swings **0.094–0.162 THz across one cell = 73 %**, against the 25 % quoted at `62:145` |
| #1 **stability** half ("…and for the stability of the map"), with #3's `\|λ\| ~ Δω/Γ > 1` | **FALSE as stated** — the true control parameter is *registration*, not Δω/Γ | ρ_raw is **non-monotonic** in Δω/Γ and **smallest on the coarsest grid** (0.13 at Δω/Γ = 58.7, where the claim predicts ≈ 59) because the coarse grid is blind. The instability lives in the *transition*, and refining past it restores ρ_raw < 1 |
| #3 alignment multiplies the gain | **mechanism TRUE, magnitude ~1.4× optimistic** | at fixed Δω, translating the pole across one cell moves ρ_raw **0.72 → 1.51** (swing 2.10 vs predicted Δω/Γ = 2.93) |

**Stability convention (correction).** `arnoldi_spectrum` returns the
Jacobian of the **raw** map F. The SCBA iterate is damped,
`x ← (1−a)x + aF(x)`, so its eigenvalue is `m = 1 − a + aλ` and the
convergence criterion is `|m| < 1`, **not** `|λ| < 1` — which is why
rungs with ρ_raw > 1 still converge at a = 0.2. Earlier statements in
this file that ρ "crosses 1" refer to ρ_raw and are *not* convergence
statements. Measured ρ_damped stays 0.83–0.95 across the whole ladder
(every rung converges); the physics signal is in ρ_raw.
| #4 the divergence is "erratic rather than geometric" | **supported** | ρ(offset) is non-monotonic: peak 1.51 at offset 0.875, minimum 0.72 at 1.0 |

**Reading:** the criterion `Δω ≲ Γ_anh` is a *quadrature-accuracy*
criterion, not a stability criterion. What actually destabilises is
pole–grid **registration** at fixed resolution (#3/#4), which is a
different statement and the one the thesis under-weights. Refining the
grid does not monotonically help stability, and on this bed it hurt
(ρ peaked at dw/Γ ≈ 7).

## Method traps found (both would have produced wrong verdicts)

1. **A naive extent ladder is confounded by re-registration.** Holding
   the *target* dw fixed while changing the top rounds `nfreq` per rung,
   shifting the actual dw by 0.4 % and the measured width by 5 % — larger
   than the truncation effect under test, and via claim #4's own
   mechanism. The ladder must hold dw *exactly* fixed and let the top
   land on a multiple of it.
2. **The existing FWHM reader quantises the width in dw**
   (`_toy_grid_study.measure_gamma` takes grid-point crossings), which on
   an extent ladder appears as a spurious constant offset (an identical
   3.7e-3 "error" at 1.5×, 2× and 3×). Interpolated crossings are
   required.

## Still open

- **#30 (highest priority)**: the §63 η=0 "grid lottery" may be a
  `b_G=1` artefact — `64:49` says that exact 161/201/361 lottery is
  "closed" by the exact band. Rescan CNT L4 at `sse_g_band=3`
  (queued on tortin, needs 3 idle nodes).
- **#23**: the adjoint-restriction claim has no system, no numbers, no
  REVIEW anywhere — needs a named A/B.
- **#43's supporting number does not exist**: `conservation.tex:297`
  quotes a d5a floor halving from 101→201 frequencies; no 101- or
  201-point d5a run exists on disk.
- **#34**: "resolved" is defined four incompatible ways; under the
  sharpest, production CNT (181 pts, Δω = 0.31 THz) sits at ~0.8
  points per measured anharmonic HWHM (`resonance_gain_distilled.npz`:
  L2 fixed point Γ_anh min 0.365, median 0.920 THz) — while `62:257`
  calls all 72 modes resolved.
- Device-scale confirmation of the two synthetic verdicts above.
