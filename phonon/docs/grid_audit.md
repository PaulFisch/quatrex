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

### RESOLUTION — the accuracy half holds, the stability half does not

| claim | verdict | measurement |
|---|---|---|
| #2 discrete pole weight `W ≈ (Δω/π)Γ/(d²+Γ²)` | **TRUE (exact)**; amplitude of the *swing* overstated ~4× | closed form reproduces the peak-bin weight to **1.0000** at dw/Γ = 0.25…64; sub-cell swing 1.02/1.25/5.0/65/862 — quadratic in dw/Γ as claimed, prefactor ≈ 1/4 |
| #1 **accuracy** half (`Δω ≲ Γ_anh` for the integrals) | **TRUE, and worse than reported** | at dw/Γ = 2.9 the measured linewidth swings **0.094–0.162 THz across one cell = 73 %**, against the 25 % quoted at `62:145` |
| #1 **stability** half ("…and for the stability of the map"), with #3's `\|λ\| ~ Δω/Γ > 1` | **FALSE as stated** | ρ(J) is **non-monotonic** in dw/Γ and **smallest on the coarsest grid**: at Γ=0.02, dw/Γ = 58.7 → ρ = **0.13** (converges) where the claim predicts ≈ 59. At Γ=0.2, ρ ≈ 0.66–0.83 over dw/Γ = 0.37…5.9 — no trend |
| #3 alignment multiplies the gain | **mechanism TRUE, magnitude ~1.4× optimistic** | at fixed dw, translating the pole across one cell moves ρ **0.72 → 1.51** (swing 2.10 vs predicted dw/Γ = 2.93) and **crosses 1** — alignment alone can destabilise a marginal cycle |
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
