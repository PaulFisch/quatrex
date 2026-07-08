# d5a η=0: the divergence seed, found by surgical frequency ablation (2026-07-07/08)

Companion to `spectral_deformation_audit.md` (the knob verdicts) and the conservation
appendix continuity work. Data: `phonon/studies/out/d5a_bisect/` (ablation tree),
`phonon/studies/out/d5a_gridladder/` (raw grid rungs), `phonon/studies/out/local_mw/`
(cnt33 bare ladder). All runs: sinw_d5a L2, T=300 K, η=1e-12, retarded=fft, linear 0.1
(unless stated), NO taper, NO window, NO masks — fully raw except the stated ablation.
Tool: `sse_zero_bands_thz` (diagnostic hard-zero of Σ in [lo,hi] windows, consistently on
the bubble input G legs, the output Σ^≷ and the KK Σ^R).

## The verdict

**The divergence seed is the sub-grid infrared: the soft twist modes at 0.0075–0.027
THz** (20–50× below the first grid bin of any affordable grid) whose 1/ω Bose weight
enters the bubble unresolved each iteration. A 1.5 THz low-frequency cutoff — the
phono3py `cutoff_frequency` analog, the one citable hard exclusion — is NECESSARY and
SUFFICIENT:

| run | ablation | outcome |
|---|---|---|
| nf181 raw | none | ABORT it 70 (drivers 47–60 THz = stretch−Si difference band) |
| nf361 raw | none | ABORT (residual ~600 by it ~50) |
| nf721 raw | none | ABORT it 90 (4.5e7) — grid refinement does NOT help |
| nf181 freeze>43THz | high-ω zeroed | ABORT it 46 — EARLIER; driver relocates to the last live bin (43.267 THz, ×2e18): squeeze-the-balloon |
| `gapzero` [[28,45]] | gap zeroed, IR live | ABORT it 62 (drivers relocate to 45–56 + 65) |
| `flatzero` [[16.5,28]] | flat Si-H bending zeroed, IR live | ABORT it 143 — flat bands are an AMPLIFIER (delay 70→143) but not the seed |
| **`ircut`** (cutoff 1.5 THz) | IR only | **descends monotonically**: 1.0 → 3.1e-4 in 150 its (≈3× per 25), heat [5.244, 5.139, 5.177], best lead balance 1.4e-5 |
| `ir_gap`, `ir_flat` | IR + extra windows | stable (150 its) — everything is quiet once the IR is cut |

Divergence mechanism (from the per-iteration spectra, QX_DIAG_SPECTRAL): the unresolved
IR weight compounds through the bubble; the growth shows up first in two-phonon
combination territory (physical support — the 24–30 and 47–60 THz windows have zero
one-phonon DOS but two-phonon JDOS comparable to in-band); once |Σ| exceeds the
dynamical-matrix scale, the GLOBAL Kramers–Kronig Re Σ^R shifts every pole (measured:
in-band G-DOS collapses 1.61→1e-3 while Σ pumps to 1e17) — which is why every non-IR
hard cut merely relocates the blow-up to its own boundary.

Amplifiers (real, but not seeds): the flat Si–H bending bands (removal delays the abort
2×); the Si–H stretch island 63.1–64.1 THz, which has EXACTLY zero 3-phonon decay phase
space (max two-phonon sum of all other bands = 53.5 THz < 63.1 ⇒ Γ_3ph(stretch) ≡ 0 —
undamped δ-poles at any grid; conversely no energy-conserving 3-phonon process involves
the stretch at all, so it is exactly decoupled from on-shell transport at this order).

## Mixer acceleration (on the contractive ircut map)

The historical "Anderson/Broyden fail at η=0" verdicts were artifacts of the MARGINAL
(IR-contaminated) map. On the cut map:

| mixer | residual @150 its | G·dω |
|---|---|---|
| linear 0.1 | 3.1e-4 | 1.9226 |
| **RRE (cycle 8, β 0.1)** | **2.6e-5** | 1.9397 |
| Anderson (depth 5, warmup 20, β 0.2) | 2.6e-2 | 1.9249 |

G·dω is mixer-invariant to ~1% — a genuine fixed point. RRE is the production
accelerator; Anderson still underperforms.

## Physics cost of the ablations (why only the IR cut is admissible)

The extra windows CHANGE the conductance — they delete real two-phonon scattering:
`ir_gap` G·dω = 2.066 (+7%), `ir_flat` = 2.271 (+18%) vs ircut 1.923. The IR cut itself
also removes real low-ω scattering; its cost is quantified the honest way — an
ω_c → 0 sweep (the phono3py-style extrapolation), which is the designated follow-up.
Alternative for the sub-ω_c physics: η_floor = c·dω adaptive smearing below ω_c only
(Phoebe/Yates-style, citable).

## cnt33 (the dispersive control, settled earlier)

Fully raw bare SCBA converges at every grid, FASTER on finer grids (220/150/111 its at
nf 181/361/541); G·dω Cauchy: 13.62 → 12.91 → 12.86 (last step 0.4%). The old
IR-occupancy-taper value (17.05, "wreg→0 extrapolation 17.4") is a +33% artifact of the
taper's Lorentzian tail — the report headline must move to the bare 12.86
(ratio to ballistic 23.73: 0.54, not 0.72). No sub-grid soft modes → no IR seed → no
regularizer needed at all.

## The η=0 recipe (post-bisection)

- dispersive (cnt33-class): fully raw + grid refinement + RRE.
- soft-mode systems (d5a-class): raw + `sse_low_freq_cutoff_thz` ≈ 1.5 THz (above the
  soft modes, below the transport-relevant acoustics — sweep ω_c for the extrapolation)
  + RRE. Everything else (windows, tapers, freezes, floors) off.
- open: the ω_c sweep + grid ladder ON the cut recipe (the flat-band resolution
  question, now testable on a converging system); the tightened-tolerance RRE run
  (150-it budget stopped at 2.6e-5, still descending).
