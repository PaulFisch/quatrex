# Spectral-deformation audit: every knob that touches Σ or G, and whether it is physics

2026-07-06. Trigger: the interface-continuity ledger (conservation appendix) and the
grid-ladder diagnostics showed that several η=0 "stabilisation" knobs deform the physics
they were meant to protect. This audit lists EVERY deformation of Σ^≷/G in the production
phonon SCBA with a physicality verdict. Companion data: `phonon/studies/out/local_mw/`
(cnt33 ablations), `phonon/studies/out/d5a_gridladder/` (d5a rungs + two-phonon DOS),
figure `d5a_grid_ladder`, `eta0_knob_{ablation,sensitivity}`.

## The two criteria

1. **Support**: the three-phonon Σ^≷(ω) is a two-phonon convolution — its physical
   support is the JOINT (sum/difference) two-phonon spectrum, NOT the harmonic
   one-phonon bands. Anything that confines Σ to the harmonic support deletes real
   combination continua (measured on d5a: the 24–30 THz "gap" bins have zero one-phonon
   DOS but two-phonon weight comparable to in-band bins).
2. **Ledger**: at a fixed point the per-interface energy continuity
   J_{k+1} = J_k + P_abs(k) must close (it does, to the SCBA residual, for the bare
   bubble); a knob that breaks it interior-only is deforming the transport physics even
   when every global gate stays green.

## Verdicts

| knob | what it deforms | verdict | status |
|---|---|---|---|
| `sse_smooth_window` + `support_taper_cells` | multiplies Σ^≷ (and the bubble input G) by a harmonic-support window | **unphysical** — deletes two-phonon continua; its half-suppressed ramp bins were the measured d5a divergence drivers (growth ×1e6 exactly on the ramp) | **REMOVED 2026-07-06** |
| `ir_taper_cells` | multiplies the CONTACT injection occupancies by ω²/(ω²+ω_reg²) | **unphysical** — Lorentzian tail suppresses scattering 10% at 3ω_reg; +33% G on cnt33 (17.05 vs the grid-converged bare 12.86); breaks the interior ledger by ~5%; no literature precedent | deprecated: 0 in every recipe; feature retained only so the committed ω_reg-sweep runs re-validate; do not use |
| `band_limit_sse` (band-top mask) + `band_support_margin_thz` (hard support mask) | zeroes Σ^≷ outside the harmonic support (hard edges) | **unphysical** for the same support reason (+ Gibbs ringing of hard edges through the KK Σ^R) | default off; flagged — remove after the raw-d5a verdict; note the cnt33 bare ladder ran with the band-top mask on, effect above 52 THz only (re-verify off) |
| `sse_freeze_occupation` | zeroes Σ at thermally-frozen bins | unphysical in principle (frozen modes still have zero-point Σ^> structure), negligible where used | default off; flagged |
| `spectral_sharp_cap` / hysteresis masks | live-A(ω) masks | unphysical + limit-cycle-prone (G-dependent mask) | default off; superseded |
| `sse_zero_bands_thz` | hard-zeroes Σ in arbitrary [lo,hi] THz windows | **diagnostic only** — the bisection tool for localising η=0 runaway seeds (added 2026-07-06); same two-phonon deletion as any hard mask | never in production |
| `sse_low_freq_cutoff_thz` | zeroes Σ below ω_c (transport stays ballistic there) | **defensible approximation** — the phono3py `cutoff_frequency` analog (PRB 91, 094306 practice); hard edge needs the consistent post-Hilbert mask (in place) | opt-in, off by default |
| `eta_ir_floor_cells` (+anneal) | broadening floor η = c·dω | **defensible numerics** — the grid-ADAPTIVE smearing of the BTE literature (Phoebe arXiv:2111.14999, Yates-style); vanishes with the grid; annealable to 0 | the designated fallback if raw d5a stays divergent |
| ω=0 single-bin zero of Σ^≷ (+ post-Hilbert) | one bin | benign (zero measure, zero heat, tames the singular acoustic G^R(0)) | keep |
| `eta` / `eta_obc` | uniform (contact) broadening | standard NEGF regularisation, η→0 extrapolable (the conservation appendix quantifies the commutator budget) | keep; η=0 program supersedes |
| retarded `"half"` | drops Re Σ^R | non-causal at η=0 (breaks the Φ-derivable pairing; −6% G at L2, limit cycles at L≥3) | `"fft"` mandatory at η=0 (already the default) |

## Non-knob approximations that DO bound the physics (inherent, quantified)

- **G-range truncation** (BT band |K−K'|≤1 in the ring): the largest known systematic —
  −18.5% on the cnt33 cutoff cube (g0 corner). Inherent to the RGF band; quantified,
  not removable at fixed solver structure.
- **NN vertex truncation** (`fc3_to_phi_blocks` nn_only): −5.7% (cutoff-cube v0 corner),
  warn-gated at build time.
- **fmax window**: benign for in-window transport (the one-sided fold makes in-window Σ
  complete when fmax ≥ the one-phonon band top); two-phonon weight ABOVE fmax only
  affects G there (no transport window). d5a needs fmax=66 (Si–H stretch) — the old
  fmax=18 truncation was the ORIGINAL source of the d5a divergence story that motivated
  the (now removed) support masks.
- **Ideal-harmonic leads** (no anharmonicity in the contacts): modelling choice
  (Luisier PRB 86, 245407), stated in the report.

## The η=0 recipe after this audit

cnt33-class (dispersive): **fully raw** — fft retarded, plain linear, no masks, no
taper, grid-refine (bare G·dω Cauchy at 12.86: 181→361→541 = 13.62/12.91/12.86, and
convergence gets FASTER on finer grids).
d5a-class (soft modes): SETTLED by the ablation bisection (d5a_eta0_bisection.md) —
raw + `sse_low_freq_cutoff_thz` ≈ 1.5 THz (the citable phono3py-style exclusion of the
sub-grid soft twist modes, the proven necessary-and-sufficient seed treatment) + RRE.
The flat bands/stretch island are amplifiers only; grids do not fix the seed.
