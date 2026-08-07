# MoS2 film: why the current "does not conserve"

2026-08-07. Reproduce: `phonon/studies/_mos2_dispersion_audit.py`,
`phonon/studies/_asr_project_film.py audit`,
`python -m phonon.studies conservation run --fc3 <build>`,
`phonon/scripts/figures/conservation_diagnosis.py`; daint jobs
4378427 (`mos2f3-bbcheck`) and 4378429 (`mos2f3-bbcheck-lm15`).
Artifacts: `phonon/studies/out/mos2_conservation/`.

## Verdict

**No assumption of the conservation derivation
(`document/src/theory/40_scba.tex` §`sub:energy_conservation`) is
violated.** The Phi-derivable bubble identity `P_in = P_out` holds on
the MoS2 film to **1e-8 ... 1e-9** when measured correctly. The
"conservation failure" was two separate things:

1. a **measurement artefact** in every saved run npz (below), and
2. the genuine per-iteration lead imbalance, which is **the SCBA
   residual expressed in energy units** — it is O(1) exactly when the
   iteration is diverging and 2e-3 when it is descending. It is not an
   independent defect.

The film's problem remains what it always was: no accessible fixed
point at eta = 0 (loop gain), not a broken conservation law.

## The three physical preconditions — all clean

| check | MoS2 film | control |
|---|---|---|
| acoustic dispersion, in-plane | alpha = 1.000, 1.000, 1.000 | Si 1.000 |
| acoustic dispersion, cross-plane | alpha = 1.004, 0.998, 1.000 (4.5x softer) | CNT **2.06, 1.97** (flexural) |
| imaginary modes, sampled 5x5 x q_z | none | none |
| imaginary modes, fine 11x11 x q_z | none | none |
| fc3 ASR, interior classes | 1.7e-5 | — |
| fc3 ASR, edge classes | 4.5e-3 (structurally unenforceable) | — |
| qfold round-trip / torus leakage | 4.0e-16 / 2.0e-16 | — |
| S3 permutation symmetry (Gamma rep) | 2.9e-16 (ls), 2.2e-16 (scp), 2.6e-16 (o4), 1.3e-16 (mos2f3) | Si 2.0e-16 |
| S3 (folded cell rep) | 5.8e-14 | — |
| H(q) hermiticity / H(-q) = H(q)* / real-space reality | 3e-14 / 2e-13 / 1.5e-14 | — |
| discrete bubble replica f64 -> f128 | 2.9e-16 -> 2.4e-19 | — |

The dispersion check **inverts** the natural hypothesis: the system
with quadratic (flexural) branches is the CNT, which conserves to
5e-16; MoS2's branches are linear and it is the one that fails. Note
the CNT's alpha ~ 2 is the same non-linear opening that invalidated
the CM-channel extraction there (`ir_residue_derivation.md`).

## The measurement artefact (this is the actionable part)

`bubble_balance_spectrum`, `final_bubble_balance` and
`slab_absorption` are produced in `phonon/studies/engine/run.py:393-397`
— i.e. **after `scba.run()` returns**, when `data.sigma_*` holds the
**mixed** self-energy of the last `_update_sigma()`. The identity
`P_in = P_out` is an algebraic property of the pair
`(Sigma = bubble[G], G)`; a mixed Sigma is not that pair, so the
post-hoc number measures `||Sigma_mixed - bubble[G]||`, i.e. the SCBA
residual, not conservation. `scba.py:975-978` says exactly this
("a mixed-iterate pairing need not balance") but only guards the
in-loop call at `scba.py:884`, which fires only under
`bubble_balance_check` (default False, and set in no MoS2 run before
2026-08-07).

Calibration on the controls, same run, two call sites:

| run | pre-mixing (in-loop) | post-hoc (saved) |
|---|---|---|
| CNT33 L4, converged | **5.3e-16 ... 5.1e-19** | 2.8e-8 |
| Si film nk9r dense | 2.1e-8 ... 6.3e-8 | 3.1e-7 |
| Si film L3 SCBA | (not recorded) | 1.5e-5 |
| MoS2 film, all runs | (never recorded until now) | **O(1) ... O(6)** |

Eight orders of inflation on a *converged* system. Consequently
`conservation_diagnosis.py`'s "BUBBLE/g_band" verdict on any MoS2
run npz is not trustworthy: it reads post-hoc arrays. The tool now
warns when `iter_bubble_balance` is absent.

## The correct measurement (`QX_BBCHECK=1`, pre-mixing)

Uniform-2001 build, o4 15-block vertex, tadpole off, eta = 0:

| iteration | bare: Sigma resid / lead bal / **bubble** | mask 1.5: Sigma resid / lead bal / **bubble** |
|---|---|---|
| 1 | 1.00 / 1.4e-14 / **4.1e-10** | 1.00 / 1.4e-14 / **1.4e-7** |
| 2 | 4485 / 0.547 / **4.8e-6** | 0.982 / 2.8e-3 / **3.0e-8** |
| 3 | 5425 / 0.501 / **2.5e-6** | 0.844 / 2.9e-3 / **2.0e-9** |
| 4 | diverged (6495) | 0.716 / 2.3e-3 / **3.1e-8** |

Read it: the **bubble stays conserving even while the iteration
explodes** (4.8e-6 at Sigma residual 4485). The lead balance tracks
the Sigma residual, not the bubble — 0.5 when the residual is 4485,
2e-3 when the residual is O(1) and falling. In the masked run, where
the iteration descends, current conservation is good (2e-3) and the
bubble balance *improves* to 2e-9.

## Consequences

- The lead-balance = 2.0 states seen in the CM-subtraction and late
  masked runs are diverging-iterate states (Sigma residual >> 1), not
  a violated identity. `balance = 2` means `h_L = -h_R`, which is what
  a blown-up Sigma produces; it is a divergence symptom.
- **Any future conservation claim must use the pre-mixing measurement**
  (`QX_BBCHECK=1`). The post-hoc arrays are a residual proxy.
- Thesis: `eq:bubble_balance` is `\cref`'d from five places
  (`results/60_eta0.tex:20,241`, `theory/50_computation.tex:166,309,818`)
  and **has no `\label` anywhere** — the one identity this campaign
  turned on is the one the thesis never writes down. It should be
  stated as `P_in = P_out` with the pre-mixing caveat, and the numbers
  above are its first measurement on the film.
