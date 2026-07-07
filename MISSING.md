# Missing plots, data, and calculations for the report

Working ledger for the report overhaul. Updated 2026-07-02 after the merge of the
report-update, the figure regeneration campaign (all document figures now have a
`phonon/studies/style.py` generator under `phonon/scripts/figures/` or
`phonon/studies/`), and the claim-by-claim verification pass (results + theory +
appendices). Resolved history is dropped; what follows is the TRUE residual state.

## A. In-flight / paused calculations on this node

1. **d11a L2 η=0 anharmonic — PAUSED 2026-07-02** (node ceded to the
   tensor-decomposed-SSE film campaign). The healthy shape is 8 ranks × 16 ring
   threads (~85 min/iteration; the 128-rank launch memory-thrashed). Resume via
   `nohup python phonon/studies/_gapruns.py` when the node frees up; a
   descending-basin verdict needs ~10–15 iterations.
2. **cnt80 L3 η=0 anharmonic (full run) — PAUSED with (1)** (queued in
   `_gapruns.py`; the 60-iteration probe descended 1.0 → 3.8e-3 with the d5a
   soft-mode kit, so convergence is expected on resume).
3. **Static-SE sweep** (`phonon/scripts/verify/_static_se_sweep.sh` →
   `phonon/scripts/out/snapshots/study_*.npz`) — regenerating the purged static
   loop/tadpole snapshots; cnt33 complete (24/24), d5a in progress. On
   completion: regenerate `static_se_study` (see B1) and verify the
   `50_static_scp.tex` claims (2.98→1.34%, 8:1:0.5, 670/1045 THz², 1298→284
   THz², 0.765 THz stabilised mode) against the fresh snapshots.

## B. Figures still to fix

1. **`static_se_study` + `static_se_tadpole_breakdown`** — blocked on A3. Review
   verdicts: the d5a panel contradicts the "loop exceeds the tadpole at every T"
   sentence (isolated- vs coupled-tadpole confusion) and the two figures carry
   inconsistent CNT loop curves. Plan: one merged figure from the fresh
   snapshots with coupled-vs-isolated tadpole clearly distinguished; drop the
   orphaned breakdown figure.
2. **`d5a_scp_convergence.png`** (orphan in the fig dir) — keep/delete after A3.

## C. Missing calculations (open, not currently scheduled)

- **cnt33 L≥4 η=0 anharmonic** — L4 reached best-iterate conservation 4.0e-6 at
  ratio 0.479 but did not close the residual gate; L5/L6 iteration-unstable
  bare (a fixed point exists at finite η). Revisit with the d5a floor stabiliser.
- **d5a η=0 at nf=361** — the floor-stabilised recipe stalls at O(1) residual on
  the doubled grid; whether the 0.016–0.032 marginal floor is structural or
  resolution-limited is OPEN (stated as such in `sec:res_eta0`).
- **Film/SiGe η→0** — the production coupled-q solver is singular below
  η≈0.4 THz (`sec:prod_open`); needs the block-recursion regularisation.
- **SrTiO₃ structure-count sweep** — the "55i–92i cm⁻¹ converging upward" range
  has no surviving sweep (only the 60-structure fit, 55.2i cm⁻¹); rerun the fit
  vs n_structures to restore the convergence claim.
- **SrTiO₃ ‖Σ_static‖≈39 THz²** — no surviving calculation (REPRODUCE M23).
- **d5a cutoff sweep on the sc4 refit** — the in-document 5× diagonal-G is the
  original fit (csv restored from git); the archived sc4 sweep gives ~77×
  (git d2854d90); a fresh rerun would put the current fit in the figure.
- **Si/Ge heterostructure phph at the native prefactor** — the +32%/+51%/±1%
  values predate the ÷4-prefactor retraction (F24); the harmonic +344–369%
  barrier number is prefactor-independent and stands.
- **cnt33 finite-η dense npz** — purged; the archived F30 tables in
  `phonon/docs/lab_notebook_archive.md` are the surviving record (now the data
  source of `cnt33_finite_eta_bias.py`).
- **cnt33 finite-η cutoff-cube npz** — purged; figure `cnt33_cutoff.pdf` kept
  from the original run, values verified against the F30 archive.

## D. Open text flags (`% REVIEW(open)` markers left in the tex)

- `80_approx.tex`: two η/ω unit-convention flags → reconcile the per-solver
  conventions once in `app:finite_analysis`.
- `conservation.tex`: sigma_convention positivity gate unquantified.
- `theory/30_scba_eta0.tex`: bare G^< sign-convention note.
- `production_coupled_q.tex`: two purged-run notes (the cnt coupled-q ladder
  0.896/0.911 — figure removed, values kept as historical record only).
- `input_gen.tex`: Mounet replacement source; Rurali2010 family-enumeration
  confirmation.
- `70_strong_srtio3.tex`: the two SrTiO₃ provenance flags of section C.

## E. Residual `\todo`s (content, not runs)

- `80_approx.tex`: consolidated approximation-cost overview figure (single axis
  across cnt33/d5a/d11a/film) — the individual sweeps all exist as figures.
- `theory` FC3-compression subsections: the error-vs-observable statement and
  two bare "discuss" placeholders.
