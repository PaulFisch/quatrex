# Pole-subtracted SCBA: current state

Snapshot after the external review (`pole_scba_implementation_review.md`) and
the fixes it prompted. Companion to `pole_scba_implemented.md`, which carries
the formulas and the full measurement record.

**Status: the frequency sector is mathematically consistent and not yet
physically validated.** Every device number quoted below predates at least one
of the fixes listed above it.

---

## 1. What was wrong, and what it cost

Five defects, found in this order. Three were introduced by earlier work in
this campaign, one by the review, one by the cell-average audit.

| # | defect | size | how found |
|---|---|---|---|
| 1 | `btd_matvec` output allocated `zeros_like(x)` | crash | first production run |
| 2 | `M(z)` assembled at all 201 frequencies at once | crash | the guard added for #1 |
| 3 | mixed background leg unmasked where the ring masks | `Sigma^>` non-PSD by 0.15 | **bisection** |
| 4 | hysteresis keyed on `id(sol)` | membership churn, limit cycle | reading the pole count per iteration |
| 5 | lesser/greater mirror used the same-component conjugate | **244 %** | external review |
| 6 | per-pole sources gave `G_PP` a spurious `1/w` tail | **18364x** at `w = 1e5` | external review |
| 7 | `M(z)` only piecewise holomorphic | **17 %** jump per stencil | external review |
| 8 | analytic sectors emitted point samples, not cell averages | **653 %** at `2 gamma/h = 0.04` | cell-average audit |

Defect 8 was flagged by nobody and is the largest at small `gamma/h`, which is
precisely the regime the sector exists for. **No device run so far includes
its fix.**

---

## 2. What is now pinned

| property | value |
|---|---|
| continuation vs production kernel | 8.1e-14 |
| sector sum `B(G,G) = SS+SR+RS+RR` | 8.7e-16 |
| production mixed kernel vs explicit ring | 3.6e-03 |
| `G_PP` subtracted == `G_PP` reinserted | 0.0 |
| residue normalisation `l^H M' r` | 1.000000 |
| left vector vs SVD null space | overlap `1 - 1e-10` |
| `G_PP` tail vs congruence | ratio 1.0000 to `w = 1e6` |
| source at the complex pole | 3.1e-14 |
| cell average vs exact integral | 0.0 at every `gamma/h` |
| bosonic relation `G^<(-w) = G^>(w)` | 8.9e-16 |
| distributed continuation, 4 MPI ranks | 1e-12 |
| Gate 0 (empty window, GPU) | bit-identical, `rtol = atol = 0` |

350 phonon tests, 423 quatrex tests. Five pre-existing failures elsewhere
(CLI x2, reference qtbm-low-rank x3), unrelated and verified pre-existing.

---

## 3. What is still open

**Physical validation.** Nothing has converged on a device. The last
measurement (`pp1`, which predates defects 8's fix and the tracker) gave

    Sigma^> worst  -1.208e-02   at the DC-mask boundary
    heat profile   31.8 28.5 39.7 54.0 73.2   -- all positive, still tilted
    residual       ~3.7e-03, drifting up slowly

The trajectory across fixes is real: `Sigma^>` worst went
`-1.506e-01 -> -4.71e-02 -> -1.208e-02`, and negative contact heat
disappeared. But the profile is not flat and the iteration does not settle.

**Phi-derivability (review 15).** Cell averaging removed the interface error,
but three quadratures -- analytic residues, cell resolvents, FFT grid -- still
meet in one discrete functional. That the continuum expression is the same
diagram does not make the discrete realisation conserving. Needs deriving.

**Observables (review 16).** `J_L`, `J_R` and the other frequency integrals
still sample narrow poles on the grid. The observable can therefore carry the
registration error the self-energy no longer does. Same cell-integral fix
applies, on the `G` path rather than the `Sigma` path.

**Contacts (review 3).** `_obc_at` samples at the anchor, which removes the
jump but keeps the contact flat in `z`. A genuine `g_s^R(z)` is the
outgoing-sheet work.

**DC cell (review 18).** The mask is a grid convention, not physics. Two
obstacles: `cell_resolvent_weights` models `R` as cell-wise constant so it
would reproduce the artefact, and the ring must drop its mask in the same
change.

**Coupled q, distributed blocks, aux grid.** Refused by config, correctly.

---

## 4. Two rules this campaign paid for

**A bed used to test a symmetry must first be checked against the physical
identity, and the check must show the WRONG relation failing by a large
margin.** Two defects were "validated" against beds constructed to satisfy the
assumption under test -- the conjugate mirror, and earlier the Hilbert fold.
Both tests passed and both were meaningless.

**Bisect before proposing a mechanism.** Four mechanisms were proposed and
refuted by measurement on defect 3, each costing a cluster round trip. Scaling
the injected term gave a strictly linear response in one job, which
constrained the answer more than all four guesses combined.

A third, smaller: **judge a hybrid against a baseline only at matched
residual**, never at a fixed iteration count. A larger self-energy legitimately
takes longer to settle, and comparing at iteration 4 read that as a defect.
