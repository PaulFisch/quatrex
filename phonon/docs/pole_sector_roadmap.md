# Pole-subtracted SCBA: what is built, what blocks a structure run, and in what order

2026-08-10. Implementation of `pole_subtracted_modal_scba.md`. Code:
`src/quatrex/phonon/pole_{kernel,nevp,keldysh,bubble,tracker,probe,sector,bridge,mixed}.py`
and `btd_linalg.py`; wiring in `phonon/solver.py`, `sse_phonon_phonon.py`,
`core/interaction.py`, `core/config.py`. 157 tests under
`tests/quatrex/phonon/test_pole_*.py` and `tests/quatrex/config/`.

## The claim, and the two numbers that justify it

A uniform frequency grid must resolve the sharpest linewidth in the problem, and
one that does not is not merely imprecise:

- the discrete convolution of two sub-grid modes **overshoots by 6.4x at
  dw/gamma = 20** and by 3590 % at dw/gamma = 0.008 -- it *gains* weight rather
  than losing it, depending on where the bins fall;
- sliding the grid under a fixed resonance swings what the grid reports by
  **7.6x**, while the pole solve returns the same pole to 0.0037 gamma.

The sector removes those modes from the grid instead: poles of `G^R` are found by
a nonlinear eigenvalue solve, subtracted from the bubble legs, convolved in
closed form, and their retarded partner reconstructed analytically. The split
`G = G_S + G_R` is exact -- it changes the representation, not the diagram.

## Status

| Piece | State |
|---|---|
| Continuation of `Sigma^R` to complex z | exact; **8.1e-14** vs the production Hilbert kernel |
| Nonlinear eigenvalue solve (Beyn + bordered Newton) | Newton 3 its to `\|M(z)\|` = 1e-14 |
| Quasiparticle seeding | 9/9 poles converge (1/9 with a fixed linewidth guess) |
| Tracker (predictor, clusters, principal angles, rescan) | live in `refresh()` |
| Keldysh cluster matrix | scalar occupations 29 % wrong at eta = 0.003, 0.46 % at 23 |
| Analytic pole-pole bubble + closed-form `Sigma^R` | exact; retarded half pinned at 1e-13 |
| Mixed sectors SR + RS | live; validated against a brute-force ring |
| Config surface, 8 refusal gates, bit-identity | live |
| `sectors` = `rr` / `rr_ss` / `rr_ss_sr` | all three implemented |

## What can run on a structure TODAY

**cnt33 and d5a only**, Gamma-only (nq = 1), **single rank**, aux grid off,
`retarded_method="fft"`, no IR floor, `sectors="rr_ss"`.

## Three hard blockers, measured not estimated

| Blocker | Effect | Fix |
|---|---|---|
| `nq > 1` refused | **MoS2 (nq = 25) and Si (nq = 81) blocked entirely** | the pole set is per-q; the vertex fold must follow it |
| mixed contraction is `O(nnz^2)`, guarded at 4096 | cnt33 L2 has ~70k nnz, so **`rr_ss_sr` refuses on every real device** | reuse the ring's block-structured GEMM at fixed frequency |
| `comm.stack > 1` refused | single rank only; production distributes frequencies | the nnz-state contraction (`pole_probe` is built for it, not wired) |

So of the four named beds, two are blocked by coupled-q, and the complete method
runs on none of them yet.

## Correctness gates that must exist before any number is quoted

None of these are written yet. They are cheap and they are what make a transport
result believable rather than plausible.

1. **Sector sum**: `B(G,G) = SS + SR + RS + RR` with consistent quadrature. The
   design note calls this the single most important test.
2. **Production PSD gate** (`psd_check`, declared and unconsumed).
   `bubble_positivity.md` is explicit that a negative occupation is a
   *structural* signature, and this sector is the first thing that can break
   positivity structurally.
3. **eps_KI**: `||Sigma^R - Sigma^A - (Sigma^> - Sigma^<)||` must sit at
   roundoff. Purely algebraic, and it guards the Sigma^R double-counting trap
   below directly.
4. **Energy balance**, `QX_BBCHECK=1`, PRE-mixing. The saved npz arrays are
   post-mixing and are an SCBA residual, not the conservation identity
   (`mos2_conservation_audit.md`).

## Ordered path

1. Pin the Phase-0 baselines (`p0cnt2`, `p0cnt4`, `p0d5a` with
   `QX_SSE_LOWMASK=1.5` -- **not** the removed `sse_low_freq_cutoff_thz`) with
   the current code and COMMIT them. Nothing downstream is interpretable
   without them.
2. Small end-to-end run: cnt33 L2, single rank, `sectors="rr_ss"`, preceded by
   the `max_poles = 0` bit-identity gate. If a zero-pole run is not byte
   identical to the baseline, nothing after it is trustworthy.
3. Write the four gates above.
4. Block-structured mixed contraction -> unlocks `rr_ss_sr` at device scale.
5. nnz-state contraction -> unlocks multi-rank, i.e. production grids.
6. Coupled-q -> unlocks MoS2 and Si.
7. The headline A/B: grid ladder with sub-cell offsets, against the traps
   already recorded -- registration lottery (report the offset SPREAD, never a
   single draw), grid-dependent `scba_tol` (guard on iteration count),
   `lead_current` is an UNWEIGHTED sum (multiply by dw), and the
   extent/resolution confound (hold fmax exactly fixed).

Steps 1-3 are days. 4-6 are the real remaining engineering.

## Four silent-wrong-answer traps found during implementation

Recorded because each produced plausible numbers and a green test.

1. **`Sigma^R` double-counting.** `core/scba.py` already adds
   `0.5*(sigma^< - sigma^>)` globally for the total, so an injected `Sigma^R`
   must carry the Kramers-Kronig half ALONE. For a pole sum that half is the
   both-retarded pole pairings -- exact, and pinned at 1e-13.
2. **`Sigma_RS` is not `Sigma_SR` doubled or transposed.** It attaches the modal
   leg to the other index pair and needs its own single-leg projections. `leg`
   and `conjugate` are independent parameters, with a test asserting the two
   sectors differ.
3. **A rational-fit route that always fell back**, reporting the cell-kernel
   answer under its own label. Its causality guard rejected every fit (a real
   background has conjugate poles, and an upper-half-plane pole is harmless) and
   its residue sum was a stub. It now raises.
4. **Three wrong references in a row** for the retarded partner: the production
   Hilbert kernel (assumes a bosonic fold the test's Delta violated by 4x), a
   truncated full-axis PV (`Delta ~ 1/omega`, so the `O(1/W)` truncation is the
   same size as the effect), and only then an exact pole-sum comparison. A
   numerical Kramers-Kronig reference cannot settle this claim.

## Method choice for the mixed sectors, measured

Exact residue reference, grid spacing h:

| gamma/h | grid | cells | moments |
|---|---|---|---|
| 20 | 5.8e-4 | 1.9e-3 | 1.8e-3 |
| 1 | 3.6e-3 | 1.3e-3 | 1.2e-3 |
| 0.2 | **0.74** | 2.7e-3 | 1.2e-3 |
| 0.04 | **6.5** | 4.6e-3 | 1.2e-3 |
| 0.008 | **36** | 5.1e-3 | 1.2e-3 |

The design note's recommended `grid` route is **not viable** -- it collapses
exactly where the sector is needed. `cells` is production (never catastrophic,
one matmul); `moments` is the refinement, flat in gamma.

## Placement rules (they differ by sector, and it is physics)

- `Sigma_SS` is added AFTER the auxiliary-grid restriction and carries its own
  closed-form retarded partner: it must never touch a grid.
- The mixed terms are added BEFORE `delta` is formed, so the existing Hilbert
  transform covers them. They carry one narrow factor against a smooth
  background, so the numerical transform is the right tool and a second analytic
  reconstruction would double-count.
