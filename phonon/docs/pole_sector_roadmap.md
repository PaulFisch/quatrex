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

## First production runs (daint debug, 2026-08-10)

Bed `pgate`: the Gamma-only 4-cell CNT from `cluster/l4gpu` (12 atoms/cell,
144 DOF, `retarded_method` forced to `fft`), `QX_NE=201`, `QX_MAXIT=4`,
`QX_BBCHECK=1`, one GH200. Two debug jobs, 1.0 nh total.

**Gate 0 passes.** Baseline (`QX_POLE=0`) against a pole-enabled run whose
window sits ABOVE the whole spectrum (`QX_POLE_WMIN=200`, `WMAX=400`, so the
screening runs and finds nothing) is bit-identical on the GPU path:

    energies, final_heat, last_heat, lead_current, iter_heat, iter_sigma_max,
    iter_bubble_balance, final_bubble_balance, slab_absorption, gr_diag_imag,
    gl_diag_imag, bubble_balance_spectrum, current_spectrum
    -- all max_rel = 0.000e+00 at rtol = atol = 0.  PARITY: PASS

Note the zero-pole gate cannot be spelled `max_poles = 0`: the field is a
`PositiveInt`. An empty window is the right way to express it anyway, because
it exercises screening and windowing rather than skipping them.

**Two real bugs, both found only by running.** Neither is a GPU bug; both
reproduce on numpy and both had passed 157 unit tests.

1. `btd_matvec` allocated its output with `zeros_like(x)`. The production
   assembly hands it blocks carrying a probe axis the vector does not have, so
   the in-place `+=` raised instead of broadcasting. Every unit test fed
   unstacked blocks. Fixed by allocating at the broadcast shape; `_matvec` now
   refuses a non-singleton stack rather than reshaping it into the row index.
2. That refusal immediately caught the second one: `set_operator_context` was
   handed `obc_blocks.retarded[0]`, which is `(n_freq, b, b)` -- the contact on
   the WHOLE grid. `M(z)` is a single matrix, so this assembled M at 201
   frequencies at once. The contacts are now sampled at the grid point nearest
   `Re z`, which is exactly the flat-contact approximation the docstring
   already claimed.

The lesson is worth keeping: the unit tests all built their own operators, so
none of them ever saw a production-shaped one. A shape that only the real
assembly produces is invisible to a suite of hand-built beds.

## Step 4 landed: the blocked mixed contraction

`_mixed_one_sector` contracts at the pattern level, `O(nnz_out * nnz_in)`,
guarded at 4096 entries -- which no real device is under (the 144-DOF smoke bed
already has ~13k). The contraction is a matrix triple product

    Sigma[I, J] = sum_{a,b} BL[I, a] M[a, b] BR[J, b]^T

in which every factor is block-banded: `M` lives on G's block-tridiagonal
pattern, and `BL`/`BR` inherit the cubic vertex's `|I - a| <= 1`. So each output
block costs a handful of `b x b` GEMMs, and the dense `(n_dof, n_dof)` vertex --
1.5 GB per pole at production size -- is never formed either.
`mixed_self_energy_blocked` is pinned to 1e-12 against the pattern-level form on
two block layouts, and the interaction bridge now calls it. The pattern-level
routine stays as the small-size reference.

## The sector engages (daint debug, job 4392169)

First end-to-end run of the pole-subtracted SCBA on a real device. Same
`pgate` bed, `sectors="rr"`, `QX_POLE_NP=8`, window 1-55 THz:

    pole sector: iteration 1, 3 cluster(s), 6 pole(s)
    pole sector: iteration 2, 3 cluster(s), 5 pole(s)
    pole sector: iteration 3, 2 cluster(s), 2 pole(s)
    lead_current = 64.27197  (baseline 64.03312, +0.37%)

Poles are found, clustered, tracked across iterations and subtracted from
the bubble legs, and the run completes its 4 iterations. The pole count
falling 6 -> 5 -> 2 is the screening working as the self-energy builds and
modes broaden past `q_out`; it is the expected direction, but the ladder in
Experiment I is what will show the split is algebraic rather than merely
plausible.

Neither leg converged in 4 iterations -- `QX_MAXIT=4` is a smoke setting, not
a convergence test. No physics conclusion should be drawn from the 0.37%
until the A/B ladder runs at matched iteration counts and a resolved
reference.

### Gate status

| Gate | State |
|---|---|
| Zero-pole bit identity | **PASS**, GPU path, rtol = atol = 0, 13 observables |
| Sector sum `SS+SR+RS+RR` | implemented + tested (`test_pole_audit.py`) |
| `eps_KI` Keldysh identity | implemented + tested; catches double-counting and non-congruent `Sigma^{<,>}` |
| Positivity | implemented, wired behind `psd_check`, tested |
| Energy balance (`QX_BBCHECK=1`) | runs; `bubble_balance_last ~ 2.5e-8` |

## What is left, and why it stopped here

- **Step 5 (nnz-state contraction)** removes the `comm.stack.size > 1`
  refusal. It is distributed code, and a single-rank debug job exercises
  none of it -- landing it without a multi-rank test would repeat exactly
  the failure this campaign already hit twice (a shape only the real
  assembly produces).
- **Step 6 (coupled-q)** is what unblocks MoS2 and Si. Both beds exist on
  daint; neither can run the sector until the pole set is made per-q and the
  vertex fold follows it.
- **Step 7 (A/B ladder)** needs 5 and 6 for the beds that matter, though a
  CNT-only ladder on `l16/l24/l32` is possible today and would already
  measure `R` on a Gamma-only device.

## Sector ladder, measured -- and SR/RS fails the energy-balance gate

All three sector settings ran to 4 iterations with zero exceptions
(`QX_MAXIT=4`, a smoke setting; nothing converged, so the currents are not
physics yet). What matters is the conservation column:

| leg | sectors | lead_current | bubble balance | heat profile |
|---|---|---|---|---|
| `base` | off | 64.03312144 | 3.2e-08 | 64.27 61.06 60.63 61.40 64.27 |
| `np0` | empty window | 64.03312144 | 3.2e-08 | identical to base |
| `rr` | `rr` | 64.27197051 | 6.8e-07 | 64.27 61.06 60.63 61.40 64.27 |
| `rrss` | `rr_ss` | 64.24186400 | 5.4e-08 | 64.14 60.99 60.60 61.44 64.34 |
| `full` | `rr_ss_sr` | 71.98161136 | **2.6e-02** | **-7.44 70.05 60.14 51.52 136.52** |

`rr` -> `rrss` behaves exactly as the decomposition predicts: `rr` alone
DROPS the SS, SR and RS processes, so its balance degrades to 6.8e-07;
adding the analytic `Sigma_SS` back recovers 5.4e-08, slightly better than
the baseline's own residual. That is evidence the pole leg subtraction and
the closed-form pole-pole channel are consistent with each other.

`full` is broken. The balance is five orders of magnitude worse than every
other leg, and the heat profile is not merely noisy -- it is NEGATIVE at the
left contact and non-monotonic across the device, which a conserving steady
state cannot be. This is the gate-4 signal firing, and it is a real defect in
the mixed sectors, not a tolerance question.

Worth being precise about what is new here: `sectors="rr_ss_sr"` had NEVER
run at device scale before this job. The pattern-level contraction refused
above 4096 entries, and every bed is past that, so the only prior evidence
for SR/RS was unit tests on hand-built beds. The blocked contraction agrees
with the pattern-level one to 1e-12, so both would inherit a shared error --
agreement between them is not evidence of correctness.

Candidates, in the order I would check them:

1. **The single-frequency source.** `core/interaction.py` evaluates the
   projected source at ONE index, `mid = argmin|freqs - Re(cl.z[0])|`, and
   uses it for the whole convolution. For a cluster carrying several poles at
   different frequencies that is crude, and it is the least defensible line in
   the mixed path.
2. **Placement.** The mixed terms are added BEFORE `delta` is formed so the
   existing Hilbert transform covers them. If that transform is also seeing
   the `Sigma_SS` contribution, or if the analytic retarded partner is added
   on top, the retarded half is double counted -- which `eps_KI` would catch
   directly and cheaply.
3. **Leg convention.** `Sigma_RS` is not `Sigma_SR` transposed, and the two
   are built from independent `leg`/`conjugate` pairs. A swapped pair would
   survive every existing test except conservation.

The next step is NOT a bigger run. It is to evaluate `eps_KI` and the sector
sum on this bed (both are implemented and cost nothing), because they
localise the error to a specific channel; a grid ladder on a non-conserving
self-energy would only measure the defect.
