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

## The SR/RS conservation break: diagnosis and fix (2026-08-10)

The break was a **missing half of the frequency axis**, not a sign, prefactor
or leg-convention error.

The mixed convolution runs over the whole axis, but the solver only ever holds
`G` for `omega >= 0`. The negative half is fixed by `R(-w) = R(w)^*` -- it is
not missing information -- but it has to be SUPPLIED, because
`cell_resolvent_weights` integrates exactly the cells it is given and its
docstring says so outright ("No bosonic mirror is applied").

Measured against a wide-axis reference on a background built to satisfy the
relation exactly: the one-sided integral is **wrong by 28%**, mirroring brings
it to **2.5e-4**. On the device that moved the bubble balance from **2.6e-02 to
2.6e-05**.

`bosonic_extend` in `pole_mixed.py` does it, and refuses a grid not anchored at
zero rather than guessing where the mirror axis is.

### Three wrong turns on the way, all worth keeping

1. **A first anti-Hermiticity probe reported the mixed sector as exactly
   Hermitian** (residual 2.0, i.e. `M^H = M`) -- an apparent factor of `i`.
   It was the TEST that was wrong: it fed a PSD Hermitian source, whereas a
   physical Keldysh source `S = V^dag Sigma^< V` is anti-Hermitian
   (`Sigma^< = i * PSD`). With the right source both SS and mixed sit at 3e-16.
   Nearly "fixed" correct code.
2. **A code audit claimed the cell kernel already returns a KK-complete
   retarded object**, so the numerical Hilbert transform was double counting.
   Wrong: `leg_partial_fractions` returns poles in BOTH half planes and the
   kernel matches `pair_convolution` to quadrature error. Checking the claim
   against the exact reference took one command and saved a wrong fix.
3. **The brute-force ring reference in `test_pole_mixed_sectors` was ALSO
   one-sided**, so the old agreement was two identical mistakes matching. It
   now integrates the full axis while the kernel gets only the non-negative
   half, so it tests the reconstruction too. Its background had to be made
   genuinely bosonic: a single Lorentzian is not, and neither is a
   conjugate-symmetric pair times a COMPLEX matrix.

### The bosonic closure is deferred, and that is measured

`bubble_clusters()` closes the pole set under `z -> -z^*` and is documented as
mandatory. Wiring it in made `rr_ss` **worse by 3400x** (bubble balance
5.4e-08 -> 1.8e-04). The reason: closure puts partners at NEGATIVE real
frequency, while the frozen source is evaluated at ONE index,
`mid = argmin|freqs - Re(z[0])|`, the positive centre. One frozen source
cannot serve both branches -- the partner's is the bosonic mirror of the
original's. Closing therefore needs a per-branch mirrored source, which is not
implemented.

Unclosed is self-consistent: the sector is DEFINED by the poles inside the
(positive) window, `g_reg` is mirrored from the positive axis, and the mixed
pole leg is exact for the poles it has. `state.legs` is kept distinct from
`state.clusters` so enabling the closure later is a one-line change plus the
source work.

## Step 5 landed differently from the design note

The note prescribed contracting in the `"nnz"` state and `dtranspose`-ing to
`"stack"`. That is unnecessary. Both terms of the continuation are LINEAR in
`Delta` with grid-only coefficients, and neither reindexes frequency (the
mirror is elementwise; the negative-frequency reflection lives in the
`z + omega_k` argument). So every rank builds the weights for the global grid,
contracts its own columns, and one `all_reduce` finishes it.

That moves `(P, nnz)` instead of the whole buffer, and avoids distributed root
finding entirely -- every rank ends with the same operator and solves the same
poles. `contract_delta`'s docstring already anticipated it.

`local_fit_weights` expresses the least-squares second-sheet continuation as a
weight matrix; pinned to `delta_local_fit` at 5e-15 for derivative orders 0-2.
Serial runs get the identity reducer, so the path is bit-identical rather than
merely equivalent. Verified on 4 real MPI ranks at 1e-12.

**Test-infrastructure trap**: the conda env links MPICH while `/usr/bin/mpirun`
is the system OpenMPI. Mixing them does not fail -- it starts N independent
one-rank jobs, so every MPI test skips with "only 1 MPI processes specified"
and the suite still reports success. Use the launcher from the same prefix as
mpi4py.

## The mixed sectors are still broken, and the cause is now identified

All six legs, same pre-mixing conservation identity (`QX_BBCHECK=1`, printed to
the log; the saved `iter_bubble_balance` agrees with it here):

| leg | config | resid | heat non-uniformity | min heat |
|---|---|---|---|---|
| `base` | pole off | 3.22e-08 | 0.060 | 60.32 |
| `rrss` | pre-fix | 5.37e-08 | 0.060 | 60.60 |
| `full` | pre-fix | 2.574e-02 | 2.316 | -7.44 |
| `rrssf` | mirror + closure | 1.847e-04 | 0.071 | 60.76 |
| `fullf` | mirror + closure | 2.582e-05 | 3.104 | -22.02 |
| `fullg` | mirror, closure deferred | 1.488e-01 | 2.789 | -9.10 |

**The bosonic mirror did NOT fix `rr_ss_sr`.** It improved the balance residual
by three orders, but the heat profile got worse, and no variant is physical:
every `rr_ss_sr` run has negative heat at the left contact. Reading the
improving metric while the physical observable moved the other way was an error
of analysis, recorded here so it is not repeated -- `P_in = P_out` is a scalar
trace identity and a spatially wrong Sigma can satisfy it.

Deferring the closure was also wrong: it flatters `rr_ss` (a staging setting)
and costs the complete method four orders. The closure is restored.

### Ruled out: the leg convention

On a leg-exchange-symmetric vertex -- `Phi[i,a,b] == Phi[i,b,a]`, which is what
hiphive's `symmetrize=True` produces and therefore what every real device uses
-- `Sigma_SR` and `Sigma_RS` are **exactly equal** (measured 0.0). They differ
only on the random, unsymmetrised Phi the unit-test beds use. So the recorded
`out + out` "error" was never wrong for real physics, and two tests asserting
SR != RS are measuring an artefact of their own bed. Both are now annotated,
and the symmetry is pinned in `test_pole_bosonic.py`.

### The actual cause: the two pole legs are different objects

`pole_keldysh_sparse` takes a **frequency-resolved** source, `(n_omega, Np,
Np)`, so `G_PP` -- the leg SUBTRACTED from the bubble -- is built with the full
frequency dependence. The mixed sector instead freezes the source at one index,
`s_l[mid]`, `mid = argmin|freqs - Re(z[0])|`.

So the pole leg removed from the ring and the pole leg put back by `Sigma_SR` /
`Sigma_RS` are not the same function. That is precisely the condition the
decomposition needs (doc Sec. 37: the SAME reconstructed G on both legs), and
violating it breaks the balance spatially rather than in the trace -- which is
exactly the signature observed. It also explains why the closure makes it
worse: more poles means more frozen-source error, and why `rr_ss` is
comparatively immune: `Sigma_SS` is localised at the poles, where the frozen
source is evaluated and is therefore accurate.

### The fix is the route the design note originally prescribed

> `Sigma_SR + Sigma_RS = B(G_S,G_R) + B(G_R,G_S)` is exactly the bilinear form
> `compute_linearized` already implements. Call it with `dG = G_PP`: zero new
> bubble code, exact, symmetric by construction, cost independent of `N_p`.

Both `G_PP` and `G_reg = G - G_PP` are already known on the grid, on the stored
pattern, with the resolved source. The mixed term is a straight convolution of
two known grid functions and needs no partial fractions and no source model at
all. The analytic pole-convolution route built instead buys cell-exact
treatment of the narrow pole, but only pays off once the polynomial source
model (`source_model` / `source_order` / `source_fit_tol`, all still
unconsumed) exists to make its leg match `G_PP`.

Next step: route `sectors="rr_ss_sr"` through `compute_linearized(dg=G_PP)`,
and keep the analytic route behind `source_model` for when it is complete.

## Correction: much of the "conservation break" was an iteration transient

The `rr_ss_sr` verdict above was reached from 4-iteration runs at mixing 0.1.
That was a bad comparison, and the error is worth recording because it cost
four hypotheses and five cluster jobs.

Re-run at 25 iterations, mixing 0.02, same bed and same code:

| | 4 it, mix 0.1 | 25 it, mix 0.02 |
|---|---|---|
| balance residual | 1.25e-01 | **5.08e-05** |
| heat non-uniformity | 2.79 | **0.42** |
| min heat | -9.1 (negative) | **50.8** (all positive) |
| lead_current | 64.69 | **63.66** (base 64.03) |

The mixed sectors add roughly 15% to Sigma (measured from the sector-sum
weights: SR + RS = 0.077 + 0.077 against SS 0.96 and RR 0.11), so the hybrid
has a LONGER transient than the baseline. Judging both at iteration 4 compared
a converged-ish baseline against a diverging hybrid transient and read the
difference as a defect in the self-energy.

None of the four fixes attempted before this was validated against a converged
run. Two of them are independently correct and stay (the bosonic mirror, 28%
error without it; per-pole sources, frozen was 27% off on a cluster's second
pole). The others were reverted or re-derived.

### What the mathematics actually says, all now pinned as tests

| check | result |
|---|---|
| sector sum `B(G,G) = SS+SR+RS+RR` | exact, **8.7e-16**, all sectors weighted |
| production kernel vs explicit ring | **3.6e-3** at `gamma/h = 40`, magnitudes within 0.2% |
| `G_PP` subtracted == `G_PP` put back | **0.0** (same object, after the partial-fraction fix) |
| `Sigma_SR` vs `Sigma_RS` on a leg-symmetric vertex | **equal**, 1.3e-15 |
| Gate 0 (empty window) | bit-identical, `rtol = atol = 0` |

So the decomposition algebra and the kernel are both correct. Whatever remains
after convergence is a question about the fixed point, not about the bubble.

### Process lesson

Judge a hybrid against a baseline only at MATCHED convergence, never at a fixed
iteration count -- a larger self-energy legitimately takes longer to settle.
And run the localising gate before proposing a mechanism: the sector sum took
one local command and would have ruled out three of the four hypotheses
immediately.

### Correction to the correction: rr_ss_sr IS broken

The "it was only a transient" reading above does not survive a longer run. At
80 iterations, mixing 0.02, on the same bed:

| leg | iters | resid | heat profile | non-uniformity |
|---|---|---|---|---|
| `cbase` | 80 | 7.11e-08 | 53.80 51.84 51.90 52.07 53.83 | 0.038 |
| `cfull` | 80 | 1.00e-04 | **-27.44** 49.00 39.06 33.62 **135.57** | **3.56** |

The 25-iteration snapshot (non-uniformity 0.42, all-positive) was itself a
transient passing through, not convergence. `rr_ss_sr` is genuinely broken:
the iteration wanders rather than settling, and the converged-ish profile is
negative at the left contact.

Note also that `cbase` reports NOT CONVERGED after 80 iterations and its
`lead_current` drifted 64.03 -> 53.81 from the 4-iteration snapshot, a 16%
move. Every number quoted from the 4-iteration runs, baseline included, was a
transient. Matched-iteration comparison on this bed is meaningless; matched
RESIDUAL is the only fair basis.

Once more the balance residual is small (1e-4) while the profile is
unphysical, which is the third time this pair has disagreed. `P_in = P_out` is
a scalar trace identity; it is necessary, not sufficient, and the heat profile
is the observable that actually discriminates. Any future gate on this sector
should read the profile first.

## Localised: the mixed sector integrated a leg the ring excludes

Found by BISECTION, after four mechanism guesses had each been proposed and
refuted by measurement (residue mis-scaling, mixed quadratures breaking PSD, a
non-congruent `G_PP`, an indefinite `G_reg`). Scaling the injected
`Sigma_SR + Sigma_RS` by `lambda`:

| lambda | `Sigma^>` worst | / lambda |
|---|---|---|
| 0.0 | -2.0e-04 (w[1], DC edge) | -- |
| 0.25 | -3.805e-02 (w[128]) | 0.152 |
| 0.5 | -7.678e-02 (w[128]) | 0.154 |
| 1.0 | -1.506e-01 (w[128]) | 0.151 |

Strictly linear, so no cancellation is involved: the term simply adds
something that should not be there. A delicate approximation failure would
have been nonlinear. That single measurement constrained the answer more than
all four mechanism guesses combined.

Cause: the ring zeroes the `omega = 0` bin of its legs -- its own comment says
that bin carries the near-singular acoustic peak, `|G^>(0)| >> neighbours` --
while `interaction.py` built `reg_l = g - g_pp` straight from the Green's
function with no mask. The two sectors were integrating different data.

### A prior bug in the gate itself

The positivity gate reported `sigma_greater worst = -1.000e+00` on the
POLE-FREE baseline. Exactly -1 is uniformly negative, i.e. a flipped sign.
This solver uses `sigma^{<,>} = +i n(+1) Gamma`, so BOTH components satisfy
`-i sigma >= 0`; the textbook `+i G^> >= 0` does not apply. Every `greater`
reading before that fix was meaningless. Now pinned with a test asserting a
wrong sign looks OBVIOUSLY wrong (worst < -0.99), so it cannot recur as a
marginal-looking number. Lesson: validate a diagnostic against a known-good
run before trusting it to localise a defect.

### Open: can the mask be dropped entirely? (Paul, 2026-08-11)

The mask is a grid convention, not physics -- it implements "omega = 0 has
measure zero in the integral", which a single grid SAMPLE violates. The code
states the device `G^<` has no `1/omega` pole, so the lesser leg is regular at
DC and its cell integral is unproblematic.

Two things stand in the way of simply removing it for the analytic route:

1. `cell_resolvent_weights` models `R` as cell-wise CONSTANT, so integrating
   the DC cell reproduces the same sampling artefact rather than curing it.
   Doing it properly means modelling the `omega -> 0` form of the leg --
   and it is `G^>`, not `G^<`, that is near-singular there
   (`G^> = G^< + (G^R - G^A)`, with the acoustic spectral weight in the
   difference).
2. Sector consistency is binding: the ring masks, and changing that breaks
   legacy bit-identity. Dropping the mask must happen on BOTH sides together,
   or the sectors disagree again -- which is exactly the 15% violation
   measured above.

So the analytic DC treatment is the right destination and a well-defined
follow-up: model the DC cell in the mixed route, drop the ring's mask behind
the same flag, and quantify the remaining O(h) disagreement.

### Mask fix verified: the linear error is gone, a quadratic one remains

Same bisection, with the mixed background leg masked like the ring's:

| lambda | `Sigma^>` before | after | factor |
|---|---|---|---|
| 0.25 | -3.81e-02 | -7.88e-04 | 48 |
| 0.5 | -7.68e-02 | -2.44e-03 | 31 |
| 1.0 | -1.506e-01 | **-9.24e-03** | 16 |

The magnitude matters less than the change in SIGNATURE: the scaling went from
linear to **quadratic** in `lambda` (predicted from the `lambda = 1` value:
5.8e-4 and 2.3e-3 against measured 7.9e-4 and 2.4e-3). A wrong sign or a
missing factor cannot produce `lambda^2`; a second-order effect can -- the
self-consistent feedback of the modified `Sigma` through `G`, or a genuine
`O(term^2)` quadrature term. So the first-order injected error is fixed and
what remains is a different, smaller problem.

Heat spread at `lambda = 1` narrowed from 30.6 to 19.0, so the observable
improved too but is not yet flat. The converged comparison is what decides
whether the residual 1e-2 violation matters.
