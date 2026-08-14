# Batching the pole solve: instructions

2026-08-14. The pole solve costs 187 s per SCBA iteration on Si against the
bubble's 7 s, and essentially none of it is arithmetic. This is the plan to fix
it. Companion to `pole_sector_observations.md`.

> **Do the convergence fix first.** The same Si run that produced these
> timings does not converge: the promoted pole set limit-cycles with period
> two (620 <-> 460 poles) while `rel Sigma` sits at O(1). See
> `pole_sector_observations.md` Sec. 8. A 20x faster pole solve is worth
> nothing until that is settled, and the fix changes how many poles are
> promoted per iteration, which is the quantity every number below scales
> with.

---

## 0. The measurement this rests on

Si `sichk_base`, 81 q, `ne = 141`, from `cluster/psi2/slurm-4464697_quatrex_times.out`:

    PhononSolver: Pole sector : 0.0000s      <- the label is broken, see Sec. 5
    PhononSolver              : 9.9665s        base arm
      ring contraction        : 7.2219s
    PhononSolver              : 186.8692s      pole arm
      ring contraction        : 7.2006s

The bubble is unchanged (7.22 -> 7.20 s). The whole 19x increase is the pole
solve.

Local profile of one `refresh()` (numpy, 9 candidates, `test_pole_sector._context_run`):

    0.116 s for 9 candidates      -> 12.9 ms per candidate
    43,488 Python calls for 9     -> ~4,830 calls per candidate

    solve_poles       0.114 s  (98 %)
      bordered_newton    9 calls, 0.113 s
      _sigma_blocks     45 calls, 0.059 s   <- 51 %
      continue_sigma    45 calls, 0.046 s
      btd_norm2         18 calls, 0.039 s
      btd_matvec       499 calls, 0.032 s
      contract_delta    45 calls, 0.027 s

45 calls for 9 candidates is ~5 Newton steps each.

**The arithmetic is negligible.** The operator is block-tridiagonal with small
blocks; one solve is a few kFLOP. Si runs 81 q x 144 candidates = **11,664 pole
solves per iteration**, at ~4,830 Python-level calls each:

    56e6 calls x ~3 us launch  ~  169 s        observed 187 s

So the time is launch and interpreter overhead, at an arithmetic intensity
roughly six orders below what the GPU needs to be busy. The bubble, by
contrast, is genuinely device-bound.

## 0.1 The kernels are ALREADY batched

This is why the fix is cheap. `pole_kernel.contract_delta` is

    out = w_pos @ flat + w_mir @ mirror,      w_pos shape (P, K)

a single GEMM over `P` poles. `continuation_weights` and `local_fit_weights`
build `(P, K)` for a vector of `z`. The primitives take the whole pole set; the
Python loop in `solve_poles` calls them with `P = 1`, 11,664 times.

Nothing needs a new kernel. The loop has to stop defeating the ones that exist.

---

## 1. Do this first: hoist a recomputed invariant

In `contract_delta`:

    mirror = bosonic_partner(a, transverse_shape).reshape(n_freq, -1)

`a` is `Delta`. It does **not depend on `z`**. So the full `(n_freq, nnz)`
bosonic mirror is rebuilt on every Newton step of every candidate of every q --
about 58,000 times per iteration -- producing the identical array each time.

Self-contained, independent of the batching, and verifiable on its own. Cache it
against the identity of `a` (or compute it once in `set_operator_context` and
pass it down). Measure before and after on the local profile above; the
`contract_delta` line should drop and `btd_matvec` should not move.

**Do not skip the measurement.** If hoisting does not move the number, the
mirror is cheaper than it looks and the assumption behind this step was wrong.

---

## 2. The batching, in order

Each step keeps the suite green. Do not proceed on a red suite -- the pole tests
are the only thing standing between this and a wrong pole set, which is far more
expensive than a slow one.

### 2.1 `bordered_newton` takes a vector of seeds

Signature becomes `z0: NDArray` of shape `(P,)`, returning `P` solutions.
Internally every quantity gains a leading `P` axis: `z`, `r`, `l`, the residual,
`dz`, `kappa`.

The ragged control flow is what forces the loop today: candidates converge after
different numbers of steps. **Do not preserve that.** Run all `P` to
`max_iter` and carry a boolean `active` mask; a converged candidate's update is
masked to zero. With batching the wasted steps are nearly free, and the
alternative -- compacting the batch as candidates drop out -- adds
synchronisation and index bookkeeping for no gain at these sizes.

Keep the per-candidate `iterations` count honest: record the step at which each
candidate first met `tol`, not `max_iter`.

### 2.2 `m_blocks` / `dm_blocks` / `_sigma_blocks` accept `(P,)`

`pole_sector.m_blocks(z)` and `_sigma_blocks(z)` already funnel into
`contract_delta`, which takes `(P, K)` weights. Pass the whole `z` vector and
return blocks with a leading `P` axis. The BTD block dict becomes
`{(i, j): array of shape (P, b_i, b_j)}`.

`continue_sigma` and `local_fit_weights` likewise: they already build `(P, K)`.

### 2.3 Batched BTD factorise and solve

`btd_linalg` needs its factorisation and solve to carry a leading batch axis.
The block operations are already `einsum`/`matmul` shaped; adding a batch axis
is mostly index work. `btd_matvec` (499 calls for 9 candidates) and `btd_norm2`
(the `norm_power` power iteration, and `left_iters` inverse iteration) must
batch too, or they become the new bottleneck -- together they are 61 ms of the
116 ms above.

### 2.4 `solve_poles` stops looping

    for k, z0 in enumerate(seeds):  ->  one call with the full seed array

`_fit_anchor` is currently pinned per seed inside the loop, so that M(z) is
holomorphic for the whole Newton solve of that candidate. **That is
load-bearing** -- deriving the stencil from `Re z` instead made `M` only
piecewise holomorphic, with a measured 17 % jump at each stencil boundary. The
batched version needs a per-candidate anchor vector, not one shared anchor.
This is the single most likely place to introduce a silent physics regression.

### 2.5 Then batch over q

`_update_pole_sector_q` loops 81 q, each with its own operator. Once 2.1-2.4
land, the q loop is the remaining serial axis. The q are fully independent, so
this is the same transformation one level up; do it only after the per-q batch
is verified, and treat `q_stride`/`q_max` as the interim mitigation.

---

## 3. Verification

Each item is a gate, not a suggestion.

* **Bit-identity on one pole.** `P = 1` through the batched path must reproduce
  the current scalar path exactly. If it does not, stop.
* **Same poles, any order.** For a real seed set, the batched solve must return
  the same `z`, `r`, `l`, `kappa` and `eps_z` as the loop, to roundoff. Compare
  as sets keyed by location, since ordering may change.
* **The anchor test.** A candidate whose seed sits near a stencil boundary must
  land where the scalar path puts it -- this is what 2.4 endangers.
* **Convergence bookkeeping.** `iterations` must be the step at which `tol` was
  met, not `max_iter`, and `converged` must agree with the scalar path
  candidate by candidate.
* **The full suite**, currently 502 phonon tests, stays green at every step.
* **The profile.** Re-run the local `refresh()` profile from Sec. 0. Python
  calls per candidate should fall by orders of magnitude, not percent. If they
  do not, the loop moved rather than disappeared.

Then re-measure on device: the same Si arm, and `PhononSolver` should approach
the bubble's ~7 s rather than 187 s.

---

## 4. What NOT to do

**Do not write a custom kernel first.** The existing primitives are already
GEMMs over the pole axis; a hand-written kernel would replace working, tested
code with new code that solves a problem the batching already removes. Revisit
only if the profile after Sec. 2 still shows a launch-bound step.

**Do not loosen `newton_tol` or `max_iter` to make it faster.** The cost is not
in the iterations; it is in how they are dispatched. Cutting them buys speed by
returning worse poles, and the acceptance work (`eps_z`, `locate_tol`) was done
precisely to stop that trade being made accidentally.

**Do not batch across q before batching within q.** The per-q batch is where
the 144x is, and it is testable against the scalar path. Batching q first mixes
two changes and leaves nothing to compare against.

---

## 5. A defect to fix while in here -- IDENTIFIED AND FIXED

    PhononSolver: Pole sector : 0.0000s

The most expensive block in the run was invisible to the profiler. On the pole
arm `PhononSolver` reports 81.9 s with `OBC` 8.96 s, `Assemble` 0.004 s and
`Selected Solve` 0.011 s under it -- about 73 s unaccounted, which is the pole
solve, and the label that should have held it read zero.

**The cause: the decorator was on the wrong method.** At `f6bd76f7`, the
commit the `psi2` log was produced from, `solver.py:630` reads

    @profiler.profile(label="PhononSolver: Pole sector", level="default", comm=comm)
    def _check_positivity(self, out: tuple) -> None:

and `_update_pole_sector` (l. 712) carries no decorator at all.
`_check_positivity` returns on its first statement unless
`pole_sector.psd_check` is set, and it is off by default. So the range was
timing an immediate return, which is why every one of the 41 readings is
*exactly* `0.0000s` and never nonzero on either arm -- a range that genuinely
covered the q loop would at least fluctuate.

Two earlier readings of this were wrong and are recorded so the next person
does not repeat them. The first blamed the decorator for not reaching
`_update_pole_sector_q`; it never reached `_update_pole_sector` either. The
second concluded the cause was unidentified because `@profiler.profile` does
sit on `_update_pole_sector` -- true of the tree only AFTER the fix below,
which had already been committed (`e4e8e05e`) when that paragraph was written.
Checking `f6bd76f7` rather than the working tree settles it.

The fix is to move the label onto `_update_pole_sector`, and to give
`_check_positivity` its own (`PhononSolver: Positivity`), so the gate is
measured where it is rather than borrowing the sector's name. Two sub-ranges,
`PhononSolver: Pole solve` and `PhononSolver: Pole legs`, split the sector into
the corrector and the leg construction, so the 187 s is attributed rather than
assumed.

The 41-against-33 count -- more `Pole sector` readings than `PhononSolver`
exits -- is NOT explained by this and remains open. It does not affect the
attribution: `profile_range` prints inline at each exit with no aggregation, so
a spurious extra reading of an immediate return adds zero either way.

None of this weakens Sec. 0: the 187 s is established by the arm-to-arm
difference (185/85 s against the base arm's 9.97 s) and by the local cProfile
of `refresh()`, which measured `solve_poles` directly at 98 % of 12.9 ms per
candidate. 11,664 candidates x 12.9 ms = 150 s, which is the observed
increment.

---

## 6. Result, measured on device (job 4468380, 2026-08-14)

Si `sichk_base`, 81 q, `ne = 141`, the same arm as `cluster/pgate/sipole2.sh`,
four SCBA iterations at `QTX_PROFILE_LEVEL=default`. Seconds per iteration,
iterations 1-3 (iteration 0 has no scattering yet and returns early):

| | `Pole sector` | of which `Pole solve` | rest (legs) | `PhononSolver` |
|---|---|---|---|---|
| q batched (81 per solve) | 34.3 / 21.2 / 35.7 | **0.77 / 0.77 / 0.41** | 33.6 / 20.4 / 35.3 | 43.4 / 30.2 / 44.7 |
| `q_batch = 1` (per-q solve) | 40.2 / 27.9 / 43.0 | **7.51 / 7.71 / 7.48** | 32.7 / 20.2 / 35.5 | 49.2 / 36.8 / 51.9 |
| sector off | 0.0 | -- | -- | 8.97 / 8.98 / 8.97 |

The bubble (`PhPh SSE: 3 ring contraction`) is 7.3-7.6 s throughout and is
unmoved, as it must be.

**The corrector is done.** Sec. 0 attributed about 177 s of the 187 s increment
to the pole solve. It is now **0.4-0.8 s**, against the bubble's 7.4 s: a
factor of roughly 250, of which about 24 comes from batching the candidates
within a q (Secs. 2.1-2.4, the `q_batch = 1` row) and a further 12 from
batching across q (Sec. 2.5). Wall clock per arm is not the number to read --
the first arm in a job also pays the one-time fc3 load and CUDA context setup,
which is why the faster arm has the longer wall time (261 s against 184 s) --
the profiled ranges are.

**The bottleneck moved rather than vanished.** `Pole sector` is still 21-36 s,
and now essentially all of it is `_build_pole_keldysh`: the leg construction,
which Sec. 0 never separated from the solve because the label was broken. It
scales with the promoted count at about **55 ms per pole** (624 poles -> 33.6 s,
437 -> 20.4 s, 580 -> 35.3 s) and is the same in both rows, as it must be,
being untouched. So `PhononSolver` is 30-45 s against the base arm's 9.0 s, and
the sector remains the dominant cost of the run.

That is the same defect one level out: `_build_pole_keldysh` loops over q in
Python and, inside each q, over clusters, calling `project_source_sparse`,
`add_contact_source`, `source_variation` and the congruence routines on one
small cluster at a time. The clusters are independent, the operations are the
same shape for all of them, and the arithmetic per call is tiny -- which is the
description Sec. 0 gives of the solve. It should be batched the same way, and
`Pole legs` (currently at `debug` level, one range per q) is the number to
judge it by.

**Open.** `q_batch` is not bit-invariant across a whole SCBA trajectory on GPU:
the promoted count reads 624 against 622 at the first iteration and 437 against
437 at the second. Batched and unbatched GEMMs differ in the last ulps and this
run is nowhere near converged (`rel Sigma` is O(25) at iteration 3), so these
four iterations cannot separate solver roundoff from ordinary trajectory
amplification. A converged A/B would settle it; nothing here depends on it,
since the batch size is a memory knob.
