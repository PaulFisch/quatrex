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

## 5. A defect to fix while in here

    PhononSolver: Pole sector : 0.0000s

The most expensive block in the run is invisible to the profiler. On the pole
arm `PhononSolver` reports 81.9 s with `OBC` 8.96 s, `Assemble` 0.004 s and
`Selected Solve` 0.011 s under it -- about 73 s unaccounted, which is the pole
solve, and the label that should hold it reads zero.

**The cause is not yet identified.** An earlier draft of this section blamed
the decorator for not reaching `_update_pole_sector_q`; that is wrong.
`@profiler.profile(label="PhononSolver: Pole sector")` sits on
`_update_pole_sector`, which wraps the whole body including the `nq > 1`
dispatch into `_update_pole_sector_q`, and neither early return above it can
fire on this arm (`_pole_enabled` is true, `delta` is nonzero after iteration
0, `extraction_only` is off). So the range does cover the work and still
reports 0.0000 s. Reproduce it locally with a q-resolved bed at
`QTX_PROFILE_LEVEL=default` before theorising further.

Fix it before the optimisation, not after -- the profile is how the work above
is judged, and right now it would send someone to optimise the bubble, which
is already fine.
