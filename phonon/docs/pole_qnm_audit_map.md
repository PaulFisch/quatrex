# The 2026-08-14 pole/QNM audit backlog, mapped onto the tree

The backlog (`pole_qnm_scba_derivation_audit_implementation_backlog.md`, 2026-08-14)
re-derives the pole/QNM SCBA and hands over 52 prose sections, a P0-P6 priority ladder
(Secs. 40-46), a T0-T10 test checklist (Sec. 47) and ten "first concrete coding
actions" (Sec. 50).

It was written against a snapshot that predates the work of 2026-08-13 and -14, so a
large part of its P0 list is already in the tree. This document records which part, so
the next review does not re-report finished work, and isolates what is genuinely
missing.

Section numbers below in the form "Sec. N" refer to the backlog. Line references are
to the tree at commit `0c9370aa`.

---

## 1. Implemented, in production

| Backlog | Where |
|---|---|
| Sec. 12, exact Lorentzian resolution gate, Eq. (12) | `pole_sector.py:435` `leg_weight_error`, `coth(pi/r) - 1 = 2/expm1(2 pi/r)`. See Sec. 2 below: it is off by default |
| Sec. 14, complex pole separation, Eq. (15) | `pole_sector.py:379` `separations`, already `abs(z_i - z_j)` and not a real-axis distance |
| Sec. 16, selected-pole retarded congruence, Eq. (18) | `leg="congruence"` is the default. `pole_legs.py:242` `congruence_legs` returns point minus cell average; injected additively at `sse_phonon_phonon.py:1790`, before the masks and before `delta`, so the numerical Hilbert transform supplies the retarded partner |
| Sec. 18-21, covariance identity, centred basis, Gram matrix | `pole_covariance.py:98` `cell_resolvent_mean`, `:116` `centred_gram`, `:153` `cell_variance`, `:191` `covariance_kernel` |
| Sec. 22, staged screen by cell variance | wired at `core/interaction.py:552`, knob `covariance_sigma_min` |
| Sec. 23, rank-one FC3 contraction | `pole_covariance.py:247` `spectrum_correction` keeps residues rank one; `pair_covariance` at `:224` is the dense oracle |
| Sec. 7, residue, normalisation, conditioning | `pole_nevp.py:338-368`. The left vector is obtained by adjoint inverse iteration on `M^H`, with `_adjoint_blocks` (`:124`) conjugate-transposing and swapping the off-diagonals, so no `M = M^T` shortcut is taken. `d = l^H M'(z) r` at `:362`, then `l /= conj(d)` so `d = 1`. `kappa` at `:368` |
| Sec. 24, q-fold convention | `q_diff_map[iq_ext, iqp] = (iq_ext - iqp) mod nq`, `sse_phonon_phonon.py:2233-2245`; the left vertex is conjugated at `:2263` |
| Sec. 25, wire `membership_frozen()` | done 2026-08-13 (`6a8ea328`), consumed at `pole_sector.py:673-676` behind `freeze_membership` |
| Sec. 26, rescans add rather than replace | done (`a98c87af`, `2260cff9`); `_seed` at `pole_sector.py:1160`, `audit_every_iteration` defaults true |
| Sec. 27, complex continuation of the scattering self-energy | `pole_sector.py:863` `continue_sigma`, backed by `pole_kernel.py:119` `continuation_weights`. The derivative is closed-form, not finite-differenced: order 1 returns `1/hi - 1/lo` at `pole_kernel.py:175` |
| Sec. 34, Keldysh positivity | `pole_audit.py:125` `psd_residual`, wired at `solver.py:646` and `:1556` behind `psd_check` |
| Sec. 40, offline gate report | `phonon/studies/_pole_gate_report.py` |
| Sec. 38, census plumbing | `extraction_only` / `QX_POLE_EXTRACT`; `_census_rows` at `pole_sector.py:1054` |

Sec. 8 asks for standardised `gamma_hwhm` / `Gamma_fwhm` names against a silent
factor-of-two error. The names are absent but the convention is already uniform:
`gamma = -Im z`, half-width, at `pole_sector.py:374, 471, 635, 1060` and
`pole_keldysh.py:120`. The one expression that looks like a factor of two is the
golden-rule seed `gamma = -Im(v^H Delta v)/(4 Omega)` at `pole_sector.py:1006`, which
is `-Im Sigma^R/(2 Omega)` combined with the exact identity `Im Sigma^R = (1/2) Im
Delta`, documented at `:965-971`. Adding the names is worthwhile; there is no loose
factor to find.

---

## 2. Implemented, but not reaching production

These are the cases where a backlog item is satisfied by code that no production run
executes. Each is a smaller piece of work than the backlog assumes, and each is a trap
for anyone reading the config and concluding a feature is active.

**The exact resolution gate is off by default.** `leg_weight_tol` defaults to `0.0`
(`config.py:295`), and `screen` (`pole_sector.py:495-513`) treats the two gates as
alternatives, not as a stack: with the tolerance at zero it falls through to
`resolution_score = gamma/(samples_per_halfwidth * h)` against `q_in`/`q_out`. The
consequence was measured before the exact gate existed and is recorded in
`pole_sector_observations.md` Sec. 1.6: the ratio rule calls 140 of 144 CNT modes
under-resolved at `h/gamma = 0.65`, where the exact worst-case line-weight error is
1.3e-04.

**Multi-cell pole support exists but is not wired.** The covariance path assigns each
pole to one nearest cell, `core/interaction.py:533-535`:

    cells = np.unique([int(np.argmin(np.abs(freqs - float(np.real(z)))))
                       for z in np.asarray(get_host(cl.z))
                       if float(np.real(z)) >= 0.0])

That is radius zero, which is what Sec. 17 warns against. `pole_local.py` has the
radius-R support, with a measured table in its own docstring (`:461-471`) at
`gamma/h = 0.02`:

| radius | residual error against exact cell integrals |
|---|---|
| 0 | 4.07e-02 |
| 1 | 1.48e-03 |
| 2 | 7.45e-04 |
| 3 | 7.10e-04 |

The module is unwired for a reason it states itself (`pole_local.py:4-20`): on the only
production CNT bed there was nothing to correct, zero of 144 modes below one grid
spacing. That premise no longer holds. `pole_sector_observations.md` Sec. 1.8 records
that Si carries a population that is simultaneously under-resolved and isolated, with
lower-quartile linewidths misrepresented by 78 % to 1290 %.

Whether the wired path's radius-zero assignment is the limiting error on Si is not
established. The residual floor on the pole arm sits near 4e-2 and the radius-zero
quadrature error above is 4.07e-02, but these are different quantities measured on
different beds and the agreement is not evidence. It is a cheap experiment once the
census says whether the narrow population survives self-consistency.

**`band_edges` is never passed.** `PoleSector` accepts it, and both `screen`
(`pole_sector.py:516-519`) and `trust_radius` (`:539`) use it, but the solver constructs
the sector without it at `solver.py:760` and `:893`. So the `edge_factor` gate never
fires and the trust radius never sees a branch point. Sec. 13's concern about poles near
a support boundary therefore has no guard in production at all.

**`window(low_freq_mask=...)`** has the parameter but neither call site (`:474`, `:985`)
supplies it, so the documented auto-resolution degrades to `4h`.

**`eps_left`** is computed and reported per candidate (`pole_nevp.py:353`) and gated on
nothing.

---

## 3. Missing

### 3.1 The outgoing sheet (Secs. 1-3, T1)

This is the one gap that changes what the computed numbers mean.

The operator the Newton solves is

    M(z) = z^2 I - D - Sigma_s(z) - Sigma_c(anchor)

`m_blocks` at `pole_sector.py:931-944`. Only the scattering self-energy is continued in
`z`. The contacts are sampled at the real grid frequency nearest a fixed fit anchor by
`_obc_at` (`:897`) and held there, and `dm_blocks` (`:946`) carries no contact
derivative, so `dSigma_c/dz` is identically zero in the Jacobian as well. The docstring
at `:730-735` states this: "continuing the lead self-energy off the axis is the
outgoing-sheet work".

The consequence is that the roots are poles of a frozen-contact operator, and the
radiative width the sector reports is a real-axis golden-rule width rather than the
width of an outgoing quasi-normal mode. For a resonance whose linewidth is dominated by
anharmonic scattering this is a good approximation. For one whose width is set by escape
into the leads it is not, and the backlog's Secs. 1-3 are entirely about the latter.

`PoleSectorConfig.sheet` promises otherwise. It is declared at `config.py:224`, settable
through `QX_POLE_SHEET` (`engine/run.py:106`), and validated at `config.py:2316` to
require `obc.algorithm="spectral"` — and read by nothing else. A run with
`sheet="outgoing"` is bit-identical to one with `sheet="physical"`. The other `sheet`
argument in the tree, `pole_kernel.py:542` and `pole_probe.py:76`, selects sheet I or II
of the *scattering* continuation and is unrelated.

Closing the gap is cheaper than it looks. `Spectral.__call__`
(`src/qttools/boundary_conditions/obc/spectral.py:473`) takes `a_ii, a_ij, a_ji` rather
than a frequency, so it can be evaluated at complex `z` by handing it complex blocks.
The obstacle is mode selection. `_find_reflected_modes` (`:198`) partitions on
real-axis heuristics: propagating requires `|Im k| < min_decay` with `min_decay = 1e-3`,
a group-velocity ratio above `min_propagation = 0.01`, and `Re dE/dk < 0`; evanescent
requires `Im k < -min_decay`. Off the real axis every mode acquires `Im k != 0`, the
propagating mask collapses and everything falls through to the decaying branch. That is
the correct analytic continuation of the outgoing set, but the thresholds are calibrated
for the axis: at Si's `Omega ~ 10 THz` a width of `gamma ~ 1e-2 THz` already pushes
`Im k` past `min_decay`. A controlled 1D lead/device regression with known Bloch factors,
which is what Sec. 3 and T1 ask for, is the way to find out whether the partition stays
continuous as `z` descends.

A related and much cheaper item: the census already produces upper-half-plane roots.
`pole_sector_observations.md` Sec. 1.8 records negative `gamma/sep` entries. `screen`
refuses them on the hard gate, but they are not counted as continuation failures, which
is the reporting Sec. 3 asks for.

### 3.2 The cheap scout (Secs. 4-6, Sec. 41)

Nothing in the tree is named or structured as a scout, and every candidate inside the
window pays for a full bordered Newton solve. The only pre-NEP filter is the window
bound itself (`pole_sector.py:989-999`).

Half of the scalar scout is nevertheless present. `harmonic_seeds`
(`pole_sector.py:954-1007`) eigendecomposes the symmetrised `D`, keeps the modes inside
the window, and forms the quadratic form `v^H Delta v` directly on the stored sparsity
pattern without densifying, over the whole spectrum in one pass. What it returns is
`Omega - i gamma` with the golden-rule width. Eq. (3) of the backlog adds the real shift
and the `s'` denominator; the structural change is to use the result as a gate rather
than only as a seed.

The modal-mixing diagnostic `eta_mix` (Sec. 6, Eq. 5) and the small-cluster linearised
scout (Sec. 5, Eq. 4) have no counterpart.

### 3.3 Sensitivities (Secs. 10-11)

The contraction itself is present and verified. `pole_tracker.py:57` `predict_shift` is
exactly `dz = l^H dSigma^R r` under `l^H M' r = 1`, and
`test_pole_tracker.py::test_predictor_is_first_order_accurate` drives it at
`eps = 1e-3, 1e-4, 1e-5` against exactly recomputed poles and asserts the relative
error falls by a decade each step. So Eq. (10) is not unimplemented mathematics.

What is missing is the decomposition. `predict_shift` is used as a PREDICTOR, applied to
the total change in the scattering self-energy between iterations, and its result is
never split by coupling channel or reported per pole. Secs. 10-11 want
`dz/dlambda_j` for `j` in {left contact, right contact, anharmonic}, which is the same
contraction against each component separately. The two contact channels cannot be
built until Sec. 3.1 is closed, because the operator's contact derivative is zero by
construction; the anharmonic channel can be built today.

### 3.4 Calibration beds (Secs. 36-37)

Neither "mandatory" bed exists. Nothing in the tree mentions Fano. Overlap is touched by
`test_pole_keldysh.py:179` and `test_pole_tracker.py:137,151`, but nothing sweeps
`|z1 - z2|/(gamma1 + gamma2)` against a dense exact resolvent, which is what would
calibrate the simple-versus-cluster transition currently fixed at 0.5.

### 3.5 Finite-support line weight (Sec. 13, Eq. 14)

Only the infinite-grid formula is implemented. The finite-support correction matters at
the acoustic end and at the `fmax` edge, which on Si is where the under-resolved
quartile sits. A partial test exists
(`test_pole_sector.py::test_the_grid_already_carries_a_barely_unresolved_line` truncates
at a finite window and uses the exact Lorentzian tail as its accuracy floor), so the
quantity is not unmeasured; it is simply not available as a gate.

### 3.6 Smaller gaps found while mapping

- Nothing brackets the log/series branch switch in the covariance kernel at
  `|s| = 0.25 * reach` and asserts continuity across it. Both branches are pinned to
  mpmath on the same bed, which bounds any jump indirectly, but the switch itself is
  untested.
- `q_stride` and `q_max` are exposed through the engine (`QX_POLE_QSTRIDE`,
  `QX_POLE_QMAX`) with no test that a strided-out q is bit-identical to baseline.
- No test runs several SCBA iterations and asserts the promoted count settles. Every
  anti-cycle gate is single-step or structural.
- There is no automated conservation test anywhere in the pole suite. Energy balance is
  the sector's most-cited failure mode and appears only as prose in docstrings; the
  real gates are offline in `phonon/studies/`.

---

## 4. Test coverage against the T0-T10 checklist

23 pytest files, 399 tests plus 16 config tests, 90 s for the whole pole set. Every
one is synthetic; there is no real-device bed in the pole suite, and device-scale
validation lives in `phonon/studies/`. `test_pole_local.py` alone is 66 s of the 90;
`-k "not pole_local"` gives the other 22 files in about 23 s.

**Run them with `QTX_ARRAY_MODULE=numpy`.** The default backend on this laptop is
cupy, under which 190 of the 399 fail: the helpers build numpy arrays and hand them to
`xp`-typed production code. That is a property of the test helpers, not of the
production path, but it means an unqualified `pytest` reads as catastrophic failure.

| Block | State |
|---|---|
| T0 sign convention | Covered in substance, never by name. `e^{-i omega t}` is pinned implicitly by `test_pole_bubble.py::_pole_transform` and the time-domain route; `Im z < 0` is asserted in six places; `gamma = -Im z` is the working definition throughout. `test_pole_retarded_assembly.py` names the stored-versus-theory sign convention and carries the negative control |
| T1 outgoing sheet | The weakest block. Continuation into `Im z < 0` is covered for the SCATTERING self-energy (`test_pole_kernel.py::test_second_sheet_is_continuous_across_the_axis`, `::test_branch_cut_jump_is_delta`). Nothing asserts a Bloch factor anywhere in `tests/`, and there is no contact-coupling continuation. `sheet="outgoing"` has one config-validation test and no numerics |
| T2 NEP | Strong except for the scout. Bordered Newton against a closed form, `d = 1`, residue by closed form AND by an independent contour integral, left vector against an SVD null space, `dz_est` as the true frequency error. The sensitivity formula IS verified by finite difference: `test_pole_tracker.py::test_predictor_is_first_order_accurate` drives `dz = l^H dSigma r` at eps 1e-3/1e-4/1e-5 and asserts first order. Scalar-versus-batched identity is covered six ways in `test_pole_batching.py`, with `::test_a_shared_anchor_would_have_been_visible` as the negative control. Missing: any scout |
| T3 resolution | The infinite formula is fully covered, including the theta function `W(r,x)` over 8 r by 4 x (`test_pole_congruence.py::test_exact_trapezoidal_line_weight`) and the closed-form inversion. Finite support is partial: `test_pole_sector.py::test_the_grid_already_carries_a_barely_unresolved_line` truncates at a finite window and uses the exact Lorentzian tail as its floor, but no closed-form truncated-window weight is derived. No test solves for a pole at the acoustic end or at a support edge; those frequencies appear only as masks |
| T4 covariance | Essentially complete, and checked against mpmath at 50-60 digits rather than against another approximation: zero cell mean, Gram PSD, the conjugate-pair degenerate limit, the small-`s` series branch, the exact `s = 0` limit, rank-one factorisation, chunk invariance, exactness as the line narrows. One gap: nothing brackets the log/series switch at `|s| = 0.25 * reach` and asserts continuity across it |
| T5 physical models | Matrix PSD congruence is covered in five files, with negative controls (`test_pole_audit.py::test_a_band_mask_breaks_positivity`). Overlapping pairs are covered by `test_pole_keldysh.py::_two_mode_d` and the avoided-crossing bed in `test_pole_tracker.py`. A broad-plus-narrow bed exists (`test_pole_congruence.py::_psd_bed`, one narrow pole and two broad ones) but nothing asserts Fano interference, an asymmetric lineshape or a zero-pole cancellation |
| T6 implementation | Dense-versus-rank-one, rectangular cluster pairs, chunk invariance (five places) and the correction vanishing in the resolved limit are all covered. The empty-pole bit identity is the strongest gate in the suite: four `assert_array_equal` tests on the real `compute` output triple, plus nine kernel-level empty-safety tests. CPU/GPU parity is absent from pytest and exists only offline in `phonon/studies/engine/parity_check.py` |
| T7 q | Per-q independence is covered thoroughly (own tracker, own contacts, own anchor, ragged candidate counts). The two-q oracle for the ORDINARY ring exists: `test_sse_phonon_phonon.py:967` `test_compute_coupled_q_matches_reference` with `nq=3`. Coupled q into the pole sector is refused, and the refusal is what is tested (`test_pole_sector.py::test_coupled_q_is_refused_rather_than_folded_wrongly`). `q_stride`/`q_max` have no baseline-identity test. The two multi-rank tests are the suite's only skips; they need `mpirun -n 4 ... --with-mpi` |
| T8 self-consistency | Membership freezing, pole-ID persistence and the structural anti-cycle gate are covered; `test_pole_sector.py::test_a_rescan_adds_candidates_and_never_replaces_the_held_set` is written directly against the measured Si period-two lock. Two gaps: nothing runs N iterations and asserts the promoted count settles, and there is no automated conservation test at all. Energy balance appears only as prose in docstrings; the real gates are offline |
| T9 frozen Si | Not run. This is the gating measurement, see Sec. 6 |
| T10 production ladder | Run once and withdrawn (`pole_sector_observations.md` Sec. 10.1): Si has no fine-grid baseline, since it diverges by `ne = 4001` under every OBC arm tried (Sec. 12) |

There is no `conftest.py` and no pytest fixture in the pole suite; every file defines
its own helpers and there is exactly one cross-file import
(`test_pole_batching.py` importing from `test_pole_sector.py`). The de-facto shared
harness for end-to-end work is `test_pole_sector.py` (`_context_run`, `_run`, `_bed`);
the canonical analytic bed `M(z) = z^2 I + i g z I - D`, where `z`, `r`, `l` and `R` all
have closed forms, lives in `test_pole_nevp.py` and is re-derived by hand in four other
files. `phonon/solver/toy_models.py` has `monatomic_chain` and `diatomic_chain`, unused
by any test, which are the obvious foundation for the missing T1 regression.

## 5. What the backlog could not know

Two structural facts constrain its later phases.

**Sec. 44 (P4, q-coupled analytic correction) is blocked.** The q-sectioned SSE path
raises `NotImplementedError` at `sse_phonon_phonon.py:947`; the missing piece, a
reduce-scatter over `comm.q` on the Sigma accumulator, is identified in the comment
above it. Coupled-q pole sectors are separately refused at `pole_sector.py:734-738`.

**Sec. 39's warning about `final_heat` is exactly right, and the mechanism is named.**
`_phonon_hw_weights` (`core/scba.py:968`) applies `|Re omega|` always and the per-bin
cell widths only when the grid is non-uniform, deliberately, so that the legacy
unweighted sum stays bit-identical on a uniform grid where the constant `dw` cancels in
every conservation ratio. Any absolute quantity built on `bubble_balance`,
`slab_absorption` or `final_heat` on a uniform grid is short one factor of `dw`.

A third, smaller point: the "contour fallback" described in `refresh`'s docstring
(`pole_sector.py:1012`) does not exist. `beyn_contour`, `ellipse_contour` and
`contour_quad_points` were removed on 2026-08-16 as unreachable; historically they were
read by nothing. The actual fallback is a harmonic re-seed and re-solve
(`pole_sector.py:1436-1443`).

---

## 6. Ordering

**Answered, 2026-08-15.** The census ran (jobs 4479538 / 4489601) and its result
is `pole_sector_observations.md` Sec. 13. In short: the bulk of the Si
population broadens away under self-consistency -- the median line goes from
unrepresentable to carried at 3 %, and the number of q with a median line the
grid cannot carry falls 46/81 to 0/81 -- but 66 of 81 q still hold at least one
line the grid cannot represent, and the survivors are OVERLAPPING rather than
isolated (median `gamma/sep` 0.751 -> 1.52).

That is the third branch of Sec. 51, not the first or the second. It also
settles the ordering below: the outgoing sheet (Sec. 3.1) and the cluster
representation matter for a tail; the multi-cell wiring (Sec. 2) does not,
because the cells it would correct are no longer the ones carrying the physics.

The census also showed that the shipped gate would have given the OPPOSITE
answer -- accepted 584 -> 504 and under-resolved 95.5 % -> 95.1 %, i.e. "the
population survived" -- because `leg_weight_tol` defaults to 0 and acceptance is
still decided by the crude `q_omega` ratio. That is Sec. 2's first entry, and it
is no longer a tidiness item.



The backlog's own decision tree (Sec. 51) turns on one measurement: does the narrow,
isolated Si population survive anharmonic self-consistency? If it does not, Secs. 43-46
are wasted. `pole_sector_observations.md` Sec. 1.8 reaches the same conclusion from the
other direction, closing on "A converged Si census is the number that would settle it".

So the order is: make the census report enough to be decisive, run it, then choose a
branch. The census plumbing exists and the run is roughly 0.5 node-hours.

Sec. 50's ten actions, re-ordered against this tree:

1. Already done (Sec. 25 wiring; the design is what failed, not the wiring).
2. Covered by `test_pole_batching.py`.
3. Naming only, no factor to fix (Sec. 1 above).
4. Real, and the largest single item (3.1).
5. Already done, closed-form (`pole_kernel.py:175`).
6. Half done, needs the gate rather than the seed (3.2).
7. Exists unwired, held for the census (Sec. 2).
8. Already rank-one on the production path (`pole_covariance.py:247`).
9. Already implemented (`pole_covariance.py:191`, `pole_bubble.py:179`).
10. This is the gating run.

---

## 7. What this pass changed (2026-08-15)

Everything below is laptop-side and default-off or diagnostic. The empty-pole
bit-identity tests still pass exactly, so no production number moves.

**A defect, not a gap.** `window()` took its upper bound from `self.freqs[-1]`, the
rank-local frequency slice, while `PhononSolver._pole_frequency_context` states the
invariant that "every rank ends up with the same operator and solves the same poles".
On a distributed run each rank therefore searched a different window and could promote
a different set. Now taken from `global_freqs`. Serial runs are unaffected, because
there `global_freqs` IS `freqs`.

**Named conventions** (Sec. 8, T0). `PoleSolution` gained `omega_pole`, `gamma_hwhm`,
`Gamma_fwhm` and `is_passive`. `tests/quatrex/phonon/test_pole_conventions.py` pins the
convention against physical requirements rather than prose: a lower-half-plane pole
carries positive spectral weight and its mirror carries negative weight; half maximum
is reached one `gamma_hwhm` off resonance; and the resolution gate consumes the half
width, checked through the closed-form inversion at three tolerances.

**Finite-support line weight** (Sec. 13, Eq. 14). `leg_weight_error_finite(gamma,
centre)` beside `leg_weight_error`. It reduces to the infinite formula where the line
is under-resolved and well inside the support, and deliberately does NOT elsewhere:
truncation cancels in the ratio, so a resolved line floors on the Euler-Maclaurin
endpoint term rather than on the pole. Measured at `h = 0.125` on a 20 THz grid, a half
width of 0.4 reads 3.4e-07 against 3.7e-09 from the infinite formula.

It also inverts near the boundary, and that is the point of having it. A broad line at
the support edge loses tail off the end of the grid and reads 1.3e-03 where the
interior formula says 3.7e-09; a narrow line there reads 2.0e-02 where the interior
formula says 0.78. The narrow case is a true statement about represented WEIGHT and not
a licence to leave the pole on the grid, since the in-band lineshape is still
mis-registered. That is why it is a census column and not a refusal gate.

**Sensitivities** (Secs. 10-11). `PoleSector.sensitivities(sols)` returns
`dz/dlambda = l^H Sigma_s^R(z) r` for the anharmonic channel, reusing the
block-tridiagonal contraction the predictor already uses, against the full `Delta`
rather than its change. Verified two ways in
`tests/quatrex/phonon/test_pole_sensitivity.py`: first-order convergence against a
re-solve at the perturbed coupling (median relative error 3.2e-07 at `eps = 1e-3`,
4.7e-08 at 1e-4), and the physical check that on a contact-free bed the one available
channel comes back carrying the entire half width, `gamma_sens/gamma = 1`.

**Census columns** (Sec. 38). `_census_rows` now carries `chi = gamma/Delta_pole`,
`leg_weight_error`, `E_finite`, `gamma_sens_anh` and `passive`; the summary prints
three more distributions and counts upper-half-plane roots as CONTINUATION FAILURES
rather than folding them into the refusal histogram.

**The two mandatory beds** (Secs. 36-37).

`test_pole_fano.py` builds one broad and one narrow mode of the same operator
(`M(z) = z^2 I + i z G - D` with non-scalar `G`), widths 0.795 and 0.0065, a factor 122
apart. It asserts the interference: the spectral function does not return to the same
level on the two sides of the narrow resonance, and the asymmetry grows with distance
(0.29 at 30 half-widths, 0.49 at 60), which a lone Lorentzian cannot do. Then it pins
the selection claim: at a spacing between the two widths the exact gate calls the broad
mode carried to better than 1e-12 and the narrow one misrepresented by more than 100 %.

`test_pole_overlap.py` sweeps the pair through the transition and calibrates the
hard-coded 0.5:

| `chi = gamma/Delta` | `eps_occ` (scalar vs coherent) | `eps_pole` (coherent vs exact) |
|---|---|---|
| 0.085 | 2.7e-02 | 7.5e-03 |
| 0.164 | 5.5e-02 | 7.4e-03 |
| 0.320 | 1.1e-01 | 7.4e-03 |
| 0.633 | 2.3e-01 | 8.2e-03 |
| 1.570 | 2.9e-01 | 8.1e-03 |
| 15.63 | 2.9e-01 | 8.0e-03 |

Two results. `eps_pole` is flat across the whole sweep, which is audit Sec. 14
measured: overlap does not break the pole representation while the roots stay distinct,
it only removes the ability to speak about the modes separately. And `eps_occ` rises
through the transition and saturates near 29 %, so 0.5 sits just past the 10 %
crossing and just before saturation -- a defensible switch point rather than an
arbitrary one. The test fails if the curve moves out from under it.

**Housekeeping.** Three unused imports and the never-called `PoleSector.predict()`
removed; the module and `refresh` docstrings no longer describe a contour fallback that
does not run; `band_edges` documents that no solver call site supplies it; the nine
config fields nothing reads are marked `INERT` in place rather than deleted, so
existing TOML still validates.

**Run the suite with `QTX_ARRAY_MODULE=numpy`.** Under the default cupy backend on this
laptop 190 of 399 pole tests fail in the helpers, not in production code.

---

## 8. Outcome of the Sec. 6 spatial recommendation (2026-08-15)

Sec. 6 above recommended the proposal's spatial leg on the strength of the
long-CNT factor-2.2 bracket. That recommendation was mis-aimed and is recorded
here rather than quietly dropped.

What was built and stands: `src/quatrex/phonon/spatial_modes.py` (device complex
bands, decay lengths, `band_range_cells`), the range measurements on Si and CNT,
`xi = v_g/gamma`, and the Phase 7 reconstruction `G(n) = V diag(lambda^n) C`
exact to roundoff on real cells. All diagnostic; none wired into a solver path.

What does not follow: that the ring truncates long-range `G`. It does not. With
a nearest-neighbour vertex the reachable leg distance is `2p + 1 = 3` and
`sse_g_band` already defaults to 3, capped at 3. The bracket is not explained by
the ring's band.

The live spatial approximation is the hard-coded pin of `Sigma` to
`|I-J| <= 1`, worth about 11 % and insensitive to the range of `G`. It is
repaired by a non-tridiagonal `Sigma` -- the proposal's Sec. 32 Schur complement
-- and not by the low-rank `G` of Secs. 33-34. Derivation and numbers:
`spatial_truncation_derivation.md`.
