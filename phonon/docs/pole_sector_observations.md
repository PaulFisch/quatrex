# Pole sector: observations

2026-08-13. Everything measured in this investigation, with the number and the
run it came from. Read `pole_sector_state_and_next_steps.md` for what to do
next and `pole_scba_routes.md` for the formulas.

---

## 1. Asymmetric, sign-indefinite heat: solved twice, for two different reasons

The same symptom has appeared from two unrelated seeds, and both are fixed. The
profile shape identifies a runaway; it does not identify its cause.

### 1.1 Pole-driven — solved by the congruence route

CNT, job 4398590, one bed, one iteration count, three routes:

| run | `final_heat` | `lead_current` |
|---|---|---|
| base | `[66.689 64.825 64.601 65.023 66.691]` | 66.690 |
| `keldysh` | `[-1553.1 561.6 6492.8 3407.8 1678.0]` | 1615.6 |
| `congruence` | `[66.674 64.808 64.584 65.007 66.677]` | 66.676 |

The wild profile is the superseded `keldysh` route. Its cause is a frozen
indefinite Keldysh remainder: `G_reg = G - G_PP` is a difference of PSD objects,
it is exact only at cell centres, and off-centre it inverts sign at a **20 %**
residue error. That inversion is the anti-damping, and `lead balance = 2.0000`
exactly — emitting into both leads — is its fingerprint.

`congruence` splits the RETARDED function instead and lets the Keldysh
components follow, so the leg is a congruence of a PSD source and cannot change
sign however bad the pole model is. It tracks base to the fourth digit.

The older `rr_ss_sr` runs show the same signature and are the same route:
`run_cfull = [-27.4 49.0 39.1 33.6 135.6]` against `run_cbase =
[53.8 51.8 51.9 52.1 53.8]`.

### 1.2 d5a — a different problem, also already solved

The `prod/sinw_d5a` arms are **not pole runs**: `retarded = half`, `eta = 0.11`,
`ne = 41`, sector off. They say nothing about the pole method. Every `_anh` arm
has a uniform physical `best` and a blown-up `final`:

| arm | `best` | `final` |
|---|---|---|
| `L2_anh` | `[2.561 2.529 2.561]` | `[31.3 2144.1 121.3]` |
| `L3_anh` | `[1.700 1.678 1.671 1.700]` | `[445 -1907 -1556 -907]` |
| `T150_anh` | `[2.161 2.129 2.161]` | `[358 -2098 -711]` |
| `T30_anh` | `[0.826 0.824 0.836]` | `[0.639 0.883 0.760]` |

`d5a_eta0_bisection.md` (2026-07-08) diagnosed this by surgical frequency
ablation. The seed is the **sub-grid infrared**: soft twist modes at
0.0075-0.027 THz, 20-50x below the first bin of any affordable grid, whose
`1/omega` Bose weight enters the bubble unresolved every iteration. A 1.5 THz
low-frequency cutoff is **necessary and sufficient** — monotone descent to
3.1e-4 in 150 iterations, heat `[5.244 5.139 5.177]`.

Grid refinement does not help: nf181 aborts at iteration 70, nf361 at ~50,
nf721 at 90. Every non-IR hard cut merely relocates the blow-up to its own
boundary.

`d5a_gridladder/nf181` (`final_heat = [-57.7 -4.06e+10 131]`) is the *raw* row
of that table — a known-diverging control arm, not a production result.

### 1.3 Why they look identical

One end state, two seeds. Once `|Sigma|` exceeds the dynamical-matrix scale the
global Kramers-Kronig `Re Sigma^R` shifts every pole, the in-band DOS collapses
(measured: 1.61 -> 1e-3 while `Sigma` pumps to 1e17), and the heat profile goes
asymmetric and sign-indefinite. Reading a mechanism off the profile is the thing
to avoid.

---

## 1.4 CNT runs stably WITH the pole sector (job 4443181, 2026-08-13)

The first CNT A/B at the production mixing (0.2), rather than the 0.02-0.1
every earlier pole run used. 150 iterations budgeted, `ne = 181`; the `cong`
arm reached 92 before the debug walltime.

| | base | `cong` |
|---|---|---|
| `rel Sigma`, it 10 | 8.8647e-01 | 8.6659e-01 |
| it 50 | 7.2735e-02 | 1.0079e-01 |
| it 90 | 2.0458e-02 | 1.9782e-02 |
| min over the run | 1.5746e-02 | 1.7212e-02 |
| lead balance, it 90 | 5.2074e-04 | 5.6097e-04 |
| **SIGN INVERSION lines** | 0 | **0** |
| `P_in`, it 90 | -3.078784e+04 | -3.077554e+04 |
| worst bubble balance residual | 3.476e-07 | 3.266e-07 |

`cong` tracks base through the whole descent, agrees on `P_in` to 0.04 %, and
never inverts. Zero sign inversions is the result: that is the failure which
killed every earlier pole run by iteration 2, and it is gone.

The promotion is also stable now, which it was not:

    it 1: 6 cluster(s), 12/144 promoted   refused: eps_z x128, grid-resolved x4
    it 2: 5 cluster(s), 11/12 promoted    refused: eps_z x1

12 of 144 under the frequency-unit acceptance against 2 under the old matrix
residual, and the tracker retains 11 of 12 across the iteration rather than
re-deciding from scratch.

**What this does NOT show.** Neither arm converges. Both descend to ~1.6e-02
and then oscillate between 1.6e-02 and 2.4e-02 without tightening -- base
included, so it is not the sector. `cnt33_L4_linear` converges at 249-311
iterations under the same recipe, but that is a different bed on tortin;
`pgate` may simply have a residual floor. A 350-iteration run is queued to
settle it.

Timing, for budgeting: 10 s per iteration on one GH200 for this bed, so 150
iterations is roughly 25 minutes and a three-arm 350-iteration comparison does
not fit in a two-hour job.

---

## 1.5 The covariance correction on device (jobs 4444772, 4445828)

20 iterations, `ne = 181`, `mix = 0.2` -- short on purpose: this answers
whether the path executes at device size, not anything about physics.

| arm | `rel Sigma` @20 | lead balance | `lead_current` | wall |
|---|---|---|---|---|
| base | 9.4272e-01 | 8.2976e-04 | 41.91546925712334 | 163 s (cold) |
| gate0 | 9.4272e-01 | 8.2976e-04 | 41.91546925712334 | 76 s |
| corr | 9.2332e-01 | 8.5696e-04 | 41.872593172402816 | 317 s |

**Gate 0 is exact.** `bubble_correction` ON with an EMPTY pole window
reproduces base to every printed digit, heat profile included. With nothing to
correct the correction is nothing, which is the precondition for the third row
meaning anything.

`corr` runs stably: no sign inversion, lead balance unchanged at 8.6e-04, heat
profile physical. It moves `lead_current` by **0.10 %** and costs about 4x the
warm baseline. Active cells vary 1-11 across iterations as the promoted set
moves.

The first attempt (4444772) raised `modal_vertex_blocks: families disagree, 4
against 24 modes` -- my own guard, firing loudly rather than returning a
number, on an assumption that was wrong: paired cells can belong to different
CLUSTERS with different pole counts, and the mixed vertex projection is
rectangular.

## 1.6 The resolution gate is mis-calibrated, and the run says so

From the same run's population diagnostic, at production mixing:

    coverage: candidates 144 -> in window 144 -> unresolved 140 -> important 140
              -> root solved 12 -> representation valid 12 -> active 12
    population: median h/gamma=0.65, median gamma/spacing=8.40, 94% overlapping

`q_omega = gamma/(2h) < 1` calls **140 of 144** under-resolved. But at
`h/gamma = 0.65` the exact worst-case line-weight error is **1.3e-04**: the
grid carries those lines to 0.013 %, however they fall between nodes. The old
rule flags almost everything and then leans on the root solve to refuse it,
which is why the refusal histogram is dominated by `eps_z` rather than by
resolution.

`leg_weight_tol` (default 0 = legacy rule) makes the test exact:

    E_leg^max(r) = coth(pi/r) - 1 = 2/(e^{2 pi/r} - 1),   r = h/gamma

refuse when that is below tolerance. Verified against the closed-form inverse
`h/gamma < 2 pi/log(1 + 2/eps)`: 1.185 for 1 %, 1.692 for 5 %, 2.064 for 10 %.

The second population number is independent and equally decisive: median
`gamma/spacing = 8.40` with **94 % overlapping**. An isolated simple pole needs
below 0.5. So CNT fails both criteria at once -- the grid does not need help,
and no isolated pole exists to give it.

---

## 1.7 The q-resolved beds, and why the census has not run yet (job 4448828)

Survey of every q-resolved bed in the tree:

| bed | q | cells | `retarded` | `ne` |
|---|---|---|---|---|
| MoS2 `mos2L2conv` / `L4conv` | 25 (`[5,5,1]`) | 2 | half | **4001** |
| MoS2 `mos2soodS4` | 25 | 2 | half | **15001** |
| Si `sichk_base` / `ext` | 81 (`[1,9,9]`) | 3 | half | **121** |
| Si `sires1001` / `si4x2` | 81 | 3 / 2 | half | 1001 / 2801 |

Two facts decide what can be asked.

**Every q bed runs `retarded_method = "half"`**, and the sector requires
`"fft"` -- without the Kramers-Kronig real part the operator is not causal and
its roots are not resonances. So no q-resolved bed has ever run in a
configuration the sector can attach to at all. That is a prerequisite, not a
detail, and it is why the fft-convergence question comes before any pole
question on these materials.

**The grids are very fine.** MoS2 at `ne = 4001` cannot have unresolved modes
almost by construction. So "is anything unresolved at the grid these beds use"
is the wrong question; the answer is trivially no. The question that decides
whether the method pays is the converse: at a COARSE grid, is there a
population the sector could carry, so the bed could run coarse instead of at
4001? Nothing else about this method is worth anything if the answer is no.

**The census did not run.** Every q raised one of two errors, and together they
locate the fault exactly:

    q (0, k): could not broadcast input array from shape (9,6,6) into shape (6,6)
    q (1, k): Index 1 is out of bounds for axis 0 with size 1

`_pole_blocks` was handed a bare `(i, j)` while the dynamical matrix carries a
leading singleton where the Keldysh buffers carry frequency, so on a `(1, 9, 9)`
stack the slice consumed the singleton and one q index -- leaving a stack where
a block was wanted -- and every `i > 0` ran off an axis of size 1. The
transverse axes sit at the END of the stack shape; the index has to be padded on
the left to whatever leading rank the buffer has. Same off-by-one in the contact
blocks, which are `(n_freq,) + nk + (b, b)`.

Fixed; unverified until the next q run, since no local bed has a q axis.

---

## 1.8 Si HAS narrow, isolated modes -- the first bed that does (job 4450385)

Extraction-only census, `sichk_base`, `ne = 121`, 81 q sampled every 20th,
`retarded = fft`, iteration 1. Five q-points:

| q | candidates | under-resolved (`q_omega < 1`) | **isolated** (`gamma/sep < 0.5`) | accepted |
|---|---|---|---|---|
| 1 | 16 | 13 | **4** | 1 |
| 2 | 18 | 16 | **6** | 7 |
| 3 | 18 | 12 | **7** | 6 |
| 4 | 18 | 10 | **7** | 6 |
| 5 | 18 | 12 | **6** | -- |

`q_omega` medians run 0.34-0.69 with **minima at 0**, and `gamma/sep` lower
quartiles run 0.013-0.63. Compare CNT, where the median `gamma/spacing` is 8.40
with 94 % overlapping and nothing is below one grid spacing.

Through the EXACT gate (`h = 15.0/120 = 0.125 THz`), the linewidths split the
population sharply:

| `gamma` [THz] | `h/gamma` | `E_leg^max` | where |
|---|---|---|---|
| 0.00286 | 43.7 | **1.29e+01** | q2 p25 |
| 0.00692 | 18.1 | **4.80e+00** | q3 p25 |
| 0.0252 | 4.96 | **7.84e-01** | q4 p25 |
| 0.0849 | 1.47 | 2.84e-02 | q3 median |
| 0.111 | 1.13 | 7.58e-03 | q2 median |
| 0.166 | 0.75 | 4.76e-04 | q1 median |

The MEDIAN mode is fine -- the grid carries it to better than 1 % -- but the
lower QUARTILE is misrepresented by 78 % to 1290 %. That is the regime the
method exists for, and it is the first time any bed has shown it. It also
explains why a median-based summary would have missed it: on CNT the median WAS
the whole story because the distribution was narrow, and here it is not.

So Si carries a population that is simultaneously under-resolved and isolated
-- both criteria at once, which is what a simple-pole representation needs and
what CNT never had. 6-7 poles per q are accepted, so across 81 q the sector
would carry several hundred.

Two caveats on the same table. The `gamma/sep` distributions have NEGATIVE
entries (-0.0119, -0.974, -0.12), i.e. candidates in the UPPER half plane;
`screen` refuses them ("pole is not in the lower half plane") but they are in
the population, so the raw quantiles include unphysical roots. And this is an
iteration-1 census: CNT taught that iteration-1 widths are a LOWER bound,
because the anharmonic width is not built yet. A converged Si census is the
number that would settle it.

## 1.9 MoS2 cannot be censused at that size, and Si's fft arm is KK-truncated

**MoS2 OOMs.** `mos2L2conv` at `ne = 401` with 25 q died with
`OutOfMemoryError: allocating 20,155,392,000 bytes (allocated so far:
87,535,582,208)`. 87 GB was already resident before the failing 20 GB request,
so this is the bed's own footprint at that grid, not the sector's -- but it
means no MoS2 census exists yet, and one will need a coarser grid, fewer q per
pass, or block distribution.

**Si's `fft` arm runs and descends monotonically**, reaching 6.10e-02 at
iteration 25 with lead balance 3.2e-05, internal spread 4.6e-03 and a uniform
heat profile [28.933 28.933 28.968 28.926]. It is slower than `half`, not
unstable. `fft` on MoS2 died the same way its census did.

**But Si's `fft` arm truncates the Kramers-Kronig integral.** Same bed, same
budget, one knob:

| | min `rel Sigma` | sign inversions | KK truncation warning |
|---|---|---|---|
| `si_half` | **1.8621e-03** | 0 | 0 |
| `si_fft` | 2.9761e-02 | 0 | **1** |

    still carries 12.3% of its peak weight at the top of the frequency grid,
    so the Kramers-Kronig integral for Re Sigma^R is truncated.

`half` never computes that integral, so it cannot fire the warning and is
unaffected by the truncation -- which makes this pair a much cleaner instance
of the grid-support hypothesis than the CNT ladder. Same bed, one knob, and the
arm that converges 16x worse is exactly the arm that computes a truncated KK
integral. Extending the grid past twice the band top and re-running `si_fft` is
the direct test, and it is now the most informative single run available.

---

## 2. Converged CNT has no narrow modes

From `phonon/scripts/data/resonance_gain_distilled.npz`, converged
`cnt33_L4_linear`, 144 modes at `dw = 0.3056 THz`:

| | min | p1 | median | max | below 1 spacing |
|---|---|---|---|---|---|
| `Gamma_tot / dw` | 1.573 | 1.867 | 6.438 | 150.4 | **0 of 144** |

`L2_fp` min 1.978, `L2_andstall` min 1.578. The narrowest mode on the device is
1.57 spacings wide and 89 % anharmonically broadened.

So the pole sector, and the local finite-cell correction, have nothing to act on
for CNT at 300 K — not "little", nothing. The earlier `h/gamma = 1.35` figure
came from an iteration-1 snapshot, before the anharmonic width is built: a lower
bound on the converged width, not an estimate of it.

**No verdict on the method can be read off any CNT bed, in either direction.**

---

## 3. What the sector fixes, when there is something to fix

Represented weight of an unresolved Lorentzian as the ring sees it, summed over
all nodes, exact total = 1:

| `h/gamma` | `delta/h` | point sample (baseline ring) | cell average (`congruence`) |
|---|---|---|---|
| 20 | 0.00 | **6.42** | 1.0000 |
| 20 | 0.25 | 0.304 | 1.0000 |
| 20 | 0.50 | 0.156 | 1.0000 |
| 100 | 0.00 | **31.8** | 1.0000 |
| 100 | 0.50 | 0.031 | 1.0000 |

The baseline gets a narrow line's total weight wrong by 6x-1000x depending only
on where it falls between nodes. The cell average is exact at every offset.

**Correction (2026-08-13).** An earlier version of this section quoted
`W_grid = r / (pi (1 + r^2 x^2))` as the closed form of that table. That is the
NEAREST-NODE weight, not the total, and it is wrong for the total by up to 2.5x
(at `r = 100, x = 0.5` it gives 0.0127 against the measured 0.0314). The exact
infinite trapezoidal sum, from
`sum_n 1/((n-x)^2 + a^2) = (pi/a) sinh(2 pi a)/(cosh(2 pi a) - cos(2 pi x))`, is

    W_inf(r, x) = sinh(2 pi / r) / (cosh(2 pi / r) - cos(2 pi x))

and it reproduces every entry of the table to four decimals. Pinned in
`test_exact_trapezoidal_line_weight`.

The gate that follows from it is exact rather than a rule of thumb. The worst
overestimate is at `x = 0`,

    E_leg^max(r) = coth(pi / r) - 1 = 2 / (e^{2 pi / r} - 1),

so a worst-case line-weight tolerance `eps` needs `h/gamma < r_eps` with

    r_eps = 2 pi / log(1 + 2/eps).

| tolerance | `h/gamma` threshold |
|---|---|
| 1 % | 1.185 |
| 5 % | 1.692 |
| 10 % | 2.064 |
| 20 % | 2.620 |

That replaces `samples_per_halfwidth`, which was an arbitrary constant doing the
same job badly.

What the cell average does **not** fix is the bubble. Ring against the exact
cell-averaged convolution at the combination frequency `2 w0`:

| pole offset in its cell | `h = 20 gamma` | `h = 200 gamma` |
|---|---|---|
| 0.00 | 1.0043 | 1.0000 |
| 0.25 | **1.7941** | 1.9754 |
| 0.50 | **0.5363** | 0.5037 |

Controlled by the sub-cell POSITION, not `h/gamma`, and it worsens with
`h/gamma`. The error is confined to cell PAIRS with both ends in a pole cell
(one leg displaced against a resolved partner costs 2 % at the worst placement,
both cost 46 %), so it is an `|P|^2` object.

On CNT that bound is tiny because almost nothing is promoted: pole-cell pairs
carry 6e-05 % of the ring's weight, 2.6 % at the single worst bin.

---

## 4. Promotion: the acceptance units were the blocker

Measured one knob at a time, job 4399332:

| arm | acceptance | trust region | promoted |
|---|---|---|---|
| legacy | scaled matrix residual | grid-tied | 2/144 |
| trust | scaled matrix residual | physical | 2/144 |
| locate | **frequency error** | grid-tied | **11/144** |
| both | frequency error | physical | 9/144 |

`eps_nep = ||M(z)r|| / ((|z|^2 + ||M||)||r||)` is a scaled matrix residual whose
denominator is `1e3-1e4 THz^2` for a phonon operator, so testing it against
1e-10 asks about matrix norms rather than about frequency. Gating on
`eps_z = |dz_est| / min(gamma, separation, h)` is worth 5.5x on its own.

The remaining refusals are genuine: `eps_z` median 9.7e-01, i.e. the median
refused candidate is displaced by about a full limiting scale. The trust region
was grid-tied (`trust_radius_cells * h`), which made grid refinement shrink the
physical pole search — a real defect, fixed, but not what was binding here.

---

## 5. Two things that were implemented and never called

Both found by review on 2026-08-13.

* **`PoleSector.audit()`** — the extraction-only census (doc Sec. 27) had no
  entry point. Now `pole_sector.extraction_only` / `QX_POLE_EXTRACT`, default
  off: it solves the candidates, prints the census as distributions, and hands
  the ring an empty pole set, so the run stays bit-identical to pole-off. It
  also walks the q axis, which is the only way the pole machinery can reach a
  q-resolved bed at all.
* **`pole_local.correct_spectrum`** — 598 lines plus 252 of tests, no caller, no
  config field. Still unwired, and **not in device-viable form**: its own
  docstring says the device path must carry rank-one `p_row`/`q_col` factors
  instead of the dense `(P,) + G.shape` residues it takes, and that substitution
  is not written. One cell pair at `Np = 16` is ~4300 dense `bilinear(x_t, y_t)`
  terms per output frequency.

Separately, **26 correctness tests were deleted without mention** by the commit
that added `pole_local` (1043 lines from `test_pole_congruence.py`, 26 of its 28
tests). Restored; all 41 pass unchanged against current source, so none had been
made obsolete.

---

## 6. Open: the `ne` ladder needs a mechanism

CNT L4, `retarded = fft`, `eta ~ 0`, production recipe:

| `ne` | outcome | `final_heat` |
|---|---|---|
| 161 | diverged | ~ -2.4e+22 |
| 181 | converged, 311 it (`g_band = 2`) | `[35.2 32.3 32.5 32.4 35.2]` |
| 201 | converged, 249 it | `[39.8 36.7 37.1 36.6 39.5]` |
| 271 | did not converge | `[51.6 47.1 47.4 46.5 50.3]` |
| 361 | diverged | ~ +3.9e+19 |

Sec. 2 rules out the narrow-resonance explanation: there is nothing to resolve.
The leading untested candidate is grid truncation — the grid runs to 55 THz
while the 3-phonon bubble has support to ~92.5 THz, so **59.5 %** is covered,
`sse_aux_grid_fmax_thz = 0` switches off the dual-grid extension that exists for
exactly this, and the KK-support warning fires on every CNT run checked. One
rerun of an `ne` rung with `sse_aux_grid_fmax_thz >= 93` would settle it.

A second confound, stated because it has not been separated: the L8/L10 failure
is attributed to `g_band = 2` across families that also differ in
`retarded_method` and `ne`.

---

## 7. q-resolved beds

`kpoint_grid != [1,1,1]` exists only for MoS2 (`[5,5,1]`, 25 q) and Si
(`[1,9,9]`, 81 q). **No run in the recorded history has used `q_comm_size > 1`**
— 94 of 94 are `qcs = 1` — so the q axis has only ever been carried in the
stack. See `pole_sector_coupled_q.md`.

---

## 8. The promoted set limit-cycles (Si, 2026-08-14)

Run `psi2` (daint 4464697), Si `sichk_base`, 81 q, `h = 0.25`, `wmax = 35`,
`retarded=fft`, `mix = 0.1`, `leg="congruence"`. Two arms on the same grid.

    base   150 it   rel Sigma -> 9.2597e-04, monotone, lead_current 400.611
    pole    34 it   rel Sigma oscillating at O(1), no downward trend

The pole arm's per-iteration pole count:

    624 431 576 433 569 436 556 442 578 462 608 466 618 463 630 479
    641 470 628 479 651 479 622 454 634 463 621 450 619 447 624 446 625

A period-2 limit cycle: ~620 poles on odd iterations, ~460 on even, ~170
poles (27 % of the set) entering and leaving every iteration for 34
iterations with no sign of settling. The wall time alternates with it
(185 s against 85 s per iteration) and so does the residual.

**Cause.** `accept="locate"` (the default since the `eps_z` work) tests
`eps_z <= locate_tol` as the FIRST gate in `PoleSector.screen`, and it was a
single hard threshold — `was_promoted` was not consulted. A hard threshold
inside a fixed-point iteration closes a feedback loop: the pole enters the
sector, its leg changes Sigma, `eps_z` drifts past the threshold, the pole is
demoted, Sigma changes back, the pole is re-promoted.

The `q_in`/`q_out` gap was built for exactly this and its config validator
says so ("the gap IS the hysteresis that stops a mode changing sector every
iteration"). It did not help: that gate runs *behind* the `eps_z` one and
never sees a pole `eps_z` has already refused. `leg_weight_tol` has the same
hole and is worse, because setting it *replaces* the `q_in`/`q_out` branch
rather than adding to it, leaving no hysteresis anywhere.

The existing hysteresis test could not catch it: it sets `dz_est = 0`, so
`eps_z` is identically zero and the gate never fires.

**Fix.** `locate_tol_out` (default 0.15 against `locate_tol` 0.05) and
`leg_weight_tol_out` (default `leg_weight_tol / 3`), both validated to lie on
the correct side, both wired into `screen` under `was_promoted`. Note the
inequality runs the other way for the two: `eps_z` refuses a *badly located*
pole so leniency raises the threshold, while `leg_weight` refuses a
*well-resolved* one so leniency lowers it. Regression tests
`test_eps_z_gate_has_hysteresis` and `test_leg_weight_gate_has_hysteresis`.

Hysteresis can only ever retain a pole that would otherwise be dropped; it
cannot admit one that `locate_tol` refused. So this does not loosen
acceptance, and a run that never re-screens a promoted pole is unchanged.

**Not yet verified on device.** The claim that this is *the* driver rests on
the mechanism plus the period-2 signature; it is confirmed when a rerun of
the `pole` arm shows the pole count settling. That rerun is the next step,
and it should come before any of the batching work in
`pole_solve_batching.md` — a 20x faster pole solve is worth nothing while the
SCBA does not converge.

### 8.1 Result (psih, 4466692, 2026-08-14)

Same bed and grid as `psi2`'s pole arm, 21 iterations before the 30-minute
debug wall. Promoted-pole count:

    309 310 301 301 305 300 306 301 299 296 359
    347 349 349 356 346 344 343 337 344 347

**The period-2 limit cycle is gone.** No alternation anywhere in 21
iterations, the band is 296-359 against `psi2`'s 431-651 sawtooth, and two
consecutive iterations (349, 349) return an identically-sized set.

Residual against the base arm at the same iteration:

    base  1.0000 0.9984 0.9695 0.7871 0.9418 0.7604 0.6317 0.8432 0.7594
          0.6715 0.5353 0.4297 0.3997 0.3825 0.3715 0.3688 0.3599 0.3404
          0.3098 0.2714 0.2352 0.1982
    psih  1.0000 0.9990 0.9744 0.9117 0.9100 0.6930 0.8004 0.6562 0.5334
          0.5105 0.4904 0.5104 0.5132 0.4709 0.3930 0.3035 0.2587 0.2291
          0.2012 0.2481 0.2927 0.3054

`psih` tracks the base arm: same order, same wobbly descent, ahead of base
between iterations 9 and 19 (0.533 against 0.759; 0.201 against 0.310) and
behind over the last three. Both arms wobble -- base rises 0.787 -> 0.942 at
iteration 5 and 0.632 -> 0.843 at iteration 8 -- so `psih`'s rise over its
last three is inside the envelope the baseline itself shows, not a turn.
Lead balance stays at 3e-04 throughout, against `psi2`'s 1e-02.

**Confounded, and this must not be forgotten.** `psih` runs `e4e8e05e`,
which carries BOTH the hysteresis fix and the in-flight batching refactor
(`BlockLayout`, `bordered_newton_batch`, vectorised `m_blocks`/`dm_blocks`/
`continue_sigma`). `psi2` ran `f6bd76f7`, which has neither. The refactor is
not numerically neutral: it changes the count at iteration 1, from 624 to
309, where hysteresis provably cannot act because `was_promoted` is False
for every candidate. So this run does not attribute the fix to either
change. The `P = 1` bit-identity gate of `pole_solve_batching.md` Sec. 3 has
not been run and is what would separate them.

Still open: a long normal-partition arm to carry `psih` to the base arm's
9.2597e-04 and compare `lead_current` against 400.611.

---

## 9. Why the poles leave: a control-flow oscillator (pdiag, 4473047)

`siladder`'s pole arm limit-cycles 641 <-> 470 and floors at `rel Sigma`
2.5e-01 where the pole-free arm on the same grid reaches 9.2597e-04. The
cause is neither the acceptance threshold nor the physics. It is the
interaction of two methods in `PoleSector`/`PoleTracker`:

* `_seed` used `harmonic_seeds()` -- discarding the held set -- whenever
  `tracker.needs_rescan()`;
* `_track` calls `tracker.update` on a warm iteration and `tracker.adopt` on
  a rescan. `update` ARMS a rescan whenever the cluster count or any cluster
  size changes; `adopt` DISARMS it.

Membership moving is what changes those counts, and membership moves every
iteration, so every warm iteration armed the next rescan, and every rescan
threw away the held poles and re-seeded from the full harmonic spectrum,
which re-found them. The sector therefore alternated between everything the
spectrum offers and whatever survived screening it. Period two, locked, for
as long as the run lasted.

Instrumented (`n_seeded` / `n_matched` / refusal histogram over q):

    it  seeded  matched  accepted  refused   seeding
     1    1456        0       626      830   harmonic
     2     626      514       421      205   warm
     3    1422      338       555      867   harmonic
     4     572      540       448      124   warm
     5    1356      421       579      777   harmonic
     6     678      586       482      196   warm
     7    1456      459       626      830   harmonic
     8     626      613       489      137   warm
     9    1379      480       648      731   harmonic

`seeded` alternates ~1400 <-> ~620, and iteration 7 revisits iteration 1
exactly (1456 seeded, 626 accepted): a genuine cycle, not drift.

**The threshold is not the sensitive knob.** `eps_z` is sharply bimodal --
accepted median 1e-14 to 1e-9, refused median 0.6 to 1.2, against a
tolerance of 0.05. The refused poles miss by one to two orders of magnitude,
not by a hair, so no widening of the `locate_tol`/`locate_tol_out` gap would
retain them. The `locate_tol_out` hysteresis added in Sec. 8 was aimed at the
wrong mechanism: the oscillation is decided in the SEEDING, before any pole
reaches `screen`.

**Why the residual floors.** Under `leg = "congruence"` the pole channel is
the point-minus-cell-average correction, so the ring convolves the cell
average of the reconstruction. An alternating pole set therefore makes
`Sigma` alternate by a finite amount, and `rel Sigma = ||Sigma_new -
Sigma_old|| / ||Sigma||` cannot fall below that jump. The 2.5e-01 floor is
the discontinuity, not slow convergence.

**Fix.** A rescan now ADDS the harmonic candidates the held set does not
already cover (within `cluster_factor * h`) instead of replacing it, so a
rescan can only ever grow the candidate set. Regression test
`test_a_rescan_adds_candidates_and_never_replaces_the_held_set`.

**Also found: `PoleTracker.membership_frozen()` has no caller.** It is the
mechanism for holding sector membership fixed across an `epoch_iterations`
epoch -- exactly the remedy for a discrete set entering a fixed-point map --
and it is dead code.

### 9.1 The fix on device (pfix, 4473281)

30 iterations, same bed and grid as `siladder`'s pole arm. Accepted poles:

    pfix      626 508 548 547 560 574 573 579 586 593 608 595 600 596 607
              595 611 604 611 596 611 612 612 612 600 610 597 589 592
    siladder  626 421 555 448 579 463 592 487 606 482 626 489 648 485 652
              488 649 487 663 511 647 482 649 474 649 474 ...

The period-two cycle is gone. `seeded` decays 1456 -> 951 -> 848 -> 805 ->
791 and then locks at ~775 instead of alternating ~1400 <-> ~620, and
`matched-as-promoted` rises to ~600 of ~600 accepted: the sector now carries
its poles across iterations rather than rediscovering them.

The residual over the last ten iterations is monotone -- 7.09 6.96 6.67 6.31
5.83 5.39 4.85 -- where `siladder` alternates 2.3 <-> 4.9 at the same point.
Smooth descent is the signature wanted, but `siladder` sits at LOWER absolute
values here, so 30 iterations does not yet say the fix converges further;
`pfix150` (4473420) runs the full 150 against base's 9.2597e-04.

Conservation, from `siladder` with `QX_BBCHECK=1`, is worth recording
separately because it isolates the defect from the physics:

    base   P_in = P_out to 10 digits,  resid 8.6e-10
    pole   P_in = P_out to  6 digits,  resid 7e-07 to 1.3e-06
           and P_in itself ALTERNATES: -2.4328e+05 <-> -2.4766e+05

So the leg is Phi-derivable and conserving; what oscillated was which poles
were in the sector, and it showed up directly in the total power at 1.8 %.
The pole arm's conservation residual is nonetheless 1000x looser than
base's, and that is still unexplained. The obvious story -- that the
diagnostic pairs `Sigma` built from cell-averaged legs with the
point-sampled `data.g_*`, so the residual measures the correction's size --
is REFUTED by the run: pairing each iteration's pole count against its
residual gives

    ~640 poles   resid  7.4  6.9  8.2  4.1  5.8  6.1  8.3  6.6   (x1e-7)
    ~470 poles   resid  8.1  8.8 10.6  9.8  9.1  9.2  9.4 13.1   (x1e-7)

More poles gives BETTER conservation, where that story predicts worse. So
the channel is not leaking -- carrying more of the spectrum in the sector
makes `Sigma` and `G` pair more consistently. (Confounded: the two phases
are different points in a limit cycle, so the pole count is not the only
thing that differs between them. Worth redoing on `pfix150`, where the set
is stable and the count can be varied deliberately through the pole
window.)

Two things seen but not chased. About 25 % of candidates are refused on
`eps_z` even with a stable set, which is expected -- the harmonic spectrum
offers every mode, not only the ones that are resonances -- but it has not
been confirmed that the refused ones are the same modes each iteration. And a
handful of candidates per iteration are refused as "not in the lower half
plane": the corrector returning an anti-damped root, harmless because it is
refused, but it means the Newton is leaving the physical sheet.
