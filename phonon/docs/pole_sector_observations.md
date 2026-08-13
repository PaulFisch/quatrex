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
