# Pole sector: state, and what production CNT is actually blocked on

2026-08-11. Companion to `pole_scba_routes.md`, which documents the three
routes' formulas and their device behaviour. This one answers a narrower
question: what works, and what has to happen before the method can be judged
on a production CNT device.

---

## 0. The headline

The method is not blocked on its own algebra. Every route's kernels are
verified, the default route is device-stable, and the failure modes that were
open a day ago are now measured rather than argued.

It is blocked on two things, and the second is the important one.

**Every pole-sector A/B so far ran on `cluster/pgate`**, at a mixing factor
and iteration budget on which the SCBA does not converge with or without the
sector. A converged `eta = 0`, Hilbert-corrected CNT recipe exists and has
existed for a while; the sector has never been run on it.

And **the converged baseline has no grid limit to be compared against.**
Refining the frequency grid does not improve the CNT answer, it destroys it
(Sec. 2). That is the pathology the pole subtraction exists to cure, so the
success criterion is not "agrees with a fine-grid baseline" — there is no
fine-grid baseline. It is "gives a stable answer on a coarse grid where the
baseline cannot".

---

## 1. What works today

### The routes

| route | state |
|---|---|
| `leg = "congruence"` (default) | Device-stable. Tracks the pole-free baseline at every iteration; `lead balance` ~1e-5, `P_in` matches `P_out` to 1e-8. Bit-identical to the grid solver on an empty pole set. |
| `leg = "congruence_analytic"` | Diverges, structurally. Three assembly defects fixed; what remains is the construction — see Sec. 4. |
| `leg = "keldysh"` | Superseded. Kept only to reproduce old runs. |

### The gates that now run every iteration

* `ring leg positivity`, per component, with its `omega_index`, the ring's own
  low-frequency mask applied, a **pole-off control** printed beside it, and a
  second reading restricted to the cells the poles touch.
* `pole registration` — the worst sub-cell pole offset, plus the fraction of
  the ring's weight sitting on pole-cell PAIRS.
* `pole sector: N/M pole(s) promoted, refused: <histogram>` — the promotion
  yield, not a bare count.
* `eps_tail`, `eps_fit` and `eps_reg` on the analytic route.
* Sector sum against a high-order quadrature of the same hybrid (offline).

Everything above was added because a number was being read without its
control. The controls are the point.

### Converged production CNT (tortin, `phonon/studies/out/anderson_test/`)

| bed | `retarded` | `eta` | `ne` | iterations |
|---|---|---|---|---|
| `cnt33_L3_linear` | fft | 0.0 | 181 | 52 (with a 2 THz low-freq mask), 209 without |
| `cnt33_L4_linear` | fft | 1e-12 | 181 | 309-311 |
| `cnt33_L5_linear` | fft | 1e-12 | 181 | 304 |
| `cnt33_L6/L7_linear` | fft | 1e-12 | 181 | 239 / 313 |

`mixing_method = "linear"`, `mixing_factor = 0.2`, `max_iterations = 450`,
`sse_g_band = 3` (the default). These converge **with** the Kramers-Kronig
half — `retarded_method = "fft"`, which is what the pole sector requires.

---

## 2. The target: grid independence, not a converged reference

CNT L4, `retarded = fft`, `eta ~ 0`, production recipe, measured
(`cluster/cnt-L4-nescan`, `cnt-L4-ne361`, `cnt-L4-gband2`):

| `ne` | outcome | `final_heat` |
|---|---|---|
| 161 | **diverged** | ~ -2.4e+22 |
| 181 | converged, 311 it (`g_band = 2`) | [35.2, 32.3, 32.5, 32.4, 35.2] |
| 201 | converged, 249 it | [39.8, 36.7, 37.1, 36.6, 39.5] |
| 271 | did not converge | [51.6, 47.1, 47.4, 46.5, 50.3] |
| 361 | **diverged** | ~ +3.9e+19 |

The SCBA is stable only in a narrow window around `ne ~ 181-201`, the answer
moves by ~13 % across it, and refining past it diverges. (The 181 entry is
`g_band = 2` against the default 3, so that one comparison is confounded; the
161/271/361 outcomes are not.)

So "converge the baseline on a fine grid and compare" is not available. The
baseline has no fine-grid limit. **That is the problem the pole subtraction is
for**, and it sets the actual success criterion:

> does the sector give an answer on a coarse grid that is STABLE under
> refinement, where the baseline is not?

A comparison at a single `ne` cannot show that, and neither can agreement with
a number the baseline only produces inside its stability window.

### Why the `pgate` A/Bs cannot answer it either

`cluster/pgate` (daint) and `cnt33_L4_linear` (tortin) are both 4-cell CNT,
both `fft`, both `eta = 0`. What differs is the budget:

| | `pgate` staging runs | production recipe |
|---|---|---|
| `mixing_factor` | 0.02 - 0.1 | **0.2** |
| iterations | 80 | **up to 450**, converging at 200-313 |
| `ne` | 201 | 181 |

At `QX_MIX=0.02, QX_MAXIT=80` the pole-free baseline plateaus at
`rel Sigma ~ 2.5e-01`; under Anderson at `2.5e-02`. Both are transients of a
run stopped an order of magnitude early. So every `pgate` conclusion about
whether the sector changes the answer — including "cong differs from baseline
only in the fourth digit" — compares two unconverged transients and is not
evidence either way.

## 3. The second blocker: promotion yield, and a hard size limit

On `pgate`, measured (job 4399102 / 4399130):

    pole sector: iteration 1, 2 cluster(s), 2/144 pole(s) promoted  refused: eps_nep x142
      cluster c0+partner source varies by 2.02e+02 across its window (tol 1.00e-01)
      cluster c1+partner source varies by 5.71e+02 across its window (tol 1.00e-01)

Two poles out of 144 candidates, and both violate the sector's own
source-smoothness gate by 2000x and 5700x. `max_poles` is inert: 2, 8 and 24
give byte-identical output, because screening binds long before the cap.

### Measured, one knob at a time (job 4399332)

| arm | acceptance | trust region | promoted |
|---|---|---|---|
| legacy | scaled matrix residual | grid-tied | 2/144 |
| trust | scaled matrix residual | physical | 2/144 |
| locate | **frequency error** | grid-tied | **11/144** |
| both | frequency error | physical | 9/144 |

**The acceptance criterion is the whole effect; the trust region does not bind
on this bed.** `eps_nep` is a scaled MATRIX residual, `||M(z)r|| /
((|z|^2 + ||M||)||r||)`, and its denominator is `1e3-1e4 THz^2` for a phonon
operator, so testing it against 1e-10 asks a question about matrix norms
rather than about frequency. Gating instead on
`eps_z = |dz_est| / min(gamma, separation, h)` -- the estimated remaining
frequency error against the smallest scale the pole must be resolved against --
is worth 5.5x on its own.

An earlier version of this section attributed 2/144 to the Newton BUDGET: with
`trust_radius_cells = 0.25`, eight steps travel at most two cells, so a distant
seed could not arrive. That mechanism is real and is fixed (the radius is now
`trust_factor * min(nearest seed, nearest band edge)` in THz, and on the
sector's own synthetic bed the old radius loses 3 of 9 poles), but the `trust`
arm above shows it is **not** what was binding on the CNT bed. The seeds there
are already close enough. Newton ITERATIONS remain untested at scale -- the
earlier attempt OOM'd before finishing, which is what the chunking fixed.

**The remaining 133 refusals are genuine, not gate artefacts.** Their `eps_z`
distribution:

    min 5.0e-02   p25 4.5e-01   median 9.7e-01   p75 2.4e+00   max 2.9e+01

The median refused candidate is displaced by about a FULL limiting scale. These
are not poles sitting just outside a tolerance; they are not located. 38 of
them fall under `eps_z = 0.5`, so loosening `locate_tol` from 0.05 to 0.5 would
give roughly 49/144 -- at the price of admitting poles displaced by half their
own linewidth, which is a materially wrong residue. That is a physics judgement
and is left as a knob (`QX_POLE_LOCTOL`) with the trade-off measured rather
than chosen silently.

### It is not the seeds either -- the bed has no narrow isolated modes

The obvious next suspect was the seeding: the real part comes from eigenvalues
of the BARE frequency-independent part, with no `Re Sigma^R` and no contacts.
It is not that. Measured on the 133 refused candidates
(`h = 55/200 = 0.275 THz`):

| | median | |
|---|---|---|
| `gamma` | 0.203 THz | |
| `h / gamma` | **1.35** | the grid nearly resolves them |
| nearest-neighbour spacing | 0.199 THz | |
| `gamma / spacing` | **2.67** | 85 % overlap their neighbour |

Both ratios say the same thing, and neither is about the solver.

**The grid already carries these lines.** A `dw`-weighted sum of point samples
recovers 98.1-101.9 % of a Lorentzian's total weight at `h/gamma = 1.35`. The
exact cell average recovers 100.0 %. So on the median refused mode the sector
can buy about 2 %. Only the narrow tail of the distribution
(`gamma = 0.013`, `h/gamma = 20.7`) is in the regime where the point sample
ranges from 15 % to 663 % and the treatment is worth a factor of 44.

**And there is no isolated pole to find.** A typical candidate's linewidth is
2.7x its distance to its nearest neighbour. In that regime a simple pole is
not well defined, a bordered Newton for one cannot localise, and
`eps_z ~ 1` is the correct report rather than a failure.

So the low yield **on this bed, at this point in the iteration** is the right
answer. `population()` now prints both ratios every iteration so this is
visible before any conclusion is drawn from a yield.

**Scope of that measurement, and an open contradiction.** It was taken on
`pgate` at SCBA iteration 1, where `Sigma_scatt` is essentially zero and the
broadening is contact-dominated -- it is not the converged linewidth
distribution, and `pgate` is not the bed the ne ladder was run on. Stated as
"CNT at 300 K has no narrow modes" it over-reaches, and it collides with
Sec. 2: if nothing were narrow, refining the grid would converge, and it does
not.

The ladder is **non-monotone** -- `ne = 161` diverges, 181 and 201 converge,
271 fails to converge, 361 diverges. Failing at BOTH ends rules out both
monotone stories: unresolved narrow lines would improve steadily with
refinement, and crowding the `omega -> 0` acoustic singularity would worsen
steadily. Neither fits.

Kramers-Kronig truncation is not the discriminator either. `sse_aux_grid_dw_thz
= 0` on these beds, so the bubble runs on the primary grid, which stops at
55 THz while the 3-phonon bubble has support to about 110; the solver warns on
every rung. But the warning does not track the outcome -- the run with the
WORST truncation (13 % of peak weight at the grid top, `cnt-L3-eta0-mask`)
converged fastest of all, in 52 iterations. It is a real defect and it is not
what is firing.

**The mechanism for the ladder is unknown.** The measurement that would settle
it is the ladder itself, on `cnt33_L4_linear`, pole sector off, with
`population()` reporting the CONVERGED `gamma` distribution, `h/gamma` and
`gamma/spacing` at every rung -- measured on that bed at that `ne` rather than
transferred from another. If narrow modes appear at the failing rungs, the
premise above is wrong there; if they do not, the ladder pathology is
something else entirely and the sector cannot be its cure.

Two things this does expose, and neither should be changed silently:

* **The resolution criterion is miscalibrated.** `q_omega = gamma/(2h) < 1`
  promotes anything with `gamma < 2h`, i.e. `h/gamma > 0.5` -- deep into the
  regime where the grid carries 98 % of the weight. The measurement above says
  the sector starts mattering around `h/gamma ~ 3-5`. `samples_per_halfwidth`
  is the knob; the calibration is a physics choice.
* **Nothing refuses an overlapping candidate.** `cluster_factor` groups poles
  after acceptance, but no criterion refuses one whose linewidth exceeds its
  spacing, where the extraction is ill-posed. 85 % of this bed's candidates
  are in that state.

**Coverage moves with it, as predicted.** At 11 promoted poles the ring weight
sitting on pole-cell PAIRS rises from 4.2 % to 15.4 % at the worst bin
(`w = 52.25`). That is the mechanism the sector needs in order to matter at
all, appearing exactly where Sec. 2 says it should.

**Raising the budget promotes more poles and then runs out of memory.**
`QX_POLE_NEWTIT=40` on the same bed died with

    cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 290,488,467,456 bytes

`sector_terms` and `sector_cell_average` materialise
`xp.take(c_sr, cols, axis=2)`, shaped `(n_omega, Np, nnz)`, and two of those
per Keldysh component per call. At `Np = 2` that is invisible; at a few dozen
poles it is hundreds of gigabytes. **The congruence route as implemented
cannot run above a handful of poles**, and it has never been asked to.

---

## 4. What is left of `congruence_analytic`

Fixed (all three were in the assembly, none in the kernels):

1. the mixed sector reached `set_pole_self_energy`, which lands after
   `delta = Sigma^> - Sigma^<` is formed, so `Sigma^R` was missing the entire
   dispersive part of `SR + RS` while `Sigma^{<,>}` had it. Now routed through
   `set_pole_mixed`, before the Hilbert transform, as `rr_ss_sr` always was;
2. the low-frequency leg mask was not applied to the background leg;
3. `mixed_scale` was ignored.

Still diverges. Two of the readings first cited here have since been withdrawn
(see `pole_scba_routes.md` Sec. 4.2): a ring leg that is indefinite is expected
for an additive remainder, and coefficient variation is not a rejection
criterion under the principal-part split. The gate that binds is the total,
`positivity sigma_lesser worst = -4.199e-01 VIOLATION`, where base and cong
read `+0.000 ok`.

The withdrawn text, kept because the reasoning is the recurring failure mode:

    ring leg positivity lesser  worst=-4.088e-01 at w[127]  pole-off control=-7.971e-04
    pole analytic leg: eps_tail=1.650e-03  eps_c_rs=9.076e-01  ABOVE source_fit_tol (1.00e-01)

The leg the ring convolves is indefinite by 0.41 against a control of 7.97e-04
— the frozen remainder, which is a difference of PSD objects. And `c_rs`
varies by 91 % of its scale across the pole window, so freezing it (which the
flattening requires) is not justified. The same gate on `congruence` reads
exactly its control every iteration.

This is a construction problem, not a wiring problem. Treat the route as
experimental and do not spend device time on it until Sec. 2 and Sec. 3 move.

---

## 5. What has been measured and cleared

Do not re-litigate these without new evidence.

* **Finite support of the residue kernel** (analytic leg integrated over
  `(-inf, inf)` while the ring sees only the stored window): sub-percent.
  The finite-window kernel, verified against numerical quadrature to 1e-10,
  moves a same-half-plane pairing by 3.3e-03 at `+-100`.
* **Conjugate-pole residue symmetry**: holds structurally, `eps_AH` at 2e-16,
  on open and closed pole sets. It depends on the shared fit anchor and the
  real design matrix, so it is pinned by a test.
* **The `-1.000` ring-leg gate**: was the gate. `-i G^>(0)` is the
  near-singular acoustic bin the ring itself masks; it fixed both the
  numerator and the normalisation, so the reading saturated at exactly -1 on
  the pole-free baseline too.
* **Zero-filled-pattern positivity**: a hard band mask is indefinite, but a
  block-banded pattern leaves the gate's two-block window fully populated, so
  it is a genuine principal submatrix. Only within-window sparsity is exposed.
* **Registration error size**: bounded for the PROMOTED set only. Pole-cell
  pairs carry 6e-05 % of the ring's weight on `pgate`, 2.6 % at the single
  worst bin. That is a bound on the 2 poles the sector promoted, **not** on
  the physical narrow-line content: 142 candidates were refused. Read with
  Sec. 2, this is the one entry on this list that is not settled -- the
  registration mechanism (a line's weight placed at its cell centre, so its
  combination frequency moves by up to a full cell) is exactly the kind of
  thing that makes an answer jump discontinuously under refinement, and the
  measured bound cannot see the 142 modes that were never promoted.

---

## 6. Next steps, in order

### N1. Reproduce the baseline's grid ladder, then run the sector on the same one

`base` on `cnt33_L4_linear` at the production recipe (`mixing_factor = 0.2`,
`max_iterations = 450`, `eta = 0`, `sse_g_band = 3`) across
`ne = 161, 181, 201, 241, 271, 361`, to re-establish Sec. 2's ladder under
matched settings — the existing points come from runs that differ in
`g_band` and in whether a low-frequency mask was on.

Then the same ladder with `leg = "congruence"`. The deliverable is the two
ladders on one axis. Success is a FLAT pole curve where the baseline curve
moves 13 % and then diverges; it is not agreement at any single `ne`.

`cnt33_L3_linear` with the 2 THz mask converges in 52 iterations and is the
cheapest place to establish the harness before spending L4 time.

Note the interaction with N2/N3: at a fixed pole window, refining `ne` changes
which modes are grid-resolved and therefore how many are promoted. The ladder
must report the promotion yield at every rung, or a flat curve could be a flat
curve for the wrong reason.

### N2. Chunk the `(n_omega, Np, nnz)` temporaries

`sector_terms` and `sector_cell_average` must chunk over the pattern the way
`pf_self_energy` already does. Until then nothing above a handful of poles can
run at all, and every promotion-yield experiment is capped by memory rather
than by physics. Small, self-contained, and it blocks N3.

### N3. Attribute the 2/144 promotion yield

With N2 done, scan `newton_max_iterations` and `trust_radius_cells` — both
now exposed as `QX_POLE_NEWTIT` / `QX_POLE_TRUST`, and **neither touches
`newton_tol`**, so no worse pole is admitted; the solve is only allowed to
finish. If the yield rises sharply, "2 of 144" was a budget. If it does not,
it is a statement about the operator and the seeds.

### N4. Then, and only then, the method comparison

With a converged bed, a solve that can carry more than a handful of poles, and
a known promotion yield: is the pole ladder flat where the baseline's is not?
That is the headline A/B (existing task #6). N1 can start immediately -- the
baseline half of the ladder needs nothing -- but the pole half is only
meaningful once N2 and N3 land, because a sector that promotes 2 modes out of
144 cannot cure a grid pathology caused by the other 142.

### Deferred

* The `|P|^2` cell-pair product-integration correction (review Sec. 38).
  Measured at ~2 % at one bin on `pgate` -- but over 2 promoted poles, so that
  number is not a bound on the physical effect. Revisit once N3 says how many
  modes the sector should be carrying.
* `congruence_analytic` — Sec. 4.
* Coupled-q support (task #5).

---

## 7. Open decisions

Neither of these is a code question.

**Does `eps_nep ~ 1e-9` mean "not a pole"?** `newton_tol = 1e-10` is tight.
Loosening it admits less-well-located poles, and a mislocated pole is what the
whole divergence investigation was about, so this is a deliberate choice
rather than a knob to turn. N3 is designed to answer it without touching the
tolerance.

**What does a source varying by 200x across a promoted pole's own window
mean?** The analytic source model presumes smoothness there. If that fails on
a converged bed too, it says the smooth-source premise does not hold for this
physics — which would be a result about the method's applicability, not a
tolerance to relax.

---

## 8. Ledger

238.51 / 300 nh, all debug. Jobs this round: 4398979 (`panal3`, the A/B after
the assembly fixes), 4399102 (`preg`, pole-cell-pair weight), 4399130
(`pyield`, promotion yield — OOM'd on the raised-budget arms, which is Sec. 3's
finding).

## 9. A correction to keep

An earlier version of this analysis stated that no converged `retarded = fft`
run existed and that the CNT baseline did not converge. Both were wrong, and
both came from searching only the daint `cluster/` tree. The converged
Hilbert-corrected CNT runs are on tortin, in
`phonon/studies/out/anderson_test/cnt33_L*_linear`. The lesson is the same one
Sec. 5 keeps recording: a negative result from a search is only as good as the
search's coverage, and the coverage has to be stated.
