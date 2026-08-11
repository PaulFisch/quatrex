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

It is blocked on the fact that **every pole-sector A/B so far has been run on
`cluster/pgate` — a bed and an iteration budget on which the SCBA does not
converge, with or without the pole sector.** A converged, `eta = 0`,
Hilbert-corrected CNT recipe exists and has existed for a while; the pole
sector has never been run on it.

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
* `eps_tail` and `eps_c_rs` on the analytic route.
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

## 2. The blocker: the A/B bed is not the production bed

`cluster/pgate` (daint) and `cnt33_L4_linear` (tortin) are both 4-cell CNT,
both `fft`, both `eta = 0`. What differs is the iteration budget:

| | `pgate` staging runs | production recipe |
|---|---|---|
| `mixing_factor` | 0.02 - 0.1 | **0.2** |
| iterations | 80 | **up to 450**, converging at 200-313 |
| `ne` | 201 | 181 |

At `QX_MIX=0.02, QX_MAXIT=80` the pole-free baseline plateaus at
`rel Sigma ~ 2.5e-01` and never converges; under Anderson it plateaus at
`2.5e-02`. Both are transients of a run that was stopped an order of magnitude
too early, not a property of the solver.

Consequence: **every conclusion drawn from `pgate` about whether the pole
sector changes the answer is a statement about two unconverged transients.**
That includes "cong differs from baseline only in the fourth digit". It is not
evidence either way.

---

## 3. The second blocker: promotion yield, and a hard size limit

On `pgate`, measured (job 4399102 / 4399130):

    pole sector: iteration 1, 2 cluster(s), 2/144 pole(s) promoted  refused: eps_nep x142
      cluster c0+partner source varies by 2.02e+02 across its window (tol 1.00e-01)
      cluster c1+partner source varies by 5.71e+02 across its window (tol 1.00e-01)

Two poles out of 144 candidates, and both violate the sector's own
source-smoothness gate by 2000x and 5700x. `max_poles` is inert: 2, 8 and 24
give byte-identical output, because screening binds long before the cap.

An `eps_nep` refusal means "the bordered Newton did not reach
`newton_tol = 1e-10` within `newton_max_iterations = 8` steps". With
`trust_radius_cells = 0.25`, eight steps travel at most two cells from the
seed, so a seed further out cannot arrive however good the pole is. The
rejected residuals run from 4.8e-10 (one step short) to 2.8e-02 (nowhere
near), which is the signature of a budget, not of an absent pole.

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

Still diverges, and the gates say why at iteration 1, before anything blows up:

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
* **Registration error size**: real but bounded. Pole-cell pairs carry 6e-05 %
  of the ring's weight on `pgate`, 2.6 % at the single worst bin.

---

## 6. Next steps, in order

### N1. Move the A/B onto a bed that converges

Run `base` / `cong` on `cnt33_L4_linear` at the production recipe —
`mixing_factor = 0.2`, `max_iterations = 450`, `ne = 181`, `eta = 0`,
`sse_g_band = 3` — and compare CONVERGED answers, not iteration-6 transients.
This is the first result that would mean anything, and it needs no new code.

`cnt33_L3_linear` with the 2 THz low-frequency mask converges in 52
iterations and is the cheapest place to start.

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

With a converged bed, a working solve and a known pole set: does the sector
change `lead_current` and the heat profile, and in which direction relative to
the grid ladder. That is the headline A/B (existing task #6) and it is not
answerable before N1-N3.

### Deferred

* The `|P|^2` cell-pair product-integration correction (review Sec. 38). Worth
  having; measured at ~2 % at one bin on `pgate`, so not urgent until a bed
  exists where poles carry weight.
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
