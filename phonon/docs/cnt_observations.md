# CNT (3,3) at 300 K: what the runs actually show

2026-08-11. A consolidation of every CNT run currently on disk — what
converges, what the answer is, what moves it, and what is not understood.
Companion to `pole_sector_state_and_next_steps.md` (which asks what the pole
sector is blocked on) and `pole_scba_routes.md` (which documents the sector's
three routes). This one is about the DEVICE, not the method.

Sources: `cluster/cnt-*/run.log` and `cluster/newton-*/run.log` (tortin,
`phonon/studies/out/anderson_test/cnt33_L*_linear`), plus `cluster/pgate` and
`cluster/p*` (daint).

---

## 1. The beds

Two families, and they are not interchangeable.

| | tortin `cnt33_L*_linear` | daint `cluster/pgate` |
|---|---|---|
| cells | 3, 4, 5, 6, 7, 8, 10 | 4 |
| `ne` | 181 | 201 |
| `energy_window_max` | 55.0 THz | 55.0 THz |
| `retarded_method` | fft | fft |
| `eta` | 0 or 1e-12 | 0 |
| `mixing_factor` | **0.2** | 0.02 - 0.1 |
| `max_iterations` | **450** | 50 (overridden to 80) |
| converges? | yes, L3-L7 | **no — the budget is 4-40x too small** |

Every pole-sector A/B so far has run on `pgate`, at a mixing factor and
iteration budget on which the SCBA does not converge with the sector on OR
off. Those comparisons are between two unconverged transients. The converged
CNT recipe is the tortin one.

`eta = 0` and `eta = 1e-12` are numerically identical here: `cnt-L3-eta0` and
`cnt-L3-gband2` differ only in that setting and produce the same 209
iterations and the same `final_heat` to every printed digit.

---

## 2. The length series — the physics result

`g_band = 2`, `ne = 181`, `eta ~ 0`, `retarded = fft`, linear mixing 0.2.
`final_heat` is the per-slab heat current; a conserving solution has it
uniform along the device.

| cells | converged in | mean heat | % of ballistic | edge-to-interior spread |
|---|---|---|---|---|
| ballistic | — | 77.67 | 100 % | 0 |
| 3 | 209 it | 37.24 | 48 % | 8.5 % |
| 4 | 311 it | 33.53 | 43 % | 8.2 % |
| 5 | 304 it | 30.61 | 39 % | 9.1 % |
| 6 | 239 it | 29.15 | 38 % | 8.3 % |
| 7 | 313 it | 26.37 | 34 % | 8.6 % |
| 8 | **diverged** | — | — | — |
| 10 | **diverged** | — | — | — |

Per-slab, for reference: L4 is `35.22, 32.33, 32.50, 32.40, 35.22` and L7 is
`28.10, 25.78, 25.72, 25.71, 25.68, 26.01, 25.89, 28.05`.

Two things to read off it.

**The series is smooth and monotone.** The current falls from 48 % to 34 % of
ballistic between 3 and 7 cells. That is the expected approach to diffusive
transport and it is the one clean physics result in this whole set.

**The heat profile is NOT uniform, by a consistent 8-9 %.** The first and last
entries sit above the interior at every length. A conserving solution cannot
do that, so roughly 8-9 % of the answer is a conservation error, and it does
not shrink with length. Sec. 4 shows where it lives.

---

## 3. Where it breaks: L8 and beyond

`L8` and `L10` do not fail to start — they nearly converge and then blow up:

| | iterations | best `rel Sigma` | final `rel Sigma` |
|---|---|---|---|
| L8 | 63 | 1.9e-01 | **4.0e+03** |
| L10 | 119 | 9.2e-02 | **8.1e+03** |

No sign inversions are logged, so this is not the anti-damping signature the
pole-sector work chased; it is a late-stage instability of the bubble-only
model, and it is the reason the length series stops at 7.

---

## 4. The low-frequency mask changes the answer by 68 %

`cnt-L3-eta0` and `cnt-L3-eta0-mask` are the same bed. The second masks the
scattering self-energy below 2 THz.

| | converged in | mean heat | spread | % of ballistic |
|---|---|---|---|---|
| no mask | 209 it | 37.24 | 8.5 % | 48 % |
| 2 THz mask | **52 it** | 62.65 | **0.6 %** | 81 % |

Three effects at once, and they point the same way:

* it converges **4x faster**;
* the heat profile becomes **essentially uniform** (0.6 % against 8.5 %), i.e.
  nearly conserving;
* and the answer rises **68 %**.

So the ~9 % conservation error of Sec. 2 lives in the sub-2-THz acoustic
region, and so does most of the convergence difficulty. Masking it removes
both — by removing the scattering there, which leaves transport below 2 THz
ballistic and is why the current jumps. **This is a choice about the physics
being modelled, not a numerical setting**, and the two published numbers for
the same device differ by 68 % depending on it. Nothing downstream should
quote a CNT heat current without stating which.

`cnt-L3-eta0-scat` (a third variant) converged in 304 iterations to 38.06,
34.56, 39.64, 37.68 — a 14 % spread, worse-conserving than either.

---

## 5. The `ne` ladder — non-monotone and unexplained

L4, same bed, varying only the frequency-grid size:

| `ne` | outcome | `final_heat` |
|---|---|---|
| 161 | **diverged** | ~ -2.4e+22 |
| 181 | converged, 311 it | 35.22, 32.33, 32.50, 32.40, 35.22 |
| 201 | converged, 249 it | 39.84, 36.68, 37.13, 36.61, 39.45 |
| 271 | did not converge | 51.64, 47.07, 47.38, 46.51, 50.29 |
| 361 | **diverged** | ~ +3.9e+19 |

(The 181 row is `g_band = 2` and the others use the default 3, so that one
comparison is confounded. The 161/271/361 outcomes are not.)

**It fails at both ends and works in the middle**, which rules out both
monotone explanations: unresolved narrow lines would improve steadily under
refinement, and crowding the `omega -> 0` acoustic singularity would worsen
steadily. And where it does converge, the answer moves 13 % from `ne` 181 to
201 and 47 % by 271 — so there is no grid-converged number here.

Two candidates checked and REJECTED:

* **Kramers-Kronig truncation.** `sse_aux_grid_dw_thz = 0` on these beds, so
  the bubble runs on the primary grid, which stops at 55 THz while the
  3-phonon bubble has support to about 110; the solver warns on nearly every
  run. But the warning does not track the outcome — the run with the WORST
  truncation (13 % of peak weight at the grid top, `cnt-L3-eta0-mask`)
  converged fastest of all, in 52 iterations. A real defect, not this one.
* **Unresolved narrow resonances.** See Sec. 6.

The mechanism is not known. The measurement that would settle it is the ladder
re-run on `cnt33_L4_linear` at matched `g_band`, pole sector off, with
`PoleSectorState.population()` reporting the CONVERGED linewidth distribution
at every rung.

---

## 6. What the pole sector sees on CNT

On `pgate` (4 cells, `ne = 201`, so `h = 0.275 THz`), at SCBA iteration 1:

| | legacy gate | frequency gate |
|---|---|---|
| promoted | 2/144 | **11/144** |

The acceptance criterion was the whole difference — `eps_nep` is a scaled
MATRIX residual whose denominator is `1e3-1e4 THz^2`, so testing it against
1e-10 is not a statement about frequency. The physical trust region, a
separate real fix, does not bind on this bed.

The 133 still refused are genuinely not located (median `eps_z` 0.98, i.e.
displaced by a full limiting scale), and the population they come from looks
like this:

| | median |
|---|---|
| `gamma` | 0.203 THz |
| `h / gamma` | 1.35 |
| nearest-neighbour spacing | 0.199 THz |
| `gamma / spacing` | 2.67 (85 % overlapping) |

At `h/gamma = 1.35` a `dw`-weighted sum of point samples already recovers
98-102 % of a Lorentzian's total weight, so the sector's exact cell average
has about 2 % to gain; and above `gamma/spacing ~ 0.5` no isolated simple pole
exists for a bordered Newton to find.

**Scope, and an open contradiction.** That was measured at SCBA iteration 1,
where `Sigma_scatt` is essentially zero and the broadening is
contact-dominated — it is not the converged distribution, and `pgate` is not
the bed the ladder of Sec. 5 was run on. Read as "CNT has no narrow modes" it
also contradicts Sec. 5: if nothing were narrow, refinement would converge.
Both cannot be right, and which one gives is the open question.

---

## 7. Summary of what is and is not established

Established:

* a smooth, monotone, converged length series L3-L7 against a ballistic
  reference, at `eta = 0` with the Kramers-Kronig half included;
* a ~9 % conservation error at every length, which the 2 THz low-frequency
  mask removes entirely while raising the answer 68 %;
* a late-stage instability at L8 and beyond;
* the pole-sector acceptance criterion was in the wrong units, worth
  2/144 -> 11/144.

Not established:

* any grid-converged CNT number — the `ne` ladder is non-monotone and moves
  the answer 13-47 % where it converges at all;
* the mechanism of that ladder;
* whether CNT at 300 K has a population of narrow isolated modes at
  CONVERGENCE, which is what decides whether the pole sector can help;
* which of the two low-frequency treatments is the physics.

The first two Not-Established items block any production CNT result,
independently of the pole sector.
