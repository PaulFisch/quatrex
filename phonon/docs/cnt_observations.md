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

Three families, and they are not interchangeable. All CNT (3,3), 12 atoms per
cell, 305/295 K.

| | daint `l16, l24, l32` | tortin `cnt33_L*_linear` | daint `pgate` |
|---|---|---|---|
| cells | 16, 24, 32 | 3 - 10 | 4 |
| `ne` | 161 (also 361) | 181 | 201 |
| `retarded_method` | **half** | **fft** | **fft** |
| `sse_g_band` | 3, or 1+taper | 2 | 3 (default) |
| `eta` | 0 | 0 or 1e-12 | 0 |
| `mixing_factor` | — | 0.2 | 0.02 - 0.1 |
| `max_iterations` | — | 450 | 50 (overridden to 80) |
| converges? | yes, all six | yes L3-L7, no L8/L10 | **no — budget 4-40x too small** |

Two things follow immediately.

**No single family is "the" CNT bed.** The longest converged runs (`l32`,
32 cells) do NOT carry the Kramers-Kronig half of `Sigma^R`; the family that
does (`cnt33_L*`) runs at `g_band = 2` and stops at 7 cells. Any statement
about "converged CNT" has to name which.

**Every pole-sector A/B so far has run on `pgate`**, at a mixing factor and
iteration budget on which the SCBA does not converge with the sector on or
off. Those comparisons are between two unconverged transients.

`eta = 0` and `eta = 1e-12` are numerically identical here: `cnt-L3-eta0` and
`cnt-L3-gband2` differ only in that setting and produce the same 209
iterations and the same `final_heat` to every printed digit.

---

## 2. Two length series, and `sse_g_band` is what separates them

There are two CNT (3,3) families on disk, both at 305/295 K, and they differ in
more than length. Do not read them as one series.

| | daint `cluster/l16, l24, l32` | tortin `cnt33_L*_linear` |
|---|---|---|
| cells | 16, 24, 32 | 3 - 10 |
| atoms/cell | 12 | 12 |
| `retarded_method` | **half** (no KK part) | **fft** |
| `ne` | 161 | 181 |
| `sse_g_band` | 3, or 1+taper | 2 |

### The `g_band` comparison — the dominant effect on conservation

Same device, same length, only the spatial bandwidth of the bubble's legs
changed. `final_heat` is per-slab; a conserving solution has it uniform.

| cells | `g_band = 3` mean | spread | `g_band = 1` + taper mean | spread |
|---|---|---|---|---|
| 16 | 8.688 | **2.4 %** | 2.878 | **39.9 %** |
| 24 | 9.156 | **5.1 %** | 2.024 | **66.1 %** |
| 32 | 10.696 | **11.8 %** | 1.545 | **82.8 %** |

All six converged (97, 96, 176 iterations for `g3`; 66, 70, 70 for `g1t`), so
this is not a convergence artefact — it is what the converged answers are.

`g_band = 1` with a Bartlett taper is catastrophically non-conserving, up to
83 % at L32, and its answer collapses with length (2.88 → 1.55) in a way the
`g3` answer does not. The taper is recorded as incorrect elsewhere; this is
the measurement.

**`g_band = 3` reaches 32 cells.** That is the correction to an earlier version
of this section, which reported the series breaking at L8. Those were
`g_band = 2` runs (`cnt-L*-gband2`), and the failure is a property of that
setting, not of the device or the method.

### But truncation still bites at length

Even at `g_band = 3` the conservation spread grows steadily with the device:
2.4 % → 5.1 % → 11.8 % from 16 to 32 cells. `sse_g_band` truncates the
self-energy at a fixed number of BLOCKS, so a longer device is not covered
better by the same band. Sec. 3 is the controlled version of that statement.

### The `g_band = 2` series (tortin, `fft`)

Kept for the record, because it is the only family with the Kramers-Kronig
half included. `ne = 181`, mixing 0.2, ballistic reference 77.67.

| cells | converged in | mean heat | % of ballistic | spread |
|---|---|---|---|---|
| 3 | 209 it | 37.24 | 48 % | 8.5 % |
| 4 | 311 it | 33.53 | 43 % | 8.2 % |
| 5 | 304 it | 30.61 | 39 % | 9.1 % |
| 6 | 239 it | 29.15 | 38 % | 8.3 % |
| 7 | 313 it | 26.37 | 34 % | 8.6 % |
| 8 | diverged after 63 it (best 1.9e-01, ends 4.0e+03) | — | — | — |
| 10 | diverged after 119 it (best 9.2e-02, ends 8.1e+03) | — | — | — |

Smooth and monotone where it converges, 8-9 % non-conserving throughout, and
it stops at 7 — at `g_band = 2`. No sign inversions are logged at L8/L10, so
that failure is not the anti-damping signature the pole-sector work chased.

**Caution on comparing the two families.** They differ in `g_band`, in `ne`,
and in whether `Sigma^R` carries its Kramers-Kronig half. The `g3` family's
mean heat also RISES with length (8.69 → 10.70), which is not what a
fixed-`dT` diffusive device should do; whether `final_heat` is a per-slab
absorbed power (which would grow with slab count) or a current has to be
settled before that is read as physics.

---

## 3. Si: the same device converges or not depending on how it is BLOCKED

The controlled version of the truncation statement, on Si. One physical device
— 8 atoms — partitioned two ways, everything else identical (`ne = 2801`,
`eta = 0`, `retarded = half`):

| run | partition | atoms/block | iterations | `rel Sigma` trajectory |
|---|---|---|---|---|
| `si4x1` | 4 blocks x 1 cell | 2 | 9 | 1.0 → **20.97, rising** |
| `si4x2` | 2 blocks x 2 cells | 4 | 12 | 1.0 → 0.317, falling |
| `si4x2b` | 2 blocks x 2 cells | 4 | 46 | 1.0 → **6.96e-03**, falling |

Nothing about the physics changed — only the block partition. With four small
blocks the SCBA diverges; with two large ones it converges by three orders.

`sse_g_band` truncates `Sigma` at a fixed number of BLOCKS, so the physical
range retained is `g_band x (cells per block)`. Halving the block size halves
that range at fixed `g_band`, and on this device that is the difference
between converging and not.

This is the concrete motivation for the spatial-decomposition work: the
quantity that matters is the retained physical range, and it is currently
controlled only indirectly, through a block count that is chosen for solver
reasons. It also explains the trend in Sec. 2 — conservation degrading from
2.4 % to 11.8 % as the CNT grows at fixed `g_band`.

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

* **`sse_g_band` is the dominant control on conservation.** At 32 cells,
  `g_band = 3` gives an 11.8 % spread and `g_band = 1` + taper gives 82.8 %,
  both converged. `g_band = 3` reaches 32 cells; `g_band = 2` stops at 7.
* **Spatial truncation is set by the retained physical range, not by
  `g_band` alone.** On Si, the same 8-atom device diverges partitioned into
  4 small blocks and converges by three orders into 2 large ones. This is the
  concrete case for the spatial-decomposition work.
* conservation still degrades with length at `g_band = 3` (2.4 % → 11.8 %
  from 16 to 32 cells), consistent with the same mechanism;
* a smooth, monotone, converged length series L3-L7 with the Kramers-Kronig
  half included, at `g_band = 2`, 8-9 % non-conserving;
* the 2 THz low-frequency mask removes the conservation error almost entirely
  (8.5 % → 0.6 %) while raising the answer 68 %;
* the pole-sector acceptance criterion was in the wrong units, worth
  2/144 -> 11/144.

Not established:

* any grid-converged CNT number — the `ne` ladder is non-monotone and moves
  the answer 13-47 % where it converges at all, and `l16` moves 2.2x between
  `ne = 161` and 361;
* the mechanism of that ladder;
* what `final_heat` rising with length in the `g3` family means — per-slab
  absorbed power would grow with slab count, a current should not;
* whether CNT at 300 K has narrow isolated modes at CONVERGENCE, which is what
  decides whether the pole sector can help;
* which of the two low-frequency treatments is the physics;
* a converged long-device run WITH the Kramers-Kronig half — no family
  currently has both.

The last item and the first two block any production CNT result,
independently of the pole sector.
