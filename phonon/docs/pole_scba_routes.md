# The three pole-sector routes: formulas as implemented, and what they do

State as of 2026-08-11. Selected by
`phonon.pole_sector.leg`. All three are bit-identical to the pole-free
baseline for an empty pole set, so no baseline result depends on the choice.

Sign convention throughout: `-i G^{<,>} >= 0` and `-i Sigma^{<,>} >= 0`, i.e.
`Sigma^dagger = -Sigma` (`bubble_positivity.md` Sec. 0).

---

## 0. What every route shares

Built in `PhononSolver._build_pole_keldysh`, per cluster of `state.legs` --
the pole set CLOSED under `z -> -conj(z)`. The closure is not decoration; see
Sec. 4.3.

The projected Keldysh source, on the stored pattern plus the dense contact
corners:

    S^{<,>}(w_k) = V^dagger Sigma_tot^{<,>}(w_k) V ,   (n_omega, Np, Np)

`Sigma_tot` is the scattering self-energy that entered the Dyson solve PLUS
both lead self-energies. Omitting the contacts is not a small error: they are
what drives the device, and without them `G^R Sigma G^A` is not `G^<` at all.

`source_variation` measures how far `S` strays from its per-pair value across
the pole window; `source_fit_tol` gates on it. Carrying a source analytically
presumes it is smooth there, and this is the measured statement of that.

Each route then produces one object, `g_pp`, which is REMOVED from the ring's
legs (`set_pole_channel`), leaving the ring to convolve `G - g_pp`. What the
three routes disagree about is what `g_pp` is and what, if anything, is added
back beside the ring.

---

## 1. `leg = "keldysh"` -- superseded, kept only to reproduce old runs

Splits the KELDYSH function directly. With `S_a = S(z_a)` shared between a
pair's two residues (`source_at_poles`),

    g_pp(w) = sum_{ab} u_a [ S_ab / (z_a - conj(z_b)) ]
                       [ 1/(w - z_a) - 1/(w - conj(z_b)) ] u_b^dagger

(`pole_keldysh_pf_sparse`), and the ring keeps

    G_reg^{<,>} = G^{<,>}(w_k) - g_pp(w_k) ,   frozen across cell k.

Sectors `SS` (`ss_self_energy_sparse`) and `SR + RS`
(`mixed_self_energy_blocked`) are added back analytically.

### Why it fails

`G_reg` is a DIFFERENCE of PSD objects, hence generically indefinite, and it
is not smooth: writing the exact relation with a retarded split
`G^R = P^R + B^R`,

    G^< = P^R S P^A + P^R S B^A + B^R S P^A + B^R S B^A ,

two of the three terms outside `P^R S P^A` still carry `P^R(w)`. Freezing
them per cell is illegitimate.

Measured (`test_pole_subcell.py`), scanning a whole cell with only the residue
corrupted:

| residue error | worst `lambda_min` of `P + R_k` | congruence | true `G^<` |
|---|---|---|---|
| exact | +2.15e-06 | +2.05e-06 | +2.02e-06 |
| +20 % | **-1.000** | +1.45e-06 | +2.02e-06 |
| x2 | **-1.000** | +5.21e-07 | +2.02e-06 |
| wrong phase | **-1.000** | +1.62e-06 | +2.02e-06 |
| half (under-subtracted) | +7.95e-06 | +7.19e-06 | +2.02e-06 |

A **twenty percent** residue error inverts the sign. Over-subtraction does it;
under-subtraction is harmless. On the device the residue is a Newton solution
of a truncated cluster fitted against an approximate source, so 20 % is not a
comfortable margin, and nothing was checking it.

---

## 2. `leg = "congruence"` -- the default, and the one that runs

Splits the RETARDED function and lets the Keldysh components follow:

    G~^R(w) = G^R(w_k) + U [ D^R(w) - D^R(w_k) ] V^dagger ,
              D^R_ab(w) = delta_ab / (w - z_a)

    G~^{<,>}(w) = G~^R(w) Sigma_k^{<,>} G~^A(w)

with `G^R(w_k)` and `Sigma_k` the stored grid samples, frozen across the cell.

Two properties follow immediately, and they are the reason for the route.

`-i G~^{<,>} >= 0` at EVERY `w`, because it is a congruence
`G~^R (-i Sigma) G~^{R dagger}` of a PSD matrix. It demands no accuracy of the
pole model: a residue wrong by a factor of two gives a worse approximation but
never a wrong sign.

The correction VANISHES at the cell centre, so `G~^{<,>}(w_k)` is the
untouched ring `G^R_k Sigma_k G^A_k` and an empty pole set reproduces the grid
solver bit-for-bit.

### The four sectors, as an identity

Regrouping with the frozen retarded background
`B^R_k = G^R(w_k) - U D^R(w_k) V^dagger`:

    RR = B^R_k Sigma B^A_k                              cell-const x cell-const
    SR = U D^R(w) [ V^dagger Sigma B^A_k ]              pole x cell-const
    RS = [ B^R_k Sigma V ] D^A(w) U^dagger              cell-const x pole
    SS = U D^R(w) [ V^dagger Sigma V ] D^A(w) U^dagger  pole x pole

and `RR + SR + RS + SS = G~^{<,>}` exactly (1e-14 on the bed,
`test_four_sectors_are_the_congruence`). `RR` is a congruence of a PSD source,
so the regular leg is PSD where the superseded one was indefinite.

`B^R_k` is never formed. It enters only through

    B^R_k Sigma V = G^R_k Sigma V - U D^R(w_k) [ V^dagger Sigma V ] ,

one sparse apply against `Np` dense columns (`apply_sparse`), and the `SR`
bracket follows from the `RS` one because `Sigma^dagger = -Sigma`:

    c_ss = V^dagger Sigma V                (Np, Np)
    c_rs = G^R_k Sigma V - U D(w_k) c_ss   (n_dof, Np)
    c_sr = -c_rs^dagger                    (Np, n_dof)

### What actually reaches the bubble

NOT the point sample. The ring's `dw`-weighted sum is a midpoint rule, so what
it wants is the CELL AVERAGE, and for a line narrower than a cell the point
sample is wrong by order one. So

    g_pp = [SR + RS + SS](w_k)  -  < SR + RS + SS >_k

leaving the ring convolving `< G~^{<,>} >_k`. No analytic sector is added
beside it: `sectors` is inert on this route and says so.

The cell weights are analytic,

    < D_a >_k     = [ Log(w_k + h/2 - z_a) - Log(w_k - h/2 - z_a) ] / h
    < D_a Db_b >_k = ( < D_a >_k - conj(< D_b >_k) ) / ( z_a - conj(z_b) )

the second by partial fractions. The poles are strictly off the real axis, so
the difference of Logs has no branch ambiguity -- the `2 pi i` cancels -- and
`z_a - conj(z_b)` has imaginary part `-(gamma_a + gamma_b) < 0`, so it never
vanishes. Verified against Gauss-Legendre to 1e-9; `h` is per-bin, so a
non-uniform grid is handled.

| leg | rel. error vs the exact cell average, `h = 20 gamma` |
|---|---|
| congruence reconstruction | 4.1e-03 |
| raw grid sample `G^<(w_k)` | 8.2e-01 |

The correction falls off like `h^2` under refinement
(`test_correction_vanishes_as_the_grid_resolves_the_line`). It is a
DISCRETISATION fix, not added physics: a grid that resolves the line must
recover the ring untouched, or the two would double-count.

### What it does not do

The pole is resolved in the leg WEIGHT, not inside the convolution. The output
frequency resolution is still the grid's, and that is not a small residual
error -- it is order one, controlled by a number nothing used to measure.

An exact cell average still puts ALL of a line's weight at the cell CENTRE. So
the combination frequency `Re(z_a + z_b)`, which is where the three-phonon
structure lands, is displaced by up to a full cell and the ring splits the peak
between two bins. Measured on the scalar Lorentzian bed, ring against the exact
cell-averaged convolution at `w = 2 w0`
(`test_cell_averaged_legs_do_not_fix_the_bubble_registration`):

| pole offset inside its cell | `h = 20 gamma` | `h = 200 gamma` |
|---|---|---|
| 0.00 (on a grid point) | 1.0043 | 1.0000 |
| 0.25 cell | **1.7941** | 1.9754 |
| 0.50 cell (on a boundary) | **0.5363** | 0.5037 |

The control parameter is the SUB-CELL POSITION, not `h/gamma`, and refining the
grid does not help -- the quarter-cell case gets worse with `h/gamma`, tending
to a factor 2. Pole placement is set by the physics, so this route's accuracy
is an accident of registration. `pole registration: worst sub-cell offset` is
the runtime number that says which column a given bed is in.

The error is localised. With only ONE leg in a pole cell the displacement costs
`O((delta/Gamma_other)^2)` -- 2 % at the worst placement against a resolved
partner -- against 46 % when both are displaced. Two orders apart
(`test_registration_error_is_dominated_by_pole_cell_PAIRS`), so the whole
effect lives on cell PAIRS `(k, m-k)` with both ends in the pole set: `|P|^2`
terms, not `|P| * N`. That is what makes an exact-in-those-cells correction
affordable, and it is the fallback if the analytic route does not hold up.

---

## 3. `leg = "congruence_analytic"` -- three assembly defects fixed, device re-check pending

Same coefficients, then frozen at the poles and flattened so the sectors can
be convolved in closed form.

### 3.1 Freezing (`coefficients_at_poles`)

The families below are only frequency independent if the coefficients are.

`c_ss` goes through `source_at_poles`: a local polynomial fit about each
pair's centre, evaluated at `z_a` and `conj(z_b)` and SHARED between the two
residues. The sharing is what keeps `c_a + c_b = 0` and the pole-pole leg
decaying like `1/w^2`.

`c_sr` and `c_rs` have no pair structure -- `c_sr[a,:]` belongs to `z_a`
alone, `c_rs[:,b]` to `conj(z_b)` alone -- so each is fitted at its own pole
by the same `delta_local_fit`. Their asymptotics are governed by a different
rule; see Sec. 4.3.

### 3.2 Flattening (`partial_fraction_legs`)

Partial-fractioning `SS`'s double pole and collecting every term that shares a
pole gives `2 Np` simple poles with RANK-ONE residues:

    G~^{<,>}(w) - B^R Sigma B^A = sum_p  p_p q_p^T / (w - zeta_p)

    zeta_p = z_a        p_p = u_a
                        q_p = c^SR_{a,:} + sum_b c^SS_ab conj(u_b)/(z_a - conj(z_b))

    zeta_p = conj(z_b)  p_p = c^RS_{:,b} - sum_a c^SS_ab u_a /(z_a - conj(z_b))
                        q_p = conj(u_b)

This is what makes the analytic convolution affordable. A leg term with an
OPEN (non-modal) index would force the cubic vertex to be re-contracted at
every frequency -- which is exactly why the original design made `G_PP` modal
on both sides. Here both families are FIXED vectors, so the vertex is
projected onto them once per iteration.

### 3.3 The sectors

The leg subtracted from the ring is `pf_leg_sample` of the SAME object the
sectors restore. `G_reg = G - G_S` is exact for any `G_S`, but only if both
sides are literally the same function.

`SS` (`pf_self_energy`):

    Sigma(w)_{mu nu} = P sum_{pq} Vbar^L_{mu,pq} Vbar^R_{nu,qp} J(zeta_p, zeta_q; w)

    Vbar^L = modal_vertex_blocks(Phi, p_row),  Vbar^R = modal_vertex_blocks(Phi, q_col)

    J(p,q;w) = int dw'/2pi  1/(w'-p) 1/(w-w'-q)
             = -i/(w - p - q)   both poles in the lower half plane
             = +i/(w - p - q)   both in the upper
             =  0               mixed

-- the same algebra as the pole-pole sector over `2 Np` poles with unit
coefficients instead of `Np` poles carrying a source matrix, because the
coefficients now live in the vertex projections. Cost `(2 Np)^4`; `vl[rows]`
is `(nnz, 2Np, 2Np)`, so the contraction is chunked over the pattern.
`max_poles` is the control.

`SR + RS` (`pf_mixed_self_energy`): the same block triple product as
`mixed_self_energy_blocked`, with the pole leg's row and column modes no
longer independent indices `(alpha, delta)` over one family but a single index
`p` over TWO. The `(Np, Np)` double loop collapses to `2 Np`, and the vertex
is projected onto `p_row` on the left and `q_col` on the right rather than
onto `u` and `conj(u)`. `q_col` already carries whatever conjugation each pole
needs, so both projections are unconjugated.

The two halves take different routes into the bubble, and that is the point.

`SS` carries its own closed-form causal partner -- the two-retarded pairing,
whose combined pole `zeta_p + zeta_q` is again in the lower half plane -- so it
goes through `set_pole_self_energy` and never touches the discrete Hilbert
transform. That IS the causal continuation of `Delta_SS = Sigma^>_SS -
Sigma^<_SS` (`acc_r = rr_g - rr_l`), not a literal `G^R * G^R`.

`SR + RS` has NO closed form: one leg is the numerical background. It therefore
goes through `set_pole_mixed`, which adds it to the raw bubble output BEFORE
`delta = Sigma^> - Sigma^<` is formed, so the existing Kramers-Kronig transform
covers it -- exactly as the `rr_ss_sr` route has always done. See Sec. 4.2:
getting this wrong was one of the three reasons the route diverged.

---

## 4. Observed behaviour

Device: CNT bed, `QX_NE=201`, `mix=0.05`, 6 iterations, `eta = 0`.

Metrics, in the order they should be read:

* `rel Sigma^R residual` -- the actual convergence criterion.
* `lead balance = |J_L - J_R| / (|J_L| + |J_R|)/2` -- **2.0000 exactly means
  sign inversion**, emitting into BOTH leads. ~1e-5 is healthy.
* `P_in` and its agreement with `P_out`. The bubble energy balance RESIDUAL is
  a ratio and cannot see a blow-up; it read 1.9e-14 once while `P_in` grew
  from 2.6e+05 to 4.4e+32.

### 4.1 `keldysh` vs `congruence` vs baseline (job 4398590)

| iter | base | congruence | keldysh |
|---|---|---|---|
| 0 | 6.41e-15 | 6.41e-15 | 6.41e-15 |
| 1 | 7.84e-06 | 7.84e-06 | 7.84e-06 |
| 2 | 8.53e-06 | 8.07e-06 | **2.0000** |
| 3 | 1.44e-05 | 1.45e-05 | **2.0000** |
| 4 | 2.41e-05 | 2.47e-05 | **2.0000** |
| 5 | 3.56e-05 | 3.65e-05 | **2.0000** |

`congruence` tracks the pole-free baseline throughout. `P_in` stays O(1e4) and
matches `P_out` to 1e-8 every iteration. `keldysh`'s `P_in` flips sign at
iteration 1 (+1.81e+04 against `P_out = -1.44e+04`, resid 1.000) and never
recovers.

The subcell metric tells the same story from the other end: the congruence
reconstruction stays at -1.8e-04 and IMPROVES across iterations (-1.8e-04 ->
-1.5e-04) where the superseded one collapses (-1.2e-02 -> -0.96).

**Caveat.** The congruence route differs from the baseline only in the fourth
digit -- `rel Sigma` 0.56983 vs 0.56966, `P_in` by ~0.1 %. Stability is
demonstrated; physical significance is NOT. And `rel Sigma` is not converging
on any route (1.0 -> 0.57 -> 0.43 -> 0.64 -> 0.84 -> 0.96), baseline included,
so this bed at `mix = 0.05` over 6 iterations gives no converged answer.

### 4.2 `congruence_analytic` (job 4398805) -- DIVERGED, and why

| iter | base | congruence | analytic |
|---|---|---|---|
| 1 | 7.84e-06 | 7.84e-06 | 7.84e-06 |
| 2 | 8.53e-06 | 8.07e-06 | **2.0000**, `rel Sigma` 3.32 |
| 3 | 1.44e-05 | 1.45e-05 | **2.0000**, `rel Sigma` 3.13 |
| 5 | 3.56e-05 | 3.65e-05 | **2.0000**, `rel Sigma` 7.95 |

`P_in` flips sign at iteration 1: +1.578e+04 against `P_out` = +1.463e+03.

The kernels were verified in ISOLATION, and that was the right conclusion --
the defects were all in the ASSEMBLY (`_pole_analytic_sectors`), and all three
are now fixed and pinned by tests.

**(a) The mixed sector never reached the Hilbert transform.** It was folded
into the `SS` accumulator and handed to `set_pole_self_energy`, which joins the
output AFTER `delta = Sigma^> - Sigma^<` is formed, while the injected retarded
partner `acc_r` is the two-retarded pairing of the POLE-POLE convolution only.
So `Sigma^R` was missing the entire dispersive part of `SR + RS` while
`Sigma^{<,>}` had it in full. That is a fluctuation-dissipation break by
construction, and `lead balance = 2.0000` is what one looks like.
(`test_analytic_mixed_sector_goes_through_the_hilbert_hook`.)

**(b) The low-frequency leg mask was not applied** to the background leg. The
`rr_ss_sr` route masks it exactly as the ring masks its own legs, with a
recorded measurement: without it `Sigma^>` is non-PSD by 0.15 at mid-band and
the violation is strictly LINEAR in the injected mixed term. The device log
agrees -- analytic route only, base and cong both read `+0.000e+00 ok`:

    positivity sigma_lesser   worst=-4.202e-01 at w[1]    VIOLATION
    positivity sigma_greater  worst=-1.000e+00 at w[127]  VIOLATION

(`test_analytic_mixed_sector_masks_the_background_leg`.)

**(c) `mixed_scale` was ignored**, so the one knob that separates "too large"
from "wrongly signed" was inert on this route.

Two things that were suspected and are NOT the cause:

* **Finite support.** `pf_self_energy` integrates the analytic leg over
  `(-inf, inf)` by residues while the mixed sectors and the ring see only the
  stored window, so the four sectors do not act on one common function. The
  finite-window kernel (`pair_convolution(window=...)`, verified against
  numerical quadrature to 1e-10 and against its own residue limit) puts a
  number on it: truncating at `+-100` moves a same-half-plane pairing by
  3.3e-03 relative, and gives the opposite-half-plane pairing -- which the
  residue form sets to exactly zero -- a magnitude 3.3e-03 of it. Sub-percent
  at a realistic window, not order one.
* **Conjugate-pole residue symmetry.** `R_{conj z} = -R_z^dagger` holds
  structurally: it requires `c_ss^dagger = -c_ss` and `c_sr = -c_rs^dagger`,
  and both survive the freeze. `source_at_poles` shares
  `0.5*(P(z_a) + P(conj z_b))` over a pair with a common anchor, and the fit's
  design matrix is real, so `conj(S[b,a]) = -S[a,b]` exactly;
  `coefficients_at_poles` fits `c_sr` at `z_a` and `c_rs` at `conj(z_a)` through
  that same design matrix, so `c_sr(z_a) = -c_rs(conj z_a)^dagger` exactly.
  Measured on the bed, open and closed pole sets alike: `eps_AH,res` and
  `eps_AH(w)` both at 2e-16
  (`test_flattened_residues_keep_the_conjugate_pole_antisymmetry`,
  `test_flattened_leg_is_anti_hermitian_on_the_real_axis`). It depends on the
  shared anchor and the real design matrix, so it is measured rather than
  argued -- changing either breaks it silently.

**What remains, and it is structural.** `reg = G(w_k) - G_S(w_k)` is frozen per
cell while `G_S(w)` varies across it. That is the same additive
signed-pole-plus-frozen-remainder construction that made `keldysh` diverge
(Sec. 1), reintroduced. The congruence route avoids it by adding no sectors at
all and correcting the leg by a cell average instead. Fixing (a)-(c) removed
three real defects; it did not remove this. Sec. 4.5 is the measurement.

### 4.5 After the fixes (job 4398979) -- still diverges, and now we know why

Same bed. `gate0` is the analytic route with an EMPTY pole window.

| | base | gate0 | cong | anal |
|---|---|---|---|---|
| `rel Sigma`, it 1 | 5.6983e-01 | 5.6983e-01 | 5.6966e-01 | 1.0148e+00 |
| `lead balance`, it 2 | 8.53e-06 | 8.53e-06 | 8.07e-06 | **2.0000** |
| `positivity sigma_*` | ok | ok | ok | **-4.20e-01 VIOLATION** |
| `P_in` vs `P_out`, it 1 | -1.353e+04 / match | match | -1.355e+04 / match | +1.760e+04 / -1.444e+04 |

`gate0` reproduces `base` digit for digit at every iteration, so the analytic
route is bit-identical to the grid solver on an empty pole set -- the
correctness precondition, now checked on this route too.

The two new gates fire, at iteration 1, before anything has diverged:

    ring leg positivity lesser   worst=-4.088e-01 at w[127]   pole-off control=-7.971e-04
    ring leg positivity greater  worst=-4.095e-01 at w[127]   pole-off control=-3.001e-03
    pole analytic leg: eps_tail=1.650e-03  eps_c_rs=9.076e-01  ABOVE source_fit_tol (1.00e-01)

**The leg the ring convolves is indefinite by 0.41**, against a pole-off
control of 7.97e-04 -- a factor 500. That is review Sec. 20 measured rather
than argued: the analytic route hands the ring the frozen remainder, and it is
a difference of PSD objects. The SAME gate on `cong` reads exactly its control
at every iteration (`[== control: gate is blind]`), i.e. the cell-average
correction does the ring no damage at all. The two routes are distinguished by
one number, which is what the gate was rebuilt for.

**`eps_c_rs = 0.908` against a tolerance of 0.1.** The mixed coefficient varies
by 91 % of its own scale across the pole window, so freezing it -- which the
partial-fraction flattening requires -- is not justified on this bed. That is
review Sec. 28, and it is a first-order reason the analytic leg is a bad
approximation independently of the positivity failure. `eps_tail = 1.7e-03`, so
the closure is doing its job and the tail is not the problem.

`congruence_analytic` is therefore not viable as constructed. The next route is
Sec. 6's `|P|^2` cell-pair correction, which forms no frozen remainder and
freezes no coefficient.

### 4.6 The registration number on the real bed

`cong`, every iteration: **0.245 to 0.265 cells**. That is the 1.79 column of
Sec. 2's table -- close to the worst placement, not the benign one. So the
production route's bubble misplaces the pole-pair contribution by about half a
cell and gets its peak wrong by order 80 %, on this bed, today. (The analytic
route's own poles drift to 0.41-0.46 cells once it starts diverging, which is
a symptom rather than a cause.)

### 4.7 ... and how little it can move (job 4399102)

The registration error is order one ONLY on ring cell pairs `(k, m-k)` with
BOTH ends in a pole cell (Sec. 2), so the error in `Sigma` is bounded by the
fraction of the ring's weight sitting on those pairs. Measured:

    pole-cell PAIRS carry 0.00381% of the ring's weight, up to 2.59% at w=41.80

So the 0.265-cell offset bounds an error in `Sigma` of about **2 % at one
frequency** and about **0.003 % integrated** -- times the ~0.8 factor. Real,
but not what is holding this bed back. (Scalar-norm proxy for the vertex
contraction, hence an upper bound: small is conclusive.)

The reason it is that small is the reason everything else on this bed is
small, and it is not a property of the quadrature:

    pole sector: iteration 1, 2 cluster(s), 2/144 pole(s) promoted  refused: eps_nep x142, weight x3
      pole sector: cluster c0+partner source varies by 2.02e+02 across its window (tol 1.00e-01)
      pole sector: cluster c1+partner source varies by 5.71e+02 across its window (tol 1.00e-01)

**142 of 144 candidates are refused on `eps_nep` alone**, against
`newton_tol = 1e-10`; the rejected residuals run from 4.8e-10 to 2.8e-02. And
the two that survive fail the source-smoothness gate by 2000x and 5700x, so
even those are carried on a source model the solver itself says is not
justified. `max_poles` is irrelevant here -- 2, 8 and 24 give byte-identical
output, because screening binds long before the cap does.

That is the honest account of why `cong` differs from the baseline only in the
fourth digit: **the sector promotes 2 of 144 candidates, and both violate its
own source gate.** Two poles out of 201 frequency bins cannot move a bubble
whose weight is spread over the whole band, whatever quadrature they get.

Consequences for what to do next:

* Sec. 6's `|P|^2` correction would be fixing a 2 %-at-one-bin error on this
  bed. Worth having, not urgent.
* The blocking questions are upstream, in the pole SOLVE and its screening:
  whether `eps_nep` at 1e-9 to 1e-8 really means "not a pole", and why the
  projected source varies by 200x across a promoted pole's own window. The
  second is not a threshold to loosen -- it says the smooth-source premise
  fails on this bed, which is a statement about the physics, not the code.
* No conclusion about the METHOD should be drawn from this bed until one of
  those moves. It is not currently a test of the pole sector.

### 4.3 The `-1.000` ring-leg gate -- RESOLVED, it was the gate

The gate read `-1.000` on the `congruence` route at every iteration, which
contradicted Sec. 2's construction. The full device log settles it. On the
**pole-off baseline**, every iteration:

    positivity g_lesser    worst=-1.842e-11 at w[72]  ok
    positivity g_greater   worst=-1.000e+00 at w[0]   VIOLATION

`-i G^>(0)` is the near-singular acoustic bin. It is indefinite AND carries the
largest eigenvalue in the whole window, so it fixes both the numerator and the
global normalisation, and `worst` saturates at exactly `-1.000`. The ring
itself excludes that bin (`conv_mask`); the gate did not, and it collapsed
lesser and greater to a single `min`. It therefore reported `-1.000` for the
baseline and for every pole-sector variant alike.

So Sec. 2 was never contradicted -- it was unmeasured. The gate now takes the
ring's own mask, reports each component with its `omega_index`, and prints the
same measure on the UNCORRECTED leg beside it; if the two agree it says
`[== control: gate is blind]` rather than showing a number that means nothing.

A separate hypothesis -- that zero-filling the sparsity pattern (`M .* X`, a
Hadamard product with a non-PSD mask) breaks positivity -- is real as a
mechanism but conditional, and the condition decides it here
(`test_when_the_zero_filled_pattern_can_and_cannot_invent_a_violation`).

A hard band mask IS indefinite: the 3x3 boxcar has eigenvalue -0.414, its
Fejer taper +0.293. But a BLOCK-BANDED pattern of bandwidth >= 1 leaves the
gate's two-block window FULLY populated -- zero entries inside it are masked --
so the window is a genuine principal submatrix and the gate is exact there.
Only a pattern sparse WITHIN the window is exposed, and then the zero fill
invents violations of order 0.1 (measured -1.9e-01 at 55 % within-window
density, against 0 for the true submatrix).

The device says which case applies: the pole-free baseline reads
`g_lesser worst = -1.842e-11`, not a healthy negative number, so the
production pattern is dense inside the window and the mask is not what fired.
A negative reading from this gate is therefore evidence of a defect -- but
only once the pattern's within-window density is known, which is why that is
now written down.

### 4.4 The tail sum rule

The analytic leg is a GLOBAL function, so its large-`w` behaviour is not
cosmetic. Measured: `max |sum_p p_p q_p^T|` IS the coefficient of the `1/w`
tail, matched to 2 % at `w = 1e5`. The true `G` decays like `1/w^2`, and a
spurious `1/w` once made `G_PP` 17x too large at `w = 1e2` and cost four
orders (`source_at_poles`).

It vanishes exactly when `sum_a u_a v_a^dagger = 0`. That cannot hold for a
generic set -- with `u` of full column rank it forces `v = 0` -- but it holds
for the BOSONICALLY CLOSED set, where the residue at `-Omega` cancels the one
at `+Omega`. Pinned in
`test_leg_tail_is_the_residue_sum_and_the_closure_kills_it`: closing the set
drives the residue sum below 1e-10 of the open one and steepens the tail from
`1/w` to `1/w^2`.

So the closure of `state.legs` is load-bearing for the analytic route, and
`residue_sum` is the runtime number that says whether a truncated set still
satisfies it. It is now printed, as `eps_tail`.

---

## 5. What is gated, and what is not

Added since the review:

* **Sector sum.** `SS + SR + RS + RR` against a direct high-order quadrature
  of the SAME hybrid `Ghat = G_S + R_h` -- not against the physical `G`, which
  would fold representation error into the same number as implementation
  error. Measured 2.1e-03 on the bed, against 0.16 for a 20 % error in `SS`,
  0.45 for a dropped mixed sector and 0.91 for a flipped one, with all three
  sectors carrying real weight (0.80 / 0.45 / 0.59 of the total). Offline,
  because the device version needs an extra ring pass.
* **`eps_tail`** -- `max |sum_p p_p q_p^T|`, printed per iteration.
* **`eps_c_rs`** -- `source_fit_tol`'s missing half. The source gate covers
  `c_ss = V^dagger Sigma V` only, while the analytic route freezes all three
  coefficients and `c_rs = G^R_k Sigma V - U D_k c_ss` carries the whole
  retarded background: it can move fast across a pole window where
  `V^dagger Sigma V` is perfectly smooth. Printed, and flagged against the
  same tolerance.
* **`pole registration`** -- the worst sub-cell pole offset, Sec. 2.
* **Ring-leg positivity** -- masked, per component, with its control.

Still open: the zero-filled-pattern positivity question (Sec. 4.3), and a
device-level sector-sum.

---

## 6. A rejected alternative: the half-amplitude Gram bubble

Factor `-i Sigma_k = L L^dagger`, set `Y(w) = G~^R(w) L`, and the Keldysh
component is `-i G~ = Y Y^dagger` by construction. Contracting the vertex
against `Y(u) (x) Y(w-u)` gives

    -i Sigma_{mu nu}(w) = int du/2pi sum_{mn} Z_{mu,mn} conj(Z_{nu,mn})  >= 0

so the bubble is manifestly PSD, and the pole dependence inside `Y` is still
integrable in closed form. The argument is correct.

It is not affordable. The reduced coefficient
`H_{mu,mn,rs} = sum_ij Phi_{mu,ij} (C_r)_{im} (C_s)_{jn}` carries an OPEN index
PAIR `(m, n)` running over the columns of `L`, i.e. over `rank(Sigma_k)`, which
is `~ n_dof` for an interior device. An `O(N_p^4)` estimate that counts only
the pole basis omits it. This is precisely the open-index blow-up
`pole_congruence.py` was designed around -- the reason both flattened families
are FIXED vectors is so the vertex is projected once per iteration rather than
carrying a device-sized index.

The affordable route to the same end is Sec. 2's `|P|^2` result: keep the FFT
ring, and replace the boxcar only on cell pairs `(k, m-k)` with both ends in
the pole set,

    Sigma_m = Sigma_m^FFT - sum_{k, m-k in P} I_km^box + sum_{k, m-k in P} I_km^exact .

Positivity survives without any of the half-amplitude machinery: every cell
contributes a PSD integrand -- the boxcar for `k not in P`, the exact integral
for `k in P` -- so the sum is PSD.

---

## 7. Ledger

Jobs: 4398590 (`pcong2`), 4398779 (`panal`, wasted -- a module-level helper
inserted into the class body truncated `PhononPhononInteraction` and all three
runs died on `AttributeError`; every local test passed because none call
`_inject_pole_sector`), 4398805 (`panal2`), 4398979 (`panal3`).
237.76 / 300 nh, all debug.
