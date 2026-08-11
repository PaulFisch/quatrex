# The three pole-sector routes: formulas as implemented, and what they do

State as of 2026-08-11, commit `2f372dd0`. Selected by
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
frequency resolution is still the grid's.

---

## 3. `leg = "congruence_analytic"` -- wired, and wrong on device

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

The retarded partner is the two-retarded pairing of `SS` alone; the mixed
sectors have no closed form, and the driver's global KK half covers them.

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

### 4.2 `congruence_analytic` (job 4398805) -- DIVERGES

| iter | base | congruence | analytic |
|---|---|---|---|
| 1 | 7.84e-06 | 7.84e-06 | 7.84e-06 |
| 2 | 8.53e-06 | 8.07e-06 | **2.0000**, `rel Sigma` 3.32 |
| 3 | 1.44e-05 | 1.45e-05 | **2.0000**, `rel Sigma` 3.13 |
| 5 | 3.56e-05 | 3.65e-05 | **2.0000**, `rel Sigma` 7.95 |

`P_in` flips sign at iteration 1: +1.578e+04 against `P_out = +1.463e+03`,
residual 0.83.

The kernels are verified in ISOLATION, so the defect is in the assembly:

* `pf_self_energy` matches a dense contraction built independently from the
  vertex, the families and the residue formula -- the one index structure a
  transposed vertex leg or a swapped `(p,q)` would survive every other test.
* `pf_mixed_self_energy` matches a brute-force ring on a bed built from the
  physics, and the residual is the REFERENCE's truncation, not the kernel's:
  1.0e-01 at `w_max = 24`, 1.4e-03 at 40, 2.3e-04 at 64.
* `partial_fraction_legs` reproduces `SR + RS + SS` at arbitrary frequencies
  from the same coefficients, to 1e-9.

### 4.3 An open contradiction, and it is the first thing to resolve

The ring-leg positivity gate reads

    ring leg positivity (cell-averaged congruence): worst = -1.000e+00

on the **`congruence`** route -- the one that runs stably -- at every
iteration.

That contradicts the construction. `< G~^{<,>} >_k` is an average of PSD
matrices and must be PSD, and the gate was added precisely as a check on the
IMPLEMENTATION rather than the maths. Either the gate is measuring the wrong
thing (wrong sign, wrong reshape, wrong distribution state, a window too
narrow for the pattern) or the leg is not what Sec. 2 claims and the route's
stability has some other cause. Until this is settled, the argument given for
why `congruence` works is not established -- only that it does work.

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
satisfies it.

---

## 5. What is not gated

**No sector-sum check runs on the analytic route.** `sector_sum_residual`
exists and would directly catch the failure mode this construction is most
exposed to -- the ring giving up more than the sectors put back. It should be
wired before anything else about Sec. 4.2 is diagnosed by inspection.

Also unverified: whether pair-shared `c_ss` and single-pole `c_sr`/`c_rs`
compose consistently inside `pf_leg_sample`. They are frozen by two different
rules (Sec. 3.1), and the leg subtracted must equal the leg restored exactly.

## 6. Ledger

Jobs: 4398590 (`pcong2`), 4398779 (`panal`, wasted -- a module-level helper
inserted into the class body truncated `PhononPhononInteraction` and all three
runs died on `AttributeError`; every local test passed because none call
`_inject_pole_sector`), 4398805 (`panal2`). 237.26 / 300 nh, all debug.
