# The pole sector diverges: observations, equations, and what is wrong

Investigation record for the anti-damping instability that appears whenever
the pole sector carries a non-empty pole set. Written after the external
review fixes landed, because the failure is now cleanly reproducible and the
earlier record is unreliable -- see Sec. 1, which explains why.

---

## 1. The metrics, and the one that was misread

Three numbers are printed each SCBA iteration. They measure different things
and only one of them is a convergence criterion.

### 1.1 Bubble energy balance -- a CONSERVATION statement

$$\varepsilon_{\rm bal} = \frac{|P_{\rm in} - P_{\rm out}|}{|P_{\rm in}|},
\qquad P = \operatorname{tr}\big[\Sigma_{ij} G_{ji}\big]$$

from `_phonon_bubble_energy_balance` (`core/scba.py`), printed as

    Bubble energy balance: P_in=... P_out=... resid=...

This is the Phi-derivability identity: energy absorbed by the three-phonon
process versus energy emitted. It is **NOT** a convergence residual, and
because it is a **ratio** it is scale invariant:

> **It cannot detect a blow-up.** In the diverging run it read `1.9e-14`
> while `P_in` grew from `2.6e+05` to `4.4e+32`. Both sides diverged
> together, so the ratio improved as the calculation destroyed itself.

Every "converging" and "machine precision" claim made earlier in this
campaign was read off this number and is void.

### 1.2 Relative self-energy residual -- the actual convergence criterion

$$\varepsilon_\Sigma = \frac{\max_{\omega,ij}|\Delta\Sigma_{ij}(\omega)|}
                            {\max_{\omega,ij}|\Sigma_{ij}(\omega)|}$$

printed on the line immediately above:

    Phonon: rel Sigma^R residual ...; lead balance ...; internal spread ...

### 1.3 Lead balance -- the sign gate

$$\varepsilon_{\rm lead} = \frac{|J_L - J_R|}{\tfrac12(|J_L| + |J_R|)}$$

Both lead currents count positive left-to-right, so a steady state has
`J_L = J_R` and `eps_lead = 0`. The difference is taken **without** absolute
values first, deliberately, so that

$$J_R = -J_L \quad\Longrightarrow\quad \varepsilon_{\rm lead} = 2 \text{ exactly}$$

`eps_lead = 2.0000` is therefore not "badly unbalanced". It is a precise
statement that the device **emits into both leads**: anti-damping. The solver
carries the same warning at the sign flip in `sse_phonon_phonon.py` --
"unflipped it injects anti-dissipation".

---

## 2. The observation

Baseline and empty-window control are healthy; the sector is not.

| run | `eps_lead` | `eps_Sigma` | verdict |
|---|---|---|---|
| pole OFF | 6.4e-15 -> 3.8e-04 | 1.00 -> 0.36 | slowly converging |
| sector ON, window ABOVE the spectrum (0 poles) | 7.8e-06 -> 1.4e-05 | 0.57 -> 0.64 | identical to baseline, Gate 0 holds |
| `rr_ss`, poles active | **2.0000** from iteration 2 | 1.00 -> 4.3 -> 18.6 -> 367 | DIVERGED after 46 it |
| `rr_ss_sr`, poles active | **2.0000** | -- | DIVERGED after 35 it |

`P_in` trajectory in the diverging run:

    -2.6e+05  ->  -8.9e+04  ->  -2.3e+06  ->  -1.7e+13  ->  -1.1e+25  ->  -4.4e+32

Final heat profiles reach `1e21`. The instability switches on exactly when the
pole set becomes non-empty, so it belongs to the sector and not to the bed.

---

## 3. The equations

### 3.1 The decomposition

The split `G = G_{PP} + G_{\rm reg}` with `G_{\rm reg} \equiv G - G_{PP}` is
exact by construction for ANY `G_PP`. Bilinearity of the bubble gives

$$B(G,G) = \underbrace{B(G_{PP},G_{PP})}_{SS}
         + \underbrace{B(G_{PP},G_{\rm reg}) + B(G_{\rm reg},G_{PP})}_{SR+RS}
         + \underbrace{B(G_{\rm reg},G_{\rm reg})}_{RR}$$

Scaling the mixed sectors by `lambda` and regrouping,

$$Q(\lambda) = \lambda\,B(G,G) + (1-\lambda)\big[B(G_{PP},G_{PP}) + B(G_{\rm reg},G_{\rm reg})\big]$$

### 3.2 Where positivity lives, and where it does not

`-i Sigma^{<,>} >= 0` is required: it is the damping rate. `B(X,X)` is PSD
when `X` is PSD, and `G` is PSD, so **`Q(1) = B(G,G)` is PSD**.

`Q(0)` is not. `G_reg` is a DIFFERENCE of PSD objects and is generically
indefinite, so `B(G_reg, G_reg)` need not be PSD. Measured with a deliberately
over-subtracted `G_PP`:

| quantity | worst normalised eigenvalue |
|---|---|
| `G_reg` | -1.00 |
| `B(G_reg, G_reg)` | **-0.708** |
| `Q(0)` | **-7.35e-02** |
| `Q(0.25)` | -1.29e-02 |
| `Q(1)` | +2.24e-04 |

**`sectors="rr_ss"` IS `Q(0)`.** Dropping the cross terms removes exactly what
restores positivity, so a non-PSD `Sigma^<` -- i.e. anti-damping -- is the
EXPECTED behaviour there once `G_PP` is a significant fraction of `G`. It
looked harmless in earlier runs only because the pole contribution was small
and broken.

That accounts for `rr_ss`. It does **not** account for `rr_ss_sr`, which is
`Q(1)` and should be PSD. That is the open problem.

### 3.3 The reconstruction, and why it inverts the sign

The sectors never see `G`. They see

    G~_h(w) = P(w) + R_k ,      R_k = G(w_k) - P(w_k)

with `P` the analytic pole sum and `R_k` frozen across cell `k`. That equals
`G` at the cell centre and nowhere else.

An earlier revision of this note blamed a mismatch between the three
quadratures (residues for `SS`, cell resolvents for `SR`/`RS`, FFT for `RR`).
That is **wrong**, and review Sec. 9 settles it: at cell centres

    SS_residue + SR_cell + RS_cell + RR_FFT  =  B(G~_h, G~_h)

exactly. The three quadratures are mutually consistent. What is not consistent
is the function they are applied to.

`R_k` is a DIFFERENCE of PSD objects, so it is generically indefinite -- and
it is frozen, while `P(w)` is not. At the centre `P(w_k)` cancels `R_k`
exactly; away from it `P` has moved and the frozen indefinite remainder is
left exposed. `SR`, `RS` and `RR` all integrate over whole cells, so the
bubble sees the exposed region and acquires **gain**, which is anti-damping,
even though every stored sample is physical.

The condition under which this fires is sharper than "the pole model is
approximate". Measured on the 2x2 bed of `test_pole_subcell.py`, corrupting
only the residue and scanning the whole cell:

| residue error | worst `lambda_min` of `P + R_k` | of the congruence | true `G^<` |
|---|---|---|---|
| exact | +2.15e-06 | +2.05e-06 | +2.02e-06 |
| +20 % | **-1.000** | +1.45e-06 | +2.02e-06 |
| x2 | **-1.000** | +5.21e-07 | +2.02e-06 |
| wrong phase | **-1.000** | +1.62e-06 | +2.02e-06 |
| half (under-subtracted) | +7.95e-06 | +7.19e-06 | +2.02e-06 |

A **twenty percent** residue error is enough to invert the sign. Over-
subtraction is what does it; under-subtraction is harmless. On the device the
residue is a Newton solution of a truncated cluster fitted against a source
that is itself only approximately constant across the pole window, so 20 % is
not a comfortable margin -- and nothing in the code was checking it.

This also disposes of the `Q(0)` attribution in Sec. 3.2. That argument is
true as far as it goes, but it is not what fires: the bisection shows `rr`
ALONE diverging, and `rr` subtracts the legs and adds nothing back. The defect
is in the leg reconstruction, and it needs only one pole.

### 3.3b The fix: decompose the RETARDED function

The Keldysh remainder is not smooth, and that is the whole of it. Writing the
exact Keldysh relation with a retarded split `G^R = P^R + B^R`,

    G^< = G^R S G^A
        = P^R S P^A  +  P^R S B^A  +  B^R S P^A  +  B^R S B^A

the remainder `G^< - P^R S P^A` still contains `P^R(w)` in two of its terms.
It carries the sharp structure. Freezing it per cell is therefore illegitimate,
whereas freezing `B^R` -- from which the poles have been removed by
construction -- is exactly the approximation the split was designed to license.

So decompose `G^R`, not `G^{<,>}`. Anchoring the split at the cell centre,

    G~^R(w) = G^R_k + U [D^R(w) - D^R(w_k)] V^dagger ,
              D^R_ab(w) = delta_ab / (w - z_a)

    G~^{<,>}(w) = G~^R(w) S^{<,>}_k G~^A(w)

with `G^R_k`, `S_k` the stored grid samples. Two properties follow immediately.

`-i G~^{<,>} >= 0` at **every** `w`, not merely at grid points, because it is a
congruence `G (-i S) G^dagger` of a PSD matrix. It holds for any pole set, any
residue, right or wrong -- which is precisely what the table above shows and
what the old form does not have.

The correction **vanishes at the cell centre**: `D^R(w_k) - D^R(w_k) = 0`, so
`G~^{<,>}(w_k)` is the untouched ring `G^R_k S_k G^A_k`. The pole supplies
sub-cell structure and nothing else, and an empty pole set reproduces the
baseline bit-for-bit rather than approximately.

Expanding the congruence gives the four sectors as an identity, with
`dD = D^R(w) - D^R(w_k)`:

    RR = G^R_k S G^A_k                      the untouched grid ring
    SR = U dD [V^dagger S G^A_k]            pole x cell-constant
    RS = [G^R_k S V] dD^dagger U^dagger     cell-constant x pole
    SS = U dD [V^dagger S V] dD^dagger U^dagger    pole x pole

`RR + SR + RS + SS = G~^{<}` to 1e-14 on the bed
(`test_four_sectors_are_the_congruence`). Every piece falls into a category
the existing machinery already handles -- pole x pole analytically, pole x
cell-constant by the cell resolvent, cell-constant x cell-constant by FFT --
so the redesign changes the COEFFICIENTS the sectors carry, not the
quadratures.

The two structural changes against the current code are that the regular leg
becomes the untouched, PSD, grid-sampled ring with no subtraction at all, and
that the pole contributions are centred on their cell rather than absolute.

Measured on the same bed at `h = 20 gamma`, on the cell average -- which is
what the bubble actually integrates:

| leg | rel. error vs exact cell average |
|---|---|
| congruence reconstruction | 4.1e-03 |
| raw grid sample `G^<(w_k)` | 8.2e-01 |

### 3.4 Point value versus cell average -- a genuine dilemma

`Sigma` is consumed in two incompatible ways.

**Pointwise**, in the Dyson solve

$$G(\omega_m) = \big[M(\omega_m) - \Sigma(\omega_m)\big]^{-1}$$

which wants the value AT `omega_m`.

**Integrated**, in the observables and the energy balance

$$P = \sum_m \omega_m\,\operatorname{tr}[\Sigma_m G_m]\,\Delta\omega$$

which wants the CELL AVERAGE, since the sum with weight `dw` is a midpoint
rule.

For a narrow pole these differ enormously. Summing `c/(w-P)` on a grid against
the exact integral:

| `2 gamma / h` | point sample | cell average |
|---|---|---|
| 0.80 | 1.24e-02 | 1.9e-16 |
| 0.40 | 1.65e-01 | 0.0 |
| 0.16 | 1.08e+00 | 1.3e-16 |
| 0.04 | 6.53e+00 | 0.0 |

The cell average is exact for the integral and **wrong for the Dyson solve**;
the point sample is the reverse. The current code emits cell averages
(`cell_average = True`), which optimises the integral and may have broken the
pointwise use. This is a design question, not a bug to be patched: the two
uses need different objects, or the pole part must be kept out of the grid
representation entirely.

---

## 4. Tested and refuted

| hypothesis | refuted by |
|---|---|
| branch cut in the cell-averaged Log | both pairings match direct quadrature to 1.8e-10; the `2 pi i` cancels in the DIFFERENCE of two Logs |
| `Sigma_SS` sign inverted relative to the ring | same sign, ratio 1.000037, rel diff 3.1e-05 against a brute-force ring on the same legs |
| the instability is a property of the bed | the pole-free baseline and the empty-window control are both healthy (`eps_lead ~ 1e-5`) |
| residue mis-scaling | `l^H M' r = 1.000000`; left vector matches an SVD null vector to `1 - 1e-10` |
| **mismatch between the three quadratures** | review Sec. 9: at cell centres `SS + SR + RS + RR = B(G~_h, G~_h)` exactly. The quadratures agree; the FUNCTION is wrong (Sec. 3.3) |
| **`rr_ss`'s failure is `Q(0)` non-positivity** | true but not what fires: `rr` ALONE diverges, and it adds nothing back |

---

## 5. The open problems, in order

1. **Wire the retarded split into the bubble.** Sec. 3.3b is derived, measured
   and pinned by tests, but the sectors still carry the old coefficients. The
   quadratures do not change; the coefficients do, and the regular leg stops
   being a subtraction.
2. **Cell-constant sources.** `S_k` is frozen per cell in the congruence.
   That is the same approximation `source_at_poles` already makes, and
   `source_fit_tol` already measures it, but the two should be the same gate.
3. **Observables** (review Sec. 16) still integrate narrow poles on the grid.
   The reconstruction of Sec. 3.3b is the object they should be integrating
   too, not only the bubble.
4. **`Phi`-derivability** (review Sec. 15) is unresolved. It is a weaker worry
   now that the leg is a congruence, but it is not answered.
5. **Sec. 3.4's dilemma is resolved, not open.** The centred split gives the
   grid sample at the centre (what Dyson wants) and the correct cell integral
   over the cell (what the bubble wants), from one object.

---

## 6. Rules this cost

* **Quote `eps_Sigma` and `|P_in|`, never `eps_bal` alone.** A ratio cannot
  see a blow-up, and this one improved monotonically while the calculation
  diverged over 27 orders of magnitude.
* **`eps_lead = 2.0` is a sign statement, not a magnitude one.** It means
  anti-damping, and it names the failure precisely.
* **Change one thing per device run.** Four changes went into the run that
  diverged, so none of them can be attributed without the bisection now
  running.
* **A symmetry test bed must be built from the physics**, and must show the
  WRONG relation failing by a large margin. Two defects were "validated"
  against beds constructed to satisfy the assumption under test.
