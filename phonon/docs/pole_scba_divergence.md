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

## 5. State, and what is still open

**Implemented (2026-08-11).** `pole_sector.leg = "congruence"`, the default.
The retarded split is built per leg, and what reaches the FFT ring is the CELL
AVERAGE of the reconstruction,

    ring leg = G^{<,>}_k - [<SR+RS+SS>_point - <SR+RS+SS>_cellavg]
             = <G~^{<,>}>_k .

An average of PSD matrices is PSD, so the leg cannot anti-damp however wrong
the pole model is. The cell weights are analytic -- `<D_a>` from a difference
of two complex Logs, `<D_a conj(D_b)>` by partial fractions -- verified
against Gauss-Legendre to 1e-9, per-bin so a non-uniform grid works, and the
correction falls off like `h^2` under refinement, which is the statement that
it is a discretisation fix and not added physics.

`leg = "keldysh"` reproduces the superseded runs. Both are bit-identical for
an empty pole set.

**Open, in order.**

1. **The pole is resolved in the leg WEIGHT, not inside the convolution.** The
   output frequency resolution is still the grid's.

   The obstruction, and how it is removed, are now settled. Any leg term with
   an OPEN (non-modal) index forces the cubic vertex to be re-contracted at
   every frequency, which is why the original design made `G_PP` modal on both
   sides. Partial-fractioning the congruence removes it: collecting every term
   that shares a pole gives

       G~(w) - B^R S B^A = sum_p p_p q_p^T / (w - zeta_p),   p = 1 .. 2 Np

   with `zeta = [z_a, conj(z_b)]` and, for `zeta_p = z_a`, row factor `u_a`
   and column factor `q_a = c^SR_{a,:} + sum_b c^SS_{ab} conj(u_b)/(z_a -
   conj(z_b))`; for `zeta_p = conj(z_b)`, row factor `y_b = c^RS_{:,b} - sum_a
   c^SS_{ab} u_a/(z_a - conj(z_b))` and column factor `conj(u_b)`.

   Every residue is rank one with one fixed row vector and one fixed column
   vector, so the vertex is projected onto the two families ONCE per
   iteration, exactly as it is onto `U` today, and the bubble is the existing
   pole-pole algebra over `2 Np` poles rather than `Np`. The convolution is
   the same residue formula, `J(p,q;w) = -i/(w - p - q)`, nonzero only for
   like-half-plane pairs.

   Cost: `(2 Np)^4`, sixteen times the pole-pole sector, and `vl[rows]` is
   `(nnz, 2Np, 2Np)`. `max_poles` is already the control for this.

   The families must be FROZEN at the poles -- one value each, not one per
   cell -- or they are not frequency independent and the whole reduction
   fails. That is the approximation `source_at_poles` already makes, and it is
   better justified here, since the poles have been removed from `B^R` by
   construction and it is the smooth part being sampled.

   `partial_fraction_legs`, `pf_leg_sample` and `pf_self_energy` implement and
   test this. Not yet wired: freezing `c_sr`/`c_rs` at the poles, switching
   `g_pp` to `pf_leg_sample` (the leg subtracted and the leg restored must be
   the SAME function), and regenerating the MIXED sectors -- with an analytic
   `G_S` the decomposition is `SS + SR + RS + RR` again, and
   `mixed_self_energy_blocked` assumes a pole leg that is modal on both sides,
   so it needs the same two-family generalisation.
2. **Cell-constant sources.** `Sigma_k` is frozen per cell in the congruence,
   the same approximation `source_at_poles` makes and `source_fit_tol`
   measures. The two should be one gate.
3. **Observables** (review Sec. 16) still integrate narrow poles on the grid.
   `<G~>` is the object they should be integrating too, not only the bubble.
4. **`Phi`-derivability** (review Sec. 15). Weaker now that the leg is a
   congruence, but unanswered.
5. **Sec. 3.4's dilemma is resolved, not open.** The centred split gives the
   grid sample at the centre, which is what Dyson wants, and the correct cell
   integral over the cell, which is what the bubble wants, from one object.

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

---

## 7. Verdict: Case A -- the off-grid reconstruction is not physical

Measured 2026-08-11 with `phonon/studies/_pole_subcell.py`, on a frozen
hybrid bed whose `G^<` is built by congruence `G^R S G^A` (PSD at every
frequency by construction) with one narrow promoted pole plus wide
background poles, so the remainder `R = G - P` is smooth as the method
assumes.

The reconstruction every sector acts on is `G_h(w) = P(w) + R_k`. It is
exact at cell centres and unconstrained between them:

| `2 gamma / h` | centre `eps_PSD` | subcell `eps_PSD` | ratio |
|---|---|---|---|
| 0.800 | +2.00e-04 | **-6.05e-02** | 302x |
| 0.160 | +8.09e-06 | **-5.29e-02** | 6.5e3 x |
| 0.040 | +5.01e-07 | **-1.64e-02** | 3.3e4 x |

The centre value is non-negative at every rung, so `G(w_k)` itself is
fine and the failure is purely off-grid. One global normalisation is used
for every row, per the rule in `pole_audit.psd_residual`.

**It is a pole/grid interaction, not a property of the bed.** Refining
the grid at fixed `gamma = 0.05` removes it monotonically, and the
reconstruction becomes PSD once the pole is resolved:

| `h` | 0.500 | 0.250 | 0.100 | 0.050 | 0.020 | 0.010 |
|---|---|---|---|---|---|---|
| `2 gamma / h` | 0.20 | 0.40 | 1.00 | 2.00 | 5.00 | 10.0 |
| subcell `eps_PSD` | -1.35e-01 | -8.12e-02 | -8.99e-03 | -2.69e-03 | -7.96e-05 | **+8.79e-05** |

A defect in the bed would not depend on `h`.

**The damage is largest exactly where the sector is designed to sit.**
Promoted poles satisfy `gamma/h < q_in` by construction, i.e. the far
left of that table. Worse, the violation is largest when the pole sits at
a cell CENTRE, which is where the subtraction is most nearly exact:

| pole offset within its cell | 0.00 h | 0.25 h | 0.50 h |
|---|---|---|---|
| subcell `eps_PSD` | -1.64e-02 | -1.58e-03 | -2.30e-04 |

That is the predicted mechanism. `P(w)` is PSD at every `w` -- 
`pole_keldysh` builds it as the congruence `(U D^R) S (U D^R)^H` -- so
indefiniteness can only enter through the frozen `R_k`, which carries a
negative eigenvalue of size `~|P(w_k)|` precisely when the centre
coincides with the pole peak. Anywhere in the cell where `P(w)` falls
below that, `P(w) + R_k` goes negative.

Negative control: over-subtracting `G_PP` by 1.5x drives the metric to
-2.2e+02, four orders beyond the faithful case, so the diagnostic does
show the wrong relation failing by a large margin.

### What this settles, and what it does not

It settles the review's first decisive question: **no**, the
reconstructed `G_h^{<,>}` is not physical between grid points. By the
review's decision tree this is Case A, so an exact evaluation of
`B(G_h, G_h)` can itself contain gain and `rr_ss_sr` is not expected to
be PSD however carefully the sectors are assembled.

It follows that the output-semantics defect of Sec. 8 below is real but
is **not** the root cause: making the four sectors one functional cannot
restore a positivity that the input representation has already lost.
Fixing it remains necessary, and is not sufficient.

The open question is now the remedy, not the diagnosis. The candidates
are to reconstruct `G^{<,>}` by congruence from a hybrid `G^R` and source
(review Sec. 22, Eq. 7) so positivity is inherited off-grid as it already
is on the pole part, or to gate promotion on subcell positivity so a
cluster is only promoted where its own reconstruction stays physical.

### Caveat

This is a synthetic bed. It reproduces the mechanism and its controls,
but the transfer to a device iterate is not yet measured.

## 8. Output semantics: the four sectors are not one functional

`core/interaction.py:250-263` passes `cell=_h` to the analytic sectors
whenever `pole_sector.cell_average` is True, while RR is the FFT ring at
`sse_phonon_phonon.py:1091-1099`, which returns a CENTRE value. So today

| `cell_average` | SS, SR/RS | RR | semantics |
|---|---|---|---|
| `False` | point | point | consistent |
| `True` (default) | cell average | point | **mixed** |

The exact averaged-RR stencil for a piecewise-constant regular part is

$$\bar\Sigma_{RR,m} = \frac{h}{2\pi}\Big[\tfrac18 S_{m-1} + \tfrac34 S_m
+ \tfrac18 S_{m+1}\Big],\qquad S_m = \sum_k R_k R_{m-k}$$

verified symbolically: the convolution of two piecewise-constant
functions is piecewise linear with breakpoints at cell centres, so its
average over the output cell is that stencil.

Note this reframes the `cell_average` bisection: `False` is the
consistent setting and `True` is the mixed one, so `False` behaving
better is what the mismatch predicts and is not evidence that averaging
is unphysical -- a positive average of PSD matrices is PSD.

## 9. The congruence reconstruction fixes it -- and is not additive

`pole_keldysh.hybrid_keldysh_congruence` reconstructs the off-grid Keldysh
function as review Eq. (7),

$$\widetilde G^R(\omega) = P^R(\omega) + R^R_k,\qquad
  \widetilde G^{\lessgtr}(\omega) = \widetilde G^R(\omega)\,
  \Sigma^{\lessgtr}_k\,\widetilde G^A(\omega)$$

freezing the RETARDED remainder and the source per cell instead of the
Keldysh remainder. Measured on the same bed, against the direct split:

| `2 gamma / h` | PSD direct | PSD congruence | err direct | err congruence |
|---|---|---|---|---|
| 0.800 | -4.84e-02 | **+4.70e-05** | 0.691 | 0.146 |
| 0.160 | -5.54e-02 | **+2.12e-06** | 2.761 | 0.142 |
| 0.040 | -5.37e-02 | **+4.00e-07** | **10.79** | 0.141 |

Two separate wins, and the second was not the one being sought.

**Positivity becomes structural.** `-i G^< = G^R (-i Sigma^<) (G^R)^H` is a
congruence, so it holds for ANY `G^R`, however coarse -- accuracy and
positivity are decoupled. The direct split has no such guarantee because
`R_k` is a difference of PSD objects.

**Accuracy stops degrading with pole width.** The direct split's error
BLOWS UP as the pole narrows -- 69 %, 276 %, 1079 % -- precisely the
regime the sector exists to serve, because it freezes the RESPONSE, which
carries the pole. Congruence freezes the SOURCE, which is smooth, so its
error is flat at ~14 % and set by `h` alone. Both agree exactly at cell
centres.

### The obstruction

This does not drop into the production sector decomposition. That
decomposition exists because the split is ADDITIVE:

$$B(G,G) = B(P,P) + B(P,R) + B(R,P) + B(R,R)$$

which is what lets `SS` be residues, `SR`/`RS` be cell resolvents and `RR`
be an FFT. The congruence form is a triple PRODUCT, not a sum, so it has
no such expansion and cannot be evaluated by those four sectors. Adopting
it for the ring means giving up the additive decomposition, not patching
it.

So the honest status is: the correct reconstruction is known, validated
and available as a function, and the efficient evaluation scheme is not
compatible with it. Two ways out, both design decisions:

1. **Gate promotion on subcell positivity.** Keep the additive scheme and
   promote a cluster only where its own reconstruction stays PSD in the
   cells it modifies -- i.e. only where `2 gamma / h` is large enough. The
   `h`-refinement table in Sec. 7 shows the threshold is near
   `2 gamma / h ~ 5-10`, which is the OPPOSITE of the regime the sector
   was built for, so this narrows the method to where the grid was nearly
   adequate anyway.
2. **Evaluate the bubble on the congruence object directly**, abandoning
   the four sectors for the promoted cells and paying for a positive-weight
   quadrature there. Correct, and expensive only on the few promoted cells.

Route 2 is the only one that keeps the method's purpose. It is a redesign
of the sector evaluation, not a fix to it.

Both `cell_average` semantics (Sec. 8) remain worth fixing, but neither
route depends on it.

## 10. The local finite-cell route: Route 2, made cheap

`/home/paul/Downloads/cheap_analytic_pole_treatment_review.md` proposes
Route 2 of Sec. 9 with the machinery that note was missing. Keep the FFT
ring untouched and REPLACE the finitely many cell-pair contributions
where the rectangle rule fails, using a closed-form finite-interval
kernel. Built and validated as `quatrex.phonon.pole_local`, on the
synthetic bed only (`phonon/studies/_pole_subcell.py` Sec. E,
`tests/quatrex/phonon/test_pole_local.py`).

The distinction from the global split is not presentational. A global
subtraction needs the leg taken out and the leg put back to be one
function, over one support, with one quadrature semantics, and each of
those is a separate way to be wrong -- the six failure modes the review
lists in its Sec. 16, of which this document has already recorded three.
Locally, both halves of every term are integrals of one model over one
interval, so the only approximation left is the local model itself. No
whole-axis residue integral, no tail constraint, no bosonic closure, no
Kramers-Kronig bypass: the correction goes into `Sigma^{<,>}` and stays
inside `delta = Sigma^> - Sigma^<`, where the existing transform covers it.

### The kernel was already in the tree

The review's Eq. (14) is, term for term, the `window=(a,b)` branch of
`pole_bubble.pair_convolution`, which this document's own earlier work
had labelled "A DIAGNOSTIC, not a production kernel". Against dense
composite Gauss-Legendre on one cell, pole cell against a smooth partner,
as relative error of the pair contribution:

| `gamma/h` | rectangle (the ring) | cell-average product | Eq. (14) |
|---|---|---|---|
| 0.400 | 5.14e-01 | 3.29e-01 | 2.5e-18 |
| 0.100 | 1.77e+00 | 5.29e-01 | 4.7e-16 |
| 0.020 | 9.41e+00 | 6.06e-01 | 1.8e-16 |
| 0.005 | 3.82e+01 | 6.21e-01 | 2.2e-16 |
| 0.001 | 1.92e+02 | 6.26e-01 | 2.2e-16 |

Moving the pole from cell centre to edge at `gamma/h = 0.02` swings the
rectangle error 9.41 -> 0.27, a factor 35; the cell-average product is
flat at ~0.61 and Eq. (14) at 1e-16.

### Three corrections to the review

**Its Eq. (7) baseline is not what the code computes.** The ring is a
rectangle rule on POINT SAMPLES (`sse_phonon_phonon.py:1751`);
`pole_sector.cell_average` governs only the analytic sectors' OUTPUT,
never the ring's input legs. Column 2 above is the baseline Eq. (7)
assumes and column 1 is what is there, so the covariance-only framing of
its Sec. 4 does not apply. The implementable form is its Sec. 18: subtract
exactly what the FFT contributed, add the exact finite-cell integral.

**The `O(M^2)` cost claim is understated.** Its Sec. 8 argues that only
pairs of pole-containing cells need correcting, because `delta G_l = 0` in
a smooth cell. A smooth leg still varies across its own cell, so its
fluctuation is not zero, and the covariance is 61 % of the exact answer
for pole against SMOOTH partner -- column 3 above IS that term. Every
partner cell of a pole cell needs correcting: `O(M n_omega)`, not
`O(M^2)`. The first-order pole-background term therefore dominates, which
is the right way round.

**The `O(P^2)` audit of its Sec. 11-12 is already satisfied.**
`pole_congruence.partial_fraction_legs` collects the `2 Np^2`
partial-fraction terms onto `2 Np` distinct pole locations with rank-one
residues, and `pf_self_energy` contracts at `(2 Np)^2`. There is no
quartic intermediate to find. Measured end to end on the local route, the
cost grows as `P^1.06` from `P = 2` to `32`, the `P^2` block not yet
dominant at that size.

### Two things the review does not contain

**The rectangle rule fails across the pole's TAIL, not only in the cell
that holds it.** A neighbouring cell sits one spacing from an unresolved
pole, where the leg still varies by order one across the cell, and the
ring integrates that with one sample too. Correcting the pole cell alone
leaves 4.07e-02; extending the corrected set by one cell on each side
leaves 1.48e-03, and by two, 7.45e-04.

| radius | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| error | 4.07e-02 | 1.48e-03 | 7.45e-04 | 7.10e-04 |
| gain over the ring | 30 | 819 | 1624 | 1705 |

So `radius = 1` is the default and 2 has converged. The cost is linear
in it.

**`rho_out` must mask pairings across the half planes.** The review's
Eq. (33) reads a pole pair's output width off `Im(zeta_p + zeta_q)`, which
vanishes for a pole and its conjugate. That pair generates no output pole
at all -- it is the pairing whose whole-axis convolution vanishes by
contour closure -- so a zero width there is an absent feature, not an
infinitely sharp one. Taken literally the gate refuses every pair on a
bosonically closed set, which is every physical set.

A third, smaller one: the background inside a corrected cell must be
interpolated from POLE-DEFLATED samples. A cell next to an unresolved pole
has a stored sample tens of times its neighbours' (measured 75.8 against
3.0 after deflation), and a polynomial fitted through the raw samples gets
worse at higher degree, not better. Deflating first restores the expected
ordering: reconstruction error 1.2e-02, 7.5e-05, 2.0e-06 at degree 0, 2, 4.

### What it achieves

On a bed whose legs are PSD at every frequency by congruence, against the
exact integrals of the SAME cells the ring sums over:

| `gamma/h` | err ring | err corrected | gain |
|---|---|---|---|
| 0.400 | 4.95e-01 | 1.50e-03 | 330 |
| 0.100 | 2.29e+00 | 4.11e-05 | 5.6e+04 |
| 0.020 | 7.34e+00 | 2.72e-06 | 2.7e+06 |
| 0.005 | 3.03e+01 | 6.25e-07 | 4.8e+07 |
| 0.001 | 1.55e+02 | 1.24e-07 | 1.2e+09 |

The requirement a sharp-resonance method has to meet -- error bounded as
`gamma/h -> 0` -- is met, and on a bed with a rational rather than exactly
representable background the corrected error is flat at 1.5e-3 down to
`gamma/h = 5e-5` while the ring saturates at 145 %.

Positivity is not the story it was for the global route, and the reason is
worth recording. The ring cannot break it: every term is `B[PSD, PSD]`,
PSD by Schur, and the rectangle weights are positive, so a sum of them is
PSD whatever the quadrature error. Measured, the exact, ring and corrected
minimum eigenvalues agree to three digits at every width. What the ring
gets wrong is the MAGNITUDE, by up to a factor 155, and an overweighted
`Sigma^<` is spurious heating rather than a wrong sign. The non-PSD
objects in Secs. 7-9 came from the pole machinery -- `G - G_PP` is a
difference, and the direct Keldysh reconstruction is indefinite off grid
-- not from the ring. The local route never forms such an object.

### The open question is the output grid, as the review says

`rho_out = 2(gamma_p + gamma_q)/h` decides whether a corrected pair's
OUTPUT is representable. Integrating the input exactly does not help if
the resulting feature falls between output samples:

| `gamma_p` | `gamma_q` | `rho_out` | peak / worst grid sample |
|---|---|---|---|
| 0.020 | 0.300 | 0.640 | 1.9 |
| 0.020 | 0.020 | 0.080 | 12.5 |
| 0.005 | 0.300 | 0.610 | 1.9 |
| 0.005 | 0.005 | 0.020 | 50.0 |

Pole against background stays near-resolved because the background carries
the width; pole against pole does not, and it is second order in the
residue. So the method is sound where it matters most and degrades where
it matters least.

The veto that suggests itself does not work, and the measurement is worth
recording. Refusing the unresolved pole-pole cell pairs leaves the ring's
value in place, and on the bed the refused set is precisely the
combination frequencies `Omega = omega_p + omega_q` and their neighbours
-- outputs 21-23 and 31-32 for pole cells 11 and 21 -- which is where the
ring is worst. Declining to correct output 32 costs a factor 7.34;
correcting it costs 2.7e-06. Away from the refused set the gate changes
nothing, 1.99e-05 against 2.72e-06. `rho_min` therefore defaults to 0.

That does not make `rho_out` idle, it makes it a diagnostic rather than a
veto. The corrected value is accurate AT every grid point even at
`rho_out = 0.08`; what an unresolved output loses is the peak BETWEEN
samples, which costs weight only once the stored `Sigma` is integrated or
handed to Dyson, and refusing the correction does not restore that weight
either. When `report["rho_out"]` says the grid cannot carry the answer,
the remedy is the rational output representation of the review's Sec. 24,
not a refusal.

### Status

Frequency-side algebra complete, exact against dense quadrature in all 27
combinations of background degree and pole count, 38 tests, full pole
suite 274 passed / 1 skipped.

NOT wired into `interaction.py`, and deliberately no config flag: the
device path needs the cubic vertex projected per pole cell, which cannot
be validated without a device run, and an option no code path reads is a
trap rather than a safeguard. `LocalLeg` also stores its residues dense,
which is the reference form; a device path carries the rank-one
`p_row`/`q_col` factors instead, and nothing outside `pair_terms`' term
list would have to change.

The prerequisite for that step is not code. The CNT bed promotes 2-11 of
144 candidates with median `h/gamma = 1.35` and 85 % of candidates
overlapping their neighbour, so it has no narrow isolated modes and no
verdict on a sharp-resonance method can be read off it. A bed where narrow
modes carry spectral weight -- most plausibly low temperature, since
anharmonic broadening scales with the Bose factor -- comes first.
