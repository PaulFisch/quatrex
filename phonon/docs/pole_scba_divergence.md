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

### 3.3 Why `Q(1)` can still fail: three quadratures

`Q(1) = B(G,G)` holds in the CONTINUUM. The discrete realisation uses three
different quadratures:

    SS      analytic residues,        J(p,q;w) = -i/(w-p-q)
    SR/RS   cell resolvents,          W_ik = [Log(z_i-w_k+h/2) - Log(z_i-w_k-h/2)]/2pi
    RR      FFT/grid convolution

so the discrete sum is `B(G,G) + (quadrature mismatch)`, and positivity is
guaranteed only for the exact object. The mismatch is not small where the
sector operates, because that is precisely where the grid fails.

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

---

## 5. The open problems, in order

1. **`rr_ss_sr` diverges although `Q(1)` is PSD in the continuum.** The
   candidates are the quadrature mismatch of Sec. 3.3 and the point-vs-cell
   dilemma of Sec. 3.4. A bisection over `cell_average`, sector, and pole
   count is running; it is the discriminating experiment.
2. **`rr_ss` is structurally non-conserving** and should be relabelled from
   "staging setting" to "expected to anti-damp": it is `Q(0)`, and the
   measured `Q(0)` is non-PSD.
3. **`Phi`-derivability of a three-quadrature scheme** (review Sec. 15) is
   unresolved and is the root of 1.
4. **Observables** (review Sec. 16) still integrate narrow poles on the grid.

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
