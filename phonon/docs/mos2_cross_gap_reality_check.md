# Is the MoS2 cross-gap cubic channel physical?

Paul asked the right question: the Fransson/Eriksson/Erhart paper (npj Comput.
Mater. **6**, 135) shows ordinary least squares over-fits when the parameter space
is large, so the cross-gap third-order coupling that OLS reports and ARD prunes to
zero might be an artefact rather than physics.

**Answer: it is physical, and it is the entire cross-plane scattering mechanism.
But the magnitude the draft quotes is inflated, the draft's supporting arguments are
the wrong ones, and the kappa_bulk numbers built on it are not safe.**

Bed: `cluster/mos2_refit/data`, 40 VASP structures, 96 atoms, cutoffs [6.0, 4.0].
Harness: `phonon/studies/_fc_regressor_sweep.py`.

---

## 1. It is not the over-fitting regime

Fit matrix **11520 x 229 = 50.3 rows per parameter**, condition number 1.39e4. The
paper's OLS failure case was ~1.5 rows per parameter with 2525 parameters. The
mechanism Paul was worried about does not apply on the parameter-count axis.

The paper's failure mode is specifically *spurious long-range* terms. Here 100 % of
the cross-gap weight sits at **3.528 A** -- the nearest interlayer S-S contact, the
shortest cross-gap distance that exists -- and nothing at the 4.0 A cutoff edge.
Only 2 of 21 third-order orbits are involved. It is the third derivative of the D3
dispersion term, which is as physical as D3's parametrisation.

## 2. OLS does inflate it, by about 15 %

Synthetic null: take the ARD model (cross-gap exactly zero), generate forces from
it, add noise matched to the OLS residual, refit OLS.

| | cross-gap Frobenius norm |
|---|---|
| OLS on real data | 21.87 |
| OLS on zero-truth data | **10.7 +/- 1.5** (24 seeds) |
| ARD on zero-truth data | 0.0000 |

So half the reported norm in magnitude (a quarter in squared weight) is noise
absorption. Three independent routes agree on the noise-free value:

* quadrature against the null: 19.07
* split-half unbiased estimator: 18.82, corrected ~17.0
* N-scaling extrapolation (`||v||^2 = T^2 + c^2/N`): **T = 17.97**

**Quote ~17-19, not 21.87.**

The floor decays with data (27.3 -> 8.6 as n goes 5 -> 40) while the excess over it
stays flat at 17.5-23.6. An artefact would decay with the floor.

## 3. ARD's zero carries no information

The decisive control. Generate from the **OLS** model, which *does* contain a 21.87
channel, and ask ARD to recover it:

> ARD returns cross_frob = **0.0000 in 24 of 24 seeds**, whether the ground truth is
> 0 or 21.867.

ARD's zero is what this estimator emits regardless of the data. It keeps 34 of 229
DOF -- only 27 of the 70 *harmonic* ones -- so it is a global collapse, not relevance
determination about the gap. At `threshold_lambda=1e6` ARD retains the block
(18.7-20.9) at OLS-level residual, so "ARD deletes it" is a statement about sklearn
defaults, not about the method.

The residual pattern says which generator produced the data. If the ARD model were
the truth, ARD would achieve RMSE 0.0202 on the real data; it achieves 0.0318, 58 %
worse. If the OLS model is the truth, ARD is predicted to give 0.0319 -- observed
0.0318, a 0.2 % match.

## 4. The forces barely constrain it -- and kappa_z is decided by it

Held-out force RMSE, cross-gap terms deleted: **+0.16 to +0.33 %**. On the [6,4]
model that is not significant when bootstrapped over the correct unit (structures):
+0.158 %, t = 1.48, p = 0.146. It becomes significant once quartic terms are added
([6,4,3.0], which removes 53 % of the residual variance): +0.447 %, p ~ 3e-4.

Now the transport observable, phono3py RTA at 300 K, 11^3 mesh. The zero-cross-gap
model is a **constrained refit** (the 21 carrying parameters set to zero, the other
208 re-optimised), so it is a self-consistent force-constant model:

| model | train RMSE | cross norm | fc3 ASR | kappa_xx | kappa_zz |
|---|---|---|---|---|---|
| least squares, full | 0.019996 | 21.87 | 1.01e-09 | 124 | **20.9** |
| cross-gap = 0, refitted | 0.020063 | 0.00 | 1.01e-09 | 161 | **10770** |
| ARD | -- | 0.00 | -- | 297 | 8736 |
| *measured bulk MoS2* | | | | *80-110* | ***2-4.8*** |

kappa_zz is mesh-converged for the full model: 23.8 / 20.9 / 20.9 at 7^3/11^3/15^3.

**A 0.33 % change in the force residual is a factor 516 in cross-plane thermal
conductivity.** Removing the channel leaves cross-plane transport with almost
nothing to scatter off. In-plane kappa moves by 30 %, cross-plane by 50000 %.

This is the paper's own warning -- that a cross-validated force error cannot rank
models -- in its most extreme form, and it is the quantitative demonstration the
draft asserts but never made.

Method note: zeroing the fc3 **array** blocks instead of the parameters breaks the
third-order acoustic sum rule and gives kappa_zz = 231 rather than 10770. Ablate in
parameter space and check the fc3 ASR is unchanged (1.01e-09 here).

## 5. What the draft gets wrong

Three arguments in `document/src/results/10_force_constants.tex` do not survive.

**The Raman rigid-layer modes cannot test this claim.** Splice test: OLS-FC2 with
ARD-FC3 gives shear/breathing 1.0988 / 1.5784 THz, bit-identical to pure OLS, while
carrying cross-gap FC3 = 0. ARD-FC2 with OLS-FC3 gives ARD's failed 1.3703 / 1.4154
with the full 21.867 present. Gamma frequencies are a function of FC2 alone and the
rotational sum rules move the order-3 block by exactly 0. The modes license
"ARD over-prunes the interlayer coupling at *second* order" -- which is a real
defect, and the right argument for the transport ladder -- not "the cross-gap FC3
channel is real".

**The cross-validated-error argument is void.** ARD's 55 % worse CV error is not
caused by cross-gap pruning: deleting only that block moves CV by 0.23 % of ARD's
deficit.

**"LS, ridge and Bayesian ridge agree to three digits" is weak evidence.** All three
are dense estimators sharing the same noise floor. The real argument is that two
*selection* methods keep the channel: LASSO 20.28, RFE 21.48. ARD is the outlier.

## 6. The kappa_bulk numbers are not safe as attributed

The draft attributes 1.868 (ARD) vs 1.140 (LS) W/m/K to the regressor. Three
problems:

1. **Confounded.** The two models differ in FC2 as much as FC3 -- fitted FC2
   interlayer Frobenius 3.6557 (LS) vs 1.8185 (ARD), rigid-layer modes 1.10/1.58 vs
   1.39/1.42 THz. ARD halves the *harmonic* interlayer coupling, and in the ARD model
   heat crosses the gap purely harmonically, so that is the dominant lever.
2. **Numerically unsupported on the LS side.** `_kappa_z_ladder` refuses `lsM4f`
   (residual 3.0e-3); 1.140 only appears under `--allow-unconverged`, from a
   two-point fit with zero degrees of freedom.
3. **The vertex is truncated.** At a 5.0 A third-order cutoff a *second* cross-gap
   shell appears carrying weight comparable to everything inside 4 A. The interlayer
   cubic coupling is not converged at the production cutoff.

Also: ~35 % of the squared cross-gap weight sits in one orbit with p = 0.12 and is
not established; the solid part is the nearest S-S orbit (F = 4.66, p = 9.8e-5).

## 7. What to run next

1. The transport splice the ladder needs: LS-FC2 with cross-gap-pruned FC3 at one
   thickness, to separate the harmonic from the anharmonic part of the 10.85 %
   fixed-length difference.
2. Converge lsM4 to 1e-3, then lsM6 -- without a third rung there is no LS
   kappa_bulk.
3. Refit at c3 = 5.0-5.5 A and rebuild the vertex (correctness issue independent of
   the ARD dispute).
4. Add a quartic order to the production fit; everything downstream currently parks
   quartic anharmonicity in cubic parameters.
