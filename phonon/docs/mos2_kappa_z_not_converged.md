# Why our MoS2 cross-plane kappa is ~4x the first-principles reference

Paul asked why we sit an order of magnitude from the true kappa_z. The honest
answer is that **kappa_zz is not a converged quantity in our setup**, and the
supercell makes it impossible to converge. In-plane is fine; cross-plane is not.

Bed: `cluster/mos2_refit/data`, 40 VASP structures, 96-atom 4x4x1 supercell of the
6-atom primitive, least squares, phono3py RTA, 300 K.

Reference points: Lindroth & Erhart PRB **94**, 115205 give kappa_z = 5.1 W/mK from
first-principles BTE; measured bulk is 2-4.8; measured in-plane ~80-110.

---

## 1. In-plane converges, cross-plane does not

Cubic-cutoff ladder at fixed 6.0 A pair cutoff:

| c3 (A) | DOF | fit RMSE | kappa_xx | kappa_zz |
|---|---|---|---|---|
| 4.0 (production) | 229 | 0.02000 | 123.8 | 20.9 |
| 4.5 | 331 | 0.01800 | 110.7 | **65.4** |
| 5.0 | 395 | 0.01792 | 101.3 | 19.9 |
| 5.5 | 867 | 0.01666 | 94.4 | 23.9 |

kappa_xx descends monotonically into the measured 80-110 band. kappa_zz swings by a
factor **3.3** with no plateau. A quantity that moves 20 -> 65 -> 20 -> 24 as the
cutoff is refined is not a prediction.

Mesh is not the problem: kappa_zz at the production cutoff is 23.8 / 20.9 / 20.9 at
7^3 / 11^3 / 15^3.

## 2. It is the harmonic interlayer coupling, and it is unstable

Pair-cutoff sweep at fixed triplet cutoff 4.0 A. The A point is the zone boundary
along the cross-plane direction; v_z is the acoustic group velocity along Gamma-A.

| c2 (A) | fit RMSE | A-point acoustic (THz) | v_z (m/s) | kappa_xx | kappa_zz |
|---|---|---|---|---|---|
| 4.0 | 0.03468 | 0.412 | 1124 | 73.4 | 24.7 |
| 5.0 | 0.02602 | **-0.317** | 17 | 84.4 | 16.2 |
| 6.0 (production) | 0.02000 | 0.775 | 2112 | 123.8 | 20.9 |

At a 5.0 A pair cutoff the cross-plane acoustic branch is **imaginary at the zone
boundary**, and the cross-plane group velocity swings over two orders of magnitude
(17 -> 2112 m/s) across cutoff choices that move kappa_xx by only ~50 %.

Since kappa_zz = sum C v_z^2 tau, an unconverged v_z enters squared. The production
model happens to land on a stable branch (no imaginary modes on the 8x8x4 mesh), but
that is luck, not convergence.

## 3. The supercell forbids fixing it

The primitive c axis is **12.294 A**, and the supercell is 4x4x**1**. The largest
pair separation representable along z without the periodic image folding back is
therefore **6.147 A**, and the production pair cutoff of 6.0 A is **98 % of that
limit**.

Where the interlayer weight sits (FC2 Frobenius, minimum image):

| pair distance | share of total FC2 | of which interlayer |
|---|---|---|
| 0-2 A | 36.9 % | 0 % |
| 2-3 A | 43.6 % | 0 % |
| 3-4 A | 14.2 % | 0.60 % |
| 4-5 A | 2.5 % | 0.23 % |
| 5.5-6.0 A | 0.5 % | 0.46 % |

Interlayer coupling is **1.28 % of the total** FC2 weight -- small and delicate, as
expected for a van der Waals solid. But of that interlayer weight, **35.5 % sits in
the last half-angstrom before the cutoff** (5.5-6.2 A), and 46.6 % at the 3-4 A S-S
contact.

Substantial weight at the cutoff edge means the interlayer coupling is not converged
in the cutoff -- and because 6.0 A is already 98 % of what the supercell supports,
**the convergence test cannot be run**. Whatever lies beyond 6.147 A is discarded
with no way to check what it was worth.

## 4. What it is not

Both of the obvious suspects are too small by an order of magnitude:

| | kappa_zz | kappa_xx |
|---|---|---|
| isotope scattering (natural Mo/S), c3=4.0 | 20.88 -> 20.10 (**-3.7 %**) | 123.8 -> 100.1 (-19 %) |
| isotope scattering, c3=5.5 | 23.93 -> 23.21 (-3.0 %) | 94.4 -> 79.2 (-16 %) |
| full linearised BTE vs RTA (c3=5.5, isotopes on) | 23.21 -> 21.80 (**-6 %**) | 79.3 -> 94.0 (+19 %) |

Isotopes matter in-plane and barely cross-plane; the BTE solver matters in-plane and
barely cross-plane. Neither accounts for a factor 4.

## 5. Consequence

Every MoS2 cross-plane number in this work rests on interlayer force constants that
are unconverged in the pair cutoff, on a supercell that cannot test the convergence.
That includes the NEGF kappa_z ladder, not only this Boltzmann cross-check: the
transport solver consumes the same force constants.

It also explains a pattern that otherwise looks contradictory -- our Boltzmann
kappa_zz sits *above* the reference while our NEGF kappa_bulk sits *below* the
measurement. Both are reading the same badly-determined interlayer constants through
different sensitivities.

## 6. The fix, and what it costs

**A 4x4x2 supercell** (192 atoms, c = 24.6 A, aliasing limit 12.3 A). That would:

* allow the pair cutoff to go past 6.147 A and the interlayer coupling to be
  converged and *tested*;
* represent next-nearest-layer coupling, which a 1-cell-thick cell cannot;
* let the cubic cutoff move past 4 A without the second cross-gap shell
  (`mos2_cross_gap_reality_check.md` section 6) colliding with the box.

Cost: the DFT is 40 structures at 96 atoms now, so this is roughly a 2x cell with a
comparable structure count -- the single most valuable outstanding calculation in the
MoS2 chain, and it is DFT, so it needs the cluster.

Cheaper interim check, no new DFT: refit at c2 = 5.5 and 6.0 with several random
train subsets and report the spread of v_z and kappa_zz. If the spread is comparable
to the cutoff-to-cutoff swing, the interlayer constants are noise-limited as well as
cutoff-limited, and the 4x4x2 becomes unavoidable.
