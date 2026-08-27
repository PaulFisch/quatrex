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

## 5b. CORRECTION (2026-08-26): the supercell is not what separates us from
## the reference

An audit of the reference protocol overturns part of section 3. Lindroth & Erhart
used a **3x3x1 supercell** -- also one primitive cell along c, so their calculation
samples q_z = 0 only, exactly as ours does -- and still obtained 5.1 W/mK. The
A-point-sampling limitation is therefore *shared with the reference* and cannot be
what makes us differ from it.

What still stands from sections 1-2 is that **our own number is not converged**:
kappa_zz swings by a factor 3.3 across cubic cutoffs and the A-point mode goes
imaginary at a 5.0 A pair cutoff. That is a property of our cluster-expansion fit
with a tunable cutoff, which the reference's finite-displacement construction does
not have -- their FCs are whatever the 54-atom cell supports, with no cutoff to
sweep.

Protocol differences, measured rather than assumed:

| | ours | Lindroth & Erhart |
|---|---|---|
| XC / vdW | PBE + D3 zero-damping (IVDW=11) | **vdW-DF-CX** (self-consistent nonlocal) |
| supercell | 4x4x1 (96 atoms) | 3x3x1 (54 atoms) |
| FC construction | hiphive cluster expansion, cutoffs [6.0, 4.0] | finite displacement, FC3 cutoff 3.8 A |
| isotopes | not included in our quoted number | **included** |
| solver | RTA | RTA |
| kappa_xx, kappa_z | 124, 20.9 | 83, 5.1 (anisotropy **16.3**, not ~30) |

Excluded by measurement: geometry (our relaxed z_S and c match theirs to 0.02 %);
harmonic velocities (our Gamma-A LA is 3022 m/s against their 3.3 km/s, i.e. 8 %
*softer*, which pushes kappa_z the wrong way); the FC3 cutoff (3.8 and 4.0 A both
contain exactly the one cross-gap S-S shell at 3.528 A and both exclude the next at
4.446 A); and the BTE solver (the reference is RTA too).

Isotopes are a real protocol difference and asymmetric: Mo has mass variance
g2 = 5.97e-4 against W's 6.97e-5, so MoS2's isotope correction is much larger than
the WS2 numbers the paper quotes. Matching the protocol **collapses the in-plane gap
to ~1.0x** -- our kappa_xx is essentially exact -- and leaves cross-plane at
**2.5-3.3x**.

**Leading remaining hypothesis: the vdW treatment.** Plain PBE has almost no
interlayer binding, so in our model 100 % of both the cross-gap restoring force and
the cross-gap *anharmonicity* comes from the additive analytic D3 pair term, whose
third derivative at 3.528 A is fixed by the damping-function form with no density
response. Getting Phi2 roughly right (our breathing 52.6 cm^-1 against 55.7 measured)
while Phi3 is ~1.8x too small is entirely possible for that functional form, and
1.8x in |Phi3| is what 3.3x in kappa_z requires. Direction is right: too little
cross-gap anharmonicity, too little cross-plane scattering, kappa_z too high.

Also worth propagating to the draft: the right comparison target is the
equilibrium-limit 4.8-5.1 W/mK, not 2.0. The 2.0 value (Liu, Choi & Cahill, JAP 116,
233107) is a TDTR number, and Jiang et al. (Adv. Mater. 29, 1701068) show the
apparent through-plane kappa of MoS2 is modulation-frequency dependent and needs a
two-channel nonequilibrium model to reach the intrinsic value.

Superseded in part by section 6: the 4x4x2 campaign shows the kappa_zz ~ 21
used above is itself noise-limited, and the honest gap against the reference is
larger than the 2.5-3.3x quoted here.

## 6. The 4x4x2 supercell: proposed as the fix, and it is not one

Sections 1-3 argued that a **4x4x2 supercell** (192 atoms, c = 24.6 A) would
settle this, because 4x4x1 samples q_z = 0 only and never constrains the A
point. That run was made: 40 rattled structures, identical INCAR to the 4x4x1
campaign, same relaxed geometry, same rattle amplitude and seed, ~20 node-hours
on tortin16 (2026-08-26/27), work_dir
`phonon/configs/mos2/fc3_hiphive_mos2_442_vasp`. It does not settle it. What it
does instead is measure how badly the cross-plane channel is determined.

**The k-meshes differ but the sampling does not.** The 4x4x1 statics used
3x3x2 on a 12.294 A c axis, the 4x4x2 used 3x3x1 on 24.588 A. Both are
12x12x2 in the primitive Brillouin zone. Cell thickness is the only variable
between the two campaigns.

### 6.1 The A-point instability is truncation, not sampling

Pair-cutoff ladder at fixed triplet cutoff 4.0 A, same script on both cells
(the 4x4x1 column reproduces section 2 exactly):

| c2 (A) | DOF | rmse | A-point acoustic (THz) | min freq on 8^3 |
|---|---|---|---|---|
| **4x4x2** | | | | |
| 4.0 | 185 | 0.03426 | 0.524 | -0.0000 |
| 5.0 | 197 | 0.02535 | **-0.124** | **-0.173** |
| 6.0 | 229 | 0.01937 | 0.802 | -0.0000 |
| 6.3 | 229 | 0.01937 | 0.802 | -0.0000 |
| **4x4x1** | | | | |
| 4.0 | 185 | 0.03468 | 0.412 | -0.0000 |
| 5.0 | 197 | 0.02602 | **-0.317** | **-0.449** |
| 6.0 | 229 | 0.02000 | 0.775 | -0.0000 |

Sampling q_z does not cure the imaginary mode; it only softens it (-0.124
against -0.317 THz). The 5.0 A cutoff cuts through the interlayer shell, and
where that cut falls has nothing to do with the cell thickness.

**The extra reach is worth nothing.** Doubling c raises the isotropic cutoff
limit only from 6.147 to 6.320 A, because the in-plane box (12.640 A) binds
once c is doubled -- and c2 = 6.3 A is bit-identical to 6.0 A (229 DOF, same
rmse to five digits). No pair shell lies in that window. The 192-atom cell
bought one extra q_z sample and no additional real-space reach.

### 6.2 kappa_zz is not determined by the data

Cubic-cutoff ladder at fixed pair cutoff 6.0 A, phono3py RTA, 300 K, 11^3 mesh,
no isotopes. The 4x4x1 column reproduces section 1 to three digits:

| c3 (A) | DOF | 4x4x1 kappa_xx | kappa_zz | 4x4x2 kappa_xx | kappa_zz |
|---|---|---|---|---|---|
| 4.0 | 229 | 124.0 | **20.88** | 129.6 | **69.55** |
| 4.5 | 331 | 110.5 | 65.44 | 116.7 | 106.16 |
| 5.0 | 395 | 101.5 | 19.88 | 115.4 | 65.46 |
| 5.5 | 867 | 94.4 | 23.93 | 97.9 | 131.74 |

In-plane transfers between cells (5-14 %, both descending into the measured
80-110 band). Cross-plane does not: the doubled cell sits ~3x higher at every
rung, and at c3 = 5.5 A gives kappa_zz > kappa_xx (131.7 against 97.9), which
is impossible for a van der Waals crystal.

Two matched-protocol DFT campaigns of the same material therefore disagree by a
factor 3 in kappa_zz while agreeing in kappa_xx. Section 1's conclusion should
be strengthened: kappa_zz is not merely unconverged in the cutoff, it is not
determined by the data.

### 6.3 What the two campaigns actually disagree about

Fitted weight at the production cutoffs [6.0, 4.0], normalised per primitive
cell, with sandwiches labelled by clustering z (the `zfrac >= 0.5` split in
`_fc_regressor_sweep.layer_split_metrics` is wrong for a cell four sandwiches
thick -- it calls the intra-cell gap "same layer"):

| | 4x4x1 | 4x4x2 | change |
|---|---|---|---|
| FC2 intra-layer | 88.8807 | 88.8934 | +0.01 % |
| FC2 interlayer | 0.9139 | 0.9250 | +1.2 % |
| FC3 intra-layer | 353.5362 | 354.5224 | +0.28 % |
| **FC3 cross-gap** | **5.4667** | **4.2349** | **-23 %** |
| FC3 cross-gap max (eV/A^3) | 0.4465 | 0.3249 | -27 % |
| Gamma shear (cm^-1), meas. 32.5 | 36.65 | 37.95 | |
| Gamma breathing (cm^-1), meas. 55.7 | 52.65 | 49.35 | |

Everything the data constrains well agrees to a few tenths of a percent. The
single quantity that moves is the cross-gap cubic channel, and it moves down by
a quarter. Less cross-gap anharmonicity means less cross-plane scattering means
larger kappa_zz -- which is the direction observed, 20.9 -> 69.6.

### 6.4 Consequence for the diagnosis in section 5b

Section 5b treated kappa_zz ~ 21 as our number, 4x the reference 5.1, and named
PBE+D3 as the leading suspect. The first half of that no longer holds.

The 4x4x2 fit is better determined on both counts that matter: 100 rows per
parameter against 50, and q_z = pi is sampled, so interlayer constants are
pinned individually rather than only through their q_z = 0 sum. The
better-determined fit puts *less* weight in the cross-gap channel and lands
kappa_zz at 65-132, i.e. **13-26x** the reference rather than 4x.

The reading that fits: part of the 4x4x1's low kappa_zz was fit noise sitting
in under-determined interlayer terms and acting as spurious cross-plane
scattering. Removing it exposes how little cross-gap anharmonicity the model
actually has. This makes the PBE+D3 hypothesis stronger, not weaker -- the
deficit in |Phi3| across the gap is larger than section 5b estimated -- but it
is now a hypothesis about a quantity we have shown we cannot measure reliably
from either force set.

**The interim check proposed at the end of the old section 6 -- refit with
random train subsets and compare the spread to the cutoff-to-cutoff swing --
has effectively been run, at the strongest possible setting: not a resampling
of one force set but two independent force sets. The answer is that the
interlayer constants are noise-limited.**

### 6.5 What would actually settle it

Not a bigger supercell. The remaining candidates, in order:

1. **The exchange-correlation treatment.** The same 40 statics with
   **vdW-DF-CX** (self-consistent nonlocal), matching the reference's protocol,
   against our additive analytic D3 pair term whose third derivative at the
   3.528 A S-S contact is fixed by the damping-function form with no density
   response. ~20 node-hours, the same cost as the run above. Not yet run.
2. **A regulariser that does not put noise in the cross-gap channel.** The
   regressor sweep (`phonon/studies/_fc_regressor_sweep.py`) is the tool; ARD
   returns cross_frob = 0.0000 and LASSO 20.28 on the 4x4x1 set, so the
   channel's fitted magnitude is regulariser-dependent by more than the
   supercell moves it. See `mos2_cross_gap_reality_check.md`.
3. Nothing in the fit geometry. Supercell thickness, pair cutoff reach and
   q_z sampling are now all measured and none of them is the lever.

Until one of those lands, every MoS2 cross-plane number in this work -- the
NEGF kappa_z ladder as much as this Boltzmann cross-check -- should be reported
as an order-of-magnitude statement at best, and the in-plane numbers, which are
reproducible across both cells and essentially exact against experiment, carry
the MoS2 story.
