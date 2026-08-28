# The spatial structure of the phonon self-energy

Can the spatial structure of `Sigma` be represented cheaply enough to replace a
truncation? This is the whole record: the index algebra, the range
measurements, and the two investigations that answer it.

It consolidates three documents, two of which are now in `attic/`
(`spatial_truncation_derivation.md`, `spatial_band_range.md`); Part 0 carries
their content. The proposals it is written against are
`~/Downloads/spatially_analytic_G_bubble_experiment_plan.md` and
`~/Downloads/matrix_free_spatial_modal_scba_plan.md` (both 2026-08-27),
untracked.

## Verdict

**The analytic / modal route is dead.** Supplying `G`'s far blocks from an
exponential fit and rebuilding `Sigma` from them (arms E and F) loses to plain
reblocking by 12x on a 1-DOF chain, and at 4 DOF per cell it does not merely
lose -- it collapses, driving the lead current to `-4e-23` and `-6e-12` against
a reference `1.7e-09`. The mechanism is in Sec. 20: the modal fit's median
far-block error rises from 9.0e-02 to 9.3e-01 between `d = 1` and `d = 4`, and
it refuses 124 of 241 frequencies. More degrees of freedom mean more
near-degenerate modes in the pencil, so the route gets worse exactly where a
real device lives.

**The bidirectional semiseparable route is a correct representation whose cost
is undecided.** It is exact at its rank, converges monotonically where
reblocking does not converge at all, and preserves the positivity structure for
free. It has never been cheaper than the incumbent on any bed measured: at
`d = 4` a 4-cell reblock costs `16x` the pin for `eps(J_L) = 4.66e-04` where the
nearest semiseparable arm costs `125x` for `5.04e-04`.

**What would settle it.** Two trends compete and only their product matters.
The rank needed at matched accuracy is FLAT in the DOF count -- 6, 8, 4 at
`d = 1, 2, 4` -- so the cost of matching a reblock falls `2197x -> 729x -> 27x`,
pointing at a crossing near `d ~ 8-16`, below a real Si or CNT cell at
`d = 24-96`. Against that, `r_Sigma` grows like `N^0.75` in device LENGTH. The
product has not been measured because **no bed survives long enough**: the
16-cell chains converge, `N = 24` diverges outright, and no real device admits
a frozen state at `eta = 0` at all. One `d = 8` bed at fixed `N` is the cheapest
decisive step.

Nothing here should be implemented in `src/quatrex/` until that measurement
exists.

---

# Part 0 — the foundation

The index algebra and the range measurements, from the two documents now in
`attic/`. Everything in Parts I and II is measured against this.

## 0.1 The support law

The cubic self-energy is

    Sigma(I,J) = sum_{K1,K2,K1',K2'} Phi_{I,K1,K2} G(K1,K1') G(K2,K2')
                 Phi_{J,K2',K1'}

over transport-cell block indices, convolved in frequency (the frequency
structure plays no part in any of this). Two supports enter. The **vertex reach
`p`**: `Phi_{I,K,L}` is nonzero only for `|I-K|, |I-L|, |K-L| <= p`, and
production has `p = 1` (`fc3_loader.py` keeps one nearest-neighbour shell). The
**leg band `b`**: `G(K,K')` is retained only for `|K-K'| <= b`, which is
`sse_g_band`, default 3, capped at 3. Chaining them,

    |I - J| <= |I - K1| + |K1 - K1'| + |K1' - J| <= p + b + p,

so

    supp(Sigma) = { |I - J| <= 2p + b }.                          (*)

Measured on a 1-DOF chain at `p = 1`, largest distance at which `Sigma` exceeds
1e-13 of its peak:

| leg band `b` | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| measured reach | 2 | 3 | 4 | 5 |
| `2p + b` | 2 | 3 | 4 | 5 |

Test: `test_the_sigma_support_law_is_two_p_plus_band`. **`Sigma` is not
tridiagonal**, and its reach grows with the leg band.

## 0.2 The three truncations, and which are exact

| # | truncation | where | status |
|---|---|---|---|
| 1 | `Phi` to `p = 1` | `fc3_loader.py:5-9` | **exact** on every bed here |
| 2 | legs to `\|K-K'\| <= b` | `sse_g_band` | **exact** for the retained output once `b >= 3` |
| 3 | output to `\|I-J\| <= 1` | `sse_phonon_phonon.py:474`, `for J in range(max(0, I-1), min(n, I+2))` | approximate; discards `2 <= \|I-J\| <= 2p+b` |

Truncation 3 is hard-coded. It does not follow from `p`, it does not follow from
`b`, and by (*) it is not a property of `Sigma`.

**Why 2 is exact and 3 is not**, which is the trap: truncation 2 is exact
*given* truncation 3. With the output pinned at `|I-J| <= 1` the reachable leg
distance is `|K-K'| <= 2p + 1 = 3`, so `b = 3` discards only links that could
not have contributed -- which is why the field is capped at 3. Split by output
distance on the same bed:

| `b` | rel. err on `\|I-J\| <= 1` | rel. err on `\|I-J\| > 1` |
|---|---|---|
| 1 | 3.18e-01 | large |
| 2 | 1.04e-01 | large |
| 3 | **0.000e+00** | large |

Test: `test_band_three_is_exact_on_the_output_band_and_lossy_off_it`. Reporting
a whole-array error overstates the leg band's cost; reporting only the retained
band hides the output pin's. Both mistakes were made, in that order (Sec. 0.6).

**Truncation 1 is exact on every bed in the tree.** Across 45 stored
`fc3_blocks.hdf5` spanning CNT, Si and MoS2 the maximum block-index offset is 1
and the dropped fraction is 0.000 % without exception. That is not the builder
truncating: `build_device_fc3_blocks` takes `vertex_cutoff=None` by default and
`build_inputs.py:153` passes no cutoff. The force constants contain no such
triplet. Three atoms with all pairwise distances within the FC3 cutoff `r_c`
span at most `r_c` along transport, so they occupy at most `ceil(r_c/L) + 1`
cells of length `L`, and

    r_c < L  =>  max block offset 1  =>  the nearest-neighbour shell is EXACT.

MoS2: `r_c = 4.0 Ang` against a primitive `c = 12.294 Ang`. Two honest limits --
the film's `[4,4,3]` fitting supercell independently bounds the offset at 1, so
the two explanations cannot be separated from the stored data (the cutoff
argument alone is sufficient and does not need the supercell); and the
force-constant metadata for CNT, SiNW and SrTiO3 is not in this repository, so
`r_c` and `L` have not been compared there.

## 0.3 What the output pin costs, and why the lever is the blocking

An earlier version reported **10.8 %** from a seven-cell device. That is
finite-size limited and should not be quoted. The tridiagonal band is `3N-2` of
`N^2` entries, so the share outside it grows with the device until `Sigma`'s
decay takes over:

| device [cells] | 7 | 10 | 14 |
|---|---|---|---|
| discarded, range 2.1 cells | 10.5 % | 29.3 % | 30.3 % |
| discarded, range 20 cells | 13.2 % | 34.2 % | 35.7 % |

**About 30 % on a device long enough to have settled**, not 11 %. Range
dependence is real but secondary -- five points between ranges of 2 and 20
cells, against thirty points from the device length -- so the pin is not
principally a long-range effect. The original seven-cell breakdown, by output
distance: 46.7 %, 42.5 %, 8.1 %, 2.0 %, 0.7 %, 0.02 % for `|I-J| = 0,1,2,3,4,>=5`.

The pin does not care how far `G` reaches. Varying the Green-function range over
a factor 3.5:

| range `xi` [cells] | 2.06 | 2.65 | 3.35 | 4.60 | 7.15 |
|---|---|---|---|---|---|
| discarded | 10.5 % | 10.4 % | 10.5 % | 10.9 % | 11.4 % |

Flat, where a long-range effect would grow. Index algebra again: for `|I-J| = 2`
take `K = I+1` and `K' = J-1 = I+1`, giving `|K-K'| = 0`. **The near tail of
`Sigma` is fed by the diagonal of `G` through the vertex's reach**, not by
long-range `G`; long-range `G` appears only in blocks the pin has already thrown
away. Test:
`test_the_discarded_output_weight_does_not_track_the_green_range`.

**The lever is the blocking.** `supp(Sigma)` is in CELLS; group `m` cells into a
block and it becomes `ceil((2p+b)/m)` in BLOCKS, so a wide enough block makes
the tridiagonal restriction discard nothing. On a 12-cell device:

| cells per block | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| discarded | 32.1 % | 5.4 % | 2.5 % | 0.30 % |

Two cells per block already removes six sevenths. The tool exists
(`phonon/studies/engine/reblock_device.py`). This is also the first statement of
the answer Part I confirms: the discarded weight moves five points with the
range and thirty with the blocking, so **the modal route addresses the smaller
term.**

Why the pin is there at all: `omega^2 I - D - Sigma` is solved by RGF, which
needs block-tridiagonal structure. Removing the pin is a change of solver
structure, not of a parameter.

The pin is an ACCURACY defect and not a stability one. `si4x1` diverges and
`si4x2` converges, and it is tempting to read that as the pin; it was tested and
it is wrong. `bubble_positivity.md` Sec. 6.7 ran the MoS2 film at six blocks
(maximal truncation, 71 % discarded) and at two (no mask at all): both abort at
iteration 28, the untruncated run carries MORE gain (13.2 % against 0.41 %), and
the `Sigma^R` residual sequences agree to five significant figures for eight
iterations. Sec. 8 reaches the same conclusion for the Si pair directly.

Block structure of the production beds, for reference:

| bed | blocks | DOF/block | cells/block |
|---|---|---|---|
| `sichk_base` | 3 | 6 | 1 |
| `si4x1` | 4 | 6 | 1 |
| `si4x2` | **2** | 12 | **2** |
| `cnt_cal`, `l4gpu` | 4 | 36 | 1 |
| `sifilm_nk9r` | 3 | 6 | 1 |

Every production bed except `si4x2` runs at one primitive cell per block.
Against the CNT ladder, discarded weight at one cell per block runs about 2 % at
L4, 11 % at L7 and 35 % at L16 -- the reported series stops at seven cells and
brackets from sixteen, which is where this crosses from a few percent to a
third. A correspondence and not a proof: the bed is a 1-DOF chain with a random
vertex, so the percentages are not the device's. What transfers is the shape.

**Caveat on all of these percentages.** They come from a 1-DOF chain with a
random dense nearest-neighbour vertex. The support law (*) and the flatness
argument are structural; the percentages are not. They also depend on the draw
order inside `toy_models.neighbour_cubic_vertex`, which is preserved
deliberately for that reason.

## 0.4 How far a damped mode actually travels

A damped propagating mode has range

    xi = v_g / gamma          [cells]

with `v_g` in cells*THz and `gamma` the half width in THz. Verified against the
complex bands in `tests/quatrex/phonon/test_spatial_modal.py` to 1e-4 in weak
damping: solving `[-H_10/lambda + (z^2 I - H_00 - Sigma^R) - H_01 lambda] v = 0`
and reading `xi = -1/ln|lambda|` off the decaying root gives the same number.
Undressed, an in-band mode sits on the unit circle and `xi` is infinite -- the
scattering self-energy is what makes a spatial truncation meaningful at all.

Si (`sichk_base`, 81 transverse q, 6 branches, 486 pairs; omega spans 0 to
15.153 THz against a bulk band top near 15.5):

| branch-max abs(v_g) [cells*THz] | min | p25 | med | p75 | max |
|---|---|---|---|---|---|
| | 0.0245 | 0.488 | 0.967 | 1.52 | 4.61 |

At the converged census median `gamma = 0.16 THz`:

| range xi [cells] | min | p25 | med | p75 | max |
|---|---|---|---|---|---|
| | 0.153 | 3.05 | 6.05 | 9.48 | 28.8 |

CNT (3,3) at Gamma, 36 DOF per cell, group velocities five times Si's (median
5.16 cells*THz). At the converged `cnt33_L4_linear` widths, `gamma = 0.481` to
`1.97 THz`:

| gamma [THz] | range xi [cells], min/med/max | band 3: branches past the band | median branch keeps |
|---|---|---|---|
| 0.481 (narrowest) | 5.9 / 10.7 / 25.5 | 100 % | 75.6 % |
| 1.97 (median) | 1.5 / 2.6 / 6.2 | 36 % | 31.8 % |

The typical CNT mode has a range near 2.6 cells and a band of 3 is roughly
adequate; the long-lived narrow modes reach 6 to 25 cells and no band between 1
and 3 touches them.

**The inference originally drawn from this is withdrawn.** These numbers were
read as evidence that `sse_g_band` truncates modes that carry heat, and
therefore as the explanation of the CNT band-ladder bracket. Sec. 0.2 settles
that `b = 3` is exact for the retained output band, so the leg band cannot be
what the ladder's arms differ by. The ranges, the reconstruction (Sec. 0.5) and
the mask-PSD bound (Sec. 0.6) are correct and general; the inference is not.

Four limits on the ranges themselves: `gamma` is a census MEDIAN applied
uniformly, so this is not a per-mode range; the dispersion is harmonic, and
under SCBA the bands shift; `xi = v_g/gamma` is a weak-damping statement that
loosens to about a percent once a mode decays within a few cells; and the CNT
numbers pair one bed's harmonic dispersion with another run's linewidth
distribution (same material and cell -- 144 = 4 x 36 -- but not the same job).

**A correction worth recording.** The first version of this measurement read the
stored keys `[nx, ny, nz]` as a transport offset plus two transverse MOMENTUM
indices. They are real-space cell offsets on all three axes, which
`cm_channel.py` settles by reading the same file and summing over transverse
offsets to reach Gamma; a transverse momentum needs a Fourier sum over `ny, nz`
first. Compounding it, the key regex required the second and third indices to be
non-negative and silently dropped every negative transverse offset -- a third of
the file. The result was 25 "q points", three reporting zero transport coupling,
and a claimed median range of 0.57 cells: the opposite conclusion, from a third
of the data under the wrong convention. The tell was in the output and was
missed: that dispersion had no acoustic branch reaching zero at Gamma, which no
translationally invariant crystal can lack.

## 0.5 The distant blocks are generated, not stored

Fitting `G(n) = V diag(lambda^n) C` from `n = 1, 2` ONLY on a 2-DOF cell with an
invertible inter-cell coupling, then predicting out to `n = 12`:

| n | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| rel. error | 1e-15 | 5e-15 | 2e-14 | 4e-14 | 6e-14 |

Roundoff over nine distances never fitted, across which the blocks fall by more
than two orders. Ten numbers reproduce what would otherwise be one dense block
per distance. A range of blocks also sums in closed form, so a long-range sum
costs `r` geometric series and never materialises what it runs over -- checked
to `n = 200`.

On real device cells, at 1.05x the band top so the quadrature reference is
regular:

| bed | DOF | decaying modes | abs(lambda) range | rel. err n=1 / 3 / 8 |
|---|---|---|---|---|
| CNT (3,3), Gamma | 36 | 36 | 7.6e-05 .. 0.145 | 1.8e-15 / 5.5e-14 / 2.5e-10 |
| Si film, transverse Gamma | 6 | 6 | 9.2e-03 .. 0.319 | 9.2e-16 / 1.3e-14 / 1.2e-12 |

The growth with `n` is the REFERENCE, not the reconstruction: the CNT quadrature
self-converges to 6.5e-10 at `n=8` and only 1.2e-06 at `n=12`, because
`|G(12)|` is 6.8e-13 and the integral is chasing its own floor. Checked rather
than assumed.

Two traps, recorded because both produce plausible output: a rank-deficient
inter-cell coupling makes the pencil degenerate (a `D_01` with one nonzero entry
gave roots collapsing to 0 and 173 and the wrong mode count); and the quadrature
reference must be a PERIODIC trapezoid, since `linspace` with both endpoints
double counts and the reference then stops decaying at 5e-6, which looks like a
Green function reaching a floor and is arithmetic.

**The rank and the fit anchor are one choice, not two.** This is the one
non-obvious rule the fitting API encodes. Modes with `|lambda| = 1e-3`
contribute `1e-9` by three cells, yet dropping them costs `1e-4` -- because the
fit was anchored at `n = 1, 2` where they are still in the data, so their weight
is pushed onto the survivors. On CNT at rank 22 of 36:

| fit anchor | rel. err at n = 5 | at n = 8 |
|---|---|---|
| n = 1, 2 | 1.2e-02 | 1.1e-02 |
| n = 3, 4 | 6.0e-05 | 3.0e-05 |
| n = 5, 6 | 2.1e-07 | 1.4e-07 |

It cuts the other way too: a fit anchored far out cannot determine the
coefficient of a mode that has already decayed there -- at `n = 7` a mode with
`|lambda| = 1e-3` contributes `1e-21`, so SHORT-range blocks degrade even at
full rank. The anchor is not a stability knob; it selects the window of
distances the representation is valid on, and the rank follows from it. Hence
`spatial_fit.modal_fit` always solves for the FULL coefficient set at the anchor
and `prune_by_amplitude` drops modes afterwards **without refitting the
survivors**. The pole sector reached the same conclusion for its own local model,
where `_fit_anchor` had to be pinned per candidate.

## 0.6 Why reweighting the mask cannot substitute for a modal sector

Three rings differing ONLY in the spatial legs, on a 7-cell gapped chain with a
dense nearest-neighbour cubic vertex. The grid forces the bed: a ring is a
convolution, so `Sigma(Omega)` needs `G` at `omega` and `Omega - omega` and the
grid must start at zero, while an exact `eta = 0` reference needs the grid to
avoid the band. A gapped chain satisfies both; an ungapped one cannot.

| `g_band` | boxcar error | modal completion | ratio |
|---|---|---|---|
| 1 | 3.18e-01 | 1.8e-16 | 1.8e+15 |
| 2 | 1.40e-01 | 1.1e-16 | 1.3e+15 |
| 3 | 7.14e-02 | 6.8e-17 | 1.1e+15 |

The boxcar is wrong by 32 % at band 1 and 7 % at band 3 **over all `Sigma`
blocks** -- but on the `|I-J| <= 1` band production actually outputs, band 3 is
exact to 0.000e+00 and the 7 % lives entirely in discarded blocks. The
whole-array figure is the right measure of a hard band in general and the wrong
measure of this kernel. The completion is exact to roundoff at every band and
beats WIDENING the boxcar by a block.

The cheap alternative to a modal representation is to keep the boxcar and
re-weight it. Two independent bounds close that route.

**At the output.** The output band is pinned at `|I-J| <= 1` whatever `g_band`
is, so the output mask is the tridiagonal Toeplitz `[w_1, 1, w_1]` with symbol
`1 + 2 w_1 cos(theta)`, non-negative only for `w_1 <= 1/2`. A weighting faithful
to a Green function of range `xi` has `w_1 = exp(-1/xi)`, so PSD-ness demands

    xi <= 1 / ln 2 = 1.4427 cells,

and every range in Sec. 0.4 is far past it. This derives the existing empirical
result rather than restating it: Bartlett has `w_1 = b/(b+1) <= 1/2` only at
`b = 1`, exactly where `test_taper_is_psd_only_at_band_one` finds it.

**On the legs.** The untruncated geometric weight is the Poisson kernel and is
strictly positive, so a geometric taper looks like the natural PSD choice.
Truncated it is not -- cutting a slowly decaying tail leaves a discontinuity. At
`lambda = 0.91` and band 4 the leg symbol reaches `-1.11`. First band at which
it turns positive:

| lambda | xi [cells] | first PSD band | band / xi |
|---|---|---|---|
| 0.30 | 0.83 | 1 | 1.20 |
| 0.50 | 1.44 | 2 | 1.39 |
| 0.68 | 2.59 | 4 | 1.54 |
| 0.80 | 4.48 | 10 | 2.23 |
| 0.91 | 10.60 | 32 | 3.02 |

The band must exceed the range by a factor that itself grows with the range --
the regime in which no truncation was needed in the first place. So the mask has
to go rather than be reshaped. It also explains, from the symbol rather than a
measurement, why `bubble_positivity.md` Sec. 6.10b found the PSD taper fixed
positivity exactly and the run still diverged: the taper repaired the legs while
the output mask stayed indefinite.

## 0.7 Error trail

Recorded because Part 0 exists on account of it. Each step was a claim about
which blocks matter, argued from a number without its mechanism.

1. Measured a boxcar leg band as costing 32 % at `b = 1` and 7 % at `b = 3` and
   reported it as evidence that the kernel truncates modes carrying heat. The
   7 % was entirely in `|I-J| >= 2` blocks production never outputs.
2. Correcting that, concluded `b = 3` is exact and therefore no spatial
   truncation is live. Exact for the legs -- but the output pin is a separate
   hard-coded truncation, and stopping at the leg band missed it.
3. The pin costs ~30 % on a settled device and is insensitive to the range of
   `G`, so the long-range modal machinery does not repair it either. The
   conclusion in (2) was right; the reason given for it was not.
4. Reported truncation 1's size as unrecorded and its guard as structurally
   dead, on the strength of every stored file showing 0.000 % dropped. The
   builder does not truncate -- `vertex_cutoff` defaults to `None` -- so the zero
   means the force constants have no triplet beyond the shell, and the guard
   reads zero because zero is the right answer.

The algebra is four lines and settles the first three; the fourth needed one
line of the builder's signature.

## 0.8 What Part 0 predicted

A low-rank modal representation of `G` answers "how do I carry `G(K,K')` at
large `|K-K'|` cheaply". Nothing in the shipped kernel asks that: truncation 2
makes long-range `G` unreachable, and truncation 3 discards the blocks where it
would have shown up. The representation is correct -- `G(n) = V diag(lambda^n) C`
reproduces real CNT and Si cells to roundoff -- and it is aimed at a question
this ring does not pose.

The live approximation is truncation 3, and repairing it means carrying a
`Sigma` that is not block-tridiagonal: a Schur complement that keeps the
tridiagonal part in the BTD solver and couples a small non-local sector to it,
rather than compressing distant `G`.

Parts I and II are that prediction tested. It holds: Sec. 11 measures the modal
arm losing to reblocking by 12x, and Sec. 20 measures it collapsing entirely at
4 DOF per cell.

---

# Part I — the analytic tail

Working record of `~/Downloads/spatially_analytic_G_bubble_experiment_plan.md`:
can the long-range spatial part of `G` be carried by complex-band modes, and
does that recover a transport-relevant part of the cubic self-energy more
cheaply than a wider explicit band or reblocking. Sections 1-12 are that
programme; the verdict is Sec. 11.

## 1. What the proposal got right, and three things it did not

### The modal-pair vertex projection is correct

The proposal's Eq. (47) survives contact with the kernel's actual index
convention. With `phi_left(a,c,e)`, `phi_right(J,d,b)`, `G_a` on `(c,b)` and
`G_b` on `(e,d)` (`phonon/solver/bubble.py:61-84`), and with
`alpha = K1-I`, `beta = K2-I`, `gamma = K2'-J`, `delta = K1'-J`, the two leg
separations are `R+delta-alpha` and `R+gamma-beta`, which is Eq. (2). The DOF
contraction factorises because `c, e` touch only the left vertex and the
`omega` leg while `b, d` touch only the right vertex and the `Omega-omega` leg.

One hazard for the implementation and not for the algebra: the proposal's
`a,b,c,d` are CELL indices and `bubble.py`'s are DOF indices. They collide.

### `zeta^R` does not leave the frequency integral

`xi_p = xi_p(omega)` and `eta_q = eta_q(Omega-omega)`, so Eq. (47) keeps
`[xi_p(omega) eta_q(Omega-omega)]^R` under the integral and the recurrence
`t_{R+1} = zeta t_R` (Eq. 49) acts on the INTEGRAND, not on `Sigma_R`. It is
still separable, so the FFT survives, but the cost is then

    explicit  ~ 49 * 3 * n_fft * n_dof^3          per output block
    analytic  ~ 49 * r^2 * n_fft log n_fft
    ratio     ~ r^2 log n_fft / (3 n_dof^3),   r = M * n_dof

with `M = 2p+b` the range of the dressed `Sigma^R`. At full rank that is 0.6x on
Si (`n_dof = 6`) and 0.1x on CNT (36). So the analytic contraction verifies the
algebra; it does not by itself buy a saving, and any saving has to be argued
against the production RGF's cost of producing far `G` blocks rather than
against the dense reference. `zeta^R` only factors out once summed over `R`,
which is the auxiliary realisation of the proposal's Sec. 28.

### The PSD sign is the opposite of the proposal's

The proposal writes `-i Sigma^{<,>} = L L^dagger`. This tree's convention is the
other one: `grids.boson_contact_self_energies_from_gamma` sets
`Sigma^< = -i n Gamma`, so `i Sigma^<` is the positive object. Measured on a
frozen chain, the negative spectral weight is

| object | of `+i M` | of `-i M` |
|---|---|---|
| `G^<` | 5.6e-04 | 9.994e-01 |
| `G^>` | 5.6e-04 | 9.994e-01 |
| `Sigma^<_tot` | 1.2e-01 | 8.8e-01 |

Getting it backwards does not fail loudly; it clips almost the whole spectrum
and returns a factor of the wrong object. The 1.2e-01 in the correct sign is
not a numerical artefact -- it is the anharmonic source's own non-positivity,
which `bubble_positivity.md` has been tracking for other reasons.

### A block-Hankel rank is not an exponent count

For `g_n = sum_p A_p xi_p^n` the Hankel matrix factors as `H = L R` with `L`
carrying `xi_p^i A_p`, so

    rank H = sum_p rank(A_p) <= r * b.

A scalar sequence's Hankel rank IS its exponent count; a block sequence's is
that count times the residue rank. Verified by planting both: four exponentials
with full-rank 2x2 residues give Hankel rank 8 and four distinct exponents at
multiplicity 2; the same four with rank-one residues give Hankel rank 4. The
whole programme turns on an exponent count, so the two numbers are reported
separately.

## 2. The shell decomposition replaces the band sweep

The proposal's E1 sweeps `g_cutoff` and differences the results. That cannot
answer the question it is asked.

- `supp Sigma^(b) = 2p + b`, so beyond `R > 2p+3` the band-3 reference is
  identically zero and the "long-propagation share" is 1 for free. The sweep is
  informative only on `2 <= R <= 2p+3`, which is the window
  Sec. 0.3 already measured as vertex-dominated.
- The partial sums are cumulative. `Sigma` is bilinear in `G`, so raising the
  band changes blocks that already existed, through interference; a difference
  of two bands is not the contribution of a shell.
- `Sigma^(b=0)` is not "the vertex-near term". `_filter_g_blocks` keeps the
  block-DIAGONAL `G_KK`, and a diagonal block of a dense inverse already
  carries the whole device's long-range physics.

Splitting the legs by distance shell instead is exact:

    Sigma_R = sum_{m,m'} Sigma_R^{(m,m')} ,   m = |K - K'| .

`compute_phph_self_energy` already knows `(K1,K1',K2,K2')` per task, so this is
an extension of the accumulation key (`se_finite.py`, `shell_bins`/`shells_out`,
default off). Measured on the real kernel: the shells sum to the undecomposed
ring at 1.9e-16, and the shelled total agrees with the default path at 2.8e-16,
the roundoff of regrouping a 36-term sum.

It also yields the finer geometry statement the derivation implies and nothing
had checked: an output at separation `R` takes weight only from leg shells in
`[R-2p, R+2p]`. Zero violations at any output distance on the analytic bed.

On a converged 8-cell chain the decomposition separates cleanly -- `R = 0..2` is
carried by shells 0-2 and `R >= 5` almost entirely by shells `4-5` and `6+` --
which is the vertex-near / propagation-tail split the proposal asks for, and
which the band sweep cannot produce.

## 3. The sizing law: 12 cells is too short

A pure-tail output block needs the leg reach `R >= R0 + 2p` AND both endpoints
clear of the edges. There are two edge effects, not one. `build_device_fc3_blocks`
emits only `0 <= K,K' < n_slabs`, so the `(alpha,beta)` sum loses terms within
`p` of an end and the projected vertex becomes `I`-dependent there; and the OBC
matches the UNDRESSED lead while the interior is dressed, so `G^R` near a
contact carries the growing branch too. With that margin `m_edge`,

    admissible I:  p + m_edge <= I  and  I + R <= N - 1 - p - m_edge
    => a pure-tail block exists only if  N >= R + 2(p + m_edge) + 1 .

At `R0 = 4, p = 1, m_edge = 2` that is `R >= 6` and `N >= 13`. A 12-cell device
has an EMPTY pure-tail region. Every tail statistic prints its admissible-`I`
count and fails loudly when it is zero.

## 4. The reference solver at eta = 0 diverges on both real beds

This constrains which frozen states can be used and was not anticipated.

`phonon/solver`'s dense SCBA, at `eta = 0` exactly, with the spectral (NEVP) OBC
and NO spatial cutoff at all (`sigma_cutoff = g_cutoff = None`):

| bed | blocking | grid | outcome |
|---|---|---|---|
| gapped 1-DOF chain, L8-L10 | -- | fmax 9, dw 0.05-0.075 | converges, resid 1e-8, conservation 3e-4 |
| Si film (transverse q=0), L16 x 6 dof | 1 cell/block | fmax 32, dw 0.133 | linear: stalls at resid 0.80; Anderson: diverges, guard aborts at iteration 24 |
| CNT (3,3) Gamma, L13 x 36 dof | 1 cell/block | fmax 98, dw 0.456 | diverges, guard aborts at iteration 7 |

Both real beds show `Gamma sign viol` on nearly every frequency sample from the
first iterations -- `Sigma^R` with the wrong sign of `Im`, i.e. gain. Neither
run has any mask to blame: this is the untruncated kernel.

That is consistent with what the tree already records rather than new. Si has no
fine-grid limit (`si-no-fine-grid-limit`), and `bubble_positivity.md` Sec. 6.11a
measured the same Si device converging at 2 cells per block and not at 1, with a
factor ~30 in iterate amplitude and no mask difference between them --
"whatever drives the Si instability is carried by the block partition itself".
Reproduced here: the reblocked device (4 blocks x 12 dof) reaches resid 0.91 in
60 iterations where the same 8 primitive cells at 1 cell/block reach 0.99, and
it does so 68x faster.

**Consequence for the programme.** The frozen state has to be the one that
physically exists, which is the state production reaches -- `sigma_cutoff = 1`,
`g_cutoff = 3`, and >= 2 cells per block on Si. That is not a compromise of the
experiment: "frozen" means the arms differ only in how `Sigma` is REPRESENTED
when evaluated on a fixed state, and the state is an input to that, not an
output of it.

### Reblocked, at the production settings: still no frozen real bed

Following the tree's own recipe -- coarser blocking plus `g_band = 3` -- on
devices reblocked exactly (`reblock_device.py` verifies the dense FC2 and FC3
operators are unchanged), with `retarded = "half"`, `sigma_cutoff = 1`,
`g_cutoff = 3`, `eta = 0`:

| bed | blocking | blocks x dof | outcome |
|---|---|---|---|
| Si film | 2 cells/block | 16 x 12 | diverges, resid 95 by iteration 16 |
| Si film | 3 cells/block | 12 x 18 | oscillates at resid ~1.1, `dJ/J` 2 %, **conservation 0.05-0.09** |
| Si film | 1 cell/block | 16 x 6 | stalls at resid 0.80 (`fft`) / diverges (Anderson) |
| CNT (3,3) | 1 cell/block | 13 x 36 | diverges at iteration 7 |
| CNT (3,3) | 2 cells/block | 8 x 72 | running |

Coarser blocking does help, and measurably: the 3-cell blocking brings the lead
balance from 1.000 -- both leads emitting, the divergence signature -- to
0.05-0.09, while the 1-cell and 2-cell blockings do not. That is the tree's
`bubble_positivity.md` Sec. 6.11a result reproduced from the other side. But the
`Sigma` residual still does not fall, so none of these is a frozen state.

Three things the dense reference does not have that production does, each a
candidate and each a change to a shared reference solver rather than to this
programme: per-frequency mixing (`low_freq_mixing_factor = 0.02` in the stored
config, and `scba_loop` has no per-frequency mixing at all); the frequency grid
(production runs `energy_window_max = 15` on Si, i.e. `fmax ~ omega_max`, where
the aliasing gate here forces `2 omega_max`); and the IR machinery
(`sse_low_freq_mask_thz`, `eta_ir_floor`), which an acoustic device at `eta = 0`
plausibly needs and which the gapped chain demonstrably does not.

So the quantitative results below are the analytic chain's. Whether they carry
to a device is open, and closing it is a question about the reference solver's
convergence, not about the spatial representation.

## 5. First result: arms A and B are bit-identical

On a converged 8-cell chain, the four-arm factorial

| arm | `sigma_cutoff` | `g_cutoff` | J_L |
|---|---|---|---|
| A | 1 | 3 | 4.184534e-10 |
| B | 1 | None | 4.184534e-10 |
| C | None | 3 | 4.274608e-10 |
| D | None | None | 4.229376e-10 |

A and B agree to every printed digit. Widening `G` while the output stays pinned
at `|I-J| <= 1` changes nothing, which is
Sec. 0.3's "the pin does not care how far `G` reaches"
reproduced on a different bed, through different code, and at the level of a
current rather than a block norm. `b = 3` is exact for the retained band, so
this is the expected answer and it is a check on the machinery.

`C -> D` -- given the pin has already been removed, does widening `G` move a
current -- came out at 1.07e-02 against a pre-registered floor of 9.10e-03 on
that bed. Not quoted as a result: an 8-cell chain has no pure-tail region
(Sec. 3) and it is a 1-DOF bed with a random vertex.

## 6. A trap: a grid sample on a lead band edge is singular at eta = 0

The group velocity vanishes at a band edge, so at `eta = 0` the surface Green's
function has no imaginary part to regularise it and the Dyson solve raises
"singular matrix". Measure zero in principle and immediate in practice, because
`fmax` and the band edges are both round numbers: the gapped chain's edge at
1.0 THz is hit exactly by `fmax = 9, nfreq_pos = 180`. The bed builder nudges
the sample count until no sample lands on an edge, ignoring edges below 1e-3 THz
(the acoustic zero at Gamma, which sits at `omega = 0`, is excluded from every
physical integral by `pos_mask`, and is regularised by `dc_handling`).

## 7. Prototypes, and what they do not settle

Two measurements taken while designing the programme, on a 40-cell scalar chain
with semi-infinite contacts and uniform dressing. They are prototypes, not
results: one degree of freedom, one frequency, and a source that is not an SCBA
self-energy.

**Contacts only.** `G^R` has Hankel rank 2 in `R` at fixed source -- not 1,
because an interior source sees both contacts -- and `G^<` also rank 2, with
exponents `{lambda, 1/lambda}` to four digits. The second branch is the retarded
root's GROWING partner, carrying the wave from the far source. So the proposal's
single-index decaying form (its Eq. 15) is not what a two-terminal device
produces.

**Uniform interior source.** `G^<` rises to rank 4, with moduli
`{|lambda|, |lambda|, 1/|lambda|, 1/|lambda|}`. A four-term fit
`(A+Br) lambda^r + (C+Dr) lambda^{-r}` does NOT reproduce the sequence, so the
structure is richer than a doubled pair and this prototype does not settle it.
That is what the source-resolved experiment is for: three arms (left contact,
right contact, anharmonic source), whose sum must reproduce the frozen `G^<` by
linearity -- measured at 3.6e-16 -- and three directional pencil estimates per
arm, where the source-resolved algebra predicts the advanced conjugates along
`J`, the retarded roots along `I`, and their products along the diagonal.

What is already clear from the algebra: the residue carries
`(lambda_m lambda_n^*)^I`, so `Sigma` is SEMISEPARABLE, not Toeplitz, and the
proposal's Eq. (53) state-space ansatz `Sigma_R = C A^{R-1} B` is the wrong
class -- it is the special case `mu_a nu_a = 1`. The semiseparable class has the
same `O(N r)` matvec through prefix/suffix recurrences, so nothing is lost by
fitting the right one.

## 8. Where the code is

| what | where |
|---|---|
| pencil -> modes: batched, arbitrary degree `2M+1`, NEVP residual as a mask | `src/quatrex/phonon/spatial_modes.py` |
| coefficient fit, amplitude pruning, geometric sums | `src/quatrex/phonon/spatial_fit.py` |
| block-Hankel rank, block-ESPRIT, semiseparable fit | `src/quatrex/phonon/spatial_hankel.py` |
| exact shell decomposition of the bubble | `phonon/solver/se_finite.py` (`shell_bins`) |
| analytic beds | `phonon/solver/toy_models.py` |
| frozen device bed + study scaffolding | `phonon/studies/_spatial_bed.py` |
| tail attribution + the four-arm factorial | `phonon/studies/_spatial_tail_tails.py` |
| source-resolved Keldysh rank | `phonon/studies/_spatial_tail_rank.py` |
| invariants | `tests/quatrex/phonon/test_spatial_tail.py` |

The pencil generalisation is the one change that alters an existing answer.
`bloch_modes` solved a quadratic; once the output pin is removed `Sigma^R` has
range `M = 2p+b > 1` and the recurrence is `sum_{n=-M}^{M} a_n lambda^n = 0`.
`qttools.nevp.NEVP` was already defined for the general case and `Full`
linearises any length -- checked on a degree-4 3x3 pencil, all 12 roots at
8.8e-14. The consequence propagates: the root count is `2Mb`, the retained
branch `Mb`, so `r = M n_dof` and not `n_dof` in every cost estimate.

## 9. First gate readings, on a converged 20-cell chain

`chain_L20`: gapped 1-DOF chain, 20 cells, cubic 6e16, 281 frequencies,
`eta = 0`, converged to `resid = 9.7e-09` with `conservation = 1.4e-03` in 66
iterations. It satisfies the sizing law (`R <= 13`). The floor pre-registered
from its own conservation error is `4.06e-03`.

### E1, the four-arm factorial

| arm | `sigma_cutoff` | `g_cutoff` | `eps(J_L)` | first-order `dJ_L/J_L` |
|---|---|---|---|---|
| A | 1 | 3 | 6.311e-03 | -1.51e-02 |
| B | 1 | None | 6.311e-03 | -1.51e-02 |
| C | None | 3 | 6.382e-03 | -1.66e-03 |

**A and B agree to every printed digit**, on this bed as on the 8-cell one.
Widening `G` while the output stays pinned at `|I-J| <= 1` changes no current at
all -- Sec. 0.3's "the pin does not care how far `G`
reaches", reproduced at the level of a current rather than a block norm, on a
different bed, through different code.

`C -> D` -- the question nothing had asked -- came out at **6.38e-03 against a
floor of 4.06e-03**, i.e. above it by a factor 1.6. Given the pin already
removed, widening `G` does move the lead current, by about 0.6 %. Marginal, and
on a bed that cannot yet be trusted for it: `G^R`'s block profile is flat out to
`R = 9` because the median modal range is 2.05 cells but the range at the
frequency carrying most of `G^<` is 636, so the bed's own tail is not resolved.
The number is reported as a gate reading, not as a physical result.

`eps_Toeplitz` is 3-9 % on `Sigma` and 1-11 % on `G` over the interior, falling
with distance. So a separation-only representation is a few-percent
approximation here rather than an exact one, which is the direction the
semiseparable structure predicts.

The shell decomposition separates cleanly at every distance: `R = 0..2` carried
by leg shells 0-2, `R = 5` by `(6+, 4-5)` at 23 %, `R >= 6` almost entirely by
`(6+, 6+)`. That is the vertex-near / propagation-tail split the proposal asks
for, and it is what the band sweep cannot produce.

### E3/E4, the Keldysh rank -- **the second gate passes**

Eleven frequencies spread over the band, interior anchor, span 14 blocks. The
source arms reproduce the frozen `G^<` to `1.2e-15`, so the decomposition is
exact and a rank can be attributed.

Median numerical rank over frequency, at four tolerances (Hankel cap 7):

| object | 1e-2 | 1e-3 | 1e-4 | 1e-6 |
|---|---|---|---|---|
| `G^R` | 2 | 3 | 4 | 7 (cap) |
| `G^<` | 4 | 5 | 7 (cap) | 7 (cap) |
| `G^>` | 4 | 5 | 7 (cap) | 7 (cap) |
| `Y = G^R L` | 5 | 7 (cap) | 7 (cap) | 7 (cap) |

Three readings.

**The Keldysh rank is bounded, not device-scaling.** Four to five exponentials
at a practical tolerance on a 20-cell device. The proposal's Sec. 39.2 stop
condition -- "Keldysh rank scales like the full device" -- is not met.

**It costs about twice the retarded rank**, 4 against 2 at 1e-2 and 5 against 3
at 1e-3. That is the pre-registered prediction from the source-resolved
derivation: two contact families, each contributing the retarded set and its
reciprocal partner.

**The positivity factor `Y` is not lower rank than what it factorises** -- 5
against 4 at 1e-2, and saturating earlier. So the factorised formulation of the
proposal's Sec. 9 buys positivity by construction and does not buy rank, which
was the other thing it was hoped for.

A rank at the cap is a lower bound and is reported as one: a Hankel matrix
cannot express more than its own size, and a span of 14 blocks gives a cap of 7.
Resolving the 1e-4 column needs a longer device.

## 10. The analytic contraction is exact where it is supposed to be

Eq. (47), verified against the ring it would replace, on a bed whose legs are
exactly modal (the gapped chain's bulk `G(n) = G(0) lambda^n` is rank one in the
Bloch factor, so any disagreement is the algebra and not the representation):

| `R` | 0 | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|---|
| rel. err | 6.7e-01 | 7.2e-02 | 3.7e-16 | 6.7e-16 | 3.2e-16 | 1.5e-16 |

Roundoff from `R = 2p` outwards, and deliberately wrong below it. The negative
control matters: a formula that happened to work everywhere would mean the
validity window `R >= R0 + 2p` was not what was being tested. Below it an
internal leg is asked for a NEGATIVE separation, where `lambda^{-n}` is the
growing partner and the modal form is not the Green function.

Three things in Eq. (47) are easy to get wrong and none of them fails loudly.
The proposal's `a,b,c,d` are CELL indices while `bubble.py`'s are DOF indices,
and they collide. The left vertex is conjugated (`se_finite.py`) and the modal
factors are not conjugated with it -- a no-op at Gamma with a real FC3 and
wrong in `se_q`. And `zeta^R` stays inside the frequency integral. That last one
decides the implementation: `analytic_tail` drives the ORDINARY kernel with
rank-one modal legs rather than reimplementing the contraction, so the frequency
convolution is the same FFT and a disagreement is unambiguous.

Pairs are screened by FC3-weighted tail amplitude (Eq. 50) and not by
`|lambda|`, because a mode near the unit circle that the vertex barely projects
onto is still irrelevant. `|zeta| >= 1` has no geometric sum and returns `inf`
unless given a device length, rather than a plausible negative number.

**This verifies the algebra. It does not demonstrate a saving** -- see the cost
argument in Sec. 1, which says the analytic route is 0.6x on Si and 0.1x on CNT
at full rank, and worse if the Keldysh legs need the doubled exponent set that
Sec. 9 measured.

## 11. The observable comparison, and criterion 5

Seven arms on the same frozen 20-cell chain, one dense Dyson/Keldysh re-solve
each, all relative to the untruncated reference D.

| arm | what | `eps(J_L)` | `|Sigma|` discarded |
|---|---|---|---|
| A | production pin, band 3 | 6.31e-03 | -- |
| B | pin, wide `G` | 6.31e-03 | -- |
| C | no pin, band 3 | 6.38e-03 | -- |
| **E** | **modal legs beyond band 3, direct fit** | **2.43e-01** | -- |
| **F** | **congruence: modal `G^R`, `G^<` rebuilt** | **3.27e-02** | -- |
| **R2** | **reblock, 2 cells/block** | **2.74e-03** | 58.8 % |
| R3 | reblock, 3 cells/block | 1.18e-02 | 47.4 % |
| R4 | reblock, 4 cells/block | 1.12e-02 | 37.8 % |

Two readings, and they point the same way.

**Reblocking at two cells per block is more accurate than the production pin**,
2.74e-03 against 6.31e-03, while discarding 58.8 % of `|Sigma|` by weight. The
discarded weight is not the error: a tridiagonal restriction at a coarser
blocking throws away more of the matrix and keeps more of the current, because
what it keeps is what the current is made of.

**The modal decompression is 38x worse than the pin it would replace.** Supplying
every leg block beyond band 3 from an exponential fit instead of from storage
moves the lead current by 24 %.

So the proposal's fifth go/no-go criterion -- "a reduced modal-pair
representation is cheaper than reblocking or direct wider-band recursion at the
same accuracy" -- is not met on this bed. It is not at the same accuracy. And
this is the criterion the tree already expected to fail:
Sec. 0.3 measured the discarded weight moving five
points with the range of `G` and thirty with the blocking, and concluded "the
modal route addresses the smaller term". That was an argument from block
weights; this is the same conclusion at the level of a current, with the modal
machinery actually built.

### The congruence route, measured

Arm E is the DIRECT fit of `G^{<,>}` -- route A of the proposal's Sec. 8, the
one it itself calls least safe. Route B continues `G^R` modally instead and
rebuilds `G~^{<,>} = G~^R Sigma_tot^{<,>} G~^A` with `G~^A` literally the
conjugate transpose, so positivity is a congruence rather than a hope. Adding it
as arm F:

| arm | `eps(J_L)` | negative spectral weight of `i G^<` |
|---|---|---|
| E, direct fit of `G^<` | 2.43e-01 | 0.090 |
| **F, congruence via modal `G^R`** | **3.27e-02** | **0.071** |
| (exact) | -- | 0.067 |
| R2, reblock 2 cells/block | 2.74e-03 | -- |

**The congruence route is 7.4x better than the direct fit and lands on the exact
positivity**, which is both of the proposal's Sec. 9 claims confirmed. Its error
is also flat in the fit tolerance -- 0.69, 0.71, 0.72 on `G^<` at
`eps = 1e-2, 1e-3, 1e-4` -- where the direct fit runs away by eight orders,
0.24, 1.75, 7.7e+07. A congruence of a bounded source cannot blow up.

**It does not overturn the verdict.** Arm F is still 5x worse than the
production pin and 12x worse than reblocking at two cells per block.

And the reason is measurable rather than a matter of tuning: **26-30 % of the
fitted residue weight sits in exponents with `|xi| > 1`.** That branch is
physical -- in a two-terminal device it is the wave from the FAR contact, which
the scalar prototype of Sec. 7 already showed -- but an outward continuation
from an interior anchor cannot carry it, because extrapolating a growing
exponent away from the anchor diverges (keeping it puts the far blocks out by a
factor 40 rather than by a few percent). Dropping it is required, and it costs a
quarter of the amplitude. No rank and no tolerance recovers that; it is a
property of continuing a one-sided sequence in a two-terminal device, and the
repair is the proposal's own "explicit boundary + modal interior" architecture
rather than a better fit.

Two limits remain. The bed is a 1-DOF chain with a random vertex, so the
percentages are the bed's. And one fit per frequency is reused at every cell
pair, which is the translation-invariance assumption `eps_Toeplitz` puts at
8-11 %.

### The ordering survives a real device, on a state that does not

The same eight arms on the reblocked Si film -- 12 blocks of 18 DOF, 36
primitive cells, the production FC3, `retarded = "half"`, `g_cutoff = 3` -- whose
frozen state DIVERGED (`resid = 1.0`, lead balance 1.0, `J_L = -3.0e-08` against
`J_R = +3.6e-08`, i.e. both leads emitting). Every absolute number there is
meaningless and none is quoted. What is legible is the ordering, because the
arms differ only in how `Sigma` is represented on one fixed `(G, Sigma)` pair,
which is a well-posed question whether or not that pair is a fixed point:

| bed | reblock | congruence | no pin | pin | direct fit |
|---|---|---|---|---|---|
| chain (converged) | 2.7e-03 | 3.3e-02 | 6.4e-03 | 6.3e-03 | 2.4e-01 |
| Si film (diverged) | 1.4e-01 | 6.4e-01 | 1.0e+00 | 2.0e+00 | 1.0e+01 |

Reblocking is the most accurate representation on both and the direct modal fit
is the worst on both, by an order of magnitude at each end. That is
corroboration and not evidence: a diverged state can order representations
correctly by accident. It is recorded because the alternative -- quoting nothing
from the only real device that ran -- would hide a consistency that does exist.

On that bed the modal continuation is in any case unusable on its own terms:
half the frequencies refuse the fit outright (241 accepted, 240 refused) and the
median far-block error among those accepted is 1.0, i.e. 100 %.

## 12. Open

- E1's `C -> D` on a bed whose own tail is resolved (`xi` of a few cells across
  the band, not 600 at the band bottom), and on a real device.
- The exponent identity: whether the recovered exponents are the advanced
  conjugates along `J` and the retarded roots along `I`. Not yet answered,
  because the operator's bands are ill-defined on this bed -- `Sigma^R` carries
  only 45 % of its weight within `|I-J| <= 1` and 90 % only by `|I-J| <= 8`, so
  there is no low-order pencil to compare against.
- The decompressor and the analytic contraction, both gated on the E6 cost
  argument in Sec. 1.
- Whether any real device admits a frozen state at all under the reference
  kernel; see Sec. 4. The CNT bed does not converge under `retarded="fft"` nor
  under production's `"half"` with the production cutoffs, and the Si film does
  not converge at one cell per block under any arm tried.
- Whether the "explicit boundary + modal interior" architecture recovers the
  30 % of weight the one-sided continuation has to drop. That is the only
  remaining construction that could change Sec. 11, and it is a different
  experiment rather than a better fit.

---

# Part II — the matrix-free programme

`~/Downloads/matrix_free_spatial_modal_scba_plan.md` (2026-08-27) proposes the
repair for Sec. 11's failure -- a bidirectional semiseparable interior between
explicit boundaries -- and on top of it a much larger ambition: never
materialise the long-range `G` or `Sigma` at all, and close the SCBA loop
through a common spatial basis so the Hilbert transform acts only on small
coefficient functions. It names two gates as decisive. Both are now measured.

## 13. Gate G1 passes: the bidirectional form is exact where one-sided fails

A two-terminal chain with a **mismatched contact**, so the interior carries a
genuine reflected wave. A matched lead is the negative control -- it carries no
reflection, so both routes are exact there and it cannot discriminate.

| lead / device spring | one-sided ESPRIT | bidirectional semiseparable | rank |
|---|---|---|---|
| 1.00 (matched) | 5e-16 | 7e-16 | (1, 1) |
| 0.63 | 3.9e-01 | 6.7e-16 | (1, 1) |
| 0.25 | 8.0e-01 | 7.1e-16 | (1, 1) |

Exact at the MINIMAL rank, which is also a correctness check: the inverse of a
block-tridiagonal matrix is block-semiseparable of rank exactly `d`, and this
is the 1-DOF case of that.

Two things had to be right, and the obvious choice was wrong in both.

**The generators must be per cell.** A homogeneous interior `A_i = Lambda`
provably cannot represent a reflecting device: a reflected wave contributes a
term going like `lambda^{i+j}`, and no product `U Lambda^{i-j-1} V` with
cell-independent `U, V` produces one. Measured: the homogeneous two-sided fit
is 40-200 % wrong on exactly the beds where the one-sided fit is.

**The direction rule is an infinitesimal retarded damping, not a group
velocity.** In-band both roots sit on the unit circle and the modulus says
nothing. Transcribing the OBC's `Re dE/dk < 0` test picks the complex
**conjugate** at every in-band frequency -- right modulus, wrong phase, a
plausible decaying tail that is 150 % wrong -- because that test selects modes
travelling INTO the lead, the opposite sense from the tail of `G` on one side of
a source. Perturbing the pencil by `+i eta` and keeping the root that moves
inside the unit circle reproduces the true ratio `G[i,i-2]/G[i,i-1]` to four
digits at every in-band frequency.

`SemiSepOperator` also supplies the `O(N r^2)` matvec that
`spatial_hankel.Semiseparable`'s docstring advertises and never had.

## 14. Gate G5, as first measured on a coarse grid -- **superseded by Sec. 17**

The verdict in this section was taken on a grid that barely resolves the bed and
does not survive a matched comparison. The numbers stand as measured; the
conclusion drawn from them does not. Kept because Sec. 17 is a correction and a
correction needs its antecedent.

`DeltaSigma(omega)` as a (spatial-operator element x frequency) matrix, on three
converged chains differing only in coupling:

| bed | cubic | `xi` med / max (cells) | live `omega` | `r@1e-3` | fraction |
|---|---|---|---|---|---|
| L16 | 2e16 | 1.08 / 3726 | 91 | 42 | 46 % |
| L16 | 5e16 | 1.07 / 743 | 91 | 53 | 58 % |
| L20 | 6e16 | 1.08 / 265 | 140 | 78 | 56 % |

The singular spectrum is flat and essentially **unchanged** across a factor 3 in
coupling and 14 in modal range: `0.87, 0.72, 0.67, 0.58` against
`0.91, 0.71, 0.66, 0.59`. The windowed fallback is a repackaging -- 140
frequencies split into 1/2/4/8/16/32 windows give total state counts
`78, 85, 97, 110, 127, 140`, rising monotonically to exactly the frequency
count. The stated mechanism was that the dominant 3-dimensional spatial subspace
turns **64 degrees (max 90) between frequencies three grid steps apart** -- a
diagnostic normalised by sample COUNT rather than frequency interval, which is
the flaw Sec. 17 identifies.

## 15. A causal `Sigma^R` action needs no common basis -- and costs the same

The common basis is needed for **generators**. It is not needed for **actions**:
the transform is linear and acts pointwise in `(i, j)` along frequency, so for a
frequency-independent probe `H[Sigma x] = H[Sigma] x`.

| probe | error |
|---|---|
| complex | 7.4e-01 |
| **real** | **3.7e-16** |

The production transform is complex-linear on its positive branch and
CONJUGATE-linear on the bosonic mirror (`core/fft_utils.py` takes
`a[::-1].conj()`), so it commutes only with a real probe. That is free: the SCBA
map is R-linear and not C-linear (`core/jfnk.py:26`), so a Krylov solver on it
already runs in the real embedding.

**It works end to end.** A Dyson solve using only the causal action -- no
`Sigma^R` ever formed, no common basis anywhere -- reproduces the explicit
widened solve to **1.7e-14** on the 20-cell chain and **4.2e-13** at `N_D = 96`.

**And it buys nothing.** One frequency pass yields the action at every frequency
at once, so the route is efficient exactly when the Krylov vectors can be shared
across frequencies. They cannot:

| bed | `N_D` | basis needed | ratio against forming `Sigma^R` |
|---|---|---|---|
| chain L20 | 20 | 20 | 1.00x |
| Si film L16 | 96 | 96 | 1.00x |

The shared space has to span the whole cell space, because different frequencies
need different directions. Preconditioning with the local operator at any
reference frequency does not shrink it at all (20 of 20 in every arm). So
`m N_w = N_D N_w` structured applications -- exactly the cost of forming
`Sigma^R` outright.

**The conclusion is symmetric and it is the programme's real difficulty.**
Causality couples every frequency. Paying for it in the REPRESENTATION needs a
common spatial basis, and that basis is not compact (Sec. 14). Paying for it in
the ACTION needs a search space spanning the cell space, which costs the same as
materialising. Both horns are measured, on this bed.

## 16. Gate G2/G4: the self-energy rank grows with device length

Off-diagonal (quasiseparable) rank, median over the band, on the 1-DOF chain at
fixed coupling and grid. The cap is the largest rank the corner block can have,
and a rank at the cap is a lower bound rather than a measurement.

| `N` | cap | `G^R` 1e-2 / 1e-3 | `G^<` | `Sigma^<` | `Sigma` as % of cap |
|---|---|---|---|---|---|
| 10 | 4 | 2 / 3 | 3 / 4 | 4 / 5 (capped) | -- |
| 16 | 10 | 2 / 3 | 4 / 5 | 6 / 8 | 60 % |
| 20 | 14 | 2 / 4 | 4 / 6 | 7 / 9 | 50 % |
| 32 | 29 | 3 / 6 | 6 / 8 | 10 / 13 | 34 % |

Over the uncapped points `N = 16 -> 32`, `Sigma` goes `6 -> 10`, i.e. about
`N^0.75`: **sublinear, and not saturating.** The fraction of the cap falls,
which is why it is not simply linear, but the absolute rank has no plateau over
a factor two in length. The document's success condition is
`r_Sigma` approximately independent of `N` (§49) and its stop condition is a
rank growing with length (§48.2-3); this is between them and on the wrong side
for an augmented Dyson, whose whole point is a local block of fixed size.

At `N = 32` the augmented block would be `d + 2 r = 21` against a 2-cell
reblock's `2d = 2`. That ratio is of block WIDTHS and understates the gap; the
cost accounting is in Sec. 18.

The joint spatial-frequency rank degrades with length too: `78/140` (56 %) at
`N = 20` becomes `108/140` (77 %) at `N = 32`, and the subspace turn rises from
64 to 78 degrees. Whatever Sec. 14 measured, a longer device makes it worse.

**One qualification, and it is the useful one.** This is the WIDE `Sigma`, with
no spatial truncation at all -- the object the programme wants to carry. The
`Sigma` production actually computes is banded at `2p + b = 5`, and a banded
matrix has quasiseparable rank at most its bandwidth, so its rank is bounded by
construction and needs no modal machinery. The representation is being asked to
compress the one object that is not already compressible, and it grows with the
device.

## 17. Correction: the common basis is compact on a resolving grid

Paul's objection to Sec. 14 was that the bed carried no long-range or sharp-peak
physics. Testing it turned up a larger effect than the one being looked for.

At **matched size and matched grid** -- 12 cells, `nfreq_pos = 600`, both
converged to `resid ~ 9.5e-09`:

| bed | live `omega` | `r@1e-2` | `r@1e-3` | `N_w / r` | singular values | subspace turn |
|---|---|---|---|---|---|---|
| dispersive chain | 600 | 33 | 51 | **11.8x** | 1.00 0.90 0.67 0.61 0.52 0.44 | 17.7 deg |
| flat band (sharp line) | 579 | 24 | 44 | **13.2x** | 1.00 0.81 0.65 0.47 0.33 0.26 | 16.2 deg |

Two things follow, and the second is the important one.

**The sharp line helps, but only a little**: 44 against 51, and a visibly faster
singular decay. The hypothesis was right in direction and is not the dominant
term.

**The dominant term is the frequency grid.** Sec. 14 measured 78 operators for
140 frequencies -- 56 %, "not compact". The same kind of bed on a grid four
times finer needs 51 for 600. The rank is a property of the FUNCTION
`DeltaSigma(omega)` and saturates once the grid resolves it; the compression
factor is `N_w / r_s`, so a coarse grid has little redundancy to exploit and a
fine one has an order of magnitude. Production runs the fine grid, because that
is what a sharp line requires.

So **gate G5 is far more favourable than Sec. 14 concluded**: a common spatial
basis of ~50 operators serves ~600 frequencies, and the Hilbert transform would
act on 50 coefficient functions instead of on the full operator at every
frequency.

Two caveats that are mine to own. The "subspace turns 64 degrees in three
samples" diagnostic of Sec. 14 is normalised by sample COUNT, not by frequency
interval, so it necessarily improves on a finer grid -- it measures the grid as
much as the physics and should be read per unit frequency. And the windowed
fallback still rises with the window count on every bed (`44 -> 168` here), so
one global basis remains the right construction; that part of Sec. 14 stands.

What this does **not** change: Sec. 15, that the causal action route costs
`N_D N_w` structured applications whatever the basis does, and Sec. 16, that the
wide `Sigma`'s own semiseparable rank grows with device length. Whether `r_s`
also grows with `N` on a resolving grid is being measured.

## 18. What the structured operator has to beat, in flops rather than widths

Sec. 16 and the proposal's Sec. 39 both set the augmented block `d + r+ + r-`
against reblocking's `2d` and read off a factor. That comparison is of block
widths, and it is not the cost. A block solve is cubic in the width and linear
in the block count, and an `m`-cell reblock divides the count by `m` while
multiplying the width by `m`:

| structure | blocks | width | leading RGF cost |
|---|---|---|---|
| production pin | `N` | `d` | `N d^3` |
| `m`-cell reblock | `N/m` | `m d` | `m^2 N d^3` |
| augmented RGF | `N` | `d + r+ + r-` | `N (d + 2r)^3` |

A 2-cell reblock therefore costs four times the pin, not twice, and the
augmented block has to come in under `4^(1/3) d ~ 1.587 d` to beat it -- that
is `r < 0.293 d` per side, nearly six times harder than the width comparison
implies. Storage is the gentler axis: at the same block count it is quadratic,
so the reblock costs `2x` the pin in memory where it costs `4x` in flops.

The break-even is `d >= 2r / (m^(2/3) - 1)`:

| `r` per side | vs 2-cell | vs 3-cell | vs 4-cell |
|---|---|---|---|
| 4 | 14 | 7 | 5 |
| 7 | 24 | 13 | 9 |
| 10 | 34 | 19 | 13 |

This is the useful reading of the arithmetic, and it is not the dismissal it
first looks like. `r ~ 7` is what the 20-cell chain gives, and `d ~ 24` is
inside the range of a real transport cell -- a Si or CNT cell carrying eight to
thirty-two atoms has `d = 24` to `96`. The comparison is not settled by
counting; it turns on whether `r_Sigma` is flat in `d` or grows with it.

One qualification, and it moves the break-even a long way. The `r` in this
table is the rank at which the representation is EXACT, so the table asks the
structured operator to be exact and the reblock only to be a reblock. That is
not the comparison; both are approximations and the question is cost at matched
accuracy. Sec. 19 measures the accuracy ladder and redoes the break-even
properly, and it comes out at roughly `d ~ 8-10` rather than 24.

### The near field does not account for the rank

The quasiseparable rank as measured in Sec. 16 charges the generators for the
near-field blocks, which the proposal's Sec. 15 operator holds explicitly
("plus the explicit diagonal/near-field blocks"). Excluding them can only lower
the rank, and if the corner were dominated by the near field it would lower it
a lot. `offdiag_rank` now takes a `band` argument -- the number of block
diagonals kept explicit, so `band = 1` is exactly the production BTD structure
-- and on a 14-cell chain:

| object | `b0` | `b1` | `b2` |
|---|---|---|---|
| `G^R` | 2 | 3 | 3 |
| `G^<` | 4 | 4 | 4 |
| `Sigma^<` | 6 | 5 | 5 |

Banding out the near field buys one rank unit on `Sigma` and then stalls. The
corner rank is long-range structure, not near-field bookkeeping. That closes
the refinement rather than motivating a search for a better one, and it means
the `r` in the break-even table can be read off the `b0` measurement to within
one.

## 19. Arm S: carrying `Sigma` itself semiseparably, and the accuracy ladder

Arms E and F fit the Green function and rebuild `Sigma` from it. Neither
compresses `Sigma`, which is the object an augmented Dyson carries and the only
one whose block width enters the RGF cost. Arm S does: `SemiSepOperator` at a
stated rank per direction, per frequency, decompressed, re-solved.

One structural point, which came out the other way from the expectation.
`Sigma^<` is anti-Hermitian, `Sigma^< = -(Sigma^<)^H`, and that is what makes
`i Sigma^<` Hermitian and lets positivity be a statement about a spectrum. The
two triangles are realised by separate truncated SVDs, so a rank cut has no
obvious reason to respect it, and the arm projects with `(M - M^H)/2` to be
safe. The projection measures as a no-op. `_sss_realisation` builds the upper
triangle by flipping the cell order and running the same algorithm, so the two
truncations cut conjugate-related subspaces and the symmetry survives to
roundoff -- checked at ranks 1, 2, 3, 5 on three shapes, and on Hermitian
inputs as well.

**Compression preserves the positivity structure for free**, which is the thing
arm F had to build a congruence to obtain. A semiseparable `Sigma` cannot be
made non-physical by truncating its rank.

On a converged 14-cell chain (`d = 1`, `resid = 9.9e-09`), against the
untruncated reference:

| arm | `eps(J_L)` | augmented width | RGF cost vs pin |
|---|---|---|---|
| R2 | 1.08e-02 | 2 | 4x |
| R3 | 4.37e-04 | 3 | 9x |
| S2 | 1.80e-02 | 5 | 125x |
| S4 | 8.22e-04 | 9 | 729x |
| S6 | 3.21e-05 | 13 | 2197x |
| S8 | 1.90e-14 | 15 | 3375x |

At `d = 1` the semiseparable loses on every row. That is what the width
arithmetic of Sec. 18 predicts and it is not interesting on its own: at `d = 1`
the block is all rank and none of it is amortised over the DOF.

**What is interesting is the shape of the two ladders.** Arm S converges
monotonically in its rank on every bed measured and reaches machine precision
at the quasiseparable rank. Reblocking does not converge in its block size.
Three beds, `eps(J_L)` for `m = 2, 3, 4`:

| bed | `m = 2` | `m = 3` | `m = 4` | `\|Sigma\|` discarded |
|---|---|---|---|---|
| chain L14 | 1.08e-02 | 4.37e-04 | -- | -- |
| chain L16 | 8.96e-04 | 1.61e-03 | 2.88e-03 | 51.8 / 38.7 / 27.9 % |
| chain L20 | 2.74e-03 | 1.18e-02 | 1.12e-02 | 58.8 / 47.4 / 37.8 % |

A coarser blocking discards strictly LESS of `Sigma` -- 51.8 % down to 27.9 %
on L16 -- and on two of the three beds it is nonetheless less accurate, while
on the third it is more. The direction is not even fixed across beds.
Reblocking removes whichever part of `Sigma` the block boundaries happen to
fall on, so a coarser blocking is not a smaller perturbation, it is a different
one, and the block size is not a knob that can be turned for another digit.
The rank is.

### The break-even, at matched accuracy

Pairing each reblock with the cheapest rank that reaches its error, rather than
asking the structured operator to be exact:

| bed | reblock | its `eps` | cost | rank needed | break-even `d` |
|---|---|---|---|---|---|
| L16 | `m = 2` | 8.96e-04 | `4x` | 6 | 20 |
| L16 | `m = 3` | 1.61e-03 | `9x` | 6 | 11 |
| L16 | `m = 4` | 2.88e-03 | `16x` | 6 | 8 |

`d >= 2r / (m^(2/3) - 1)` as before, but with `r` the rank that MATCHES that
reblock rather than the rank that is exact. The break-even lands between 8 and
20 depending on which reblock is the incumbent, against 24 for the exact-rank
reading of Sec. 18, and a real Si or CNT transport cell carries `d = 24` to
`96`. The honest form of this is a range, not a number, and it is a range
whose top is at the bottom of the physical one.

This is a bed of one degree of freedom, and the whole result turns on whether
`r` is flat in `d`. If `r` is a property of the physical range of `Sigma` it
should be, and the augmented block `d + 2r` then grows only linearly while the
reblock's `m d` grows in proportion. If instead `r` grows like `d` -- if the
rank is counting DOF rather than range -- the ratio is constant and no device
is large enough. That measurement is running; Sec. 16's `r_Sigma` growing like
`N^0.75` in device LENGTH is a separate and less favourable fact, and it is the
one that bounds how long a device this can be used on.

### Positivity and causality, measured rather than argued

`Sigma^R` is never fitted in arm S: `solve_arm` rebuilds it with
`build_retarded` from the compressed `Sigma^{<,>}`, so it is Kramers-Kronig by
construction. Combined with the symmetry result above, both of the properties
arm F needed a congruence for come free. Measured on the 16-cell bed, negative
spectral weight of `i Sigma^<` over positive frequencies, and the `Gamma_Sigma`
causality diagnostic:

| arm | `eps(J_L)` | neg. weight | causality violations |
|---|---|---|---|
| D (reference) | -- | 0.6859 | 240 pts, max 5.8e-02 |
| R2 | 8.96e-04 | 0.6792 | 240 pts, max 6.1e-02 |
| R3 | 1.61e-03 | 0.6795 | 240 pts, max 5.3e-02 |
| R4 | 2.88e-03 | 0.6794 | 240 pts, max 5.9e-02 |
| S2 | 8.14e-03 | 0.6894 | 240 pts, max 4.2e-02 |
| S4 | 5.32e-03 | 0.6858 | 240 pts, max 5.7e-02 |
| S6 | 2.03e-04 | 0.6859 | 240 pts, max 5.8e-02 |
| S8 | 5.70e-15 | 0.6859 | 240 pts, max 5.8e-02 |
| F | 7.34e-04 | 0.5641 | 240 pts, max 5.6e-02 |

**The diagnostics do not discriminate, and the reason is the bed.** The
untruncated reference itself carries 0.686 of negative weight and violates the
`Gamma_Sigma` condition at 240 of 241 grid points. That is the same
`Gamma sign viol` the SCBA logs report throughout, and it is a property of a
frozen toy bed at `eta = 0`, not of any compression. No arm can be scored
against a reference that fails the test.

What the column does support is the negative statement, which is the one that
matters here: **no arm makes it worse**. Arm S tracks the reference to within
0.004 at rank 2 and reproduces it to four decimals from rank 6, so truncating
the rank does not manufacture negative weight. Arm F sits 0.12 BELOW the
reference, which is not an improvement but a departure -- the congruence
imposes a positivity the reference does not have, and a route that disagrees
with the exact answer in the safe direction is still disagreeing with it.

## 20. `r_Sigma` against the DOF count, which is what the cost turns on

Sec. 18 left the verdict on one number: whether the semiseparable rank is a
property of the physical RANGE of `Sigma`, in which case the augmented block
`d + 2r` grows slower than the reblock's `m d` and a large enough cell wins, or
of the DOF COUNT, in which case the ratio is constant and no device is large
enough.

Three converged 16-cell beds, `d = 1, 2, 4`, all arms on each. The row that
matters is the cost of the cheapest `S` arm that MATCHES a 2-cell reblock's
error, since that is the like-for-like comparison:

| `d` | R2 `eps` | cost | matching `S` | its width | its cost | break-even `d` |
|---|---|---|---|---|---|---|
| 1 | 8.96e-04 | `4x` | S6 | 13 | `2197x` | 20 |
| 2 | 5.74e-03 | `4x` | S8 | 18 | `729x` | 27 |
| 4 | 4.67e-03 | `4x` | S4 | 12 | **`27x`** | 14 |

**The rank needed does not grow with `d`** -- 6, 8, 4 -- so the cost of
matching a reblock falls by two orders of magnitude over two doublings of the
DOF count: `2197x`, `729x`, `27x`. That is the favourable branch of Sec. 18's
alternative. The rank is counting range, not degrees of freedom, and the
augmented block therefore grows like `d + const` against the reblock's `m d`.

**It has not crossed yet.** At `d = 4` reblocking still wins on every row: R3
costs `9x` for 1.65e-03 where S4 costs `27x` for 1.17e-03, and R4 costs `16x`
for 4.66e-04 where the nearest `S` is S8 at `125x` for 5.04e-04. What has
changed is the size of the gap -- roughly `500x` at `d = 1`, roughly `3x` at
`d = 4`. One more doubling would put the matching arm near parity, and a real
Si or CNT transport cell carries `d = 24` to `96`.

So the honest verdict is neither of the two the plan anticipated. The
programme is not dead on cost arithmetic, and it is not demonstrated either:
the trend over the only three DOF counts measured points at a crossing
somewhere around `d ~ 8-16`, below a physical cell, and no bed has been run
there. That is the measurement this now rests on, and it is one bed away.

Two caveats that belong beside the table. All three beds are 16 cells, and
Sec. 16 measured `r_Sigma` growing like `N^0.75` in device LENGTH, which works
against this at exactly the sizes a real device has -- the DOF trend and the
length trend push opposite ways and only the product matters. And arms E and F
both return `eps = 1.00` at `d = 4`: the modal decompression and the
congruence, which were the original programme, degrade to worthless on a
multi-DOF bed while the semiseparable arm improves. That contrast is the
clearest single statement of what changed between Part I and Part II.

## 21. Where the analytic route actually fails: arms E and F at 4 DOF

Sections 11 and 19 measure the modal arms losing on a 1-DOF chain. On a
converged 16-cell bed at 4 DOF per cell they do not lose, they collapse:

| bed | arm E (`eps J_L`) | arm F | far-block err, median | frequencies refused |
|---|---|---|---|---|
| `d = 1` | 4.43e-02 | 7.34e-04 | 9.0e-02 | 151 / 241 |
| `d = 2` | 1.27e-02 | 4.10e-03 | -- | -- |
| `d = 4` | **1.00e+00** | **1.00e+00** | **9.3e-01** | 124 / 241 |

The `eps = 1.00` is not a crash. Both arms ran and returned a lead current of
`-3.9e-23` (E) and `-5.7e-12` (F) against a reference `1.7e-09` -- the sign is
wrong and the magnitude is gone. Arm F's diagnostics at `d = 4` read
`far-block err 9.30e-01 median / 1.02e+02 max`, `G^< err 7.27e+02`.

The mechanism is the pencil. More degrees of freedom per cell mean more modes at
similar `|lambda|`, so the fit is closer to degenerate: the median far-block
error rises an order of magnitude between `d = 1` and `d = 4`, and the fit
refuses half the frequency grid outright. **The analytic route degrades with
exactly the quantity a real device has more of.** Sec. 0.8 predicted the modal
route addressed the smaller term; this is stronger than that -- at realistic DOF
it addresses nothing.

## 22. The converged rank at 4 DOF

The `d = 4` bed converged (`resid = 8.79e-09`, `conservation = 8.35e-05`):

| object | `r@1e-2` | `r@1e-3` | `r@1e-4` | cap |
|---|---|---|---|---|
| `Sigma^<` | 19 | 25 | 29 | 52 |
| `Sigma^<` band 1 | 17 | 24 | 27 | 48 |
| `Sigma^<` band 2 | 17 | 22 | 25 | 44 |

Augmented width `d + 2r = 38` at band 1, i.e. `857x` the pin for an EXACT
representation. That is the Sec. 18 reading and it is the wrong comparison; the
iso-accuracy reading is Sec. 20, where rank 4 suffices to match a 2-cell reblock
at `27x`. The two numbers answer different questions and both are reported
because quoting either alone misleads.

The rank is robust to convergence: a 6-iteration probe of the same bed gave the
same `17`.

## 23. The length axis is blocked

Sec. 16 measured `r_Sigma` growing like `N^0.75` over `N = 16 -> 32`, and
Sec. 20 measured the DOF trend running the other way. Their product is what
decides the programme, and it has not been measured because the beds do not
exist.

Attempting `N = 24` and `N = 48` at the settings that converge at `N = 16-20`:
the run reached `resid ~ 7e-3` around iteration 43, then diverged --
`J = -5.59e-01 W`, `conservation = 1.0000`, `max|Sigma^R| = 7.6e+06` by
iteration 107, and it never recovered -- still at `conservation = 1.0000` and
`max|Sigma^R| = 6.2e+06` at iteration 432, when it was killed rather than run
out to its 800-iteration cap. A separate attempt at `N = 20` on a four-times finer grid
(`nfreq_pos = 600`) diverged the same way, `J = 1.1e-01 W` with
`max|Sigma^R| = 9.2e+04`, where the same bed converges at 140 points.

This is the same wall Sec. 4 hits on the real beds and that the tree has hit
before: no toy chain holds a frozen state much past `N ~ 20` at a coupling
strong enough to have a self-energy, and no real device holds one at `eta = 0`
at all. **Every number in Parts I and II is therefore on 12-to-20-cell synthetic
chains**, and the length extrapolation is one measured decade with no
confirmation available.

Still pending at the time of writing: the `d = 6` bed, which would add a fourth
point to Sec. 20's DOF ladder. It is stated as pending rather than guessed.

---

# Part III — verdict

## 24. What is settled

**The analytic / modal route is dead.** It loses to reblocking by 12x at
`d = 1` (Sec. 11) and collapses entirely at `d = 4` (Sec. 21). Part 0 predicted
it would address the smaller term; the measurement is worse than the prediction.

**The bidirectional repair is real.** Gate G1 passes -- the two-sided form is
exact at rank (1,1) where the one-sided continuation is 40-98 % wrong
(Sec. 13) -- which confirms the diagnosis that 26-30 % of the residue weight
sits in the far-contact branch `|xi| > 1`.

**Arm S is a correct representation.** It converges monotonically in rank to
machine precision, and reblocking does not converge in block size at all: across
three beds a coarser blocking discards strictly less of `Sigma` and is
nonetheless less accurate on two of them and more on the third (Sec. 19). And
the realisation preserves `Sigma^< = -(Sigma^<)^H` under truncation to 4e-14 --
positivity survives compression for free, which is what arm F had to build a
congruence to obtain.

**The common basis is compact on a resolving grid** -- ~50 operators for ~600
frequencies (Sec. 17), after correcting the coarse-grid artefact of Sec. 14.

**The causal action route works and does not pay.** The identity holds to
3.7e-16 for real probes and fails at 0.74 for complex ones (Sec. 15), but the
Krylov basis reaches `N_D` exactly, so a solve costs `N_D` matvecs and the
compression buys nothing at solve time.

## 25. What is not settled, and the one measurement that would settle it

Arm S has never been cheaper than the incumbent on any bed. At `d = 4`:

| arm | `eps(J_L)` | RGF cost vs pin |
|---|---|---|
| 2-cell reblock | 4.67e-03 | `4x` |
| 3-cell reblock | 1.65e-03 | `9x` |
| 4-cell reblock | 4.66e-04 | `16x` |
| S4 | 1.17e-03 | `27x` |
| S8 | 5.04e-04 | `125x` |

Reblocking wins every row. What has changed is the size of the gap: roughly
`500x` at `d = 1`, roughly `3x` at `d = 4`, because the rank needed at matched
accuracy is flat in `d` (6, 8, 4) while the reblock's width is not.

So the decisive quantity is `r_Sigma(d, N)` jointly, and the cheapest step is
**one `d = 8` bed at fixed `N = 16`**. If the matching rank stays near 4, the
augmented block undercuts a 2-cell reblock somewhere near `d ~ 8-16` and a real
Si or CNT cell at `d = 24-96` is past it, and an implementation has a case. If
it flattens at the `d = 4` gap, the programme is finished and this document is
the deliverable.

Until then nothing here should be ported into `src/quatrex/`: it would be a
representation that is correct, positivity-preserving, and about three times too
expensive in the only regime measured, with an unquantified length penalty on
top.

---

# Appendix — how the campaign was run

Not about the physics; about the four days of compute behind it, and three
things that cost more than they should have.

**The reference kernel is the whole cost.** `phonon/solver/se_finite.py` -- the
dense reference the studies need, because production returns at most three
off-diagonals and a question about long-range `G` cannot be asked of it -- is
98 % of an SCBA iteration: 0.83 s against 0.01 s for the Kramers-Kronig
transform and 0.01 s for the Dyson solve on a 16-cell 1-DOF bed. It issues one
`bubble_dense_from_fft` call per `(I, J, iq, kind, K1, K1', K2, K2')` tuple:
251384 calls for six iterations at `d = 4`.

**The thread pool was subtracting.** Scanned on a 256-core tortin node, the
per-task loop is fastest at ONE worker for `d = 1, 2, 4` and at two for `d = 6`:

| `d` | 1 thr | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 1 | 1.00x | 0.84 | 0.90 | 0.84 | 0.80 | 0.53 | 0.41 |
| 2 | 1.00x | 0.52 | 0.21 | 0.16 | **0.15** | 0.17 | 0.17 |
| 4 | 1.00x | 0.64 | 0.36 | 0.27 | **0.26** | 0.30 | 0.30 |
| 6 | 1.00x | **1.30** | 0.77 | 0.67 | 0.55 | 0.59 | 0.58 |

Every study before 2026-08-28 ran at `--threads 16`, i.e. at a quarter of serial
speed at `d = 4`. Fixing it took a `d = 4` bed from iteration 17 in four hours to
iteration 115 in three.

**OpenBLAS dies at import on those nodes.** The conda build supports at most 256
threads and sizes its pool from the core count, so on a 256-core node it
exhausts its own per-thread buffer table -- "Program is Terminated. Because you
tried to allocate too many memory regions", repeated, then SIGSEGV, before a
line of the run's own output. It must be capped in the ENVIRONMENT, because
OpenBLAS reads the variable at load time and the in-process `threadpoolctl`
limit the bubble runner already applies is too late. `tortin.py` now exports 32.

**Batching is not a CPU optimisation; it is what makes a GPU usable.** Carrying
the task axis as a batch measures 2.5x at `d = 1`, where the per-task arrays are
`(n_fft, 1, 1)` and Python overhead is everything, and 0.68-0.90x at `d = 4-6`,
where numpy is already at its FLOP/memory bound. On a GH200, complex128
throughout:

| `d` | CPU per-task | GPU batched | GPU per-task |
|---|---|---|---|
| 1 | 1.49 GF/s | 19.5 (13.0x) | **0.09x** |
| 2 | 1.84 | 10.2 (5.5x) | **0.47x** |
| 4 | 7.92 | 573.7 (**72.5x**) | 1.32x |
| 6 | 11.34 | 1263.0 (**111.4x**) | 4.15x |

A port WITHOUT batching would be slower than the CPU at `d = 1, 2`: 21000
launches of a few microseconds' work each. Batched and verified end to end, a
`d = 6` bed ran 389 SCBA iterations in 21 minutes on a GH200 against ~270 s per
iteration on tortin, ~84x, matching the kernel figure. Enabled by
`QX_PHPH_BATCHED=1` / `QX_PHPH_XP=cupy`, both opt-in, bit-identical to the
per-task path on numpy and 6e-16 on the GPU.

**None of this applies to production.** `src/quatrex/phonon/sse_phonon_phonon.py`
already flattens the `(q', quad)` task axis into strided-batched GEMMs with
resident stacking and a cupy scatter-add (`_contract_tau_q_batched`), on by
default via `sse_dense_q_batched`. The slow path was only ever the dense
reference, and the dense reference exists because it is the only thing that can
answer the question Part 0 poses.
