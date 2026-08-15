# How far a damped mode travels, and how long the band is

The spatial half of the method proposal asks for a modal representation of
long-range propagation. Before building one it is worth measuring the thing it
would fix: the distance a mode actually reaches, against the number of blocks
`sse_g_band` keeps.

## The relation

A damped propagating mode has range

    xi = v_g / gamma          [cells]

with `v_g` the group velocity in cells*THz and `gamma` the half width in THz.
Verified against the complex bands in `tests/quatrex/phonon/test_spatial_modal.py`
to 1e-4 in weak damping: solving the pencil
`[-H_10/lambda + (z^2 I - H_00 - Sigma^R) - H_01 lambda] v = 0` and reading
`xi = -1/ln|lambda|` off the decaying root gives the same number.

Undressed, an in-band mode sits on the unit circle and `xi` is infinite -- no
block range truncates it, and what a boxcar of range `b` discards is
`sum_{n>b} 1`, which diverges for every `b`. The scattering self-energy is what
makes a spatial truncation meaningful at all.

The useful consequence: the pole census already reports `gamma` per mode on
every bed it has run, so the required band follows from a measurement already
taken, given the harmonic group velocity.

## Measured on Si (`sichk_base`, 2026-08-15)

`phonon/studies/_band_range_report.py` over the stored dynamical matrix: 81
transverse q (9x9), 6 branches each, 486 (q, branch) pairs, all coupled along
transport. The dispersion checks out -- omega spans 0 to 15.153 THz against a
bulk Si band top near 15.5, with three acoustic branches at zero and three
optical at the top at transverse Gamma.

| branch-max abs(v_g) [cells*THz] | min | p25 | med | p75 | max |
|---|---|---|---|---|---|
| | 0.0245 | 0.488 | 0.967 | 1.52 | 4.61 |

At `gamma = 0.16 THz`, the converged Si census median half width
(`pole_sector_observations.md` Sec. 13):

| range xi [cells] | min | p25 | med | p75 | max |
|---|---|---|---|---|---|
| | 0.153 | 3.05 | 6.05 | 9.48 | 28.8 |

| `sse_g_band` | branches reaching further than the band | median branch keeps |
|---|---|---|
| 1 | 99.6 % | 84.8 % |
| 2 | 96.3 % | 71.8 % |
| 3 | 77.6 % | 60.9 % |

"Keeps" is `exp(-b/xi)`: the fraction of the mode the truncation was supposed to
remove and did not. At the production `g_band = 3` the median branch is cut
while 61 % of it is still there.

## Measured on CNT (3,3) -- the bed the bracket belongs to

`cluster/cnt_cal/dynamical_matrix.mat`, transport along z, Gamma-only (a CNT has
no transverse periodicity), 36 DOF per cell = 12 atoms. That the converged L4
census counts 144 modes = 4 x 36 is what confirms the cell pairs with the
linewidths quoted below.

Group velocities are five times Si's:

| branch-max abs(v_g) [cells*THz] | min | p25 | med | p75 | max |
|---|---|---|---|---|---|
| | 2.86 | 4.45 | 5.16 | 6.82 | 12.3 |

The converged `cnt33_L4_linear` widths (`cnt_observations.md` Sec. 6) run
`gamma/dw` = 1.573 (min) to 6.438 (median) at `dw = 0.3056 THz`, i.e.
gamma = 0.481 to 1.97 THz. `Gamma_tot` there is the HALF width despite the
capital -- `_resonance_gain_study.py` computes it as `half_width_at`,
`-Im Sigma(Omega)/(2 Omega)` -- so no factor of two is owed.

| gamma [THz] | range xi [cells], min/med/max | band 3: branches past the band | median branch keeps |
|---|---|---|---|
| 0.481 (narrowest mode) | 5.9 / 10.7 / 25.5 | 100 % | 75.6 % |
| 1.97 (median mode) | 1.5 / 2.6 / 6.2 | 36 % | 31.8 % |

So the answer depends on which mode is asked about, and the split is the
interesting part. The TYPICAL CNT mode has a range near 2.6 cells and a band of
3 is roughly adequate for it. The LONG-LIVED modes -- the narrow ones, which are
also the ones that carry heat furthest -- reach 6 to 25 cells, and no band
between 1 and 3 touches them.

That lines up with the ladder's own numbers more closely than it was set up to.
The two arms of the bracket are a band-3 boxcar and a band-1 taper. At the
narrow end of the distribution the band-3 arm keeps 76 % of what it should have
removed and the band-1 arm keeps 91 %: the arms differ precisely in how much
long-range coherence survives, which is what a bracket between an upper bound
"contaminated by non-causal gain" and a lower bound with "halved coherence"
describes. And the series converges to seven cells and brackets from sixteen,
with a narrow-mode range of about eleven cells sitting between.

It also predicts the direction of the anomaly. The lead current on the band-3
arm GROWS with length, 38.4 -> 44.1 -> 56.2 from L16 to L32
(`gpu_campaign_2026-07.md` Sec. 7). If the band cuts modes whose range exceeds
it, the fraction of each mode wrongly retained grows as the device lengthens
past the band, which is the direction observed rather than the opposite one.

## What that does and does not say

It is consistent with the phenomenology it was meant to explain. A range of 6 to
29 cells means a truncation is harmless on a device shorter than that and starts
to bite beyond it -- and the CNT series converges to seven cells and brackets its
answer by a factor 2.2 from sixteen (`document/src/results/64_gband.tex`). The
orders line up without being tuned to.

Four things it does not establish.

* `gamma = 0.16` is the census MEDIAN applied uniformly. Real widths vary by
  mode, and pairing a per-mode `v_g` with one global `gamma` is an
  approximation, not a per-mode range.
* The dispersion is harmonic. Under SCBA the bands themselves shift, and the
  proposal's Eq. (144) substitution is exactly the statement that they do.
* `xi = v_g/gamma` identifies a range with a lifetime, which is a weak-damping
  statement; it loosens to about a percent once a mode decays within a few
  cells.
* The CNT numbers pair one bed's harmonic dispersion with another run's
  linewidth distribution. Same material and same cell -- 144 = 4 x 36 -- but
  not the same job, and the widths come from a fitted spectral function rather
  than from the pole solve.
* None of this is a correction to the transport numbers. It says a truncation
  is active where it was assumed harmless; it does not say by how much the
  answer moves, which needs the modal representation actually built.

## A correction worth recording

The first version of this measurement was wrong and its numbers should not be
quoted. It read the stored keys `[nx, ny, nz]` as a transport offset plus two
transverse MOMENTUM indices; they are real-space cell offsets on all three
axes, which is settled by `cm_channel.py` reading the same file and summing
over the transverse offsets to reach Gamma. A transverse momentum needs a
Fourier sum over `ny, nz` before the transport dispersion can be formed.

Compounding it, the key regex required the second and third indices to be
non-negative, so it silently dropped every negative transverse offset -- a
third of the file. The result was 25 "q points", three of them reporting zero
transport coupling, and a claimed median range of 0.57 cells: the opposite
conclusion, from a third of the data read under the wrong convention.

The tell was in the output and was missed on the first pass: that dispersion had
NO acoustic branch reaching zero at Gamma, which no phonon dispersion of a
translationally invariant crystal can lack.

---

## Phase 7: the distant blocks are generated, not stored (2026-08-15)

The acceptance test the proposal asks for -- "initially use it only to
reconstruct distant blocks; do not alter the SCBA kernel; validate against
exact distant G_ij" -- on a 2-DOF cell with an invertible inter-cell coupling.

Fitting `G(n) = V diag(lambda^n) C` from `n = 1` and `n = 2` ONLY, and then
predicting every block out to `n = 12`:

| n | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| rel. error | 1e-15 | 5e-15 | 2e-14 | 4e-14 | 6e-14 |

Roundoff over nine distances that were never fitted, across which the blocks
fall by more than two orders. That is Eq. (158) exactly: two mode vectors and
two coefficient rows -- ten numbers -- reproduce what would otherwise be one
dense block per distance.

The rank is not a tuning parameter. Dropping to one mode fails at 1e-3 on the
very blocks it was fitted to, so `r` is the number of decaying modes and not a
knob. And a range of blocks sums in closed form (Eq. 160 at the matrix level),
so a long-range sum costs `r` geometric series and never materialises what it
runs over -- checked to n = 200.

Two traps in building this, recorded because both produce plausible output:

* A rank-deficient inter-cell coupling makes the pencil degenerate. The first
  bed used a `D_01` with one nonzero entry; its roots collapsed to 0 and 173
  and the mode count was wrong.
* The quadrature reference must be a PERIODIC trapezoid. Using `linspace` with
  both endpoints double counts, and the reference then stops decaying at 5e-6
  -- which looks like a Green function reaching a floor and is arithmetic. The
  test now asserts its own reference is converged, 1024 against 4096 nodes.

## On real device cells (2026-08-15)

The same reconstruction against the stored dynamical matrices, at 1.05x the
band top so the quadrature reference is regular.

| bed | DOF | decaying modes | abs(lambda) range | rel. err n=1 / 3 / 8 |
|---|---|---|---|---|
| CNT (3,3), Gamma | 36 | 36 | 7.6e-05 .. 0.145 | 1.8e-15 / 5.5e-14 / 2.5e-10 |
| Si film, transverse Gamma | 6 | 6 | 9.2e-03 .. 0.319 | 9.2e-16 / 1.3e-14 / 1.2e-12 |

The growth with `n` is the REFERENCE, not the reconstruction: the CNT
quadrature self-converges to 6.5e-10 at n=8 and only 1.2e-06 at n=12 (4096
against 16384 nodes), because `|G(12)|` is 6.8e-13 and the integral is chasing
its own floor. Checked rather than assumed.

### The rank and the fit anchor are one choice, not two

Truncating the mode set looked free and is not. Modes with `|lambda| = 1e-3`
contribute `1e-9` by three cells, yet dropping them costs `1e-4` -- because the
fit was anchored at `n = 1, 2` where they are still present in the data, so
their weight is pushed onto the survivors. Moving the anchor past their range
recovers it. On CNT at rank 22 of 36:

| fit anchor | rel. err at n = 5 | at n = 8 |
|---|---|---|
| n = 1, 2 | 1.2e-02 | 1.1e-02 |
| n = 3, 4 | 6.0e-05 | 3.0e-05 |
| n = 5, 6 | 2.1e-07 | 1.4e-07 |

It cuts the other way too. A fit anchored far out cannot determine the
coefficient of a mode that has already decayed there -- at `n = 7` a mode with
`|lambda| = 1e-3` contributes `1e-21`, so the design carries no information
about it and SHORT-range blocks degrade even at full rank.

So the anchor is not a stability knob. It selects the window of distances the
representation is valid on, and the rank follows from it: keep the modes alive
at the anchor, drop the rest, and use the result only at or beyond it. The pole
sector reached the same conclusion for its own local model, where `_fit_anchor`
had to be pinned per candidate.

Practical consequence for Phase 8: the band-truncated blocks that need
replacing are the DISTANT ones, so the anchor belongs at the band edge, and the
rank needed there is 22 of 36 on CNT rather than all of them.

---

## Phase 8: the ring sees what the band removes (2026-08-15)

The proposal's Phase 8 is "use factorized G_S inside the FC3 ring", with the
dense-vertex version required correct first. This is that version, at the
kernel level: three rings differing ONLY in the spatial legs, on a 7-cell
chain with a dense nearest-neighbour cubic vertex.

The frequency grid forces the bed. A ring is a convolution, so `Sigma(Omega)`
needs `G` at `omega` and at `Omega - omega` and the grid has to start at zero;
an exact `eta = 0` reference needs the grid to avoid the band. A GAPPED chain
satisfies both -- an on-site pinning puts the band at
`[w0, sqrt(w0^2 + 4 k_s)]` and the grid sits below `w0`, where the
Brillouin-zone integrand never vanishes. The ungapped chain used earlier
cannot: its acoustic branch reaches zero, so any grid starting at zero runs
through the band.

Green-function ranges 2.0 to 4.6 cells across the grid:

| `g_band` | boxcar error | modal completion | ratio |
|---|---|---|---|
| 1 | 3.18e-01 | 1.8e-16 | 1.8e+15 |
| 2 | 1.40e-01 | 1.1e-16 | 1.3e+15 |
| 3 | 7.14e-02 | 6.8e-17 | 1.1e+15 |

The boxcar gets the self-energy wrong by 32 % at band 1 and still 7 % at
band 3. The completion is exact to roundoff at every band, and it beats
WIDENING the boxcar by a block -- which costs a whole extra block per cell pair
where the completion costs one root and one anchor block.

The error also grows with the range of `G`, which is the mechanism behind a
band ladder that converges on short devices and brackets on long ones.

Not a production claim. This is a 1-DOF synthetic chain with a random dense
vertex, so 32 % is a property of this bed and not of CNT; what transfers is
that the completion is exact and the mechanism is the one measured on the real
cells above. Wiring it into `SigmaPhononPhonon` behind a flag, against the
`set_pole_channel` / `set_cm_channel` seams, is the next step and the first one
that would move a production number.
