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
