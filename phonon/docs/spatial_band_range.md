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
* This is Si. The CNT linewidths are different and have not been put through
  the same calculation.

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
