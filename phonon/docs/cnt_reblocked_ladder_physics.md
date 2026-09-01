# CNT reblocked ladder physics atlas

**Status:** generated from the completed 2026-09-01 ladder continuations.

The reproducible generator is
`phonon/scripts/figures/cnt_reblocked_ladder_physics.py`. Its compact input is
`phonon/scripts/data/cnt_reblocked_ladder_physics.npz`; the PNG, PDF, CSV and
JSON outputs are written to
`phonon/studies/out/fig/cnt_reblocked_ladder_physics/`.

## Inputs and scope

The length ladder uses two primitive cells per solver slab, the full stored
FC3 support, `sse_g_band = 3`, a 161-point 0 to 55 THz grid, zero artificial
broadening and linear mixing 0.1. The matched ballistic spectrum is `c16-ball`
on the same grid. The complete deep spectra are L16, L24, L32, L64 and L128.
L48 reached iteration 222 before the job failed, so its integrated current is
included as a live state but it is omitted from every spectral plot.

The linewidth and mode-coupling panels use the independently distilled
`resonance_gain_distilled.npz` data. The L4 state supplies 144 resolved device
linewidths. The L2 fixed point supplies the mode-to-mode partial-linewidth
matrix. Its partner mode is summed over, so the matrix identifies source and
receiving frequency sectors but not a unique three-mode decay process.

The causal reference uses the separate converged, one-cell, retarded-FFT CNT
bubble snapshots. It is not joined to the half-retarded length curve. It is an
anchor for the frequency shift, linewidth and temperature dependence that the
half-retarded ladder cannot provide consistently.

## Figure set

| Figure | Physics content |
|---|---|
| `ladder_transport` | Conductance, ballistic ratio, apparent MFP, residual and lead balance |
| `spectral_transport` | Ballistic and anharmonic transmission, spectral heat current, suppression and cumulative current at L16, L64 and L128 |
| `spectral_length_evolution` | Frequency-resolved attenuation over the full ladder, heat-current frequency sectors and current quantiles |
| `local_spectral_state` | Per-stored-DOF LDOS, cumulative LDOS, spectral temperature and occupation at representative lengths |
| `mode_linewidths` | Contact and anharmonic widths, grid resolution, anharmonic width share and spectral-fit parity |
| `mode_to_mode_scattering` | Full partial-linewidth matrix, frequency-sector transfer, source strength and dominant links |
| `causal_fft_reference` | Causal ballistic versus anharmonic current, modal width and shift, and temperature dependence |

## Length dependence

The matched ballistic conductance is 1.56784 nW/K per tube. The deep ladder is
strictly decreasing over every available point:

| cells | length (nm) | G (nW/K) | G/Gball | residual | lead balance | state |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 3.935 | 0.76569 | 0.4884 | 6.49e-8 | 2.41e-4 | final snapshot |
| 24 | 5.903 | 0.69812 | 0.4453 | 2.26e-7 | 5.10e-4 | final snapshot |
| 32 | 7.870 | 0.64952 | 0.4143 | 7.38e-8 | 6.75e-4 | final snapshot |
| 48 | 11.806 | 0.58247 | 0.3715 | 1.76e-6 | 2.39e-3 | live iteration 222 |
| 64 | 15.741 | 0.53467 | 0.3410 | 1.75e-6 | 3.48e-3 | final snapshot |
| 128 | 31.482 | 0.46326 | 0.2955 | 9.01e-6 | 6.46e-3 | final snapshot |

The first `1e-3` stopping points increasingly overestimate the current. The
deep change is 1.6 percent at L16, 6.9 percent at L32, 19.2 percent at L64 and
25.5 percent at L128. The earlier plateau was therefore premature stopping,
not length convergence.

A single ballistic-diffusive fit is not supported. Applying
`G/Gball = 1/(1 + L/lambda)` point by point gives an apparent lambda of 3.76,
4.74, 5.57, 6.98, 8.15 and 13.20 nm from L16 through L128. A material MFP
should not grow by a factor 3.5 with the length used to infer it. This can
reflect a broad mode-dependent MFP distribution, remaining fixed-point drift,
or the non-causal half-retarded approximation. It must not be reported as one
intrinsic CNT MFP.

## Spectral transport

The 7 to 20 THz sector carries the largest share at every complete deep point,
but its fraction falls from 53.7 percent at L16 to 46.6 percent at L128. The
0 to 7 THz fraction rises from 25.7 to 36.9 percent. The 20 to 35 THz fraction
falls from 17.7 to 14.1 percent, while frequencies above 35 THz carry only
2.4 to 2.9 percent.

The median heat-current frequency moves from 11.28 THz at L16 to 9.81 THz at
L128. The 90 percent frequency moves from 25.53 to 23.69 THz. The length
dependence is therefore frequency selective: higher-frequency optical current
is removed faster, leaving a progressively larger low-frequency share.

The representative LDOS and occupation panels average over the DOF stored by
rank 0. The distributed runs save only one block-rank slice of the diagonal
Green function, so these panels are valid frequency samples but not full
spatial profiles. The occupation stays close to the 300 K Bose function and
the spectral temperature stays near the 295 to 305 K contact window.

## Linewidth and mode coupling

The L4 linewidth distribution is resolved on the 0.3056 THz grid. The minimum,
1st percentile, median and maximum `Gamma_tot / df` are 1.57, 1.87, 6.44 and
150.4. No mode is narrower than one grid cell. The median anharmonic share of
the total width is 70.9 percent, with strong mode-to-mode variation.

The source-frequency decomposition is structured rather than uniform. For a
receiving mode in 7 to 20 THz, 77.9 percent of the summed source weight comes
from the same low sector. The 20 to 35 THz receiving sector draws 40.4 percent
from low, 42.5 percent from mid and 17.2 percent from high modes. The 35 to
50 THz receiving sector draws 58.0 percent from high, 25.7 percent from mid
and 16.3 percent from low modes. Low and high modes are mainly self-sector
coupled, while the middle sector is strongly mixed.

## Causal reference

The separate retarded-FFT point is converged at 300 K and gives
`Ganh/Gball = 0.7411`. Its on-shell bubble HWHM ranges from 0.18 to 12.59 THz,
with a median 2.01 THz. The real bubble shifts representative modes by -2.97
to +3.48 THz, which is precisely the physics omitted by the ladder's half
reconstruction. In the causal temperature sweep the conductance ratio falls
from 0.938 at 30 K to 0.741 at 300 K and 0.713 at 600 K.

## Interpretation limits

Every deep ladder endpoint misses the requested `1e-8` self-energy gate, and
L48 has no final `run.npz`. More importantly, the ladder uses the non-causal
half-retarded reconstruction. Its monotone attenuation and spectral trends are
useful diagnostics, but the conductance curve and apparent MFP are not final
material predictions. A causal retarded-FFT ladder that reaches the fixed
point is still required for that claim.
