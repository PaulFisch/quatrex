# The MoS2 cross-plane ladder, with the box mask off

## The ladder did not need re-running -- it had already been re-run

H6 (`bubble_positivity.md` Sec. 6.8-6.10d) made the whole cross-plane campaign
suspect: `phonon.interaction_cutoff` masks the stored sparsity pattern of every
matrix, `G` included, so any run whose cutoff falls inside the device carries a
non-PSD `G` and a non-PSD `Sigma` and its current is not a physical number.

Whether a given run is affected is decided by geometry alone, before any
physics: the mask is inactive iff the cutoff exceeds the largest
transport-direction separation in the device.
`phonon/studies/_cutoff_mask_audit.py` builds the grid through the solver's own
loader, computes the pattern with `compute_sparsity_pattern`, and reports the
fill fraction. Over the 39 MoS2 configs on disk it splits the campaign cleanly:

| family | cutoff | fill | status |
|---|---|---|---|
| `mos2f3*`, `mos2film_L3_*` (3 layers, span 33.86 A) | 10 A | 0.469 | MASK ACTIVE |
| `mos2f6*` (6 layers, span 70.75 A) | 10 A | 0.252 | MASK ACTIVE |
| `cvM2b` (span 21.57 A) | 30 A | **1.000** | dense |
| `cvM4e` (span 46.16 A) | 48 A | **1.000** | dense |
| `cvM6b` (span 70.75 A) | 72 A | **1.000** | dense |
| `mos2L*conv`, `mos2sood*`, `mos2f4dense` | 30-75 A | 1.000 | dense |

20 of the 39 are masked, and every one of them is from the pre-H6 campaign. The
`cvM*` series **is** the re-run: launched 2026-08-10/11, two days after H6 was
identified, with cutoffs chosen per thickness. The audit and the positivity gate
agree where both have been applied -- `mos2psd10` (fill 0.469) is the run whose
`Sigma` sits at -0.99, `mos2psd40` (fill 1.000) the one that stays at +0.000e+00.

## Three points, and the first test of linearity

The two-point fit of 2026-08-10 had zero degrees of freedom; linearity was
assumed. The third rung exists -- `cvM6b`, 6 layers, converged in 52 iterations
-- and had never been folded in. `phonon/studies/_kappa_z_ladder.py` does the
bridge and the fit, and refuses to fit any run whose mask is active.

| run | t [nm] | J_raw | iterations | residual | R [m2K/GW] |
|---|---|---|---|---|---|
| `cvM2b` | 2.4588 | 590.3940 | 10 | 5.11e-06 | 13.8161 |
| `cvM4e` | 4.9176 | 539.0484 | 38 | 9.16e-06 | 15.1322 |
| `cvM6b` | 7.3764 | 507.9516 | 52 | 9.45e-06 | 16.0586 |

All three at `ne = 6001`, `df = 0.004` THz, 24 THz window, 5x5 q-mesh,
T 305/295 K, `eta = 0`, so every bridge constant is common and cancels in the
slope.

Fitting `R(t) = R_c + t/kappa_bulk`:

* **kappa_bulk = 2.193 W/m/K**
* **R_c = 12.760 m2K/GW**
* residuals **-0.47 %, +0.86 %, -0.40 %**

The residuals are the point. With three points and two parameters there is one
degree of freedom, so this is the first time the assumed linearity has been
tested rather than imposed, and it holds to under one percent. `kappa_bulk` sits
inside the literature c-axis range (~2-4 W/m/K) and `R_c` near the Sood-type
boundary resistance (~10 m2K/GW).

The two-point fit shifts by 17 %: 1.868 W/m/K against 2.193. That is the
expected sensitivity -- `kappa_bulk` comes from differences between R values
that are themselves 80-92 % contact -- and it is why the third point mattered.

`R_c` is **92.4 %** of the resistance at 2 layers, 84.3 % at 4 and 79.5 % at 6,
so `kappa_z,eff` from any single thickness (0.178, 0.325, 0.459 W/m/K) is mostly
interface and is not a material number. Only the slope is bulk.

## What a fourth point would cost

The ladder as it stands cost about 12.2 node-hours: `cvM2b` 2 nodes for 7:52,
`cvM4e` 2 nodes for 1:22:33, `cvM6b` **8 nodes** for 1:08:36. The cost is
superlinear -- the 6-layer rung needed four times the nodes of the 4-layer one
and still took an hour -- so an 8-layer point is a 15-25 node-hour proposition
and needs more than one node, i.e. explicit approval.

With one degree of freedom already spent and the residuals under 1 %, a fourth
point buys a second test of linearity rather than a first one. The larger
uncertainty is elsewhere: two of the three currents differ by 9.5 % and 5.8 %,
so a 1 % DIFFERENTIAL error in any lead current moves `kappa_bulk` by roughly
10 %, while a common-mode error passes through to `R_c` and leaves the slope
alone.

Reproduce with::

    python -m phonon.studies._cutoff_mask_audit cluster/*/quatrex_config.toml
    python -m phonon.studies._kappa_z_ladder cluster/cvM2b cluster/cvM4e cluster/cvM6b
