# Re-runs the report would benefit from

Each item is carried as a `\todo[inline]` at the point in `document/` where its
number would go. Ordered by what the report gains, not by cost.

## 1. The third least-squares MoS2 rung

**Gains:** the degree of freedom. The two-point least-squares fit gives
`kappa_bulk = 1.1395 [1.1174, 1.1622] W/m/K` and `R_c = 10.159 m2K/GW` with
zero degrees of freedom, so it cannot test its own linearity -- and on the
sparsifying ladder the third point moved `kappa_bulk` by 17 % (1.868 -> 2.193).
Until it exists, 1.14 is a two-point number.

**Blocked on:** cost, not method. `lsM6` needs ~9.5 min/iteration on 16 nodes
and of order 65 iterations, ~165 node-hours against ~96 remaining under the
400 cap. See `sse_memory_scaling.md`.

**Cheaper routes, in order:** (a) finish `lsM4` to the 1e-3 the shorter rungs
reach, which tightens the existing two-point bar; (b) flatten the 5x5
transverse mesh to one axis so `q_distributed` applies, which divides both the
perm cache and the legs; (c) then `lsM6`.

## 1b. RESOLVED -- the dual-grid ladder and the PSD cutoff taper

Both were run (`gl_*`, 12 jobs; `tap*`, 7 jobs) and neither was written up.
Both are now in the report: the auxiliary grid is the expensive and
unconverged axis (results §3.4), and a PSD cutoff mask restores positivity
without rescuing the run (results §3.1). See `report_removed.md`.

## 2. The static loop and tadpole, without broadening

**Gains:** turns results §5.2 from a ratio into values. The sweep behind
`2.4 : 1 : 1.1` (bubble : loop : tadpole on the CNT at 300 K) and the soft-wire
comparison was run on the dense solver at `eta_factor 0.5`.

**Blocked on:** nothing. The production solver computes both terms.

## 3. The nanowire ballistic curves, without broadening

**Gains:** removes the one figure in the main chapter whose underlying sweep
carries a broadening the rest does not (`r2_ballistic_wires`). The ratio
(1.558-1.709, 1.643 at 300 K) and the shape are robust; the absolute
conductance is not fully.

**Blocked on:** the third-order inputs are not on the laptop
(`fc3_sinw100_d5a_sc4_vasp` and friends). `phonon/studies/ballistic.py` now
takes `--eta-factor 0` for whoever has them.

## 4. DONE -- `srtio3_rattle_renorm`, regenerated

The blocker was never missing data. The committed `fcp.fcp` holds the fit; what
was absent is the `fc3.hdf5` the loader resolves, which the reap normally writes
beside it. Two obstacles, both now removed:

* the `.fcp` files were pickled by a numpy whose `_frombuffer` takes a fifth
  argument -- the axis permutation for arrays stored in `'K'` (keep-layout)
  order -- and numpy 2.1's takes four, so `ForceConstantPotential.read` died
  with a TypeError. `phonon/scripts/derive_fc3_from_fcp.py` supplies the
  five-argument form and checks its inverse against a synthetic non-contiguous
  array before reading anything;
* the supercell is rebuilt with the pipeline's own `_build_supercell` +
  `structure_to_ase`, so atom ordering matches the original fit.

The derived tensors are committed beside the fits (198 KB each) and pass their
gates: FC2 ASR 7.1e-15 / 3.6e-15, FC2 pair symmetry exactly 0, FC3 residual
identical across all three axes (so it is the fit residual, not an axis
artefact). The figure then reproduces both numbers the report quotes -- the AFD
mode at R is -55.2i cm^-1 on the small-rattle fit against the quoted 55i, and
+38.8 cm^-1 on the large one against the quoted +39.

`make_all.py` now regenerates 23 of 23 referenced figures; this was the last
one outstanding, and it was also the last main-chapter figure still at
pre-style-pass width (382.8 x 267.6 pt, now 300.1 x 209.7).

## 5. DONE -- the mixer-campaign figure

Recorded here as a correction. An earlier note in this file said the campaign
panels "exist as archived PDFs with no surviving generator". That was wrong:
`phonon/studies/_campaign_figures.py` generates them from
`studies/out/anderson_test/mixer_campaign_{L2,d5a_v2}/<scheme>/run.log`, and
that data is on disk. Results section 4.5 now carries
`r4_mixer_campaign`, built by a generator under the make_all gate. It shows
the inversion the section claims: on the nanotube plain linear reaches
9.6e-04 while Anderson plateaus at 1.1e-01; on the nanowire Anderson is the
worst of four at 7.9e-03 and RRE the best at 3.3e-04.

## 6. The long-chain band ladder, `b_G = 4, 5`

**Gains:** closes the factor-2.2 bracket of results §3.2. At 16 cells and
beyond the boxcar is an upper bracket carrying non-causal gain and the taper a
lower bracket with halved coherence, and neither is a converged-in-band
transport result.

**Blocked on:** nothing. Mechanical with the GPU machinery.

## 7. Bulk silicon conductivity from this solver in the diffusive limit

**Gains:** the end-to-end test that has never been run, and the only thing that
separates what the force constants cost from what the transport method costs.
Currently the largest unattributed difference in the work: bulk Si sits 25-30 %
below experiment near 300 K, the vertex is exact against phono3py to 1.1e-13,
and the Boltzmann result from the same force constants is ~110 W/mK.

**Blocked on:** device size for a diffusive limit.

## 8. The spectral sector at low temperature

**Gains:** decides whether the construction of `70_spectral_sector.tex` is
useful or merely correct. At 300 K neither system has a population that is
simultaneously unresolvable and isolated: self-consistency broadens the lines
by 4.5x into the grid's reach while raising the median width-to-spacing ratio
from 0.75 to 1.52.

**Blocked on:** choosing a bed. The widths must not grow.

## 9. `QX_POLE_PSD=1` in production

**Gains:** the positivity gate exists, is wired, is cheap, and has run in four
short debug jobs (`mos2psd*`, 1.32 nh total) and nowhere else. Every production run should carry
it, since it is the diagnostic that turns a divergence into a statement about
which leg lost positivity.

**Blocked on:** nothing. One environment variable.

## 10. Wide code lines in the package appendices

**Gains:** typography. Building the document (198 pages, lualatex + biber,
clean: 0 errors, 0 undefined references, 0 undefined citations) leaves 77
overfull hboxes, 43 of them over 20pt and the worst at 240pt. They are almost
entirely long dotted Python identifiers and file paths in `input_gen.tex`,
`finite_analysis.tex` and `phonon_solver.tex` -- code documentation that has
no break opportunity and runs into the margin. The main chapters are clean;
their four long strings now carry explicit `\allowbreak` points.

**Blocked on:** finding a remedy that does not break the build. Two were
tried and reverted: making `.` `_` `/` active inside `\texttt` (conflicts
with the other packages -- 50 errors) and `\usepackage[htt]{hyphenat}`
(halves the count, to 38 boxes and 14 over 20pt, but raises 34 "Improper
discretionary list" errors against the class's LuaLaTeX font setup). The
options not yet tried are `seqsplit` applied selectively, or simply adding
`\allowbreak` to the appendix identifiers the way the main chapters now do,
which is mechanical but touches on the order of a hundred sites.

## 11. Separate the open-boundary solver from the Si fine-grid divergence

**Gains:** closes the one candidate results section 4.4 leaves open. At
`ne = 8001` on `si4x2` the spectral surface Green's function misses its own
defining recursion by 65 % (relative 6.5e-01), which is exactly the regime
where grid points crowd the band edges and a contour-based nonlinear
eigenvalue solve struggles. Whether that drives the `nf^4.8` divergence or
merely accompanies it is not established.

**Partly answered (2026-08-16).** The two coarse-grid controls are recorded in
`pole_sector_observations.md`: at `ne = 141` `si4x2` runs clean under the
spectral OBC (0 warnings, O(1) residuals, heat 12.17), so the divergence is a
property of the GRID and not of the 2x2 blocking; and `sancho-rubio` returns
NaN on this bed at BOTH spacings, so it fails unconditionally and cannot serve
as the independent arm. Results §4.3 now says this.

**Still blocked on:** an arm that works at both spacings. Varying
`nevp_solver` (`beyn` / `full`) at fixed `obc.algorithm = spectral` is the
remaining lever; `QX_OBC_ALG` / `QX_NEVP` and
`phonon/studies/engine/obc_probe.sh` are in place. The original probe's logs
are not on the analysis machine.
