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

## 4. `srtio3_rattle_renorm`, regenerated

**Gains:** the one main-chapter figure not regenerated at thesis width in the
2026-08-16 pass, so it is the one that will look different on the page.

**Blocked on:** `fc3_hiphive_srtio3_small_vasp` is not local.

## 5. The long-chain band ladder, `b_G = 4, 5`

**Gains:** closes the factor-2.2 bracket of results §3.2. At 16 cells and
beyond the boxcar is an upper bracket carrying non-causal gain and the taper a
lower bracket with halved coherence, and neither is a converged-in-band
transport result.

**Blocked on:** nothing. Mechanical with the GPU machinery.

## 6. Bulk silicon conductivity from this solver in the diffusive limit

**Gains:** the end-to-end test that has never been run, and the only thing that
separates what the force constants cost from what the transport method costs.
Currently the largest unattributed difference in the work: bulk Si sits 25-30 %
below experiment near 300 K, the vertex is exact against phono3py to 1.1e-13,
and the Boltzmann result from the same force constants is ~110 W/mK.

**Blocked on:** device size for a diffusive limit.

## 7. The spectral sector at low temperature

**Gains:** decides whether the construction of `70_spectral_sector.tex` is
useful or merely correct. At 300 K neither system has a population that is
simultaneously unresolvable and isolated: self-consistency broadens the lines
by 4.5x into the grid's reach while raising the median width-to-spacing ratio
from 0.75 to 1.52.

**Blocked on:** choosing a bed. The widths must not grow.

## 8. `QX_POLE_PSD=1` in production

**Gains:** the positivity gate exists, is wired, is cheap, and has run in three
0.33-node-hour debug jobs and nowhere else. Every production run should carry
it, since it is the diagnostic that turns a divergence into a statement about
which leg lost positivity.

**Blocked on:** nothing. One environment variable.
