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

## The 8-layer point: blocked, and the reason is not the budget

Asked to run it, I traced what it needs before launching. Two things came out,
and the second is the important one.

### The build recipe is recoverable -- and it was nearly lost

The 8-layer device inputs do not exist and must be built. The reap they have to
be built from is **not on either cluster**: `cluster/mos2_film_reap` is absent
from daint entirely and from tortin's scratch, and neither surviving MoS2 film
reap on tortin (`mos2_film_reap_scp`, `mos2_film_reap_o4`) reproduces the
ladder's dynamical matrix. It survives only in the laptop's `cluster/` tree.

Identified by measurement rather than by name. Rebuilding `H_00`/`H_01` at
`nk = 5` from each candidate and comparing against the stored
`dynamical_matrix.mat` over all 75 real-space blocks:

| reap | fit | worst `|dD|` vs the ladder |
|---|---|---|
| `cluster/mos2_film_reap` | **ardr** | **0.0000e+00** (bit-exact, 75/75) |
| `cluster/mos2_film_reap_ls` | least-squares | 1.04e+00 (rel 9.1e-03) |

So the recipe is `build_inputs.py --system mos2film --nslabs N --nk 5` against
`cluster/mos2_film_reap`, then a re-block to 2 blocks. That is now recorded;
before this it existed nowhere.

Both reaps were copied to daint on 2026-08-15 (md5-verified) so they are no
longer single-copy: `cluster/mos2_film_reap` (ARDR, 8.6 MB) and
`cluster/mos2_film_reap_ls` (least squares, 11.5 MB).

### The ladder's vertex has no coupling across the van der Waals gap

The reap that reproduces the ladder is the **ARDR** fit, and ARDR is a
sparsifying regression that pruned MoS2's cross-gap third-order parameters to
exact zero even though the 3.528 A S-S distance is inside the 4.0 A cutoff.
That pruning propagates all the way into the device inputs. Measured on the
stored `fc3_blocks.hdf5` of every rung:

| rung | vertex blocks | off-diagonal | cross-layer `|Phi|` weight inside a block |
|---|---|---|---|
| `mos2f2dense` (cvM2) | 2 | 0 | -- (one layer per block) |
| `mos2f4dense` (cvM4) | 2 | 0 | **0.000000 %** |
| `mos2f6x3` (cvM6) | 2 | 0 | **0.000000 %** |

100.000000 % of the three-phonon weight is intra-layer, at every rung. `fc2`
across the gap is nonzero (harmonic interlayer transport works), so in this
model heat crosses the van der Waals gap **purely harmonically**, and every
anharmonic scattering event happens inside a layer.

The least-squares refit of the same 40-structure data does not do this. Summing
`|Phi|` over each reference atom's whole neighbourhood and splitting by layer:

| reap | cross-layer weight | Mo | each of the four S |
|---|---|---|---|
| `mos2_film_reap` (ardr) | **0.000000 %** | 0 | 0 |
| `mos2_film_reap_ls` (least squares) | **1.185434 %** | 0 | **2.272335 %** |

The cross-gap coupling sits entirely on the four gap-facing sulfurs and none of
it on the molybdenums, which is where the geometry says it should be. The ARDR
zero is a regression artefact, not a physical statement.

That is a first-order caveat on `kappa_bulk = 2.193 W/m/K`. It is a real number
for the model that was run, and the linearity test stands, but it is not a
measurement of MoS2 cross-plane *anharmonic* transport, because the gap-crossing
anharmonic channel is identically absent rather than small.

An 8-layer point built the same way inherits this exactly. It would buy a second
test of linearity for a model missing that channel.

### What to run instead, and what it costs

The least-squares reap keeps the cross-gap couplings (`cross_frob` 21.9 -> 18.8
across a 56 -> 80 structure extension; ARDR's is exactly 0, with CV force error
23-35 % worse) and is the fit the thesis calls production. Rebuilding the ladder
on `mos2_film_reap_ls` is what produces a defensible cross-plane number.

Neither option fits the current budget:

| option | builds | SCBA | node-hours |
|---|---|---|---|
| 8th-layer point on the ARDR ladder | 1 | cold + warm at ~16 nodes | ~25-40 |
| rebuild the ladder on least squares | 3-4 | 3-4 cold + warm pairs | ~60-100 |

The daint ledger stands at 276.09/300, so **23.91 nh remain** and neither fits.
The ledger charges nodes x walltime AT SUBMISSION, and `daint.py status` reports
CSCS's actual consumption at **197.22 nh** -- an over-charge of 78.87 nh, which
is structural and expected. The documented remedy is a
`Running total from here: **N nh**` line, applied twice before (2026-08-08 at
103.00, 2026-08-10 at 126.0), both times by Paul. Applying it again would free
that 78.87 nh. That is a call for Paul, not something to do silently, so nothing
has been launched.

## The least-squares rebuild: inputs, and the gates they passed

Built 2026-08-15. One source device at one cell per block, then re-blocked --
`reblock_device.py` merges the q-fold rather than recomputing it, so the
expensive object is produced once:

    build_inputs.py --system mos2film --nslabs 8 --nk 5 \
        --fc3-subdir ../cluster/mos2_film_reap_ls --out cluster/mos2f8_ls_x1
    reblock_device.py --src cluster/mos2f8_ls_x1 --cells N --per-block N/2 \
        --out cluster/lsM{N}

The build had to run on the laptop: daint's venv has no `phonopy` (the job died
in seconds) and tortin had no node idle enough to claim under the sharing
policy. Peak 5.5 GB against 51 GB free, 8 of 16 cores, threads pinned; the fold
took about ninety seconds.

Harmonic gates on the source: IDFT round-trip `1.71e-15`, acoustic modes at
Gamma exactly zero, `||H00(G)|| = 454.5 THz^2` against the ARDR build's `454.4`
-- the same crystal. `||H01(G)||` is 1.9 against 3.0, which is the interlayer
coupling the two fits disagree about and the same disagreement the rigid-layer
modes show.

Re-block gates, on every rung: slab translational equivalence OK, FC2 re-block
exact, fc3 merge exact (the dense operator is unchanged by the re-partition).
Worth noting because `reblock_device.py`'s docstring justifies replication with
"the fit prunes the vdW-gap fc3 to exact zero" -- true for ARDR, not for LS, so
the assertion was doing real work here and it passed.

| rung | cells | dof/block | span | cutoff | fill | qfold | vertex blocks | cross-layer inside a block |
|---|---|---|---|---|---|---|---|---|
| `lsM2` | 2 | 18 | 21.57 A | 30 A | 1.000 | 0.44 GB | **8** (2 diag + 6 off) | n/a, 1 cell/block |
| `lsM4` | 4 | 36 | 46.16 A | 48 A | 1.000 | 3.48 GB | **8** (2 diag + 6 off) | **0.520591 %** |
| `lsM6` | 6 | 54 | 70.75 A | 72 A | 1.000 | 11.73 GB | **8** (2 diag + 6 off) | **0.692919 %** |

Against the ARDR ladder's 2 blocks, all diagonal, 0.000000 % cross-layer. The
channel that was identically absent is now present, and the box mask is inactive
on all three.

The eight-block source itself carries 50 vertex blocks over its 8 slabs (8
diagonal + 42 off-diagonal at offsets `(+-1, 0)`, `(0, +-1)`, `(+-1, +-1)`) where
ARDR would have 8. That 6.25x is the cost the correct vertex carries.

## lsM2: the correct vertex does not converge under fixed-point iteration

Job 4490547, 2 nodes, `eta = 0`, `ne = 6001`, anderson depth 2, `QX_MIX=0.1`,
`QX_POLE_PSD=1`. TIMEOUT at 1:00:13 after 19 iterations. Three things it was run
to measure:

**Cost multiplier: 4.0x.** 3.16 min/iteration against the ARDR L2 rung's 47 s,
matching the 8-vs-2 vertex block count.

**It orbits.** The residual never improves on its first value:

| it | 1 | 4 | 7 | 8 | 10 | 15 | 17 |
|---|---|---|---|---|---|---|---|
| residual | 0.110 | 0.234 | 0.176 | **0.568** | 0.120 | 0.114 | 0.221 |

`sigma_best` holds `res = 1.1005e-01` from iteration 1, the best of the run.
That is a limit cycle, not slow convergence.

**Positivity fails in two distinct modes.** From iteration 4:

* persistent and mild at `w[734] = 2.936 THz`, growing -6.2e-05 -> -1.6e-03 and
  then sitting there;
* intermittent and severe at `w[1] = 0.004 THz` -- iterations 7, 17, 18, reaching
  **-2.38e-02** with `g_lesser` at -7.1e-03 -- and each one coincides with a
  residual spike.

**A hypothesis of mine, falsified.** `mos2-ardr-no-cross-gap-fc3` recorded on
2026-08-03 that "without cross-gap fc3 the film SCBA descends; with it, it
orbits". I proposed that this was confounded by H6, since it predates the
box-mask discovery by five days. It is not: this run has `fill = 1.000`, the
mask is inactive, and it orbits anyway. The cross-gap vertex itself is what makes
the fixed point hard, and the 2026-08-03 observation stands on its own.

**REVISED the same evening: the orbit is the mixer, not the vertex.** The same
bed at anderson depth **8** (job 4490605, warm from `sigma_best`) descends to
`6.58e-03` by iteration 37 and has **no positivity violation at any of its 38
iterations** -- `sigma_lesser`, `sigma_greater` and `g_lesser` all sit at
`+0.000e+00` throughout, with only `g_greater`'s constant `-1.374e-14` at bin 0,
the same sub-tolerance harmonic artefact the ARDR runs carried at `-8.755e-11`.

So the depth-2 result is one data point about one mixer setting, not a verdict on
the vertex, and the positivity violations went away with the orbit rather than
being a property of the cross-gap coupling. What survives is narrow: the LS
vertex needs more Anderson history than the ARDR one, which is what a map with
four times the scattering channels should need.

**Newton is not available on this path.** `newton.py::_newton_step` raises
`NotImplementedError: mixing_method='newton' (exact JVP) requires the numpy
backend`, and daint jobs run `QTX_ARRAY_MODULE=cupy` (job 4490794: warmup runs,
then it throws on the first Newton step). Two further traps in its defaults:
`newton_switch_tol = 1e-2` means Newton engages only once the residual reaches
1 % of its first value, so on an orbiting run it never engages and the job is
silently damped Picard; and `newton_max_krylov = 30` costs up to 30 JVPs per
step, each about one map application. None of that matters here because Anderson
works, but it rules Newton out as the fallback for the longer rungs.

## The fixed-length A/B: the least-squares fit lowers cross-plane resistance by 11 %

`lsM2f` (job 4491085) is `lsM2b`'s state finished off at a loose tolerance so it
writes a current -- `lsM2b` itself ran 57 correct iterations and hit the wall
before the output stage, which produced no transport number at all.

| run | vertex | t | J_raw | residual | lead balance | R |
|---|---|---|---|---|---|---|
| `cvM2b` | ARDR, 2 blocks all diagonal | 2.4588 nm | 590.3940 | 5.11e-06 | 1.52e-04 | **13.8161** m2K/GW |
| `lsM2f` | least squares, 8 blocks | 2.4588 nm | 662.2560 | 4.81e-04 | 1.04e-03 | **12.3169** m2K/GW |

Same geometry, same 2-block partition, same `ne = 6001` on the same 24 THz
window, same 5x5 transverse mesh, same 305/295 K, `eta = 0` both, box mask
inactive on both. One variable: which force-constant fit the device was built
from.

**R falls by 10.85 %.**

Two things this is NOT.

It is not "the cross-gap fc3 channel is worth 11 %". The two reaps differ in
`fc2` as well as `fc3` -- `||H01(G)||` is 1.9 against 3.0 THz^2, and the
rigid-layer modes move from 1.39/1.42 to 1.10/1.58 THz. Isolating the
third-order channel alone would need a third build, LS `fc2` with the `fc3`
pruned the ARDR way. What the 11 % measures is the whole difference between the
two fits, and the fit that produces it is the one that matches the measured
rigid-layer frequencies.

It is not a converged-to-converged comparison. `lsM2f` stops at a Sigma residual
of 4.81e-04 where `cvM2b` reached 5.11e-06. The bound that matters is the
conservation one: lead balance 1.04e-03 against 1.52e-04, so the LS current
carries about 0.1 % of imbalance-bounded uncertainty against the ARDR side's
0.015 %. Both are far below the 11 % effect.

`_kappa_z_ladder.py` now refuses to fit `R(t)` across runs at the same
thickness -- polyfit returned `kappa_bulk = 0.3764 W/m/K` from these two points
before the guard, which is a fit to a vertical line and means nothing.

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
