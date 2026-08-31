# The CNT (3,3) campaign: what it changes, and against what

2026-08-28. Every CNT run in the corpus fails four gates at once, and no
single-variable A/B exists for any of them. This is the campaign that
separates them, on one bed, changing one thing at a time. Companion to
`run_audit_2026-08.md` (which found the gates), `cnt_observations.md` (which
consolidates what the old runs show) and `production_beds.md` (which
prescribes the settings).

## 1. The bed

Sixteen CNT (3,3) transport cells, twelve atoms and 36 DOF each, transport
along z, `nk = [1,1,1]`, 305/295 K, `eta = 0`. The geometry is
`cluster/l16` on Alps, the same one the `l16f-*` family ran on 2026-07-31.
Each arm gets its own directory under `cluster/`, with the geometry symlinked
and its own config, because `scba.py:1375` writes per-iteration dumps into
`output_dir` and two arms sharing a bed would collide.

Measured on the bed's own dynamical matrix rather than taken from a note:

| quantity | value |
|---|---|
| `omega_max` | 46.335 THz |
| `2*omega_max` (bubble support) | 92.67 THz |
| Gamma modes below 1e-3 THz | 3 translations, plus a twist at 0.0265 THz |
| device orbital span along z | 38.12 A |

The twist mode matters for the infrared gate: at 300 K its Bose factor is of
order 200, and at `dw = 0.34375 THz` it sits thirteen spacings below the
first grid point, so it is never resolved and never enters except through the
masked `omega = 0` bin.

## 2. The four gates, and what each arm changes

### Extent

The primary grid stops at 55 THz and the bubble is supported to 92.67, so it
covers **59.4 %** of the support and the Kramers-Kronig integral for
`Re Sigma^R` is truncated. `sse_aux_grid_fmax_thz` exists for exactly this:
it extends the auxiliary bubble grid without adding Dyson solves
(`config.py:1596-1603`). Every tortin and Alps CNT bed leaves it at 0.0, and
the one family that sets it (`_run_cnt33_length_gband.py:33`) uses 88.0,
sized against a stale 43.73 THz band top, which is why `L8_g3` still logs
1.8-2.1 % truncation.

The campaign sets `--aux-fmax 97`, and sets `--aux-dw` to exactly the primary
spacing `55/(nfreq-1)`. The auxiliary grid therefore **extends without
refining**: `grid_audit.md` measures aux density as the unconverged and
expensive axis, carrying +4.4 to +10.2 % against the aux-off bubble, and
folding a refinement into the extent fix would confound the two.

### The Kramers-Kronig half

`l16f-g3`, `l24f-g3`, `l32f-g3` and the whole `cnt33_long_gband3` ladder run
`retarded_method = "half"`. `Sigma^R` then carries no real part at all, so
those runs have linewidths and no frequency shifts. `c16-fft` and everything
after it runs `fft`.

### The box mask -- measured, and it does nothing on CNT

`interaction_cutoff` is the shipped 10.0 A default (`config.py:1442`) on
every CNT run ever performed: `write_config.py` has no flag for it and never
emits the key, so it cannot appear in a config by accident. It feeds
`compute_sparsity_pattern(grid, cutoff, strategy="box")`
(`core/scba.py:112-118`), which masks on the transport-axis separation alone
(`core/utils.py:45-49`).

`c16-cut40` raises it to 40 A, which is 100 % fill on this 38.12 A device,
and changes nothing else. **The two arms are bit-identical.** Twenty
iterations of `rel Sigma^R residual`, lead balance, internal spread and lead
current agree to every printed digit, while the logs confirm the setting took
effect (`Max Interaction Cutoff: 10.0` against `40.0`). The arm was cancelled
once the identity was established.

The reason is that the mask can only cut what the solver stores, and the
solver is block-tridiagonal. What is at risk is therefore the **block band**,
not the device span:

| cells per block | block band `\|dz\|` | 10 A cutoff |
|---|---|---|
| 1 | 3.69 A | never bites |
| 2 | 8.61 A | never bites |
| 3 | 13.53 A | **bites** |
| 4 | 18.45 A | **bites** |

The CNT (3,3) transport period is 2.4595 A, so one block is thin and two are
still inside the default. MoS2's 2H cell is 12.294 A along z, so a single
MoS2 block already exceeds 10 A -- which is why H6 is a MoS2 defect and does
not transfer to CNT, and why the device-span fill fraction (46.1 % at L16,
24.8 % at L32) is the wrong diagnostic despite matching the MoS2 rungs that
diverged. The elements the mask discards on CNT are outside the retained band
and structurally zero.

Two consequences. The anomalous gain of the L16/L24/L32 ladder is **not** a
box-mask artefact, so the attribution in `30_truncation.tex:136-149` to
spatial truncation stands. And the rule to carry forward is stated in blocks,
not in device length: a CNT bed at three or more cells per block must set
`interaction_cutoff` explicitly, and one at two or fewer need not.

### Blocking

No CNT run has ever used more than one transport cell per block. The
32.1/5.4/2.5/0.30 % ladder that motivates the gate was measured on the
synthetic 1-DOF chain of `_spatial_bed.py`, not on a device.
`sse_g_band` is clamped to `n_blocks - 1` at three sites
(`solver.py:195`, `core/scba.py:125`, `sse_phonon_phonon.py:393`), so band 3
needs four blocks while two cells per block needs eight cells; sixteen cells
as eight blocks of two is the first geometry that satisfies both **and**
admits `block_comm_size = 2`.

`cluster/c16x2` was built for this by
`reblock_device.py --src cluster/c16-cut40 --cells 16 --per-block 2 --tdir z`
and passed every gate the tool applies: slab translational equivalence on the
fc3, the FC2 re-block exact on the transverse key, and the fc3 merge exact
(106 primitive to 50 merged blocks), giving 8 blocks x 72 DOF = 576, the same
576 DOF as the 16 x 36 bed. It inherits `interaction_cutoff = 40.0` and the
auxiliary grid from its source, so it changes exactly one variable against
`c16-cut40`.

Two traps worth recording. `--tdir` takes a **letter** --
`reblock_device.py:203` does `"xyz".index(tdir)` -- so the `--tdir 0` in
`cluster/alps/do_reblock.sh` would raise. And the tool's config rewrite is a
literal `/cluster/<name>` string replace, so a source bed outside a path
component named `cluster` would silently emit a config still pointing at the
source.

## 3. The arms

All 16 cells, `eta = 0`, one node of four GH200, `bcs = 1`, `QX_POLE_PSD=1`
and `QX_BBCHECK=1` throughout. Each arm changes one variable against
`c16-kk`.

| arm | change |
|---|---|
| `c16-half` | `retarded=half`, aux off -- the control, reproducing `l16f-g3` |
| `c16-fft` | `retarded=fft`, aux off: KK on, but on a 55 THz grid |
| `c16-kk` | + aux grid to 97 THz -- the corrected bed |
| `c16-cut40` | + `interaction_cutoff = 40.0` |
| `c16-ball` | `QX_BALLISTIC=1` at the same grid -- the Landauer reference |
| `c16-g1`, `c16-g2` | the band ladder at fixed everything else |
| `c16-ne241`, `c16-ne361` | the grid ladder, extent defect removed |
| `c16x2` | 8 blocks x 2 cells |

`sse_g_band` is not emitted by `write_config.py` either and has to travel in
`QX_GBAND` (`run.py:49`); the code default has been 3 since `82761380`.

The driver is `phonon/studies/_run_cnt_campaign.py` (`--stage`, `--prep`,
`--reblock`, `--launch`, dry-run unless `--go`); the reducer is
`phonon/studies/_cnt_campaign_report.py`.

## 4. What the arms are measured against

| observable | standing value | source |
|---|---|---|
| ballistic conductance | 5.54 `g_Q` at 300 K, length-independent to 11 digits | `20_baseline.tex:20-34` |
| `G_anh/G_ball`, L = 2..7 | 0.569 / 0.500 / 0.453 / 0.417 / 0.396 / 0.362 | `30_truncation.tex:126-128` |
| band-3 length ladder | 13.19 / 15.16 / 19.33 at L = 16/24/32, **rising** | `gband_ladder.npz` |
| interior spread, same ladder | 6.3 % / 11.0 % / 17.5 % | same |
| linewidths | 144 modes, narrowest 1.57 spacings, median 6.4 | `50_structure.tex:193-196` |
| sub-2-THz share | the 2 THz mask moves `r` from 0.500 to 0.809 | `40_selfconsistency.tex:213-217` |

The rising current is what the campaign has to explain. A fixed-`dT` device
should not gain with length, and the mask fill falls across exactly the same
ladder.

Two keys must not be read from `run.npz` on this campaign: `iter_heat` and
`iter_sigma_max` are rank-0-local frequency slices, and the arms run on four
ranks. Per-iteration history comes from the log through
`phonon/studies/pipeline.py::parse_scba_trace`. Mode linewidths are not in
`run.npz` at all -- they need a `Sigma` snapshot and a Dyson re-solve
(`_resonance_gain_study.py`), which is why `c16-kk` carries `QX_SAVE_SIGMA`.

## 5. Results

### The control reproduces the archive exactly

`c16-half` converged in **97 iterations** to `lead_current =
38.361263551244036`, residual 9.6263e-04, lead balance 9.5126e-05, interior
spread 6.2745e-02. `l16f-g3`, run 2026-07-31 on eight GPUs, gives 97
iterations and 38.36126355124404. The two agree to fifteen digits on four
GPUs against eight, so the stack decomposition is reduction-order invariant
here and nothing in the code has moved the bed since July. Everything
downstream is comparable to the archive.

Peak GPU memory 7.97 GB of 96, and the run took under fifteen minutes.

### The cutoff does nothing

See Sec. 2. `c16-kk` and `c16-cut40` are bit-identical over twenty
iterations; the arm was cancelled.

### Positivity is violated on the converged control

`QX_POLE_PSD=1` reports, at the converged iterate of `c16-half`,
`sigma_lesser` and `sigma_greater` clean at `+0.000e+00` but
`g_lesser` at `-2.298e-05` (w[2]) and `g_greater` at `-2.017e-05` (w[0]).
The self-energies are positive-semidefinite and the Green's functions are
not, which locates the violation after the Dyson solve rather than in the
bubble. It is small, it sits at the bottom of the grid where the infrared
channel lives, and it is present in a run that meets `sigma_convergence_tol`
-- so it is a property of this bed at `eta = 0`, not of the corrected
settings. `spatial_representation.md` Sec. 19 reports the same thing on the
frozen chain beds, where the untruncated reference itself carries 0.686 of
negative weight.

### The ballistic reference, and two report numbers confirmed

`c16-ball` (`QX_BALLISTIC=1`, same bed, same grid) gives `lead_current =
68.834`, i.e. `I = 23.662` in the archive's integral units, lead balance
1.55e-14 and interior spread 2.8e-14 -- a ballistic run conserves exactly, as
`conservation.tex:169-176` says it should.

Two independent checks pass. In physical units it is **5.522 `g_Q`** at
300 K against the report's **5.54** (`20_baseline.tex:20-34`), computed on a
different length, a different grid and through a different reducer. And
against `units_parity/cnt33_L4` (four cells, `ne = 361`) it gives `I = 23.575`
where this bed at sixteen cells and `ne = 161` gives 23.662, so the ballistic
current is length-independent to 0.4 % across a fourfold change in length,
the residual being the grid rather than the length.

### The anomalous gain, in ratio form

The matched reference is what the length ladder never had. Dividing the
committed `gband_ladder.npz` rows by it:

| L | `I_int` | `r = G/G_ball` | interior spread |
|---|---|---|---|
| 16 | 13.187 | **0.557** | 6.3 % |
| 24 | 15.158 | **0.641** | 11.0 % |
| 32 | 19.327 | **0.817** | 17.5 % |

`r` rises with length. Anharmonic scattering cannot move a device towards its
ballistic limit as it gets longer, and on this trend the ratio would reach 1
before fifty cells. The band-1 Bartlett arm on the same reference goes the
other way -- 0.257, 0.237, 0.233 -- but at 47 %, 72 % and 86 % interior
spread. The two bracket rather than converge, which is what
`bubble_positivity.md:203-214` predicts of a PSD tridiagonal mask against a
boxcar, and neither is a transport result.

Stating it as a ratio is what makes it unambiguous: the earlier form,
"current rises 13.19 to 19.33", could in principle have been a normalisation
that grows with slab count. `G/G_ball` cannot be, because the reference is
measured on the same bed and is length-independent to 0.4 %.

### The Kramers-Kronig half costs a factor of two, and a long plateau

`c16-fft` ran its full 400-iteration cap without reaching `sigma_tol = 1e-3`,
ending at residual 5.30e-02 and `lead_current = 17.95`. `c16-kk` is at
2.32e-02 after 344. Both descend, but through a plateau: the per-40-iteration
residual minimum sits between 6e-2 and 1.5e-1 from roughly iteration 40 to
320 in both arms, and only then falls. The control converged in 97.

So the half-rule converges on this bed and the Kramers-Kronig form does not,
within 400 iterations at linear mixing 0.1. That is the reverse of
`40_selfconsistency.tex:16-27`, which reports the half rule limit-cycling at
O(1e-1) on a three-cell tube under every mixer while the KK form converges in
296 iterations. Whatever the mechanism, it is not length-independent.

The effect on the answer is large and in the physical direction. Against the
same ballistic reference:

| arm | `Sigma^R` | `lead_current` | `r = G/G_ball` |
|---|---|---|---|
| `c16-half` (converged) | half | 38.361 | 0.557 |
| `c16-fft` (residual 5.3e-2) | fft, KK truncated 1.8 % | 17.950 | ~0.261 |
| `c16-kk` (residual 2.3e-2) | fft, KK support complete | 17.658 | ~0.257 |

The real part of `Sigma^R` roughly halves the conductance, and the two fft
arms agree with each other to 1.7 % while differing from the half rule by a
factor of 2.2. The extent fix therefore matters much less than the KK half
itself -- which is worth stating, because the extent is the gate the code
warns about and the KK half is the one it does not.

Both fft numbers are iterates at 1e-2, not fixed points, so the ratio is
indicative rather than final.

### The mixer sweep: the two gates come apart

`40_selfconsistency.tex:233-236` measured the mixers on a TWO-cell tube and
found plain damped linear best with Anderson two orders above it. Repeated on
the sixteen-cell corrected bed, with `QX_MIXMETHOD` and everything else held:

| mixer | iterations | `rel Sigma^R` | lead balance | interior spread |
|---|---|---|---|---|
| linear | 400 | 3.29e-02 | **1.19e-04** | 0.122 |
| anderson | 552 | 8.64e-01 | 7.04e-01 | 1.021 |
| rre | 600 | 6.74e-02 | 1.44e-01 | 0.254 |
| **broyden** | 600 | **7.44e-04** | **2.02e-01** | 0.270 |

Anderson is again the worst, so that part of the report's finding survives the
length. The interesting row is Broyden. The SCBA gates on two conditions
(`scba.py:943-944`): `rel_sigma < sigma_convergence_tol` **and**
`balance < heat_flow_conservation_tol`. Broyden is the only arm to clear the
first -- 7.44e-04 against 1e-3 -- and it misses the second by a factor of
twenty. The run reports NOT CONVERGED, correctly.

That is the behaviour `config.py:518-525` advertises: Broyden, RPM, RRE and
JFNK "can LAND an iteration-UNSTABLE fixed point (Jacobian |lambda|>1) that
damped/Anderson mixing cannot reach". What this bed adds is that the landed
point is not the physical one. Linear mixing, which is nowhere near the
self-energy tolerance, sits a hundred times INSIDE the conservation gate;
Broyden reaches the self-energy tolerance at twenty times outside it. The two
mixers are converging to different things, and only one of them conserves
energy.

Whether they approach a common point is not established: no fft arm has yet
satisfied both gates, so there is no converged fft reference to compare them
against. The half-rule control does satisfy both (9.63e-04 and 9.51e-05), so
a conserving fixed point exists on this device -- for that form of `Sigma^R`.

### Gate (a): two cells per block, measured on CNT for the first time

`c16x2h` is the sixteen-cell device blocked as eight blocks of two cells,
against `c16-half` as sixteen blocks of one. Same cells, same grid, same
`half` rule, same band 3, `eta = 0`. The only other difference is
`interaction_cutoff` 40 against the 10 default, which Sec. 2 measured inert at
this blocking (the block band is 8.61 A).

| | `c16-half` | `c16x2h` |
|---|---|---|
| blocking | 16 x 1 | **8 x 2** |
| iterations to converge | 98 | 101 |
| `rel Sigma^R` | 9.63e-04 | 9.95e-04 |
| lead balance | 9.51e-05 | 5.04e-04 |
| **interior heat spread** | **6.27 %** | **0.05 %** |
| `I_int` | 13.187 | 11.741 |
| `r = G/G_ball` | 0.557 | **0.496** |
| `kappa_eff` | 8.0 W/m/K | 7.1 W/m/K |

Both converge, in essentially the same number of iterations, so this is a
comparison of two converged answers and not of a convergence rate. Two cells
per block make the device **125 times more conserving** -- the interior spread
falls from 6.27 % to 0.05 % -- and lower the current by **11 %**.

This is the first measurement of gate (a) on CNT, and it is the first CNT run
in the corpus with two cells per block. It also bears directly on the
anomalous gain of Sec. 5: the one-cell-per-block answer is 11 % high at
sixteen cells, and its non-conservation grows with length (6.3 %, 11.0 %,
17.5 % at 16, 24, 32 cells) in step with the rising `r`. A spatial-truncation
error that grows with device length and inflates the current is exactly the
shape of the anomaly, and reblocking removes it here.

What this does not yet establish is the length trend at two cells per block.
`r = 0.496` at sixteen cells still sits above the `g_band = 2` family's 0.362
at seven, so the ladder is not yet monotone. The 24- and 32-cell rungs at
`8 x 2` blocking are the measurement that would settle it.

### A rank-locality trap the block-first layout introduces

`gr_diag_imag` and `gl_diag_imag` are gathered over the stack axis but remain
**rank-local in the BLOCK axis**: `c16x2h` at `bcs = 2` stores 288 of the
device's 576 DOF, i.e. the four blocks its block-rank 0 owns. Any post-hoc
LDOS, occupation or local-temperature analysis
(`phonon/postproc/local_observables.py`) on a `bcs > 1` run silently uses half
the device. `_cnt_campaign_report.py` therefore takes the blocking from
`structure.xyz` and not from the array shape. This matters more now than it
did, because Sec. 6 makes block-first the default layout.

### The remaining mixers

`jfnk` and `rpm` both terminate with `SIGN INVERSION` -- `rpm` at an interior
spread of 1.6e+14 -- so neither is usable on this bed. `newton` stopped at 61
iterations with `rel Sigma^R` 2.12e-01 and a clean lead balance of 1.00e-03.
`low_freq_mixing_thz = 2.0` (`c16-kk-lfm`) did **not** help: 2.82e-01 after
400 iterations against linear's 3.29e-02, with a lead balance an order of
magnitude worse (1.16e-02 against 1.19e-04). The infrared marginal mode is
therefore not what holds the fft iteration back.

## 6. GPU settings, measured on this bed

`gpu_campaign_2026-07.md` Sec. 6 ranks the OBC cache first, "up to ~2x
end-to-end at production grids". On this bed it is worth 7.3 %, and the
reason is worth recording. `c16-obc` repeats `c16-half` with
`QX_OBC_MEMO=cache` and `QX_MAXBATCH=100000`:

| | iteration | ring contraction | PhononSolver |
|---|---|---|---|
| `auto` memoizer, `max_batch 512` | 4.393 s | 3.644 s | 0.639 s |
| `cache` memoizer, `max_batch 100000` | **4.072 s** | 3.644 s | **0.331 s** |

The OBC halves, exactly as advertised, but **the ring contraction is 90 % of
the iteration on a 36-DOF block and neither knob touches it**. The residual
trajectory is unchanged to five significant figures over eight iterations, so
the cache is numerically free here: `eta = 0`, fixed leads and
`obc_scattering_contacts = False` are the stated precondition for the mode
(`config.py:713-729`), and it is self-invalidating if they stop holding.

### Block-first beats stack-first on this bed

`gpu_campaign_2026-07.md` Sec. 6 item 2 recommends a "stack-first rank
layout, block axis only for memory", and every CNT run in the corpus uses
`bcs = 1`. Measured here at a fixed four GPUs, sixteen blocks, `ne = 161`:

| `bcs` x stack | iteration | ring | PhononSolver |
|---|---|---|---|
| 1 x 4 | 4.083 s | 3.641 s | 0.352 s |
| 2 x 2 | 3.635 s | 3.447 s | 0.120 s |
| **4 x 1** | **3.439 s** | **2.820 s** | 0.152 s |

Block-first is **15.8 % faster end to end and 22.5 % faster on the ring**, and
all three produce identical residual trajectories to five significant figures,
so this is a decomposition choice and not a numerical one. The bed permits it
because `sse_phonon_phonon.py:402-418` needs every block rank to own at least
`g_band + 1 = 4` blocks, and sixteen blocks over four ranks gives exactly four
each.

The mechanism is the GEMM batch. `ring_contract_pre` issues its three GEMMs
with the tau slice as the batch dimension, so stack parallelism divides the
batch while leaving the launch count -- fixed by the vertex quads -- alone. At
`bcs = 1` each launch carries 40 tau points; at `bcs = 4`, stack is 1 and each
carries all 161. The gain is not linear in either variable: halving the launch
count and doubling the batch (`bcs = 2`) buys only 5.3 % on the ring, while
the second step to a full-length batch buys 18 %. Small batches on a 36-DOF
block are badly underutilised and the penalty falls away sharply once the
batch is long enough, which is the same effect from the other side as
Sec. 6 item 4's `b >= 63` recommendation and as the dense solver's own
task-batching result.

Rank scaling, from the archived `scale-s*` benches on the L4 bed at
`ne = 241`: 3.863 / 2.086 / 1.096 / 0.618 s per iteration at 1 / 2 / 4 / 8
ranks, i.e. 93 %, 88 % and 78 % parallel efficiency. On this bed the archive's
eight-rank `l16f-g3` ran at 3.15 s against 4.39 s here on four, so **one node
is 1.43x the more node-hour-efficient and two nodes is 1.39x the faster in
wall time**. Four ranks is therefore right while node-hours bind and wrong
when the walltime does. `nccl` and `device_mpi` are within 1.3 % of each
other at four ranks.

Two levers deliberately not taken. `sse_ring_dtype = "complex64"` gives
10-30 % end to end but `gpu_campaign_2026-07.md` Sec. 5 rules it out for
`eta = 0` conservation-grade numbers -- it moves the bubble-balance residual
from ~1e-14 to <1e-6, and that gate is the honesty anchor of this campaign.
And the fused-ring kernel of Sec. 8/10 measured below cuBLAS, so the 90 % is
not currently addressable. The structural lever that IS available is block
size: Sec. 6 item 4 puts `b >= 63` at 2-2.4x the ring efficiency of `b = 36`,
which is what the reblocked `c16x2` (`b = 72`) has.

## 7. What this campaign does not settle

The FC3 **fit** cutoff. `cnt33_vasp.yaml:74` fits FC3 with `r_c = 2.5 A`
against a transport cell of `L = 2.4595 A`, so the `r_c < L` criterion that
makes the nearest-neighbour shell exact
(`spatial_representation.md:275-283`) fails by 1.6 %. What caps the
shell at offset 1 is the `[1,1,3]` fitting supercell, not the cutoff. Testing
it needs a refit at `[1,1,5]` and new VASP runs. The upstream reap
`phonon/configs/cnt/fc3_hiphive_cnt33_vasp/` is not on the analysis machine,
which also blocks `phonon/studies/linewidths.py` (the only phono3py
comparison) and `ballistic.py --wires cnt33`. `cluster/cnt33_fcq` separately
records `vasp.encut = 400 eV` against a recommended 520.

Nor does it settle the infrared treatment. The 2 THz mask moves the answer by
68 % and takes the conservation spread from 8.5 % to 0.6 %; `sse_cm_subtraction`
is the derived alternative and has never converged on a device.
