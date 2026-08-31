# Reblocked CNT length ladder

**Status:** scaling, the 16--128-cell ladder and the matched 96-cell support
audit are complete, 2026-08-31.

## Recorded ladder definition

The ladder keeps the full stored CNT support instead of reducing it:

* two primitive transport cells per 72-DOF solver block;
* `sse_g_band = 3`;
* the complete stored FC3 support after exact re-partitioning;
* the 161-point uniform 0--55 THz convolution grid;
* zero broadening, half retarded reconstruction and the exact four-ring
  greater-from-lesser identity;
* plain linear mixing with factor 0.1.

This is a diagnostic of the archived half-retarded ladder, not the final
physical recipe. The half reconstruction omits the real Kramers-Kronig part
of the retarded self-energy. At zero broadening it is non-causal and the long
matched-length runs below show broad residual recurrences. A causal `fft`
retarded ladder is required before interpreting the current as a mean-free-
path result.

The auxiliary grid is disabled for this reference, not deleted. It remains an
experimental representation that must reproduce this uniform-grid result
before it can replace it.

The scaling, 16-cell and 24-cell runs use source commit
`67d602993a7b65a37eddd5a70ba4be17492ef52b`. The 32--64-cell runs use
`16f80537`, which only makes shared output-directory creation race-safe.
The 128-cell result uses `55d083e4`, which retains permutation-cache entries
only for the output pairs owned by each block rank.
The focused complete-band, reblocking and auxiliary-grid suite passes locally:
21 tests passed.

## Exact input construction

`phonon/studies/engine/reblock_device.py` builds every rung from the primitive
`cluster/c16-half` archive. It verifies the dense finite-device FC2 operator
and compares every merged FC3 tensor slice with its primitive source.

| Primitive cells | Solver blocks | DOF/block | Primitive FC3 blocks | Merged FC3 blocks |
|---:|---:|---:|---:|---:|
| 16 | 8 | 72 | 106 | 50 |
| 24 | 12 | 72 | 162 | 78 |
| 32 | 16 | 72 | 218 | 106 |
| 48 | 24 | 72 | 330 | 162 |
| 64 | 32 | 72 | 442 | 218 |
| 128 | 64 | 72 | 890 | 442 |

## Strong scaling of the 16-cell rung

Jobs `4565860`, `4566126`, `4566250` and `4566307` ran three identical cold
SCBA iterations on 1, 2, 4 and 8 Daint nodes. Each node used four GH200 GPUs.
The block communicator stayed at size two, so additional ranks divided the
frequency stack. Times are medians of iterations 1 and 2 and exclude the first
call setup.

| Nodes | GPUs | Local tau points | Ring (s) | Ring speedup | Ring efficiency | Iteration (s) | Iteration speedup | Iteration efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 161 | 8.0224 | 1.00 | 100% | 8.2798 | 1.00 | 100% |
| 2 | 8 | 81 | 4.1110 | 1.95 | 97.6% | 4.3896 | 1.89 | 94.3% |
| 4 | 16 | 41 | 2.3227 | 3.45 | 86.3% | 2.5661 | 3.23 | 80.7% |
| 8 | 32 | 21 | 1.2924 | 6.21 | 77.6% | 1.4921 | 5.55 | 69.4% |

The final lead current after three forced iterations is
`17.99784778298` in every run. The relative Sigma residual
(`5.8149e-1`) and lead balance (`3.2498e-5`) also agree, so the distributed
paths preserve the calculation.

Scaling is close to ideal through two nodes and remains useful at eight. The
ring dominates the steady iteration, but each stack rank has only 21 tau
points at 32 GPUs. Kernel-launch, reduction and fixed solver costs then become
large compared with useful local contraction work. The warm solver is already
only 0.09--0.17 s and does not control the result. Four nodes are the better
node-hour choice for this small rung; eight nodes minimize elapsed time.

## Convergence ladder

The production gate is simultaneous convergence of the retarded self-energy
and heat-current conservation. A short scaling run is not a physics result.

| Primitive cells | Job | Nodes | Iterations | Steady iteration (s) | Relative Sigma residual | Lead balance | Lead current |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4566515 | 1 | 101 | 8.2267 | 9.9529e-4 | 5.0396e-4 | 11.74109 |
| 24 | 4567378 | 1 | 108 | 12.9810 | 9.6155e-4 | 9.0888e-4 | 10.93254 |
| 32 | 4567827 | 2 | 107 | 9.1759 | 9.6364e-4 | 1.2808e-3 | 10.52394 |
| 48 | 4567863 | 4 | 109 | 8.0203 | 9.9810e-4 | 1.8286e-3 | 10.07512 |
| 64 | 4567913 | 8 | 108 | 6.1271 | 9.7888e-4 | 1.9466e-3 | 9.98664 |
| 128 | 4568041 | 16 | 112 | 8.9041 | 9.7601e-4 | 9.4711e-4 | 9.38597 |

The 16-cell result reproduces the archived 101-iteration convergence and
contact balance with the cleaned current code. The 24-cell rung also
converges without losing contact balance, and the 32-cell balance is 0.128%
rather than the 6.27% measured for the bad 16x1 blocking. The balance remains
below 0.2% through 128 cells. Current decreases monotonically by 20.1% from 16
to 128 cells. The 48--64-cell change is only 0.88%, but the 64--128-cell change
is 6.01%. The apparent 64-cell plateau is therefore not length convergence.
Distributed final and minimum-residual Sigma checkpoints remain on Daint for
restart and audit.

The measured production allocation ladder is 1, 1, 2, 4, 8 and 16 nodes. The
48-cell run reached 92% maximum device usage on four nodes, whereas the
64-cell run stayed below 84% on eight. The first 128-cell attempt reached a
96.18 GB CuPy pool and failed while building a cache replicated across block
ranks. Restricting the cache to owned output pairs reduced the successful
run's pool peak to 60.24 GB. A three-iteration 16-cell parity run, job
`4568119`, reproduced the old lead current `17.99784778298`, residual
`5.8149e-1` and lead balance `3.2498e-5`; its pool peak fell from 21.82 to
16.76 GB.

The first 32-cell attempt, job `4567821`, exposed a multi-rank race in shared
output-directory creation before SCBA started. Commit `16f80537` makes that
operation idempotent and adds a regression test. Job `4567827` passed the old
failure point and produced the converged result above.

## Current accounting audit

The production convergence gate is the contact balance. It integrates the
direct Meir-Wingreen current at the left and right contacts. Reblocking does
not change this definition. A ballistic iteration gives the same current,
`23.66164`, for two, three and four cells per block, with contact balances
between `8.2e-14` and `9.4e-13`. Reintegrating the saved endpoint spectra
reproduces the saved endpoint currents to `7e-15` or better.

The distributed RGF path stores only the two contact currents and fills the
interior interface slots with NaN. The former max-minus-min calculation
ignored those NaNs and therefore reported the contact balance again under the
name `internal_spread`. It did not measure current variation inside the
device. Commit `915bc59d` reports NaN when the complete interface profile is
unavailable and adds regression tests.

The dense offline observable can combine bond current and slab absorption
into a telescoped current. The distributed production path does not compute
slab absorption when the block communicator has more than one rank, so no
telescoped interior profile exists for these runs. This is a missing
diagnostic, but it does not enter or invalidate the contact-current balance.
The roughly 2% mismatch of the four-cell blocking is therefore a real mismatch
between its two contact currents, not an internal-spread artifact.

Converged checkpointing had a separate consistency bug. The loop exits before
mixing when a state passes its gate, so the old `sigma_final` saved the raw map
output while the logged observables belonged to the measured input iterate.
Commit `c33748af` saves the measured iterate for converged or divergent runs.
The `sigma_best` files used for the final comparisons already had the correct
semantics.

## Separating length and support convergence

Reblocking is not a neutral partition change. The self-energy output is fixed
to neighbouring solver blocks and the selected Green-function band is fixed
in solver-block units. Increasing primitive cells per block therefore widens
both masks in physical cells. A block-width comparison must keep the physical
device length fixed; its change measures missing spatial support, not a mean
free path. A length ladder at fixed blocking then measures the remaining
transport-length dependence.

For `g_band = 3`, the primitive-cell coverage is:

| Cells/block | Full Sigma distance | Partial Sigma distance | Full G distance | Partial G distance |
|---:|---:|---:|---:|---:|
| 2 | 0--2 | 3 | 0--6 | 7 |
| 3 | 0--3 | 4--5 | 0--9 | 10--11 |
| 4 | 0--4 | 5--7 | 0--12 | 13--15 |

A matched 96-cell device is divisible by two, three and four. Jobs `4568155`,
`4568156` and `4568158` built and checked all three representations from the
same primitive archive. Every input has 3,456 device DOF, the same dense
finite-device FC2 operator and all 666 primitive FC3 blocks.

| Cells/block | Solver blocks | Block DOF | Merged FC3 blocks | Block ranks | Nodes |
|---:|---:|---:|---:|---:|---:|
| 2 | 48 | 72 | 330 | 2 | 16 |
| 3 | 32 | 108 | 218 | 4 | 32 |
| 4 | 24 | 144 | 162 | 6 | 48 |

The measured production timings are substantially better than the projection.
The medians exclude the first iteration.

| Cells/block | Job | Iteration (s) | Ring (s) | Phonon solver (s) |
|---:|---:|---:|---:|---:|
| 2 | 4568160 | 6.6937 | 6.1063 | 0.3109 |
| 3 | 4568197 | 7.3523 | 6.4774 | 0.1939 |
| 4 | 4568162 | 9.0525 | 7.6161 | 0.1912 |

At the original `1e-3` Sigma threshold, x2 and x3 pass both convergence
gates. The x4 row is its first state below `1e-3`; it fails the 1% contact
balance gate and therefore is not a converged result.

| Cells/block | State | Sigma residual | Lead balance | Lead current |
|---:|---:|---:|---:|---:|
| 2 | iteration 109 | 9.7354e-4 | 1.8754e-3 | 10.431891 |
| 3 | iteration 105 | 9.7046e-4 | 5.3512e-3 | 9.394144 |
| 4 | iteration 112 | 9.9667e-4 | 1.3346e-2 | 9.451577 |

The x3 and x4 currents differ by only 0.61% at this matched residual, while
x2 is 10.4% higher than x3. Deeper continuations show that `1e-3` stops the
slow fixed-point mode far too early, so this table must not be interpreted as
the support-converged physics result.

## Deep half-retarded trajectories

Jobs `4568506` and `4568565` followed x2 and x3 for 300 states around and
after their first local residual minima. Both residuals rise and then fall
again. Neither run reaches the `1e-8` fixed-point threshold, and the lead
current decreases at every recorded step.

| Cells/block | State | Iteration | Sigma residual | Lead balance | Lead current |
|---:|---|---:|---:|---:|---:|
| 2 | first trough | 0 | 5.3421e-6 | 3.2513e-3 | 7.861726 |
| 2 | crest | 59 | 7.9879e-6 | 3.8172e-3 | 7.590795 |
| 2 | endpoint | 299 | 2.7273e-6 | 4.5414e-3 | 7.117351 |
| 3 | first trough | 8 | 1.7406e-5 | 7.4193e-3 | 8.050669 |
| 3 | crest | 85--86 | 5.8375e-5 | 8.1691e-3 | 7.619447 |
| 3 | endpoint | 299 | 1.7698e-5 | 8.8593e-3 | 7.141144 |

The x2 endpoint residual is 49% below its first trough, but its current is
9.47% lower. The x3 endpoint residual is still 1.68% above its first trough,
but its current is 11.30% lower. A local residual minimum is therefore not a
fixed-point observable for this map.

The first-trough x2 and x3 currents differ by 2.40%. At the endpoints they
differ by only 0.334%, with x3 slightly higher. Increasing the slab from two
to three primitive cells does not produce a resolved current decrease in
this matched-length comparison. The earlier large difference at `1e-3` was
mostly premature stopping, not evidence that x2 omitted important support.

This agreement is useful for choosing the spatial representation, but the
endpoint value is not a final transport result. Both states are still moving
and use the non-causal half-retarded reconstruction. The next physics ladder
must use the causal `fft` reconstruction and converge the actual fixed point.
