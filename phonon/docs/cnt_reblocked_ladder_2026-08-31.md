# Reblocked CNT length ladder

**Status:** scaling and 16--128-cell ladder complete, 2026-08-31

## Production definition

The ladder keeps the conserving CNT support instead of reducing it:

* two primitive transport cells per 72-DOF solver block;
* `sse_g_band = 3`;
* the complete stored FC3 support after exact re-partitioning;
* the 161-point uniform 0--55 THz convolution grid;
* zero broadening, half retarded reconstruction and the exact four-ring
  greater-from-lesser identity;
* plain linear mixing with factor 0.1.

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
(`5.8149e-1`) and internal heat-current spread (`3.2498e-5`) also agree,
so the distributed paths preserve the calculation.

Scaling is close to ideal through two nodes and remains useful at eight. The
ring dominates the steady iteration, but each stack rank has only 21 tau
points at 32 GPUs. Kernel-launch, reduction and fixed solver costs then become
large compared with useful local contraction work. The warm solver is already
only 0.09--0.17 s and does not control the result. Four nodes are the better
node-hour choice for this small rung; eight nodes minimize elapsed time.

## Convergence ladder

The production gate is simultaneous convergence of the retarded self-energy
and heat-current conservation. A short scaling run is not a physics result.

| Primitive cells | Job | Nodes | Iterations | Steady iteration (s) | Relative Sigma residual | Internal spread | Lead current |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4566515 | 1 | 101 | 8.2267 | 9.9529e-4 | 5.0396e-4 | 11.74109 |
| 24 | 4567378 | 1 | 108 | 12.9810 | 9.6155e-4 | 9.0888e-4 | 10.93254 |
| 32 | 4567827 | 2 | 107 | 9.1759 | 9.6364e-4 | 1.2808e-3 | 10.52394 |
| 48 | 4567863 | 4 | 109 | 8.0203 | 9.9810e-4 | 1.8286e-3 | 10.07512 |
| 64 | 4567913 | 8 | 108 | 6.1271 | 9.7888e-4 | 1.9466e-3 | 9.98664 |
| 128 | 4568041 | 16 | 112 | 8.9041 | 9.7601e-4 | 9.4711e-4 | 9.38597 |

The 16-cell result reproduces the archived 101-iteration convergence and
conservation quality with the cleaned current code. The 24-cell rung also
converges without losing conservation, and the 32-cell spread is 0.128% rather
than the 6.27% measured for the bad 16x1 blocking. The spread remains below
0.2% through 128 cells. Current decreases monotonically by 20.1% from 16 to
128 cells. The 48--64-cell change is only 0.88%, but the 64--128-cell change is
6.01%. The apparent 64-cell plateau is therefore not length convergence.
Distributed final and minimum-residual Sigma checkpoints remain on Daint for
restart and audit.

The measured production allocation ladder is 1, 1, 2, 4, 8 and 16 nodes. The
48-cell run reached 92% maximum device usage on four nodes, whereas the
64-cell run stayed below 84% on eight. The first 128-cell attempt reached a
96.18 GB CuPy pool and failed while building a cache replicated across block
ranks. Restricting the cache to owned output pairs reduced the successful
run's pool peak to 60.24 GB. A three-iteration 16-cell parity run, job
`4568119`, reproduced the old lead current `17.99784778298`, residual
`5.8149e-1` and spread `3.2498e-5`; its pool peak fell from 21.82 to 16.76 GB.

The first 32-cell attempt, job `4567821`, exposed a multi-rank race in shared
output-directory creation before SCBA started. Commit `16f80537` makes that
operation idempotent and adds a regression test. Job `4567827` passed the old
failure point and produced the converged result above.

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

A matched 96-cell device is divisible by two, three and four. Keeping 32
frequency-stack ranks and distributing the block axis gives the following
feasible comparison, projected from job `4568041`:

| Cells/block | Solver blocks | Block DOF | Block ranks | Nodes | Walltime request | Projected steady iteration |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 48 | 72 | 2 | 16 | 00:30 | 6--7 s |
| 3 | 32 | 108 | 4 | 32 | 00:30 | 10--11 s |
| 4 | 24 | 144 | 6 | 48 | 00:40 | 16--18 s |

The three reservations total 56 node-hours, plus about 0.5 node-hours for
input construction. The widest case has the same projected per-rank
permutation-cache load as the successful 128-cell, two-cell run.
