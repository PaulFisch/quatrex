# CNT deep and causal continuation

**Status:** launched on 2026-09-01. Results remain provisional until the jobs
finish and their non-checkpoint outputs are pulled.

## Questions

This campaign tests two independent numerical questions:

1. Does the middle-length L48 point continue toward the requested `1e-8`
   self-energy tolerance, or does its residual turn around on a deeper fixed
   point trajectory?
2. Does the causal FFT, including the complete 97 THz Kramers-Kronig support,
   converge if it is continued with weaker mixing or enabled only after a
   tightly converged half-retarded seed?

These are convergence experiments. They do not make the half-retarded and
causal currents physically interchangeable.

## L48 chunked continuation

Job 4574448, `cnt-l48x2-deeper-chunked`, starts from the live best checkpoint
of `cnt-l48x2-deep300-r3`. That checkpoint has 16 rank shards and a recorded
best residual of `1.8058e-6`. The last completed iteration before the earlier
allocation failed was iteration 222, with residual `1.7565e-6`, lead balance
`2.3864e-3` and legacy lead current `8.790528`.

The continuation uses the unchanged L48 physical problem: two primitive cells
per slab, `sse_g_band = 3`, zero broadening, half-retarded self-energy and
linear mixing 0.1. It runs at most four 150-iteration chunks. Each chunk is a
new 16-rank process and continues from the previous chunk's final state. This
preserves the fixed-point trajectory while releasing the CuPy memory pool
before it can reach the approximately 25.8 GB process allocation observed in
the failed 223-iteration run. A chunk stops the chain early if it reaches the
joint convergence gates.

The allocation is four nodes for at most six hours, or 24 committed
node-hours. Sigma checkpoints remain only on Daint. The local pull command
excludes them.

The committed controller is
`phonon/studies/cnt_l48_deeper_chunked_job.sh`.

## Causal FFT audit

The historical C16 trajectories separate the relevant failure modes:

| arm | iterations | minimum residual | final residual | final lead balance | final legacy current |
|---|---:|---:|---:|---:|---:|
| half | 98 | `9.6263e-4` | `9.6263e-4` | `9.5126e-5` | 38.3613 |
| FFT, 55 THz support | 400 | `5.3035e-2` | `5.3035e-2` | `2.1495e-3` | 17.9504 |
| FFT, 97 THz auxiliary support | 400 | `2.2284e-2` | `3.2892e-2` | `1.1881e-4` | 17.2289 |
| Broyden, 97 THz support | 600 | `7.4449e-4` | `7.4449e-4` | `2.0230e-1` | 16.5435 |

The corrected KK trajectory reached its minimum at iteration 346 and then
rose mildly while retaining clean lead balance. The Broyden trajectory reduced
the map residual but moved into a strongly non-conserving state. A small map
residual alone is therefore not a sufficient acceptance gate.

The `c16-kk-restart-audit` controller runs three stages on one node and four
ranks:

1. Continue the existing corrected-KK final checkpoint for 500 iterations
   with linear mixing 0.05.
2. Recompute the half-retarded seed to `1e-5` tolerance, with at most 200
   iterations and linear mixing 0.1, saving a restartable final state.
3. Switch that state to the corrected 97 THz causal FFT and run 500 iterations
   with linear mixing 0.05.

Both causal stages save full-frequency iteration diagnostics for the incoming
DOS, retarded and lesser self-energy magnitudes, and the raw and windowed
lesser Green function. These arrays distinguish a global underdamped orbit
from instability confined to a frequency sector. Both stages retain final and
best checkpoints on Daint, but only `run.npz` and logs will be pulled locally.

The controller requests one node for at most 3.5 hours. It will be submitted
when the second Quatrex queue slot opens.

The committed controller is
`phonon/studies/cnt_c16_kk_restart_audit_job.sh`.

## Acceptance criteria

An endpoint is physically usable only if all of the following agree:

- the relative retarded self-energy residual reaches its requested tolerance;
- the two contact currents cancel within the heat-flow gate;
- the integrated bubble energy balance closes;
- the current and frequency-resolved state stop drifting over the final
  iterations.

Internal interface spread is diagnostic and is not substituted for contact
current cancellation. If the two causal paths approach the same conserving
state, the delayed-KK start is a useful continuation method. If they reach
different conserving states, the result indicates multiple basins and needs a
separate stability analysis. If neither residual falls, the spectral histories
will identify the frequency sector to target in the next solver experiment.
