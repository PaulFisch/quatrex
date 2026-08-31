# Beds that clear every gate, and what they cost

> Historical plan, 2026-08-31: this document includes the rejected CM
> subtraction gate and is not a production recipe. The regenerated manifests
> no longer use that gate. See `code_catalogue.md` for the current method.

2026-08-27, from `run_audit_2026-08.md`. That audit found no run in the
corpus satisfying gates (a) and (b) together. This is what such a run looks
like, per system, with the arithmetic that fixes the size.

## The arithmetic

`sse_g_band` is clamped to `n_blocks - 1` at three sites, so

    g_band = 3  =>  n_blocks >= 4
    gate (a)    =>  cells_per_block >= 2
    ------------------------------------
    device      >= 8 transport cells

and one further constraint that only appears at run time:
`sse_phonon_phonon.py:402-418` refuses unless every block rank owns at
least `g_band + 1 = 4` blocks. **At four blocks that forces
`block_comm_size = 1`** -- no block parallelism at all. The first geometry
that admits `bcs = 2` at `g_band = 3` is eight blocks of two cells, i.e. a
sixteen-cell device.

So there are two rungs, not one: a minimum bed that proves the point and
does not parallelise, and a scalable bed that costs four times the length.

The blocking is produced offline:

    python phonon/studies/engine/reblock_device.py \
        --src <existing 8-cell bed> --cells 8 --per-block 2 --out <new bed>

which rewrites `num_transport_cells` to 4 and rebuilds the matrix and the
FC3 blocks. It is unit-tested (`tests/quatrex/phonon/test_reblock_device.py`)
and has produced three beds: `mos2f6x2` and `mos2f6x3`, neither of which
was then run at four or more blocks, and **`cluster/sifilm8x2`, built
2026-08-27 -- 4 blocks of 2 cells, the first geometry in the tree that
clears both gates.** It has not been run. One thing to change before it is:
its config carries `energy_window_max = 15.0`, which is
`write_config.py`'s sifilm default and sits at or below the Si band top
(a Gershgorin bound on its own dynamical matrix gives
`omega_max <= 15.71 THz`). See the Si section.

## Si film -- run this first

The cheapest bed by a wide margin: the transport cell is 2 atoms, so a
two-cell block is 12 DOF against MoS2's 36, and `perm_cache` goes as `b^3`.

| | value |
|---|---|
| bed | `sichk_base` geometry, transport along x, `nk = [1, 9, 9]` |
| device | 8 cells as `num_transport_cells = 4`, 2 cells per block |
| `interaction_cutoff` | irrelevant: the fcc cell's 1.37 A extent along x means the box never truncates (`bubble_positivity.md` Sec. 6.11) |
| grid | `nfreq = 121`; **`fmax` >= 31**, not the 15.0 the bed inherits |
| mixing | **0.02 linear**, or Anderson after a few damped iterations |
| also set | `QX_GBAND=3`, `QX_POLE_PSD=1`, `QX_SSE_CMSUB=1` |

The geometry already exists: `cluster/sifilm8x2`, built 2026-08-27 by
`reblock_device.py`, 4 blocks of 2 cells, `nk = [1, 9, 9]`, eta = 0.

**Raise its window before running it.** The config carries
`energy_window_max = 15.0`, inherited from `write_config.py`'s sifilm
default. Si's band top is about 15.3 THz -- a Gershgorin bound on this
bed's own dynamical matrix gives `omega_max <= 15.71` -- so the window does
not cover the harmonic spectrum, let alone the `2 omega_max ~ 31 THz` the
bubble is supported on. `grid_audit.md`'s extent ladder puts the cost of
`top/omega_max = 1.0` at 4.7e-2 in the current and off the table in the
linewidth; the corrected 20 THz window used by `sichk_ext` is still only
1.3x. Set `--fmax 31` (or `QX_WMAX=31`). Note also that the code's own
`_check_kk_grid_support` warning did **not** fire on either `sichk_base`
(15.0) or `sichk_ext` (20.0), so on this bed its silence is not evidence
that the extent is adequate.

Cost anchor: `sichk_base` (3 blocks x 1 cell, `nf = 121`) converged in 69
iterations to 9.6e-08, and Alps has two Si films converged at the exact
band -- `sifilm5s2` (5 blocks, 9.4e-04) and `sifilm8s` (8 blocks,
9.3e-04), both `nf = 121`, one cell per block. So the band gate alone is
not what breaks this system, and `sifilm8x2` changes exactly one variable
against `sifilm8s`. Four blocks of two cells is 2x the block DOF and 8x the
per-block cost, so of order a few node-hours on tortin.

The grid rung matters more here than the blocking. `grid_audit.md` measures
this exact bed diverging as `nf^4.8` -- 9.6e-08 at 121, ABORT at 15001 -- and
establishes that it is an **iteration** problem, not a truncation one:
positivity is intact at every resolution and `mixing_factor = 0.02` turns
the divergence into a monotone descent. So run the 121 rung first to
establish the fixed point, then walk the grid up from the previous rung's
solution. Doing the two together confounds them.

This is also the first bed on which `QX_SSE_CMSUB=1` can be a real test:
the CM subtraction is derived precisely to remove the DC channel that makes
refinement diverge, and Si is the system where refinement demonstrably
diverges.

## CNT (3,3) -- the one bed with a converged control

`phonon/studies/out/cnt33_gband_length/L8_g3` is the **only converged
`g_band = 3` run in the corpus**: 8 blocks x 1 cell, `nf = 361`,
`fmax = 55`, `aux_fmax = 88`, eta = 0, converged in 362 iterations to
9.87e-04, 1447 min on 64 tortin ranks. Reblocking the same eight cells to
4 x 2 changes exactly one variable, against a control that is known to
converge.

| | value |
|---|---|
| device | the L8 bed reblocked to `num_transport_cells = 4`, 2 cells per block |
| grid | `nfreq = 361`, `fmax = 55`, `aux_fmax = 88` as in L8_g3 |
| caveat | that configuration still carries **1.8 %** KK truncation, above the 1 % gate; `fmax` or `aux_fmax` needs raising to clear gate (c) |

Two things to know before spending the time. The block DOF goes 36 -> 72, so
the ring cost per pair goes as `b^3` and the run is roughly eight times
heavier per pair with half as many pairs -- budget a multiple of the 24 h
that L8_g3 took, not a fraction. And `bcs` must be 1 at four blocks, which
removes the axis that made the 64-rank run cheap; the practical version may
be the sixteen-cell bed at 8 blocks x 2 cells with `bcs = 2`.

`cluster/cnt-nescan-g3` already answers what the grid does at `g_band = 3`
on the L4 bed: J drifts 8.6 % from `ne = 161` to 361 and only `ne = 161`
converges within 300 iterations. Expect the same on L8; the grid rung is
not free even with the exact band.

## MoS2 film -- the expensive one, and the reason to do the other two first

| | value |
|---|---|
| device | 8 cells as `num_transport_cells = 4` over a 12-atom (2-cell) block, transport along z |
| `interaction_cutoff` | **>= 30 A** -- 22 A is the measured floor, 21 A still diverges at 98.6 % fill |
| grid | `nfreq = 2001`, aux grid **off** |
| nearest existing | `mosreach` (8 cells as 2 x 4, `g_band = 1`) and `mos2f8_ls_x1` (8 x 1, `g_band = 3`) |

The grid choice is settled and is the cheap part: `grid_audit.md` measures
2001 primary points reproducing 15001 to **0.02 %** on this system, by two
independent legs (a direct A/B and decimation of the converged reference).
The auxiliary grid is the opposite -- unconverged and expensive, carrying
+4.4 to +10.2 % against the aux-off bubble at `aux_dw` 0.005 to 0.02. "Aux
on" is not a production setting.

The cost is the problem. `perm_cache` is `32 b^3 (nq^2/P_q) Q` bytes and is
replicated on every rank; at `b = 36` (a 2-cell MoS2 block) `lsM4` peaked
at 82.5 GB/rank and `lsM6` OOM'd on 16 nodes
(`sse_memory_scaling.md`). A 4-block device has the same `b` as `lsM4` but
cannot use `bcs > 1`, so the per-rank budget is at best what `lsM4` needed.
The levers, in the order that document establishes them: `q_comm_size = 2`
(helps at `b = 36`, hurts at `b = 54`), then flattening the 5x5 transverse
mesh to one axis so `q_distributed` applies.

Do the Si film and the CNT first. They settle whether `g_band = 3` plus
2 cells per block actually changes the answer, on beds where the run
finishes, before that question is asked at MoS2 prices.

## What every one of them must carry

    QX_GBAND=3            gate (b); not emitted by write_config.py, so it
                          must be in the environment or it silently reverts
                          to the code default
    QX_POLE_PSD=1         turns a divergence into a statement about which
                          leg lost positivity; costs one variable
    QX_SSE_CMSUB=1        gate (d), which no run in the corpus satisfies
    --eta 0               and no eta_ir_floor, no eta ramp

and the config must set `interaction_cutoff` explicitly, because the
shipped default is 10.0 A and that is the H6 rung.

Check the first iteration's log for the `_check_kk_grid_support` warning
before letting a long run proceed: if it fires, gate (c) has already
failed and there is no auto-extension in the production path to save it.
