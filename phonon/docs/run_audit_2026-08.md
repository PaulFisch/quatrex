# The run corpus audited against the gates the method now has

2026-08-27. Tools: `phonon/scripts/audit_runs.py` (verdicts),
`phonon/scripts/archive_runs.py` (prune and move). Manifests:
`phonon/scripts/data/run_manifest_{cluster,alps,studies_out,scripts_out}.csv`,
one row per run with every gate resolved. Archive index:
`cluster/attic/INDEX.csv`.

Revised the same day, after CSCS access was restored: the Alps corpus is
272 further directories and 742.6 GB, and it holds the runs that decide two
of the questions below. Its metadata (configs, logs, job scripts,
summaries -- 31 MB for all 272) is mirrored at `cluster/alps/` and audited
from there; no bulk was pulled.

Four requirements were established over July and August 2026 that most of
the corpus predates, and there was nothing on disk that separated the runs
which still satisfy them from the ones that do not. This is that separation.

## 1. The gates

| # | gate | established in |
|---|---|---|
| a | >= 2 transport cells per BTD block | `bubble_positivity.md` Sec. 8, `spatial_truncation_derivation.md` |
| b | `sse_g_band = 3` | `report_rerun_backlog.md` Sec. 6 |
| c | grid resolves the line, extent reaches ~2 omega_max | `grid_audit.md` |
| d | the DC channel subtracted, not smeared | `ir_residue_derivation.md` |
| e | eta = 0 | `CLAUDE.md` |
| f | MoS2 `interaction_cutoff >= 22 A` | `bubble_positivity.md` Sec. 6.8-6.10 |

Gate (c) is machine-checkable from the code's own
`_check_kk_grid_support` warning, which fires above 1 % of peak bubble
weight at the top of the grid. Gate (b) is not recoverable from any config:
`sse_g_band` appears in **0 of 157** stored TOMLs because
`write_config.py` does not emit it, and `run.npz` does not store it either.
The audit resolves it in this order -- `QX_GBAND` from the `RUN env` line,
then the run name (the tortin campaign encoded it there and nowhere else:
`cnt-L4-gband2`, `l16f-g3`), then the code default for the run's date,
which flipped 1 -> 3 at commit `82761380` on 2026-08-01 -- and then applies
the clamp.

## 2. Gates (a) and (b) compete for the same cells

`sse_g_band` is clamped to `n_blocks - 1` at three independent sites
(`src/quatrex/phonon/solver.py:195`, `core/scba.py:125`,
`phonon/sse_phonon_phonon.py:393`). So `g_band = 3` needs at least four
blocks while gate (a) needs at least two cells in each of them: together,
**at least eight transport cells**. Every long run in the corpus spends its
cells on one axis or the other.

| runs | blocks x cells | effective `g_band` | |
|---|---|---|---|
| `mos2L4conv`, `lsM4*`, `cvM4e`, `mos2f6x3`, `cvM6b` | 2 x 2..3 | **1** | (a) yes, (b) no |
| `mos2f6x1`, `mos2f8_ls_x1`, `si4x1`, `l16f-g3`, `cnt33_gband_length/L8_g3` | 4..32 x 1 | 3 | (a) no, (b) yes |
| `mos2f3*`, `sichk_*`, `sires*`, `sifilm3*` | 3 x 1 | 2 | neither |

The corpus is 604 directories -- 170 live under `cluster/`, 53 in
`cluster/attic/`, 93 campaign arms under `phonon/studies/out/`, 18 under
`phonon/scripts/out/` and 272 on Alps. **378 of them reached the SCBA
loop**, and

    runs satisfying (a) and (b) together: 0

Gate failures over those 378: `no-cm-subtraction` 351, `gband` 328,
`cells_per_block` 309, `h6-cutoff` 98, `extent-truncated` 58, `timeout` 56,
`crash` 47, `oom` 20, `finite-eta` 3, `ir-floor` 1.

`sse_g_band` is absent from **0 of 101** Alps configs as well, and only
five Alps runs set `QX_GBAND` at all, so the Alps corpus is dated against
the same 2026-08-01 flip.

## 3. What each system has

Over all four trees, archived directories included. `solver` counts those
that entered the SCBA loop; the rest hold geometry, reaps or analysis.

| system | dirs | solver | converged | at `g_band = 3` | at >= 2 cells/block |
|---|---|---|---|---|---|
| MoS2 film | 215 | 175 | 24 | 3 | 42 |
| CNT (3,3) | 146 | 102 | 39 | 24 | 0 |
| Si film | 73 | 56 | 14 | 4 | 7 |
| Si nanowire (d5a/d11a) | 47 | 14 | 0 | 0 | 0 |
| CNT (8,0), SrTiO3 | 4 | 0 | 0 | 0 | 0 |
| unattributed | 119 | 31 | 7 | 16 | 0 |

"unattributed" is directories with no config and a name the classifier does
not recognise -- mostly benchmark and probe beds. The nanowire directories
predate the `RUN` provenance lines, and `conserving_vertex_findings.md`
already declares every anharmonic number from them superseded by the
2026-06-12 vertex fix.

### The CNT band ladder is the one place gate (b) was varied cleanly

From `phonon/studies/out/cnt33_{gband_length,long_gband3}` (ne = 361,
fmax = 55, `aux_fmax = 88`, eta = 0, one cell per block):

| arm | b_G | outcome | final residual |
|---|---|---|---|
| L8_g1 | 1 boxcar | diverged | 3.2e+06 |
| L8_g1t | 1 + Bartlett | converged, 107 it | 9.69e-04 |
| L8_g2 | 2 | diverged | 3.8e+03 |
| **L8_g3** | **3** | **converged, 362 it** | **9.87e-04** |
| L10_g1 | 1 boxcar | diverged | 3.5e+05 |
| L10_g1t | 1 + Bartlett | converged, 128 it | 9.97e-04 |
| L10_g2 | 2 | diverged | 2.1e+03 |
| L10_g3 | 3 | not converged in 600 it | 2.28e-04 |
| L16_g3 / L24_g3 | 3 | not converged in 600 it | 1.25e-03 / 5.71e-03 |

`b_G = 3` is the only band setting that converges a long tube without the
taper, and the taper arm is the one `report_rerun_backlog.md` Sec. 6
retired for converging to the wrong answer. L8 is the longest tube that
reaches `scba_tol` at all.

Eleven runs in the whole corpus converged at `g_band = 3`, and every one of
them is one cell per block: the CNT ladder (`L8_g3`; `l16f-g3`,
`l24f-g3`, `l32f-g3` at ne = 161 and `l16f-g3-361` at ne = 361, all on
Alps), `cnt-nescan-g3` at ne = 161, and -- newly surfaced from Alps --
**`sifilm5s2` and `sifilm8s`, Si films at 5 and 8 blocks, ne = 121,
converged to 9.4e-04 and 9.3e-04.** The Si film converges at the exact band
on a coarse grid, which is the fact the recommended bed rests on.

### grid_audit open item #30, answered

`cluster/cnt-nescan-g3` (2026-08-07/08, tortin) rescans the CNT L4 grid at
`sse_g_band = 3`, which `grid_audit.md` listed as its highest-priority open
item and which had not been pulled to the laptop. In the integral
convention (`J = lead_current x dw`, the trap that section documents):

| ne | dw [THz] | J | iterations | converged | last residual |
|---|---|---|---|---|---|
| 161 | 0.344 | 11.046 | 201 | yes | 9.92e-04 |
| 181 | 0.306 | 10.761 | 300 | no | 1.09e-02 |
| 201 | 0.275 | 10.775 | 300 | no | 1.17e-02 |
| 271 | 0.204 | 10.229 | 300 | no | 4.51e-02 |
| 361 | 0.153 | 10.101 | 300 | no | 2.23e-02 |

The exact band does not remove the grid dependence: J falls 8.6 % from
ne = 161 to 361, monotonically apart from the 181/201 pair. It does change
its character -- what remains is a drift, not a lottery. The caveat is
large: only ne = 161 converged, so four of the five rows are iterates at
residual 1e-2 to 5e-2 and not transport numbers. The comparison is honest
for the question "does the exact band close it" and not for any value.

## 4. What is not invalidated

Failing a gate invalidates a transport number. It does not invalidate a
measurement whose subject is the defect itself, and these stand as they are:

* the `mos2f2c12 ... c22` cutoff ladder -- its subject is gate (f), so
  running below 22 A is the experiment, not a fault;
* `sires501/1001/2001` and `sichk_cut/res` -- the Si `nf^4.8` divergence
  ladder of `grid_audit.md`, whose subject is gate (c);
* `mos2f6x1` against `mos2f6x3` -- the same six cells partitioned six ways
  and two ways, which is the (a)-versus-(b) trade measured directly;
* the `newton-*`, `mix-*`, `rre-*` and `l4-*` families -- mixer behaviour
  on a fixed bed, independent of whether that bed is the production one;
* `mos2L2conv` and `mos2L4conv` -- the two converged eta = 0 MoS2 fixed
  points. They ran at `g_band = 1` and are not production numbers, but they
  are the evidence that an eta = 0 fixed point exists for this system at
  all, which `observations_2026-08-15.md` Sec. 5 corrects an earlier denial
  of. Both are in `cluster/attic/gband/` with logs and `run.npz` intact.

## 5. The archive

`archive_runs.py` ran on `cluster/` on 2026-08-27.

**Pruned:** 81 files, 61.4 GB, of `qfold_vertices.npz` (rebuildable by
`build_inputs.py` from the source reap) and `sigma_best*.npz` (transient
warm-start state). `fc3_blocks.hdf5` was deliberately kept -- it is the
vertex, several audit scripts open it through a variable path, and the
whole corpus of it is 5.5 GB. Files a committed script names by path are
kept, which is what protects
`cluster/prod/geom/sifilm_L3_nk9/qfold_vertices.npz`
(`decomposed_sse_conservation.py`) and `cluster/mos2f3scp/*`
(`_asr_project_film_job.sh`). No log, config, `job.sh`, `summary.json` or
`run*.npz` was deleted.

**Moved:** 53 directories into `cluster/attic/<class>/`, class being the
first gate the run fails.

| class | n | |
|---|---|---|
| `gband` | 34 | ran at `sse_g_band` 1 or 2 |
| `h6-cutoff` | 12 | MoS2 below 22 A |
| `inputs-only` | 5 | geometry or warm-start state, no log on the laptop |
| `blocking` | 2 | one cell per block and nothing else disqualifying |

A directory named by any committed `.py` or `.sh` is never moved, and the
detector is over-inclusive on purpose: generators spell the path three ways
and only one carries the tree prefix (`"cluster/mos2f3"`,
`CL / "cnt-L3-gband2/run.log"`, `CL / f"newton-pc-{arm}/run.log"`), so the
audit probes the bare name and every separator-terminated prefix against an
f-string brace. A false positive costs nothing; a false negative deletes a
figure's input.

`cluster/` went from 70 GB to 25 GB by `du`. The two measures disagree
because the tree is sparse: the same content is 86 GB by `st_size` and
70 GB by allocated blocks, so 61.4 GB of file size frees 45 GB of disk.
`make_all.py` regenerates
24 of 24 referenced figures after the prune, generator for generator
identical to the pre-prune run, which is the gate that it broke nothing.

`phonon/studies/out` (4.1 GB) and `phonon/scripts/out` were audited and
left alone: they are largely git-tracked, and the one untracked bulk,
`anderson_test/jprobe_snaps` (2.7 GB), is kept deliberately -- its
`MANIFEST.md` is its index and its snapshots are the only record of the
Jacobian measurement quoted in `observations_2026-08-15.md` Sec. 4.

## 6. The laptop mirror is not the corpus

Two remote corpora, neither of them mirrored.

**tortin** (`/usr/scratch/mont-fort11/pfischill/quatrex/cluster`): 30 GB in
165 directories. Fifteen were pulled during this audit because committed
code names them or because they answer an open question; after that, 98
directories holding 28.2 GB have no counterpart on the laptop at all. Only
12 remote directories are redundant with the archive, and they total under
a gigabyte, so `cluster/REMOTE_PRUNE.sh` is a review list rather than a
delete list.

**Alps** (`/capstor/scratch/cscs/pfischil/quatrex/cluster`): **272
directories and 742.6 GB** -- an order of magnitude more than the laptop
holds, and until this audit entirely unaudited, because `ssh daint` had
been failing on an expired CSCS key. Its metadata is now mirrored at
`cluster/alps/` (31 MB for all 272 directories: configs, logs, job
scripts, summaries, structures) and carries its own manifest,
`run_manifest_alps.csv`.

Be exact about what survives what. `cluster/` is gitignored in its
entirety, so **the mirror is laptop-local**; the committed durable record
of those 742 GB is the 53 KB manifest, which carries every gate for every
run but not the logs behind them. CSCS scratch is purged periodically. If
the logs themselves should outlive both the scratch and this laptop, the
mirror needs `git add -f cluster/alps` or a tarball beside
`cluster/attic/document-last-tracked-*.tar.gz`; that is a call about
putting 34 MB of logs in the repo, not something the audit should decide.

Refresh the mirror with

```
rsync -az --include='*/' --include='quatrex_config.toml' --include='*.sh' \
      --include='slurm-*.out' --include='*.log' --include='summary.*' \
      --include='structure.xyz' --include='*.npy' --include='*.csv' \
      --exclude='*' \
      daint:/capstor/scratch/cscs/pfischil/quatrex/cluster/ cluster/alps/
python phonon/scripts/audit_runs.py --root cluster/alps --sizes <du-file>
```

What that 742.6 GB actually is, measured file by file:

| | GB | files | |
|---|---|---|---|
| `sigma_best*.npz` | 400.3 | 296 | transient warm-start state |
| `qfold_vertices*.npz` | 54.0 | 27 | rebuildable by `build_inputs.py` |
| `run*.npz` | 10.3 | 257 | **the results** |
| `fc3_blocks.hdf5` | 1.3 | 33 | the vertex |

**61 % of the tree is state no result depends on, and the results are
10 GB.** `cluster/REMOTE_PRUNE_alps.sh` prunes the first two in place --
one `find -delete`, with the two qfold files committed scripts read by path
excluded -- and leaves every result and log. Whole-directory deletion is
offered as an optional second part and is not the recommendation.

Three things only the Alps side records:

* the six `mos2f3*-cmsub*` runs, which are the whole device history of
  gate (d) on MoS2 (Sec. 7);
* the missing OBC probe logs. `report_rerun_backlog.md` item 11 says "the
  original probe's logs are not on the analysis machine" -- they are
  `obcfull`, `obcsr`, `obc8k2n`, `si8kprobe`, `speccoarse` and `srcoarse`
  on Alps, and the item's remaining lever (`nevp_solver` at fixed
  `obc.algorithm = spectral`) has in fact been pulled: `obcfull` is
  `QX_NEVP=full`, `obcsr` is `QX_OBC_ALG=sancho-rubio`, both at
  `ne = 8001`, `wmax = 35`, `retarded = fft`;
* the converged `g_band = 3` Si films `sifilm5s2` and `sifilm8s`.

Also on tortin only: `bench-decomp` and `bench-legacy`, which
`report_removed.md` records as gone and whose numbers were transcribed into
figure generators as literals; and the `mos2film_L{3,6,10,16}_nk{5,7,9}_o4`
quartic-vertex campaign (23 GB), in no manifest, figure or document.

## 7. Four records corrected

**The CM subtraction has run on a device eight times, and not once on a
bed that could answer the question.** `ir_residue_derivation.md` Sec. 6
lists gate V5 as "staged on tortin, waits on the pool watcher" and
`QX_SSE_CMSUB` appears in no config anywhere. It appears in eight logs.

| run | where | system | `interaction_cutoff` | ne | outcome |
|---|---|---|---|---|---|
| `cnt-cmnull` | tortin | CNT33 L4 | 10 A (no effect on CNT) | 181 | 3 it, not converged |
| `cnt-cmnull2` | tortin | CNT33 L4 | 10 A | 181 | 3 it, not converged |
| `mos2f3-cmsub` | Alps | MoS2 L3 | **10 A** | 267 | crashed |
| `mos2f3-cmsub2` | Alps | MoS2 L3 | **10 A** | 267 | timeout, res 9.50e-01 |
| `mos2f3-cmsub3` | Alps | MoS2 L3 | **10 A** | 267 | timeout, res 3.51e-01 |
| `mos2f3-u2001-cmsub` | Alps | MoS2 L3 | **10 A** | 2001 | crashed |
| `mos2f3-u2001-cmsub2` | Alps | MoS2 L3 | **10 A** | 2001 | timeout, res 4.25e+01 |
| `mos2f3-u2001-cmsub3` | Alps | MoS2 L3 | **10 A** | 2001 | timeout, res 1.99e+00 |

Every MoS2 arm sits at `interaction_cutoff = 10.0 A` -- the H6 rung, where
`bubble_positivity.md` Sec. 6.10c measures `-i G^<` already non-PSD at
iteration 0 with Sigma identically zero. So the subtraction was applied to
a Green's function the box mask had already broken, and its failure to
converge says nothing about the subtraction. This is the same confound as
the five `QX_SSE_LOWMASK` runs of `observations_2026-08-15.md` Sec. 4, and
it has the same reading: **masking or subtracting the infrared does not
repair what the box mask breaks.**

The falsification clause of `ir_residue_derivation.md` Sec. 6 -- "if the
exactly-subtracted film still orbits at eta = 0, the residual marginal
interlayer gain is physical" -- must therefore **not** be read as fired. It
requires a clean discretisation, and none of these six had one. On the CNT
pair the confound is absent (the 10 A box never truncates a tube's own
cell) but the test is still not a verdict: three iterations at
`mixing_factor = 1.0`, the bare control also left the warm-start state
(J 35.21 -> 37.62), and the log carries a 20.9 % KK truncation.

Gate (d) is therefore untested on every bed in the corpus, and the cheapest
place to test it is a cutoff >= 30 A MoS2 film or the Si film, where the
mask is the identity.

**There is no fmax auto-extension in production.** `auto_extend_fmax`
exists only in the standalone `phonon/solver/dense.py:1317`. The production
path has `_check_kk_grid_support` (`sse_phonon_phonon.py:1903`), which
warns and does nothing. Every run that carries that warning kept its
truncated grid.

**`interaction_cutoff` still defaults to 10.0 A** (`config.py:1325`,
`:1430`, `:1442`) -- the H6 rung that `bubble_positivity.md` Sec. 6.10
shows breaks the positivity of `G` before the self-energy exists. Fourteen
audited MoS2 runs sit at exactly that value, most of them by omitting the
key rather than by choosing it.

**The si4x2 fine-grid probe did not run at `g_band = 3`.**
`report_rerun_backlog.md` item 11 and results Sec. 4.3 rest on the
2026-08-14 retest of the Si fine-grid divergence -- `si4x2`,
`retarded = fft`, `wmax = 35`, `sse_g_band = 3`, `ne = 8001` -- as the run
that removed every objection. `si4x2` is `num_transport_cells = 2`, so
`block_sizes` has length 2 and all three call sites clamp the band to
`min(3, 1) = 1`. The config asked for 3 and the solver used 1. The 2x2
blocking and band 3 cannot coexist on a four-cell device at all, which is
the same arithmetic as Sec. 2. The conclusions that survive are the ones
that do not depend on the band: the spectral OBC is exonerated
(`beyn` and `full` agree bit-for-bit), `sancho-rubio` returns NaN at both
spacings, and the `ne = 141` control is clean. Whether the divergence
survives an exact band is untested.

## 8. Reproducing this

```
python phonon/scripts/audit_runs.py                              # cluster
python phonon/scripts/audit_runs.py --root phonon/studies/out --nested
python phonon/scripts/audit_runs.py --root cluster/alps --sizes <du-file>
python phonon/scripts/audit_runs.py --check                      # CSV is current
python phonon/scripts/archive_runs.py                            # dry run
```

One extraction rule earned by the Alps pass: a driver can put several arms
in **one** log (`cnt-cmnull` runs a bare arm and a `QX_SSE_CMSUB=1` arm into
the same file), so the audit takes the UNION of every `RUN env` line in a
file and records the arm count. Reading only the first line reports the
control's settings as the run's, which is how the CNT CM-subtraction test
was missed on the first pass. 42 directories in the corpus are multi-arm.

The manifests are committed; `cluster/` is not. `--check` re-derives and
diffs, so a manifest that no longer matches the tree is a hard error rather
than a stale table.
