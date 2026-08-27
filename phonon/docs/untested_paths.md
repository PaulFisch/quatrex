# What has no evidence behind it

2026-08-27, written alongside `run_audit_2026-08.md`. Three separate
things get confused under the word "untested", and they need different
fixes, so they are kept apart here: a knob that no device run has ever
exercised, a module that no test imports, and a default that is known to be
wrong. Device claims are checked against
`phonon/scripts/data/run_manifest_*.csv` and `cluster/attic/INDEX.csv`, not
against the config files -- the config file is not the record of what ran.

## 1. No device evidence

| knob | how it is set | device record |
|---|---|---|
| `sse_cm_subtraction` | `QX_SSE_CMSUB` | **eight runs, none on a bed that could answer** -- see below |
| `pole_sector.band_edges = "lead"` | config | none; added 2026-08-15 and never enabled |
| `sse_aux_restrict = "sample"` | config | one A/B at 8 iterations, residual 0.34-0.46, no arm converged (`grid_audit.md` Sec. "Device scale") |
| `reblock_device.py --per-block >= 2` | offline tool | unit-tested; produced `mos2f6x2`, `mos2f6x3` and (2026-08-27) `sifilm8x2`. The first two were then run at 2-3 blocks, so `g_band` clamped to 1-2; `sifilm8x2` is the first 4-block, 2-cell bed and has not been run |
| `frequency_grid = "file"` together with `sse_g_band = 3` | config + `QX_GBAND` | none; every non-uniform-grid run is a MoS2 film at 3 blocks or fewer |
| `low_freq_mixing_thz` | config only | 0.0 in every config on both clusters; the IR *mixing* knob, distinct from the IR *mask*, has never been switched on |

**The CM subtraction, precisely.** Eight device runs set `QX_SSE_CMSUB=1`:
`cnt-cmnull` and `cnt-cmnull2` on tortin, and `mos2f3-cmsub{,2,3}` and
`mos2f3-u2001-cmsub{,2,3}` on Alps. None settles gate (d).

* **All six MoS2 arms ran at `interaction_cutoff = 10.0 A`** -- the H6 rung,
  where `bubble_positivity.md` Sec. 6.10c measures `-i G^<` non-PSD at
  iteration 0 with Sigma identically zero. The subtraction was applied to a
  Green's function the box mask had already broken. None converged
  (residuals 3.5e-01, 9.5e-01, 2.0e+00, 4.3e+01; two crashed), and that
  failure is uninformative about the subtraction.
* The two CNT arms avoid the confound -- the 10 A box never truncates a
  tube's own cell -- but ran three iterations at `mixing_factor = 1.0`, the
  **bare control also left the warm-start state** (J 35.21 -> 37.62), and
  the log carries a 20.9 % KK truncation warning.

So the falsification clause of `ir_residue_derivation.md` Sec. 6 has NOT
fired, and gate (d) -- which every run in the corpus fails -- has never been
tested on a clean bed. The cheapest clean bed is the Si film, where the box
mask is the identity because the fcc cell's extent along x is 1.37 A.

**Correction to `report_rerun_backlog.md` item 9.** That item says the
positivity gate "has run in four short debug jobs (`mos2psd*`) and nowhere
else". It has run in twelve directories and printed `positivity` lines in
all of them -- the whole least-squares MoS2 ladder (`lsM2`, `lsM2a/b/f/n`,
`lsM4c/f/w`, 2026-08-15/16) carried `QX_POLE_PSD=1`. What remains true is
that no CNT or Si run has ever carried it.

**Correction to item 11.** That item says the OBC probe's "logs are not on
the analysis machine". They are on Alps and now mirrored at `cluster/alps/`:
`obcfull` (`QX_NEVP=full`), `obcsr` (`QX_OBC_ALG=sancho-rubio`), `obc8k2n`,
`si8kprobe`, `speccoarse`, `srcoarse` -- all at `ne = 8001`, `wmax = 35`,
`retarded = fft`. The lever the item calls remaining has already been
pulled.

## 2. No test imports these

From a sweep of `tests/` for imports, not for the module name as a word.

`phonon/solver/`:
`zero_modes.py` (the translation projector, `project_self_energy`,
`translation_leakage`), `causality.py`, `cutoffs.py`, `diagnostics.py`.
All four are reached at runtime through `solver/dense.py`; none is named by
an import in any test.

`src/quatrex/phonon/`:
`qfold.py` -- tests pass `qfold=` tuples directly, so `load_qfold` /
`save_qfold`, which every production run goes through, are unexercised.
`bubble_factored.py` is imported only for `_convolve_q`; the rest is
reached through `SigmaPhononPhonon`.

`src/quatrex/core/`:
`scba.py` (the SCBA driver itself, including the `frequency_grid = "file"`
ingestion and the uniformity refusal), `phonon_jvp.py`, `observables.py`,
`transport.py`.

`phonon/studies/engine/`:
`write_config.py` -- it writes every production TOML in the corpus and has
no test at all. `build_inputs.py`, `run.py` and `parity_check.py` likewise.

`phonon/phonon_inputs/`:
`force_constants.py`, `dfpt.py`, `pipeline.py`, `io.py`, `structure.py`,
`validation.py`, `cli.py`, `fc3_factor_device.py`.

**`PhononSolver` is never constructed.** `grep -rn "PhononSolver(" tests/`
returns nothing. The pole and spatial suites reach its static helpers
through `object.__new__(PhononSolver)`, so `solve()`, `_compute_obc`,
`_assemble_system_matrix`, `_selected_solve` and the eta ramps have unit
coverage of exactly zero, and their only evidence is that device runs
produce plausible numbers.

## 3. Structural gap in CI

`.github/workflows/tests.yaml` runs `tests/qttools` and `tests/quatrex`
only. `tests/phonon_inputs` -- the dense reference solver, `se_finite`, the
hiphive path and the whole `finite_analysis` chain pipeline -- **never runs
in CI**, on any event. Coverage is collected with `--source=src`, so
everything under `phonon/` is invisible to it even when it does run.
`justfile`'s `test-cov` / `test-mpi` targets do cover all of `tests/`, so
the gap is in the workflow, not in the suite.

Optional dependencies silently remove coverage when absent: `phonopy`
(`test_btd_fold.py` skips its whole module, `test_reblock_device.py`),
`torch` + `tensorly` (`test_fc3_compression.py`, i.e. the entire INDSCAL
fitter suite), `hiphive` (`test_hiphive_rattle_phonon.py`), `mpmath`
(`test_pole_covariance.py`). None is in `pyproject.toml`.

The phonon SSE tests are backend-aware and degrade under CuPy rather than
skipping: `test_sse_phonon_phonon.py:1650` sets `exact = xp.__name__ ==
"numpy"`, and commit `fd987dcc` records that two `bit_identical` gates were
reading GPU jitter. No CI job runs on a GPU.

## 4. Defaults that are known to be wrong

`interaction_cutoff = 10.0` (`src/quatrex/core/config.py:1325`, `:1430`,
`:1442`). `bubble_positivity.md` Sec. 6.10c measures `-i G^<` non-PSD at
**-2.167e-01** at 0.113 THz with Sigma identically zero at exactly this
cutoff, against -3.5e-16 at 40 A. It is the shipped default, it is not
emitted by `write_config.py`, and fourteen audited MoS2 runs inherited it.
Any device longer than the cutoff is affected; Si along x is not, because
the fcc cell's 1.37 A extent means the box never truncates.

`pole_sector.leg_weight_tol` defaulted to 0 for the whole pole campaign,
which is the crude `q_omega = gamma/(2h)` rule rather than the exact
line-weight gate; `observations_2026-08-15.md` Sec. 1.2 shows the two
disagree about a physics conclusion. Now 0.05.

`sse_g_band_taper` is reachable and warns only when `g_band > 1`. The taper
arm was retired on 2026-08-17 for converging to the wrong answer (interior
heat spread 47/72/86 % against 6/11/18 % for the boxcar band 3), but the
`"bartlett"` value is still accepted at band 1, where it does not warn.

## 5. What would close each of these cheapest

1. `QX_SSE_CMSUB=1` on a bed where the box mask is the identity and a
   converged control already exists. That is the **Si film**: `sifilm8s`
   (8 blocks, `ne = 121`, `g_band = 3`) converged to 9.3e-04 on Alps, and
   Si transports along x where the 10 A box never truncates. Run the
   subtraction against that control and the pre-registered null test
   becomes meaningful for the first time. The CNT `L8_g3` bed is the
   alternative, at 362 iterations.
2. A test that constructs a `PhononSolver` on the toy chain and runs one
   `solve()`, which would put `_compute_obc`, `_assemble_system_matrix` and
   the eta ramps under coverage at once.
3. `write_config.py` round-trip test: write a config for each system, load
   it through `QuatrexConfig`, assert the keys the gates depend on are
   present. Two of the four gates are currently unrecoverable from its
   output.
4. Add `tests/phonon_inputs` to `.github/workflows/tests.yaml` and widen
   coverage to `--source=src,phonon`.
5. Change the `interaction_cutoff` default, or make `write_config.py` emit
   it explicitly so no run inherits 10.0 by omission.
