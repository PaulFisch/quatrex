# Regenerating the report figures

Every figure referenced by the report lives in `document/fig/transport_sweeps/`
and has a committed generator script that reads only committed data — no solver
re-runs are needed to re-plot. Figures NOT referenced by the report were moved
to `document/fig/attic/` (see its README).

## One command

```bash
python phonon/scripts/figures/make_all.py
```

runs every generator, then verifies that every figure referenced by the tex
exists and was refreshed, and lists anything that was not. Each generator also
prints the derived numbers that appear in the text (the claim-verification
trail). Styling is centralised in `phonon/studies/style.py` (colour-blind-safe
palette, png+pdf output).

## Figure → generator → data

All generators are in `phonon/scripts/figures/`; data paths relative to the
repo root. "notebook" = `phonon/docs/lab_notebook_archive.md`, the surviving
record of runs whose raw outputs were purged in the studies reorganisation
(commit 843c3069) — those values are hard-coded with the notebook section
cited in the script.

| figure(s) | generator | data |
|---|---|---|
| eta0_convergence_methods, eta0_cnt33_cutoff, d5a_gamma_anh | `eta0_convergence.py` | `phonon/studies/out/conv1e10/*.log` (cnt33_smooth_L2, d5a rpm/jfnk/diag/tgrow), `phonon/scripts/verify/d5a_gamma_anh.npz` |
| eta0_knob_ablation, eta0_knob_sensitivity | `eta0_knob_ablation.py` | `phonon/studies/out/convergence/L{2,3}e0_*.{log,npz}` (half-vs-fft x mixer matrix), `phonon/studies/out/conv1e10/{L2_taper3,L2_taper4s,cnt33_smooth_L2}.log`+npz, `phonon/scripts/out/prod/cnt33_eta0/L{2,3}_anh.{log,npz}` |
| local_vs_mw_current | `local_vs_mw_current.py` | `phonon/studies/out/local_mw/L3_eta{0,07}/run.npz` (launch recipe in `phonon/scripts/verify/local_vs_mw_current.py`; needs the 2026-07-03 `slab_absorption` snapshot key) |
| eta0_cnt33_transmission, eta0_cnt33_transport, eta0_cnt33_ir_plateau | `eta0_physics.py` | `phonon/scripts/out/prod/cnt33_eta0/` (summary.json + L2/L3/T-sweep npz), `…/prod/geom/cnt33_L2/dynamical_matrix.mat` |
| eta0_cnt33_conservation_iter, eta0_cnt33_ratio_eta, eta0_cnt33_conservation_T, conservation_bubble_replica | `conservation_figs.py` | `phonon/scripts/out/prod/cnt33_eta0/` (+ the matched-η table from `conservation.ratio_eta`, hard-coded) |
| sinw_d5a_eta0_panel | `sinw_d5a_eta0_panel.py` | `phonon/studies/out/conv1e10/sinw_d5a_L2_eta0_{jfnk_tgrow,jfnk_warm,jfnk_t005,jfnk_ptc01,jfnk_tgrow_repro,floor_anneal}.{npz,log}` |
| cnt33_temperature_g, cnt33_temperature_ratio | `cnt33_finite_eta_bias.py` | notebook F30 (dense reference) + `prod/cnt33_eta0/summary.json` |
| ballistic_vs_T_d5_d11, ballistic_vs_length_d5_d11 | `ballistic_wires.py` | `phonon/studies/out/ballistic_curves.npz` (regenerable in ~10 min: `python -m phonon.studies ballistic run --wires d5a d11a`) |
| srtio3_eta0_transport | `srtio3_eta0_transport.py` | `phonon/scripts/out/prod/srtio3_eta0/summary.csv` |
| static_se_study | `static_se_figs.py` | `phonon/scripts/data/static_se_summary.npz` (compact; the full `scripts/out/snapshots/study_*.npz` set is ~430 MB, local-only, regenerable in ~3 h via `phonon/scripts/verify/_static_se_sweep.sh`) |
| systems_panel | `systems_panel.py` | the `structure:` sections of `phonon/configs/{si_primitive/hiphive_big,sinw/sinw100_d5a_vasp_sc4,sinw/sinw100_d11a_vasp_sc4,cnt/cnt33_vasp,mos2/mos2_bulk_vasp.gen.b,perovskite/srtio3_small_vasp}.yaml` (no run data: the panels are the structures the force constants were fitted on) |
| fc_method_dispersion_si | `fc_method_dispersion_si.py` | `phonon/reaps/si_primitive_work/fc2.hdf5`, `phonon/configs/si_primitive/dfpt/ph.out` |
| phph_physics_si | `phph_physics_si.py` | `phonon/reaps/si_primitive_work/kappa-m191919.hdf5` |
| phph_NU_gruneisen_si | `phph_NU_gruneisen_si.py` | `phonon/reaps/si_primitive_work/{phono3py.yaml,fc2.hdf5,fc3.hdf5}` (~5 s phono3py recompute) |
| d11a_decomp_ganh, d11a_decomp_conservation | `d11a_decomposition.py` | `phonon/configs/sinw/reaps/sinw100_d11a_vasp_sc4/transport_quality/transport_quality.csv` |
| decomp_kernel_speedup, decomp_cost_scaling | `decomposed_sse_cost.py` | in-script literals from the standalone ring benchmark (`phonon/studies/_bench_factored_sse.py`, logs `cluster/bench-decomp`, `cluster/bench-legacy`, Γ/b=63 shape) — the benchmark contracts the ring in isolation, so it is not derivable from any run.npz; SCBA points from `phonon/scripts/data/decomposed_sse.csv` |
| decomp_compression | `decomposed_sse_compression.py` | in-script literals: INDSCAL fit residuals + factor/qfold file sizes from the geometry build log (`cluster/sifilm-L10-geom/run.log`) |
| decomp_rank_error, decomp_amplification | `decomposed_sse_error.py` | one-shot Σ error: literals from `phonon/studies/_rank_error_sse.py`; self-consistent errors: `phonon/scripts/data/decomposed_sse.csv` |
| decomp_scba_convergence, decomp_harmonic, decomp_observables | `decomposed_sse_convergence.py`, `decomposed_sse_harmonic.py`, `decomposed_sse_observables.py` | `phonon/scripts/data/decomposed_sse_spectra.npz` |
| cutoff_sse_d5_d11 | `cutoff_sse.py` | d11a: `…/sinw100_d11a_vasp_sc4/cutoffs/cutoffs_sweep.csv`; d5a: `phonon/reaps/hiphive_sinw100_d5a_vasp/cutoffs/cutoffs_sweep.csv` (restored from git) |
| cnt_blocking_kk | `cnt_blocking_kk.py` | `cluster/{c16-ball,c16-half,c16x2h,c16-kk}/run.npz` — the 2026-08-28 CNT campaign on 4 GH200 (see `phonon/docs/cnt_campaign_2026-08.md`). One 16-cell (3,3) device at eta=0, band 3, ne=161; the arms change the blocking (16x1 vs 8x2) and the retarded form (half vs Kramers-Kronig), one variable each. |
| solver_scaling | `solver_scaling.py` | `phonon/scripts/out/rgf_vs_dense_scaling.csv` + retired dist_scaling literals (in-script, provenance noted) |
| phph_scaling, phph_q_scaling, prod_phph_scaling, phph_memory | `phph_scaling_figs.py` | `phonon/scripts/data/prod_scaling_results.csv` (restored from git) + in-script literals with provenance |
| si_film_vs_guo | `si_film_vs_guo.py` | notebook F23/F29 + Guo PRB 102, 195412 (2020) reference values (hard-coded, cited) |
| si_film_conductance_b | `si_film_conductance.py` | notebook F23 archived ballistic points (hard-coded, cited) |
| prod_qfilm_qconv, prod_qfilm_conservation | `prod_qfilm.py` | the appendix's own tables (tab:prod_film/tab:prod_dense; raw session purged) |
| sinw_d5a_ballistic_plateau | `sinw_d5a_ballistic_plateau.py` | `phonon/studies/out/conv1e10/sinw_d5a_L2_{eta0_diag,irsub2_smoke}_ball.npz` |
| srtio3_rattle_renorm | `srtio3_rattle_renorm.py` | `phonon/configs/perovskite/fc3_hiphive_srtio3{_small,}_vasp/fcp.fcp` + config yamls |

## Which runs are still current

Every `cluster/` path above is a run directory, and most of the corpus is
superseded by the correctness gates established in July and August 2026.
`phonon/docs/run_audit_2026-08.md` carries the verdicts;
`phonon/scripts/data/run_manifest_{cluster,studies_out,scripts_out}.csv`
carry them one row per run, regenerated by

```bash
python phonon/scripts/audit_runs.py [--root <tree>] [--nested] [--check]
```

Superseded directories were moved to `cluster/attic/<class>/` with their
logs, configs and `run.npz` intact (index: `cluster/attic/INDEX.csv`), and
the regenerable geometry (`qfold_vertices.npz`, `sigma_best*.npz`) was
pruned everywhere a committed script does not name the file by path. Every
generator above was re-run afterwards: 24 of 24 referenced figures
regenerate, generator for generator identical to the pre-prune run.

Supplementary (attic): `transmission_physicality.py` regenerates the full
η=0-transmission physicality audit figure into `fig/attic/`.

The `decomp_*` data artifacts are produced once from the campaign outputs by
`phonon/scripts/verify/_extract_decomposed_sse.py --runs cluster`, which reduces
~10 MB/leg of `run.npz` to a CSV plus q-summed spectra (~0.5 MB) so the
generators read only committed data. Note it takes the per-iteration residual and
lead-balance traces from the run LOGS, not the npz: `iter_heat`/`iter_sigma_max`
are stored as the rank-0-local frequency slice, and these runs put one frequency
per rank, so rank 0 holds ω=0 alone (zero heat). `iter_bubble_balance` is
all-reduced and is safe to read from the npz.

## Figures whose RAW data cannot be re-measured without new solver runs

Every referenced figure regenerates from the repo. For the following, however,
the generator draws on the archived record rather than raw sweep outputs (the
raw files were purged before this policy existed) — re-measuring them would
need the listed run:

| figure | archived source | to re-measure |
|---|---|---|
| cnt33_temperature_g/ratio (dense reference series) | notebook F30 tables | dense cnt33 T-sweep (~1 h) + ladder (L3 ≈ 6.8 h) |
| si_film_vs_guo, si_film_conductance_b | notebook F23/F29 | film ballistic/anharmonic sweeps (days; coupled-q) |
| prod_qfilm_qconv/conservation | appendix tables | coupled-q film nk-sweep (± zero-mode projection) |
| solver_scaling (right panel), phph_* scaling literals | retired benchmark logs (git) | re-run the benchmark drivers (hours) |

Run-summary overlays from `phonon/studies/transport.py` (`fig_sinw`,
`fig_film`) now save under their own `prod_sinw_*` / `prod_si_film_vs_guo`
names so a stale production sweep can never overwrite the curated
generators above.
