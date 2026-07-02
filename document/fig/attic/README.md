# Figure attic

Figures that are NOT referenced by the report. Moved out of
`fig/transport_sweeps/` (which contains exactly the referenced set) during the
2026-07-02 overhaul. Status per family:

| figure(s) | status |
|---|---|
| `eta0_cnt33_transmission_physicality` | supplementary audit figure, regenerable (`phonon/scripts/figures/transmission_physicality.py`); its content is folded into `eta0_cnt33_transmission` (N(ω) staircase) and `eta0_cnt33_ir_plateau` (quantised plateau) |
| `sinw_d5a_bands`, `sinw_d5a_eta0_*` (7 diagnostics) | regenerable (`phonon/studies/_eta0_diag_plots.py` + `phonon/studies/out/conv1e10/*.npz`); superseded by the unified `sinw_d5a_eta0_panel` |
| `static_se_tadpole_breakdown` | superseded by the merged `static_se_study` (fresh snapshot sweep); the old figure's data was purged and its CNT loop curve disagreed with `static_se_study`'s |
| `prod_cnt_ladder` | source run purged (unconverged finite-η production sweep); removed from the coupled-q appendix, superseded by the η=0 ladder in `sec:res_eta0` |
| `si_film_conductance_a` | strict subset of `si_film_vs_guo`; dropped from the text |
| `cnt_transport`, `cnt_cutoff`, `length_scaling`, `d5a_scp_convergence` | pre-reorganisation working figures; data purged, values preserved in `phonon/docs/lab_notebook_archive.md` and in the regenerated replacements |
