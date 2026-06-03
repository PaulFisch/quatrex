# Results-Section Production Runbook

End-to-end recipe for producing the figures that go into
`document/src/results.tex § Finite-Structure Validation`. Three headline
systems plus one cross-system overlay.

## Known limitations and what to compensate for

Read this once before publishing any result.

1. **Σ^R reconstruction.** The default in
   `transport_metrics.transport_trace_from_sigma` is now
   `sigma_retarded_method="fft"`, which captures both the imaginary part
   (broadening) and the real part (Hilbert-transform level shift). The
   `"half"` option (older default) drops the level shift; only use it
   for cutoff-sweep *trend* comparisons where the shift cancels.

2. **ASR-broken FC3.** The hiphive least-squares fit on the SiNW had
   ~ 60 % FC3 ASR residual. Pass `asr_project=True` to
   `loader.load_quatrex_blocks` (and through `build_sse_inputs` /
   `run_sse_sparsity` / `run_cutoffs`) **for all SiNW results**. Without
   this, Σ(ω→0) carries a Drude-like artefact and low-ω transport
   numbers are not trustworthy. The `physical_tests.fc3_asr_legs`
   warning fires when this matters.

3. **Lead broadening is synthetic.** `transport_metrics._lead_broadening`
   uses a scalar wide-band-limit `Γ = γ·I` on the first/last slab. This
   is *not* a real bulk-Si contact self-energy. T(ω) absolute magnitudes
   are not directly comparable to literature; use it for cutoff-sweep
   *relative* comparisons and document the limitation. For absolute
   numbers, call
   `phonon_inputs.anharmonic.anharmonic_transmission_finite` with
   Sancho-Rubio leads instead.

4. **Bose factor at low ω.** The synthetic GF evaluates `n_B(ω_n)/(2ω_n)`
   at the mode frequency. For modes with `ω_n ≪ kT/ℏ` (≈ 6.3 THz at
   300 K) the weight grows as `kT/(2ℏω_n²)` — finite but large. Modes
   below `THZ_ACOUSTIC_BAND = 0.01 THz` are dropped (translational
   zeros); modes between 0.01 THz and the acoustic onset of the
   structure carry inflated spectral weight. For SiNW this is a few
   marginal acoustic modes; consider raising
   `synthetic_gf_dense(drop_threshold_thz=0.5)` if low-ω artefacts
   appear in T(ω).

5. **Lorentzian width.** Default `eta_thz` is `2 × dω` — fine for cutoff
   sweeps where adjacent modes are smeared together by design. For
   results figures of `physical_dispersion.pdf` and
   `sse_sigma_*.pdf`, override with `--eta-thz 0.05` (≈ phonon
   linewidth scale) so individual peaks resolve.

6. **Born approximation, not SCBA.** The cutoff sweeps evaluate Σ on a
   non-self-consistent (synthetic eigenmode) G. For full SCBA convergence
   call `phonon_inputs.anharmonic.anharmonic_transmission_finite` (which
   iterates Σ↔G via the Dyson equation). The chain test
   `test_sse_baseline_vs_dense_path` pins the bubble agreement at the
   Born level.

## Pre-flight (every system)

```bash
# 0. Sanity-check the YAML against the parameter-validation framework.
python -m finite_analysis validate phonon/configs/<family>/<name>.yaml
#    Exit code 0 = OK; 1 = warn (proceed); 2 = error (fix first).
```

The validator checks ENCUT/ecutwfc, kpoints, hiphive sample count and
cutoffs, displacement amplitude, supercell extent, etc. against
literature-recommended thresholds (`POTCAR_ENMAX_EV` table verified against
VASP 6.4).

## 1. SiNW(100) — main physics story

### Cluster phase (~6–10 h on 4 nodes)

```bash
cd phonon/
python -m phonon_inputs pipeline --config configs/sinw/sinw100_d5a_vasp.yaml
#   relax → fc3-sow → sbatch run_vasp.sh → fc3-reap (hiphive route)
```

The reap lands in the directory specified by
`configs/sinw/sinw100_d5a_vasp.yaml :: hiphive.work_dir` (default
`./fc3_hiphive_sinw100_d5a_vasp`). Move it into the canonical reap tree:

```bash
mv fc3_hiphive_sinw100_d5a_vasp reaps/hiphive_sinw100_d5a_vasp
```

### Pre-flight gates (must all pass)

1. `python -m finite_analysis validate configs/sinw/sinw100_d5a_vasp.yaml` exits 0 or 1 (no errors).
2. After running the analyses (next step), inspect
   `out/sinw100/physical/physical.json`:
   - `dispersion.n_imaginary == 0` (no FC2 instabilities), or document why.
   - `fc3_asr_legs.leg_j_rel < 0.05` and `leg_k_rel < 0.05`.
   - `fc2_psd.n_neg_eigvals == 0`.

### Validation phase (minutes)

```bash
python -m finite_analysis \
    --config configs/sinw/sinw100_d5a_vasp.yaml \
    --fc3-path reaps/hiphive_sinw100_d5a_vasp/fc3.hdf5 \
    --out-dir document/src/fig/finite_analysis/sinw100 \
    --analyses all \
    --rank-sweep 2,4,8,16,32            # PCP off by default; pass --include-pcp to audit it

```

### Figures into results.tex

| File (under `document/src/fig/finite_analysis/sinw100/`) | results.tex subsection |
|---|---|
| `fc_quality/sparsity_fc2_heatmap.pdf` | FC2/FC3 sparsity |
| `fc_quality/sparsity_fc3_decay_1d.pdf` | FC2/FC3 sparsity |
| `fc_quality/sparsity_fc3_scatter_3d.pdf` | FC2/FC3 sparsity |
| `fc_quality/fc_quality_dispersion.pdf` | Phonon spectrum sanity |
| `fc_quality/fc_quality_fc2_distance.pdf` | FC convergence |
| `fc_quality/fc_quality_fc3_distance.pdf` | FC convergence |
| `decomposition/decomp_frob_vs_params.pdf` | Compressibility |
| `decomposition/decomp_nnz_vs_eps.pdf` | Compressibility |
| `sse_sparsity/sse_sigma_heatmap_synth.pdf` | Σ block structure |
| `cutoffs/cutoffs_sigma_frob_vs_cutoff.pdf` | Cutoff sensitivity |
| `cutoffs/cutoffs_transport.pdf` | Cutoff sensitivity |

## 2. Si chain — pedagogical fixture

### Cluster phase (~30 min, single node)

```bash
cd phonon/
python -m phonon_inputs pipeline --config configs/chain/si_chain.yaml
mv fc3_si_chain_vasp reaps/si_chain_vasp
```

Or simply run the in-memory analytic fixture (no DFT) — covered by
`tests/phonon_inputs/finite_analysis/test_chain_pipeline.py`.

### Validation phase

```bash
python -m finite_analysis \
    --config configs/chain/si_chain.yaml \
    --fc3-path reaps/si_chain_vasp/fc3.hdf5 \
    --out-dir document/src/fig/finite_analysis/si_chain \
    --analyses all
```

### Pre-flight gates

Same as SiNW. Chain has no LA softening so `n_imaginary == 0` should be tight.

### Figures

The chain produces the cleanest pedagogical FC3-decay plot
(`sparsity_fc3_decay_1d.pdf`) and is also where the
`test_synthetic_gf_correlator_normalisation` test's analytic agreement
plot belongs (proving the Lorentzian prefactor; produced by the test
itself, not the CLI — copy from `tests/.../*.png` if visualising).

## 3. CNT(3,3) finite — sp² test

### Setup + cluster phase (~12 h on 4 nodes)

```bash
cd phonon/
# Generate the finite CNT YAML if not yet present:
python examples/setup_cnt_finite.py --n-cells 4
#   Writes: configs/cnt/cnt33_finite_n4_vasp.yaml

# Override ENCUT for carbon (validator will warn):
# Edit configs/cnt/cnt33_finite_n4_vasp.yaml :: vasp.encut → 520

python -m phonon_inputs pipeline --config configs/cnt/cnt33_finite_n4_vasp.yaml
mv fc3_cnt33_finite_n4_vasp reaps/cnt33_finite_n4_vasp
```

### Validation phase

```bash
python -m finite_analysis \
    --config configs/cnt/cnt33_finite_n4_vasp.yaml \
    --fc3-path reaps/cnt33_finite_n4_vasp/fc3.hdf5 \
    --out-dir document/src/fig/finite_analysis/cnt33_finite \
    --analyses all
```

### Pre-flight

Validator will warn re: ENCUT (default 400 eV is below 1.3 × C ENMAX = 520 eV).
Bump to 520 eV before running DFT.

## 4. Cross-system overlays

After all three single-system analyses produce their
`document/src/fig/finite_analysis/<system>/` trees, run

```bash
cd phonon/
python -m finite_analysis.cross_system_plots \
    --systems sinw100,si_chain,cnt33_finite \
    --base-dir ../document/src/fig/finite_analysis \
    --out-dir ../document/src/fig/finite_analysis
```

Produces three cross-system PDFs:

- `cross_fc3_decay.pdf` — FC3 magnitude vs triplet-diameter, all three systems on one log-y panel.
- `cross_decomposition_ranks.pdf` — relative Frobenius decomposition error vs parameter count, three systems × six methods.
- `cross_summary_table.pdf` — one-page table of headline numbers (n_atoms, FC3 max, ASR residual, decomposition rank for 1 % error).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `validate` reports `vasp.encut` warn | POTCAR ENMAX higher than 1.3 × encut | Bump `vasp.encut` |
| `validate` reports `kpoints_scf` warn along periodic axis | Under 4 along that axis | Bump kpoint count |
| `physical.dispersion.n_imaginary > 0` | FC2 indefinite | Re-relax structure with tighter `forc_conv_thr`; check `fc2_psd.min_eigval` for the worst mode |
| `physical.fc3_asr_legs.leg_j_rel > 0.05` | hiphive fit broke ASR | Either accept (Σ will carry Drude weight at ω→0) or post-ASR-project FC3 in a custom step |
| Cutoff sweep too slow | bubble is O(n_blocks⁴ × n_freq²) | Reduce `--n-freq-pos` (chain works at 32; SiNW at 64); cap rank sweep |
| `sse_sparsity --quatrex` fails to import | quatrex stack not on PYTHONPATH | Pass `--skip-quatrex-run` (synthetic-GF analysis still runs) |
