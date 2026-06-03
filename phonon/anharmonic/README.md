# `phonon/anharmonic/`

Production transport scripts and regression tests for the legacy
`phonon_inputs.anharmonic` SCBA path.

The validation-pipeline tooling lives in `phonon/finite_analysis/`
(see `phonon/finite_analysis/RUNBOOK.md` and
`phonon/finite_analysis/RESULTS_RUNBOOK.md`); this directory is
specifically for transport calculations that drive the original
`phonon_inputs.anharmonic.anharmonic_transmission_*` entry points
end-to-end.

## Production drivers

| File | Purpose |
|---|---|
| `run_anharmonic.py` | Full SCBA transport on the Si primitive cell (q-path). Reads `reaps/si_primitive_work/fc3.hdf5` and runs `anharmonic_transmission_q` over a configurable q-mesh. Produces `anharmonic_results.png`. |
| `run_sinw100_phph.py` | SCBA transport on the H-passivated SiNW(100). Reads `reaps/hiphive_sinw100_vasp_larger/fc3.hdf5`. |

## Regression / diagnostic tests

| File | What it pins |
|---|---|
| `test_anharmonic_multislab.py` | Multi-slab SCBA on Si primitive (transport along [011]). Exercises `anharmonic_transmission_finite` with `n_slabs > 1`. |
| `test_ballistic_grid.py` | Ballistic transmission convergence in `(n_freq, n_q)` grid resolution. |
| `test_convolution_correctness.py` | FFT bubble convolution vs analytic Gaussian-Gaussian baseline. |
| `test_self_validation.py` | SCBA self-consistency: $\Sigma$ on input $G$ matches $\Sigma$ on the iterated $G$. |
| `test_waring_diagnostic.py` | Waring-decomposition spectrum on the Si primitive FC3. |

All tests are runnable as standalone scripts from the repo root, e.g.:
```bash
python phonon/anharmonic/test_anharmonic_multislab.py
```

## Relationship with `phonon/finite_analysis/`

`finite_analysis/` is the user-facing validation pipeline (sparsity,
decomposition, physical invariants, SSE structure, cutoff sensitivity,
Sancho–Rubio + SCBA transport with the Hilbert-reconstructed Σ^R).
The five tests here exercise the legacy `phonon_inputs.anharmonic`
machinery directly and serve as integration tests on the underlying
SCBA + transport code that finite_analysis builds on top of.
