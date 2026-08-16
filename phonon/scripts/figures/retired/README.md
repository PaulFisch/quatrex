# Retired figure generators

`make_all.py` scans `phonon/scripts/figures/*.py` (flat, non-underscore) and
runs every one. These produce figures the report no longer references, so
leaving them in that directory meant the gate failed on every run for
outputs nobody reads.

| script | produced | why it is here |
|---|---|---|
| `eta0_knob_ablation.py` | `eta0_knob_ablation`, `eta0_knob_sensitivity` | the IR-taper knob study. The 2026-07-06 spectral-deformation audit found the taper unphysical (+33 % on G against the grid-converged bare value), so the ablation's framing is superseded; results section 4.2 reports it as a negative result instead |
| `d5a_grid_ladder.py` | `d5a_grid_ladder` | its dynamical-matrix inputs under `studies/out/d5a_gridladder/` were purged; the surviving grid claims are results section 3.3 |
| `conservation_diagnosis.py` | `--out` CLI target | never a report generator: it requires `--bare` and `--dressed` run directories and was only ever a diagnostic |

Nothing here is deleted. Move a script back up one level to put it under the
gate again.
