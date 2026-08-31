# Phonon code catalogue

Status: implemented and regression-tested after merging `upstream/dev` at
`9d957a9f`, 2026-08-31.

This catalogue records the ownership of the branch-only code after the
cleanup. The production target is the unmodified three-phonon SCBA
interaction on a resolved frequency grid and with spatial support large enough
to contain the interaction. Stabilising masks, artificial broadening and
continuation schedules are not part of that target.

## Production core

These pieces implement the retained physical path in `src/`.

| Area | Files | Decision |
|---|---|---|
| Phonon Dyson solver | `src/quatrex/phonon/solver.py` | Keep the harmonic operator, spectral OBC, Bose contacts, selected solve and heat current. Remove the experimental branches listed below. |
| Three-phonon self-energy | `sse_phonon_phonon.py`, `q_contraction.py`, `contraction_support.py` | Keep the raw FFT bubble, complete absorption/emission fold, batched coupled-q contraction, exact four-ring identity and distributed execution. Communication and q contraction now have separate modules. |
| Bubble kernels | `bubble.py`, `bubble_factored.py` | Keep the vectorized NumPy/CuPy kernels. A factored vertex is valid only when its source representation is certified; rank truncation is not a production default. |
| Vertex input | `fc3_loader.py`, `qfold.py`, `vertex_factors.py`, `vertex_q_resample.py` | Keep loading, q folding and exact representation transforms. Approximate factor fits remain opt-in and must report their error. |
| Spatial layout | `microblocks.py` | Keep the exact grouped-Dyson/primitive-bubble layout. This is the preferred route to retain the required physical support without paying for dense merged FC3 blocks. |
| Units and grids | `units.py`, `quatrex/grid/energies.py` | Keep the THz/eV conversions and integration weights. Production uses a resolved uniform convolution grid. |
| SCBA integration | `core/interaction.py`, `core/scba.py`, `core/config.py` | Keep a small phonon interaction adapter, fixed-point convergence and physical diagnostics. Separate phonon code from electron SCBA where practical. |
| Distributed selected solve | `qttools/greens_function_solver/{rgf,rgf_dist}.py` and q distribution changes | Keep band three as the validated production default. Allow other layouts to request their complete support without imposing band three as a solver-wide ceiling. |

Band three remains the production default. It is exact for the retained
adjacent-output contraction in the measured layouts. Generalising the selected
solver does not change those results. It only lets another layout request more
support when its vertex structure requires it.

Two support rules apply to every production layout:

1. Band three remains exact for the measured adjacent-output layouts. The
   selected solver no longer imposes three as a global ceiling, so another
   layout can request the wider Green-function support its FC3 offsets require.
   Unsupported self-energy output ranges must fail construction rather than be
   dropped silently.
2. `phonon.interaction_cutoff` remains an electron-phonon compatibility
   setting, but phonon transport no longer uses it to mask the Green-function
   sparsity. The complete requested block band is built directly. The MoS2
   runs showed that the old 10 Angstrom box mask could break positivity before
   the self-energy was applied.

## Remove from production

The following paths altered the physical map or were introduced as convergence
workarounds. They have been deleted from the public configuration and runtime.
Negative-result evidence remains in the docs and run manifests.

| Feature | Runtime/config surface | Evidence and action |
|---|---|---|
| Center-of-mass subtraction | `cm_channel.py`, `ir_subtraction.py`, `sse_cm_subtraction`, the hooks in `solver.py`, `interaction.py` and `sse_phonon_phonon.py` | No clean device run converged with it, and it is not part of the requested raw interaction. Remove. |
| Uniform broadening schedules | `eta_ramp_iterations`, `eta_final`, `eta_obc_ramp_iterations`, `eta_obc_final` | Finite broadening biases transport and the reported calculations use zero broadening. Remove the schedules and run at zero broadening. |
| Infrared broadening floor | `eta_ir_floor_cells`, `eta_ir_floor_final_cells`, `eta_ir_floor_ramp_iterations` and solver injection | A grid-dependent stabilizer and annealing path. Remove. |
| Interaction annealing | `sse_ramp_iterations`, `sse_vertex_scale`, cross-slab scaling and continuation hooks | Changes the nonlinear map during iteration. Remove from production. |
| Frequency masks and special mixing | `sse_low_freq_mask_thz`, `low_freq_mixing_thz`, `low_freq_mixing_factor` | Deletes real scattering channels or changes only selected bins. Remove. |
| Spatial and cutoff tapers | `sse_g_band_taper`, `interaction_cutoff_taper` | The tested tapers change the answer and do not repair the underlying support error. Remove. |
| Static SCP/tadpole additions | `static_self_energy.py` and the SCP/FC4 switches in the three-phonon runtime | These are separate physical models, not the raw cubic interaction. Move to experiments until they have an unbroadened production validation. |
| Advanced root-finder campaign knobs | Broyden, RPM, RRE, JFNK/Newton campaign controls and continuation-specific safeguards | Useful research tools, but not part of the plain production solver. Keep reusable numerical algorithms outside the phonon runtime and expose only after an independent need is established. |

The single zero-frequency bin may still be handled as a quadrature endpoint,
but this should be expressed by the integration rule rather than as a general
masking feature.

## Experimental code

Experimental implementations live under explicit experimental namespaces and
are loaded only when their disabled-by-default configuration is selected.

| Programme | Current files | Verdict |
|---|---|---|
| Analytic pole sector | `src/quatrex/phonon/experimental/pole/` and its opt-in hooks | The warm bulk broadens but a strongly mixed unresolved tail remains. The simple-pole route is not reliable for production. |
| Passive auxiliary states | `src/quatrex/phonon/experimental/auxiliary_scba.py` | Algebraic tests pass, but real-Si constant-source errors are 10-16 percent and wholesale promotion costs more than reblocking. Preserve as a selective-cluster experiment. |
| Modal and semiseparable spatial tails | `src/quatrex/phonon/experimental/spatial/` | One-sided fitting and post-hoc compression lose to exact reblocking. The direct-generator construction remains an unclosed research idea. Preserve only its oracle and derivation. |
| Adaptive/nonuniform collision integration | study modules under `phonon/studies/` | The reduced P1 oracle is accurate, but the current production bridge is not a faster conserving collision backend. Keep out of runtime. |
| Auxiliary frequency grid | `sse_aux_grid_*` and `_prepare_nonuniform_production.py` | Keep as an opt-in experiment. The Si 113/121 case was accurate, but two CNT cases failed lead conservation and saved little time because the FC3 ring retained the full auxiliary grid. Auxiliary-spacing convergence is still open. |
| Ballistic and static audits | `src/quatrex/phonon/experimental/ballistic_audit.py` and study helpers | Verification tooling only; these are not solver dependencies. |
| SCBA root finders | `src/quatrex/experimental/mixers/` | Broyden, RPM, RRE, JFNK and exact Newton remain research tools. Production linear and Anderson runs load none of their implementations. |

## Tooling and data

- `phonon/phonon_inputs/` is input-generation tooling. Keep the force-constant
  and structure pipeline, but remove options that only generate deprecated
  masks, broadening schedules or continuation runs.
- `phonon/studies/` and `phonon/scripts/` contain both reusable verification
  and one-off campaign drivers. Retain reproducible oracles and current figure
  generators; move closed campaigns to an attic or delete them when the run
  manifest provides the durable record.
- `phonon/scripts/data/run_manifest_*.csv` is the historical run catalogue.
  Its mask, broadening and CM-subtraction columns describe old runs, not
  current production controls.
- Long historical documents belong in `phonon/docs/attic/`. The active docs
  should describe only the retained production method, its convergence
  requirements and the experimental verdicts above.

## Verification contract

Each refactor increment must compare against the current raw path with every
deprecated option disabled. At minimum it must preserve:

1. dense and factored bubble parity for an exact vertex;
2. NumPy/CuPy and serial/distributed-q parity;
3. Keldysh symmetry, retarded assembly and equilibrium detailed balance;
4. ballistic Caroli/Meir-Wingreen agreement;
5. bubble energy balance and internal heat-current continuity;
6. unchanged results when solver blocks are regrouped while physical FC3 and
   Green-function support are held fixed.

The last item is the acceptance test for the solver redesign. A faster result
that changes when the same physical device is reblocked is not a valid
refactor.

## Verification result

The final NumPy regression on 2026-08-31 passed 1,193 tests with 64 expected
skips across phonon, configuration, SCBA and selected-solve modules. The
distributed selected solver passed 45 tests on each of three MPI ranks. The
coupled-q and experimental pole paths passed 10 tests on each of two MPI
ranks. A focused regression also verifies that phonon transport ignores the
legacy geometric cutoff and requests the complete selected block band.
