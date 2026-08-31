# Phonon code catalogue

Status: refactor baseline after merging `upstream/dev` at `9d957a9f`,
2026-08-31.

This catalogue records the intended ownership of the branch-only code before
it is simplified. The production target is the unmodified three-phonon SCBA
interaction on a resolved frequency grid and with spatial support large enough
to contain the interaction. Stabilising masks, artificial broadening and
continuation schedules are not part of that target.

## Production core

These pieces implement the physical path and should remain in `src/` after
being shortened and tested.

| Area | Files | Decision |
|---|---|---|
| Phonon Dyson solver | `src/quatrex/phonon/solver.py` | Keep the harmonic operator, spectral OBC, Bose contacts, selected solve and heat current. Remove the experimental branches listed below. |
| Three-phonon self-energy | `src/quatrex/phonon/sse_phonon_phonon.py` | Keep the raw FFT bubble, complete absorption/emission fold, dense coupled-q contraction, exact four-ring identity and distributed execution. Split this file into small components. |
| Bubble kernels | `bubble.py`, `bubble_factored.py` | Keep the vectorized NumPy/CuPy kernels. A factored vertex is valid only when its source representation is certified; rank truncation is not a production default. |
| Vertex input | `fc3_loader.py`, `qfold.py`, `vertex_factors.py`, `vertex_q_resample.py` | Keep loading, q folding and exact representation transforms. Approximate factor fits remain opt-in and must report their error. |
| Spatial layout | `microblocks.py` | Keep the exact grouped-Dyson/primitive-bubble layout. This is the preferred route to retain the required physical support without paying for dense merged FC3 blocks. |
| Units and grids | `units.py`, `quatrex/grid/energies.py` | Keep the THz/eV conversions and integration weights. Production uses a resolved uniform convolution grid. |
| SCBA integration | `core/interaction.py`, `core/scba.py`, `core/config.py` | Keep a small phonon interaction adapter, fixed-point convergence and physical diagnostics. Separate phonon code from electron SCBA where practical. |
| Distributed selected solve | `qttools/greens_function_solver/{rgf,rgf_dist}.py` and q distribution changes | Keep the wider selected Green-function blocks required by the FC3 support. Replace the fixed maximum band of three with support derived from the actual vertex/layout. |

Two current approximations must be removed from the core design rather than
hidden behind better defaults:

1. `sse_g_band` is capped at three solver blocks and the generated self-energy
   is pinned to adjacent solver blocks. Reblocking changes both physical
   supports. The new solver must derive Green-function and self-energy support
   from the FC3 block offsets and preserve it exactly.
2. `phonon.interaction_cutoff` defaults to 10 Angstrom. The MoS2 runs show that
   this box mask can break positivity before the self-energy is applied.
   Production should consume the loaded FC3 support as-is; an analysis-only
   cutoff belongs in study code.

## Remove from production

The following paths alter the physical map or were introduced as convergence
workarounds. They should be deleted from the public configuration and runtime.
Tests that merely preserve their old behavior should be removed with them;
negative-result evidence stays in the docs and run manifests.

| Feature | Runtime/config surface | Evidence and action |
|---|---|---|
| Center-of-mass subtraction | `cm_channel.py`, `ir_subtraction.py`, `sse_cm_subtraction`, the hooks in `solver.py`, `interaction.py` and `sse_phonon_phonon.py` | No clean device run converged with it, and it is not part of the requested raw interaction. Remove. |
| Uniform broadening schedules | `eta_ramp_iterations`, `eta_final`, `eta_obc_ramp_iterations`, `eta_obc_final` | Finite broadening biases transport and the reported calculations use zero broadening. Remove the schedules and run at zero broadening. |
| Infrared broadening floor | `eta_ir_floor_cells`, `eta_ir_floor_final_cells`, `eta_ir_floor_ramp_iterations` and solver injection | A grid-dependent stabilizer and annealing path. Remove. |
| Interaction annealing | `sse_ramp_iterations`, `sse_vertex_scale`, cross-slab scaling and continuation hooks | Changes the nonlinear map during iteration. Remove from production. |
| Frequency masks and special mixing | `sse_low_freq_mask_thz`, `low_freq_mixing_thz`, `low_freq_mixing_factor` | Deletes real scattering channels or changes only selected bins. Remove. |
| Spatial and cutoff tapers | `sse_g_band_taper`, `interaction_cutoff_taper` | The tested tapers change the answer and do not repair the underlying support error. Remove. |
| Auxiliary primary grid | `sse_aux_grid_*` runtime path | It moves narrow-line resolution cost to a uniform auxiliary grid and did not pass the CNT conservation gate. Retain only as a study oracle. |
| Static SCP/tadpole additions | `static_self_energy.py` and the SCP/FC4 switches in the three-phonon runtime | These are separate physical models, not the raw cubic interaction. Move to experiments until they have an unbroadened production validation. |
| Advanced root-finder campaign knobs | Broyden, RPM, RRE, JFNK/Newton campaign controls and continuation-specific safeguards | Useful research tools, but not part of the plain production solver. Keep reusable numerical algorithms outside the phonon runtime and expose only after an independent need is established. |

The single zero-frequency bin may still be handled as a quadrature endpoint,
but this should be expressed by the integration rule rather than as a general
masking feature.

## Experimental code

Experimental modules should not be imported, configured or executed by the
default solver. They will be moved under an explicit experimental namespace
with focused oracle tests and a short result note.

| Programme | Current files | Verdict |
|---|---|---|
| Analytic pole sector | `pole_*.py`, `btd_linalg.py`, pole hooks in `solver.py`, `interaction.py`, `sse_phonon_phonon.py`, and `PoleSectorConfig` | The warm bulk broadens but a strongly mixed unresolved tail remains; the simple-pole production route is not reliable. Preserve as an experiment, disabled structurally rather than by a default flag. |
| Passive auxiliary states | `auxiliary_scba.py` | Algebraic tests pass, but real-Si constant-source errors are 10-16 percent and wholesale promotion costs more than reblocking. Preserve as a selective-cluster experiment. |
| Modal and semiseparable spatial tails | `spatial_*.py` | One-sided fitting and post-hoc compression lose to exact reblocking. The direct-generator construction remains an unclosed research idea. Preserve only its oracle and derivation. |
| Adaptive/nonuniform collision integration | study modules under `phonon/studies/` | The reduced P1 oracle is accurate, but the current production bridge is not a faster conserving collision backend. Keep out of runtime. |
| Ballistic and static audits | `ballistic_audit.py` and related study helpers | Move to verification tooling; these are not solver dependencies. |

## Tooling and data

- `phonon/phonon_inputs/` is input-generation tooling. Keep the force-constant
  and structure pipeline, but remove options that only generate deprecated
  masks, broadening schedules or continuation runs.
- `phonon/studies/` and `phonon/scripts/` contain both reusable verification
  and one-off campaign drivers. Retain reproducible oracles and current figure
  generators; move closed campaigns to an attic or delete them when the run
  manifest provides the durable record.
- `phonon/scripts/data/run_manifest_*.csv` is the run catalogue. Its current
  `no-cm-subtraction` failure reason is obsolete and must be removed when the
  runtime subtraction is deleted.
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
