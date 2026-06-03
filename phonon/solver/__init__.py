""" Dense NEGF/SSE solver for (anharmonic) phonon transport.

Public API
----------

* :func:`transmission`         -- unified SCBA driver (q-mesh + n_slabs).
* :func:`transmission_finite`  -- wrapper: q_mesh=(1,1) (Gamma-only device).
* :func:`transmission_q`       -- wrapper: transversely-periodic q-mesh.
* :func:`scba_loop`            -- shared fixed-point loop.
* :func:`gamma_project_M_blocks` -- Gamma-only supercell->primitive
  projection of the FC3 vertex.
* :func:`compare_q11_to_finite` -- regression check.
* :func:`bubble_dense`         -- FFT 3-phonon bubble kernel.
* :func:`build_retarded`       -- Sigma^R reconstruction (``half|pv|fft``).
* :func:`sancho_rubio`,
  :func:`sancho_rubio_batch`,
  :func:`compute_obc_batch`    -- lead self-energy helpers.
* :func:`build_frequency_grid`,
  :func:`bose_full_axis`       -- frequency-axis utilities.
"""

from .bubble import bubble_dense
from .cutoffs import CutoffPolicy, apply_fc3_cutoffs, diagonalise_g_blocks
from .dense import (
    compare_q11_to_finite,
    gamma_project_M_blocks,
    load_fc3_raw,
    scba_loop,
    transmission,
    transmission_finite,
    transmission_q,
)
from .diagnostics import (
    check_broadening_sign,
    check_full_axis_symmetry,
    symmetrize_lesser_greater,
)
from .grids import (
    bose_full_axis,
    boson_contact_self_energies_from_gamma,
    build_frequency_grid,
)
from .leads import (
    ballistic_transmission_z2,
    build_device_hamiltonian,
    compute_obc_batch,
    sancho_rubio,
    sancho_rubio_batch,
    solve_green_batch,
    solve_green_functions,
)
from .retarded import build_retarded, hilbert_transform_axis
from .se_finite import (
    compute_phph_self_energy,
    compute_phph_self_energy_finite_multi_slab,
)
from .se_q import compute_phph_self_energy_q_dense_multi_slab
from .zero_modes import (
    build_dynamical_zero_mode_projector,
    build_translation_projector,
    project_self_energy,
    translation_leakage,
    translation_vectors,
)

__all__ = [
    # Public entry points
    "transmission",
    "transmission_finite",
    "transmission_q",
    "scba_loop",
    "compare_q11_to_finite",
    "gamma_project_M_blocks",
    "load_fc3_raw",
    # Bubble kernel
    "bubble_dense",
    # Cutoff policy
    "CutoffPolicy",
    "apply_fc3_cutoffs",
    "diagonalise_g_blocks",
    # Retarded reconstruction
    "build_retarded",
    "hilbert_transform_axis",
    # Leads / OBC
    "sancho_rubio",
    "sancho_rubio_batch",
    "build_device_hamiltonian",
    "compute_obc_batch",
    "solve_green_functions",
    "solve_green_batch",
    "ballistic_transmission_z2",
    # Grids / Bose
    "build_frequency_grid",
    "bose_full_axis",
    "boson_contact_self_energies_from_gamma",
    # Diagnostics
    "check_broadening_sign",
    "check_full_axis_symmetry",
    "symmetrize_lesser_greater",
    # Self-energy kernel (unified) + back-compat wrappers
    "compute_phph_self_energy",
    "compute_phph_self_energy_finite_multi_slab",
    "compute_phph_self_energy_q_dense_multi_slab",
    # Zero-mode (rigid-translation) handling
    "build_translation_projector",
    "build_dynamical_zero_mode_projector",
    "project_self_energy",
    "translation_vectors",
    "translation_leakage",
]
