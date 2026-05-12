"""Unified dense NEGF/SSE solver for phonon transport.

This is the canonical home for the dense reference solver. The
production block-sparse / MPI variant lives at
:mod:`quatrex.phonon.sse_phonon_phonon` and imports the same bubble
kernel from :mod:`quatrex.phonon.bubble` (re-exported here as
:func:`bubble_dense`).

Public API
----------

* :func:`transmission_finite` — Γ-only SCBA driver.
* :func:`transmission_q`       — q-resolved SCBA driver.
* :func:`scba_loop`            — shared fixed-point loop.
* :func:`gamma_project_M_blocks` — Γ-only supercell→primitive
  projection of the FC3 vertex.
* :func:`compare_q11_to_finite` — regression check.
* :func:`bubble_dense`         — FFT 3-phonon bubble kernel.
* :func:`build_retarded`       — Σ^R reconstruction (``half|pv|fft``).
* :func:`sancho_rubio`,
  :func:`sancho_rubio_batch`,
  :func:`compute_obc_batch`    — lead self-energy helpers.
* :func:`build_frequency_grid`,
  :func:`bose_full_axis`       — frequency-axis utilities.
"""

from .bubble import bubble_dense
from .cutoffs import CutoffPolicy, apply_fc3_cutoffs, diagonalise_g_blocks
from .dense import (
    compare_q11_to_finite,
    gamma_project_M_blocks,
    load_fc3_raw,
    scba_loop,
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
from .se_finite import compute_phph_self_energy_finite
from .se_q import compute_phph_self_energy_q_dense

__all__ = [
    # Public entry points
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
    # Self-energy drivers
    "compute_phph_self_energy_finite",
    "compute_phph_self_energy_q_dense",
]
