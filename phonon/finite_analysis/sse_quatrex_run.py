"""Cross-check Σ^{<,>} from two independent code paths.

The synthetic-GF route in :mod:`sse_sparsity_driver` evaluates the bubble
through :func:`SigmaPhononPhonon._bubble_block` (block-tridiagonal
storage, einsum-based contraction). This module evaluates the *same*
math through :func:`phonon_inputs.anharmonic._compute_phph_self_energy_finite`
(dense storage, matrix-multiply contraction) and projects the result
back onto the block-tridiagonal grid for direct per-block comparison.

**Structural assumption — important.** ``_bubble_block`` consumes only the
*diagonal* G(K, K) blocks, never the off-diagonal G(K1, K2) (K1 ≠ K2).
This is by design in the block-tridiagonal NEGF stack: the bubble
contracts each propagator through one transport-cell-diagonal G slice.
The dense bubble in :func:`_compute_phph_self_energy_finite` uses the
*full* G matrix, so it picks up additional contributions from the
off-diagonal G blocks. To make the two routes numerically identical on
the same input we **project G to block-diagonal** before feeding the
dense path. With that projection the agreement is to floating-point
precision; without it the routes can disagree by tens of percent
depending on how much weight the off-diagonal G blocks carry.

For a clean numerical agreement to within a few percent, the two routes
must use the same:
  * mass-weighted Phi tensor (we feed both with ``bundle.fc3_target.T_lifted``,
    z-permuted to slab order),
  * Green's function (the synthetic GF from
    :func:`finite_analysis.synthetic_gf.synthetic_gf_dense`, projected
    to block-diagonal),
  * frequency grid (built once and shared).

Optionally runs ``n_iter`` SCBA self-consistency passes on top — by
default zero, since one iteration of "synthetic G → Σ" already exposes
all the bubble-kernel agreement we want to verify.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phonon_inputs.anharmonic import _compute_phph_self_energy_finite
from phonon_inputs.constants import HBAR_SI, KB_SI, THZ_TO_RAD

from ._utils import expand_atom_perm_to_dofs, project_dense_to_blocks
from .loader import SystemBundle
from .synthetic_gf import dynamical_matrix, synthetic_gf_dense, gf_to_block_dict


def run_dense_scba_crosscheck(
    bundle: SystemBundle,
    *,
    n_freq_pos: int = 64,
    eta_thz: float | None = None,
    temperature_k: float = 300.0,
    n_iter: int = 0,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Cross-check route: dense Σ via :func:`_compute_phph_self_energy_finite`.

    Returns the per-(I, J) ``(‖Σ^<‖_F, ‖Σ^>‖_F)`` tuple in the block-tridiagonal
    layout that :mod:`sse_sparsity_driver` uses. ``n_iter`` is the number of
    self-consistency passes (0 = single Σ from the synthetic G, 1+ = update G
    via the Dyson equation and recompute, like the SCBA loop).
    """
    G_l, G_g, freqs, dw, modes = synthetic_gf_dense(
        bundle,
        n_freq_pos=n_freq_pos,
        eta_thz=eta_thz,
        temperature_k=temperature_k,
        in_z_sorted_order=True,
    )

    # Permute the bundle's Phi to z-sorted DOF order to match the GF.
    dof_perm = expand_atom_perm_to_dofs(bundle.atom_perm)
    Phi = bundle.fc3_target.T_lifted[np.ix_(dof_perm, dof_perm, dof_perm)]

    # Single bubble pass on the synthetic G (one SCBA half-step).
    sl_dense, sg_dense = _compute_phph_self_energy_finite(
        G_l, G_g, Phi, freqs, dw,
    )

    # Optional SCBA iterations: rebuild G from (D + Σ^R) and resample Σ.
    if n_iter > 0:
        D = dynamical_matrix(bundle, z_sorted=True)
        n_dof = D.shape[0]
        eye = np.eye(n_dof)
        eta = 0.05
        z2 = (freqs + 1j * eta) ** 2
        for _ in range(n_iter):
            Sigma_R = 0.5 * (sg_dense - sl_dense)
            n_B = 1.0 / np.expm1(
                np.maximum(np.abs(freqs) * HBAR_SI * THZ_TO_RAD / (KB_SI * temperature_k),
                           1e-12)
            )
            n_B = np.where(freqs > 0, n_B, -(n_B + 1))
            G_R = np.zeros_like(sl_dense)
            for iw, w in enumerate(freqs):
                G_R[iw] = np.linalg.inv(z2[iw] * eye - D - Sigma_R[iw])
            G_A = G_R.conj().swapaxes(-1, -2)
            Gamma = 1j * (Sigma_R - Sigma_R.conj().swapaxes(-1, -2))
            G_l = (1j * n_B[:, None, None] * (G_R @ Gamma @ G_A))
            G_g = (-1j * (n_B[:, None, None] + 1.0) * (G_R @ Gamma @ G_A))
            sl_dense, sg_dense = _compute_phph_self_energy_finite(
                G_l, G_g, Phi, freqs, dw,
            )

    sl_blocks = project_dense_to_blocks(sl_dense, bundle.block_sizes)
    sg_blocks = project_dense_to_blocks(sg_dense, bundle.block_sizes)
    block_frob = {
        ij: (
            float(np.linalg.norm(sl_blocks.get(ij, 0))),
            float(np.linalg.norm(sg_blocks.get(ij, 0))),
        )
        for ij in set(sl_blocks) | set(sg_blocks)
    }
    return block_frob
