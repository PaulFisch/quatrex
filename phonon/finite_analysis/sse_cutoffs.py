"""Cutoff-sweep harness for the 3-phonon bubble.

The bubble integrand

    Σ^{<,>}_{IJ}(ω) = i ℏ / 2 × Σ_{K₁,K₁',K₂,K₂'} Φ^{(I,K₁,K₂)}
                        · G^{<,>}(K₁,K₁')(ω) · G^{<,>}(K₂,K₂')(ω)
                        · Φ^{(J,K₂',K₁')}

is delegated to the canonical kernel :func:`quatrex.phonon.bubble.bubble_dense`
(the same kernel that backs ``transmission_finite``). This module wraps
the kernel with the cutoff policy from :mod:`phonon.solver.cutoffs`
(``diag_G_in_se``, ``fc3_nn_only``, ``fc3_distance_cutoff``,
``fc3_magnitude_threshold``) and lets the caller pick a
``sigma_block_distance`` to opt out of the block-tridiagonal restriction
when auditing how much Σ weight lives off the NN band.

**Frequency-grid convention.** Inputs ride on the symmetric
``[-fmax, ..., -Δω, 0, Δω, ..., fmax]`` axis built by
:func:`phonon.solver.grids.build_frequency_grid`. The ω=0 sample is
zeroed inside ``bubble_dense`` (``zero_freq_idx=mid``) and the IFFT is
sliced ``[mid : mid + ne]`` so Σ lives on the same axis as G.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from quatrex.phonon.bubble import bubble_dense
from quatrex.phonon.units import bubble_prefactor_thz

from solver.cutoffs import (
    CutoffPolicy,
    apply_fc3_cutoffs as _apply_fc3_cutoffs_policy,
    diagonalise_g_blocks as _diagonalise_g_blocks_policy,
)

from ._utils import min_image_distance_matrix
from .loader import SystemBundle
from .synthetic_gf import gf_to_block_dict


# --------------------------------------------------------------------------- #
# Phi cutoffs                                                                 #
# --------------------------------------------------------------------------- #


def atomic_distances_z_sorted(bundle: SystemBundle) -> np.ndarray:
    """Pairwise atomic distances in Å, in z-sorted atom order."""
    perm = bundle.atom_perm
    pos = bundle.sc_positions[perm]
    return min_image_distance_matrix(pos, bundle.sc_cell)


def _dof_to_atom(dof_idx: int) -> int:
    return dof_idx // 3


def apply_fc3_cutoffs(
    phi_blocks: dict[tuple[int, int, int], np.ndarray],
    block_sizes: np.ndarray,
    *,
    distances_atom: np.ndarray | None = None,
    distance_cutoff_A: float | None = None,
    magnitude_threshold: float | None = None,
    fc3_nn_only: bool = False,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Compatibility wrapper around :func:`phonon.solver.cutoffs.apply_fc3_cutoffs`.

    Translates the legacy keyword surface (``distance_cutoff_A``,
    ``magnitude_threshold``) into a :class:`CutoffPolicy`.
    """
    policy = CutoffPolicy(
        fc3_nn_only=fc3_nn_only,
        fc3_distance_cutoff=distance_cutoff_A,
        fc3_magnitude_threshold=magnitude_threshold,
    )
    return _apply_fc3_cutoffs_policy(
        phi_blocks, block_sizes,
        policy=policy, distances_atom=distances_atom,
    )


# --------------------------------------------------------------------------- #
# G cutoffs                                                                   #
# --------------------------------------------------------------------------- #


def diagonalise_g_blocks(
    g_blocks: dict[tuple[int, int], np.ndarray],
    *,
    keep_diag_blocks_only: bool = True,
) -> dict[tuple[int, int], np.ndarray]:
    """Return a copy where every diagonal G(K,K) block is masked to its
    DOF-diagonal. Off-diagonal (K, K') blocks are dropped if
    ``keep_diag_blocks_only`` (default), otherwise zeroed.

    Note: the canonical block-level projection used by the unified
    solver is :func:`phonon.solver.cutoffs.diagonalise_g_blocks`, which
    simply drops the off-(K, K') entries without zeroing the diagonal
    block's off-DOF-diagonal entries. The extra DOF-diagonal masking
    here is preserved for the legacy ``diag_G_everywhere`` audit; new
    callers should prefer the policy version.
    """
    out: dict[tuple[int, int], np.ndarray] = {}
    for (I, J), block in g_blocks.items():
        if I == J:
            n_dof = block.shape[1]
            mask = np.eye(n_dof, dtype=bool)
            new = np.zeros_like(block)
            new[:, mask] = block[:, mask]
            out[(I, J)] = new
        elif not keep_diag_blocks_only:
            out[(I, J)] = np.zeros_like(block)
        # else: drop
    return out


# --------------------------------------------------------------------------- #
# Bubble driver                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class SSEResult:
    sigma_lesser: dict[tuple[int, int], np.ndarray]
    sigma_greater: dict[tuple[int, int], np.ndarray]
    block_frob: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict)


def _bubble_block_standalone(
    phi_left: np.ndarray, phi_right: np.ndarray,
    G_inner_a: np.ndarray, G_inner_b: np.ndarray,
    n_fft: int, prefactor: complex,
) -> np.ndarray:
    """Bubble block on the symmetric ω-axis.

    Thin wrapper over the canonical kernel
    :func:`quatrex.phonon.bubble.bubble_dense` that pins the dense
    reference convention (zero the ω=0 sample before FFT, slice
    ``[mid : mid + ne]`` from the IFFT). The historical inline einsum
    implementation has been deleted — see git history pre-Phase 3.
    """
    ne = G_inner_a.shape[0]
    if ne % 2 != 1:
        raise ValueError(
            f"Symmetric-grid bubble requires odd ne; got ne={ne}. "
            "Use _build_frequency_grid (which always returns odd) to avoid this."
        )
    mid = ne // 2
    return bubble_dense(
        phi_left=phi_left,
        phi_right=phi_right,
        G_a=G_inner_a,
        G_b=G_inner_b,
        n_fft=n_fft,
        prefactor=prefactor,
        out_slice=slice(mid, mid + ne),
        zero_freq_idx=mid,
    )


def compute_sse_with_cutoffs(
    phi_blocks: dict[tuple[int, int, int], np.ndarray],
    g_lesser_blocks: dict[tuple[int, int], np.ndarray],
    g_greater_blocks: dict[tuple[int, int], np.ndarray],
    block_sizes: np.ndarray,
    dw_thz: float,
    *,
    policy: CutoffPolicy | None = None,
    diag_G_in_se: bool = False,
    diag_G_everywhere: bool = False,
    fc3_nn_only: bool = True,
    fc3_distance_cutoff: float | None = None,
    fc3_magnitude_threshold: float | None = None,
    distances_atom: np.ndarray | None = None,
    sigma_block_distance: int = 1,
) -> SSEResult:
    """Compute Σ^{<,>} blocks under the requested cutoffs.

    Either pass a :class:`CutoffPolicy` (preferred — the unified solver
    API) or the legacy individual keyword arguments. Mixing the two is
    not supported: when ``policy`` is given the four legacy
    cutoff kwargs are ignored. ``diag_G_everywhere`` and
    ``sigma_block_distance`` remain top-level kwargs because they govern
    the bubble *driver*, not the FC3 vertex.

    ``sigma_block_distance`` sets the largest ``|I - J|`` for which Σ blocks
    are produced (default ``1`` → block-tridiagonal, matching the NEGF
    transport solver's assumption). Pass a larger value to *audit* how much
    of the bubble lives in off-tridiagonal Σ; the result still respects the
    NN-tridiagonal phi support, so contributions above ``|I - J| > 2`` are
    only non-zero through G blocks of finite range.

    The order of operations:
      1. Apply FC3 cutoffs (NN, distance, magnitude) to ``phi_blocks``.
      2. Apply diagonal-G projection if requested.
      3. Loop over (I, J) with ``|I - J| <= sigma_block_distance`` and
         accumulate the bubble contribution from every (K1, K2) triplet
         that has a (J, K2, K1) partner.
    """
    if policy is not None:
        diag_G_in_se = policy.diag_G_in_se
        fc3_nn_only = policy.fc3_nn_only
        fc3_distance_cutoff = policy.fc3_distance_cutoff
        fc3_magnitude_threshold = policy.fc3_magnitude_threshold

    if diag_G_everywhere:
        gl = diagonalise_g_blocks(g_lesser_blocks, keep_diag_blocks_only=True)
        gg = diagonalise_g_blocks(g_greater_blocks, keep_diag_blocks_only=True)
    elif diag_G_in_se:
        gl = diagonalise_g_blocks(g_lesser_blocks, keep_diag_blocks_only=False)
        gg = diagonalise_g_blocks(g_greater_blocks, keep_diag_blocks_only=False)
    else:
        gl = dict(g_lesser_blocks)
        gg = dict(g_greater_blocks)

    block_sizes = np.asarray(block_sizes, dtype=int)
    n_blocks = block_sizes.size

    # DOF-order alignment: phi_blocks come from load_quatrex_blocks which
    # always z-sorts via bundle.atom_perm. The G blocks must use the same
    # ordering — check by matching block shapes against block_sizes.
    if g_lesser_blocks:
        sample_key = next(iter(g_lesser_blocks))
        sample_block = g_lesser_blocks[sample_key]
        I, J = sample_key
        expected_shape = (block_sizes[I], block_sizes[J])
        if sample_block.shape[1:] != expected_shape:
            raise ValueError(
                f"G block {sample_key} has DOF shape {sample_block.shape[1:]} "
                f"but block_sizes implies {expected_shape}. The G blocks must "
                "be in the same z-sorted DOF order as the phi blocks — pass "
                "in_z_sorted_order=True to synthetic_gf_dense (the default)."
            )

    # Filter phi_blocks for NN constraint (already enforced by upstream
    # loader, but reapply here when caller relaxes / supplies dense triplets).
    phi = {
        k: v for k, v in phi_blocks.items()
        if not fc3_nn_only or (
            abs(k[0] - k[1]) <= 1 and abs(k[0] - k[2]) <= 1
            and abs(k[1] - k[2]) <= 1
        )
    }
    phi = apply_fc3_cutoffs(
        phi, block_sizes,
        distances_atom=distances_atom,
        distance_cutoff_A=fc3_distance_cutoff,
        magnitude_threshold=fc3_magnitude_threshold,
    )

    # Build (I, J) -> [(K1, K2, K1', K2', phi_left, phi_right)] enumerating
    # the FULL bubble. For each phi_left = phi[(I, K1, K2)] and
    # phi_right = phi[(J, K2', K1')] we sum over all (K1', K2') for which
    # G(K1, K1') and G(K2, K2') are present in the supplied G dict — that
    # is, we iterate over the full support of G rather than restricting to
    # K1 = K1' / K2 = K2' (the original quatrex SigmaPhononPhonon behaviour,
    # which can be recovered with ``diag_G_in_se=True``). Any (J, K2', K1')
    # not present in phi_blocks is silently skipped (zero contribution).
    g_keys = set(gl.keys())
    if not diag_G_in_se and not diag_G_everywhere:
        # The "off-diagonal-G" path (default) requires the G dict to cover at
        # least every NN-tridiagonal (K, K') pair, otherwise we silently miss
        # bubble contributions. ``synthetic_gf.gf_to_block_dict(..., nn_only=
        # False)`` provides the full (n_blocks × n_blocks) coverage; ``True``
        # gives only NN. Here we accept either NN-tridiagonal *or* full.
        # When ``diag_G_everywhere=True`` the diagonalisation above intentionally
        # strips the off-diagonals, so the check is skipped for that config.
        nn_required = {
            (K, Kp) for K in range(n_blocks)
            for Kp in range(max(0, K - 1), min(n_blocks, K + 2))
        }
        missing = nn_required - g_keys
        if missing:
            raise ValueError(
                "compute_sse_with_cutoffs(diag_G_in_se=False) needs the G dict "
                "to cover every NN-tridiagonal (K, K') pair. Missing: "
                f"{sorted(missing)[:8]}{'...' if len(missing) > 8 else ''}. "
                "Use gf_to_block_dict(..., nn_only=False) (or True) to build G."
            )
    d_sigma = max(0, int(sigma_block_distance))
    pair_index: dict[tuple[int, int], list] = {}
    for (I, K1, K2), phi_left in phi.items():
        for J in range(max(0, I - d_sigma), min(n_blocks, I + d_sigma + 1)):
            k1_range = (K1,) if diag_G_in_se else range(n_blocks)
            for K1_prime in k1_range:
                if (K1, K1_prime) not in g_keys:
                    continue
                k2_range = (K2,) if diag_G_in_se else range(n_blocks)
                for K2_prime in k2_range:
                    if (K2, K2_prime) not in g_keys:
                        continue
                    phi_right = phi.get((J, K2_prime, K1_prime))
                    if phi_right is None:
                        continue
                    pair_index.setdefault((I, J), []).append(
                        (K1, K2, K1_prime, K2_prime, phi_left, phi_right)
                    )

    n_freq = next(iter(gl.values())).shape[0]
    n_fft = 2 * n_freq - 1
    prefactor = bubble_prefactor_thz(dw_thz)

    sl_out: dict[tuple[int, int], np.ndarray] = {}
    sg_out: dict[tuple[int, int], np.ndarray] = {}
    for (I, J), pairs in pair_index.items():
        for K1, K2, K1p, K2p, phi_left, phi_right in pairs:
            for gx, sx_dict in (
                (gl, sl_out), (gg, sg_out),
            ):
                block = _bubble_block_standalone(
                    phi_left=phi_left, phi_right=phi_right,
                    G_inner_a=gx[(K1, K1p)], G_inner_b=gx[(K2, K2p)],
                    n_fft=n_fft, prefactor=prefactor,
                )
                sx_dict[(I, J)] = sx_dict.get(
                    (I, J), np.zeros_like(block)
                ) + block

    block_frob = {
        ij: (
            float(np.linalg.norm(sl_out.get(ij, 0))),
            float(np.linalg.norm(sg_out.get(ij, 0))),
        )
        for ij in set(sl_out) | set(sg_out)
    }
    return SSEResult(sigma_lesser=sl_out, sigma_greater=sg_out, block_frob=block_frob)


# --------------------------------------------------------------------------- #
# Sweep driver                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class CutoffConfig:
    label: str
    diag_G_in_se: bool = False
    diag_G_everywhere: bool = False
    fc3_nn_only: bool = True
    fc3_distance_cutoff: float | None = None
    fc3_magnitude_threshold: float | None = None
    sigma_block_distance: int = 1


def standard_cutoff_grid() -> list[CutoffConfig]:
    """The default cutoff grid documented in the plan."""
    return [
        CutoffConfig(label="baseline"),
        CutoffConfig(label="diag_G_in_se", diag_G_in_se=True),
        CutoffConfig(label="diag_G_everywhere", diag_G_everywhere=True),
        CutoffConfig(label="mag_thresh_1e-2", fc3_magnitude_threshold=1e-2),
        CutoffConfig(label="mag_thresh_1e-3", fc3_magnitude_threshold=1e-3),
        CutoffConfig(label="mag_thresh_1e-4", fc3_magnitude_threshold=1e-4),
    ]


def run_sse_cutoffs(
    bundle: SystemBundle,
    g_lesser_blocks: dict[tuple[int, int], np.ndarray],
    g_greater_blocks: dict[tuple[int, int], np.ndarray],
    phi_blocks: dict[tuple[int, int, int], np.ndarray],
    dw_thz: float,
    *,
    cutoff_grid: Sequence[CutoffConfig] | None = None,
) -> dict[str, SSEResult]:
    """Run :func:`compute_sse_with_cutoffs` for every entry in the grid."""
    if cutoff_grid is None:
        cutoff_grid = standard_cutoff_grid()
    distances = atomic_distances_z_sorted(bundle)
    results: dict[str, SSEResult] = {}
    for cfg in cutoff_grid:
        results[cfg.label] = compute_sse_with_cutoffs(
            phi_blocks, g_lesser_blocks, g_greater_blocks,
            bundle.block_sizes, dw_thz,
            diag_G_in_se=cfg.diag_G_in_se,
            diag_G_everywhere=cfg.diag_G_everywhere,
            fc3_nn_only=cfg.fc3_nn_only,
            fc3_distance_cutoff=cfg.fc3_distance_cutoff,
            fc3_magnitude_threshold=cfg.fc3_magnitude_threshold,
            distances_atom=distances,
            sigma_block_distance=cfg.sigma_block_distance,
        )
    return results


# --------------------------------------------------------------------------- #
# Input assembly                                                              #
# --------------------------------------------------------------------------- #


def build_sse_inputs(
    bundle: SystemBundle,
    *,
    n_freq_pos: int,
    eta_thz: float | None,
    temperature_k: float,
    truncation_warn: float | None = None,
    asr_project: bool = False,
):
    """Build the standard inputs (phi blocks, full G blocks, frequency grid)
    that both :func:`compute_sse_with_cutoffs` and :func:`run_sse_cutoffs`
    consume.

    Both `gl_blocks` and `gg_blocks` are returned with ``nn_only=False`` so
    the bubble can pick up contributions from off-tridiagonal G blocks; the
    cutoff sweep restricts via the ``diag_G_in_se`` / ``diag_G_everywhere``
    flags as needed.

    Set ``asr_project=True`` to ASR-project FC3 along legs 2 and 3 before
    the block-tridiagonal cut — required when the upstream FC3 fit
    violates ASR (cf. :data:`constants.ASR_REL_RESIDUAL_WARN`); otherwise
    Σ(ω→0) carries a spurious Drude-like weight.
    """
    from .constants import SSE_TRUNCATION_WARN
    from .loader import load_quatrex_blocks
    from .synthetic_gf import synthetic_gf_dense

    if truncation_warn is None:
        truncation_warn = SSE_TRUNCATION_WARN
    phi_blocks = load_quatrex_blocks(
        bundle, truncation_warn=truncation_warn, asr_project=asr_project,
    )
    G_l, G_g, freqs, dw, modes = synthetic_gf_dense(
        bundle, n_freq_pos=n_freq_pos, eta_thz=eta_thz,
        temperature_k=temperature_k,
    )
    gl_blocks = gf_to_block_dict(G_l, bundle.block_sizes, nn_only=False)
    gg_blocks = gf_to_block_dict(G_g, bundle.block_sizes, nn_only=False)
    return phi_blocks, gl_blocks, gg_blocks, freqs, dw, modes


def block_frob_diff(reference: SSEResult, other: SSEResult) -> dict[tuple[int, int], float]:
    """Per-(I,J) ‖Σ^< - Σ^<_ref‖_F / ‖Σ^<_ref‖_F."""
    out: dict[tuple[int, int], float] = {}
    for ij in set(reference.sigma_lesser) | set(other.sigma_lesser):
        a = reference.sigma_lesser.get(ij)
        b = other.sigma_lesser.get(ij)
        ref_norm = float(np.linalg.norm(a)) if a is not None else 0.0
        if a is None and b is None:
            out[ij] = 0.0
        elif a is None:
            out[ij] = float("inf")
        elif b is None:
            out[ij] = 1.0
        else:
            diff = float(np.linalg.norm(a - b))
            out[ij] = diff / (ref_norm or 1.0)
    return out
