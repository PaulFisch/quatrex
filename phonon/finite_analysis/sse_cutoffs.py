"""Phonon-phonon SSE under various cutoffs.

The 3-phonon bubble in block-tridiagonal storage is

    Σ^{<,>}_{IJ}(ω) = i ℏ / 2 ×
                  Σ_{K1, K1', K2, K2'}  Φ^{(I,K1,K2)}
                                      · G^{<,>}(K1, K1')(ω)
                                      · G^{<,>}(K2, K2')(ω)
                                      · Φ^{(J, K2', K1')}.

Note the **four**-index inner sum: with G in block-tridiagonal storage
(|K - K'| ≤ 1), each diagonal-pair contribution is augmented by
contributions from the off-diagonal G blocks. The original quatrex
:func:`SigmaPhononPhonon._bubble_block` is the K1 = K1', K2 = K2'
diagonal-only sub-case; this module enumerates the full sum by default
and exposes a ``diag_G_in_se`` flag to recover the diagonal-only
restriction for direct comparison.

Five cutoff knobs are exposed:

  * ``diag_G_in_se``: keep only the diagonal G blocks (K1 = K1', K2 = K2')
    inside the bubble — recovers the original block-tridiagonal NEGF
    approximation.
  * ``diag_G_everywhere``: same restriction applied globally before any
    computation.
  * ``fc3_nn_only``: drop block triplets that are not nearest-neighbour
    (``|I-J|, |I-K|, |J-K| > 1``); already the default of
    :func:`fc3_to_phi_blocks`.
  * ``fc3_distance_cutoff``: drop FC3 entries whose triplet diameter
    (max pairwise distance among the three atoms) exceeds the cutoff.
  * ``fc3_magnitude_threshold``: drop FC3 entries whose magnitude is below
    ``threshold × max|Φ|``.

Outputs are returned as a :class:`SSEResult` carrying the per-(I,J)
Σ^{<,>} block dict so the caller can compare two configurations
block-wise.

**Frequency-grid convention.** This module operates on the symmetric
``[-fmax, ..., -dω, 0, dω, ..., fmax]`` axis built by
:func:`phonon_inputs.anharmonic._build_frequency_grid`. The ω=0 sample is
zeroed before convolution and the IFFT output is sliced ``[mid:mid+ne]``
so the result lives on the same symmetric grid. The legacy one-sided
quatrex convention lives in
:func:`finite_analysis._compat.bubble_block_legacy_quatrex` for
bit-for-bit cross-checks against ``SigmaPhononPhonon._bubble_block``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon
from quatrex.phonon.units import bubble_prefactor_thz

from ._utils import expand_atom_perm_to_dofs, min_image_distance_matrix
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
) -> dict[tuple[int, int, int], np.ndarray]:
    """Return a copy of ``phi_blocks`` with the requested cutoffs applied.

    ``distance_cutoff_A`` and ``magnitude_threshold`` are independent
    masks; both, either, or neither may be active. Empty blocks are
    dropped from the dict.
    """
    if not phi_blocks:
        return {}
    block_sizes = np.asarray(block_sizes, dtype=int)
    offsets = np.concatenate(([0], np.cumsum(block_sizes)))

    if magnitude_threshold is not None:
        max_abs = max(np.abs(b).max() for b in phi_blocks.values()) or 1.0
        mag_floor = magnitude_threshold * max_abs
    else:
        mag_floor = None

    out: dict[tuple[int, int, int], np.ndarray] = {}
    for (I, J, K), block in phi_blocks.items():
        modified = block.copy()
        if distance_cutoff_A is not None:
            if distances_atom is None:
                raise ValueError(
                    "distances_atom required for distance_cutoff_A"
                )
            # For every (a, b, c) entry inside the block, look up the atomic
            # triplet distance via DOF→atom mapping.
            i_atoms = (offsets[I] + np.arange(block.shape[0])) // 3
            j_atoms = (offsets[J] + np.arange(block.shape[1])) // 3
            k_atoms = (offsets[K] + np.arange(block.shape[2])) // 3
            d_ij = distances_atom[i_atoms[:, None], j_atoms[None, :]]
            d_ik = distances_atom[i_atoms[:, None], k_atoms[None, :]]
            d_jk = distances_atom[j_atoms[:, None], k_atoms[None, :]]
            diam = np.maximum(
                np.maximum(d_ij[:, :, None], d_ik[:, None, :]),
                d_jk[None, :, :],
            )
            modified = np.where(diam > distance_cutoff_A, 0.0, modified)
        if mag_floor is not None:
            modified = np.where(np.abs(modified) < mag_floor, 0.0, modified)
        if np.any(modified):
            out[(I, J, K)] = modified
    return out


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
    ``keep_diag_blocks_only`` (default), otherwise zeroed."""
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
    """Pure FFT bubble for the symmetric-ω finite_analysis convention.

    Shape contract:
        phi_left   : (b_I, b_K1, b_K2)
        phi_right  : (b_J, b_K2_prime, b_K1_prime)
        G_inner_a  : (n_freq, b_K1, b_K1_prime)         — G(K1, K1')
        G_inner_b  : (n_freq, b_K2, b_K2_prime)         — G(K2, K2')

    G is on the symmetric ``[-fmax, ..., -dω, 0, dω, ..., fmax]`` axis built
    by :func:`phonon_inputs.anharmonic._build_frequency_grid`. The ω=0 sample
    is zeroed before convolution and the IFFT output is sliced ``[mid:mid+ne]``
    so the result lives on the same symmetric grid.

    The legacy one-sided convention used by quatrex
    :func:`SigmaPhononPhonon._bubble_block` lives in
    :func:`finite_analysis._compat.bubble_block_legacy_quatrex` for
    bit-for-bit cross-checks.
    """
    ne = G_inner_a.shape[0]
    bI, bK1, bK2 = phi_left.shape
    bJ, bK2_prime, bK1_prime = phi_right.shape
    assert G_inner_a.shape == (ne, bK1, bK1_prime)
    assert G_inner_b.shape == (ne, bK2, bK2_prime)
    # _build_frequency_grid always returns 2*nfreq_pos+1 (odd) samples; the
    # symmetric-grid bookkeeping below depends on a well-defined ω=0 sample.
    assert ne % 2 == 1, (
        f"Symmetric-grid bubble requires odd ne; got ne={ne}. "
        "Use _build_frequency_grid (which always returns odd) to avoid this."
    )

    mid = ne // 2
    Ga_clean = G_inner_a.copy()
    Gb_clean = G_inner_b.copy()
    Ga_clean[mid] = 0.0
    Gb_clean[mid] = 0.0

    Ga_pad = np.zeros((n_fft, bK1, bK1_prime), dtype=complex)
    Gb_pad = np.zeros((n_fft, bK2, bK2_prime), dtype=complex)
    Ga_pad[:ne] = Ga_clean
    Gb_pad[:ne] = Gb_clean

    Ga_fft = np.fft.fft(Ga_pad, axis=0)
    Gb_fft = np.fft.fft(Gb_pad, axis=0)

    A = np.einsum("ace,wed->wacd", phi_left, Gb_fft)
    B = np.einsum("wacd,wcb->wabd", A, Ga_fft)
    S_hat = np.einsum("wabd,Jdb->waJ", B, phi_right)

    return prefactor * np.fft.ifft(S_hat, axis=0)[mid:mid + ne]


def compute_sse_with_cutoffs(
    phi_blocks: dict[tuple[int, int, int], np.ndarray],
    g_lesser_blocks: dict[tuple[int, int], np.ndarray],
    g_greater_blocks: dict[tuple[int, int], np.ndarray],
    block_sizes: np.ndarray,
    dw_thz: float,
    *,
    diag_G_in_se: bool = False,
    diag_G_everywhere: bool = False,
    fc3_nn_only: bool = True,
    fc3_distance_cutoff: float | None = None,
    fc3_magnitude_threshold: float | None = None,
    distances_atom: np.ndarray | None = None,
) -> SSEResult:
    """Compute Σ^{<,>} blocks under the requested cutoffs.

    The order of operations:
      1. Apply FC3 cutoffs (NN, distance, magnitude) to ``phi_blocks``.
      2. Apply diagonal-G projection if requested.
      3. Loop over (I, J) ∈ NN block-tridiagonal pairs and accumulate the
         bubble contribution from every (K1, K2) triplet that has a
         (J, K2, K1) partner.
    """
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
    if not diag_G_in_se:
        # The "off-diagonal-G" path (default) requires the G dict to cover at
        # least every NN-tridiagonal (K, K') pair, otherwise we silently miss
        # bubble contributions. ``synthetic_gf.gf_to_block_dict(..., nn_only=
        # False)`` provides the full (n_blocks × n_blocks) coverage; ``True``
        # gives only NN. Here we accept either NN-tridiagonal *or* full.
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
    pair_index: dict[tuple[int, int], list] = {}
    for (I, K1, K2), phi_left in phi.items():
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
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
