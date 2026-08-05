# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Centre-of-mass (CM) channel of the open phonon device.

Derivation and measurements: ``phonon/docs/ir_residue_derivation.md``.
At eta = 0 the lead-injected translation modes give the device Green's
functions an exact rank-r infrared channel

    -i G^{<,>}(w)|_T  =  C2 / w^2  -+  V_T^{-1} / w  +  O(1),
    C2 = V_T^{-1} ( sum_alpha 2 c_alpha V_{alpha,T} ) V_T^{-1},

(T = the near-null modes of the static lead-screened stiffness
``K = D00 + D01 + D10``; ``Gamma_alpha(w) = 2 w V_alpha`` the linear
lead-broadening opening; ``c_alpha = k_B T_alpha / h``). The full
crystal's cubic acoustic sum rule annihilates this channel exactly
(uniform motion does not scatter); the device-truncated vertex cannot,
which makes the bare bubble integrand non-integrable and is the
measured driver of the eta = 0 grid-refinement divergence.

The surgical treatment subtracts the EXACT CM-subsystem pair

    S^R(w)     = [ w^2 + i w V_T ]^{-1}                 (on T)
    S^{<}(w)   = S^R [ i sum_a 2 w n_a(w) V_{a,T} ] S^A
    S^{>}(w)   = S^R [ i sum_a 2 w (n_a(w)+1) V_{a,T} ] S^A

from the SSE bubble legs at the q = Gamma pair (zero add-back; Dyson
and observables keep the full G). ``S`` is fold- and (at equal
temperatures) KMS-exact by construction; its lesser tail decays as
1/w^4 beyond the damping scale ``V_T``, so nothing away from the
channel is touched. Everything here is computed rank-locally and
deterministically from the run's own inputs -- no fit, no free
parameter, no broadening.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import loadmat

from quatrex.phonon.ir_subtraction import bose

if TYPE_CHECKING:
    from quatrex.core.config import QuatrexConfig

# probe frequencies (THz) for the Gamma(w)/2w -> V_alpha extraction;
# fitted as V(w) = V0 + a*w^2 (validated to 6e-4 on the MoS2 film and
# machine precision on chain fixtures, phonon/studies/_ir_residue_check.py)
_PROBE_W = (2e-3, 4e-3, 8e-3, 1.6e-2)
# |eigenvalue| < _NULL_TOL * |K| counts as a translation/null mode
_NULL_TOL = 1e-4


def load_gamma_transport_blocks(
    config: "QuatrexConfig",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gamma-point transport blocks (d00, d01, d10) of the lead cell.

    Reads ``<input_dir>/dynamical_matrix.mat`` (offset-keyed real-space
    blocks) and sums the transverse offsets with unit (Gamma) phases,
    bucketing by the transport-axis offset. Raises if the transport
    coupling extends beyond nearest cells (the OBC superblock
    assumption).
    """
    path = config.input_dir / "dynamical_matrix.mat"
    raw = loadmat(str(path))
    tdir = str(config.device.transport_direction).strip().lower()
    ti = "xyz".index(tdir)
    blocks: dict[int, np.ndarray] = {}
    for key, val in raw.items():
        if key.startswith("__"):
            continue
        offs = [int(x) for x in re.findall(r"-?\d+", key)]
        n_t = offs[ti]
        if abs(n_t) > 1:
            raise ValueError(
                f"cm_channel: transport-axis offset {n_t} in {path.name}; "
                "the CM channel builder assumes nearest-cell transport "
                "coupling (OBC superblock convention)."
            )
        blocks.setdefault(n_t, np.zeros(val.shape, complex))
        blocks[n_t] += val
    d00, d01, d10 = blocks[0], blocks[1], blocks[-1]
    if not np.allclose(d00, d00.conj().T, atol=1e-9 * np.linalg.norm(d00)):
        raise ValueError("cm_channel: D00 at Gamma is not Hermitian.")
    if not np.allclose(d10, d01.conj().T, atol=1e-9 * np.linalg.norm(d01)):
        raise ValueError("cm_channel: D10 != D01^dagger at Gamma.")
    return d00, d01, d10


def translation_null_modes(
    d00: np.ndarray, d01: np.ndarray, d10: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Near-null modes of the lead-summed stiffness K = D00 + D01 + D10.

    These are the mass-weighted uniform translations (K t = 0 exactly up
    to the fc2's own acoustic-sum-rule quality; measured 3e-6 on the
    MoS2 film, 4e-16 on chain fixtures). Returns (t, eigs) with ``t``
    of shape (r, b), orthonormal rows.
    """
    k_lead = (d00 + d01 + d10).real
    k_lead = 0.5 * (k_lead + k_lead.T)
    evals, evecs = np.linalg.eigh(k_lead)
    scale = float(np.linalg.norm(k_lead))
    null = np.abs(evals) < _NULL_TOL * scale
    if not null.any():
        raise ValueError(
            "cm_channel: no near-null translation modes found in the "
            f"lead stiffness (smallest |eig| {np.abs(evals).min():.3e} "
            f"vs tol {_NULL_TOL * scale:.3e}); CM subtraction is not "
            "applicable to this system."
        )
    return evecs[:, null].T.copy(), evals[null]


def _spectral_obc():
    from qttools.boundary_conditions.obc import Spectral
    from qttools.nevp import Full

    # The spectral (NEVP mode-matching) solver selects the retarded
    # branch by group velocity/decay and is exact at eta = 0 -- the
    # validated branch for the probe solves regardless of the run's
    # own OBC algorithm choice.
    return Spectral(nevp=Full(), block_sections=1)


def lead_velocity_matrices(
    d00: np.ndarray, d01: np.ndarray, d10: np.ndarray, t_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(V_LT, V_RT): the linear lead-broadening opening on T.

    Gamma_alpha(w) = i(Sigma^R_alpha - Sigma^A_alpha) = 2 w V_alpha
    + O(w^3); V extracted on the probe ladder with a w^2 fit.
    """
    obc = _spectral_obc()
    b = d00.shape[-1]
    eye = np.eye(b)
    flip = lambda a: np.flip(a, axis=(-2, -1))  # noqa: E731
    ws = np.asarray(_PROBE_W)
    z2 = (ws * ws).astype(complex)
    m_00 = z2[:, None, None] * eye - d00[None]
    m_01 = np.broadcast_to(-d01, m_00.shape).copy()
    m_10 = np.broadcast_to(-d10, m_00.shape).copy()
    g_00 = np.asarray(obc(m_00, m_01, m_10, "left"))
    sig_l = m_10 @ g_00 @ m_01
    g_nn = np.asarray(obc(flip(m_00), flip(m_10), flip(m_01), "right"))
    sig_r = m_01 @ flip(g_nn) @ m_10

    out = []
    for sig in (sig_l, sig_r):
        gam = (1j * (sig - np.conj(np.swapaxes(sig, -2, -1)))).real
        v_w = np.einsum("ai,wij,bj->wab", t_cell, gam, t_cell) / (
            2.0 * ws[:, None, None])
        coef = np.polynomial.polynomial.polyfit(
            ws ** 2, v_w.reshape(len(ws), -1), 1)
        v0 = coef[0].reshape(t_cell.shape[0], t_cell.shape[0])
        out.append(0.5 * (v0 + v0.T))
    return out[0], out[1]


def cm_sigma_pair(
    w: float,
    v_l: np.ndarray,
    v_r: np.ndarray,
    t_left: float,
    t_right: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(M^<, M^>) of the CM channel at frequency w != 0, r x r on T.

    Production (occupation-positive) sign convention:
    S^{<,>} = S^R [ +i sum_a 2 w n_a^{<,>} V_a ] S^A.
    """
    r = v_l.shape[0]
    v_t = v_l + v_r
    sr = np.linalg.inv((w * w) * np.eye(r) + 1j * w * v_t)
    n_l, n_r = bose(w, t_left), bose(w, t_right)
    drive_l = 2.0 * w * (n_l * v_l + n_r * v_r)
    drive_g = 2.0 * w * ((n_l + 1.0) * v_l + (n_r + 1.0) * v_r)
    sa = sr.conj().T
    return 1j * sr @ drive_l @ sa, 1j * sr @ drive_g @ sa


def compute_cm_channel(
    config: "QuatrexConfig",
    n_blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Full CM-channel data for a run: (t_dev, V_LT, V_RT, T_L, T_R).

    ``t_dev`` (r, n_blocks * b): the device-tiled orthonormal
    translation modes. Rank-local and deterministic (identical on every
    rank): reads the run's own ``dynamical_matrix.mat`` and solves the
    probe OBC with the spectral solver.
    """
    d00, d01, d10 = load_gamma_transport_blocks(config)
    t_cell, eigs = translation_null_modes(d00, d01, d10)
    v_l, v_r = lead_velocity_matrices(d00, d01, d10, t_cell)
    ev_l = np.linalg.eigvalsh(v_l)
    if ev_l.min() < -1e-8 * max(ev_l.max(), 1e-300):
        raise ValueError(
            f"cm_channel: V_LT not PSD (eigs {ev_l}); the linear-opening "
            "extraction failed for this lead model.")
    t_dev = np.tile(t_cell, (1, n_blocks))
    t_dev /= np.linalg.norm(t_dev, axis=1, keepdims=True)
    # V is extracted in the CELL translation basis; the S field lives on
    # the DEVICE-tiled normalised modes t_dev = tile(t_cell)/sqrt(n),
    # whose contact-block components carry 1/sqrt(n) each -- the lead
    # damping in that basis is V_cell / n_blocks. (Validated against the
    # measured C2 residue to 6e-4, _ir_residue_check.py.)
    v_l = v_l / n_blocks
    v_r = v_r / n_blocks
    return (t_dev, v_l, v_r,
            float(config.phonon.left_temperature),
            float(config.phonon.right_temperature))
