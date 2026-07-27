"""Embedded-wire force-constant tools: surface pinning + sum-rule policy.

Phase-1/3 machinery of the d5a twist-mode plan:

* :func:`add_surface_pinning` -- emulate an embedding by harmonic
  springs on the surface (H-shell) atoms of the block dynamical matrix.
* :func:`reimpose_translational_asr` -- subtract the minimal Hermitian
  on-site correction so the sqrt(m)-weighted translations are exact
  zero modes again. Applied after pinning (or after extracting the
  wire sub-block of an embedded calculation), this removes the
  uniform-translation part of the environment coupling while KEEPING
  its orientation-dependent (twist / flexural) stiffening -- the
  controlled "wire in a matrix" model with intact momentum
  conservation along the transport axis.
* :func:`gamma_spectrum`, :func:`twist_gap` -- validation helpers.

Conventions: blocks are the transport ``dynamical_matrix.mat`` dict
{(nx,ny,nz): H} in THz^2, mass-weighted (D = M^-1/2 Phi M^-1/2), as
emitted by ``get_btd_blocks_folded``. The translational null vectors of
D are sqrt(m) (x) e_alpha (NOT uniform vectors -- Si:H = 28:1).
"""
from __future__ import annotations

import numpy as np

# amu masses for the species we handle here.
_MASS = {"Si": 28.0855, "H": 1.008, "O": 15.999}


def _sqrtm_translations(masses: np.ndarray) -> np.ndarray:
    """Orthonormal sqrt(m)-weighted translation vectors, shape (3N, 3)."""
    n = masses.size
    T = np.zeros((3 * n, 3))
    sm = np.sqrt(masses)
    for a in range(3):
        T[a::3, a] = sm
    T /= np.linalg.norm(T, axis=0, keepdims=True)
    return T


def read_structure_xyz(path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(symbols, positions (N,3) A, masses (N,) amu) from extended xyz."""
    lines = open(path).read().splitlines()
    n = int(lines[0])
    syms, pos = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        syms.append(parts[0])
        pos.append([float(x) for x in parts[1:4]])
    masses = np.array([_MASS[s] for s in syms])
    return syms, np.array(pos), masses


def add_surface_pinning(
    blocks: dict[tuple[int, int, int], np.ndarray],
    masses: np.ndarray,
    surface_idx: np.ndarray,
    k_pin: float,
    directions: str = "xy",
) -> dict[tuple[int, int, int], np.ndarray]:
    """Add harmonic pinning springs to surface atoms (on-site block only).

    A physical spring of constant ``k_pin`` (units THz^2 * amu, i.e.
    Phi-level) anchored to a rigid frame adds Phi_ii += k_pin * P to the
    force constants of atom i, hence D_ii += (k_pin / m_i) * P to the
    mass-weighted dynamical matrix -- light (H) atoms feel it 28x
    stronger than Si, as a shell contacting the passivation layer would.

    Parameters
    ----------
    blocks : the .mat block dict (THz^2); NOT modified in place.
    masses : (N,) amu.
    surface_idx : indices of the pinned atoms.
    k_pin : spring constant, THz^2 * amu.
    directions : subset of "xyz" -- which Cartesian components to pin
        (transverse "xy" for a wire along z).

    Returns
    -------
    New block dict with the modified on-site (0,0,0) block.
    """
    out = {k: v.copy() for k, v in blocks.items()}
    onsite_key = next(k for k in out if all(c == 0 for c in k))
    H0 = out[onsite_key]
    dirs = ["xyz".index(c) for c in directions]
    for i in np.asarray(surface_idx, dtype=int):
        for a in dirs:
            j = 3 * i + a
            H0[j, j] += k_pin / masses[i]
    return out


def reimpose_translational_asr(
    blocks: dict[tuple[int, int, int], np.ndarray],
    masses: np.ndarray,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Subtract the minimal Hermitian on-site correction restoring exact
    sqrt(m)-weighted translational zero modes.

    Let S = sum_n H(n) (the Gamma matrix) and T the orthonormal
    translation frame. The defect D_def = S @ T is removed by the
    minimal Hermitian correction

        C = D T^H + T D^H - T (T^H D) T^H            (with D = D_def)

    which satisfies C T = D_def exactly when T^H D_def is Hermitian --
    true for Hermitian S. The corrected on-site block is H0 - C: the
    uniform-translation coupling of any added environment (pinning,
    extracted embedding) is cancelled; rotational / higher-moment parts
    survive.
    """
    out = {k: v.copy() for k, v in blocks.items()}
    onsite_key = next(k for k in out if all(c == 0 for c in k))
    S = sum(out[k] for k in out)
    S = 0.5 * (S + S.conj().T)
    T = _sqrtm_translations(np.asarray(masses))
    D_def = S @ T
    M = T.conj().T @ D_def
    M = 0.5 * (M + M.conj().T)  # symmetrise (Hermitian S guarantees it)
    C = D_def @ T.conj().T + T @ D_def.conj().T - T @ M @ T.conj().T
    out[onsite_key] = out[onsite_key] - C
    return out


def gamma_spectrum(blocks: dict) -> np.ndarray:
    """Signed frequencies (THz) of the Gamma dynamical matrix."""
    S = sum(blocks[k] for k in blocks)
    S = 0.5 * (S + S.conj().T)
    w2 = np.linalg.eigvalsh(S)
    return np.sign(w2) * np.sqrt(np.abs(w2))


def twist_gap(
    blocks: dict,
    positions: np.ndarray,
    masses: np.ndarray,
    axis: str = "z",
) -> float:
    """Frequency (THz) of the Gamma mode with the largest overlap with
    the rigid rotation about the wire axis (the twist)."""
    S = sum(blocks[k] for k in blocks)
    S = 0.5 * (S + S.conj().T)
    w2, V = np.linalg.eigh(S)
    ai = "xyz".index(axis)
    perp = [i for i in range(3) if i != ai]
    c = positions[:, perp].mean(axis=0)
    n = masses.size
    u = np.zeros(3 * n)
    sm = np.sqrt(masses)
    # u_i ~ sqrt(m_i) * (axis_hat x r_i): tangential displacement.
    u[perp[0]::3] = -sm * (positions[:, perp[1]] - c[1])
    u[perp[1]::3] = sm * (positions[:, perp[0]] - c[0])
    u /= np.linalg.norm(u)
    overlaps = np.abs(V.conj().T @ u) ** 2
    m = int(np.argmax(overlaps))
    w2m = w2[m]
    return float(np.sign(w2m) * np.sqrt(abs(w2m)))


def pin_for_twist_gap(
    blocks: dict,
    positions: np.ndarray,
    masses: np.ndarray,
    surface_idx: np.ndarray,
    target_gap_thz: float,
    axis: str = "z",
    tol: float = 0.02,
    max_iter: int = 60,
) -> tuple[dict, float, float]:
    """Bisect k_pin so the ASR-corrected twist gap hits ``target_gap_thz``.

    Returns (pinned+ASR blocks, k_pin, achieved gap).
    """
    def gap_of(k_pin: float) -> tuple[dict, float]:
        b = add_surface_pinning(blocks, masses, surface_idx, k_pin)
        b = reimpose_translational_asr(b, masses)
        return b, twist_gap(b, positions, masses, axis)

    lo, hi = 0.0, 1.0
    while gap_of(hi)[1] < target_gap_thz and hi < 1e6:
        hi *= 4.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        b, g = gap_of(mid)
        if abs(g - target_gap_thz) <= tol * target_gap_thz:
            return b, mid, g
        if g < target_gap_thz:
            lo = mid
        else:
            hi = mid
    return b, mid, g
