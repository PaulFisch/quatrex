"""Surface pinning + translational-ASR policy: the twist gaps, the
translations stay exact.

Test wire: 4 atoms per cell on a square ring (radius R) around the z
axis, coupled by CENTRAL pairwise springs (in-cell ring bonds +
nearest-cell axial/diagonal bonds). Central springs are rotationally
invariant, so the free wire has FOUR exact zero modes at Gamma: three
translations + the rigid twist. Pinning the ring atoms transversely and
re-imposing the translational ASR must gap ONLY the twist.
"""
import numpy as np
import pytest

from phonon_inputs.embedded_extract import (
    add_surface_pinning,
    gamma_spectrum,
    pin_for_twist_gap,
    reimpose_translational_asr,
    twist_gap,
)

R = 2.0     # ring radius, A
C = 3.0     # period along z, A
K = 5.0     # spring constant (THz^2 * amu per A^2 -- units arbitrary)
M = np.array([28.0, 1.0, 28.0, 1.0])  # mixed masses (Si/H-like)


def _ring_positions():
    ang = np.arange(4) * np.pi / 2
    return np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros(4)], axis=1)


def _central_fc(d):
    """FC 3x3 block of a central spring along bond vector d: -K r r^T."""
    r = d / np.linalg.norm(d)
    return -K * np.outer(r, r)


def _build_blocks():
    """H(0), H(+-1) for the ring wire with central springs (mass-weighted)."""
    pos = _ring_positions()
    n = 4
    H = {(0, 0, 0): np.zeros((3 * n, 3 * n)),
         (0, 0, 1): np.zeros((3 * n, 3 * n)),
         (0, 0, -1): np.zeros((3 * n, 3 * n))}
    onsite = [np.zeros((3, 3)) for _ in range(n)]

    def bond(i, j, cell, dvec):
        blk = _central_fc(dvec)
        H[(0, 0, cell)][3*i:3*i+3, 3*j:3*j+3] += blk / np.sqrt(M[i] * M[j])
        onsite[i] -= blk / M[i]  # ASR: on-site balances (mass-weighted)

    for i in range(n):
        jn = (i + 1) % n                        # in-cell ring bond
        bond(i, jn, 0, pos[jn] - pos[i])
        bond(jn, i, 0, pos[i] - pos[jn])
    for i in (0, 1):                            # in-cell cross diagonals:
        jd = i + 2                              # kill the rhombus shear
        bond(i, jd, 0, pos[jd] - pos[i])        # (central springs keep
        bond(jd, i, 0, pos[i] - pos[jd])        # rotational invariance)
    for i in range(n):
        jn = (i + 1) % n
        # axial bond to the same atom in the next cell
        bond(i, i, 1, np.array([0.0, 0.0, C]))
        bond(i, i, -1, np.array([0.0, 0.0, -C]))
        # diagonal bond to the neighbouring atom in the next cell
        d = pos[jn] - pos[i] + np.array([0.0, 0.0, C])
        bond(i, jn, 1, d)
        bond(jn, i, -1, -d)
    for i in range(n):
        H[(0, 0, 0)][3*i:3*i+3, 3*i:3*i+3] += onsite[i]
    return H, pos


@pytest.fixture(scope="module")
def wire():
    return _build_blocks()


def test_free_wire_has_four_zero_modes(wire):
    H, pos = wire
    w = gamma_spectrum(H)
    assert np.abs(w[:4]).max() < 1e-6, f"expected 4 zero modes, got {w[:6]}"
    assert w[4] > 0.1
    assert abs(twist_gap(H, pos, M)) < 1e-6


def test_pinning_plus_asr_gaps_only_the_twist(wire):
    H, pos = wire
    pinned = add_surface_pinning(H, M, np.arange(4), k_pin=2.0)
    fixed = reimpose_translational_asr(pinned, M)
    w = gamma_spectrum(fixed)
    # translations exactly restored ...
    assert np.abs(w[:3]).max() < 1e-6, f"translations not zero: {w[:5]}"
    # ... the twist is gapped ...
    tg = twist_gap(fixed, pos, M)
    assert tg > 0.05, f"twist not gapped: {tg}"
    # ... no imaginary modes anywhere.
    assert w.min() > -1e-6
    # Hermiticity preserved.
    for k, v in fixed.items():
        pass
    S = sum(fixed[k] for k in fixed)
    assert np.abs(S - S.conj().T).max() < 1e-10


def test_pinning_without_asr_gaps_translations_too(wire):
    """Documents WHY the ASR step matters: bare pinning gaps everything."""
    H, pos = wire
    pinned = add_surface_pinning(H, M, np.arange(4), k_pin=2.0)
    w = gamma_spectrum(pinned)
    # xy-pinning leaves the z-translation as the single exact zero mode;
    # the two transverse translations (and the twist) are lifted.
    n_zero = int(np.sum(np.abs(w) < 1e-4))
    assert n_zero == 1, f"expected exactly one zero mode, got {n_zero}: {w[:5]}"
    assert w[1] > 1e-3, f"transverse translations not lifted: {w[:5]}" 


def test_pin_for_twist_gap_hits_target(wire):
    H, pos = wire
    target = 0.5
    fixed, k_pin, gap = pin_for_twist_gap(
        H, pos, M, np.arange(4), target_gap_thz=target)
    assert abs(gap - target) <= 0.02 * target
    assert k_pin > 0
    w = gamma_spectrum(fixed)
    assert np.abs(w[:3]).max() < 1e-6
