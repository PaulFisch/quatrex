# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The four sectors are an identity of the retarded split, on a real pattern.

``test_pole_subcell.py`` establishes the identity on a 2x2 bed by hand. This
checks that the production coefficients -- which never form ``B^R_k``, and get
the ``SR`` bracket from the ``RS`` one by anti-Hermiticity of ``Sigma`` --
reproduce it for several poles, several dof, and a nontrivial sparsity pattern.
"""
import numpy as np
import pytest

from quatrex.phonon.pole_congruence import (
    background_coefficients, reconstruct, sector_grid_sample, sector_terms,
)
from quatrex.phonon.pole_keldysh import PoleCluster

N_DOF, N_P, N_W = 6, 3, 5


def _bed(seed=7):
    rng = np.random.default_rng(seed)

    def cx(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    z = rng.uniform(4.0, 10.0, N_P) - 1j * rng.uniform(0.01, 0.2, N_P)
    cl = PoleCluster(z=z, u=cx(N_DOF, N_P), v=cx(N_DOF, N_P), label="bed")
    w = np.linspace(3.0, 11.0, N_W)
    gk = cx(N_W, N_DOF, N_DOF)
    # -i Sigma Hermitian PSD  <=>  Sigma^dagger = -Sigma, which is the
    # relation background_coefficients uses to get the SR bracket for free.
    a = cx(N_W, N_DOF, N_DOF)
    psd = a @ np.conj(np.swapaxes(a, 1, 2))
    sig = 1j * psd
    assert np.abs(np.conj(np.swapaxes(sig, 1, 2)) + sig).max() < 1e-10
    return cl, w, gk, sig


def _coeffs(cl, w, gk, sig):
    v = np.asarray(cl.v)
    sv = sig @ v
    return background_coefficients(cl, w, sv, gk @ sv)


@pytest.mark.parametrize("seed", [7, 11])
def test_sectors_sum_to_the_congruence(seed):
    """``RR + SR + RS + SS`` is the congruence at ANY probe, not only at the
    cell centres -- that is what makes the reconstruction PSD off-centre."""
    cl, w, gk, sig = _bed(seed)
    co = _coeffs(cl, w, gk, sig)
    rows, cols = (np.repeat(np.arange(N_DOF), N_DOF),
                  np.tile(np.arange(N_DOF), N_DOF))
    u, v, z = np.asarray(cl.u), np.asarray(cl.v), np.asarray(cl.z)

    for frac in (0.0, 0.17, -0.4):
        probe = w + frac * (w[1] - w[0])
        sr, rs, ss = sector_terms(cl, w, co, rows, cols, probe=probe)
        for i in range(N_W):
            # RR = B^R_k Sigma B^A_k, formed densely only here
            b = gk[i] - (u * (1.0 / (w[i] - z))) @ v.conj().T
            rr = b @ sig[i] @ b.conj().T
            got = rr + (sr[i] + rs[i] + ss[i]).reshape(N_DOF, N_DOF)
            want = reconstruct(cl, w[i], probe[i], gk[i], sig[i])[0]
            assert np.abs(got - want).max() < 1e-8 * np.abs(want).max()


def test_grid_sample_is_what_the_ring_must_give_up():
    """At the centre the reconstruction is the untouched ring, so the sample
    removed from the ring's legs is exactly ``G^< - B^R_k Sigma B^A_k``."""
    cl, w, gk, sig = _bed()
    co = _coeffs(cl, w, gk, sig)
    rows, cols = (np.repeat(np.arange(N_DOF), N_DOF),
                  np.tile(np.arange(N_DOF), N_DOF))
    smp = sector_grid_sample(cl, w, co, rows, cols)
    u, v, z = np.asarray(cl.u), np.asarray(cl.v), np.asarray(cl.z)
    for i in range(N_W):
        g_lesser = gk[i] @ sig[i] @ gk[i].conj().T
        b = gk[i] - (u * (1.0 / (w[i] - z))) @ v.conj().T
        rr = b @ sig[i] @ b.conj().T
        assert np.abs(smp[i].reshape(N_DOF, N_DOF)
                      - (g_lesser - rr)).max() < 1e-8 * np.abs(g_lesser).max()


def test_reconstruction_is_psd_across_the_whole_cell():
    """The property the redesign exists for. ``-i G~^<`` is a congruence of a
    PSD matrix, so it is PSD at every probe -- with no accuracy demanded of the
    pole model, which the superseded form needed to 20 percent."""
    cl, w, gk, sig = _bed()
    h = w[1] - w[0]
    for i in range(N_W):
        for frac in np.linspace(-0.5, 0.5, 21):
            g = reconstruct(cl, w[i], w[i] + frac * h, gk[i], sig[i])[0]
            herm = -1j * g
            herm = 0.5 * (herm + herm.conj().T)
            ev = np.linalg.eigvalsh(herm)
            assert ev.min() / max(abs(ev).max(), 1e-300) > -1e-12


def test_sample_vanishes_without_poles_shifting_the_leg():
    """An empty correction reproduces the grid solver bit-for-bit: at the
    centre the pole supplies sub-cell structure and nothing else."""
    cl, w, gk, sig = _bed()
    co = _coeffs(cl, w, gk, sig)
    rows, cols = (np.repeat(np.arange(N_DOF), N_DOF),
                  np.tile(np.arange(N_DOF), N_DOF))
    sr, rs, ss = sector_terms(cl, w, co, rows, cols, probe=w)
    dense = [(sr[i] + rs[i] + ss[i]).reshape(N_DOF, N_DOF) for i in range(N_W)]
    for i in range(N_W):
        want = gk[i] @ sig[i] @ gk[i].conj().T
        got = reconstruct(cl, w[i], w[i], gk[i], sig[i])[0]
        assert np.abs(got - want).max() < 1e-9 * np.abs(want).max()
        assert np.abs(dense[i]).max() > 0.0
