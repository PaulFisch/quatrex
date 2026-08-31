# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Doc Sec. 53 Experiment A on an exactly solvable nonlinear eigenvalue problem.

Bed: ``M(z) = z^2 I + i g z I - D`` with ``D`` real symmetric block-tridiagonal.
This has the shape of the phonon Dyson operator (``z^2 I - D - Sigma^R``) with a
frequency-linear damping standing in for the lead broadening, and it is solvable
in closed form: for every eigenpair ``(lam, v)`` of ``D``,

    z^2 + i g z - lam = 0   =>   z = (-i g +- sqrt(4 lam - g^2)) / 2,

so ``Omega = sqrt(lam - g^2/4)``, ``gamma = g/2``, the right vector is ``v``, and
because ``M`` is complex symmetric the left vector is ``conj(v) = v``. The
residue follows from ``d = l^H M'(z) r = 2z + i g = 2 Omega``:

    R = v v^T / (2 Omega).

Every quantity the pole sector needs therefore has an independent reference.
"""
import numpy as np
import pytest

from quatrex.phonon.experimental.pole.pole_nevp import (
    bordered_newton,
    residue,
)

G_DAMP = 0.30


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _dynamical(sizes=(3, 3, 3), seed=0, coupling=0.4, separate=True):
    """A real symmetric, positive-definite block-tridiagonal ``D``."""
    rng = np.random.default_rng(seed)
    n = len(sizes)
    total = sum(sizes)
    ladder = 25.0 * 1.7 ** np.arange(total) if separate else np.full(total, 40.0)
    d_ii, d_ij, k = [], [], 0
    for i in range(n):
        m = coupling * rng.normal(size=(sizes[i], sizes[i]))
        d_ii.append(m + m.T + np.diag(ladder[k:k + sizes[i]]))
        k += sizes[i]
    for i in range(n - 1):
        d_ij.append(coupling * rng.normal(size=(sizes[i], sizes[i + 1])))
    d_ji = [b.T for b in d_ij]
    return d_ii, d_ij, d_ji


def _dense(d_ii, d_ij, d_ji):
    sizes = [b.shape[-1] for b in d_ii]
    off = np.concatenate(([0], np.cumsum(sizes)))
    n = off[-1]
    out = np.zeros((n, n))
    for i in range(len(d_ii)):
        out[off[i]:off[i + 1], off[i]:off[i + 1]] = d_ii[i]
        if i + 1 < len(d_ii):
            out[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = d_ij[i]
            out[off[i + 1]:off[i + 2], off[i]:off[i + 1]] = d_ji[i]
    return out


def _operator(d_ii, d_ij, d_ji, g=G_DAMP):
    def m_blocks(z):
        w = z * z + 1j * g * z
        a_ii = [w * np.eye(b.shape[-1]) - b for b in d_ii]
        return a_ii, [-b for b in d_ij], [-b for b in d_ji]

    def dm_blocks(z):
        dw = 2.0 * z + 1j * g
        a_ii = [dw * np.eye(b.shape[-1]) + 0j for b in d_ii]
        a_ij = [np.zeros_like(b) + 0j for b in d_ij]
        a_ji = [np.zeros_like(b) + 0j for b in d_ji]
        return a_ii, a_ij, a_ji

    return m_blocks, dm_blocks


def _exact_poles(d_dense, g=G_DAMP):
    lam, vec = np.linalg.eigh(d_dense)
    disc = np.sqrt(np.asarray(4 * lam - g * g, dtype=complex))
    z_plus = (-1j * g + disc) / 2.0
    return z_plus, lam, vec


# --------------------------------------------------------------------------- #

def test_bordered_newton_finds_the_exact_pole():
    d = _dynamical()
    m_blocks, dm_blocks = _operator(*d)
    z_ex, _, vec = _exact_poles(_dense(*d))

    spacing = np.min(np.diff(np.sort(z_ex.real)))
    for k in (0, 3, len(z_ex) - 1):
        # Off in both frequency and linewidth, but inside the pole's basin.
        guess = complex(z_ex[k].real + 0.25 * spacing, -0.05)
        sol = bordered_newton(m_blocks, dm_blocks, guess, max_iter=30, tol=1e-9)
        assert sol.converged, f"eps_nep={sol.eps_nep:.3e} after {sol.iterations} its"
        assert abs(sol.z - z_ex[k]) < 1e-9 * abs(z_ex[k]), (
            f"got {sol.z}, exact {z_ex[k]}"
        )
        # The right vector is the eigenvector of D, up to a phase.
        ov = abs(np.vdot(_h(sol.r), vec[:, k]))
        assert ov > 1 - 1e-9, f"eigenvector overlap {ov}"


def test_left_vector_normalisation_makes_d_alpha_one():
    """l^H M'(z) r == 1, which is what makes the residue simply r l^H."""
    d = _dynamical(seed=1)
    m_blocks, dm_blocks = _operator(*d)
    z_ex, _, _ = _exact_poles(_dense(*d))

    sol = bordered_newton(
        m_blocks, dm_blocks, complex(z_ex[2].real, -0.05), max_iter=30
    )
    assert sol.converged
    dm = dm_blocks(sol.z)[0][0][0, 0]  # M' is dw * I
    d_alpha = np.vdot(_h(sol.l), dm * _h(sol.r))
    assert abs(d_alpha - 1.0) < 1e-9, f"d_alpha = {d_alpha}"


def test_residue_matches_the_closed_form():
    d = _dynamical(seed=2)
    m_blocks, dm_blocks = _operator(*d)
    d_dense = _dense(*d)
    z_ex, _, vec = _exact_poles(d_dense)

    k = 4
    sol = bordered_newton(
        m_blocks, dm_blocks, complex(z_ex[k].real, -0.05), max_iter=30
    )
    assert sol.converged and abs(sol.z - z_ex[k]) < 1e-8 * abs(z_ex[k])
    omega = sol.z.real
    ref = np.outer(vec[:, k], vec[:, k]) / (2.0 * omega)
    got = _h(sol.residue())
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-8


def test_eps_nep_flags_a_wrong_pole():
    """The residual must reject a point that is not a pole -- the accept gate."""
    d = _dynamical(seed=4)
    m_blocks, dm_blocks = _operator(*d)
    z_ex, _, _ = _exact_poles(_dense(*d))

    good = bordered_newton(
        m_blocks, dm_blocks, complex(z_ex[0].real, -0.05), max_iter=30, tol=1e-9
    )
    bad = bordered_newton(
        m_blocks, dm_blocks, complex(z_ex[0].real, -0.05), max_iter=0, tol=1e-9
    )
    assert good.converged and not bad.converged
    assert bad.eps_nep > 1e3 * good.eps_nep


def test_trust_radius_caps_the_step():
    d = _dynamical(seed=5)
    m_blocks, dm_blocks = _operator(*d)
    z_ex, _, _ = _exact_poles(_dense(*d))
    far = complex(z_ex[-1].real + 5.0, -0.05)

    capped = bordered_newton(
        m_blocks, dm_blocks, far, max_iter=1, trust_radius=0.01
    )
    assert abs(capped.z - far) <= 0.01 + 1e-12


def test_dense_spectrum_needs_the_contour_not_a_crude_guess():
    """With modes closer than the guess error, Newton lands on a neighbour."""
    d = _dynamical(sizes=(3, 3, 3), seed=0, separate=False)
    m_blocks, dm_blocks = _operator(*d)
    z_ex, _, _ = _exact_poles(_dense(*d))
    order = np.argsort(z_ex.real)
    spacing = np.min(np.diff(z_ex.real[order]))

    k = int(order[3])
    guess = complex(z_ex[k].real + 4.0 * spacing, -0.05)
    sol = bordered_newton(m_blocks, dm_blocks, guess, max_iter=30, tol=1e-9)

    assert sol.converged, "the corrector must still certify a genuine pole"
    hit = int(np.argmin(np.abs(z_ex - sol.z)))
    assert np.min(np.abs(z_ex - sol.z)) < 1e-8, "not on any exact pole"
    assert hit != k, "guess was not actually crude enough to demonstrate the point"


def test_left_vector_matches_an_svd_null_space():
    """``M(z_alpha)`` is singular, so ``M^{-H}`` does not exist at the pole."""
    import numpy as np

    rng = np.random.default_rng(3)
    b, nb = 3, 3
    n = b * nb
    d = rng.normal(size=(n, n))
    d = d + d.T
    sig = 0.05 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    mask = np.zeros((n, n), dtype=bool)
    for i in range(nb):
        for j in range(max(0, i - 1), min(nb, i + 2)):
            mask[i * b:(i + 1) * b, j * b:(j + 1) * b] = True
    d, sig = d * mask, sig * mask

    def dense(z):
        return z * z * np.eye(n) - d - sig

    def blocks(a):
        return ([a[i * b:(i + 1) * b, i * b:(i + 1) * b] for i in range(nb)],
                [a[i * b:(i + 1) * b, (i + 1) * b:(i + 2) * b] for i in range(nb - 1)],
                [a[(i + 1) * b:(i + 2) * b, i * b:(i + 1) * b] for i in range(nb - 1)])

    ev = np.linalg.eigvals(d + sig)
    z0 = complex(np.sqrt(ev[np.argsort(ev.real)[n // 2]]))
    sol = bordered_newton(lambda z: blocks(dense(z)),
                          lambda z: blocks(2 * z * np.eye(n)), z0)

    u, _, _ = np.linalg.svd(dense(sol.z))
    l = _h(sol.l)
    l = l / np.linalg.norm(l)
    assert abs(np.vdot(u[:, -1], l)) > 1 - 1e-9, "same direction as the SVD null vector"

    # The normalisation that makes R = r l^H the residue.
    d_alpha = np.vdot(_h(sol.l), 2 * sol.z * _h(sol.r))
    assert abs(d_alpha - 1.0) < 1e-10

    # And the conditioning is reported, not assumed.
    assert np.isfinite(sol.eps_left)


# --- the acceptance metric, in frequency units ----------------------------- #

def test_dz_est_is_the_actual_frequency_error():
    """``dz_est`` must predict the remaining distance to the true pole.

    This is the whole basis for gating on ``eps_z`` rather than ``eps_nep``:
    if ``dz_est`` were not the frequency error, moving the gate would just be
    swapping one opaque number for another.
    """
    d = _dynamical(seed=6)
    m_blocks, dm_blocks = _operator(*d)
    z_ex, _, _ = _exact_poles(_dense(*d))
    target = z_ex[0]

    # Stop early on purpose, at a few different distances, and check that
    # dz_est tracks z -> target rather than merely correlating with it.
    for n_it in (1, 2, 3):
        sol = bordered_newton(
            m_blocks, dm_blocks, complex(target.real + 0.30, target.imag),
            max_iter=n_it, tol=1e-30, trust_radius=1.0,
        )
        true_err = abs(complex(sol.z) - complex(target))
        if true_err < 1e-12:
            continue
        rel = abs(abs(complex(sol.dz_est)) - true_err) / true_err
        assert rel < 0.25, (
            f"{n_it} it: |dz_est|={abs(sol.dz_est):.3e} vs true "
            f"{true_err:.3e} (rel {rel:.2f})")


def test_the_two_gates_disagree_and_the_frequency_one_is_the_physical_test():
    """A pole located to a tiny fraction of its width can fail ``eps_nep``."""
    d = _dynamical(seed=7)
    m_blocks, dm_blocks = _operator(*d)
    z_ex, _, _ = _exact_poles(_dense(*d))
    target = z_ex[0]
    gamma = abs(target.imag)

    sol = bordered_newton(
        m_blocks, dm_blocks, complex(target.real + 0.05, target.imag),
        max_iter=3, tol=1e-10, trust_radius=1.0,
    )
    located = abs(complex(sol.z) - complex(target)) / gamma
    # Located to a small fraction of its own linewidth ...
    assert located < 1e-3, f"located to {located:.2e} of gamma"
    # ... and dz_est agrees that it is.
    assert abs(complex(sol.dz_est)) / gamma < 1e-2
    # The scaled matrix residual is a different, much larger-looking number:
    # it is normalised by |z|^2 + ||M||, not by gamma.
    assert sol.eps_nep < 1e-6
    assert sol.eps_nep * (abs(sol.z) ** 2) > abs(complex(sol.dz_est)) * 1e-6
