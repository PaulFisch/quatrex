"""Analytically-tractable toy phonon systems for verifying the dense SCBA solver.

These small systems have 3-phonon physics that can be checked against
closed-form expressions, so they serve as the ground truth for the
solver-verification scripts and the physics regression tests:

  * :func:`single_oscillator` -- one harmonic mode with an on-site cubic
    term.  The 3-phonon self-energy is a pure frequency self-convolution;
    its weak-coupling broadening matches the Fermi golden rule.
  * :func:`monatomic_chain` -- one DOF per cell, nearest-neighbour
    springs.  Its acoustic branch reaches ``omega = 0`` at Gamma, so it
    exercises the zero-mode / DC handling.
  * :func:`diatomic_chain` -- two DOF per cell; acoustic + optical
    branches.

All quantities use the dense solver's internal units: frequencies in
THz, dynamical matrices and self-energies in THz**2, retarded Green's
functions in THz**-2.  A toy carries the on-site (``h00``) and
inter-cell (``h01``) dynamical-matrix blocks plus a fully-symmetric
cubic vertex ``phi``; the helpers below turn those into harmonic
Green's functions and KMS-consistent ``G^{<,>}``.  Callers can then
drive :mod:`phonon.solver.bubble` / :mod:`phonon.solver.se_finite` /
:mod:`phonon.solver.retarded` directly, or assemble a finite device
(``h00``/``h01`` -> :func:`phonon.solver.leads.build_device_hamiltonian`
+ :func:`phonon.solver.leads.compute_obc_batch`) and run
:func:`phonon.solver.dense.scba_loop_dev`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grids import bose_full_axis


# ---------------------------------------------------------------------------
# Toy model container
# ---------------------------------------------------------------------------


@dataclass
class ToyModel:
    """A small harmonic + cubic phonon system in the solver's THz units.

    Attributes
    ----------
    name
        Short identifier.
    n_dof
        Degrees of freedom per transport cell (= ``3 * n_atoms`` for a
        realistic system, but a toy may use fewer).
    masses
        Per-atom masses (advisory; ``h00``/``h01``/``phi`` are already
        mass-weighted).
    h00
        On-site dynamical-matrix block, shape ``(n_dof, n_dof)`` [THz**2].
    h01
        Inter-cell hopping block (cell ``l`` -> ``l+1``), shape
        ``(n_dof, n_dof)`` [THz**2].
    phi
        Fully index-symmetric cubic vertex, shape
        ``(n_dof, n_dof, n_dof)``.
    description
        Human-readable summary.
    """

    name: str
    n_dof: int
    masses: np.ndarray
    h00: np.ndarray
    h01: np.ndarray
    phi: np.ndarray
    description: str = ""

    def gamma_omega2(self) -> np.ndarray:
        """omega**2 eigenvalues of the q=0 (Gamma) dynamical matrix."""
        d_gamma = self.h00 + self.h01 + self.h01.conj().T
        return np.linalg.eigvalsh(d_gamma)

    def isolated_omega2(self) -> np.ndarray:
        """omega**2 eigenvalues of the isolated on-site block ``h00``."""
        return np.linalg.eigvalsh(self.h00)

    def device_dynamical_matrix(self, n_slabs: int) -> np.ndarray:
        """Block-tridiagonal device matrix for ``n_slabs`` identical cells."""
        from .leads import build_device_hamiltonian

        return build_device_hamiltonian(self.h00, self.h01, n_slabs)

    def phi_dev_blocks(
        self, n_slabs: int,
    ) -> dict[tuple[int, int, int], np.ndarray]:
        """On-site cubic vertex placed on every slab: ``{(I, I, I): phi}``.

        The toy cubic term is purely on-site, so the device vertex has
        one triplet per slab and no inter-slab coupling.
        """
        return {(i, i, i): self.phi for i in range(n_slabs)}


# ---------------------------------------------------------------------------
# Named toy systems
# ---------------------------------------------------------------------------


def single_oscillator(
    omega0_thz: float = 6.0, cubic: float = 0.5, mass: float = 1.0,
) -> ToyModel:
    """One isolated harmonic mode with an on-site cubic anharmonicity.

    ``h01 = 0`` (no transport): used for the bubble-level physics checks
    (prefactor, convolution, detailed balance, golden-rule broadening)
    where the 3-phonon self-energy is a pure self-convolution of the
    single-mode Green's function.
    """
    h00 = np.array([[omega0_thz ** 2]], dtype=float)
    h01 = np.zeros((1, 1), dtype=float)
    phi = np.array([[[cubic]]], dtype=float)
    return ToyModel(
        name="single_oscillator",
        n_dof=1,
        masses=np.array([mass], dtype=float),
        h00=h00,
        h01=h01,
        phi=phi,
        description=(
            f"isolated harmonic mode at {omega0_thz} THz, "
            f"on-site cubic g={cubic}"
        ),
    )


def monatomic_chain(
    omega_max_thz: float = 8.0, cubic: float = 0.4, mass: float = 1.0,
) -> ToyModel:
    """1D monatomic chain: one DOF per cell, nearest-neighbour springs.

    The acoustic dispersion is ``omega(q)**2 = 4 k sin**2(q/2)`` with
    ``k = omega_max**2 / 4``; it reaches ``omega = 0`` at Gamma, so this
    toy exercises the zero-mode / DC handling.  The cubic term is
    on-site.
    """
    k = omega_max_thz ** 2 / 4.0
    h00 = np.array([[2.0 * k]], dtype=float)
    h01 = np.array([[-k]], dtype=float)
    phi = np.array([[[cubic]]], dtype=float)
    return ToyModel(
        name="monatomic_chain",
        n_dof=1,
        masses=np.array([mass], dtype=float),
        h00=h00,
        h01=h01,
        phi=phi,
        description=(
            f"1D monatomic chain, acoustic branch up to {omega_max_thz} "
            f"THz (omega->0 at Gamma), on-site cubic g={cubic}"
        ),
    )


def diatomic_chain(
    spring: float = 20.0,
    mass_light: float = 1.0,
    mass_heavy: float = 2.0,
    cubic: float = 0.3,
) -> ToyModel:
    """1D diatomic chain: two DOF per cell, acoustic + optical branches.

    Two unequal masses joined by identical nearest-neighbour springs.
    The optical branch opens a gap, so the toy has well-separated modes
    for the controllability / conservation checks.
    """
    m1, m2 = float(mass_light), float(mass_heavy)
    off = -spring / np.sqrt(m1 * m2)
    h00 = np.array(
        [[2.0 * spring / m1, off], [off, 2.0 * spring / m2]], dtype=float
    )
    # Cell l atom-2 couples to cell l+1 atom-1.
    h01 = np.array([[0.0, 0.0], [off, 0.0]], dtype=float)
    phi = np.zeros((2, 2, 2), dtype=float)
    phi[0, 0, 0] = cubic
    phi[1, 1, 1] = cubic
    return ToyModel(
        name="diatomic_chain",
        n_dof=2,
        masses=np.array([m1, m2], dtype=float),
        h00=h00,
        h01=h01,
        phi=phi,
        description=(
            f"1D diatomic chain, masses ({m1}, {m2}), spring {spring}, "
            f"on-site cubic g={cubic}"
        ),
    )


# ---------------------------------------------------------------------------
# Harmonic Green's functions and KMS-consistent G^{<,>}
# ---------------------------------------------------------------------------


def harmonic_green_retarded(
    dyn_matrix: np.ndarray, z2_arr: np.ndarray,
) -> np.ndarray:
    """Retarded Green's function of a harmonic system.

    ``G^R(omega) = [ (omega + i eta)**2 I - D ]**-1``, evaluated on the
    complex grid ``z2_arr`` from :func:`phonon.solver.grids.build_frequency_grid`.

    Parameters
    ----------
    dyn_matrix
        Harmonic dynamical matrix ``D``, shape ``(n, n)`` [THz**2].
        Use the isolated ``h00`` for a closed system or
        :meth:`ToyModel.device_dynamical_matrix` for a finite device.
    z2_arr
        Complex ``(omega + i eta)**2`` grid, shape ``(n_freq,)``.

    Returns
    -------
    g_ret : complex ndarray, shape ``(n_freq, n, n)``.
    """
    n = dyn_matrix.shape[0]
    eye = np.eye(n)
    a = z2_arr[:, None, None] * eye[None] - dyn_matrix[None]
    return np.linalg.inv(a)


def spectral_function(g_ret: np.ndarray) -> np.ndarray:
    """Phonon spectral function ``A = i (G^R - G^A)``.

    ``A`` is positive semi-definite for ``omega > 0`` and obeys the
    bosonic odd-symmetry ``A(-omega) = -A(omega)^T`` on the symmetric
    grid built with ``z2(-omega) = conj(z2(omega))``.
    """
    g_adv = g_ret.conj().transpose(0, 2, 1)
    return 1j * (g_ret - g_adv)


def equilibrium_lesser_greater(
    g_ret: np.ndarray, freqs_thz: np.ndarray, temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """KMS-consistent ``(G^<, G^>)`` for a system in thermal equilibrium.

    Fluctuation-dissipation theorem for bosons:

        G^<(omega) = -i n_B(omega)        A(omega)
        G^>(omega) = -i (n_B(omega) + 1)  A(omega)

    These satisfy the detailed-balance relation
    ``G^>(omega) = exp(hbar omega / kT) G^<(omega)`` exactly on the grid
    (because ``1 + n_B = exp(beta hbar omega) n_B``), which makes them
    the ground truth for the bubble's no-double-counting check.

    The ``omega = 0`` sample is returned as zero (``n_B(0) = 0``
    placeholder) -- consistent with the solver's DC convention.
    """
    spectral = spectral_function(g_ret)
    n = bose_full_axis(freqs_thz, temperature)[:, None, None]
    g_lesser = -1j * n * spectral
    g_greater = -1j * (n + 1.0) * spectral
    return g_lesser, g_greater


# ---------------------------------------------------------------------------
# Generic symmetric cubic vertex (for robustness tests beyond the toys)
# ---------------------------------------------------------------------------


def symmetric_cubic_vertex(
    n_dof: int, rng: np.random.Generator, scale: float = 1.0,
) -> np.ndarray:
    """A random cubic vertex symmetrized over all 3! index permutations.

    The physical FC3 is fully index-symmetric; this builder produces a
    generic tensor with that property so the bubble can be exercised
    beyond the diagonal on-site toys.
    """
    raw = rng.standard_normal((n_dof, n_dof, n_dof))
    sym = (
        raw
        + raw.transpose(0, 2, 1)
        + raw.transpose(1, 0, 2)
        + raw.transpose(1, 2, 0)
        + raw.transpose(2, 0, 1)
        + raw.transpose(2, 1, 0)
    ) / 6.0
    return scale * sym


# ---------------------------------------------------------------------------
# Gapped beds for the spatial-tail programme
# ---------------------------------------------------------------------------
#
# The ring is a convolution, so Sigma(Omega) needs G at omega AND at
# Omega - omega and the frequency grid has to start at zero. An exact eta = 0
# reference needs the grid to avoid the band. A GAPPED chain satisfies both --
# an on-site pinning puts the band at [w0, sqrt(w0**2 + 4 k_s)] and the grid
# sits below w0, where the Brillouin-zone integrand never vanishes. The
# ungapped ``monatomic_chain`` above cannot: its acoustic branch reaches zero,
# so any grid starting at zero runs through the band.
#
# Promoted from ``tests/quatrex/phonon/test_spatial_modal.py``, where they were
# module-private, because the spatial-tail STUDIES need the same beds as the
# tests and one definition is the point. Behaviour is unchanged: the tests that
# consumed the originals still pass against these.


def gapped_chain(w0: float = 1.0, k_s: float = 4.0, cubic: float = 0.0):
    """1-DOF chain with an on-site pinning ``w0``: band ``[w0, sqrt(w0^2+4k)]``.

    One DOF per cell, so a block IS a cell -- which is what makes the block
    bookkeeping of the self-energy support law readable.
    """
    h00 = np.array([[w0 ** 2 + 2.0 * k_s]], dtype=float)
    h01 = np.array([[-k_s]], dtype=float)
    phi = np.array([[[cubic]]], dtype=float)
    lo, hi = float(w0), float(np.sqrt(w0 ** 2 + 4.0 * k_s))
    return ToyModel(
        name="gapped_chain",
        n_dof=1,
        masses=np.array([1.0], dtype=float),
        h00=h00,
        h01=h01,
        phi=phi,
        description=(
            f"1D chain pinned at w0={w0}, spring {k_s}; band [{lo:.4g}, "
            f"{hi:.4g}] THz, so a grid below {lo:.4g} is regular at eta = 0"
        ),
    )


def gapped_chain_root(omega: float, w0: float = 1.0, k_s: float = 4.0) -> complex:
    """Decaying Bloch factor of :func:`gapped_chain` (real, in ``(0, 1)``).

    ``scimath.sqrt`` and the ``argmin(|lambda|)`` selection are both
    load-bearing: the discriminant turns negative inside the band, and the
    reciprocal partner must not be returned.
    """
    a, b, c = k_s, omega * omega - (w0 ** 2 + 2.0 * k_s), k_s
    disc = np.lib.scimath.sqrt(b * b - 4.0 * a * c)
    lam = np.array([(-b + disc) / (2.0 * a), (-b - disc) / (2.0 * a)])
    return complex(lam[np.argmin(np.abs(lam))])


def gapped_chain_green(omega: float, n_max: int, w0: float = 1.0,
                       k_s: float = 4.0, n_k: int = 4096) -> np.ndarray:
    """``G(n)``, ``n = 0..n_max``, by PERIODIC-trapezoid BZ quadrature.

    Periodic, not ``linspace``: including both endpoints double counts and the
    reference then stops decaying at ~5e-6, which looks like a Green function
    reaching a floor and is arithmetic. Independent of
    :func:`gapped_chain_root`, so a modal reconstruction checked against it is
    a real comparison.
    """
    k = 2.0 * np.pi * np.arange(n_k) / n_k
    denom = omega ** 2 - (w0 ** 2 + 2.0 * k_s) + 2.0 * k_s * np.cos(k)
    return np.array([np.sum(np.exp(1j * k * n) / denom) / n_k
                     for n in range(n_max + 1)])


def bulk_green_blocks(a_ii, a_ij, a_ji, n_max: int, n_k: int = 2048) -> dict:
    """``{n: G(n)}`` of an infinite periodic chain, by the same quadrature.

    ``a_xx`` are SYSTEM-matrix blocks in the OBC convention, the ones
    :func:`quatrex.phonon.spatial_modes.bloch_modes` takes -- so a caller
    cannot accidentally feed the pencil one convention and the reference
    another.
    """
    a_ii = np.asarray(a_ii)
    k = 2.0 * np.pi * np.arange(n_k) / n_k
    ph = np.exp(1j * k)[:, None, None]
    a_k = a_ii[None] + np.asarray(a_ij)[None] * ph + np.asarray(a_ji)[None] * np.conj(ph)
    inv = np.linalg.inv(a_k)
    return {n: (inv * np.exp(1j * k * n)[:, None, None]).sum(0) / n_k
            for n in range(n_max + 1)}


# A 2-DOF cell whose inter-cell coupling is INVERTIBLE. A rank-deficient D_01
# makes the pencil degenerate -- roots collapse to 0 and infinity and the mode
# count is wrong -- so the coupling is chosen full rank and callers assert it.
CHAIN2_D00 = np.array([[2.6, -0.4], [-0.4, 2.2]])
CHAIN2_D01 = np.array([[-0.9, -0.25], [-0.2, -0.8]])


def chain2_system_blocks(omega: float, d00=None, d01=None):
    """``(a_ii, a_ij, a_ji)`` of the 2-DOF chain at ``omega``."""
    d00 = CHAIN2_D00 if d00 is None else d00
    d01 = CHAIN2_D01 if d01 is None else d01
    return (omega * omega * np.eye(d00.shape[0]) - d00, -d01, -d01.conj().T)


def chain2_band_top(d00=None, d01=None, n_k: int = 2001) -> float:
    """Top of the 2-DOF chain's band, for placing a grid outside it."""
    d00 = CHAIN2_D00 if d00 is None else d00
    d01 = CHAIN2_D01 if d01 is None else d01
    top = 0.0
    for x in np.linspace(0.0, np.pi, n_k):
        dk = d00 + d01 * np.exp(1j * x) + d01.conj().T * np.exp(-1j * x)
        top = max(top, np.sqrt(np.linalg.eigvalsh(
            0.5 * (dk + dk.conj().T)).max()))
    return float(top)


def neighbour_cubic_vertex(n_cell: int, seed: int = 5) -> np.ndarray:
    """Cubic vertex coupling each cell to its neighbours, index-symmetric.

    ``Phi[i, a, b]`` nonzero for ``|i-a| <= 1`` and ``|i-b| <= 1``, i.e. vertex
    reach ``p = 1`` -- the production reach.

    The draw happens INSIDE the bounds check, so the random sequence depends on
    ``n_cell``. That is not incidental: the measured pin-versus-length numbers
    in ``phonon/docs/spatial_truncation_derivation.md`` were taken with this
    ordering, and reordering the draws moves them. Preserved deliberately.
    """
    rng = np.random.default_rng(seed)
    phi = np.zeros((n_cell, n_cell, n_cell))
    for i in range(n_cell):
        for a in (i - 1, i, i + 1):
            for b in (i - 1, i, i + 1):
                if 0 <= a < n_cell and 0 <= b < n_cell:
                    phi[i, a, b] = rng.normal()
    return (phi + phi.transpose(0, 2, 1)) / 2.0
