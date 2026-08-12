# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Keldysh structure of the pole sector: the cluster occupation matrix.

The nonequilibrium content of a pole is **not** a scalar occupation. Starting
from the retarded split :math:`G^R = P^R + B^R` with

.. math::
    P^R(\omega) = U D^R(\omega) V^\dagger, \qquad
    D^R_{\alpha\beta} = \frac{\delta_{\alpha\beta}}{\omega - z_\alpha},

where :math:`U = [r_1 \dots r_{N_p}]`, :math:`V = [l_1 \dots l_{N_p}]` and the
pairs are normalised by :math:`l_\alpha^\dagger M'(z_\alpha) r_\alpha = 1`, the
exact Keldysh equation :math:`G^{\lessgtr} = G^R \Sigma^{\lessgtr} G^A` gives for
the pole-pole sector

.. math::
    G_{PP}^{\lessgtr}(\omega) = U D^R(\omega) S^{\lessgtr}(\omega) D^A(\omega) U^\dagger,
    \qquad S^{\lessgtr} = V^\dagger \Sigma_{\rm tot}^{\lessgtr} V,

i.e.

.. math::
    G_{PP}^{\lessgtr}(\omega) = \sum_{\alpha\beta} r_\alpha
        \frac{S^{\lessgtr}_{\alpha\beta}(\omega)}
             {(\omega - z_\alpha)(\omega - z_\beta^*)} r_\beta^\dagger .

The off-diagonal :math:`S_{\alpha\beta}` are modal coherences. Replacing them by
independent scalar occupations :math:`n_\alpha` discards exactly those terms, and
is justified only when the poles are well separated compared with their widths
*and* the projected source is nearly diagonal. Under a temperature bias neither
is guaranteed, so the matrix form is the default and
:func:`coherence_metric` is what licenses the scalar reduction.

Two properties matter for the rest of the solver:

* :math:`G_{PP}^{\lessgtr} = (U D^R) S^{\lessgtr} (U D^R)^\dagger` is a
  **congruence** of the projected source, so it inherits its semidefiniteness.
  That is safer than assigning modal weights independently and clipping negative
  occupations afterwards.
* The split used downstream is ``G_S = G_PP``, ``G_R = G_direct - G_PP``. Both
  halves are then computable, they sum to the untouched ``G`` identically, and
  the pole-background interference is retained rather than dropped.

Sign convention follows the solver: :math:`-i G^{\lessgtr} \succeq 0`
(``phonon/docs/bubble_positivity.md`` Sec. 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qttools import NDArray, xp

__all__ = [
    "PoleCluster",
    "project_source",
    "modal_denominator",
    "pole_retarded",
    "pole_keldysh",
    "occupation_matrix",
    "coherence_metric",
    "source_poly_fit",
    "source_poly_eval",
]


@dataclass
class PoleCluster:
    """A group of poles carried together with their biorthogonal vectors.

    Attributes
    ----------
    z : NDArray
        ``(Np,)`` pole locations in the lower half plane.
    u : NDArray
        ``(n_dof, Np)`` right vectors ``r_alpha``.
    v : NDArray
        ``(n_dof, Np)`` left vectors ``l_alpha``, normalised so that
        ``l^H M'(z) r == 1`` and hence ``R_alpha = r_alpha l_alpha^H``.
    label : str
        Free-form identifier, used for tracking across SCBA iterations.

    """

    z: NDArray
    u: NDArray
    v: NDArray
    label: str = field(default="")

    def __post_init__(self):
        self.z = xp.asarray(self.z, dtype=xp.complex128).reshape(-1)
        self.u = xp.asarray(self.u, dtype=xp.complex128)
        self.v = xp.asarray(self.v, dtype=xp.complex128)
        if self.u.shape != self.v.shape:
            raise ValueError(
                f"u {self.u.shape} and v {self.v.shape} must have the same shape."
            )
        if self.u.shape[-1] != self.z.shape[0]:
            raise ValueError(
                f"{self.u.shape[-1]} vectors for {self.z.shape[0]} poles."
            )
        if bool(xp.any(xp.imag(self.z) >= 0.0)):
            raise ValueError("retarded poles must lie in the lower half plane.")

    @property
    def n_poles(self) -> int:
        return int(self.z.shape[0])

    @property
    def n_dof(self) -> int:
        return int(self.u.shape[0])

    @property
    def omega(self) -> NDArray:
        """Resonance frequencies ``Re z``."""
        return xp.real(self.z)

    @property
    def gamma(self) -> NDArray:
        """Half-widths ``-Im z`` (positive)."""
        return -xp.imag(self.z)

    def residues(self) -> NDArray:
        """``(Np, n_dof, n_dof)`` residue matrices ``R_alpha = r_alpha l_alpha^H``."""
        return xp.einsum("ia,ja->aij", self.u, xp.conj(self.v))

    def isolation(self) -> NDArray:
        r"""Isolation ratio :math:`\eta_\alpha = \min_{\beta\neq\alpha}
        |z_\alpha - z_\beta| / (\gamma_\alpha + \gamma_\beta)` (doc Eq. 136).

        Below ``eta_iso`` the poles must be treated as one coherent cluster
        rather than independently.
        """
        z, g = self.z, self.gamma
        if self.n_poles == 1:
            return xp.full((1,), np.inf)
        d = xp.abs(z[:, None] - z[None, :])
        w = g[:, None] + g[None, :]
        ratio = d / xp.where(w > 0, w, 1.0)
        # Mask the diagonal by selection, not by adding inf: 0 * inf is nan.
        eye = xp.eye(self.n_poles, dtype=bool)
        ratio = xp.where(eye, xp.asarray(np.inf), ratio)
        return xp.min(ratio, axis=1)


def project_source(sigma: NDArray, v: NDArray) -> NDArray:
    r"""Projected Keldysh source :math:`S^{\lessgtr} = V^\dagger \Sigma^{\lessgtr} V`.

    Parameters
    ----------
    sigma : NDArray
        ``(..., n_dof, n_dof)`` total Keldysh self-energy (scattering plus both
        contacts) on the frequency grid.
    v : NDArray
        ``(n_dof, Np)`` left vectors.

    Returns
    -------
    NDArray
        ``(..., Np, Np)`` projected source.

    """
    vh = xp.conj(v).T
    return vh @ sigma @ v


def modal_denominator(omega: NDArray, z: NDArray) -> tuple[NDArray, NDArray]:
    r"""Retarded and advanced modal denominators.

    Returns ``(dr, da)`` with ``dr[..., a] = 1/(omega - z_a)`` and
    ``da[..., b] = 1/(omega - conj(z_b))``; these are the diagonals of
    :math:`D^R` and :math:`D^A`.
    """
    w = xp.asarray(omega, dtype=xp.complex128).reshape(-1, 1)
    zz = xp.asarray(z, dtype=xp.complex128).reshape(1, -1)
    return 1.0 / (w - zz), 1.0 / (w - xp.conj(zz))


def pole_retarded(omega: NDArray, cluster: PoleCluster) -> NDArray:
    r"""Pole part of the retarded Green's function, :math:`P^R = U D^R V^\dagger`.

    Returns ``(n_omega, n_dof, n_dof)``.
    """
    dr, _ = modal_denominator(omega, cluster.z)
    return xp.einsum("ia,wa,ja->wij", cluster.u, dr, xp.conj(cluster.v))


def pole_keldysh(
    omega: NDArray, cluster: PoleCluster, source: NDArray
) -> NDArray:
    r"""Pole-pole Keldysh Green's function, doc Eqs. (83)/(85).

    .. math::
        G_{PP}^{\lessgtr}(\omega) = U D^R S^{\lessgtr} D^A U^\dagger

    Evaluated as a congruence ``(U D^R) S (U D^R)^H`` so the semidefiniteness of
    the source is inherited exactly rather than approximately.

    Parameters
    ----------
    omega : NDArray
        ``(n_omega,)`` real frequencies.
    cluster : PoleCluster
    source : NDArray
        ``(n_omega, Np, Np)`` projected source, or ``(Np, Np)`` to freeze it
        across the window (the smooth-source approximation, doc Eq. 91).

    Returns
    -------
    NDArray
        ``(n_omega, n_dof, n_dof)``.

    """
    dr, _ = modal_denominator(omega, cluster.z)
    s = xp.asarray(source, dtype=xp.complex128)
    if s.ndim == 2:
        s = s[None].repeat(dr.shape[0], axis=0)
    if s.shape[-2:] != (cluster.n_poles, cluster.n_poles):
        raise ValueError(
            f"source has shape {s.shape}, expected (..., {cluster.n_poles}, "
            f"{cluster.n_poles})."
        )
    # L = U D^R, then G = L S L^H -- an explicit congruence.
    lmat = cluster.u[None] * dr[:, None, :]
    return lmat @ s @ xp.conj(lmat).swapaxes(-2, -1)


def hybrid_keldysh_congruence(
    omega: NDArray,
    cluster: PoleCluster,
    r_retarded: NDArray,
    source: NDArray,
    cell_index: NDArray,
) -> NDArray:
    r"""Reconstruct :math:`G^{\lessgtr}` off-grid by congruence, doc review Eq. (7).

    .. math::
        \widetilde G^R(\omega) = P^R(\omega) + R^R_k, \qquad
        \widetilde G^{\lessgtr}(\omega)
        = \widetilde G^R(\omega)\,\Sigma^{\lessgtr}_k\,\widetilde G^A(\omega)

    The alternative in use elsewhere freezes the KELDYSH remainder,
    ``R_k = G^<(w_k) - P^<(w_k)``, and reconstructs ``P^<(w) + R_k``. That is
    exact at the cell centre and unconstrained between centres: ``R_k`` is a
    difference of PSD objects, hence indefinite, and near a narrow pole
    ``P^<`` moves by orders of magnitude across one cell while ``R_k`` is
    frozen. Measured on the bed in ``phonon/studies/_pole_subcell.py``, that
    reconstruction reaches ``eps_PSD = -5.4e-02`` with a relative error of
    1079 % at ``2 gamma/h = 0.04``.

    This form has neither failure. Positivity is structural --
    ``-i G^< = G^R (-i Sigma^<) (G^R)^H`` is a congruence, so it holds for ANY
    ``G^R``, however inaccurate -- and what is frozen per cell is the SOURCE,
    which is smooth, rather than the RESPONSE, which carries the pole. On the
    same bed it stays positive (``+4.0e-07``) with a flat ~14 % error
    independent of the pole width.

    Both agree exactly at cell centres, where ``P^R(w_k) + R^R_k = G^R(w_k)``.

    Parameters
    ----------
    omega : NDArray
        ``(n_omega,)`` real frequencies, NOT required to lie on the grid.
    cluster : PoleCluster
    r_retarded : NDArray
        ``(n_cells, n_dof, n_dof)`` frozen retarded remainder
        ``G^R(w_k) - P^R(w_k)`` per cell.
    source : NDArray
        ``(n_cells, n_dof, n_dof)`` cell-constant Keldysh source. The PSD
        object is ``-i Sigma^{<}``, so this carries the usual factor of i;
        the returned function inherits exactly the semidefiniteness it has.
    cell_index : NDArray
        ``(n_omega,)`` index of the cell containing each frequency.

    Returns
    -------
    NDArray
        ``(n_omega, n_dof, n_dof)``.

    """
    w = xp.asarray(omega, dtype=xp.float64).reshape(-1)
    idx = xp.asarray(cell_index, dtype=xp.int64).reshape(-1)
    if idx.shape[0] != w.shape[0]:
        raise ValueError(
            f"cell_index has {idx.shape[0]} entries for {w.shape[0]} "
            "frequencies."
        )
    rr = xp.asarray(r_retarded, dtype=xp.complex128)
    ss = xp.asarray(source, dtype=xp.complex128)
    if rr.shape[0] != ss.shape[0]:
        raise ValueError(
            f"r_retarded has {rr.shape[0]} cells, source has {ss.shape[0]}."
        )
    gr = pole_retarded(w, cluster) + rr[idx]
    return gr @ ss[idx] @ xp.conj(gr).swapaxes(-2, -1)


def occupation_matrix(
    omega: NDArray, cluster: PoleCluster, source: NDArray, weights: NDArray | None = None
) -> NDArray:
    r"""Pole-cluster covariance matrix, doc Eq. (88).

    .. math::
        N_{\mathcal C} = \frac{i}{2\pi}\int_{\mathcal W} d\omega\, D^R S^< D^A

    The window must be finite: under the frozen-source model the integrand does
    not decay, so the caller restricts ``omega`` to the cluster window (a few
    tens of half-widths).

    Returns
    -------
    NDArray
        ``(Np, Np)``.

    """
    dr, da = modal_denominator(omega, cluster.z)
    s = xp.asarray(source, dtype=xp.complex128)
    if s.ndim == 2:
        s = s[None].repeat(dr.shape[0], axis=0)
    integrand = dr[:, :, None] * s * da[:, None, :]
    w = xp.asarray(omega, dtype=float)
    if weights is None:
        weights = xp.gradient(w) if w.shape[0] > 1 else xp.ones_like(w)
    return (1j / (2.0 * np.pi)) * xp.einsum("w,wab->ab", weights, integrand)


def coherence_metric(n_matrix: NDArray) -> float:
    r"""Coherence indicator, doc Eq. (87).

    .. math::
        \epsilon_{\rm coh} = \frac{\|\operatorname{offdiag} N\|_F}{\|N\|_F}

    Only when this is small may a cluster be reduced to independent scalar modal
    populations. Under a temperature bias, keeping the matrix form is safer.
    """
    n = xp.asarray(n_matrix)
    total = float(xp.linalg.norm(n))
    if total == 0.0:
        return 0.0
    off = n - xp.diag(xp.diag(n))
    return float(xp.linalg.norm(off) / total)


def source_poly_fit(
    omega: NDArray, source: NDArray, centre: float, scale: float, order: int = 2
) -> tuple[NDArray, float]:
    r"""Least-squares polynomial model of the projected source (doc Sec. 15, Option B).

    Fits :math:`S(\omega) \simeq \sum_{m=0}^{p} S_m ((\omega-\Omega_c)/h)^m` over
    the cluster window. All pole-sector integrals of such a model are moments of
    known rational functions and are available in closed form, so the narrow
    denominator never touches a grid.

    A rational (AAA/vector-fitting) model would be more compact but is rejected
    here: its classic failure mode is a spurious upper-half-plane pole, which
    would break the causality of the reconstructed retarded self-energy.

    Parameters
    ----------
    omega : NDArray
        ``(n_omega,)`` sample frequencies in the window.
    source : NDArray
        ``(n_omega, Np, Np)`` sampled source.
    centre, scale : float
        Expansion point and scaling, normally the cluster centre and the grid
        spacing.
    order : int, optional
        Polynomial degree. Default 2.

    Returns
    -------
    coeff : NDArray
        ``(order + 1, Np, Np)`` coefficients.
    residual : float
        Relative Frobenius residual of the fit. The caller demotes the cluster
        when this exceeds its tolerance rather than approximating silently.

    """
    t = (xp.asarray(omega, dtype=float) - centre) / scale
    vander = xp.stack([t**m for m in range(order + 1)], axis=1).astype(xp.complex128)
    s = xp.asarray(source, dtype=xp.complex128)
    flat = s.reshape(s.shape[0], -1)
    coeff, *_ = xp.linalg.lstsq(vander, flat, rcond=None)
    resid = float(
        xp.linalg.norm(vander @ coeff - flat) / max(float(xp.linalg.norm(flat)), 1e-300)
    )
    return coeff.reshape((order + 1,) + s.shape[1:]), resid


def source_poly_eval(
    omega: NDArray, coeff: NDArray, centre: float, scale: float
) -> NDArray:
    """Evaluate a :func:`source_poly_fit` model."""
    t = (xp.asarray(omega, dtype=float) - centre) / scale
    powers = xp.stack([t**m for m in range(coeff.shape[0])], axis=1)
    return xp.einsum("wm,m...->w...", powers.astype(xp.complex128), coeff)
