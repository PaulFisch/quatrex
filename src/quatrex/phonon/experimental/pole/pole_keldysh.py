# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Keldysh structure of the pole sector: the cluster occupation matrix.

The nonequilibrium content of a pole is not a scalar occupation. The exact
Keldysh equation gives the pole-pole sector as ``U D^R S D^A U^dagger`` with
``S = V^dagger Sigma_tot V``, whose off-diagonal entries are modal coherences.
Replacing them by independent scalar occupations discards exactly those terms,
and is justified only when the poles are well separated compared with their
widths AND the projected source is nearly diagonal. Under a temperature bias
neither is guaranteed, so the matrix form is the default and
:func:`coherence_metric` is what licenses a scalar reduction.
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


def occupation_matrix(
    omega: NDArray, cluster: PoleCluster, source: NDArray, weights: NDArray | None = None
) -> NDArray:
    r"""Pole-cluster covariance matrix, doc Eq. (88).

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
    r"""Least-squares polynomial model of the projected source (doc Sec. 15,
    Option B).

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
