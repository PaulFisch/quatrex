# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Continuation of the pole set across SCBA iterations.

Poles move, cross, broaden and merge as the self-energy updates. The naive
approach -- re-solve and sort by frequency -- fails for a structural reason:
near a crossing the individual eigenvectors rotate arbitrarily under a tiny
perturbation while the invariant SUBSPACE stays smooth. So this module tracks
subspaces, not labels.

The design is predictor / corrector / subspace match / contour fallback:

1. **Predictor** (doc Eq. 43). With the biorthogonal normalisation
   :math:`l_\alpha^\dagger M'(z_\alpha) r_\alpha = 1` the first-order response of
   a pole to a change in the scattering self-energy is simply

   .. math:: \delta z_\alpha = l_\alpha^\dagger\,\Delta\Sigma_s^R(z_\alpha)\,r_\alpha,

   which costs one projection and is a far better starting point than a global
   re-solve.
2. **Corrector**: bordered Newton, in :mod:`quatrex.phonon.pole_nevp`.
3. **Cluster and match**: poles closer than :math:`c_{\rm cl}(\gamma_\alpha+\gamma_\beta)`
   are carried as one cluster, and clusters are matched between iterations by
   principal angles between their invariant subspaces (doc Eqs. 57-58). Isolated
   poles may be matched individually by the combined displacement/overlap cost
   of Eq. (61).
4. **Fallback**: a failed correction, a cardinality change or a subspace jump
   triggers a contour rescan.

The pole set is a deterministic FUNCTION of the mixed self-energy, so it is
recomputed every iteration and merely warm-started from the previous one. It is
deliberately **not** part of the mixed state vector: adding it would make the
Anderson/RRE least-squares rank-deficient and would silently invalidate the
exact Jacobian-vector product used by the Newton mixer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from qttools import NDArray, xp

__all__ = [
    "predict_shift",
    "cluster_poles",
    "subspace_basis",
    "principal_angles",
    "subspace_distance",
    "match_cost",
    "match_poles",
    "TrackedCluster",
    "PoleTracker",
]


def predict_shift(l: NDArray, r: NDArray, dsigma: NDArray) -> complex:
    r"""First-order pole shift under a self-energy update, doc Eq. (43).

    .. math:: \delta z_\alpha = l_\alpha^\dagger\,\Delta\Sigma_s^R(z_\alpha)\,r_\alpha

    Valid with the normalisation ``l^H M'(z) r = 1``, which makes the
    denominator ``d_alpha`` unity.

    Parameters
    ----------
    l, r : NDArray
        ``(n_dof,)`` left and right vectors of the pole.
    dsigma : NDArray
        ``(n_dof, n_dof)`` change in the retarded scattering self-energy,
        evaluated at the pole.

    Returns
    -------
    complex

    """
    return complex(xp.vdot(l, dsigma @ r))


def cluster_poles(
    z: NDArray, factor: float = 3.0
) -> list[list[int]]:
    r"""Group poles that overlap, doc Eq. (55).

    Single-linkage on :math:`|z_\alpha - z_\beta| \le c_{\rm cl}(\gamma_\alpha+\gamma_\beta)`.
    Overlapping poles must be carried together: their individual eigenvectors are
    not a stable numerical object even where their span is.

    Parameters
    ----------
    z : NDArray
        ``(Np,)`` pole locations (lower half plane).
    factor : float, optional
        ``c_cl``. Default 3.

    Returns
    -------
    list[list[int]]
        Index groups, each sorted, ordered by their lowest member.

    """
    zz = np.asarray(_host(z)).reshape(-1)
    n = zz.size
    if n == 0:
        return []
    gam = -zz.imag

    # Adjacency, then its transitive closure by repeated squaring. The union-
    # find this replaces walked every PAIR in Python -- O(Np^2) calls per q,
    # and there is one q per transverse momentum. Squaring reaches the closure
    # in ceil(log2(n)) boolean matmuls, so the call count grows with the
    # LOGARITHM of the pole count rather than its square.
    adj = (np.abs(zz[:, None] - zz[None, :])
           <= factor * (gam[:, None] + gam[None, :]))
    adj |= np.eye(n, dtype=bool)
    reach = adj
    for _ in range(int(np.ceil(np.log2(n))) if n > 1 else 0):
        nxt = reach @ reach
        if np.array_equal(nxt, reach):
            break
        reach = nxt

    # A component is named by its lowest member, which also orders the groups.
    label = np.argmax(reach, axis=1)
    order = np.unique(label)
    return [np.nonzero(label == r)[0].tolist() for r in order]


def _host(a):
    return a.get() if hasattr(a, "get") else a


def subspace_basis(vectors: NDArray) -> NDArray:
    """Orthonormal basis of the span of a set of column vectors."""
    q, _ = xp.linalg.qr(xp.asarray(vectors, dtype=xp.complex128))
    return q


def principal_angles(q1: NDArray, q2: NDArray) -> NDArray:
    r"""Principal angles between two subspaces, doc Eqs. (57)-(58).

    The singular values of :math:`Q_1^\dagger Q_2` are the cosines of the
    principal angles. A small largest angle means the same cluster has been
    tracked even if the individual modes exchanged labels.

    Parameters
    ----------
    q1, q2 : NDArray
        ``(n_dof, k)`` orthonormal bases.

    Returns
    -------
    NDArray
        ``(k,)`` angles in radians, ascending.

    """
    s = xp.linalg.svd(xp.conj(q1).T @ q2, compute_uv=False)
    return xp.arccos(xp.clip(xp.real(s), -1.0, 1.0))


def subspace_distance(vectors_a: NDArray, vectors_b: NDArray) -> float:
    """Largest principal angle between the spans of two vector sets (rad)."""
    qa, qb = subspace_basis(vectors_a), subspace_basis(vectors_b)
    if qa.shape[1] != qb.shape[1]:
        return float(np.pi / 2)
    return float(xp.max(principal_angles(qa, qb)))


def match_cost(
    z_old: NDArray,
    r_old: NDArray,
    z_new: NDArray,
    r_new: NDArray,
    *,
    w_z: float = 1.0,
    w_u: float = 1.0,
    s_z: float | None = None,
) -> NDArray:
    r"""Assignment cost between isolated candidates, doc Eq. (61).

    .. math::
        C_{\alpha\beta} = w_z \frac{|z_\alpha - z_\beta|}{s_z}
                        + w_u\left(1 - |\hat r_\alpha^\dagger \hat r_\beta|\right)

    Only for well-separated poles; clusters are matched by subspace instead.

    Parameters
    ----------
    z_old, z_new : NDArray
        Pole locations of the two iterations.
    r_old, r_new : NDArray
        ``(n_dof, N)`` right vectors, matching columns.
    w_z, w_u : float, optional
        Weights of the displacement and overlap terms.
    s_z : float, optional
        Displacement scale. Defaults to the median half-width, so the cost is
        measured in linewidths rather than in THz.

    Returns
    -------
    NDArray
        ``(N_old, N_new)`` cost matrix.

    """
    zo = xp.asarray(z_old).reshape(-1, 1)
    zn = xp.asarray(z_new).reshape(1, -1)
    if s_z is None:
        gam = xp.concatenate([-xp.imag(zo).reshape(-1), -xp.imag(zn).reshape(-1)])
        s_z = float(xp.median(gam)) or 1.0
    disp = xp.abs(zo - zn) / s_z

    ro = r_old / xp.linalg.norm(r_old, axis=0, keepdims=True)
    rn = r_new / xp.linalg.norm(r_new, axis=0, keepdims=True)
    overlap = xp.abs(xp.conj(ro).T @ rn)
    return w_z * disp + w_u * (1.0 - overlap)


def match_poles(
    z_old: NDArray, r_old: NDArray, z_new: NDArray, r_new: NDArray, **kw
) -> NDArray:
    """Optimal global assignment of old to new poles under :func:`match_cost`.

    Returns
    -------
    NDArray
        ``(N,)`` such that old pole ``i`` maps to new pole ``out[i]``.

    """
    cost = np.asarray(_host(match_cost(z_old, r_old, z_new, r_new, **kw)))
    rows, cols = linear_sum_assignment(cost)
    out = np.full(cost.shape[0], -1, dtype=int)
    out[rows] = cols
    return out


@dataclass
class TrackedCluster:
    """One pole cluster carried across SCBA iterations."""

    cid: int
    z: NDArray
    r: NDArray
    l: NDArray
    basis: NDArray
    state: str = "active"
    age: int = 0
    last_angle: float = 0.0

    @property
    def size(self) -> int:
        return int(np.asarray(_host(self.z)).size)


@dataclass
class PoleTracker:
    """State machine over pole clusters (doc Sec. 8).

    Parameters
    ----------
    cluster_factor : float
        ``c_cl`` of Eq. (55).
    angle_tol : float
        Largest principal angle still accepted as the same cluster.
    rescan_iterations : int
        Force a contour audit every N updates even when tracking looks healthy.
    epoch_iterations : int
        Hold membership fixed for this many updates. Sector membership must not
        change every iteration: an approximate implementation is not invariant
        under repartitioning, so a mode that jumps changes the fixed-point map.

    """

    cluster_factor: float = 3.0
    angle_tol: float = 0.35
    rescan_iterations: int = 10
    epoch_iterations: int = 5

    clusters: list[TrackedCluster] = field(default_factory=list)
    iteration: int = 0
    epoch: int = 0
    _next_id: int = 0
    rescan_reasons: list[str] = field(default_factory=list)

    # -- queries ----------------------------------------------------------- #

    def needs_rescan(self) -> bool:
        """Whether a contour audit is due this update."""
        return (
            not self.clusters
            or self.iteration % self.rescan_iterations == 0
            or bool(self.rescan_reasons)
        )

    def membership_frozen(self) -> bool:
        """Whether sector membership is held fixed for the rest of this epoch."""
        return bool(self.iteration % self.epoch_iterations)

    # -- update ------------------------------------------------------------ #

    def adopt(self, z: NDArray, r: NDArray, l: NDArray) -> list[TrackedCluster]:
        """Replace the tracked set from a fresh solve (initialisation / rescan)."""
        self.clusters = []
        for grp in cluster_poles(z, self.cluster_factor):
            idx = xp.asarray(grp)
            vecs = xp.take(r, idx, axis=1)
            self.clusters.append(
                TrackedCluster(
                    cid=self._next_id,
                    z=xp.take(z, idx),
                    r=vecs,
                    l=xp.take(l, idx, axis=1),
                    basis=subspace_basis(vecs),
                )
            )
            self._next_id += 1
        self.rescan_reasons = []
        return self.clusters

    def update(
        self, z: NDArray, r: NDArray, l: NDArray
    ) -> list[TrackedCluster]:
        """Re-cluster a corrected pole set and match it to the previous one.

        Records a rescan reason -- rather than silently re-labelling -- whenever
        the cluster count changes, a cluster changes size, or its subspace turns
        by more than ``angle_tol``.
        """
        self.iteration += 1
        previous = list(self.clusters)
        groups = cluster_poles(z, self.cluster_factor)
        self.rescan_reasons = []

        if len(groups) != len(previous):
            self.rescan_reasons.append(
                f"cluster count {len(previous)} -> {len(groups)}"
            )

        fresh: list[TrackedCluster] = []
        for gi, grp in enumerate(groups):
            idx = xp.asarray(grp)
            vecs = xp.take(r, idx, axis=1)
            basis = subspace_basis(vecs)
            cid, angle = self._next_id, 0.0
            if gi < len(previous):
                prev = previous[gi]
                if prev.size == len(grp):
                    angle = float(xp.max(principal_angles(prev.basis, basis)))
                    cid = prev.cid
                    if angle > self.angle_tol:
                        self.rescan_reasons.append(
                            f"cluster {cid} subspace turned {angle:.3f} rad"
                        )
                else:
                    self.rescan_reasons.append(
                        f"cluster {prev.cid} size {prev.size} -> {len(grp)}"
                    )
            if cid == self._next_id:
                self._next_id += 1
            fresh.append(
                TrackedCluster(
                    cid=cid, z=xp.take(z, idx), r=vecs,
                    l=xp.take(l, idx, axis=1), basis=basis,
                    age=(previous[gi].age + 1 if gi < len(previous) else 0),
                    last_angle=angle,
                )
            )
        self.clusters = fresh
        return fresh
