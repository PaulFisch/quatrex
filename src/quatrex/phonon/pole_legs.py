# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The congruence leg, for every q and every cluster in one pass.

:mod:`~quatrex.phonon.pole_congruence` states the physics one cluster at a
time, which is how it is read and tested. Production cannot be driven that way:
looping over q and clusters in Python cost 6.85 million calls and 33 s per SCBA
iteration on Si (81 q, ~600 poles), against a 7.4 s bubble.

The arithmetic is unchanged. What changes is that the number of Python calls no
longer depends on the number of q, clusters, poles, frequencies or nonzeros --
only on the device blocking, through
:class:`~quatrex.phonon.pole_probe.BlockLayout`. Every step is one gather, one
matmul or one elementwise expression over the whole (q, cluster) stack.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qttools import NDArray, xp

from quatrex.phonon.pole_probe import BlockLayout

__all__ = ["ClusterBatch", "ClusterViews", "CoefficientViews",
           "congruence_legs", "source_fit"]

# Padded pole columns carry zero u and v, so their location is irrelevant to
# every contraction. It only has to stay off the real axis and out of the way
# of z_a - conj(z_b), which is why it sits in the lower half plane.
_PAD_Z = -1.0j


class ClusterViews(Sequence):
    """Per-cluster slices of a batched array, taken only when read.

    The leg results come out as one padded array per q. The consumers want a
    list per cluster, and building that list eagerly is a Python call and a
    device slice per cluster -- which is the cost this module exists to
    remove, reintroduced at the last step for a diagnostic that usually never
    runs. So the list is a view: it costs nothing until something indexes it.
    """

    def __init__(self, batched: NDArray, sizes: list[int], trailing: int = 2):
        self._b, self._sizes, self._t = batched, list(sizes), int(trailing)

    def __len__(self) -> int:
        return len(self._sizes)

    def __getitem__(self, m):
        if isinstance(m, slice):
            return [self[i] for i in range(*m.indices(len(self)))]
        p = self._sizes[m]
        cut = (Ellipsis,) + (slice(0, p),) * self._t
        return self._b[m][cut]


class CoefficientViews(Sequence):
    """``(c_sr, c_rs, c_ss)`` per cluster, sliced only when read.

    The three background coefficients have DIFFERENT trailing shapes --
    ``(Np, n_dof)``, ``(n_dof, Np)`` and ``(Np, Np)`` -- so each is trimmed on
    the axes that carry poles and left alone on the one that carries degrees
    of freedom. Getting that transposed silently hands the mixed sectors a
    coefficient of the wrong rank.
    """

    def __init__(self, coefficients, sizes: list[int]):
        self._c, self._sizes = tuple(coefficients), list(sizes)

    def __len__(self) -> int:
        return len(self._sizes)

    def __getitem__(self, m):
        if isinstance(m, slice):
            return [self[i] for i in range(*m.indices(len(self)))]
        p = self._sizes[m]
        c_sr, c_rs, c_ss = (c[m] for c in self._c)
        return (c_sr[..., :p, :], c_rs[..., :, :p], c_ss[..., :p, :p])


@dataclass
class ClusterBatch:
    """Pole clusters of several q, padded to one rectangle.

        Attributes
        ----------
        z : NDArray
            ``(Q, M, P)`` pole locations; padded entries hold ``_PAD_Z``.
        u, v : NDArray
            ``(Q, M, n_dof, P)`` right and left vectors; padded entries are zero,
            which is what makes the padding inert.
        live : NDArray
            ``(Q, M)`` whether that slot is a real cluster.
        counts : list[int]
            Clusters per q, for unpacking the result.
    """

    z: NDArray
    u: NDArray
    v: NDArray
    live: NDArray
    counts: list[int]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.z.shape)

    @classmethod
    def from_clusters(cls, per_q, n_dof: int) -> "ClusterBatch":
        """Stack ``per_q[q][m]`` clusters, padding the ragged axes.

        One concatenate and one scatter, not a write per cluster: on a device
        the latter is a kernel launch per cluster, which is the cost this whole
        module exists to remove.
        """
        counts = [len(cl) for cl in per_q]
        n_q = len(per_q)
        m_max = max(counts) if counts else 0
        flat = [c for cl in per_q for c in cl]
        p_max = max((int(c.z.shape[0]) for c in flat), default=0)
        if m_max == 0 or p_max == 0:
            empty = xp.zeros((n_q, 0, 0), dtype=xp.complex128)
            return cls(z=empty,
                       u=xp.zeros((n_q, 0, n_dof, 0), dtype=xp.complex128),
                       v=xp.zeros((n_q, 0, n_dof, 0), dtype=xp.complex128),
                       live=xp.zeros((n_q, 0), dtype=bool), counts=counts)

        # Where each cluster's poles land in the (Q, M, P) rectangle.
        npoles = np.array([int(c.z.shape[0]) for c in flat], dtype=np.int64)
        per_q_counts = np.array(counts, dtype=np.int64)
        # slot = q * m_max + m, without a pass over the q.
        slot = (np.repeat(np.arange(n_q, dtype=np.int64) * m_max, per_q_counts)
                + np.arange(int(per_q_counts.sum()), dtype=np.int64)
                - np.repeat(np.cumsum(per_q_counts) - per_q_counts,
                            per_q_counts))
        start = np.repeat(slot * p_max, npoles)
        within = (np.arange(int(npoles.sum()))
                  - np.repeat(np.cumsum(npoles) - npoles, npoles))
        dest = xp.asarray(start + within)

        z = xp.full(n_q * m_max * p_max, _PAD_Z, dtype=xp.complex128)
        z[dest] = xp.concatenate([c.z for c in flat])
        u = xp.zeros((n_q * m_max * p_max, n_dof), dtype=xp.complex128)
        v = xp.zeros_like(u)
        u[dest] = xp.swapaxes(xp.concatenate([c.u for c in flat], axis=1), 0, 1)
        v[dest] = xp.swapaxes(xp.concatenate([c.v for c in flat], axis=1), 0, 1)

        live = np.zeros(n_q * m_max, dtype=bool)
        live[slot] = True
        return cls(
            z=z.reshape(n_q, m_max, p_max),
            u=xp.swapaxes(u.reshape(n_q, m_max, p_max, n_dof), -1, -2),
            v=xp.swapaxes(v.reshape(n_q, m_max, p_max, n_dof), -1, -2),
            live=xp.asarray(live.reshape(n_q, m_max)), counts=counts)

    def pole_counts(self) -> list[list[int]]:
        """Poles per cluster, per q -- for slicing the padded results back."""
        return self._counts_per_cluster


def _pole_factors(z: NDArray, omega: NDArray, widths: NDArray | None):
    r"""``(d1, d2)``: the pole factors, at the cell centres or averaged.

    ``widths is None`` gives the point sample ``1/(w - z)`` and its outer
    product; otherwise the exact cell averages of
    :func:`~quatrex.phonon.pole_congruence.cell_weights`, written over the
    whole (q, cluster) stack.
    """
    w = xp.asarray(omega, dtype=xp.complex128)[None, None, :, None]
    zz = z[:, :, None, :]
    if widths is None:
        d1 = 1.0 / (w - zz)
        return d1, d1[..., :, None] * xp.conj(d1)[..., None, :]
    h = xp.asarray(widths, dtype=xp.float64)[None, None, :, None]
    d1 = (xp.log(w + 0.5 * h - zz) - xp.log(w - 0.5 * h - zz)) / h
    # z_a - conj(z_b) has imaginary part -(gamma_a + gamma_b) < 0, so it never
    # vanishes: the retarded poles are strictly in the lower half plane.
    gap = z[:, :, None, :, None] - xp.conj(z)[:, :, None, None, :]
    d2 = (d1[..., :, None] - xp.conj(d1)[..., None, :]) / gap
    return d1, d2


def _sectors(layout: BlockLayout, u_blk, u_shift, c_rs_blk, c_ss, d1, d2):
    r"""``SR + RS + SS`` as a band tensor, for one choice of pole factors.

    ``RS`` is formed and ``SR`` taken as its negated conjugate transpose --
    ``c_sr`` is defined as ``-conj(c_rs^T)``, so this is an identity, not an
    approximation. ``SS`` is the only genuinely bilinear term.
    """
    nb, bm = layout.n_blocks, layout.bmax
    uc = xp.conj(u_shift)                                   # (Q,M,nb,3*bm,P)

    # RS: sum_a c_rs[.., row, a] conj(d1[a]) conj(u[col, a])
    rs = (c_rs_blk * xp.conj(d1)[..., None, None, :]) @ xp.swapaxes(uc, -1, -2)[:, :, None]
    rs = _to_band(rs, nb, bm)
    sr = -xp.conj(layout.band_transpose(rs))

    # SS: sum_ab u[row, a] K[a, b] conj(u[col, b]), K = d2 * c_ss
    t = u_blk[:, :, None] @ (d2 * c_ss)[..., None, :, :]
    ss = t @ xp.swapaxes(uc, -1, -2)[:, :, None]
    return sr + rs + _to_band(ss, nb, bm)


def _to_band(x: NDArray, nb: int, bm: int) -> NDArray:
    """``(..., nb, bm, 3*bm) -> (..., nb, 3, bm, bm)``."""
    return xp.swapaxes(x.reshape(x.shape[:-1] + (3, bm)), -3, -2)


def congruence_legs(
    batch: ClusterBatch,
    layout: BlockLayout,
    sigma: NDArray,
    g_retarded: NDArray,
    corners: tuple,
    omega: NDArray,
    cell_widths: NDArray,
):
    r"""The congruence leg for every q and cluster at once.

        Parameters
        ----------
        batch : ClusterBatch
            Padded clusters, ``(Q, M, P)``.
        layout : BlockLayout
            The shared pattern-to-band map.
        sigma : NDArray
            ``(Q, n_omega, nnz)`` Keldysh self-energy on the stored pattern.
        g_retarded : NDArray
            ``(Q, n_omega, nnz)`` retarded Green's function on the same pattern.
        corners : tuple
            ``((block, i), ...)`` dense contact self-energies and the block row
            they sit on. ``block`` is ``(Q, n_omega, b, b)``.
        omega, cell_widths : NDArray
            ``(n_omega,)`` cell centres and widths.

        Returns
        -------
        sample : NDArray
            ``(Q, M, n_omega, nnz)`` the grid sample MINUS the cell average -- what
            the ring must give up so that what it keeps is the PSD congruence.
        source : NDArray
            ``(Q, M, n_omega, P, P)`` the projected source ``V^dagger Sigma_tot V``.
        coefficients : tuple
            ``(c_sr, c_rs, c_ss)`` over the whole stack.
    """
    n_q, m_max, p_max = batch.shape
    if m_max == 0 or p_max == 0:
        w = int(xp.asarray(omega).shape[0])
        z = xp.zeros((n_q, 0, w, layout.nnz), dtype=xp.complex128)
        return z, xp.zeros((n_q, 0, w, 0, 0), dtype=xp.complex128), ()

    # Sigma_tot: the scattering self-energy PLUS both contacts. Dropping the
    # contacts drops the injection that drives the device.
    sig_band = layout.band(sigma)
    for block, i in corners:
        if block is None:
            continue
        b = int(xp.asarray(block).shape[-1])
        sig_band[:, :, i, 1, :b, :b] += xp.asarray(block)

    v, u = batch.v, batch.u
    sv = layout.apply_band(sig_band[:, None], v[:, :, None])
    gsv = layout.apply_band(layout.band(g_retarded)[:, None], sv)

    # c_ss IS the projected source: both are V^dagger Sigma_tot V.
    c_ss = xp.swapaxes(xp.conj(v), -1, -2)[:, :, None] @ sv
    d, _ = _pole_factors(batch.z, omega, None)
    c_rs = gsv - (u[:, :, None] * d[..., None, :]) @ c_ss
    c_sr = -xp.conj(xp.swapaxes(c_rs, -1, -2))

    nb, bm = layout.n_blocks, layout.bmax
    u_blk = layout.to_blocks(u)
    u_shift = layout.band_neighbours(u_blk).reshape(
        u_blk.shape[:-3] + (nb, 3 * bm, p_max))
    c_rs_blk = layout.to_blocks(c_rs)

    d1, d2 = _pole_factors(batch.z, omega, None)
    point = _sectors(layout, u_blk, u_shift, c_rs_blk, c_ss, d1, d2)
    a1, a2 = _pole_factors(batch.z, omega, cell_widths)
    average = _sectors(layout, u_blk, u_shift, c_rs_blk, c_ss, a1, a2)

    return layout.unband(point - average), c_ss, (c_sr, c_rs, c_ss)


def source_fit(batch: ClusterBatch, source: NDArray, omega: NDArray,
               window: int = 4, order: int = 2) -> NDArray:
    r"""``eps_fit`` per (q, cluster): how far the source strays from its pair
    value.

        Returns
        -------
        NDArray
            ``(Q, M)`` relative residual, zero for a padded slot.
    """
    from quatrex.phonon.pole_kernel import LocalFitPlan

    n_q, m_max, p_max = batch.shape
    if m_max == 0 or p_max == 0:
        return xp.zeros((n_q, m_max))
    w = xp.asarray(omega, dtype=xp.float64)
    n_w = int(w.shape[0])

    # (Q, M, P, P) -> one pair per entry; the source column each pair reads.
    z = batch.z
    za = xp.broadcast_to(z[..., :, None], (n_q, m_max, p_max, p_max))
    zb = xp.conj(xp.broadcast_to(z[..., None, :], (n_q, m_max, p_max, p_max)))
    anchor = 0.5 * (xp.real(za) + xp.real(zb))
    flat_anchor = anchor.reshape(-1)
    n_pair = int(flat_anchor.shape[0])

    plan = LocalFitPlan(w, xp.concatenate([flat_anchor, flat_anchor]),
                        order=order, window=window)
    row, idx, pos = plan.compact_weights(
        xp.concatenate([za.reshape(-1), zb.reshape(-1)]))

    # src laid out one row per pair: (2*n_pair, n_omega)
    per_pair = xp.swapaxes(xp.swapaxes(source, -3, -1), -3, -2).reshape(-1, n_w)
    sel = xp.concatenate([per_pair, per_pair], axis=0)
    gathered = xp.take_along_axis(sel, idx, axis=1)
    vals = xp.sum(row * xp.where(pos[:, None], gathered, xp.conj(gathered)),
                  axis=1)
    pair = 0.5 * (vals[:n_pair] + vals[n_pair:])              # (Q*M*P*P,)

    # ... and the worst deviation of the samples in each pair's own window.
    centre = xp.abs(0.5 * (xp.real(za) + xp.real(zb))).reshape(-1)
    k0 = xp.argmin(xp.abs(w[None, :] - centre[:, None]), axis=1)
    span = xp.arange(-window, window + 1)
    # Clamping at the grid edge only REPEATS a sample already inside the
    # window, and the reduction is a maximum, so the answer is unchanged.
    win = xp.clip(k0[:, None] + span[None, :], 0, n_w - 1)
    local = xp.take_along_axis(per_pair, win, axis=1)
    neg = (0.5 * (xp.real(za) + xp.real(zb))).reshape(-1) < 0.0
    local = xp.where(neg[:, None], xp.conj(local), local)

    dev = xp.abs(local - pair[:, None]).reshape(n_q, m_max, p_max, p_max, -1)
    scale = xp.abs(source).reshape(n_q, m_max, -1).max(axis=-1)
    worst = dev.reshape(n_q, m_max, -1).max(axis=-1)
    return xp.where(scale > 0, worst / xp.where(scale > 0, scale, 1.0), 0.0)
