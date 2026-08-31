# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Connecting the pole sector to the bubble, on the stored sparsity pattern.

The solver produces pole clusters; the bubble consumes two things:

``G_PP^{<,>}``
    subtracted from the legs, so the FFT ring only sees the smooth remainder
    ``G_R = G - G_PP``;
``Sigma_SS^{<,>,R}``
    added back analytically, so the pole-pole channel never touches a grid.

Everything here contracts **directly on the stored nnz pattern**. That is not a
micro-optimisation: the intermediate dense objects would be
``(n_omega, n_dof, n_dof)``, which at production size (2001 frequencies, 300
degrees of freedom) is 2.9 GB for a single buffer. The projected source
``S = V^dagger Sigma V`` is only ``(n_omega, N_p, N_p)`` and the reconstruction
back onto the pattern is a per-entry contraction, so the whole path is
``O(n_omega * (N_p^2 + nnz))``.

The projected source must be the **total** Keldysh source -- the scattering
self-energy that entered the Dyson solve plus both contacts -- because that is
what ``G^{<,>} = G^R Sigma_tot^{<,>} G^A`` is built from. Dropping the contact
part would silently remove the injection that drives the device.
"""
from __future__ import annotations

import numpy as np

from qttools import NDArray, xp

from quatrex.phonon.experimental.pole.pole_bubble import modal_convolution
from quatrex.phonon.experimental.pole.pole_keldysh import PoleCluster, modal_denominator
from quatrex.phonon.units import HBAR_SI

__all__ = [
    "source_at_poles",
    "source_variation",
    "pole_keldysh_pf_sparse",
    "mixed_vertex_blocks",
    "mixed_self_energy_sparse",
    "project_source_sparse",
    "add_contact_source",
    "pole_keldysh_sparse",
    "modal_vertex_blocks",
    "ss_self_energy_sparse",
    "analytic_prefactor",
]


def analytic_prefactor() -> complex:
    r"""Prefactor of the ANALYTIC bubble, ``i*hbar/2``.

    The production prefactor is ``0.5j*hbar*dw/(2*pi)``: the ``dw/(2*pi)`` turns
    the discrete sum into ``int dw'/(2*pi)``. The analytic convolution evaluates
    that integral in closed form, so it must NOT carry the grid spacing --
    including it here would make the analytic sector scale with a grid it never
    touches.
    """
    return 0.5j * HBAR_SI


def _host(a):
    return a.get() if hasattr(a, "get") else a


def project_source_sparse(
    sigma_values: NDArray, rows: NDArray, cols: NDArray, v: NDArray
) -> NDArray:
    r"""``S(w) = V^dagger Sigma(w) V`` from sparse values, without densifying.

        Parameters
        ----------
        sigma_values : NDArray
            ``(n_omega, nnz)`` self-energy on the stored pattern.
        rows, cols : NDArray
            ``(nnz,)`` global row/column indices.
        v : NDArray
            ``(n_dof, Np)`` left vectors.

        Returns
        -------
        NDArray
            ``(n_omega, Np, Np)``.
    """
    vr = xp.conj(xp.take(v, xp.asarray(rows), axis=0))     # (nnz, Np)
    vc = xp.take(v, xp.asarray(cols), axis=0)              # (nnz, Np)
    return xp.einsum("ka,wk,kb->wab", vr, sigma_values, vc)


def add_contact_source(
    source: NDArray, corner: NDArray, v: NDArray, offset: int
) -> NDArray:
    r"""Add a dense contact block's contribution to the projected source.

        Parameters
        ----------
        source : NDArray
            ``(n_omega, Np, Np)`` accumulator.
        corner : NDArray
            ``(n_omega, b, b)`` contact self-energy on its diagonal block.
        v : NDArray
            ``(n_dof, Np)`` left vectors.
        offset : int
            Row index at which the block starts.
    """
    b = int(corner.shape[-1])
    vb = v[offset:offset + b]                              # (b, Np)
    return source + xp.einsum("ia,wij,jb->wab", xp.conj(vb), corner, vb)


def pole_keldysh_sparse(
    omega: NDArray,
    cluster: PoleCluster,
    source: NDArray,
    rows: NDArray,
    cols: NDArray,
) -> NDArray:
    r"""``G_PP^{<,>}`` evaluated only on the stored pattern.

    ``G_PP = (U D^R) S (U D^R)^dagger`` is a congruence, so the source's
    semidefiniteness is inherited exactly. Only the pattern entries are formed.

    Returns
    -------
    NDArray
        ``(n_omega, nnz)``.

    """
    dr, _ = modal_denominator(omega, cluster.z)            # (n_omega, Np)
    lr = xp.take(cluster.u, xp.asarray(rows), axis=0)      # (nnz, Np)
    lc = xp.conj(xp.take(cluster.u, xp.asarray(cols), axis=0))
    # G[w,k] = sum_ab  U[r_k,a] dr[w,a] S[w,a,b] conj(dr[w,b]) conj(U[c_k,b])
    mid = dr[:, :, None] * source * xp.conj(dr)[:, None, :]
    return xp.einsum("ka,wab,kb->wk", lr, mid, lc)


def modal_vertex_blocks(
    phi_blocks: dict, block_sizes: NDArray, u: NDArray, conjugate: bool,
    v: NDArray | None = None,
) -> NDArray:
    r"""Project the block-sparse cubic vertex onto the modal basis."""
    sizes = np.asarray(_host(block_sizes), dtype=int)
    off = np.concatenate(([0], np.cumsum(sizes)))
    n_dof, npp = int(off[-1]), int(u.shape[1])
    uu = xp.conj(u) if conjugate else u
    vv = uu if v is None else (xp.conj(v) if conjugate else v)
    nqq = int(vv.shape[1])
    out = xp.zeros((n_dof, npp, nqq), dtype=xp.complex128)
    for (i, k1, k2), blk in phi_blocks.items():
        b = xp.asarray(blk, dtype=xp.complex128)
        u1 = uu[off[k1]:off[k1 + 1]]
        u2 = vv[off[k2]:off[k2 + 1]]
        out[off[i]:off[i + 1]] += xp.einsum("mab,aA,bB->mAB", b, u1, u2)
    return out


def ss_self_energy_sparse(
    omega: NDArray,
    cluster: PoleCluster,
    source_a: NDArray,
    source_b: NDArray,
    vl: NDArray,
    vr: NDArray,
    rows: NDArray,
    cols: NDArray,
    prefactor: complex | None = None,
    retarded_only: bool = False,
    cell: float | None = None,
) -> NDArray:
    r"""Analytic pole-pole self-energy, evaluated on the stored pattern.

        Returns
        -------
        NDArray
            ``(n_omega, nnz)``.
    """
    if prefactor is None:
        prefactor = analytic_prefactor()
    c = modal_convolution(omega, cluster, source_a, source_b,
                          retarded_only=retarded_only, cell=cell)
    left = xp.take(vl, xp.asarray(rows), axis=0)           # (nnz, Np, Np)
    right = xp.take(vr, xp.asarray(cols), axis=0)          # (nnz, Np, Np)
    return prefactor * xp.einsum("kAB,kGD,wADBG->wk", left, right, c)


def mixed_vertex_blocks(
    phi_blocks: dict, block_sizes: NDArray, u: NDArray, *, leg: int,
    conjugate: bool,
) -> NDArray:
    r"""Vertex with ONE leg projected onto the modal basis (doc Eq. 119)."""
    if leg not in (0, 1):
        raise ValueError(f"leg must be 0 or 1 (got {leg}).")
    sizes = np.asarray(_host(block_sizes), dtype=int)
    off = np.concatenate(([0], np.cumsum(sizes)))
    n_dof, npp = int(off[-1]), int(u.shape[1])
    uu = xp.conj(u) if conjugate else u
    out = xp.zeros((n_dof, n_dof, npp), dtype=xp.complex128)
    for (i, k1, k2), blk in phi_blocks.items():
        b = xp.asarray(blk, dtype=xp.complex128)
        if leg == 0:
            # Reduce the first leg (block k1); the second (k2) survives.
            out[off[i]:off[i + 1], off[k2]:off[k2 + 1]] += xp.einsum(
                "mab,aA->mbA", b, uu[off[k1]:off[k1 + 1]])
        else:
            # Reduce the second leg (block k2); the first (k1) survives.
            out[off[i]:off[i + 1], off[k1]:off[k1 + 1]] += xp.einsum(
                "mab,bA->maA", b, uu[off[k2]:off[k2 + 1]])
    return out


def _mixed_one_sector(
    omega: NDArray,
    cluster: PoleCluster,
    source: NDArray,
    g_reg: NDArray,
    freqs: NDArray,
    bl: NDArray,
    br: NDArray,
    rows: NDArray,
    cols: NDArray,
    g_partner: NDArray,
    prefactor: complex | None = None,
    max_nnz: int = 4096,
) -> NDArray:
    r"""ONE mixed sector, given its pair of single-leg vertices.

        Parameters
        ----------
        g_reg : NDArray
            ``(n_freq, nnz)`` regular Green's function ``G - G_PP`` on the pattern.
        bl, br : NDArray
            ``(n_dof, n_dof, Np)`` single-leg modal vertices from
            :func:`mixed_vertex_blocks`.
        max_nnz : int
            Guard. The pattern-level contraction here is ``O(nnz^2)`` per pole pair,
            which is fine for the small devices this sector is currently gated to
            and hopeless at production size. The production route reuses the ring's
            block-structured GEMM at fixed frequency; refusing loudly is better than
            quietly running for hours.
    """
    from quatrex.phonon.experimental.pole.pole_audit import transpose_index
    from quatrex.phonon.experimental.pole.pole_bubble import _split_leg
    from quatrex.phonon.experimental.pole.pole_mixed import bosonic_extend, mixed_convolution_batched

    nnz = int(np.asarray(_host(rows)).size)
    if nnz > max_nnz:
        raise NotImplementedError(
            f"sr_rs_self_energy_sparse: {nnz} pattern entries exceeds the "
            f"{max_nnz} guard. This routine contracts at the pattern level, "
            "which is O(nnz^2); the production path must reuse the ring's "
            "block-structured contraction at fixed frequency."
        )
    if prefactor is None:
        prefactor = analytic_prefactor()

    g_reg, freqs = bosonic_extend(
        g_reg, g_partner, freqs, transpose_index=transpose_index(rows, cols))

    poles, coeffs = _split_leg(cluster, source)
    npp = cluster.n_poles
    r_idx, c_idx = xp.asarray(rows), xp.asarray(cols)
    out = None
    for al in range(npp):
        for dl in range(npp):
            m = None
            for j in range(2):
                term = mixed_convolution_batched(
                    omega, complex(poles[al, dl, j]), complex(coeffs[al, dl, j]),
                    g_reg, freqs,
                )
                m = term if m is None else m + term
            # Frequency-local contraction onto the OUTPUT pattern entries.
            left = xp.take(bl[..., al], r_idx, axis=0)     # (nnz_out, n_dof)
            right = xp.take(br[..., dl], c_idx, axis=0)    # (nnz_out, n_dof)
            lhs = xp.take(left, r_idx, axis=1)             # (nnz_out, nnz_in)
            rhs = xp.take(right, c_idx, axis=1)
            contrib = xp.einsum("ok,wk,ok->wo", lhs, m, rhs)
            out = contrib if out is None else out + contrib
    return prefactor * out


def mixed_self_energy_sparse(
    omega: NDArray,
    cluster: PoleCluster,
    source: NDArray,
    g_reg: NDArray,
    g_partner: NDArray,
    freqs: NDArray,
    phi_blocks: dict,
    block_sizes: NDArray,
    rows: NDArray,
    cols: NDArray,
    prefactor: complex | None = None,
    max_nnz: int = 4096,
) -> NDArray:
    r"""``Sigma_SR + Sigma_RS`` -- the mixed sectors, as a symmetric pair."""
    m_vertex = mixed_vertex_blocks
    kw = dict(freqs=freqs, rows=rows, cols=cols, g_partner=g_partner,
              prefactor=prefactor, max_nnz=max_nnz)
    sr = _mixed_one_sector(
        omega, cluster, source, g_reg,
        bl=m_vertex(phi_blocks, block_sizes, cluster.u, leg=0, conjugate=False),
        br=m_vertex(phi_blocks, block_sizes, cluster.u, leg=1, conjugate=True),
        **kw,
    )
    rs = _mixed_one_sector(
        omega, cluster, source, g_reg,
        bl=m_vertex(phi_blocks, block_sizes, cluster.u, leg=1, conjugate=False),
        br=m_vertex(phi_blocks, block_sizes, cluster.u, leg=0, conjugate=True),
        **kw,
    )
    return sr + rs



def block_offsets(block_sizes: NDArray) -> NDArray:
    """Cumulative block offsets, ``(n_blocks + 1,)`` on the host."""
    sizes = np.asarray(_host(block_sizes), dtype=int)
    return np.concatenate(([0], np.cumsum(sizes)))


def blocks_from_pattern(
    values: NDArray, rows: NDArray, cols: NDArray, block_sizes: NDArray
) -> dict[tuple[int, int], NDArray]:
    """Scatter ``(n_omega, nnz)`` pattern values into dense ``(n_omega, b_i,
    b_j)`` blocks, one per occupied block pair."""
    off = block_offsets(block_sizes)
    r = np.asarray(_host(rows), dtype=np.int64)
    c = np.asarray(_host(cols), dtype=np.int64)
    br = np.searchsorted(off, r, side="right") - 1
    bc = np.searchsorted(off, c, side="right") - 1
    out: dict[tuple[int, int], NDArray] = {}
    for key in {(int(i), int(j)) for i, j in zip(br, bc)}:
        i, j = key
        sel = np.flatnonzero((br == i) & (bc == j))
        blk = xp.zeros(
            (values.shape[0], int(off[i + 1] - off[i]), int(off[j + 1] - off[j])),
            dtype=values.dtype,
        )
        blk[:, xp.asarray(r[sel] - off[i]), xp.asarray(c[sel] - off[j])] = \
            values[:, xp.asarray(sel)]
        out[key] = blk
    return out


def pattern_from_blocks(
    blocks: dict[tuple[int, int], NDArray],
    rows: NDArray,
    cols: NDArray,
    block_sizes: NDArray,
    n_omega: int,
    dtype=None,
) -> NDArray:
    """Gather dense blocks back onto ``(n_omega, nnz)`` pattern values.

    Block pairs absent from ``blocks`` contribute zero -- that is how the
    ``|I - J| <= 1`` output pin is applied, rather than by masking afterwards.
    """
    off = block_offsets(block_sizes)
    r = np.asarray(_host(rows), dtype=np.int64)
    c = np.asarray(_host(cols), dtype=np.int64)
    br = np.searchsorted(off, r, side="right") - 1
    bc = np.searchsorted(off, c, side="right") - 1
    if dtype is None:
        dtype = next(iter(blocks.values())).dtype if blocks else xp.complex128
    out = xp.zeros((n_omega, r.size), dtype=dtype)
    for (i, j), blk in blocks.items():
        sel = np.flatnonzero((br == i) & (bc == j))
        if sel.size == 0:
            continue
        out[:, xp.asarray(sel)] = blk[
            :, xp.asarray(r[sel] - off[i]), xp.asarray(c[sel] - off[j])]
    return out


def mixed_vertex_block_dict(
    phi_blocks: dict, block_sizes: NDArray, u: NDArray, *, leg: int,
    conjugate: bool,
) -> dict[tuple[int, int], NDArray]:
    """Block-sparse form of :func:`mixed_vertex_blocks`.

    Same object, ``(b_i, b_j, Np)`` per occupied block pair instead of one
    dense ``(n_dof, n_dof, Np)`` array. ``leg`` and ``conjugate`` mean exactly
    what they do there and remain independent.
    """
    if leg not in (0, 1):
        raise ValueError(f"leg must be 0 or 1 (got {leg}).")
    off = block_offsets(block_sizes)
    npp = int(u.shape[1])
    uu = xp.conj(u) if conjugate else u
    out: dict[tuple[int, int], NDArray] = {}
    for (i, k1, k2), blk in phi_blocks.items():
        b = xp.asarray(blk, dtype=xp.complex128)
        surv = k2 if leg == 0 else k1
        red = k1 if leg == 0 else k2
        spec = "mab,aA->mbA" if leg == 0 else "mab,bA->maA"
        contrib = xp.einsum(spec, b, uu[off[red]:off[red + 1]])
        key = (int(i), int(surv))
        if key in out:
            out[key] = out[key] + contrib
        else:
            out[key] = contrib
    return out


def _mixed_one_sector_blocked(
    omega: NDArray,
    cluster: PoleCluster,
    source: NDArray,
    g_reg: NDArray,
    bl: dict[tuple[int, int], NDArray],
    br: dict[tuple[int, int], NDArray],
    freqs: NDArray,
    rows: NDArray,
    cols: NDArray,
    block_sizes: NDArray,
    g_partner: NDArray,
    prefactor: complex | None = None,
) -> NDArray:
    """ONE mixed sector via the block triple product. See
    :func:`_mixed_one_sector` for the physics; this is the same object at
    device-scale cost."""
    from quatrex.phonon.experimental.pole.pole_audit import transpose_index
    from quatrex.phonon.experimental.pole.pole_bubble import _split_leg
    from quatrex.phonon.experimental.pole.pole_mixed import bosonic_extend, mixed_convolution_batched

    if prefactor is None:
        prefactor = analytic_prefactor()

    g_reg, freqs = bosonic_extend(
        g_reg, g_partner, freqs, transpose_index=transpose_index(rows, cols))

    poles, coeffs = _split_leg(cluster, source)
    npp = cluster.n_poles
    n_omega = int(omega.shape[0])

    m_blocks = {}
    for al in range(npp):
        for dl in range(npp):
            m = None
            for j in range(2):
                term = mixed_convolution_batched(
                    omega, complex(poles[al, dl, j]), complex(coeffs[al, dl, j]),
                    g_reg, freqs,
                )
                m = term if m is None else m + term
            m_blocks[(al, dl)] = blocks_from_pattern(m, rows, cols, block_sizes)

    # Output pattern -> which block pairs must be produced.
    off = block_offsets(block_sizes)
    r = np.asarray(_host(rows), dtype=np.int64)
    c = np.asarray(_host(cols), dtype=np.int64)
    out_pairs = {(int(i), int(j)) for i, j in zip(
        np.searchsorted(off, r, side="right") - 1,
        np.searchsorted(off, c, side="right") - 1)}

    acc: dict[tuple[int, int], NDArray] = {}
    for (i_out, j_out) in out_pairs:
        total = None
        for al in range(npp):
            for dl in range(npp):
                for (a, b), mab in m_blocks[(al, dl)].items():
                    left = bl.get((i_out, a))
                    right = br.get((j_out, b))
                    if left is None or right is None:
                        continue
                    term = xp.einsum(
                        "ip,wpq,jq->wij",
                        left[..., al], mab, right[..., dl])
                    total = term if total is None else total + term
        if total is not None:
            acc[(i_out, j_out)] = total

    return prefactor * pattern_from_blocks(
        acc, rows, cols, block_sizes, n_omega, dtype=xp.complex128)


def mixed_self_energy_blocked(
    omega: NDArray,
    cluster: PoleCluster,
    source: NDArray,
    g_reg: NDArray,
    g_partner: NDArray,
    freqs: NDArray,
    phi_blocks: dict,
    block_sizes: NDArray,
    rows: NDArray,
    cols: NDArray,
    prefactor: complex | None = None,
) -> NDArray:
    """``Sigma_SR + Sigma_RS`` at device scale.

    Drop-in replacement for :func:`mixed_self_energy_sparse` with no ``nnz``
    guard: the pattern-level O(nnz^2) intermediate is replaced by block
    triple products. Pinned against the pattern-level form, which stays as the
    small-size reference.
    """
    kw = dict(freqs=freqs, rows=rows, cols=cols, block_sizes=block_sizes,
              g_partner=g_partner, prefactor=prefactor)
    vd = mixed_vertex_block_dict
    sr = _mixed_one_sector_blocked(
        omega, cluster, source, g_reg,
        bl=vd(phi_blocks, block_sizes, cluster.u, leg=0, conjugate=False),
        br=vd(phi_blocks, block_sizes, cluster.u, leg=1, conjugate=True),
        **kw,
    )
    rs = _mixed_one_sector_blocked(
        omega, cluster, source, g_reg,
        bl=vd(phi_blocks, block_sizes, cluster.u, leg=1, conjugate=False),
        br=vd(phi_blocks, block_sizes, cluster.u, leg=0, conjugate=True),
        **kw,
    )
    return sr + rs


def source_at_poles(
    source: NDArray, freqs: NDArray, cluster: PoleCluster,
    order: int = 2, window: int = 4,
) -> NDArray:
    r"""The projected source for each pole PAIR, continued to the poles.

        Returns
        -------
        NDArray
            ``(Np, Np)``.
    """
    from quatrex.phonon.experimental.pole.pole_kernel import LocalFitPlan

    z = xp.asarray(cluster.z, dtype=xp.complex128)
    npp = int(z.shape[0])
    src = xp.asarray(source, dtype=xp.complex128)
    flat = src.reshape(src.shape[0], -1)

    a_idx = xp.repeat(xp.arange(npp), npp)
    b_idx = xp.tile(xp.arange(npp), npp)
    za, zbb = z[a_idx], xp.conj(z[b_idx])
    anchor = 0.5 * (xp.real(za) + xp.real(zbb))

    plan = LocalFitPlan(freqs, xp.concatenate([anchor, anchor]),
                        order=order, window=window)
    w_pos, w_mir = plan.weights(xp.concatenate([za, zbb]))

    # Column of the flattened source that pair (a, b) reads.
    col = a_idx * npp + b_idx
    sel = xp.take(flat, xp.concatenate([col, col]), axis=1).T     # (2Np^2, K)
    vals = xp.sum(w_pos * sel + w_mir * xp.conj(sel), axis=1)
    n = npp * npp
    return (0.5 * (vals[:n] + vals[n:])).reshape(npp, npp)


def source_variation(
    source: NDArray, freqs: NDArray, cluster: PoleCluster, window: int = 4
) -> float:
    r"""How far the projected source strays from its per-pair value."""
    w = xp.asarray(freqs, dtype=xp.float64)
    z = xp.asarray(cluster.z, dtype=xp.complex128)
    src = xp.asarray(source, dtype=xp.complex128)
    pair = source_at_poles(source, freqs, cluster)

    scale = float(xp.abs(src).max())
    if scale == 0.0:
        return 0.0

    centre = 0.5 * (xp.real(z)[:, None] + xp.real(z)[None, :])    # (Np, Np)
    flat_c = xp.abs(centre).reshape(-1)                           # (Np^2,)
    k0 = xp.argmin(xp.abs(w[None, :] - flat_c[:, None]), axis=1)
    span = xp.arange(-window, window + 1)
    idx = xp.clip(k0[:, None] + span[None, :], 0, int(w.shape[0]) - 1)

    local = xp.take(src.reshape(src.shape[0], -1), idx, axis=0)   # (Np^2, win, Np^2)
    ab = xp.arange(int(flat_c.shape[0]))
    local = local[ab[:, None], xp.arange(int(idx.shape[1]))[None, :], ab[:, None]]
    # A pair whose centre is negative is served by the mirrored branch.
    neg = (centre.reshape(-1) < 0.0)[:, None]
    local = xp.where(neg, xp.conj(local), local)
    return float(xp.abs(local - pair.reshape(-1)[:, None]).max()) / scale


def pole_keldysh_pf_sparse(
    omega: NDArray,
    cluster: PoleCluster,
    source: NDArray,
    rows: NDArray,
    cols: NDArray,
) -> NDArray:
    r"""``G_PP`` in the PARTIAL-FRACTION representation the sectors use.

        Parameters
        ----------
        s_a, s_b : NDArray
            ``(Np, Np)`` source at the row pole and at the column pole, from
            :func:`source_at_poles`.

        Returns
        -------
        NDArray
            ``(n_omega, nnz)`` on the stored pattern.
    """
    w = xp.asarray(omega, dtype=xp.complex128)[:, None, None]
    za = cluster.z[None, :, None]
    zb = xp.conj(cluster.z)[None, None, :]
    gap = cluster.z[:, None] - xp.conj(cluster.z)[None, :]
    ss = xp.asarray(source, dtype=xp.complex128)
    ca, cb = ss / gap, -ss / gap          # equal and opposite: 1/w^2 by design
    f = ca[None] / (w - za) + cb[None] / (w - zb)          # (n_omega, Np, Np)

    lr = xp.take(cluster.u, xp.asarray(rows), axis=0)      # (nnz, Np)
    lc = xp.conj(xp.take(cluster.u, xp.asarray(cols), axis=0))
    return xp.einsum("ka,wab,kb->wk", lr, f, lc)
