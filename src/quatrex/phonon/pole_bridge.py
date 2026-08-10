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

from quatrex.phonon.pole_bubble import modal_convolution
from quatrex.phonon.pole_keldysh import PoleCluster, modal_denominator
from quatrex.phonon.units import HBAR_SI

__all__ = [
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

    The lead self-energies live only on the first and last diagonal blocks and
    are held as dense corners rather than on the pattern, so they are projected
    separately. Omitting them would drop the injection that drives the device.

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
    phi_blocks: dict, block_sizes: NDArray, u: NDArray, conjugate: bool
) -> NDArray:
    r"""Project the block-sparse cubic vertex onto the modal basis.

    ``Vbar[mu, alpha, beta] = sum_{ab} Phi[mu, a, b] u[a, alpha] u[b, beta]``
    (doc Eq. 116), accumulated block by block so the dense ``(n_dof,)^3`` tensor
    is never formed.

    ``conjugate`` selects the right-hand factor, whose modal vectors are
    conjugated -- that is what makes the contraction a congruence and carries
    the positivity statement (``bubble_positivity.md`` Thm 1).
    """
    sizes = np.asarray(_host(block_sizes), dtype=int)
    off = np.concatenate(([0], np.cumsum(sizes)))
    n_dof, npp = int(off[-1]), int(u.shape[1])
    uu = xp.conj(u) if conjugate else u
    out = xp.zeros((n_dof, npp, npp), dtype=xp.complex128)
    for (i, k1, k2), blk in phi_blocks.items():
        b = xp.asarray(blk, dtype=xp.complex128)
        u1 = uu[off[k1]:off[k1 + 1]]
        u2 = uu[off[k2]:off[k2 + 1]]
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
) -> NDArray:
    r"""Analytic pole-pole self-energy, evaluated on the stored pattern.

    .. math::
        \Sigma_{SS}(\omega)_{\mu\mu'} = \frac{i\hbar}{2}
          \sum_{\alpha\beta\gamma\delta} \bar\Phi_{\mu,\alpha\beta}
          \bar\Phi^*_{\mu',\gamma\delta}\,
          C_{\alpha\delta\beta\gamma}(\omega)

    with ``C`` the closed-form modal convolution. Only pattern entries
    ``(rows[k], cols[k])`` are produced, so the ``|I-J| <= 1`` output band the
    solver can actually consume is respected automatically.

    Returns
    -------
    NDArray
        ``(n_omega, nnz)``.

    """
    if prefactor is None:
        prefactor = analytic_prefactor()
    c = modal_convolution(omega, cluster, source_a, source_b,
                          retarded_only=retarded_only)
    left = xp.take(vl, xp.asarray(rows), axis=0)           # (nnz, Np, Np)
    right = xp.take(vr, xp.asarray(cols), axis=0)          # (nnz, Np, Np)
    return prefactor * xp.einsum("kAB,kGD,wADBG->wk", left, right, c)


def mixed_vertex_blocks(
    phi_blocks: dict, block_sizes: NDArray, u: NDArray, *, leg: int,
    conjugate: bool,
) -> NDArray:
    r"""Vertex with ONE leg projected onto the modal basis (doc Eq. 119).

    ``leg`` and ``conjugate`` are INDEPENDENT, because the two mixed sectors
    reduce different legs. For the ring
    ``Phi_L[a,c,e] A[c,b] B[e,d] Phi_R[J,d,b]``:

    ===========  ==================================  ==================================
    sector       left factor                         right factor
    ===========  ==================================  ==================================
    ``Sigma_SR`` ``B_a[a,e] = sum_c Phi_L[a,c,e]u``  ``B*_d[J,d] = sum_b Phi_R[J,d,b]u*``
                 (``leg=0``)                          (``leg=1``, conjugate)
    ``Sigma_RS`` ``C_b[a,c] = sum_e Phi_L[a,c,e]u``  ``D_g[J,b] = sum_d Phi_R[J,d,b]u*``
                 (``leg=1``)                          (``leg=0``, conjugate)
    ===========  ==================================  ==================================

    Tying ``leg`` to ``conjugate`` would silently compute ``Sigma_SR`` twice.

    Returns ``(n_dof, n_dof, Np)``: the free device index, then the surviving
    vertex leg, then the mode.
    """
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
    prefactor: complex | None = None,
    max_nnz: int = 4096,
) -> NDArray:
    r"""ONE mixed sector, given its pair of single-leg vertices.

    Use :func:`mixed_self_energy_sparse`, which evaluates ``Sigma_SR`` and
    ``Sigma_RS`` as the symmetric pair the conserving decomposition requires.

    .. math::
        \Sigma_{SR}(\omega)_{aJ} = \sum_{\alpha\delta}\sum_{ed}
            B_\alpha[a,e]\, M_{\alpha\delta}[e,d](\omega)\, B^*_\delta[J,d]

    with ``M`` the pole convolved against the regular Green's function. The
    convolution is done ONCE per pole pair as a matmul over the grid (the narrow
    denominator is integrated over each cell, never sampled); what remains is a
    frequency-LOCAL bilinear contraction, so there is no second convolution.

    ``Sigma_RS`` is the same object with the legs exchanged, and is obtained by
    transposing rather than recomputing.

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
    from quatrex.phonon.pole_bubble import leg_partial_fractions
    from quatrex.phonon.pole_mixed import mixed_convolution_batched

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

    poles, coeffs = leg_partial_fractions(cluster, source)
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
    freqs: NDArray,
    phi_blocks: dict,
    block_sizes: NDArray,
    rows: NDArray,
    cols: NDArray,
    prefactor: complex | None = None,
    max_nnz: int = 4096,
) -> NDArray:
    r"""``Sigma_SR + Sigma_RS`` -- the mixed sectors, as a symmetric pair.

    These are the pole-background three-phonon processes. They must be evaluated
    together and with the same quadrature: dropping either, or approximating one
    differently from the other, breaks the Phi-derivable energy balance that
    makes the decomposition conserving (doc Sec. 37).

    Both reduce to the same object -- the pole convolved against the regular
    Green's function, ``M = int dw'/2pi F(w') G_R(w-w')`` -- because convolution
    commutes; they differ only in which vertex legs carry the modal index. So
    ``M`` is formed once per pole pair and reused.
    """
    m_vertex = mixed_vertex_blocks
    kw = dict(freqs=freqs, rows=rows, cols=cols, prefactor=prefactor,
              max_nnz=max_nnz)
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
