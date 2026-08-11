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
    "source_at_poles",
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
    g_partner: NDArray,
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
    from quatrex.phonon.pole_audit import transpose_index
    from quatrex.phonon.pole_bubble import _split_leg
    from quatrex.phonon.pole_mixed import bosonic_extend, mixed_convolution_batched

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

    # See _mixed_one_sector_blocked: the negative half comes from the PARTNER
    # Keldysh component, transposed.
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


# ---------------------------------------------------------------------------
# Block-structured mixed contraction
#
# ``_mixed_one_sector`` contracts at the pattern level, which is O(nnz_out *
# nnz_in) and hopeless at device scale (nnz ~ 1e5 gives 1e10 entries in a
# single intermediate). The contraction is really a matrix triple product,
#
#     Sigma[I, J] = sum_{a, b} BL[I, a] M[a, b] BR[J, b]^T,
#
# and every factor is block-banded: M lives on G's block-tridiagonal pattern,
# and BL/BR inherit the cubic vertex's |I - a| <= 1. So each output block needs
# only a handful of b x b GEMMs, and the dense (n_dof, n_dof) vertex -- 1.5 GB
# per pole at production size -- is never formed either.
# ---------------------------------------------------------------------------

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
    from quatrex.phonon.pole_audit import transpose_index
    from quatrex.phonon.pole_bubble import _split_leg
    from quatrex.phonon.pole_mixed import bosonic_extend, mixed_convolution_batched

    if prefactor is None:
        prefactor = analytic_prefactor()

    # The convolution runs over the WHOLE frequency axis; the solver only holds
    # G for omega >= 0. The negative half is fixed by the bosonic relation
    # G^<(q,-w) = G^>(-q,w)^T -- it comes from the PARTNER component,
    # transposed, NOT from conjugating this one. Extended once here, not per
    # pole pair.
    g_reg, freqs = bosonic_extend(
        g_reg, g_partner, freqs, transpose_index=transpose_index(rows, cols))

    poles, coeffs = _split_leg(cluster, source)
    npp = cluster.n_poles
    n_omega = int(omega.shape[0])

    # The pole set is shared by every sector, so both the convolved M and its
    # block view are built once and reused across output blocks.
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
    source: NDArray, freqs: NDArray, cluster: PoleCluster
) -> NDArray:
    r"""The projected source for each pole PAIR, at that pair's own frequency.

    One value per ``(alpha, beta)``, shared by both residues of the pair. That
    sharing is not a convenience -- it is what keeps ``G_PP`` decaying like the
    object it models.

    Writing the leg as ``c_a/(w - z_a) + c_b/(w - conj(z_b))`` with
    ``c_a = S_a/gap`` and ``c_b = -S_b/gap``, the large-``w`` behaviour is
    ``(c_a + c_b)/w``. The congruence form ``S/((w-z_a)(w-conj(z_b)))`` decays
    as ``1/w^2``, so the two agree only if ``c_a + c_b = 0``, i.e. only if the
    SAME source serves both residues.

    Using a different source at each pole leaves ``c_a + c_b != 0`` and gives
    ``G_PP`` a spurious ``1/w`` tail. Measured against the congruence: 17x too
    large at ``w = 1e2``, 579x at ``3e3``, **18364x** at ``1e5``. That is what
    made ``rr_ss`` regress from 5.4e-08 to 4.9e-05 when per-pole sources were
    introduced.

    The pair frequency is the midpoint of the two poles' real parts. For the
    dominant ``alpha == beta`` terms that is exactly ``Re z_alpha``, so the
    per-pole accuracy is kept where it matters; only the cross terms, whose
    coefficients are suppressed by a large ``gap``, are averaged.

    Negative-frequency partners (from the bosonic closure) are served by
    ``S(-w) = S(w)^*``.

    Returns
    -------
    NDArray
        ``(Np, Np)``.

    """
    w = np.asarray(_host(freqs), dtype=float)
    z = np.asarray(_host(cluster.z))
    src = xp.asarray(source, dtype=xp.complex128)

    centre = 0.5 * (z.real[:, None] + np.conj(z).real[None, :])
    idx = np.abs(w[None, None, :] - np.abs(centre)[:, :, None]).argmin(axis=-1)
    npp = int(z.size)
    out = xp.stack([
        xp.stack([src[int(idx[a, b]), a, b] for b in range(npp)])
        for a in range(npp)
    ])
    neg = xp.asarray(centre < 0.0)
    return xp.where(neg, xp.conj(out), out)


def pole_keldysh_pf_sparse(
    omega: NDArray,
    cluster: PoleCluster,
    source: NDArray,
    rows: NDArray,
    cols: NDArray,
) -> NDArray:
    r"""``G_PP`` in the PARTIAL-FRACTION representation the sectors use.

    :func:`pole_keldysh_sparse` builds ``U D^R S(w) D^A U^dag`` from the
    frequency-resolved source. The analytic sectors cannot represent that: they
    split every leg into simple poles, which carries only a rational source. So
    the two are different functions whenever ``S`` varies with frequency --
    measured 7e-3 apart on a source with a 2%/THz slope, against 7e-16 for a
    constant one.

    That difference is not a small inaccuracy, it is a broken decomposition.
    ``G_reg = G - G_PP`` is exact for ANY ``G_PP``, so the sector sum
    ``B(G,G) = SS + SR + RS + RR`` holds only if the leg SUBTRACTED from the
    ring and the leg the sectors PUT BACK are literally the same object. Using
    the resolved form on one side and partial fractions on the other violates
    doc Sec. 37's "same reconstructed G on both legs", and it breaks the
    balance spatially while leaving the scalar ``P_in = P_out`` nearly intact.

    This builds the leg from the same coefficients
    :func:`~quatrex.phonon.pole_bubble.leg_partial_fractions` produces, so the
    two agree to roundoff by construction.

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
