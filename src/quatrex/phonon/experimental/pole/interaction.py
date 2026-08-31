# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Analytic-pole injection for the three-phonon interaction."""

import numpy as np
from qttools import xp
from qttools.comm import comm
from qttools.utils.gpu_utils import get_host


def data_rows_cols(scba: "SCBA"):
    """Row/column indices of the shared sparsity pattern."""
    return scba.data.sigma_lesser.rows, scba.data.sigma_lesser.cols


def inject_pole_sector(ssp, scba) -> None:
    """Hand the solver's pole clusters to the bubble.

    The solver has already found the poles, projected the Keldysh source and
    reduced ``G_PP`` onto the sparsity pattern. What is left is the piece
    that needs the VERTEX, which lives here: contract the analytic pole-pole
    self-energy and inject both halves.

    The split is exact -- ``G_S + G_R`` is the untouched ``G`` -- so the
    bubble sees a different representation of the same object, not a
    different object.
    """
    ps = getattr(scba.config.phonon, "pole_sector", None)
    if ps is None or not ps.enabled:
        return
    solver = getattr(scba, "phonon_solver", None)
    state = getattr(solver, "pole_state", None)
    if state is None or not state.clusters:
        return

    from quatrex.phonon.experimental.pole.pole_bridge import (
        mixed_self_energy_blocked, modal_vertex_blocks,
        source_at_poles, ss_self_energy_sparse,
    )

    # (1) The legs: remove the pole sector before the FFT ring sees them.
    ssp.set_pole_channel(state.g_pp_lesser, state.g_pp_greater)
    if getattr(ps, "bubble_correction", "none") == "local_covariance":
        _bubble_covariance_correction(scba, state, ssp)
    if getattr(ps, "leg", "congruence") == "congruence_analytic":
        _pole_analytic_sectors(scba, state, ssp)
        return
    if getattr(ps, "leg", "congruence") == "congruence":
        return
    if ps.sectors == "rr":
        return

    rows, cols = data_rows_cols(scba)
    freqs = xp.asarray(solver.local_frequencies, dtype=float)
    shape = scba.data.sigma_lesser.data.shape
    acc_l = acc_g = acc_r = None
    for cl, s_l, s_g in zip(state.legs, state.source_lesser,
                            state.source_greater):
        vl = modal_vertex_blocks(ssp.phi_blocks, ssp.block_sizes, cl.u,
                                 conjugate=False)
        vr = modal_vertex_blocks(ssp.phi_blocks, ssp.block_sizes, cl.u,
                                 conjugate=True)
        sa = source_at_poles(s_l, freqs, cl)
        sb = source_at_poles(s_g, freqs, cl)
        _h = float(xp.real(freqs[1] - freqs[0])) if freqs.shape[0] > 1 else 0.0
        kw = dict(rows=rows, cols=cols,
                  cell=_h if getattr(ps, "cell_average", True) else None)
        ss_l = ss_self_energy_sparse(freqs, cl, sa, sa, vl, vr, **kw)
        ss_g = ss_self_energy_sparse(freqs, cl, sb, sb, vl, vr, **kw)
        # The CAUSAL part of each, from the two-retarded pole pairings.
        rr_l = ss_self_energy_sparse(freqs, cl, sa, sa, vl, vr,
                                     retarded_only=True, **kw)
        rr_g = ss_self_energy_sparse(freqs, cl, sb, sb, vl, vr,
                                     retarded_only=True, **kw)
        acc_l = ss_l if acc_l is None else acc_l + ss_l
        acc_g = ss_g if acc_g is None else acc_g + ss_g
        rr = rr_g - rr_l
        acc_r = rr if acc_r is None else acc_r + rr

    kk_half = acc_r - 0.5 * (acc_g - acc_l)
    ssp.set_pole_self_energy(acc_l.reshape(shape), acc_g.reshape(shape),
                             kk_half.reshape(shape))
    if ps.sectors != "rr_ss_sr":
        return

    g_l = scba.data.g_lesser.data.reshape(freqs.shape[0], -1)
    g_g = scba.data.g_greater.data.reshape(freqs.shape[0], -1)
    reg_l = g_l - state.g_pp_lesser.reshape(g_l.shape)
    reg_g = g_g - state.g_pp_greater.reshape(g_g.shape)
    leg_mask = xp.abs(freqs) < 1e-6
    if bool(leg_mask.any()):
        reg_l = reg_l.copy(); reg_l[leg_mask] = 0.0
        reg_g = reg_g.copy(); reg_g[leg_mask] = 0.0
    mx_l = mx_g = None
    for cl, s_l, s_g in zip(state.legs, state.source_lesser,
                            state.source_greater):
        common = dict(freqs=freqs, phi_blocks=ssp.phi_blocks,
                      block_sizes=ssp.block_sizes, rows=rows, cols=cols)
        a = mixed_self_energy_blocked(
            freqs, cl, source_at_poles(s_l, freqs, cl),
            reg_l, reg_g, **common)
        b = mixed_self_energy_blocked(
            freqs, cl, source_at_poles(s_g, freqs, cl),
            reg_g, reg_l, **common)
        mx_l = a if mx_l is None else mx_l + a
        mx_g = b if mx_g is None else mx_g + b
    scale = float(getattr(ps, "mixed_scale", 1.0))
    ssp.set_pole_mixed((scale * mx_l).reshape(shape),
                       (scale * mx_g).reshape(shape))

def _pole_analytic_sectors(scba, state, ssp) -> None:
    """SS + SR + RS for the partial-fraction leg, all analytic.

    The leg the solver removed from the ring is
    ``sum_p p_p q_p^T/(w - zeta_p)``; this restores its three bubble sectors,
    so the decomposition ``B(G,G) = SS + SR + RS + RR`` closes. Removing more
    than is put back is what the sector-sum gate exists to catch, and it is
    the failure this whole construction started from.

    The two halves take DIFFERENT routes into the bubble, and that is the
    point of the split below.

    ``SS`` -- the pole-pole convolution -- carries its own closed-form causal
    partner (the two-retarded pairing, whose combined pole ``zeta_p + zeta_q``
    is again in the lower half plane), so it goes through
    :meth:`set_pole_self_energy` and never touches the discrete Hilbert
    transform.

    ``SR + RS`` has NO closed-form causal partner: one leg is the numerical
    background. It must therefore join the raw bubble output BEFORE
    ``delta = sigma^> - sigma^<`` is formed, so the existing Kramers-Kronig
    transform covers it -- which is exactly what :meth:`set_pole_mixed` does,
    and what the ``rr_ss_sr`` route has always done. Routing it through
    ``set_pole_self_energy`` instead lands it after ``delta`` is built
    (``sse_phonon_phonon.py``), and then ``Sigma^R`` is missing the entire
    dispersive part of the mixed sector while ``Sigma^{<,>}`` has it. That is
    a fluctuation-dissipation break by construction, and it is what
    ``lead balance = 2.0000`` measured on job 4398805.
    """
    from quatrex.phonon.experimental.pole.pole_bridge import modal_vertex_blocks
    from quatrex.phonon.experimental.pole.pole_congruence import (
        pf_mixed_self_energy, pf_self_energy,
    )

    solver = scba.phonon_solver
    ps = scba.config.phonon.pole_sector
    rows, cols = data_rows_cols(scba)
    freqs = xp.asarray(solver.local_frequencies, dtype=float)
    shape = scba.data.sigma_lesser.data.shape
    h = float(xp.real(freqs[1] - freqs[0])) if freqs.shape[0] > 1 else 0.0
    cell = h if getattr(ps, "cell_average", True) else None
    g_l = scba.data.g_lesser.data.reshape(freqs.shape[0], -1)
    g_g = scba.data.g_greater.data.reshape(freqs.shape[0], -1)
    reg_l = g_l - state.g_pp_lesser.reshape(g_l.shape)
    reg_g = g_g - state.g_pp_greater.reshape(g_g.shape)
    leg_mask = xp.abs(freqs) < 1e-6
    if bool(leg_mask.any()):
        reg_l = reg_l.copy(); reg_l[leg_mask] = 0.0
        reg_g = reg_g.copy(); reg_g[leg_mask] = 0.0

    acc_l = acc_g = acc_r = None
    mx_l = mx_g = None
    for pf_l, pf_g in zip(state.pf_lesser, state.pf_greater):
        for pf, reg, partner, slot in ((pf_l, reg_l, reg_g, "l"),
                                       (pf_g, reg_g, reg_l, "g")):
            zeta, p_row, q_col = pf
            vl = modal_vertex_blocks(ssp.phi_blocks, ssp.block_sizes, p_row,
                                     conjugate=False)
            vr = modal_vertex_blocks(ssp.phi_blocks, ssp.block_sizes, q_col,
                                     conjugate=False)
            ss = pf_self_energy(freqs, zeta, vl, vr, rows, cols, cell=cell)
            mx = pf_mixed_self_energy(
                freqs, zeta, p_row, q_col, reg, partner, freqs,
                ssp.phi_blocks, ssp.block_sizes, rows, cols)
            # The CAUSAL part of the POLE-POLE sector, in closed form.
            rr = pf_self_energy(freqs, zeta, vl, vr, rows, cols,
                                retarded_only=True, cell=cell)
            if slot == "l":
                acc_l = ss if acc_l is None else acc_l + ss
                mx_l = mx if mx_l is None else mx_l + mx
                acc_r = -rr if acc_r is None else acc_r - rr
            else:
                acc_g = ss if acc_g is None else acc_g + ss
                mx_g = mx if mx_g is None else mx_g + mx
                acc_r = rr if acc_r is None else acc_r + rr

    kk_half = acc_r - 0.5 * (acc_g - acc_l)
    ssp.set_pole_self_energy(acc_l.reshape(shape), acc_g.reshape(shape),
                             kk_half.reshape(shape))
    scale = float(getattr(ps, "mixed_scale", 1.0))
    ssp.set_pole_mixed((scale * mx_l).reshape(shape),
                       (scale * mx_g).reshape(shape))


def _bubble_covariance_correction(scba, state, ssp) -> None:
    """What the cell-averaged ring leaves behind, added back on active pairs.

    The ring works from cell means; the exact cell-pair integral is that plus
    the covariance of the two subcell fluctuations. This adds the second term
    and touches nothing else -- no leg is modified, nothing is subtracted, and
    an empty active set gives exactly zero. See
    :mod:`quatrex.phonon.experimental.pole.pole_covariance`.

    The active set is built on the EXTENDED axis: each promoted cell enters
    once at ``+omega_k`` from this Keldysh component and once at ``-omega_k``
    from its PARTNER, transposed, which is the same fold
    ``Sigma^<(-w) = Sigma^>(w)^T`` the ring applies to its own legs. Without
    the negative entries the difference-frequency channel is silently absent.
    """
    from quatrex.phonon.experimental.pole.pole_congruence import partial_fraction_legs_percell
    from quatrex.phonon.experimental.pole.pole_covariance import cell_variance, spectrum_correction

    ps = scba.config.phonon.pole_sector
    solver = scba.phonon_solver
    rows, cols = data_rows_cols(scba)
    freqs = np.asarray(get_host(solver.local_frequencies), dtype=float)
    if freqs.size < 2:
        return
    h = float(freqs[1] - freqs[0])
    shape = scba.data.sigma_lesser.data.shape
    floor = float(getattr(ps, "covariance_sigma_min", 0.0) or 0.0)

    built = {"l": [], "g": []}
    for cl, co_l, co_g in zip(state.legs, state.c_lesser, state.c_greater):
        cells = np.unique([int(np.argmin(np.abs(freqs - float(np.real(z)))))
                           for z in np.asarray(get_host(cl.z))
                           if float(np.real(z)) >= 0.0])
        if cells.size == 0:
            continue
        for tag, co in (("l", co_l), ("g", co_g)):
            sub = tuple(np.asarray(get_host(a))[cells] for a in co)
            zeta, p_row, q_col = partial_fraction_legs_percell(cl, sub)
            for j, k in enumerate(cells):
                built[tag].append((float(freqs[k]), np.asarray(get_host(zeta)),
                                   np.asarray(get_host(p_row))[j],
                                   np.asarray(get_host(q_col))[j]))

    if not built["l"] and not built["g"]:
        return

    def _screen(entries):
        if not entries:
            return []
        var = [cell_variance(np.einsum("ip,jp->pij", p, q), z, c, h)
               for c, z, p, q in entries]
        top = max(var) if var else 0.0
        return [e for e, v in zip(entries, var) if v >= floor * top]

    out = {}
    for tag, partner in (("l", "g"), ("g", "l")):
        pos = _screen(built[tag])
        neg = [(-c, -z, -q, p) for c, z, p, q in _screen(built[partner])]
        corr, rep = spectrum_correction(freqs, pos + neg, ssp.phi_blocks,
                                        ssp.block_sizes, rows, cols, h)
        out[tag] = np.asarray(get_host(corr))
        if comm.rank == 0 and tag == "l":
            print(f"  bubble correction: {len(pos)} active cells (+{len(neg)} "
                  f"mirrored), {rep['applied']} pairs applied, "
                  f"{rep['out_of_range']} off-grid", flush=True)
    ssp.set_bubble_correction(xp.asarray(out["l"]).reshape(shape),
                              xp.asarray(out["g"]).reshape(shape))
