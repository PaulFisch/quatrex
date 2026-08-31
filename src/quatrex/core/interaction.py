# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Interaction registry for the SCBA loop.

Each Interaction instance owns its scattering self-energy
machinery and reads the appropriate Green's functions and writes into the
self-energy buffers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from qttools import xp
from qttools.comm import comm
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host

from quatrex.coulomb_screening import CoulombScreeningSolver, PCoulombScreening
from quatrex.electron import (
    SigmaCoulombScreening,
    SigmaFock,
    SigmaPhonon,
)
from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

if TYPE_CHECKING:
    from quatrex.core.scba import SCBA

profiler = Profiler()


class Interaction(ABC):
    """Abstract scattering interaction in an SCBA iteration.
    """

    #: Name of the interaction (used for profiling labels).
    name: str = "interaction"

    @abstractmethod
    def compute(self, scba: "SCBA") -> None:
        """Apply this interaction to the SCBA state.
        """
        ...


class CoulombScreeningInteraction(Interaction):
    """GW screened Coulomb interaction for the electron subsystem.
    """

    name = "coulomb_screening"

    def __init__(
        self,
        config,
        electron_energies,
        coulomb_screening_energies,
        coulomb_matrix,
        sparsity_pattern,
    ) -> None:
        if coulomb_matrix.distribution_state != "nnz":
            coulomb_matrix.dtranspose()

        self.sigma_fock = SigmaFock(
            config,
            coulomb_matrix,
            electron_energies,
        )
        if coulomb_matrix.distribution_state == "nnz":
            coulomb_matrix.dtranspose()

        self.p_coulomb_screening = PCoulombScreening(
            config,
            coulomb_screening_energies,
        )
        self.coulomb_screening_solver = CoulombScreeningSolver(
            config,
            coulomb_matrix,
            coulomb_screening_energies,
            sparsity_pattern=sparsity_pattern,
        )
        self.sigma_coulomb_screening = SigmaCoulombScreening(
            config,
            electron_energies,
        )

    @profiler.profile(label="Interaction: Coulomb screening", level="default", comm=comm)
    def compute(self, scba: "SCBA") -> None:
        data = scba.data

        data.p_greater.allocate_data()
        data.p_lesser.allocate_data()
        data.p_retarded_hermitian.allocate_data()

        self.p_coulomb_screening.compute(
            data.g_lesser,
            data.g_greater,
            out=(data.p_lesser, data.p_greater, data.p_retarded_hermitian),
        )

        data.w_greater.allocate_data()
        data.w_lesser.allocate_data()

        self.coulomb_screening_solver.solve(
            data.p_lesser,
            data.p_greater,
            data.p_retarded_hermitian,
            out=(data.w_lesser, data.w_greater),
        )

        scba._compute_coulomb_screening_observables()

        data.p_lesser.free_data()
        data.p_greater.free_data()
        data.p_retarded_hermitian.free_data()

        self.sigma_fock.compute(
            data.g_lesser,
            out=(data.sigma_retarded_hermitian,),
        )

        self.sigma_coulomb_screening.compute(
            data.g_lesser,
            data.g_greater,
            data.w_lesser,
            data.w_greater,
            out=(
                data.sigma_lesser,
                data.sigma_greater,
                data.sigma_retarded_hermitian,
            ),
        )

        data.w_greater.free_data()
        data.w_lesser.free_data()


class PhononPhononInteraction(Interaction):
    """3-phonon scattering self-energy (cubic-anharmonic NEGF model).
    """

    name = "phonon_phonon"

    def __init__(
        self, config, phonon_energies, block_sizes,
    ) -> None:
        self.sigma_phonon_phonon = SigmaPhononPhonon(
            config,
            phonon_energies,
            block_sizes=np.asarray(get_host(block_sizes)),
        )

    @profiler.profile(label="Interaction: Phonon-Phonon", level="default", comm=comm)
    def compute(self, scba: "SCBA") -> None:
        data = scba.data
        self._inject_pole_sector(scba)
        self.sigma_phonon_phonon.compute(
            data.g_lesser,
            data.g_greater,
            out=(
                data.sigma_lesser,
                data.sigma_greater,
                data.sigma_retarded_hermitian,
            ),
        )

    def _inject_pole_sector(self, scba: "SCBA") -> None:
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

        ssp = self.sigma_phonon_phonon
        # (1) The legs: remove the pole sector before the FFT ring sees them.
        ssp.set_pole_channel(state.g_pp_lesser, state.g_pp_greater)
        # Independent of `leg`: the ring can be corrected whether or not the
        # legs also carry a cell-average correction, because this ADDS the
        # covariance the ring omits rather than replacing anything it computed.
        if getattr(ps, "bubble_correction", "none") == "local_covariance":
            _bubble_covariance_correction(scba, state, ssp)
        if getattr(ps, "leg", "congruence") == "congruence_analytic":
            _pole_analytic_sectors(scba, state, ssp)
            return
        if getattr(ps, "leg", "congruence") == "congruence":
            # The pole channel here is the point-minus-cell-average
            # correction, so the ring convolves the cell average of the
            # congruence reconstruction -- what a dw-weighted sum wants, and
            # what the raw grid sample gets wrong by order one for an
            # under-resolved line. Being an average of PSD matrices, the leg is
            # PSD however bad the pole model is, which is the difference from
            # the superseded route.
            #
            # It does NOT resolve the pole inside the convolution: the output
            # resolution is still the grid's. Carrying SR/RS analytically needs
            # a vertex contraction against per-pole pattern-valued
            # coefficients, which is not built.
            return
        if ps.sectors == "rr":
            # Deliberately incomplete: SS/SR/RS are not added back, so this
            # DROPS real three-phonon processes. It is a staging setting whose
            # purpose is to measure how large they are.
            return

        # (2) The analytic pole-pole channel, contracted against the vertex on
        # the stored pattern.
        rows, cols = data_rows_cols(scba)
        freqs = xp.asarray(solver.local_frequencies, dtype=float)
        shape = scba.data.sigma_lesser.data.shape
        acc_l = acc_g = acc_r = None
        # state.legs, not state.clusters: the sources are indexed alongside
        # the bosonically closed set the solver built them from.
        for cl, s_l, s_g in zip(state.legs, state.source_lesser,
                                state.source_greater):
            vl = modal_vertex_blocks(ssp.phi_blocks, ssp.block_sizes, cl.u,
                                     conjugate=False)
            vr = modal_vertex_blocks(ssp.phi_blocks, ssp.block_sizes, cl.u,
                                     conjugate=True)
            # Per-pole sources, NOT one frozen value at the cluster centre:
            # each leg must carry the source where its own pole sits. A frozen
            # value is wrong as soon as a cluster holds more than one pole, and
            # badly wrong once the set is closed under z -> -z^*, where the
            # partner sits at -Omega. It is what made G_PP and the pole leg the
            # sectors put back different functions, which breaks the SPATIAL
            # balance while leaving the scalar P_in = P_out nearly intact.
            # One source per pole PAIR: shared by both residues, which is what
            # keeps G_PP decaying like 1/w^2 (see source_at_poles).
            sa = source_at_poles(s_l, freqs, cl)
            sb = source_at_poles(s_g, freqs, cl)
            # Cell width, so the analytic sector lands in the SAME
            # representation the grid solver integrates: piecewise constant
            # with weight dw, not a point sample.
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

        # (3) Inject only the KRAMERS-KRONIG HALF of Sigma^R. The driver
        # (core/scba.py) already adds 0.5*(sigma^< - sigma^>) globally for the
        # total stored self-energy, which now includes this analytic term, so
        # supplying the full Sigma_SS^R here would double-count the half term
        # and break causality.
        kk_half = acc_r - 0.5 * (acc_g - acc_l)
        ssp.set_pole_self_energy(acc_l.reshape(shape), acc_g.reshape(shape),
                                 kk_half.reshape(shape))
        if ps.sectors != "rr_ss_sr":
            return

        # (4) The mixed pole-background sectors. They must be evaluated as a
        # symmetric pair with the same quadrature, or the Phi-derivable energy
        # balance that makes the decomposition conserving is lost.
        g_l = scba.data.g_lesser.data.reshape(freqs.shape[0], -1)
        g_g = scba.data.g_greater.data.reshape(freqs.shape[0], -1)
        reg_l = g_l - state.g_pp_lesser.reshape(g_l.shape)
        reg_g = g_g - state.g_pp_greater.reshape(g_g.shape)
        # Mask the background leg EXACTLY as the ring masks its own legs. The
        # omega = 0 bin carries the near-singular acoustic peak and the ring
        # excludes it; feeding the mixed convolution an unmasked leg makes the
        # two sectors integrate different data and injects that peak into
        # Sigma. Measured: without this, Sigma^> is non-PSD by 0.15 at mid-band,
        # strictly linear in the injected term.
        leg_mask = xp.abs(freqs) < 1e-6
        if bool(leg_mask.any()):
            reg_l = reg_l.copy(); reg_l[leg_mask] = 0.0
            reg_g = reg_g.copy(); reg_g[leg_mask] = 0.0
        mx_l = mx_g = None
        # state.legs, not state.clusters: the sources are indexed alongside
        # the bosonically closed set the solver built them from.
        for cl, s_l, s_g in zip(state.legs, state.source_lesser,
                                state.source_greater):
            common = dict(freqs=freqs, phi_blocks=ssp.phi_blocks,
                          block_sizes=ssp.block_sizes, rows=rows, cols=cols)
            # Blocked, not pattern-level: the pattern contraction is
            # O(nnz_out * nnz_in) and refuses above 4096 entries, which no
            # real device is under. Same object, pinned to 1e-12 against the
            # pattern form in test_pole_blocked.py.
            # Each component is extended onto the negative axis from its
            # Keldysh PARTNER: G^<(q,-w) = G^>(-q,w)^T. Passing the same
            # component twice would restore the conjugate mirror, which is
            # correct for Delta but wrong for a lesser/greater leg by 244 %.
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


def data_rows_cols(scba: "SCBA"):
    """Row/column indices of the shared sparsity pattern."""
    return scba.data.sigma_lesser.rows, scba.data.sigma_lesser.cols


class PseudoScatteringPhononInteraction(Interaction):
    """Pseudo-scattering phonon self-energy
    """

    name = "phonon_pseudo"

    def __init__(self, config, electron_energies) -> None:
        self.sigma_phonon = SigmaPhonon(config, electron_energies)

    @profiler.profile(label="Interaction: Phonon (pseudo)", level="default", comm=comm)
    def compute(self, scba: "SCBA") -> None:
        data = scba.data
        self.sigma_phonon.compute(
            data.g_lesser,
            data.g_greater,
            out=(
                data.sigma_lesser,
                data.sigma_greater,
                data.sigma_retarded_hermitian,
            ),
        )


def build_interactions(scba: "SCBA") -> list[Interaction]:
    """Construct the list of enabled :class:`Interaction` instances.
    """
    config = scba.config
    interactions: list[Interaction] = []

    if config.scba.coulomb_screening:
        interactions.append(scba._coulomb_screening_interaction)

    if config.scba.phonon:
        if config.phonon.model == "negf":
            interactions.append(scba._phonon_phonon_interaction)
        elif config.phonon.model == "pseudo-scattering":
            interactions.append(scba._pseudo_scattering_phonon_interaction)

    if config.scba.photon:
        raise NotImplementedError(
            "Photon interaction not implemented in the registry yet."
        )

    return interactions



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
    # Mask the background leg EXACTLY as the ring masks its own legs
    # (sse_phonon_phonon: gl_in[conv_mask] = 0), for the same reason as the
    # rr_ss_sr route above: the omega = 0 bin carries the near-singular
    # acoustic spectral peak and the ring excludes it, so an unmasked leg
    # makes the two sectors integrate different data and injects that peak
    # straight into Sigma. Measured there: Sigma^> non-PSD by 0.15 at
    # mid-band, strictly LINEAR in the injected mixed term.
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

    # Inject only the KRAMERS-KRONIG HALF of the pole-pole Sigma^R:
    # core/scba.py already adds 0.5*(sigma^< - sigma^>) globally over the
    # stored total. acc_l/acc_g are SS alone here, which is what acc_r is the
    # causal partner OF -- the mixed sectors get theirs from the transform.
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
        # Only the cells that hold a promoted pole carry subcell structure
        # worth correcting; the rest are already smooth on the grid.
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
        # The negative half is the PARTNER component folded:
        # G^<(-w) = G^>(w)^T, so a leg sum_p R_p/(u - zeta_p) placed at -w_k
        # becomes sum_p (-R_p^T)/(u + zeta_p) -- poles at -zeta, residues
        # negated and TRANSPOSED. For a rank-one residue the transpose is just
        # the two factors swapped, (p (x) q)^T = q (x) p, so no pattern
        # permutation is needed and the sign rides on one of them.
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
