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
        self, config, phonon_energies, block_sizes, dynamical_matrix=None,
        orbital_grid=None,
    ) -> None:
        self.sigma_phonon_phonon = SigmaPhononPhonon(
            config,
            phonon_energies,
            block_sizes=np.asarray(get_host(block_sizes)),
            dynamical_matrix=dynamical_matrix,
            orbital_grid=orbital_grid,
        )

    @profiler.profile(label="Interaction: Phonon-Phonon", level="default", comm=comm)
    def compute(self, scba: "SCBA") -> None:
        data = scba.data
        if (
            getattr(scba.config.phonon, "sse_cm_subtraction", False)
            and self.sigma_phonon_phonon._cm_channel is None
        ):
            # Build the CM-channel data once from the run's own inputs
            # (rank-local, deterministic; see quatrex.phonon.cm_channel).
            from quatrex.phonon.cm_channel import compute_cm_channel

            n_blocks = len(self.sigma_phonon_phonon.block_sizes)
            self.sigma_phonon_phonon.set_cm_channel(
                *compute_cm_channel(scba.config, n_blocks)
            )
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

        from quatrex.phonon.pole_bridge import (
            mixed_self_energy_blocked, modal_vertex_blocks,
            source_at_poles, ss_self_energy_sparse,
        )

        ssp = self.sigma_phonon_phonon
        # (1) The legs: remove the pole sector before the FFT ring sees them.
        ssp.set_pole_channel(state.g_pp_lesser, state.g_pp_greater)
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
            # Per-pole sources, NOT a single frozen value at the cluster
            # centre. The exact residue at z_alpha is S(z_alpha)/gap, so each
            # leg must carry the source where its own pole sits; one frozen
            # value is wrong as soon as a cluster holds more than one pole, and
            # badly wrong once the set is closed under z -> -z^* (the partner
            # is at -Omega while the frozen value was read at +Omega).
            #
            # This is what made G_PP -- built from the frequency-resolved
            # source -- and the pole leg put back by the sectors different
            # functions, which breaks the SPATIAL balance while leaving the
            # scalar P_in = P_out identity nearly intact.
            # One source per pole PAIR: shared by both residues, which is what
            # keeps G_PP decaying like 1/w^2 (see source_at_poles).
            sa = source_at_poles(s_l, freqs, cl)
            sb = source_at_poles(s_g, freqs, cl)
            kw = dict(rows=rows, cols=cols)
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
        # Mask the background leg EXACTLY as the ring masks its own legs
        # (sse_phonon_phonon: gl_in[conv_mask] = 0). The omega = 0 bin carries
        # the near-singular acoustic spectral peak -- the ring's own comment
        # says |G^>(0)| >> neighbours -- and the ring excludes it. Feeding the
        # mixed convolution an UNMASKED leg makes the two sectors integrate
        # different data, and injects that peak straight into Sigma.
        #
        # Measured: without this, Sigma^> is non-PSD by 0.15 at mid-band and
        # the violation is strictly LINEAR in the injected mixed term, i.e. no
        # cancellation at all -- the signature of a term that simply should
        # not be there.
        low = max(1e-6, float(getattr(ssp, "_low_freq_mask", 0.0) or 0.0))
        leg_mask = xp.abs(freqs) < low
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
