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
        self.sigma_phonon_phonon.compute(
            data.g_lesser,
            data.g_greater,
            out=(
                data.sigma_lesser,
                data.sigma_greater,
                data.sigma_retarded_hermitian,
            ),
        )


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
