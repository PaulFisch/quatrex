# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Interaction registry for the SCBA loop.

Each ``Interaction`` instance owns its scattering self-energy
machinery (and any auxiliary Dyson solver, e.g.\\ the screened
Coulomb solve for GW) and, when called from inside the SCBA loop,
reads the appropriate Green's functions and writes into the
self-energy buffers held by :class:`quatrex.core.scba.SCBAData`.

The registry exists so that the SCBA driver does not have to branch
on which interactions are enabled. Adding a new interaction
(e.g.\\ an electron-phonon SSE coupling the electron and phonon
subsystems) is a matter of subclassing :class:`Interaction` and
appending an instance to ``SCBA.interactions`` -- no SCBA-side
changes are needed.

A future :class:`ElectronPhononInteraction` that couples the
electron and phonon subsystems in a single SCBA iteration plugs in
through the same interface: its ``compute`` reads from both
subsystems' Green's functions and writes the e-ph contribution into
the appropriate ``sigma_*`` buffer on each side.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from qttools import xp
from qttools.comm import comm
from qttools.profiling import Profiler

from quatrex.coulomb_screening import CoulombScreeningSolver, PCoulombScreening
from quatrex.electron import (
    SigmaCoulombScreening,
    SigmaFock,
    SigmaPhonon,
)
from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from quatrex.core.scba import SCBA

profiler = Profiler()


class Interaction(ABC):
    """Abstract scattering interaction in an SCBA iteration.

    A concrete interaction owns its scattering-self-energy machinery
    and, when its :meth:`compute` is called, additively writes its
    contribution into the SCBA's self-energy buffers. Multiple
    interactions are summed by the SCBA loop in the order they are
    registered.
    """

    #: Name of the interaction (used for profiling labels).
    name: str = "interaction"

    @abstractmethod
    def compute(self, scba: "SCBA") -> None:
        """Apply this interaction to the SCBA state.

        Concrete subclasses read from ``scba.data.g_*`` (and any
        cross-subsystem buffers needed for e-ph-style couplings) and
        additively accumulate into ``scba.data.sigma_*``.
        """
        ...


class CoulombScreeningInteraction(Interaction):
    """GW-style screened Coulomb interaction for the electron subsystem.

    Owns the polarisation kernel, the Dyson solve for the screened
    interaction $W$, the bare-exchange (Fock) self-energy, and the
    screened self-energy. ``compute`` reproduces the body of the old
    :meth:`SCBA._compute_coulomb_screening_interaction` bit-for-bit so
    the registry refactor is transparent for single-subsystem runs.
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
        # The Fock self-energy reads its Coulomb matrix in the ``nnz``
        # distribution; the polarisation/screening Dyson + retarded/
        # lesser/greater pipeline runs on the electron grid.
        if coulomb_matrix.distribution_state != "nnz":
            coulomb_matrix.dtranspose()

        self.sigma_fock = SigmaFock(
            config,
            coulomb_matrix,
            electron_energies,
        )

        # Transpose back so the rest of the SCBA setup sees the
        # original layout (matches the legacy code path).
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
        data.p_retarded.allocate_data()

        self.p_coulomb_screening.compute(
            data.g_lesser,
            data.g_greater,
            out=(data.p_lesser, data.p_greater, data.p_retarded),
        )

        data.w_greater.allocate_data()
        data.w_lesser.allocate_data()

        self.coulomb_screening_solver.solve(
            data.p_lesser,
            data.p_greater,
            data.p_retarded,
            out=(data.w_lesser, data.w_greater),
        )

        # Observation hook: writes polarisation/screening densities
        # into the observables container if the corresponding output
        # flags are set.
        scba._compute_coulomb_screening_observables()

        data.p_lesser.free_data()
        data.p_greater.free_data()
        data.p_retarded.free_data()

        self.sigma_fock.compute(
            data.g_lesser,
            out=(data.sigma_retarded,),
        )

        self.sigma_coulomb_screening.compute(
            data.g_lesser,
            data.g_greater,
            data.w_lesser,
            data.w_greater,
            out=(
                data.sigma_lesser,
                data.sigma_greater,
                data.sigma_retarded,
            ),
        )

        data.w_greater.free_data()
        data.w_lesser.free_data()


class PhononPhononInteraction(Interaction):
    """3-phonon scattering self-energy (cubic-anharmonic NEGF model).

    Wraps :class:`quatrex.phonon.sse_phonon_phonon.SigmaPhononPhonon`.
    """

    name = "phonon_phonon"

    def __init__(self, config, phonon_energies, block_sizes) -> None:
        self.sigma_phonon_phonon = SigmaPhononPhonon(
            config,
            phonon_energies,
            block_sizes=np.asarray(block_sizes),
        )

    @profiler.profile(label="Interaction: Phonon-Phonon", level="default", comm=comm)
    def compute(self, scba: "SCBA") -> None:
        data = scba.data
        self.sigma_phonon_phonon.compute(
            data.g_lesser,
            data.g_greater,
            out=(
                data.sigma_lesser,
                data.sigma_greater,
                data.sigma_retarded,
            ),
        )


class PseudoScatteringPhononInteraction(Interaction):
    """Pseudo-scattering phonon self-energy (legacy non-NEGF model).

    Wraps :class:`quatrex.electron.SigmaPhonon`, which adds a
    pseudo-scattering phonon contribution to the electron self-energy.
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
                data.sigma_retarded,
            ),
        )


def build_interactions(scba: "SCBA") -> list[Interaction]:
    """Construct the list of enabled :class:`Interaction` instances.

    Called once from ``SCBA.__init__`` after the auxiliary state
    (Coulomb matrix, phonon FC3 path, electron energies, ...) has
    been set up. The order of the returned list determines the order
    in which interactions accumulate into ``sigma_*`` within a single
    SCBA iteration; for the currently registered interactions the
    contributions commute (each writes into independent buffers or
    accumulates additively), so the order has no physical effect.
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
