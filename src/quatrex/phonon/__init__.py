# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the classes and methods for phonon calculations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quatrex.phonon.polarization import PiPhonon
    from quatrex.phonon.solver import PhononSolver

__all__ = ["PhononSolver", "PiPhonon"]


def __getattr__(name: str):
    """Load public phonon classes on demand."""
    if name == "PhononSolver":
        from quatrex.phonon.solver import PhononSolver
        return PhononSolver
    if name == "PiPhonon":
        from quatrex.phonon.polarization import PiPhonon
        return PiPhonon
    raise AttributeError(name)
