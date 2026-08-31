# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes function to get electron energies based on the configuration."""

import os

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.utils.mpi_utils import distributed_load
from quatrex.core.config import QuatrexConfig


def frequency_cell_widths(frequencies: NDArray) -> NDArray:
    """Per-bin quadrature cell widths of an ascending frequency grid.

    Interior bins get the midpoint-to-midpoint width
    ``(w[i+1] - w[i-1]) / 2``; the two edge bins get their single
    neighbouring gap, so a UNIFORM grid yields exactly ``dw`` for EVERY
    bin (including the edges) and ``sum(f * widths)`` reproduces the
    legacy ``sum(f) * dw`` Riemann sum bit-for-bit up to the reordering
    of the scalar multiply. On a non-uniform grid this is the piecewise
    cell measure the frequency integrals need.
    """
    w = xp.asarray(frequencies, dtype=float)
    if w.shape[0] < 2:
        return xp.ones_like(w)
    widths = xp.empty_like(w)
    widths[1:-1] = 0.5 * (w[2:] - w[:-2])
    widths[0] = w[1] - w[0]
    widths[-1] = w[-1] - w[-2]
    return widths


def is_uniform_grid(frequencies: NDArray, rtol: float = 1e-9) -> bool:
    """True if the grid spacing is constant to relative tolerance."""
    w = xp.asarray(frequencies, dtype=float)
    if w.shape[0] < 3:
        return True
    dw = float(w[1] - w[0])
    if dw == 0.0:
        return False
    return bool(xp.max(xp.abs(xp.diff(w) - dw)) <= rtol * abs(dw))


def get_electron_energies(config: QuatrexConfig) -> NDArray:
    """Get the electron energies based on the configuration.
    If an energy window is specified in the configuration, it generates
    the energies using linspace. Otherwise, it attempts to load the energies
    from a file named 'electron_energies.npy' in the input directory.

    Parameters
    ----------
    config : QuatrexConfig
        The Quatrex configuration.

    Returns
    -------
    electron_energies : NDArray
        Array of electron energies.

    Raises
    -------
    ValueError
        If both or neither of `energy_window_num` and `energy_window_num_per_rank` are set.
    FileNotFoundError
        If the energies file is not found and no energy window is specified.

    """

    if (config.electron.energy_window_max is not None) and (
        config.electron.energy_window_min is not None
    ):
        if config.electron.energy_window_num is not None:
            if config.electron.energy_window_num_per_rank is not None:
                raise ValueError(
                    "Should **exclusively** set electron `energy_window_num` or `energy_window_num_per_rank` in the config."
                )
            electron_energies = xp.linspace(
                config.electron.energy_window_min,
                config.electron.energy_window_max,
                config.electron.energy_window_num,
            )
        elif config.electron.energy_window_num_per_rank is not None:
            energy_window_num = (
                config.electron.energy_window_num_per_rank * comm.stack.size
            )
            electron_energies = xp.linspace(
                config.electron.energy_window_min,
                config.electron.energy_window_max,
                energy_window_num,
            )
        else:
            raise ValueError(
                "Should set electron `energy_window_num` or `energy_window_num_per_rank` in the config."
            )
    else:
        energies_path = config.input_dir / "electron_energies.npy"
        if os.path.isfile(energies_path):
            electron_energies = distributed_load(energies_path)
        else:
            raise FileNotFoundError(
                f"Could not find electron energies file at {energies_path}. Please provide an energy window in the config."
            )

    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    xp.save(config.output_dir / "electron_energies.npy", electron_energies)

    return electron_energies
