# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Transfer data between primary and auxiliary frequency grids."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from qttools import NDArray, xp
from quatrex.grid.energies import frequency_cell_widths


@dataclass(frozen=True)
class AuxiliaryGrid:
    frequencies: NDArray
    interpolation: tuple[NDArray, NDArray, NDArray, NDArray]
    restriction: tuple[str, NDArray | tuple[NDArray, NDArray]]

    def interpolate(self, data: NDArray) -> NDArray:
        """Interpolate axis zero from the primary grid."""
        low, high, weight, valid = self.interpolation
        shape = (weight.size,) + (1,) * (data.ndim - 1)
        weight = weight.reshape(shape)
        return (
            (1.0 - weight) * data[low] + weight * data[high]
        ) * valid.reshape(shape)

    def restrict(self, data: NDArray) -> NDArray:
        """Transfer axis zero back to the primary grid."""
        kind, operator = self.restriction
        if kind == "adjoint":
            flat = data.reshape(data.shape[0], -1)
            return (operator @ flat).reshape((operator.shape[0],) + data.shape[1:])
        index, weight = operator
        shape = (weight.size,) + (1,) * (data.ndim - 1)
        weight = weight.reshape(shape)
        return (1.0 - weight) * data[index] + weight * data[index + 1]


def make_auxiliary_grid(
    primary: NDArray,
    spacing: float,
    max_frequency: float = 0.0,
    restriction: Literal["adjoint", "sample"] = "adjoint",
) -> AuxiliaryGrid:
    """Build a zero-anchored uniform grid and its transfer operators."""
    primary = xp.asarray(primary, dtype=float)
    top = max(float(primary[-1]), max_frequency)
    frequencies = xp.arange(int(np.ceil(top / spacing - 1e-9)) + 1) * spacing

    high = xp.clip(
        xp.searchsorted(primary, frequencies, side="left"), 1, primary.size - 1
    )
    low = high - 1
    gap = primary[high] - primary[low]
    weight = xp.clip(
        (frequencies - primary[low]) / xp.where(gap > 0, gap, 1.0), 0.0, 1.0
    )
    valid = (frequencies >= primary[0] - 1e-12 * spacing) & (
        frequencies <= primary[-1] + 1e-12 * spacing
    )
    interpolation = (low, high, weight, valid)

    if restriction == "sample":
        position = primary / spacing
        index = xp.clip(xp.floor(position).astype(int), 0, frequencies.size - 2)
        transfer = ("sample", (index, xp.clip(position - index, 0.0, 1.0)))
    elif restriction == "adjoint":
        row_weight = frequency_cell_widths(primary) * primary
        column_weight = spacing * frequencies
        operator = xp.zeros((primary.size, frequencies.size))
        columns = xp.arange(frequencies.size)
        operator[low, columns] = (1.0 - weight) * valid * column_weight
        operator[high, columns] = weight * valid * column_weight
        denominator = xp.where(row_weight[:, None] > 0.0, row_weight[:, None], 1.0)
        operator = xp.where(row_weight[:, None] > 0.0, operator / denominator, 0.0)
        transfer = ("adjoint", operator)
    else:
        raise ValueError(f"Unknown auxiliary-grid restriction {restriction!r}.")

    return AuxiliaryGrid(frequencies, interpolation, transfer)
