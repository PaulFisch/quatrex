"""Independent Bloch-mode audit of a ballistic Si-film production run.

The production Caroli audit deliberately reuses the lead self-energies.  This
study instead diagonalises the real-space harmonic dynamical matrix on a dense
transport-wavevector mesh.  A positive-frequency crossing with positive group
velocity is one incident channel.  The resulting mode count and its Landauer
integral do not use a surface Green function, a contact self-energy, or a
device Green function.

Example
-------
OPENBLAS_NUM_THREADS=8 python phonon/studies/_si_ballistic_mode_count.py \\
    --matrix cluster/si-l5-q9-r128-in/dynamical_matrix.mat \\
    --run cluster/si-l5-ballistic-q9-w20-dw003125-caroli/run.npz \\
    --nk 1025,2049,4097 \\
    --output phonon/studies/out/si_ballistic_q9_mode_count.json
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

PLANCK = 6.626_070_15e-34
BOLTZMANN = 1.380_649e-23
THZ = 1.0e12
DEFAULT_AREA_M2 = 1.294_665_716_618_031e-19


def load_real_space_blocks(path: Path) -> dict[tuple[int, int, int], np.ndarray]:
    """Read the offset-keyed dynamical-matrix blocks written by Quatrex."""
    raw = loadmat(path)
    blocks = {
        tuple(int(i) for i in ast.literal_eval(key)): np.asarray(value, complex)
        for key, value in raw.items()
        if key.startswith("[")
    }
    if not blocks or any(len(offset) != 3 for offset in blocks):
        raise ValueError(f"no three-dimensional offset blocks in {path}")
    return blocks


def transverse_blocks(
    blocks: dict[tuple[int, int, int], np.ndarray], qy: float, qz: float
) -> dict[int, np.ndarray]:
    """Fourier transform the two transverse lattice coordinates."""
    transport_offsets = sorted({offset[0] for offset in blocks})
    return {
        rx: sum(
            block * np.exp(2j * np.pi * (ry * qy + rz * qz))
            for (ix, ry, rz), block in blocks.items()
            if ix == rx
        )
        for rx in transport_offsets
    }


def bloch_bands(
    transformed: dict[int, np.ndarray], reduced_k: np.ndarray
) -> np.ndarray:
    """Return sorted non-negative frequencies in THz on a Bloch path."""
    matrix = sum(
        block[None, :, :]
        * np.exp(2j * np.pi * rx * reduced_k)[:, None, None]
        for rx, block in transformed.items()
    )
    matrix = 0.5 * (matrix + matrix.conj().transpose(0, 2, 1))
    return np.sqrt(np.clip(np.linalg.eigvalsh(matrix), 0.0, None))


def positive_mode_count(bands: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Count positive-slope crossings of each requested frequency.

    Sorted eigenvalues can exchange branch labels at a degeneracy.  The total
    positive variation and the crossing count are invariant under that
    exchange except exactly at the discontinuous channel threshold.
    """
    out = np.zeros(frequencies.shape, dtype=float)
    for branch in bands.T:
        below = branch[None, :] < frequencies[:, None]
        out += np.sum(below[:, :-1] & ~below[:, 1:], axis=1)
    return out


def bose(frequency_thz: np.ndarray, temperature_k: float) -> np.ndarray:
    x = PLANCK * THZ * frequency_thz / (BOLTZMANN * temperature_k)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return 1.0 / np.expm1(x)


def frequency_weighted_occupation_difference(
    frequency_thz: np.ndarray, left_temperature: float, right_temperature: float
) -> np.ndarray:
    """Return ``nu * (n_L - n_R)`` with its analytic zero limit."""
    frequency_thz = np.asarray(frequency_thz, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = frequency_thz * (
            bose(frequency_thz, left_temperature)
            - bose(frequency_thz, right_temperature)
        )
    zero = frequency_thz == 0.0
    out[zero] = (
        BOLTZMANN
        * (left_temperature - right_temperature)
        / (PLANCK * THZ)
    )
    return out


def positive_variation_current(
    bands: np.ndarray, left_temperature: float, right_temperature: float
) -> float:
    """Integrate the finite-bias Landauer current along positive band pieces."""
    lo = bands[:-1]
    hi = bands[1:]
    positive = hi > lo
    midpoint = 0.5 * (lo + hi)
    occupation = bose(midpoint[positive], left_temperature) - bose(
        midpoint[positive], right_temperature
    )
    return float(
        np.sum(
            PLANCK
            * THZ
            * midpoint[positive]
            * occupation
            * THZ
            * (hi[positive] - lo[positive])
        )
    )


def production_spectrum_conductance(
    frequencies: np.ndarray,
    widths: np.ndarray,
    current_spectrum: np.ndarray,
    area_m2: float,
    left_temperature: float,
    right_temperature: float,
) -> tuple[np.ndarray, float]:
    """Reduce a saved production MW spectrum to SI conductance.

    The solver stores a q-resolved number-current spectrum.  The transverse
    axes are averaged, as in ``phonon.postproc.units``, before multiplying by
    ``h nu dnu``.  Keeping this small reducer here avoids importing any
    production Green-function or contact implementation into the mode oracle.
    """
    spectrum = np.real(np.asarray(current_spectrum))
    while spectrum.ndim > 2:
        spectrum = spectrum.mean(axis=1)
    currents = PLANCK * THZ**2 * np.sum(
        np.asarray(widths)[:, None]
        * np.asarray(frequencies)[:, None]
        * spectrum,
        axis=0,
    )
    delta_temperature = left_temperature - right_temperature
    if delta_temperature == 0.0:
        raise ValueError("a nonzero temperature difference is required")
    lead_current = 0.5 * (abs(currents[0]) + abs(currents[-1]))
    conductance = lead_current / (area_m2 * delta_temperature) / 1.0e6
    return currents, float(conductance)


def mode_integral_audit(
    matrix_path: Path,
    q_mesh: tuple[int, int],
    nk_values: list[int],
    area_m2: float,
    left_temperature: float,
    right_temperature: float,
) -> dict[str, object]:
    """Landauer conductance from Bloch bands without a production run."""
    blocks = load_real_space_blocks(matrix_path)
    for nk in nk_values:
        if nk < 3 or nk % 2 == 0:
            raise ValueError("each nk must be odd and at least three")
    finest_nk = max(nk_values)
    strides: dict[int, int] = {}
    for nk in nk_values:
        if (finest_nk - 1) % (nk - 1):
            raise ValueError("nk refinements must be nested in the finest mesh")
        strides[nk] = (finest_nk - 1) // (nk - 1)

    reduced_k = np.linspace(-0.5, 0.5, finest_nk)
    currents = {nk: 0.0 for nk in nk_values}
    for iqy, iqz in np.ndindex(q_mesh):
        transformed = transverse_blocks(
            blocks, iqy / q_mesh[0], iqz / q_mesh[1]
        )
        bands = bloch_bands(transformed, reduced_k)
        for nk in nk_values:
            currents[nk] += positive_variation_current(
                bands[:: strides[nk]], left_temperature, right_temperature
            )

    refinements: list[dict[str, float | int]] = []
    for nk in nk_values:
        current = currents[nk] / (q_mesh[0] * q_mesh[1])
        conductance = current / (
            area_m2 * (left_temperature - right_temperature)
        )
        refinements.append(
            {"nk": nk, "conductance_mw_m2k": conductance / 1.0e6}
        )
    return {
        "matrix": str(matrix_path),
        "q_mesh": [1, *q_mesh],
        "area_m2": area_m2,
        "temperatures_k": [left_temperature, right_temperature],
        "refinements": refinements,
        "method": "positive Bloch-band variation",
    }


def audit(
    matrix_path: Path,
    run_path: Path,
    nk_values: list[int],
    area_m2: float = DEFAULT_AREA_M2,
) -> dict[str, object]:
    blocks = load_real_space_blocks(matrix_path)
    with np.load(run_path) as run:
        frequencies = np.asarray(run["energies"], float)
        caroli = (
            np.asarray(run["caroli_transmission"], float)
            if "caroli_transmission" in run else None
        )
        current_spectrum = (
            np.asarray(run["current_spectrum"])
            if "current_spectrum" in run else None
        )
        q_mesh = tuple(int(i) for i in run["q_mesh"])
        left_temperature = float(run["left_temperature"])
        right_temperature = float(run["right_temperature"])
        source_commit = str(run["source_commit"])
        widths = np.asarray(run["frequency_cell_widths"], float)
    if caroli is None and current_spectrum is None:
        raise ValueError("run has neither a Caroli nor a production spectrum")
    if q_mesh[0] != 1 or (
        caroli is not None and caroli.shape[1:] != q_mesh[1:]
    ):
        raise ValueError("run is not a two-dimensional transverse-q film")

    # The production shift 1/2 - 1/(2n) makes the mesh i/n.  Evaluating the
    # finite real-space Fourier polynomial there avoids importing the MPI
    # production loader and is algebraically identical to its q transform.
    qy = np.arange(q_mesh[1], dtype=float) / q_mesh[1]
    qz = np.arange(q_mesh[2], dtype=float) / q_mesh[2]
    transformed = {
        index: transverse_blocks(blocks, qy[index[0]], qz[index[1]])
        for index in np.ndindex(q_mesh[1:])
    }

    for nk in nk_values:
        if nk < 3 or nk % 2 == 0:
            raise ValueError("each nk must be odd and at least three")
    finest_nk = max(nk_values)
    strides = {}
    for nk in nk_values:
        if (finest_nk - 1) % (nk - 1):
            raise ValueError("nk refinements must be nested in the finest mesh")
        strides[nk] = (finest_nk - 1) // (nk - 1)

    reduced_k = np.linspace(-0.5, 0.5, finest_nk)
    currents = {nk: 0.0 for nk in nk_values}
    finest_modes = np.zeros_like(caroli) if caroli is not None else None
    for index, q_blocks in transformed.items():
        bands = bloch_bands(q_blocks, reduced_k)
        if finest_modes is not None:
            finest_modes[(slice(None),) + index] = positive_mode_count(
                bands, frequencies
            )
        for nk in nk_values:
            currents[nk] += positive_variation_current(
                bands[:: strides[nk]], left_temperature, right_temperature
            )

    refinements: list[dict[str, float | int]] = []
    for nk in nk_values:
        current = currents[nk] / (q_mesh[1] * q_mesh[2])
        conductance = current / (
            area_m2 * (left_temperature - right_temperature)
        )
        refinements.append(
            {
                "nk": nk,
                "conductance_mw_m2k": conductance / 1.0e6,
            }
        )
    finest_conductance = float(refinements[-1]["conductance_mw_m2k"])
    result: dict[str, object] = {
        "matrix": str(matrix_path),
        "run": str(run_path),
        "source_commit": source_commit,
        "q_mesh": list(q_mesh),
        "area_m2": area_m2,
        "temperatures_k": [left_temperature, right_temperature],
        "refinements": refinements,
    }
    if current_spectrum is not None:
        production_current, production_conductance = (
            production_spectrum_conductance(
                frequencies,
                widths,
                current_spectrum,
                area_m2,
                left_temperature,
                right_temperature,
            )
        )
        result.update({
            "production_current_w": production_current.tolist(),
            "production_conductance_mw_m2k": production_conductance,
            "finest_mode_to_production_conductance_relative": (
                finest_conductance / production_conductance - 1.0
            ),
        })
    if caroli is not None and finest_modes is not None:
        difference = finest_modes - caroli
        caroli_norm = float(np.linalg.norm(caroli))
        weighted_occupation = frequency_weighted_occupation_difference(
            frequencies, left_temperature, right_temperature
        )
        # The stored production current omits the exactly-zero bin.  Its
        # Caroli transmission is ill-conditioned at the acoustic threshold,
        # whereas the continuum mode integral includes its analytic limit.
        weighted_occupation[frequencies == 0.0] = 0.0
        caroli_current = PLANCK * THZ**2 * np.sum(
            widths[:, None, None]
            * weighted_occupation[:, None, None]
            * caroli
        ) / (q_mesh[1] * q_mesh[2])
        caroli_conductance = caroli_current / (
            area_m2 * (left_temperature - right_temperature)
        )
        result.update({
            "caroli_grid_dc_dropped_conductance_mw_m2k": (
                caroli_conductance / 1.0e6
            ),
            "finest_mode_to_caroli_conductance_relative": (
                finest_conductance / (caroli_conductance / 1.0e6) - 1.0
            ),
            "mode_to_caroli_spectral_relative_l2": (
                float(np.linalg.norm(difference)) / caroli_norm
            ),
            "mode_to_caroli_exact_bin_fraction": float(
                np.mean(np.abs(difference) <= 1.0e-6)
            ),
            "mode_count_min": float(np.min(finest_modes)),
            "mode_count_max": float(np.max(finest_modes)),
            "caroli_transmission_min": float(np.min(caroli)),
            "caroli_transmission_max": float(np.max(caroli)),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument(
        "--q-mesh", type=int,
        help="square transverse mesh for a mode-only audit without --run",
    )
    parser.add_argument("--temperatures", default="305,295")
    parser.add_argument("--nk", default="1025,2049,4097")
    parser.add_argument("--area", type=float, default=DEFAULT_AREA_M2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    nk_values = [int(value) for value in args.nk.split(",")]
    if args.run is not None:
        result = audit(args.matrix, args.run, nk_values, args.area)
    else:
        if args.q_mesh is None:
            parser.error("give --run or --q-mesh")
        temperatures = [float(value) for value in args.temperatures.split(",")]
        if len(temperatures) != 2:
            parser.error("--temperatures must be left,right")
        result = mode_integral_audit(
            args.matrix,
            (args.q_mesh, args.q_mesh),
            nk_values,
            args.area,
            temperatures[0],
            temperatures[1],
        )
    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
