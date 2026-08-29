#!/usr/bin/env python3
"""Audit a production Si-film ballistic run against Caroli and literature.

The run must have been made with ``QX_BALLISTIC=1`` and
``QX_DIAG_CAROLI=1``.  The script performs the physical-unit integration
twice, once from the production Meir--Wingreen spectrum and once from the
independently assembled Caroli spectrum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "phonon"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from phonon.postproc.units import heat_current_watts


SI_FILM_AREA_M2 = 1.294665716618031e-19

# Table III of Guo et al., Phys. Rev. B 102, 195412 (2020).  These values
# expose the residual frequency/q convergence of that calculation rather
# than presenting its final point without context.
GUO_TABLE_III_MW_M2K = {
    "61_q4": 1130.18,
    "81_q4": 1090.59,
    "81_q6": 1069.35,
    "101_q4": 1074.60,
    "101_q6": 1040.03,
    "101_q8": 1053.53,
    "121_q6": 1061.24,
    "121_q8": 1065.81,
}

# Surviving eta=0 Quatrex results on the same FC2 input.  They are used only
# as regression values; the new run remains the primary result.
HISTORICAL_QUATREX_MW_M2K = {
    "q9_L3": 995.953812735210,
    "q9_L5": 995.953812735219,
    "q9_L8": 995.953812735228,
    "q13_L3": 1002.752845954465,
}


def _lead_average(values: np.ndarray) -> float:
    return 0.5 * (abs(float(values[0])) + abs(float(values[-1])))


def audit_snapshot(path: str | Path,
                   area_m2: float = SI_FILM_AREA_M2) -> dict:
    data = np.load(path, allow_pickle=True)
    required = {
        "energies", "frequency_cell_widths", "current_spectrum",
        "caroli_transmission", "caroli_current_spectrum", "t_left",
        "t_right",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"{path}: missing ballistic audit arrays: {missing}")

    frequencies = np.asarray(data["energies"], dtype=float)
    weights = np.asarray(data["frequency_cell_widths"], dtype=float)
    mw_current = heat_current_watts(
        frequencies, data["current_spectrum"], weights)
    caroli_spectrum = np.asarray(data["caroli_current_spectrum"])[..., None]
    caroli_current = heat_current_watts(
        frequencies, caroli_spectrum, weights)
    delta_t = float(data["t_left"] - data["t_right"])
    if delta_t == 0:
        raise ValueError("a nonzero temperature difference is required")

    mw_g = _lead_average(mw_current) / delta_t / area_m2 * 1e-6
    caroli_g = abs(float(caroli_current[0])) / delta_t / area_m2 * 1e-6
    q_mesh = np.asarray(data["q_mesh"], dtype=int).tolist()
    final_guo = GUO_TABLE_III_MW_M2K["121_q8"]
    historical_q9 = HISTORICAL_QUATREX_MW_M2K["q9_L5"]
    transmission = np.asarray(data["caroli_transmission"], dtype=float)
    return {
        "run": str(Path(path)),
        "source_commit": str(data["source_commit"]) if "source_commit" in data else "",
        "q_mesh": q_mesh,
        "frequency_points": int(frequencies.size),
        "frequency_max_thz": float(frequencies[-1]),
        "frequency_spacing_thz": float(frequencies[1] - frequencies[0]),
        "temperatures_k": [float(data["t_left"]), float(data["t_right"])],
        "area_m2": float(area_m2),
        "mw_current_w": mw_current.tolist(),
        "caroli_current_w": float(caroli_current[0]),
        "mw_conductance_mw_m2k": float(mw_g),
        "caroli_conductance_mw_m2k": float(caroli_g),
        "integrated_caroli_mw_relative": float(abs(caroli_g - mw_g) / mw_g),
        "spectral_caroli_mw_relative_l2": float(data["caroli_mw_relative_l2"]),
        "spectral_caroli_mw_active_max_relative": float(
            data["caroli_mw_active_max_relative"]),
        "transmission_min": float(np.min(transmission)),
        "transmission_max": float(np.max(transmission)),
        "relative_to_historical_q9_L5": float(mw_g / historical_q9 - 1.0),
        "relative_to_guo_121_q8": float(mw_g / final_guo - 1.0),
        "historical_quatrex_mw_m2k": HISTORICAL_QUATREX_MW_M2K,
        "guo_table_iii_mw_m2k": GUO_TABLE_III_MW_M2K,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--area", type=float, default=SI_FILM_AREA_M2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_snapshot(args.run, args.area)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
