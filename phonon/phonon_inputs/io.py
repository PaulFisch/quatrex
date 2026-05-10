"""I/O utilities for saving and loading transport and decomposition results.

Provides a simple NPZ + JSON sidecar pattern:
- Arrays (spectra, self-energies) → .npz
- Scalars and metadata → .json sidecar

Usage:
    from phonon_inputs.io import save_transport_results, load_transport_results

    # After SCBA converges:
    save_transport_results(result_dict, Path("results/dense_4x4"))

    # Later, for plotting:
    result = load_transport_results(Path("results/dense_4x4"))
"""

import json
from pathlib import Path

import numpy as np


def save_transport_results(results: dict, path: Path, metadata: dict = None):
    """Save SCBA transport results to disk.

    Creates:
        {path}/transport.npz  — array data
        {path}/transport.json — scalar data + metadata

    Parameters
    ----------
    results : dict
        Return dict from any *_anharmonic_transmission function.
        Expected keys: freqs_thz, spectral_heat_current,
        spectral_heat_current_ballistic, thermal_conductance_anharmonic,
        thermal_conductance_ballistic, heat_flow_conservation, etc.
    path : Path
        Output directory (created if needed).
    metadata : dict, optional
        Extra info to store (method name, parameters, timing, etc.).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    arrays = {}
    scalars = {}

    for k, v in results.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, (int, float, bool, str)):
            scalars[k] = v
        elif isinstance(v, np.floating):
            scalars[k] = float(v)
        elif isinstance(v, np.integer):
            scalars[k] = int(v)
        elif isinstance(v, (list, tuple)):
            # Try to convert to array
            try:
                arrays[k] = np.array(v)
            except (ValueError, TypeError):
                scalars[k] = v

    np.savez_compressed(path / "transport.npz", **arrays)

    json_data = {"scalars": scalars}
    if metadata is not None:
        json_data["metadata"] = metadata
    with open(path / "transport.json", "w") as f:
        json.dump(json_data, f, indent=2, default=str)


def load_transport_results(path: Path) -> dict:
    """Load transport results saved by save_transport_results.

    Returns
    -------
    dict
        Combined dict with arrays and scalars.
    """
    path = Path(path)
    result = {}

    npz_path = path / "transport.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            for k in data.files:
                result[k] = data[k]

    json_path = path / "transport.json"
    if json_path.exists():
        with open(json_path) as f:
            json_data = json.load(f)
        result.update(json_data.get("scalars", {}))
        if "metadata" in json_data:
            result["_metadata"] = json_data["metadata"]

    return result


def save_decomposition(data: dict, path: Path):
    """Save PCP or SVD decomposition for reuse.

    Parameters
    ----------
    data : dict
        For PCP: A_modes, lambdas, info (from fit_pcp)
        For SVD: F_list, H, svals (from decompose_fc3_supercell)
    path : Path
        Output directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    arrays = {}
    scalars = {}

    for k, v in data.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, dict):
            # Nested dict (e.g., info from fit_pcp)
            for kk, vv in v.items():
                if isinstance(vv, np.ndarray):
                    arrays[f"{k}__{kk}"] = vv
                elif isinstance(vv, (int, float, bool, str, np.floating, np.integer)):
                    scalars[f"{k}__{kk}"] = float(vv) if isinstance(vv, (np.floating, float)) else vv
        elif isinstance(v, (list, tuple)):
            # List of arrays (e.g., F_list from SVD)
            for i, item in enumerate(v):
                if isinstance(item, np.ndarray):
                    arrays[f"{k}__{i}"] = item
        elif isinstance(v, (int, float, bool, str, np.floating, np.integer)):
            scalars[k] = float(v) if isinstance(v, (np.floating, float)) else v

    np.savez_compressed(path / "decomposition.npz", **arrays)

    with open(path / "decomposition.json", "w") as f:
        json.dump(scalars, f, indent=2, default=str)


def load_decomposition(path: Path) -> dict:
    """Load decomposition saved by save_decomposition.

    Returns
    -------
    dict
        Reconstructed data dict. Nested dicts (info__key) are
        re-nested. Lists (key__0, key__1, ...) are re-listed.
    """
    path = Path(path)
    result = {}

    npz_path = path / "decomposition.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            for k in data.files:
                if "__" in k:
                    parent, child = k.split("__", 1)
                    if child.isdigit():
                        # List item
                        result.setdefault(parent, [])
                        idx = int(child)
                        lst = result[parent]
                        while len(lst) <= idx:
                            lst.append(None)
                        lst[idx] = data[k]
                    else:
                        # Nested dict
                        result.setdefault(parent, {})[child] = data[k]
                else:
                    result[k] = data[k]

    json_path = path / "decomposition.json"
    if json_path.exists():
        with open(json_path) as f:
            scalars = json.load(f)
        for k, v in scalars.items():
            if "__" in k:
                parent, child = k.split("__", 1)
                result.setdefault(parent, {})[child] = v
            else:
                result[k] = v

    return result
