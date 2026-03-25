#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from plot_contact_vs_bulk_projected import (
    build_q_points,
    bulk_projected_bands,
    contact_bands_for_q,
    load_phonon,
    reconstruct_basis_masses,
)
from extract_harmonic_fc import construct_device_fc_matrix
from extract_transport_blocks import build_transverse_hoppings, choose_reference_cell


def hungarian_min_cost(cost_matrix):
    """Return (assignment, min_cost) for a square cost matrix.

    assignment[i] = chosen column for row i.
    """

    import numpy as np

    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise RuntimeError("Hungarian solver expects a square 2D cost matrix.")

    n = cost.shape[0]
    u = np.zeros(n + 1, dtype=float)
    v = np.zeros(n + 1, dtype=float)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=float)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = np.full(n, -1, dtype=int)
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1

    min_cost = float(sum(cost[i, assignment[i]] for i in range(n)))
    return assignment, min_cost


def matched_frequency_error(omega_contact, omega_bulk):
    """Compute RMS/max mismatch after optimal branch matching at each k."""

    import numpy as np

    if omega_contact.shape != omega_bulk.shape:
        raise RuntimeError(
            f"Shape mismatch in matched_frequency_error: {omega_contact.shape} vs {omega_bulk.shape}"
        )

    nk, nm = omega_contact.shape
    matched_sq = []
    matched_abs = []
    for ik in range(nk):
        c = omega_contact[ik]
        b = omega_bulk[ik]
        cost = (c[:, None] - b[None, :]) ** 2
        assignment, _ = hungarian_min_cost(cost)
        diffs = c - b[assignment]
        matched_sq.extend((diffs**2).tolist())
        matched_abs.extend(np.abs(diffs).tolist())

    matched_sq_arr = np.asarray(matched_sq, dtype=float)
    matched_abs_arr = np.asarray(matched_abs, dtype=float)
    return float(np.sqrt(np.mean(matched_sq_arr))), float(np.max(matched_abs_arr))


def build_same_formalism_reference(
    phonon,
    force_constants,
    metadata: dict,
    deltas,
):
    """Rebuild transverse hoppings from the same formalism used in extraction."""

    import numpy as np

    axis = str(metadata["transport_axis"])
    num_cells = int(metadata["device_cells"])
    if num_cells < 2:
        raise RuntimeError("device_cells in metadata must be at least 2.")

    device_matrix_ref, metadata_ref = construct_device_fc_matrix(
        phonon=phonon,
        force_constants=force_constants,
        axis=axis,
        num_cells=num_cells,
    )

    dof_per_cell = int(metadata_ref["degrees_of_freedom_per_transport_cell"])
    num_cells_ref = device_matrix_ref.shape[0] // dof_per_cell
    reference_cell = choose_reference_cell(num_cells_ref, None)

    r0 = reference_cell * dof_per_cell
    K00 = device_matrix_ref[r0 : r0 + dof_per_cell, r0 : r0 + dof_per_cell]
    couplings_by_delta: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for delta in deltas:
        d = int(delta)
        right_cell = reference_cell + d
        if right_cell >= num_cells_ref:
            continue
        c0 = right_cell * dof_per_cell
        K0d = device_matrix_ref[r0 : r0 + dof_per_cell, c0 : c0 + dof_per_cell]
        Kd0 = device_matrix_ref[c0 : c0 + dof_per_cell, r0 : r0 + dof_per_cell]
        couplings_by_delta[d] = (K0d, Kd0)

    if not couplings_by_delta:
        raise RuntimeError("No longitudinal couplings available in same-formalism reference.")

    dt_ref, K00_ref, K0d_ref, Kd0_ref, deltas_ref, _ = build_transverse_hoppings(
        metadata=metadata_ref,
        K00=K00,
        couplings_by_delta=couplings_by_delta,
        dof_per_cell=dof_per_cell,
    )

    return dt_ref, deltas_ref, K00_ref, K0d_ref, Kd0_ref


def tensor_diff_stats(a, b):
    import numpy as np

    diff = np.asarray(a) - np.asarray(b)
    abs_diff = np.abs(diff)
    denom = float(np.linalg.norm(np.asarray(b)))
    return {
        "max_abs": float(np.max(abs_diff)),
        "frobenius": float(np.linalg.norm(diff)),
        "relative_frobenius": float(np.linalg.norm(diff) / (denom + 1e-30)),
    }


def inspect_force_constants_interpretation(harmonic_dir: Path, phonon, force_constants):
    """Inspect raw FORCE_CONSTANTS layout against phonopy's p2s/compact conventions."""

    import numpy as np

    info = {
        "force_constants_shape": [int(v) for v in force_constants.shape],
        "num_primitive_atoms": int(len(phonon.primitive)),
        "num_supercell_atoms": int(len(phonon.supercell)),
    }

    shape = force_constants.shape
    if len(shape) == 4 and shape[0] == len(phonon.primitive) and shape[1] == len(phonon.supercell):
        info["storage_type"] = "compact"
    elif len(shape) == 4 and shape[0] == len(phonon.supercell) and shape[1] == len(phonon.supercell):
        info["storage_type"] = "full"
    else:
        info["storage_type"] = "unknown"

    p2s_map = np.asarray(phonon.primitive.p2s_map, dtype=int)
    info["p2s_map_1based"] = [int(v + 1) for v in p2s_map.tolist()]

    fc_file = harmonic_dir / "FORCE_CONSTANTS"
    if not fc_file.exists():
        info["force_constants_file_found"] = False
        return info

    info["force_constants_file_found"] = True
    info["force_constants_file"] = str(fc_file)

    with fc_file.open("r", encoding="utf-8") as f:
        header = [int(x) for x in f.readline().split()]
        if len(header) == 1:
            header = [header[0], header[0]]

        file_n0, file_n1 = int(header[0]), int(header[1])
        info["file_header_n0_n1"] = [file_n0, file_n1]

        row_indices = []
        for i in range(file_n0):
            first_label = None
            for j in range(file_n1):
                tokens = f.readline().split()
                if not tokens:
                    raise RuntimeError("Unexpected EOF while parsing FORCE_CONSTANTS block labels.")
                if first_label is None:
                    first_label = int(tokens[0])
                for _ in range(3):
                    _ = f.readline()
            if first_label is None:
                raise RuntimeError("Failed to parse FORCE_CONSTANTS row label.")
            row_indices.append(first_label)

    info["file_first_label_each_row_1based"] = [int(v) for v in row_indices]
    info["file_row_labels_match_p2s_map"] = bool(
        len(row_indices) == len(p2s_map) and np.array_equal(np.asarray(row_indices, dtype=int), p2s_map + 1)
    )
    return info


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check contact lead dispersion consistency against folded phonopy bulk bands "
            "for selected transverse q points."
        )
    )
    parser.add_argument("--harmonic-dir", type=Path, required=True)
    parser.add_argument("--transverse-hoppings", type=Path, required=True)
    parser.add_argument("--device-metadata", type=Path, required=True)
    parser.add_argument(
        "--q-point",
        type=float,
        nargs=2,
        action="append",
        metavar=("Q1", "Q2"),
        help="Transverse reduced q-point (repeatable).",
    )
    parser.add_argument(
        "--q-grid",
        type=int,
        nargs=2,
        default=None,
        metavar=("NQ1", "NQ2"),
        help="Uniform transverse q-grid points (i/NQ1, j/NQ2).",
    )
    parser.add_argument(
        "--gamma-centered",
        action="store_true",
        help="Use Gamma-centered q-grid: ((i+0.5)/NQ1-0.5, (j+0.5)/NQ2-0.5).",
    )
    parser.add_argument("--k-min", type=float, default=-0.5)
    parser.add_argument("--k-max", type=float, default=0.5)
    parser.add_argument("--k-num", type=int, default=201)
    parser.add_argument(
        "--bulk-axis-scale",
        type=float,
        default=None,
        help="Scale factor mapping lead k to primitive bulk q on transport axis.",
    )
    parser.add_argument(
        "--fail-rms-threshold",
        type=float,
        default=None,
        help="If set, exit with nonzero code when any q-point RMS exceeds this threshold (THz).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("contact_bulk_consistency_report.json"),
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required.") from exc

    if args.k_num < 2:
        raise RuntimeError("--k-num must be at least 2")
    if args.k_max <= args.k_min:
        raise RuntimeError("--k-max must be greater than --k-min")

    phonon, yaml_path = load_phonon(args.harmonic_dir)
    primitive = phonon.primitive
    harmonic_dir = args.harmonic_dir.expanduser().resolve()
    force_constants = phonon.force_constants

    transverse_path = args.transverse_hoppings.expanduser().resolve()
    if not transverse_path.exists():
        raise RuntimeError(f"Transverse hopping file not found: {transverse_path}")
    with np.load(transverse_path) as hop:
        if "dt" not in hop or "K00" not in hop:
            raise RuntimeError(f"Missing dt/K00 arrays in {transverse_path}")
        dt = np.asarray(hop["dt"], dtype=float)
        K00_hop = np.asarray(hop["K00"], dtype=complex)

        if all(name in hop for name in ["deltas", "K0d", "Kd0"]):
            deltas = np.asarray(hop["deltas"], dtype=int)
            K0d_hop = np.asarray(hop["K0d"], dtype=complex)
            Kd0_hop = np.asarray(hop["Kd0"], dtype=complex)
        else:
            if "K01" not in hop or "K10" not in hop:
                raise RuntimeError(f"Missing K01/K10 arrays in {transverse_path}")
            deltas = np.asarray([1], dtype=int)
            K0d_hop = np.asarray([np.asarray(hop["K01"], dtype=complex)], dtype=complex)
            Kd0_hop = np.asarray([np.asarray(hop["K10"], dtype=complex)], dtype=complex)

    metadata_path = args.device_metadata.expanduser().resolve()
    if not metadata_path.exists():
        raise RuntimeError(f"Device metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    axis_index = {"a": 0, "b": 1, "c": 2}[metadata["transport_axis"]]
    repeats = [int(v) for v in metadata["supercell_repeats"]]

    primitive_masses = [float(primitive.masses[i]) for i in range(len(primitive))]
    basis_masses, transverse_dirs = reconstruct_basis_masses(
        device_metadata=metadata,
        primitive_masses=primitive_masses,
        axis_index=axis_index,
    )

    dof_basis = K00_hop.shape[1]
    if len(basis_masses) != dof_basis:
        raise RuntimeError(
            f"Basis mass length mismatch: {len(basis_masses)} vs hopping dof {dof_basis}."
        )

    inv_sqrt_m = np.diag(1.0 / np.sqrt(np.asarray(basis_masses, dtype=float)))
    freq_factor = float(getattr(phonon, "unit_conversion_factor", 1.0))

    axis_scale = args.bulk_axis_scale
    if axis_scale is None:
        axis_scale = 1.0 / float(repeats[axis_index])

    q_points = build_q_points(args)
    k_grid = np.linspace(args.k_min, args.k_max, args.k_num)

    same_formalism = None
    K00_ref_for_freq = None
    K0d_ref_for_freq = None
    Kd0_ref_for_freq = None
    if force_constants is None:
        raise RuntimeError("Phonon object has no force_constants for same-formalism reference.")

    dt_ref, deltas_ref, K00_ref, K0d_ref, Kd0_ref = build_same_formalism_reference(
        phonon=phonon,
        force_constants=force_constants,
        metadata=metadata,
        deltas=deltas,
    )

    if not np.array_equal(dt_ref, dt):
        raise RuntimeError("Same-formalism reference dt grid does not match extracted dt grid.")
    if not np.array_equal(deltas_ref, deltas):
        raise RuntimeError("Same-formalism reference deltas do not match extracted deltas.")

    same_formalism = {
        "K00": tensor_diff_stats(K00_hop.real, K00_ref),
        "K0d": tensor_diff_stats(K0d_hop.real, K0d_ref),
        "Kd0": tensor_diff_stats(Kd0_hop.real, Kd0_ref),
    }
    K00_ref_for_freq = np.asarray(K00_ref, dtype=complex)
    K0d_ref_for_freq = np.asarray(K0d_ref, dtype=complex)
    Kd0_ref_for_freq = np.asarray(Kd0_ref, dtype=complex)

    mismatch_rows = []
    for q_perp in q_points:
        omega_contact = contact_bands_for_q(
            k_grid=k_grid,
            q_perp=q_perp,
            dt=dt,
            K00_hop=K00_hop,
            K0d_hop=K0d_hop,
            Kd0_hop=Kd0_hop,
            deltas=deltas,
            inv_sqrt_m=inv_sqrt_m,
            freq_factor=freq_factor,
        )
        omega_bulk = bulk_projected_bands(
            phonon=phonon,
            axis_index=axis_index,
            transverse_dirs=transverse_dirs,
            q_perp=q_perp,
            k_grid=k_grid,
            axis_scale=axis_scale,
            transport_repeat=int(repeats[axis_index]),
        )

        diff = omega_contact - omega_bulk
        rms_direct = float(np.sqrt(np.mean(diff**2)))
        max_direct = float(np.max(np.abs(diff)))
        rms_matched, max_matched = matched_frequency_error(omega_contact, omega_bulk)

        row = {
            "q_perp": [float(q_perp[0]), float(q_perp[1])],
            "rms_direct_thz": rms_direct,
            "max_abs_direct_thz": max_direct,
            "rms_matched_thz": rms_matched,
            "max_abs_matched_thz": max_matched,
        }

        if K00_ref_for_freq is not None and K0d_ref_for_freq is not None and Kd0_ref_for_freq is not None:
            omega_same = contact_bands_for_q(
                k_grid=k_grid,
                q_perp=q_perp,
                dt=dt,
                K00_hop=K00_ref_for_freq,
                K0d_hop=K0d_ref_for_freq,
                Kd0_hop=Kd0_ref_for_freq,
                deltas=deltas,
                inv_sqrt_m=inv_sqrt_m,
                freq_factor=freq_factor,
            )
            diff_same = omega_contact - omega_same
            row["same_formalism_rms_direct_thz"] = float(np.sqrt(np.mean(diff_same**2)))
            row["same_formalism_max_abs_direct_thz"] = float(np.max(np.abs(diff_same)))
            same_rms_matched, same_max_matched = matched_frequency_error(omega_contact, omega_same)
            row["same_formalism_rms_matched_thz"] = same_rms_matched
            row["same_formalism_max_abs_matched_thz"] = same_max_matched

        mismatch_rows.append(row)

    rms_direct_values = [row["rms_direct_thz"] for row in mismatch_rows]
    max_direct_values = [row["max_abs_direct_thz"] for row in mismatch_rows]
    rms_matched_values = [row["rms_matched_thz"] for row in mismatch_rows]
    max_matched_values = [row["max_abs_matched_thz"] for row in mismatch_rows]
    summary = {
        "harmonic_yaml": str(yaml_path),
        "transverse_hoppings": str(transverse_path),
        "device_metadata": str(metadata_path),
        "axis": metadata["transport_axis"],
        "axis_scale": float(axis_scale),
        "longitudinal_deltas": [int(value) for value in deltas.tolist()],
        "k_min": float(args.k_min),
        "k_max": float(args.k_max),
        "k_num": int(args.k_num),
        "q_points": [[float(q[0]), float(q[1])] for q in q_points],
        "mismatch": mismatch_rows,
        "same_formalism_reference": same_formalism,
        "global": {
            "mean_rms_direct_thz": float(sum(rms_direct_values) / len(rms_direct_values)),
            "max_rms_direct_thz": float(max(rms_direct_values)),
            "max_abs_direct_thz": float(max(max_direct_values)),
            "mean_rms_matched_thz": float(sum(rms_matched_values) / len(rms_matched_values)),
            "max_rms_matched_thz": float(max(rms_matched_values)),
            "max_abs_matched_thz": float(max(max_matched_values)),
        },
    }

    summary["force_constants_interpretation"] = inspect_force_constants_interpretation(
        harmonic_dir=harmonic_dir,
        phonon=phonon,
        force_constants=force_constants,
    )

    if same_formalism is not None:
        same_direct = [row["same_formalism_rms_direct_thz"] for row in mismatch_rows]
        same_matched = [row["same_formalism_rms_matched_thz"] for row in mismatch_rows]
        same_max_direct = [row["same_formalism_max_abs_direct_thz"] for row in mismatch_rows]
        same_max_matched = [row["same_formalism_max_abs_matched_thz"] for row in mismatch_rows]
        summary["global"]["same_formalism_mean_rms_direct_thz"] = float(
            sum(same_direct) / len(same_direct)
        )
        summary["global"]["same_formalism_max_rms_direct_thz"] = float(max(same_direct))
        summary["global"]["same_formalism_max_abs_direct_thz"] = float(max(same_max_direct))
        summary["global"]["same_formalism_mean_rms_matched_thz"] = float(
            sum(same_matched) / len(same_matched)
        )
        summary["global"]["same_formalism_max_rms_matched_thz"] = float(max(same_matched))
        summary["global"]["same_formalism_max_abs_matched_thz"] = float(max(same_max_matched))

    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved consistency report: {report_path}")
    print(f"mean RMS direct (THz): {summary['global']['mean_rms_direct_thz']:.6f}")
    print(f"max RMS direct (THz): {summary['global']['max_rms_direct_thz']:.6f}")
    print(f"max abs direct (THz): {summary['global']['max_abs_direct_thz']:.6f}")
    print(f"mean RMS matched (THz): {summary['global']['mean_rms_matched_thz']:.6f}")
    print(f"max RMS matched (THz): {summary['global']['max_rms_matched_thz']:.6f}")
    print(f"max abs matched (THz): {summary['global']['max_abs_matched_thz']:.6f}")
    if same_formalism is not None:
        print("Same-formalism tensor comparison:")
        print(
            "  K00 max|d|={:.3e}, relF={:.3e}".format(
                same_formalism["K00"]["max_abs"], same_formalism["K00"]["relative_frobenius"]
            )
        )
        print(
            "  K0d max|d|={:.3e}, relF={:.3e}".format(
                same_formalism["K0d"]["max_abs"], same_formalism["K0d"]["relative_frobenius"]
            )
        )
        print(
            "  Kd0 max|d|={:.3e}, relF={:.3e}".format(
                same_formalism["Kd0"]["max_abs"], same_formalism["Kd0"]["relative_frobenius"]
            )
        )
        print("Same-formalism frequency comparison:")
        print(
            "  mean RMS matched (THz): {:.3e}".format(
                summary["global"]["same_formalism_mean_rms_matched_thz"]
            )
        )
        print(
            "  max RMS matched (THz): {:.3e}".format(
                summary["global"]["same_formalism_max_rms_matched_thz"]
            )
        )
        print(
            "  max abs matched (THz): {:.3e}".format(
                summary["global"]["same_formalism_max_abs_matched_thz"]
            )
        )

    fc_interp = summary.get("force_constants_interpretation", {})
    if fc_interp:
        print("FORCE_CONSTANTS interpretation:")
        print(
            "  storage type: {} | shape={}"
            .format(fc_interp.get("storage_type"), fc_interp.get("force_constants_shape"))
        )
        print(
            "  row labels match p2s_map: {}".format(
                fc_interp.get("file_row_labels_match_p2s_map")
            )
        )

    if args.fail_rms_threshold is not None:
        threshold = float(args.fail_rms_threshold)
        if summary["global"]["max_rms_matched_thz"] > threshold:
            print(
                "Consistency check failed: "
                f"max RMS matched {summary['global']['max_rms_matched_thz']:.6f} > {threshold:.6f}",
                file=sys.stderr,
            )
            return 2

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
