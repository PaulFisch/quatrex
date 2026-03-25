#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay contact lead bands with bulk projected bands along the same transport "
            "direction at selected transverse q points."
        )
    )
    parser.add_argument(
        "--harmonic-dir",
        type=Path,
        required=True,
        help="Directory containing phonopy YAML and FORCE_CONSTANTS/FORCE_SETS.",
    )
    parser.add_argument(
        "--transverse-hoppings",
        type=Path,
        required=True,
        help="NPZ from extract_transport_blocks.py --export-transverse-hoppings.",
    )
    parser.add_argument(
        "--device-metadata",
        type=Path,
        required=True,
        help="Device metadata JSON from extract_harmonic_fc.py.",
    )
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
    parser.add_argument("--k-min", type=float, default=-0.5, help="Minimum lead reduced k.")
    parser.add_argument("--k-max", type=float, default=0.5, help="Maximum lead reduced k.")
    parser.add_argument("--k-num", type=int, default=201, help="Number of k samples.")
    parser.add_argument(
        "--bulk-axis-scale",
        type=float,
        default=None,
        help=(
            "Scale factor mapping lead k to primitive bulk q on transport axis. "
            "Default is 1/repeat_axis."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("contact_vs_bulk_projected.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path for per-q mismatch metrics.",
    )
    return parser


def load_phonon(harmonic_dir: Path):
    try:
        import phonopy
    except ImportError as exc:
        raise RuntimeError("phonopy is required.") from exc

    harmonic_dir = harmonic_dir.expanduser().resolve()
    yaml_candidates = [harmonic_dir / "phonopy_disp.yaml", harmonic_dir / "phonopy.yaml"]
    yaml_path = next((path for path in yaml_candidates if path.exists()), None)
    if yaml_path is None:
        raise RuntimeError(f"Missing phonopy yaml in {harmonic_dir}")

    fc = harmonic_dir / "FORCE_CONSTANTS"
    fs = harmonic_dir / "FORCE_SETS"
    if fc.exists():
        phonon = phonopy.load(phonopy_yaml=str(yaml_path), force_constants_filename=str(fc))
    elif fs.exists():
        phonon = phonopy.load(phonopy_yaml=str(yaml_path), force_sets_filename=str(fs))
    else:
        phonon = phonopy.load(phonopy_yaml=str(yaml_path))

    return phonon, yaml_path


def build_q_points(args):
    q_points = [(float(q[0]), float(q[1])) for q in (args.q_point or [])]
    if args.q_grid is not None:
        nq1, nq2 = int(args.q_grid[0]), int(args.q_grid[1])
        if nq1 <= 0 or nq2 <= 0:
            raise RuntimeError("--q-grid values must be positive.")
        for i in range(nq1):
            for j in range(nq2):
                if args.gamma_centered:
                    q1 = (i + 0.5) / nq1 - 0.5
                    q2 = (j + 0.5) / nq2 - 0.5
                else:
                    q1 = i / nq1
                    q2 = j / nq2
                q_points.append((q1, q2))

    if not q_points:
        q_points = [(0.0, 0.0)]

    seen = set()
    unique = []
    for q1, q2 in q_points:
        key = (round(float(q1), 12), round(float(q2), 12))
        if key in seen:
            continue
        seen.add(key)
        unique.append((float(q1), float(q2)))
    return unique


def reconstruct_basis_masses(device_metadata: dict, primitive_masses, axis_index: int):
    order = device_metadata.get("transport_atom_order")
    if not isinstance(order, list):
        raise RuntimeError("device metadata missing transport_atom_order")

    transverse_dirs = [d for d in (0, 1, 2) if d != axis_index]
    grouped: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {}
    for atom_idx, atom in enumerate(order):
        offset = atom.get("cell_offset")
        if not isinstance(offset, list) or len(offset) != 3:
            raise RuntimeError("invalid cell_offset in metadata")
        primitive_index = int(atom["primitive_atom_index"])
        trans_key = (int(offset[transverse_dirs[0]]), int(offset[transverse_dirs[1]]))
        basis_key = (int(offset[axis_index]), primitive_index)
        grouped.setdefault(trans_key, []).append((basis_key, atom_idx))

    ref_key = (0, 0) if (0, 0) in grouped else sorted(grouped.keys())[0]
    reference_basis = sorted(grouped[ref_key], key=lambda item: item[0])
    basis_order = [item[0] for item in reference_basis]

    masses = []
    for _, primitive_index_1based in basis_order:
        m = float(primitive_masses[primitive_index_1based - 1])
        masses.extend([m, m, m])

    return masses, transverse_dirs


def contact_bands_for_q(
    k_grid,
    q_perp,
    dt,
    K00_hop,
    K0d_hop,
    Kd0_hop,
    deltas,
    inv_sqrt_m,
    freq_factor,
):
    import numpy as np

    q1, q2 = q_perp
    phase_per_hop = np.exp(2j * np.pi * (q1 * dt[:, 0] + q2 * dt[:, 1]))
    K00q = np.tensordot(phase_per_hop, K00_hop, axes=(0, 0))
    K0d_q = np.tensordot(phase_per_hop, K0d_hop, axes=(0, 1))
    Kd0_q = np.tensordot(phase_per_hop, Kd0_hop, axes=(0, 1))

    n_mode = K00q.shape[0]
    omega = np.zeros((len(k_grid), n_mode), dtype=float)

    for idx, k in enumerate(k_grid):
        phase_k = np.exp(2j * np.pi * k)
        D = K00q.copy()
        for index, delta in enumerate(deltas):
            D = D + K0d_q[index] * (phase_k ** int(delta))
            D = D + Kd0_q[index] * (phase_k.conjugate() ** int(delta))
        D = 0.5 * (D + D.conjugate().T)
        D = inv_sqrt_m @ D @ inv_sqrt_m
        eigvals = np.linalg.eigvalsh(D)
        eigvals = np.maximum(eigvals.real, 0.0)
        omega[idx, :] = np.sqrt(eigvals) * freq_factor

    return omega


def bulk_projected_bands(
    phonon,
    axis_index: int,
    transverse_dirs,
    q_perp,
    k_grid,
    axis_scale,
    transport_repeat: int,
):
    import numpy as np

    if transport_repeat < 1:
        raise RuntimeError("transport_repeat must be >= 1")

    q1, q2 = q_perp
    folded_sets = []
    for fold in range(transport_repeat):
        q_points = []
        for k in k_grid:
            q = [0.0, 0.0, 0.0]
            q[axis_index] = float(k * axis_scale + fold / transport_repeat)
            q[transverse_dirs[0]] = float(q1)
            q[transverse_dirs[1]] = float(q2)
            q_points.append(q)

        phonon.run_qpoints(q_points, with_eigenvectors=False)
        qdict = phonon.get_qpoints_dict()
        folded_sets.append(np.asarray(qdict["frequencies"], dtype=float))

    freqs = np.concatenate(folded_sets, axis=1)
    freqs = np.sort(freqs, axis=1)
    return np.maximum(freqs, 0.0)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib and numpy are required.") from exc

    if args.k_num < 2:
        raise RuntimeError("--k-num must be at least 2")
    if args.k_max <= args.k_min:
        raise RuntimeError("--k-max must be greater than --k-min")

    phonon, yaml_path = load_phonon(args.harmonic_dir)
    primitive = phonon.primitive

    transverse_path = args.transverse_hoppings.expanduser().resolve()
    if not transverse_path.exists():
        raise RuntimeError(f"Transverse hopping file not found: {transverse_path}")
    hop = np.load(transverse_path)
    required = ["dt", "K00", "K01", "K10"]
    dt = np.asarray(hop["dt"], dtype=float)
    K00_hop = np.asarray(hop["K00"], dtype=complex)
    if all(name in hop for name in ["deltas", "K0d", "Kd0"]):
        deltas = np.asarray(hop["deltas"], dtype=int)
        K0d_hop = np.asarray(hop["K0d"], dtype=complex)
        Kd0_hop = np.asarray(hop["Kd0"], dtype=complex)
    else:
        for name in required:
            if name not in hop:
                raise RuntimeError(f"Missing array {name} in {transverse_path}")
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

    nplot = len(q_points)
    ncol = 2 if nplot > 1 else 1
    nrow = math.ceil(nplot / ncol)

    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.5 * nrow), dpi=150, squeeze=False)
    mismatch_report = []

    for idx, q_perp in enumerate(q_points):
        row, col = divmod(idx, ncol)
        ax = axes[row][col]

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

        for b in range(omega_contact.shape[1]):
            ax.plot(k_grid, omega_contact[:, b], color="tab:blue", linewidth=0.8, alpha=0.9)
        for b in range(omega_bulk.shape[1]):
            ax.plot(k_grid, omega_bulk[:, b], color="tab:orange", linewidth=1.0, linestyle="--", alpha=0.9)

        diff = omega_contact - omega_bulk
        rms = float(np.sqrt(np.mean(diff**2)))
        max_abs = float(np.max(np.abs(diff)))
        mismatch_report.append(
            {
                "q_perp": [float(q_perp[0]), float(q_perp[1])],
                "rms_thz": rms,
                "max_abs_thz": max_abs,
            }
        )

        ax.set_title(
            f"q⊥=({q_perp[0]:.3f}, {q_perp[1]:.3f})\n"
            f"RMS={rms:.3f} THz, Max={max_abs:.3f} THz"
        )
        ax.set_xlabel("Lead reduced k")
        ax.set_ylabel("Frequency (THz)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        if idx == 0:
            ax.plot([], [], color="tab:blue", label="Contact bands (from K blocks)")
            ax.plot([], [], color="tab:orange", linestyle="--", label="Bulk projected bands")
            ax.legend(loc="best", fontsize=8)

    for idx in range(nplot, nrow * ncol):
        row, col = divmod(idx, ncol)
        axes[row][col].axis("off")

    fig.suptitle(
        "Contact vs bulk-projected bands\n"
        f"axis={metadata['transport_axis']}, bulk axis scale={axis_scale:.6f}",
        fontsize=11,
    )
    fig.tight_layout()

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    report_path = args.report
    if report_path is None:
        report_path = output_path.with_name(f"{output_path.stem}_mismatch_report.json")
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "harmonic_yaml": str(yaml_path),
                "transverse_hoppings": str(transverse_path),
                "device_metadata": str(metadata_path),
                "axis": metadata["transport_axis"],
                "axis_scale": float(axis_scale),
                "longitudinal_deltas": [int(value) for value in deltas.tolist()],
                "q_points": [[float(q[0]), float(q[1])] for q in q_points],
                "mismatch": mismatch_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Harmonic yaml: {yaml_path}")
    print(f"Transverse hoppings: {transverse_path}")
    print(f"Device metadata: {metadata_path}")
    print(f"q points: {q_points}")
    print(f"Saved comparison plot: {output_path}")
    print(f"Saved mismatch report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
