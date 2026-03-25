#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline: construct device FC matrix, extract q-grid transport blocks, "
            "and run Sancho-Rubio transport for each transverse q point."
        )
    )
    parser.add_argument("--harmonic-dir", type=Path, required=True)
    parser.add_argument("--device-axis", choices=("a", "b", "c"), required=True)
    parser.add_argument("--device-cells", type=int, required=True)
    parser.add_argument(
        "--device-matrix",
        type=Path,
        default=None,
        help="Output device matrix path. Default: <harmonic-dir>/device_fc_matrix_<axis>_<cells>.npy",
    )
    parser.add_argument(
        "--no-device-asr",
        action="store_true",
        help="Disable integrated ASR in device matrix construction.",
    )

    parser.add_argument("--q-grid", type=int, nargs=2, required=True, metavar=("NQ1", "NQ2"))
    parser.add_argument("--gamma-centered", action="store_true")
    parser.add_argument("--max-neighbor", type=int, default=1)
    parser.add_argument(
        "--transport-prefix",
        type=Path,
        default=None,
        help="Prefix for transport blocks. Default: <harmonic-dir>/transport_blocks_<device-matrix-stem>",
    )

    parser.add_argument("--omega-min", type=float, default=0.01)
    parser.add_argument("--omega-max", type=float, default=20.0)
    parser.add_argument("--omega-num", type=int, default=400)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max-iter", type=int, default=300)

    parser.add_argument(
        "--spectra-dir",
        type=Path,
        default=None,
        help="Output directory for q-resolved spectra. Default: <harmonic-dir>/qgrid_transport_spectra",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Output summary JSON path. Default: <spectra-dir>/pipeline_qgrid_transport_summary.json",
    )
    return parser


def run_cmd(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=str(cwd), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def build_python_cmd(python_exe: str, script_name: str, *args: str) -> list[str]:
    return [python_exe, script_name, *args]


def q_label(q: list[float]) -> str:
    def p(value: float) -> str:
        return f"{value:+.6f}".replace("+", "p").replace("-", "m").replace(".", "d")

    return f"q_{p(float(q[0]))}_{p(float(q[1]))}"


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    harmonic_dir = args.harmonic_dir.expanduser().resolve()
    if not harmonic_dir.exists():
        raise RuntimeError(f"Harmonic dir not found: {harmonic_dir}")
    if args.device_cells < 2:
        raise RuntimeError("--device-cells must be >= 2")
    if args.max_neighbor < 1:
        raise RuntimeError("--max-neighbor must be >= 1")

    workspace = Path(__file__).resolve().parent
    py = sys.executable

    device_matrix = args.device_matrix
    if device_matrix is None:
        device_matrix = harmonic_dir / f"device_fc_matrix_{args.device_axis}_{args.device_cells}.npy"
    device_matrix = device_matrix.expanduser().resolve()
    device_metadata = device_matrix.with_name(f"{device_matrix.stem}_metadata.json")

    transport_prefix = args.transport_prefix
    if transport_prefix is None:
        transport_prefix = harmonic_dir / f"transport_blocks_{device_matrix.stem}"
    transport_prefix = transport_prefix.expanduser().resolve()
    transport_summary = transport_prefix.with_name(f"{transport_prefix.name}_summary.json")

    spectra_dir = args.spectra_dir
    if spectra_dir is None:
        spectra_dir = harmonic_dir / "qgrid_transport_spectra"
    spectra_dir = spectra_dir.expanduser().resolve()
    spectra_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.summary
    if summary_path is None:
        summary_path = spectra_dir / "pipeline_qgrid_transport_summary.json"
    summary_path = summary_path.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Device FC construction (ASR integrated by default in extract_harmonic_fc.py)
    cmd_device = build_python_cmd(
        py,
        "extract_harmonic_fc.py",
        "--harmonic-dir",
        str(harmonic_dir),
        "--device-axis",
        args.device_axis,
        "--device-cells",
        str(args.device_cells),
        "--device-output",
        str(device_matrix),
        "--device-format",
        "npy",
    )
    if args.no_device_asr:
        cmd_device.append("--device-no-asr")
    run_cmd(cmd_device, workspace)

    # 2) Extract blocks + q-grid q-resolved K blocks
    cmd_blocks = build_python_cmd(
        py,
        "extract_transport_blocks.py",
        "--device-matrix",
        str(device_matrix),
        "--metadata",
        str(device_metadata),
        "--output-prefix",
        str(transport_prefix),
        "--max-neighbor",
        str(args.max_neighbor),
        "--export-transverse-hoppings",
        "--q-grid",
        str(int(args.q_grid[0])),
        str(int(args.q_grid[1])),
    )
    if args.gamma_centered:
        cmd_blocks.append("--gamma-centered")
    run_cmd(cmd_blocks, workspace)

    if not transport_summary.exists():
        raise RuntimeError(f"Transport summary not found: {transport_summary}")
    transport_info = json.loads(transport_summary.read_text(encoding="utf-8"))
    q_blocks = transport_info.get("q_resolved_blocks", [])
    if not q_blocks:
        raise RuntimeError("No q-resolved blocks found in transport summary.")

    # 3) Run Sancho-Rubio transport for each q block
    spectra = []
    for item in q_blocks:
        q = [float(item["q"][0]), float(item["q"][1])]
        label = q_label(q)
        out_npz = spectra_dir / f"transport_{label}.npz"

        cmd_tr = build_python_cmd(
            py,
            "phonon_transport_sancho_rubio.py",
            "--k00",
            str(item["K00"]),
            "--k01",
            str(item["K01"]),
            "--k10",
            str(item["K10"]),
            "--device-cells",
            str(args.device_cells),
            "--omega-min",
            str(args.omega_min),
            "--omega-max",
            str(args.omega_max),
            "--omega-num",
            str(args.omega_num),
            "--eta",
            str(args.eta),
            "--tol",
            str(args.tol),
            "--max-iter",
            str(args.max_iter),
            "--output",
            str(out_npz),
        )
        run_cmd(cmd_tr, workspace)

        spectra.append(
            {
                "q": q,
                "label": label,
                "K00": str(item["K00"]),
                "K01": str(item["K01"]),
                "K10": str(item["K10"]),
                "spectrum": str(out_npz),
                "metadata": str(out_npz.with_name(f"{out_npz.stem}_metadata.json")),
            }
        )

    summary = {
        "harmonic_dir": str(harmonic_dir),
        "device_axis": args.device_axis,
        "device_cells": int(args.device_cells),
        "device_asr_enabled": not bool(args.no_device_asr),
        "device_matrix": str(device_matrix),
        "device_metadata": str(device_metadata),
        "transport_prefix": str(transport_prefix),
        "transport_summary": str(transport_summary),
        "q_grid": [int(args.q_grid[0]), int(args.q_grid[1])],
        "gamma_centered": bool(args.gamma_centered),
        "max_neighbor": int(args.max_neighbor),
        "omega_min": float(args.omega_min),
        "omega_max": float(args.omega_max),
        "omega_num": int(args.omega_num),
        "eta": float(args.eta),
        "tol": float(args.tol),
        "max_iter": int(args.max_iter),
        "spectra_dir": str(spectra_dir),
        "spectra": spectra,
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Device matrix: {device_matrix}")
    print(f"Transport summary: {transport_summary}")
    print(f"q-resolved spectra: {len(spectra)}")
    print(f"Saved pipeline summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
