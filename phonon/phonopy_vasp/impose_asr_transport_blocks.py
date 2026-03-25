#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Impose translational acoustic sum rule (ASR) on transport lead blocks by "
            "projecting D(0)=K00+K01+K10 to the subspace orthogonal to rigid translations."
        )
    )
    parser.add_argument("--k00", type=Path, required=True, help="Input K00 .npy path")
    parser.add_argument("--k01", type=Path, required=True, help="Input K01 .npy path")
    parser.add_argument("--k10", type=Path, required=True, help="Input K10 .npy path")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Output prefix. Defaults to <k00_dir>/<k00_stem>_asr with files "
            "<prefix>_K00.npy, <prefix>_K01.npy, <prefix>_K10.npy"
        ),
    )
    parser.add_argument(
        "--keep-k10-consistent",
        action="store_true",
        help="After ASR correction set K10 = K01^T to enforce exact transpose consistency.",
    )
    return parser


def load_block(path: Path):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required.") from exc

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise RuntimeError(f"File not found: {resolved}")
    arr = np.load(resolved)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise RuntimeError(f"Block must be square: {resolved}, got {arr.shape}")
    return arr.astype(float, copy=False), resolved


def translation_basis(num_dof: int):
    import numpy as np

    if num_dof % 3 != 0:
        raise RuntimeError(f"DoF {num_dof} is not divisible by 3.")

    n_atoms = num_dof // 3
    t = np.zeros((num_dof, 3), dtype=float)
    for cart in range(3):
        t[cart::3, cart] = 1.0

    # Orthonormal columns.
    u = t / np.sqrt(float(n_atoms))
    return u


def asr_residual_metrics(d0):
    import numpy as np

    nd = d0.shape[0]
    u = translation_basis(nd)
    right = d0 @ u
    left = u.T @ d0
    row_sum = d0.sum(axis=1)
    return {
        "max_abs_row_sum": float(np.max(np.abs(row_sum))),
        "rms_row_sum": float(np.sqrt(np.mean(row_sum**2))),
        "norm_right_residual": float(np.linalg.norm(right)),
        "norm_left_residual": float(np.linalg.norm(left)),
    }


def impose_asr_on_d0(d0):
    import numpy as np

    nd = d0.shape[0]
    u = translation_basis(nd)
    p = u @ u.T
    q = np.eye(nd, dtype=float) - p
    d0_asr = q @ d0 @ q
    d0_asr = 0.5 * (d0_asr + d0_asr.T)
    return d0_asr


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required.") from exc

    k00, k00_path = load_block(args.k00)
    k01, k01_path = load_block(args.k01)
    k10, k10_path = load_block(args.k10)

    if k00.shape != k01.shape or k00.shape != k10.shape:
        raise RuntimeError(f"Shape mismatch: K00={k00.shape}, K01={k01.shape}, K10={k10.shape}")

    d0 = k00 + k01 + k10
    before = asr_residual_metrics(d0)

    d0_asr = impose_asr_on_d0(d0)
    delta = d0_asr - d0

    k00_new = k00 + delta
    k01_new = k01.copy()
    k10_new = k10.copy()

    if args.keep_k10_consistent:
        k10_new = k01_new.T.copy()
        # Re-adjust K00 to preserve corrected D(0).
        k00_new = d0_asr - k01_new - k10_new

    d0_new = k00_new + k01_new + k10_new
    after = asr_residual_metrics(d0_new)

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = k00_path.with_name(f"{k00_path.stem}_asr")
    output_prefix = output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_k00 = output_prefix.with_name(f"{output_prefix.name}_K00.npy")
    out_k01 = output_prefix.with_name(f"{output_prefix.name}_K01.npy")
    out_k10 = output_prefix.with_name(f"{output_prefix.name}_K10.npy")
    np.save(out_k00, k00_new)
    np.save(out_k01, k01_new)
    np.save(out_k10, k10_new)

    report = {
        "input": {
            "K00": str(k00_path),
            "K01": str(k01_path),
            "K10": str(k10_path),
        },
        "output": {
            "K00": str(out_k00),
            "K01": str(out_k01),
            "K10": str(out_k10),
        },
        "asr_residual_before": before,
        "asr_residual_after": after,
        "delta_k00_max_abs": float(np.max(np.abs(delta))),
        "delta_k00_frobenius": float(np.linalg.norm(delta)),
        "keep_k10_consistent": bool(args.keep_k10_consistent),
    }

    report_path = output_prefix.with_name(f"{output_prefix.name}_asr_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved K00: {out_k00}")
    print(f"Saved K01: {out_k01}")
    print(f"Saved K10: {out_k10}")
    print(f"Saved ASR report: {report_path}")
    print("ASR residual before:")
    print(f"  max abs row sum: {before['max_abs_row_sum']:.6e}")
    print(f"  rms row sum    : {before['rms_row_sum']:.6e}")
    print(f"  ||D0 U||       : {before['norm_right_residual']:.6e}")
    print(f"  ||U^T D0||     : {before['norm_left_residual']:.6e}")
    print("ASR residual after:")
    print(f"  max abs row sum: {after['max_abs_row_sum']:.6e}")
    print(f"  rms row sum    : {after['rms_row_sum']:.6e}")
    print(f"  ||D0 U||       : {after['norm_right_residual']:.6e}")
    print(f"  ||U^T D0||     : {after['norm_left_residual']:.6e}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
