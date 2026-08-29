#!/usr/bin/env python3
"""Compare a primitive-factor Si run with a dense primitive oracle.

The comparison is deliberately made on saved production arrays.  It therefore
includes q folding, frequency normalisation, the primitive-to-grouped mapping
and the retarded reconstruction, rather than testing the FC3 fit in isolation.

Examples
--------
::

    python phonon/scripts/si_micro_gate.py \
        --reference-run cluster/si-l5-q9-dense-in/probe.npz \
        --candidate-run cluster/si-l5-q9-r128-in/probe.npz \
        --reference-sigma cluster/si-l5-q9-dense-in/sigma.npz \
        --candidate-sigma cluster/si-l5-q9-r128-in/sigma.npz \
        --json-out phonon/scripts/data/si_l5_q9_r128_gate.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SIGMA_KEYS = ("sigma_lesser", "sigma_greater", "sigma_retarded")


def relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return the Frobenius error relative to the dense reference."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"array shapes differ: {reference.shape} and {candidate.shape}"
        )
    scale = np.linalg.norm(reference.ravel())
    error = np.linalg.norm((candidate - reference).ravel())
    return float(error / max(float(scale), np.finfo(float).tiny))


def antihermiticity_defect(array: np.ndarray) -> float:
    """Return ``||A+A^H||/||A||`` for flattened square matrix blocks."""
    array = np.asarray(array)
    width = int(round(np.sqrt(array.shape[-1])))
    if width * width != array.shape[-1]:
        raise ValueError(f"last dimension {array.shape[-1]} is not square")
    matrix = array.reshape(*array.shape[:-1], width, width)
    defect = matrix + np.swapaxes(matrix.conj(), -1, -2)
    return float(
        np.linalg.norm(defect.ravel())
        / max(float(np.linalg.norm(matrix.ravel())), np.finfo(float).tiny)
    )


def negativity_fraction(array: np.ndarray) -> float:
    """Measure the negative weight of the phonon ``-i Sigma`` source.

    Quatrex uses the convention in which both saved lesser and greater
    phonon sources should give positive semidefinite ``-i Sigma``.  Imperfect
    frozen references are not projected.  The factor gate compares this
    diagnostic with the dense value and reports only the increase as a defect.
    """
    array = np.asarray(array)
    width = int(round(np.sqrt(array.shape[-1])))
    if width * width != array.shape[-1]:
        raise ValueError(f"last dimension {array.shape[-1]} is not square")
    matrix = array.reshape(-1, width, width)
    negative = 0.0
    total = 0.0
    for block in matrix:
        source = -1j * block
        source = 0.5 * (source + source.conj().T)
        values = np.linalg.eigvalsh(source)
        negative += float(np.maximum(-values, 0.0).sum())
        total += float(np.abs(values).sum())
    return negative / max(total, np.finfo(float).tiny)


def _scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    return value.item() if value.size == 1 else value


def _relative_scalar(reference: float | None, candidate: float | None):
    if reference is None or candidate is None:
        return None
    return abs(float(candidate) - float(reference)) / max(
        abs(float(reference)), np.finfo(float).tiny
    )


def _spectral_current_error(reference, candidate):
    if "current_spectrum" not in reference.files:
        return None
    if "current_spectrum" not in candidate.files:
        return None
    return relative_l1(
        np.asarray(reference["current_spectrum"]),
        np.asarray(candidate["current_spectrum"]),
    )


def relative_l1(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return the elementwise L1 difference normalised by reference weight."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"array shapes differ: {reference.shape} and {candidate.shape}"
        )
    return float(
        np.abs(candidate - reference).sum()
        / max(float(np.abs(reference).sum()), np.finfo(float).tiny)
    )


def compare(
    reference_run: Path,
    candidate_run: Path,
    reference_sigma: Path,
    candidate_sigma: Path,
) -> dict:
    """Return all certification metrics for one factor candidate."""
    with (
        np.load(reference_run, allow_pickle=True) as ref_run,
        np.load(candidate_run, allow_pickle=True) as cand_run,
        np.load(reference_sigma) as ref_sigma,
        np.load(candidate_sigma) as cand_sigma,
    ):
        metrics: dict[str, object] = {
            "reference_run": str(reference_run),
            "candidate_run": str(candidate_run),
            "reference_sigma": str(reference_sigma),
            "candidate_sigma": str(candidate_sigma),
            "reference_vertex": _scalar(ref_run, "vertex_representation"),
            "candidate_vertex": _scalar(cand_run, "vertex_representation"),
            "candidate_rank": _scalar(cand_run, "sse_vertex_rank"),
            "reference_converged": bool(_scalar(
                ref_run, "converged", False)),
            "candidate_converged": bool(_scalar(
                cand_run, "converged", False)),
            "reference_iterations": _scalar(ref_run, "n_iter"),
            "candidate_iterations": _scalar(cand_run, "n_iter"),
            "reference_lead_current": _scalar(ref_run, "lead_current"),
            "candidate_lead_current": _scalar(cand_run, "lead_current"),
        }
        current_gate = bool(
            metrics["reference_converged"] and metrics["candidate_converged"]
        )
        metrics["current_gate_applicable"] = current_gate
        metrics["lead_current_relative_error"] = (
            _relative_scalar(
                metrics["reference_lead_current"],
                metrics["candidate_lead_current"],
            ) if current_gate else None
        )
        metrics["spectral_current_l1_error"] = (
            _spectral_current_error(ref_run, cand_run)
            if current_gate else None
        )
        for key in SIGMA_KEYS:
            metrics[f"{key}_relative_error"] = relative_error(
                ref_sigma[key], cand_sigma[key]
            )
        for label, sigma in (("reference", ref_sigma), ("candidate", cand_sigma)):
            for key in ("sigma_lesser", "sigma_greater"):
                metrics[f"{label}_{key}_antihermiticity"] = (
                    antihermiticity_defect(sigma[key])
                )
                metrics[f"{label}_{key}_negativity"] = negativity_fraction(
                    sigma[key]
                )
        for key in ("sigma_lesser", "sigma_greater"):
            metrics[f"additional_{key}_negativity"] = max(
                0.0,
                float(metrics[f"candidate_{key}_negativity"])
                - float(metrics[f"reference_{key}_negativity"]),
            )
        return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--reference-sigma", type=Path, required=True)
    parser.add_argument("--candidate-sigma", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    metrics = compare(
        args.reference_run,
        args.candidate_run,
        args.reference_sigma,
        args.candidate_sigma,
    )
    encoded = json.dumps(metrics, indent=2, sort_keys=True)
    print(encoded)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
