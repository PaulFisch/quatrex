"""Spatial compression gate on the frozen converged long Si-film Sigma.

This is intentionally a snapshot study, not an SCBA modification.  The L8
production archive stores the full 48 x 48 self-energy for every frequency and
transverse q.  We decode its deterministic DSDB full-block ordering, retain the
primitive nearest-cell band exactly, and compress only the residual sibling
blocks of a binary HODLR tree.  The comparison includes the exact support that
a two-cell reblock would put into a block-tridiagonal Dyson operator.

Run on an Alps compute node (the compressed snapshot is about 4.3 GB)::

    QTX_ARRAY_MODULE=numpy PYTHONPATH=src \
      python phonon/studies/_si_long_film_spatial_gate.py \
      --sigma cluster/sifilm8s/sigma_best.rank0.npz \
      --pole-states cluster/si-aux-l8b/poles.npz \
      --output cluster/si-long-spatial/spatial_gate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def decode_full_blocks(values: np.ndarray, n_cells: int, cell_dof: int):
    """DSDBCOO full pattern -> ordinary dense matrix."""
    values = np.asarray(values)
    expected = (n_cells * cell_dof) ** 2
    if values.size != expected:
        raise ValueError(
            f"snapshot has {values.size} nnz, expected full {expected}")
    blocks = values.reshape(n_cells, n_cells, cell_dof, cell_dof)
    return blocks.transpose(0, 2, 1, 3).reshape(
        n_cells * cell_dof, n_cells * cell_dof)


def cell_mask(n_cells: int, cell_dof: int, predicate) -> np.ndarray:
    mask = np.zeros((n_cells * cell_dof,) * 2, dtype=bool)
    for i in range(n_cells):
        si = slice(i * cell_dof, (i + 1) * cell_dof)
        for j in range(n_cells):
            if predicate(i, j):
                sj = slice(j * cell_dof, (j + 1) * cell_dof)
                mask[si, sj] = True
    return mask


def hodlr_residual(carrier: np.ndarray, n_cells: int, cell_dof: int,
                   tol: float):
    """Exact near band plus binary sibling SVDs of its residual."""
    near = cell_mask(n_cells, cell_dof, lambda i, j: abs(i - j) <= 1)
    approximation = np.where(near, carrier, 0.0)
    residual = np.where(near, 0.0, carrier)
    levels: dict[int, list[int]] = {}
    factors = []

    def visit(lo: int, hi: int, level: int):
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        left = slice(lo * cell_dof, mid * cell_dof)
        right = slice(mid * cell_dof, hi * cell_dof)
        block = residual[left, right]
        u, s, vh = np.linalg.svd(block, full_matrices=False)
        if s.size and s[0] > 0.0:
            sq = np.cumsum((s * s)[::-1])[::-1]
            total = float(sq[0])
            rank = s.size
            for r in range(s.size + 1):
                rem = 0.0 if r == s.size else float(sq[r])
                if np.sqrt(rem / total) <= tol:
                    rank = r
                    break
        else:
            rank = 0
        levels.setdefault(level, []).append(int(rank))
        if rank:
            upper = (u[:, :rank] * s[:rank]) @ vh[:rank]
            approximation[left, right] = upper
            approximation[right, left] = upper.conj().T
            factors.append((upper.shape[0], upper.shape[1], rank))
        visit(lo, mid, level + 1)
        visit(mid, hi, level + 1)

    visit(0, n_cells, 0)
    # A truncated sibling SVD can leak into the zeroed boundary entries that
    # belong to the primitive near band.  In the operator representation those
    # entries are sparse corrections to the low-rank factors; restoring them
    # here makes the advertised exact near field literal.
    approximation[near] = carrier[near]
    # Symmetry-aware storage: one triangle of the exact near band, plus one
    # U/V pair per upper sibling block.  Also report the conservative explicit
    # two-direction count used by a generic non-symmetric implementation.
    near_upper = int(np.count_nonzero(np.triu(near)))
    factor_upper = int(sum((m + n) * r for m, n, r in factors))
    return approximation, levels, {
        "symmetry_aware": near_upper + factor_upper,
        "explicit_directions": int(np.count_nonzero(near)) + 2 * factor_upper,
    }


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def analyse_matrix(sigma: np.ndarray, n_cells: int, cell_dof: int):
    # Quatrex convention Sigma^< = +i C, C >= 0.
    carrier = -1j * sigma
    anti = _rel(sigma.conj().T, -sigma)
    carrier = 0.5 * (carrier + carrier.conj().T)
    scale = max(float(np.linalg.norm(carrier, 2)), 1e-300)
    ref_floor = float(np.linalg.eigvalsh(carrier)[0] / scale)
    near = cell_mask(n_cells, cell_dof, lambda i, j: abs(i - j) <= 1)
    reblock = cell_mask(
        n_cells, cell_dof, lambda i, j: abs(i // 2 - j // 2) <= 1)
    out = {
        "antihermiticity_error": anti,
        "reference_psd_floor": ref_floor,
        "hard_band1_discarded_frobenius": _rel(
            np.where(near, carrier, 0.0), carrier),
        "two_cell_reblock_discarded_frobenius": _rel(
            np.where(reblock, carrier, 0.0), carrier),
        "shell_fraction": {},
        "storage": {
            "dense": int(carrier.size),
            "hard_band1_explicit": int(np.count_nonzero(near)),
            "two_cell_reblock_explicit": int(np.count_nonzero(reblock)),
        },
        "hodlr": {},
    }
    norm = max(float(np.linalg.norm(carrier)), 1e-300)
    for distance in range(n_cells):
        shell = cell_mask(
            n_cells, cell_dof, lambda i, j, d=distance: abs(i - j) == d)
        out["shell_fraction"][str(distance)] = float(
            np.linalg.norm(np.where(shell, carrier, 0.0)) / norm)
    for tol in (1e-2, 1e-3, 1e-4):
        approx, levels, storage = hodlr_residual(
            carrier, n_cells, cell_dof, tol)
        floor = float(np.linalg.eigvalsh(0.5 * (approx + approx.conj().T))[0]
                      / scale)
        out["hodlr"][str(tol)] = {
            "operator_error": _rel(approx, carrier),
            "rank_by_level": {str(k): v for k, v in levels.items()},
            "storage": storage,
            "storage_over_reblock": {
                key: float(value / max(np.count_nonzero(reblock), 1))
                for key, value in storage.items()
            },
            "additional_normalised_negativity": min(0.0, floor - ref_floor),
        }
    far = np.where(near, 0.0, carrier)
    s = np.linalg.svd(far, compute_uv=False)
    ranks = {}
    if s.size and s[0] > 0.0:
        tail = np.cumsum((s * s)[::-1])[::-1]
        for tol in (1e-2, 1e-3, 1e-4):
            ranks[str(tol)] = next(
                (r for r in range(s.size + 1)
                 if np.sqrt((0.0 if r == s.size else tail[r]) / tail[0]) <= tol),
                s.size)
    out["global_far_rank"] = ranks
    return out


def choose_cases(data: np.ndarray, frequencies: np.ndarray,
                 pole_path: str | None):
    cases = []
    # Off-resonant/high-frequency sample.
    cases.append(("off_resonant", int(np.argmin(abs(frequencies - 14.5))), 0, 0))
    # Broad/large-weight sample without materialising another full-size array.
    best = (-1.0, 0, 0, 0)
    for iw in range(1, data.shape[0]):
        norms = np.linalg.norm(data[iw].reshape(-1, data.shape[-1]), axis=1)
        iq = int(np.argmax(norms))
        if float(norms[iq]) > best[0]:
            q = np.unravel_index(iq, data.shape[1:-1])
            best = (float(norms[iq]), iw, int(q[0]), int(q[1]))
    cases.append(("broad_max", best[1], best[2], best[3]))
    if pole_path:
        poles = np.load(pole_path, allow_pickle=False)
        z = np.asarray(poles["poles"], dtype=complex)
        po = np.asarray(poles["pole_offsets"], dtype=int)
        qi = np.asarray(poles["q_index"], dtype=int)
        candidates = []
        for ic in range(po.size - 1):
            for zp in z[po[ic]:po[ic + 1]]:
                if zp.real > 0.3:
                    candidates.append((abs(zp.imag), zp.real, *qi[ic]))
        if candidates:
            _gamma, centre, q0, q1 = min(candidates)
            cases.append(("narrow_pole", int(np.argmin(abs(frequencies - centre))),
                          int(q0), int(q1)))
    # Stable order, no duplicate matrix.
    unique = []
    seen = set()
    for case in cases:
        key = case[1:]
        if key not in seen:
            unique.append(case); seen.add(key)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", required=True)
    parser.add_argument("--pole-states")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-cells", type=int, default=8)
    parser.add_argument("--cell-dof", type=int, default=6)
    parser.add_argument("--wmax", type=float, default=15.0)
    args = parser.parse_args()
    archive = np.load(args.sigma, allow_pickle=False)
    lesser = np.asarray(archive["sigma_lesser"])
    frequencies = np.linspace(0.0, args.wmax, lesser.shape[0])
    report = {"shape": list(lesser.shape), "cases": {}}
    for label, iw, q0, q1 in choose_cases(
            lesser, frequencies, args.pole_states):
        sigma = decode_full_blocks(
            lesser[iw, q0, q1], args.n_cells, args.cell_dof)
        row = analyse_matrix(sigma, args.n_cells, args.cell_dof)
        row.update({"frequency_index": iw, "frequency_thz": frequencies[iw],
                    "q_index": [q0, q1]})
        report["cases"][label] = row
        print(label, json.dumps(row, indent=2), flush=True)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"WROTE {target}")


if __name__ == "__main__":
    main()
