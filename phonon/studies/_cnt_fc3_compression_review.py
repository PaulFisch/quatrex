"""Finite-CNT FC3 compression audit used by the acceleration review.

This is deliberately a study, not a production factor generator.  It reads
the committed four-cell CNT(3,3) device vertex, extracts one translationally
representative primitive-cell tensor over offsets ``(-1, 0, 1)``, and records
the structure which a useful factorisation must retain:

* exact offset-pair support and acoustic/permutation identities;
* atom-triplet sparsity of every occupied 36^3 block;
* mode-unfolding ranks of the on-site and two directed cross-cell orbits;
* an S2-preserving support-block Tucker diagnostic, including its global ASR
  defect rather than a post-hoc projection;
* optional, separately fitted INDSCAL ranks of the *global* offset tensor.

The last item reproduces the old flat-CP question on the actual CNT vertex.
It is intentionally opt-in because ranks through 256 take a few minutes on a
desktop CPU.  Each rank is fitted independently: CP/INDSCAL fits are not a
nested basis and truncating one high-rank fit is not the same experiment.

Examples
--------
Fast structural audit::

    PYTHONPATH=src:phonon python phonon/studies/_cnt_fc3_compression_review.py

Full global-rank audit::

    PYTHONPATH=src:phonon python phonon/studies/_cnt_fc3_compression_review.py \
        --fit-ranks 16,32,64,96,128,160,192,256 --json /tmp/cnt_fc3.json

Restart-rich finite-CNT audit::

    PYTHONPATH=src:phonon python phonon/studies/_cnt_fc3_compression_review.py \
        --fit-ranks 64,128,256 --restarts 2 --max-iter 250 \
        --lbfgs-iters 150 --seed-base 1729
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERTEX = (
    ROOT / "phonon/studies/out/anderson_test/cnt33_L4_inputs/fc3_blocks.hdf5"
)
DEFAULT_D11A = (
    ROOT / "phonon/configs/sinw/reaps/sinw100_d11a_vasp_sc4"
    / "transport_quality/transport_quality.csv"
)


def load_blocks(path: str | Path) -> tuple[dict[tuple[int, int, int], np.ndarray], np.ndarray]:
    """Load the production FC3 HDF5 schema without touching the file."""
    import h5py

    blocks: dict[tuple[int, int, int], np.ndarray] = {}
    with h5py.File(path, "r") as h5:
        sizes = np.asarray(h5["meta/block_sizes"], dtype=int)
        for name in h5["fc3_blocks"]:
            ds = h5["fc3_blocks"][name]
            key = tuple(int(ds.attrs[x]) for x in ("I", "J", "K"))
            value = np.asarray(ds)
            if np.max(np.abs(value.imag), initial=0.0) > 1e-13 * max(
                np.max(np.abs(value.real), initial=0.0), 1.0
            ):
                raise ValueError(f"{path}: complex FC3 block {key}")
            blocks[key] = value.real
    return blocks, sizes


def representative_offset_tensor(
    blocks: dict[tuple[int, int, int], np.ndarray],
    sizes: np.ndarray,
    offsets: tuple[int, ...] = (-1, 0, 1),
) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    """Return ``T[external, offset/internal, offset/internal]``.

    The external cell is chosen away from the finite-device boundary.  Missing
    offset pairs are exact structural zeros, not discarded small entries.
    """
    if len(set(int(x) for x in sizes)) != 1:
        raise ValueError("the audit expects uniform primitive block sizes")
    b = int(sizes[0])
    candidates = [
        i for i in range(len(sizes))
        if all(0 <= i + d < len(sizes) for d in offsets)
    ]
    if not candidates:
        raise ValueError("device has no interior cell for the requested offsets")
    ext = candidates[len(candidates) // 2]
    out = np.zeros((b, len(offsets) * b, len(offsets) * b), dtype=float)
    support: list[tuple[int, int]] = []
    for jp, dj in enumerate(offsets):
        for kp, dk in enumerate(offsets):
            value = blocks.get((ext, ext + dj, ext + dk))
            if value is None:
                continue
            out[:, jp * b:(jp + 1) * b, kp * b:(kp + 1) * b] = value
            support.append((dj, dk))
    return out, ext, support


def relative_s2_error(tensor: np.ndarray) -> float:
    norm = np.linalg.norm(tensor)
    return float(np.linalg.norm(tensor - tensor.transpose(0, 2, 1)) / (norm or 1.0))


def relative_asr_errors(tensor: np.ndarray) -> tuple[float, float]:
    """Mass-uniform CNT ASR on the two contracted supercell legs."""
    n0, n1, n2 = tensor.shape
    if n1 != n2 or n1 % 3:
        raise ValueError("contracted dimensions must be equal atom x Cartesian axes")
    n_atoms = n1 // 3
    shaped = tensor.reshape(n0, n_atoms, 3, n_atoms, 3)
    norm = np.linalg.norm(tensor) or 1.0
    return (
        float(np.linalg.norm(shaped.sum(axis=1)) / norm),
        float(np.linalg.norm(shaped.sum(axis=3)) / norm),
    )


def atom_triplet_support(block: np.ndarray, relative_tol: float = 1e-12) -> dict:
    """Count nonzero 3x3x3 Cartesian blocks of one cell triple."""
    if len(set(block.shape)) != 1 or block.shape[0] % 3:
        raise ValueError("expected a cubic atom x Cartesian FC3 block")
    n_atoms = block.shape[0] // 3
    cart = block.reshape(n_atoms, 3, n_atoms, 3, n_atoms, 3)
    cart = cart.transpose(0, 2, 4, 1, 3, 5)
    flat = cart.reshape(n_atoms, n_atoms, n_atoms, 27)
    norms = np.linalg.norm(flat, axis=-1)
    cutoff = relative_tol * float(norms.max(initial=0.0))
    active = norms > cutoff
    kept = float(np.linalg.norm(flat[active]) ** 2 / (np.linalg.norm(flat) ** 2 or 1.0))
    return {
        "active": int(np.count_nonzero(active)),
        "total": int(active.size),
        "fill": float(np.count_nonzero(active) / active.size),
        "squared_norm_kept": kept,
        "pairs_per_external_atom": [int(active[i].sum()) for i in range(n_atoms)],
    }


def mode_rank_summary(block: np.ndarray) -> list[dict]:
    """Unfolding ranks needed to retain several squared-norm fractions."""
    result = []
    for mode in range(3):
        matrix = np.moveaxis(block, mode, 0).reshape(block.shape[mode], -1)
        singular = np.linalg.svd(matrix, compute_uv=False)
        cumulative = np.cumsum(singular ** 2) / (np.sum(singular ** 2) or 1.0)
        result.append({
            "mode": mode,
            "rank_90": int(np.searchsorted(cumulative, 0.9) + 1),
            "rank_99": int(np.searchsorted(cumulative, 0.99) + 1),
            "rank_999": int(np.searchsorted(cumulative, 0.999) + 1),
            "rank_9999": int(np.searchsorted(cumulative, 0.9999) + 1),
        })
    return result


def _s2_hosvd(block: np.ndarray, rank: int) -> tuple[np.ndarray, int]:
    """S2-preserving Tucker/HOSVD approximation of one ``(a,c,c)`` block.

    The two internal legs use one shared subspace.  This is a deliberately
    small reference construction: it measures whether local block terms are
    promising before introducing a nonlinear orbit fit.
    """
    n0, n1, n2 = block.shape
    if n1 != n2:
        raise ValueError("S2 HOSVD requires equal internal dimensions")
    r0 = min(int(rank), n0)
    r1 = min(int(rank), n1)
    u0 = np.linalg.svd(block.reshape(n0, -1), full_matrices=False)[0][:, :r0]
    unfold1 = np.moveaxis(block, 1, 0).reshape(n1, -1)
    unfold2 = np.moveaxis(block, 2, 0).reshape(n2, -1)
    covariance = unfold1 @ unfold1.T + unfold2 @ unfold2.T
    _, v = np.linalg.eigh(covariance)
    shared = v[:, -r1:]
    core = np.einsum(
        "ia,jb,kc,ijk->abc", u0, shared, shared, block, optimize=True
    )
    reconstructed = np.einsum(
        "ia,jb,kc,abc->ijk", u0, shared, shared, core, optimize=True
    )
    # One external and one shared internal factor plus the core.
    parameters = n0 * r0 + n1 * r1 + r0 * r1 * r1
    return reconstructed, int(parameters)


def _tucker_hosvd(block: np.ndarray, rank: int) -> tuple[np.ndarray, int]:
    """Ordinary three-leg HOSVD used for one directed offset-pair block."""
    factors = []
    ranks = []
    for mode in range(3):
        matrix = np.moveaxis(block, mode, 0).reshape(block.shape[mode], -1)
        r = min(int(rank), matrix.shape[0])
        factors.append(np.linalg.svd(matrix, full_matrices=False)[0][:, :r])
        ranks.append(r)
    core = np.einsum(
        "ia,jb,kc,ijk->abc", factors[0], factors[1], factors[2], block,
        optimize=True,
    )
    reconstructed = np.einsum(
        "ia,jb,kc,abc->ijk", factors[0], factors[1], factors[2], core,
        optimize=True,
    )
    parameters = int(np.prod(ranks) + sum(
        n * r for n, r in zip(block.shape, ranks)
    ))
    return reconstructed, parameters


def block_tucker_sweep(
    tensor: np.ndarray,
    block_dof: int,
    ranks: tuple[int, ...] = (8, 12, 16, 24, 32, 36),
) -> list[dict]:
    """Fit occupied offset-pair blocks while retaining support and S2 exactly.

    Only the upper offset pairs are fitted; the lower pairs are their exact
    internal-leg transposes.  Diagonal pairs use a shared internal subspace.
    The format is not yet globally ASR constrained, so its ASR residual is an
    explicit decision metric rather than silently projected away.
    """
    b = int(block_dof)
    if tensor.shape != (b, 3 * b, 3 * b):
        raise ValueError("expected the three-offset primitive tensor")
    tensor_norm = np.linalg.norm(tensor) or 1.0
    occupied = []
    for j in range(3):
        for k in range(j, 3):
            value = tensor[:, j * b:(j + 1) * b, k * b:(k + 1) * b]
            if np.linalg.norm(value) > 1e-14 * tensor_norm:
                occupied.append((j, k, value))

    rows = []
    for rank in ranks:
        reconstructed = np.zeros_like(tensor)
        parameters = 0
        for j, k, value in occupied:
            if j == k:
                fitted, count = _s2_hosvd(value, rank)
            else:
                fitted, count = _tucker_hosvd(value, rank)
            reconstructed[:, j * b:(j + 1) * b, k * b:(k + 1) * b] = fitted
            if j != k:
                reconstructed[:, k * b:(k + 1) * b, j * b:(j + 1) * b] = (
                    fitted.transpose(0, 2, 1)
                )
            parameters += count
        asr_j, asr_k = relative_asr_errors(reconstructed)
        rows.append({
            "rank": int(rank),
            "relative_frobenius_error": float(
                np.linalg.norm(tensor - reconstructed) / tensor_norm
            ),
            "s2_error": relative_s2_error(reconstructed),
            "asr_error_j": asr_j,
            "asr_error_k": asr_k,
            "parameters": int(parameters),
            "stored_scalar_ratio": float(parameters / tensor.size),
            "unique_upper_blocks": len(occupied),
        })
    return rows


def exact_atom_sparse_summary(block: np.ndarray, relative_tol: float = 1e-12) -> dict:
    """Storage of an exact indexed 3x3x3 atom-triplet representation."""
    support = atom_triplet_support(block, relative_tol=relative_tol)
    values = 27 * support["active"]
    # Three uint16 atom indices are a conservative six bytes per active term;
    # quote value-equivalent float64 storage separately from the raw ratio.
    index_float64_equivalent = 0.75 * support["active"]
    return {
        **support,
        "value_scalars": int(values),
        "value_scalar_ratio": float(values / block.size),
        "indexed_float64_equivalent_ratio": float(
            (values + index_float64_equivalent) / block.size
        ),
    }


def global_indscal_sweep(
    tensor: np.ndarray,
    ranks: list[int],
    *,
    max_iter: int = 100,
    n_restarts: int = 0,
    lbfgs_iters: int = 1,
    seed_base: int = 0,
) -> list[dict]:
    """Fit each requested global INDSCAL rank independently."""
    from phonon_inputs.fc3_compression import fit_indscal, target_from_dense

    scaled = tensor / (np.linalg.norm(tensor) or 1.0)
    target = target_from_dense(scaled, n_super=scaled.shape[1] // 3)
    rows = []
    for rank in ranks:
        start = time.perf_counter()
        result = fit_indscal(
            target,
            rank=int(rank),
            n_restarts=n_restarts,
            max_iter=max_iter,
            lbfgs_iters=lbfgs_iters,
            tol=1e-9,
            seed=int(seed_base) + int(rank),
            enforce_asr=True,
        )
        rows.append({
            "rank": int(rank),
            "relative_frobenius_error": float(result.rel_err),
            "seconds": float(time.perf_counter() - start),
            "n_restarts": int(n_restarts),
            "lbfgs_iters": int(lbfgs_iters),
            "seed": int(seed_base) + int(rank),
        })
    return rows


def d11a_transport_history(path: str | Path) -> list[dict]:
    """Retain the old finite-wire evidence without treating it as CNT data."""
    if not Path(path).exists():
        return []
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    dense = next(row for row in rows if row["method"] == "dense")
    out = [{
        "method": "dense",
        "rank": 0,
        "conductance_error": 0.0,
        "conservation_error": float(dense["conservation_err"]),
    }]
    for row in rows:
        if row["method"] == "dense" or str(row["rank"]).split("_")[0] != "16":
            continue
        out.append({
            "method": row["method"],
            "rank": row["rank"],
            "vertex_error": float(row["frob_rel_err"]),
            "conductance_error": float(row["G_anh_rel_err_vs_dense"]),
            "conservation_error": float(row["conservation_err"]),
            "ballistic_collapse": row["ballistic_collapse"] == "True",
        })
    return out


def run(
    vertex: Path,
    fit_ranks: list[int],
    max_iter: int,
    d11a: Path,
    *,
    n_restarts: int = 0,
    lbfgs_iters: int = 1,
    seed_base: int = 0,
) -> dict:
    blocks, sizes = load_blocks(vertex)
    tensor, external, support = representative_offset_tensor(blocks, sizes)
    b = int(sizes[0])

    # Three actual orbit representatives in the local device export.  The two
    # cross-cell tensors have equal norms but are directed, not equal under a
    # bare Cartesian permutation in the stored atom gauge.
    onsite = blocks[(external, external, external)]
    plus = blocks[(external, external, external + 1)]
    minus = blocks[(external, external + 1, external + 1)]
    orbit_data = {}
    for name, value in (("onsite", onsite), ("cross_001", plus), ("cross_011", minus)):
        orbit_data[name] = {
            "frobenius_norm": float(np.linalg.norm(value)),
            "atom_triplets": exact_atom_sparse_summary(value),
            "mode_ranks": mode_rank_summary(value),
        }

    asr_j, asr_k = relative_asr_errors(tensor)
    result = {
        "source": str(vertex),
        "block_dof": b,
        "external_cell": external,
        "offset_support": [list(pair) for pair in support],
        "global_tensor_shape": list(tensor.shape),
        "global_s2_error": relative_s2_error(tensor),
        "global_asr_error_j": asr_j,
        "global_asr_error_k": asr_k,
        "orbits": orbit_data,
        "support_block_tucker": block_tucker_sweep(tensor, b),
        "global_indscal": global_indscal_sweep(
            tensor,
            fit_ranks,
            max_iter=max_iter,
            n_restarts=n_restarts,
            lbfgs_iters=lbfgs_iters,
            seed_base=seed_base,
        )
        if fit_ranks else [],
        "prior_d11a_rank16": d11a_transport_history(d11a),
    }
    return result


def _parse_ranks(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(x) for x in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertex", type=Path, default=DEFAULT_VERTEX)
    parser.add_argument("--d11a-csv", type=Path, default=DEFAULT_D11A)
    parser.add_argument("--fit-ranks", type=_parse_ranks, default=[])
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=0)
    parser.add_argument("--lbfgs-iters", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = run(
        args.vertex,
        args.fit_ranks,
        args.max_iter,
        args.d11a_csv,
        n_restarts=args.restarts,
        lbfgs_iters=args.lbfgs_iters,
        seed_base=args.seed_base,
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json is not None:
        args.json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
