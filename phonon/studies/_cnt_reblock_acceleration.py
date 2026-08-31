"""Audit cheaper ways to reproduce the conserving CNT 8 x 2 blocking.

This is a private, read-only study.  It combines three pieces of evidence:

* the exact primitive-cell mask induced by grouping two cells per solver block;
* the production timing logs for ``c16-half`` and ``c16x2h``;
* the saved ``c16x2h`` self-energy, used to measure the size and numerical
  rank of the blocks which a one-cell solver drops.

No production solver option is changed.  The pure helper functions are kept
small so their support and cost arithmetic can be unit-tested independently of
the multi-gigabyte optional snapshot.

Run from the repository root::

    python phonon/studies/_cnt_reblock_acceleration.py \
        --json /tmp/cnt_reblock_acceleration.json

Use ``--skip-sigma`` when the archived ``sigma_best.rank*.npz`` files are not
available.  The timing/support part then remains reproducible.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / "cluster/c16-half"
DEFAULT_REBLOCK = ROOT / "cluster/c16x2h"


def grouped_mask(ncells: int, cells_per_block: int,
                 block_band: int) -> np.ndarray:
    """Return the primitive-cell mask induced by a solver block band."""
    cell = np.arange(ncells, dtype=int)
    blocks = cell // int(cells_per_block)
    return np.abs(blocks[:, None] - blocks[None, :]) <= int(block_band)


def distance_coverage(mask: np.ndarray) -> dict[int, dict[str, float | int]]:
    """Count retained upper-diagonal primitive pairs at every separation."""
    n = int(mask.shape[0])
    out: dict[int, dict[str, float | int]] = {}
    for distance in range(n):
        total = n - distance
        kept = int(np.count_nonzero(np.diag(mask, k=distance)))
        if kept:
            out[distance] = {
                "kept": kept,
                "total": total,
                "fraction": kept / total,
            }
    return out


def cnt33_primitive_vertices(ncells: int) -> set[tuple[int, int, int]]:
    """CNT(3,3) nearest-cell FC3 topology used by the archived L16 run.

    The two crossed offset pairs ``(-1,+1)`` and ``(+1,-1)`` are absent.
    This seven-pair topology gives 106 primitive vertex blocks and exactly the
    2104 quads printed by ``c16-half`` at ``g_band=3``.
    """
    offsets = tuple(
        (a, b)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        if not a * b < 0
    )
    return {
        (i, i + a, i + b)
        for i in range(ncells)
        for a, b in offsets
        if 0 <= i + a < ncells and 0 <= i + b < ncells
    }


def merge_vertices(vertices: Iterable[tuple[int, int, int]],
                   cells_per_block: int) -> set[tuple[int, int, int]]:
    """Map exact primitive vertex blocks to unique grouped block triples."""
    m = int(cells_per_block)
    return {(i // m, j // m, k // m) for i, j, k in vertices}


def count_ring_quads(
    vertices: set[tuple[int, int, int]],
    nblocks: int,
    output_pair: Callable[[int, int], bool],
    green_pair: Callable[[int, int], bool],
    owner_pair: Callable[[int, int], bool] | None = None,
) -> tuple[int, int]:
    """Count the production ``_phi_pair_index`` quads and output pairs."""
    quads = 0
    pairs: set[tuple[int, int]] = set()
    for i, k1, k2 in vertices:
        for j in range(nblocks):
            if not output_pair(i, j):
                continue
            if owner_pair is not None and not owner_pair(i, j):
                continue
            for k1p in range(nblocks):
                if not green_pair(k1, k1p):
                    continue
                for k2p in range(nblocks):
                    if (green_pair(k2, k2p)
                            and (j, k2p, k1p) in vertices):
                        quads += 1
                        pairs.add((i, j))
    return quads, len(pairs)


def atom_sparse_ring_ratio(
    vertices: set[tuple[int, int, int]],
    nblocks: int,
    output_pair: Callable[[int, int], bool],
    green_pair: Callable[[int, int], bool],
    owner_pair: Callable[[int, int], bool] | None = None,
    onsite_fill: float = 228 / 1728,
    cross_fill: float = 76 / 1728,
) -> float:
    """Ideal MAC ratio when the two FC3 actions use atom-triplet sparsity.

    The production dense ring is three equal ``d^4`` GEMMs.  Sparse FC3 only
    changes the first and second vertex actions; the final ``T @ U`` remains
    dense.  A quad with left/right triplet fills ``f_l`` and ``f_r`` therefore
    costs ``(1 + f_l + f_r) / 3`` of the dense MAC count.  This is an
    arithmetic bound, not a throughput claim for a future sparse GPU kernel.
    """
    quads = 0
    units = 0.0

    def fill(vertex: tuple[int, int, int]) -> float:
        i, j, k = vertex
        return onsite_fill if i == j == k else cross_fill

    for left in vertices:
        i, k1, k2 = left
        for j in range(nblocks):
            if not output_pair(i, j):
                continue
            if owner_pair is not None and not owner_pair(i, j):
                continue
            for k1p in range(nblocks):
                if not green_pair(k1, k1p):
                    continue
                for k2p in range(nblocks):
                    right = (j, k2p, k1p)
                    if green_pair(k2, k2p) and right in vertices:
                        quads += 1
                        units += 1.0 + fill(left) + fill(right)
    if not quads:
        return 0.0
    return float(units / (3.0 * quads))


def atom_sparse_quad_categories(
    vertices: set[tuple[int, int, int]],
    nblocks: int,
    output_pair: Callable[[int, int], bool],
    green_pair: Callable[[int, int], bool],
    owner_pair: Callable[[int, int], bool] | None = None,
) -> dict[str, int]:
    """Count onsite/cross left-right vertex combinations in the ring tasks."""
    counts = {
        "onsite-onsite": 0,
        "onsite-cross": 0,
        "cross-onsite": 0,
        "cross-cross": 0,
    }

    def kind(vertex: tuple[int, int, int]) -> str:
        i, j, k = vertex
        return "onsite" if i == j == k else "cross"

    for left in vertices:
        i, k1, k2 = left
        for j in range(nblocks):
            if not output_pair(i, j):
                continue
            if owner_pair is not None and not owner_pair(i, j):
                continue
            for k1p in range(nblocks):
                if not green_pair(k1, k1p):
                    continue
                for k2p in range(nblocks):
                    right = (j, k2p, k1p)
                    if green_pair(k2, k2p) and right in vertices:
                        counts[f"{kind(left)}-{kind(right)}"] += 1
    return counts


def atom_sparse_layout_categories(
    vertices: set[tuple[int, int, int]],
    nblocks: int,
    output_pair: Callable[[int, int], bool],
    green_pair: Callable[[int, int], bool],
    owner_pair: Callable[[int, int], bool] | None = None,
) -> dict[str, int]:
    """Count the 30-row/52-row cross layouts used by the CUDA prototype."""
    labels = ("onsite", "cross30", "cross52")
    counts = {f"{left}-{right}": 0 for left in labels for right in labels}

    def kind(vertex: tuple[int, int, int]) -> str:
        i, k1, k2 = vertex
        if i == k1 == k2:
            return "onsite"
        # The sparse first action groups FC3 axes (outer, second-internal) and
        # contracts the first-internal axis.  On the stored CNT gauge, blocks
        # with K2=I have 52 active output atom rows; all other cross orbits
        # have 30.  The right action receives its already-swapped vertex tuple
        # and obeys the same classification.
        return "cross52" if k2 == i else "cross30"

    for left in vertices:
        i, k1, k2 = left
        for j in range(nblocks):
            if not output_pair(i, j):
                continue
            if owner_pair is not None and not owner_pair(i, j):
                continue
            for k1p in range(nblocks):
                if not green_pair(k1, k1p):
                    continue
                for k2p in range(nblocks):
                    right = (j, k2p, k1p)
                    if green_pair(k2, k2p) and right in vertices:
                        counts[f"{kind(left)}-{kind(right)}"] += 1
    return counts


def dense_ring_gflop(nquads: int, ntau: int, dof: int,
                     rings_per_quad: int = 6) -> float:
    """Production three-GEMM complex-flop model for equal block sizes."""
    # Three b^4 GEMMs per ring and eight real flops per complex MAC.
    return (nquads * rings_per_quad * 8 * ntau * 3 * dof**4) / 1e9


def ideal_auxiliary_break_even_rank(dof: int,
                                    cells_per_block: int = 2) -> float:
    """Rank limit for ``N(d+2r)^3 < (N/m)(md)^3``.

    It is an optimistic bound: sparse-extension overhead and selected inverse
    work are ignored.
    """
    m = float(cells_per_block)
    return 0.5 * dof * (m ** (2.0 / 3.0) - 1.0)


_TIMING_LABELS = {
    "solver": "PhononSolver :",
    "ring": "PhPh SSE: 3 ring contraction :",
    "sse": "SigmaPhononPhonon :",
    "iteration": "SCBA: Iteration :",
}


def timing_medians(path: Path, warmup: int = 2) -> dict[str, float]:
    """Read post-warm-up medians from a Quatrex profiler text log."""
    text = path.read_text(errors="replace")
    out: dict[str, float] = {}
    for key, label in _TIMING_LABELS.items():
        values = np.asarray([
            float(x) for x in re.findall(re.escape(label) + r"\s*([0-9.]+)s",
                                          text)
        ])
        if values.size <= warmup:
            raise ValueError(f"not enough {label!r} samples in {path}")
        out[key] = float(np.median(values[warmup:]))
        out[key + "_samples"] = int(values.size)
    return out


def _timing_file(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("*_quatrex_times.out"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected one *_quatrex_times.out in {run_dir}, found {len(files)}")
    return files[0]


def _rank_block_pairs(world_rank: int, nblocks: int = 8,
                      block_comm_size: int = 2) -> list[tuple[int, int]]:
    """BCOO block order for the archived arrow-distributed dense pattern."""
    block_rank = world_rank % block_comm_size
    section = nblocks // block_comm_size
    lo = block_rank * section
    # rank 0 owns every arrow touching its section; rank 1 owns the remaining
    # bottom-right section.  The cutoff=40 archived pattern is block dense.
    return [
        (i, j)
        for i in range(nblocks)
        for j in range(nblocks)
        if min(i, j) >= lo and (block_rank == block_comm_size - 1
                                or min(i, j) < lo + section)
    ]


def _far_correction(block: np.ndarray, primitive_dof: int = 36) -> np.ndarray:
    """Remove the primitive |i-j|<=1 subblock from an upper 2-cell link."""
    d = int(primitive_dof)
    out = np.zeros_like(block)
    out[..., :d, :d] = block[..., :d, :d]       # primitive distance 2
    out[..., :d, d:] = block[..., :d, d:]       # alternating distance 3
    out[..., d:, d:] = block[..., d:, d:]       # primitive distance 2
    return out


def _frobenius_ranks(matrices: np.ndarray,
                     tolerances=(1e-2, 1e-3, 1e-4)) -> dict[str, dict[str, float]]:
    """Ranks required to bound each matrix's relative Frobenius tail."""
    singular = np.linalg.svd(matrices, compute_uv=False)
    squared = singular * singular
    total = np.sqrt(squared.sum(axis=1))
    # Sum the discarded singular values directly.  Computing this as
    # ``total**2 - cumulative`` loses all relative accuracy at the last few
    # ranks and can leave a spurious nonzero final tail; ``argmax`` would then
    # report rank one when no entry satisfies a tight tolerance.
    reverse = np.cumsum(squared[:, ::-1], axis=1)[:, ::-1]
    tail_squared = np.zeros_like(squared)
    tail_squared[:, :-1] = reverse[:, 1:]
    tail = np.sqrt(tail_squared)
    out: dict[str, dict[str, float]] = {}
    for tol in tolerances:
        ranks = np.argmax(tail <= tol * total[:, None], axis=1) + 1
        ranks[total <= np.finfo(float).tiny] = 0
        out[f"{tol:.0e}"] = {
            "min": int(ranks.min()),
            "median": float(np.median(ranks)),
            "p90": float(np.quantile(ranks, 0.9)),
            "max": int(ranks.max()),
        }
    return out


def analyse_sigma_snapshot(run_dir: Path, primitive_dof: int = 36,
                           sample_count_per_stack_rank: int = 11,
                           include_keldysh: bool = True) -> dict:
    """Measure the grouped-only self-energy weight and its per-link rank."""
    files = [run_dir / f"sigma_best.rank{rank}.npz" for rank in range(4)]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("missing saved CNT Sigma slices: "
                                + ", ".join(missing))

    d = int(primitive_dof)
    grouped_dof = 2 * d
    power = {"all": 0.0, "near_d01": 0.0, "new_d2": 0.0, "new_d3": 0.0}
    sampled: dict[str, list[np.ndarray]] = {"retarded": []}
    if include_keldysh:
        sampled["delta"] = []

    for rank, path in enumerate(files):
        pairs = _rank_block_pairs(rank)
        with np.load(path) as archive:
            retarded = archive["sigma_retarded"]
            values = {"retarded": retarded}
            if include_keldysh:
                values["delta"] = (archive["sigma_lesser"]
                                   - archive["sigma_greater"])

        blocks = retarded.reshape(retarded.shape[0], len(pairs),
                                  grouped_dof, grouped_dof)
        for index, (bi, bj) in enumerate(pairs):
            block = blocks[:, index]
            power["all"] += float(np.vdot(block, block).real)
            for ui in range(2):
                for uj in range(2):
                    distance = abs((2 * bi + ui) - (2 * bj + uj))
                    sub = block[:, ui*d:(ui+1)*d, uj*d:(uj+1)*d]
                    amount = float(np.vdot(sub, sub).real)
                    if distance <= 1:
                        power["near_d01"] += amount
                    elif distance == 2:
                        power["new_d2"] += amount
                    elif distance == 3:
                        power["new_d3"] += amount

        sample_ids = np.unique(np.linspace(
            0, retarded.shape[0] - 1,
            sample_count_per_stack_rank, dtype=int))
        for name, array in values.items():
            shaped = array.reshape(array.shape[0], len(pairs),
                                   grouped_dof, grouped_dof)
            for index, (bi, bj) in enumerate(pairs):
                if bj == bi + 1:
                    sampled[name].append(_far_correction(
                        shaped[sample_ids, index], d))
        del retarded, blocks, values
        gc.collect()

    result: dict[str, object] = {
        "power": power,
        "power_fraction": {
            key: value / power["all"] for key, value in power.items()
            if key != "all"
        },
    }
    new_power = power["new_d2"] + power["new_d3"]
    result["new_far_power_fraction"] = new_power / power["all"]
    result["new_far_frobenius_ratio"] = np.sqrt(new_power / power["all"])

    ranks: dict[str, object] = {}
    for name, chunks in sampled.items():
        matrices = np.concatenate(chunks, axis=0)
        primitive = np.concatenate(
            (matrices[:, :d, :d], matrices[:, :d, d:],
             matrices[:, d:, d:]), axis=0)
        ranks[name] = {
            "sampled_grouped_links": int(matrices.shape[0]),
            "assembled_far": _frobenius_ranks(matrices),
            "primitive_far_blocks": _frobenius_ranks(primitive),
        }
    result["ranks"] = ranks
    return result


def support_and_cost_model(ncells: int = 16, primitive_dof: int = 36) -> dict:
    """Return the exact CNT support masks, quad counts and flop models."""
    vertices = cnt33_primitive_vertices(ncells)
    merged = merge_vertices(vertices, 2)
    nsuper = ncells // 2

    baseline_quads, baseline_pairs = count_ring_quads(
        vertices, ncells,
        output_pair=lambda i, j: abs(i - j) <= 1,
        green_pair=lambda i, j: abs(i - j) <= 3,
    )

    merged_cases: dict[str, dict[str, float | int]] = {}
    for green_band in (1, 2, 3):
        quads, pairs = count_ring_quads(
            merged, nsuper,
            output_pair=lambda i, j: abs(i - j) <= 1,
            green_pair=lambda i, j, b=green_band: abs(i - j) <= b,
            owner_pair=lambda i, j: min(i, j) < nsuper // 2,
        )
        merged_cases[f"g{green_band}"] = {
            "slow_rank_quads": quads,
            "slow_rank_pairs": pairs,
            "gflop_6ring": dense_ring_gflop(quads, 161,
                                                   2 * primitive_dof),
            "gflop_4ring": dense_ring_gflop(quads, 161,
                                                   2 * primitive_dof, 4),
        }

    # Exact algebraic microblocking: keep the grouped masks, but enumerate the
    # ring over primitive FC3/G/Sigma subblocks on the slow block rank.
    micro_quads, micro_pairs = count_ring_quads(
        vertices, ncells,
        output_pair=lambda i, j: abs(i // 2 - j // 2) <= 1,
        green_pair=lambda i, j: abs(i // 2 - j // 2) <= 3,
        owner_pair=lambda i, j: min(i // 2, j // 2) < nsuper // 2,
    )
    micro_sparse_ratio = atom_sparse_ring_ratio(
        vertices, ncells,
        output_pair=lambda i, j: abs(i // 2 - j // 2) <= 1,
        green_pair=lambda i, j: abs(i // 2 - j // 2) <= 3,
        owner_pair=lambda i, j: min(i // 2, j // 2) < nsuper // 2,
    )
    micro_sparse_categories = atom_sparse_quad_categories(
        vertices, ncells,
        output_pair=lambda i, j: abs(i // 2 - j // 2) <= 1,
        green_pair=lambda i, j: abs(i // 2 - j // 2) <= 3,
        owner_pair=lambda i, j: min(i // 2, j // 2) < nsuper // 2,
    )
    micro_sparse_layouts = atom_sparse_layout_categories(
        vertices, ncells,
        output_pair=lambda i, j: abs(i // 2 - j // 2) <= 1,
        green_pair=lambda i, j: abs(i // 2 - j // 2) <= 3,
        owner_pair=lambda i, j: min(i // 2, j // 2) < nsuper // 2,
    )

    return {
        "primitive_vertices": len(vertices),
        "merged_vertices": len(merged),
        "sigma_output_coverage": distance_coverage(
            grouped_mask(ncells, 2, 1)),
        "green_coverage": distance_coverage(grouped_mask(ncells, 2, 3)),
        "baseline": {
            "quads": baseline_quads,
            "pairs": baseline_pairs,
            "gflop_6ring": dense_ring_gflop(
                baseline_quads, 81, primitive_dof),
        },
        "merged": merged_cases,
        "microblocked_exact": {
            "slow_rank_quads": micro_quads,
            "slow_rank_pairs": micro_pairs,
            "gflop_6ring": dense_ring_gflop(
                micro_quads, 161, primitive_dof),
            "gflop_4ring": dense_ring_gflop(
                micro_quads, 161, primitive_dof, 4),
            "atom_sparse_ideal_mac_ratio": micro_sparse_ratio,
            "atom_sparse_quad_categories": micro_sparse_categories,
            "atom_sparse_layout_categories": micro_sparse_layouts,
        },
        "ideal_solver_flop_ratio": 4.0,
        "ideal_solver_storage_ratio": 2.0,
        "auxiliary_break_even_rank": ideal_auxiliary_break_even_rank(
            primitive_dof, 2),
    }


def run(baseline_dir: Path = DEFAULT_BASELINE,
        reblock_dir: Path = DEFAULT_REBLOCK,
        include_sigma: bool = True,
        include_keldysh: bool = True) -> dict:
    model = support_and_cost_model()
    baseline_timing = timing_medians(_timing_file(baseline_dir))
    reblock_timing = timing_medians(_timing_file(reblock_dir))

    ring = reblock_timing["ring"]
    nonring = reblock_timing["iteration"] - ring
    merged = model["merged"]
    model_g3 = float(merged["g3"]["gflop_6ring"])
    model_g1 = float(merged["g1"]["gflop_6ring"])

    baseline_throughput = (float(model["baseline"]["gflop_6ring"])
                           / baseline_timing["ring"])
    micro_ring = (float(model["microblocked_exact"]["gflop_6ring"])
                  / baseline_throughput)
    sparse_ratio = float(
        model["microblocked_exact"]["atom_sparse_ideal_mac_ratio"])
    micro_sparse_ring = micro_ring * sparse_ratio
    # The final T@U (one of the original three equal d^4 GEMMs) still runs at
    # dense throughput.  Only the extra fraction above 1/3 is handled by the
    # sparse vertex kernels.  These thresholds are therefore relative useful
    # MAC throughput for those two kernels, not for the whole ring.
    sparse_extra_ratio = sparse_ratio - 1.0 / 3.0
    merged_target_ratio = ring / micro_ring
    c16_target_ratio = (
        (baseline_timing["iteration"] - nonring)
        / (micro_ring * 4.0 / 6.0)
    )
    predictions = {
        # Exact identity, current g3 approximation unchanged.
        "g3_four_ring_exact_iteration_s": nonring + ring * 4.0 / 6.0,
        # Diagnostic approximation changes: grouped output stays, G shrinks.
        "g1_six_ring_iteration_s": nonring + ring * model_g1 / model_g3,
        "g1_four_ring_iteration_s": (
            nonring + ring * model_g1 / model_g3 * 4.0 / 6.0),
        # Exact grouped approximation, conservative estimate using measured
        # d=36 throughput from c16-half rather than d=72 throughput.
        "microblocked_exact_iteration_s": nonring + micro_ring,
        "microblocked_exact_four_ring_iteration_s": (
            nonring + micro_ring * 4.0 / 6.0),
        # Arithmetic lower-bound projection at equal useful-MAC throughput.
        # Real sparse GPU throughput must be measured separately.
        "atom_sparse_ideal_iteration_s": nonring + micro_sparse_ring,
        "atom_sparse_ideal_four_ring_iteration_s": (
            nonring + micro_sparse_ring * 4.0 / 6.0),
        "microblocked_exact_ring_s": micro_ring,
        "atom_sparse_ideal_ring_s": micro_sparse_ring,
        "sparse_action_throughput_to_beat_merged_four_ring": (
            sparse_extra_ratio / (merged_target_ratio - 1.0 / 3.0)),
        "sparse_action_throughput_to_beat_c16_iteration": (
            sparse_extra_ratio / (c16_target_ratio - 1.0 / 3.0)),
        "baseline_d36_model_gflop_per_s": baseline_throughput,
    }

    out = {
        "model": model,
        "timing": {"c16_half": baseline_timing,
                   "c16x2h": reblock_timing},
        "measured_iteration_cost_ratio": (
            reblock_timing["iteration"] / baseline_timing["iteration"]),
        "predictions": predictions,
    }
    if include_sigma:
        out["sigma_snapshot"] = analyse_sigma_snapshot(
            reblock_dir, include_keldysh=include_keldysh)
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _print_report(result: dict) -> None:
    model = result["model"]
    timing = result["timing"]
    print("CNT 16 x 1 -> 8 x 2 support")
    print("  Sigma output primitive distances:",
          model["sigma_output_coverage"])
    print("  selected-G primitive distances:", model["green_coverage"])
    print(f"  archived quad check: {model['baseline']['quads']} "
          f"(expected 2104)")
    print(f"  merged g3 slow-rank quads: "
          f"{model['merged']['g3']['slow_rank_quads']} (expected 513)")
    print("\nPost-warm-up timing medians (s)")
    for name, row in timing.items():
        print(f"  {name:10s} iteration={row['iteration']:.4f} "
              f"ring={row['ring']:.4f} solver={row['solver']:.4f}")
    print(f"  measured reblock cost ratio: "
          f"{result['measured_iteration_cost_ratio']:.3f}x")
    print("\nConservative timing projections (s/iteration)")
    for key, value in result["predictions"].items():
        if key.endswith("iteration_s"):
            print(f"  {key:38s} {value:.3f}")
    if "sigma_snapshot" in result:
        sigma = result["sigma_snapshot"]
        print("\nSaved reblocked Sigma")
        print(f"  newly retained farther power fraction: "
              f"{sigma['new_far_power_fraction']:.6f}")
        print(f"  newly retained farther Frobenius ratio: "
              f"{sigma['new_far_frobenius_ratio']:.6f}")
        for name, ranks in sigma["ranks"].items():
            print(f"  {name} assembled-far ranks:", ranks["assembled_far"])
        print(f"  optimistic Woodbury break-even rank: "
              f"{model['auxiliary_break_even_rank']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--reblock-dir", type=Path, default=DEFAULT_REBLOCK)
    parser.add_argument("--skip-sigma", action="store_true")
    parser.add_argument("--skip-keldysh-rank", action="store_true",
                        help="analyse Sigma^R only (uses less memory)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = run(args.baseline_dir, args.reblock_dir,
                 include_sigma=not args.skip_sigma,
                 include_keldysh=not args.skip_keldysh_rank)
    _print_report(result)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(_jsonable(result), indent=2) + "\n")


if __name__ == "__main__":
    main()
