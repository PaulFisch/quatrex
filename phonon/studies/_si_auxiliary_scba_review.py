"""Real-Si rank and spatial gate for the production auxiliary SCBA path.

The input is written by ``QX_SAVE_POLE_STATES`` in ``engine/run.py``.  It
contains the q-resolved poles, coherent right/left subspaces and the complete
frequency-dependent projected Keldysh sources from one frozen production
solve.  This study asks the two questions a scalar pole census cannot answer:

1. can each source be represented as a passive constant cluster carrier; and
2. after the real q fold and production tensor-decomposed FC3 contraction,
   how many *physical* auxiliary directions survive?

The raw Kronecker state is never assembled globally.  For every external q we
accumulate its frequency-integrated PSD covariance

    C_q = sum_s U_s W_s U_s^H,

where ``s`` runs over all q/cluster pairs and ``W_s`` is the exact Gramian of
the Kronecker-sum output state.  Eigenvalue truncation of ``C_q`` is the best
possible congruence compression of the physical coupling for this integrated
metric.  Consequently a rank that is already too high here is a firm no-go
for the more constrained frequency-resolved/passive compression.

Example::

    QTX_ARRAY_MODULE=numpy PYTHONPATH=src \
      python phonon/studies/_si_auxiliary_scba_review.py \
      --case L3=cluster/si-aux-l3/poles.npz \
      --case L8=cluster/si-aux-l8/poles.npz \
      --vertices cluster/si-aux-inputs/decomposed_vertices.npz \
      --output phonon/studies/out/si_auxiliary_scba_review.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quatrex.phonon.auxiliary_scba import lyapunov_gramian
from quatrex.phonon.pole_bridge import source_at_poles, source_variation
from quatrex.phonon.pole_keldysh import PoleCluster
from quatrex.phonon.vertex_factors import load_decomposed


@dataclass
class FrozenCluster:
    q: tuple[int, ...]
    z: np.ndarray
    u: np.ndarray
    v: np.ndarray
    source_lesser: np.ndarray
    source_greater: np.ndarray
    label: str

    @property
    def rank(self) -> int:
        return int(self.z.size)


@dataclass
class FrozenCase:
    name: str
    frequencies: np.ndarray
    block_sizes: np.ndarray
    q_shape: tuple[int, ...]
    clusters: list[FrozenCluster]

    @property
    def n_cells(self) -> int:
        return int(self.block_sizes.size)

    @property
    def cell_dof(self) -> int:
        values = np.unique(self.block_sizes)
        if values.size != 1:
            raise ValueError("the Si q-fold rank gate requires uniform cells")
        return int(values[0])


def load_case(spec: str) -> FrozenCase:
    if "=" not in spec:
        raise ValueError("--case must be NAME=path")
    name, raw_path = spec.split("=", 1)
    data = np.load(raw_path, allow_pickle=False)
    po = np.asarray(data["pole_offsets"], dtype=int)
    so = np.asarray(data["source_offsets"], dtype=int)
    w = np.asarray(data["local_frequencies"], dtype=float)
    qidx = np.asarray(data["q_index"], dtype=int)
    labels = np.asarray(data["labels"]).astype(str)
    clusters = []
    for i in range(po.size - 1):
        ps = slice(po[i], po[i + 1])
        r = int(po[i + 1] - po[i])
        ss = slice(so[i], so[i + 1])
        expected = w.size * r * r
        if int(so[i + 1] - so[i]) != expected:
            raise ValueError(
                f"cluster {i} source has {so[i + 1] - so[i]} values, "
                f"expected {expected}")
        clusters.append(FrozenCluster(
            q=tuple(int(k) for k in qidx[i]),
            z=np.asarray(data["poles"][ps], dtype=complex),
            u=np.asarray(data["coupling_u"][:, ps], dtype=complex),
            v=np.asarray(data["coupling_v"][:, ps], dtype=complex),
            source_lesser=np.asarray(
                data["source_lesser"][ss], dtype=complex).reshape(w.size, r, r),
            source_greater=np.asarray(
                data["source_greater"][ss], dtype=complex).reshape(w.size, r, r),
            label=str(labels[i]),
        ))
    return FrozenCase(
        name=name, frequencies=w,
        block_sizes=np.asarray(data["block_sizes"], dtype=int),
        q_shape=tuple(int(k) for k in data["q_shape"]), clusters=clusters)


def _relative_floor(q: np.ndarray) -> float:
    q = 0.5 * (q + q.conj().T)
    ev = np.linalg.eigvalsh(q)
    return float(ev[0] / max(float(ev[-1]), 1e-300))


def _carrier(source: np.ndarray, case: FrozenCase,
             cluster: FrozenCluster) -> np.ndarray:
    cl = PoleCluster(z=cluster.z, u=cluster.u, v=cluster.v,
                     label=cluster.label)
    # Quatrex stores Sigma^x = +i C^x, hence C^x = -i Sigma^x.
    q = -1j * np.asarray(source_at_poles(
        source, case.frequencies, cl), dtype=complex)
    return 0.5 * (q + q.conj().T)


def _rank_for_frobenius(eigenvalues: np.ndarray, tol: float) -> int:
    ev = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    if not np.any(ev):
        return 0
    sq = ev * ev
    tail = np.cumsum(sq[::-1])[::-1]
    total = float(tail[0])
    for rank in range(ev.size + 1):
        rem = 0.0 if rank == ev.size else float(tail[rank])
        if np.sqrt(rem / total) <= float(tol):
            return rank
    return int(ev.size)


def _effective_cells(u: np.ndarray, block_sizes: np.ndarray) -> np.ndarray:
    off = np.concatenate(([0], np.cumsum(block_sizes)))
    weight = np.stack([
        np.sum(np.abs(u[off[i]:off[i + 1]]) ** 2, axis=0)
        for i in range(block_sizes.size)
    ])
    denom = np.sum(weight * weight, axis=0)
    return np.sum(weight, axis=0) ** 2 / np.maximum(denom, 1e-300)


def _modal_coupling(vf, q1: int, q2: int, ua: np.ndarray,
                    ub: np.ndarray, n_cells: int, cell_dof: int) -> np.ndarray:
    """Production factored FC3 applied to two modal subspaces."""
    ra, rb = ua.shape[1], ub.shape[1]
    out = np.zeros((n_cells * cell_dof, ra * rb), dtype=complex)
    pos = vf.offset_index()
    support = vf.meta.get("support_pairs")
    if support is not None:
        support = {(int(a), int(b)) for a, b in support}
    dt = np.asarray(vf.D * vf.lambdas[None, :])
    for i in range(n_cells):
        acc = np.zeros((cell_dof, ra, rb), dtype=complex)
        for d1 in (int(x) for x in vf.offsets):
            k1 = i + d1
            if not 0 <= k1 < n_cells:
                continue
            u1 = ua[k1 * cell_dof:(k1 + 1) * cell_dof]
            bproj = np.einsum(
                "ir,ia->ra", vf.UB[pos[d1], q1], u1, optimize=True)
            for d2 in (int(x) for x in vf.offsets):
                if support is not None and (d1, d2) not in support:
                    continue
                k2 = i + d2
                if not 0 <= k2 < n_cells:
                    continue
                u2 = ub[k2 * cell_dof:(k2 + 1) * cell_dof]
                cproj = np.einsum(
                    "ir,ib->rb", vf.UC[pos[d2], q2], u2, optimize=True)
                acc += np.einsum(
                    "mr,ra,rb->mab", dt, bproj, cproj, optimize=True)
        out[i * cell_dof:(i + 1) * cell_dof] = acc.reshape(cell_dof, -1)
    return out


def _cluster_combination_count(poles: list[complex], factor: float = 1.0) -> int:
    """Greedy linewidth-scaled count; diagnostic, not a pole fit."""
    ordered = sorted((complex(z) for z in poles), key=lambda z: z.real)
    groups: list[list[complex]] = []
    for z in ordered:
        for group in groups:
            centre = sum(group) / len(group)
            if abs(z.real - centre.real) <= factor * (
                    abs(z.imag) + abs(centre.imag)):
                group.append(z)
                break
        else:
            groups.append([z])
    return len(groups)


def analyse_case(case: FrozenCase, vf=None, tolerances=(1e-2, 1e-3, 1e-4)):
    nq = int(np.prod(case.q_shape)) if case.q_shape else 1
    by_q: dict[int, list[tuple[FrozenCluster, np.ndarray, np.ndarray]]] = {
        q: [] for q in range(nq)}
    floors, variations, effective = [], [], []
    invalid = 0
    for cl in case.clusters:
        qflat = (int(np.ravel_multi_index(cl.q, case.q_shape))
                 if case.q_shape else 0)
        ql = _carrier(cl.source_lesser, case, cl)
        qg = _carrier(cl.source_greater, case, cl)
        floor = min(_relative_floor(ql), _relative_floor(qg))
        floors.append(floor)
        pc = PoleCluster(cl.z, cl.u, cl.v, cl.label)
        variations.append(max(
            float(source_variation(
                cl.source_lesser, case.frequencies, pc)),
            float(source_variation(
                cl.source_greater, case.frequencies, pc))))
        effective.extend(_effective_cells(cl.u, case.block_sizes).tolist())
        if floor < -1e-8:
            invalid += 1
            continue
        # Remove only roundoff-level negativity and record it above.  A source
        # below -1e-8 is excluded, never silently projected into the study.
        def clean(q):
            ev, vec = np.linalg.eigh(q)
            return (vec * np.maximum(ev, 0.0)[None, :]) @ vec.conj().T
        by_q[qflat].append((cl, clean(ql), clean(qg)))

    input_ranks = [sum(cl.rank for cl, _ql, _qg in by_q[q]) for q in range(nq)]
    result = {
        "n_cells": case.n_cells,
        "cell_dof": case.cell_dof,
        "n_q": nq,
        "n_clusters": len(case.clusters),
        "n_invalid_passive_clusters": invalid,
        "source_psd_floor_min": float(min(floors, default=0.0)),
        "source_variation": {
            "median": float(np.median(variations)) if variations else 0.0,
            "p90": float(np.percentile(variations, 90)) if variations else 0.0,
            "max": float(max(variations, default=0.0)),
        },
        "input_poles_per_q": {
            "median": float(np.median(input_ranks)),
            "max": int(max(input_ranks, default=0)),
            "total": int(sum(input_ranks)),
        },
        "input_mode_effective_cells": {
            "median": float(np.median(effective)) if effective else 0.0,
            "p90": float(np.percentile(effective, 90)) if effective else 0.0,
            "max": float(max(effective, default=0.0)),
        },
    }
    if vf is None or invalid:
        result["qfold_status"] = (
            "not run" if vf is None else
            "refused because at least one source is materially non-passive")
        return result
    if int(vf.n_kpts) != nq or int(vf.D.shape[0]) != case.cell_dof:
        raise ValueError("vertex factors do not match this Si case")

    raw_ranks, merged_counts = [], []
    ranks = {str(t): [] for t in tolerances}
    eig_spectra = []
    for qe in range(nq):
        cov = np.zeros((case.n_cells * case.cell_dof,) * 2, dtype=complex)
        raw = 0
        combination_poles: list[complex] = []
        for q1 in range(nq):
            q2 = int(vf.q_diff_map[qe, q1])
            for ca, qla, qga in by_q[q1]:
                for cb, qlb, qgb in by_q[q2]:
                    qa, qb = qla + qga, qlb + qgb
                    wa = lyapunov_gramian(ca.z, qa)
                    wb = lyapunov_gramian(cb.z, qb)
                    qo = np.kron(wa, qb) + np.kron(qa, wb)
                    zo = (ca.z[:, None] + cb.z[None, :]).reshape(-1)
                    uo = _modal_coupling(
                        vf, q1, q2, ca.u, cb.u,
                        case.n_cells, case.cell_dof)
                    wo = lyapunov_gramian(zo, qo)
                    cov += uo @ wo @ uo.conj().T / float(nq)
                    raw += int(zo.size)
                    combination_poles.extend(zo.tolist())
        cov = 0.5 * (cov + cov.conj().T)
        ev = np.linalg.eigvalsh(cov)[::-1]
        scale = max(float(ev[0]) if ev.size else 0.0, 1e-300)
        if ev.size and float(ev[-1] / scale) < -1e-9:
            raise ValueError(
                f"q={qe} accumulated output covariance is not PSD: "
                f"lambda_min/lambda_max={ev[-1] / scale:.3e}")
        ev = np.maximum(ev, 0.0)
        raw_ranks.append(raw)
        merged_counts.append(_cluster_combination_count(combination_poles))
        eig_spectra.append((ev / scale).tolist())
        for tol in tolerances:
            ranks[str(tol)].append(_rank_for_frobenius(ev, tol))

    rank_summary = {
        key: {"median": float(np.median(value)), "max": int(max(value))}
        for key, value in ranks.items()}
    # Conservative arithmetic gate: the current prototype performs the base
    # RGF plus one extra BTD factorization.  Compare its dominant terms with a
    # two-cell reblock, whose cubic width cost is 4*N*d^3.
    def cost(r):
        n, d = case.n_cells, case.cell_dof
        aux = 2 * n * d**3 + 6 * n * d**2 * r + 2 * r**3
        reblock = 4 * n * d**3
        return float(aux / reblock)
    result.update({
        "qfold_status": "complete",
        "raw_output_states_per_q": {
            "median": float(np.median(raw_ranks)), "max": int(max(raw_ranks))},
        "linewidth_merged_output_poles_per_q": {
            "median": float(np.median(merged_counts)),
            "max": int(max(merged_counts))},
        "integrated_physical_rank": rank_summary,
        "woodbury_over_two_cell_reblock_cost": {
            key: {
                "median_rank": cost(int(np.ceil(value["median"]))),
                "max_rank": cost(int(value["max"])),
            } for key, value in rank_summary.items()
        },
        "normalised_covariance_eigenvalues": eig_spectra,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True,
                        help="NAME=QX_SAVE_POLE_STATES.npz")
    parser.add_argument("--vertices",
                        help="production decomposed_vertices.npz; omit for the input gate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    vf = load_decomposed(args.vertices) if args.vertices else None
    report = {}
    for spec in args.case:
        case = load_case(spec)
        report[case.name] = analyse_case(case, vf=vf)
        print(case.name, json.dumps(report[case.name], indent=2)[:5000], flush=True)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"WROTE {target}")


if __name__ == "__main__":
    main()
