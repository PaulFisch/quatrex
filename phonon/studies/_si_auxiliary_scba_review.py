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
    archived_source_fit: float = np.nan

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
            archived_source_fit=(float(data["source_fit"][i])
                                 if "source_fit" in data else np.nan),
        ))
    return FrozenCase(
        name=name, frequencies=w,
        block_sizes=np.asarray(data["block_sizes"], dtype=int),
        q_shape=tuple(int(k) for k in data["q_shape"]), clusters=clusters)


def _passive_source_fit(source: np.ndarray, case: FrozenCase,
                        cluster: FrozenCluster):
    """Fit one constant passive source on the *real* frequency axis.

    ``source_at_poles`` is the correct analytic continuation for the existing
    partial-fraction correction, but its value at a complex pole is not a
    covariance and need not be PSD.  Using it as ``Q`` therefore made the first
    version of this gate reject every real Si cluster.  An auxiliary-state
    realization instead needs a real-axis PSD source.  We take a positive
    response-weighted average over the pole window, which is PSD whenever the
    sampled source is PSD, and measure the resulting error after the complete
    ``U D(.) [.] D(.)^H U^H`` congruence.
    """
    w = np.asarray(case.frequencies, dtype=float)
    src = np.asarray(source, dtype=complex)
    carrier = -1j * src
    herm = 0.5 * (carrier + carrier.conj().swapaxes(-1, -2))
    scale = max(float(np.linalg.norm(
        herm.reshape(herm.shape[0], -1), axis=1).max()), 1e-300)
    anti = float(np.linalg.norm(carrier - carrier.conj().swapaxes(-1, -2))
                 / max(np.linalg.norm(carrier), 1e-300))
    floor = min(float(np.linalg.eigvalsh(h)[0]) for h in herm) / scale

    centres = np.abs(np.real(np.asarray(cluster.z, dtype=complex)))
    dw = float(np.median(np.diff(w))) if w.size > 1 else 1.0
    margin = max(2.0 * dw, 4.0 * float(np.max(np.abs(np.imag(cluster.z)))))
    keep = ((w >= max(0.0, float(centres.min()) - margin))
            & (w <= float(centres.max()) + margin))
    if not np.any(keep):
        keep[int(np.argmin(abs(w - float(np.mean(centres)))))] = True
    wk = w[keep]
    ck = herm[keep]
    widths = np.ones_like(wk) * dw
    d = 1.0 / (wk[:, None] - np.asarray(cluster.z)[None, :])
    # A scalar sensitivity weight preserves passivity of the average.  The
    # actual error below is evaluated with the full, generally non-orthogonal
    # physical coupling U.
    weight = widths * np.sum(np.abs(d) ** 2, axis=1) ** 2
    q = np.einsum("w,wab->ab", weight, ck) / max(float(weight.sum()), 1e-300)
    q = 0.5 * (q + q.conj().T)

    num = den = 0.0
    for wi, ci, di, hi in zip(wk, ck, d, widths):
        del wi
        r = cluster.u * di[None, :]
        exact = r @ ci @ r.conj().T
        fitted = r @ q @ r.conj().T
        num += float(hi) * float(np.linalg.norm(fitted - exact) ** 2)
        den += float(hi) * float(np.linalg.norm(exact) ** 2)
    error = float(np.sqrt(num / max(den, 1e-300)))
    return q, floor, anti, error


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
            bproj = np.asarray(vf.UB[pos[d1], q1]).T @ u1
            for d2 in (int(x) for x in vf.offsets):
                if support is not None and (d1, d2) not in support:
                    continue
                k2 = i + d2
                if not 0 <= k2 < n_cells:
                    continue
                u2 = ub[k2 * cell_dof:(k2 + 1) * cell_dof]
                cproj = np.asarray(vf.UC[pos[d2], q2]).T @ u2
                # The old three-operand einsum recomputed a contraction path
                # for every cell/offset/cluster tuple.  Flatten the two modal
                # legs and use one GEMM; this is the same contraction and is
                # the form a production tensor kernel would use.
                modal_outer = (bproj[:, :, None] * cproj[:, None, :]).reshape(
                    dt.shape[1], -1)
                acc += (dt @ modal_outer).reshape(cell_dof, ra, rb)
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


def analyse_case(case: FrozenCase, vf=None, tolerances=(1e-2, 1e-3, 1e-4),
                 external_q_count: int = 0):
    nq = int(np.prod(case.q_shape)) if case.q_shape else 1
    by_q: dict[int, list[tuple[FrozenCluster, np.ndarray, np.ndarray]]] = {
        q: [] for q in range(nq)}
    floors, anti_errors, fit_errors, archived_fits, effective = [], [], [], [], []
    invalid = 0
    all_input_ranks = [0 for _ in range(nq)]
    for cl in case.clusters:
        qflat = (int(np.ravel_multi_index(cl.q, case.q_shape))
                 if case.q_shape else 0)
        all_input_ranks[qflat] += cl.rank
        ql, fl, al, el = _passive_source_fit(cl.source_lesser, case, cl)
        qg, fg, ag, eg = _passive_source_fit(cl.source_greater, case, cl)
        floor = min(fl, fg)
        floors.append(floor)
        anti_errors.append(max(al, ag))
        fit_errors.append(max(el, eg))
        archived_fits.append(float(cl.archived_source_fit))
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

    archived_finite = [x for x in archived_fits if np.isfinite(x)]
    result = {
        "n_cells": case.n_cells,
        "cell_dof": case.cell_dof,
        "n_q": nq,
        "n_clusters": len(case.clusters),
        "n_invalid_passive_clusters": invalid,
        "passivity_test": "real-axis projected source over the full sampled grid",
        "source_psd_floor_min": float(min(floors, default=0.0)),
        "source_antihermiticity_error_max": float(max(anti_errors, default=0.0)),
        "passive_constant_source_congruence_error": {
            "median": float(np.median(fit_errors)) if fit_errors else 0.0,
            "p90": float(np.percentile(fit_errors, 90)) if fit_errors else 0.0,
            "max": float(max(fit_errors, default=0.0)),
        },
        "legacy_complex_continuation_source_fit": {
            "median": (float(np.median(archived_finite))
                       if archived_finite else None),
            "p90": (float(np.percentile(archived_finite, 90))
                    if archived_finite else None),
            "max": (float(max(archived_finite))
                    if archived_finite else None),
        },
        "input_poles_per_q": {
            "median": float(np.median(all_input_ranks)),
            "max": int(max(all_input_ranks, default=0)),
            "total": int(sum(all_input_ranks)),
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
            "refused because at least one real-axis source is materially non-passive")
        return result
    if int(vf.n_kpts) != nq or int(vf.D.shape[0]) != case.cell_dof:
        raise ValueError("vertex factors do not match this Si case")

    if 0 < int(external_q_count) < nq:
        count = int(external_q_count)
        # Cover the mesh while deliberately including the q points with the
        # largest input state count.  This is a conservative long-film rank
        # screen, not a replacement for the full q fold.
        spread = np.linspace(0, nq - 1, count, dtype=int).tolist()
        ranked = sorted(range(nq), key=lambda q: (-all_input_ranks[q], q))
        external_q = []
        for pair in zip(ranked, spread):
            for q in pair:
                if q not in external_q:
                    external_q.append(q)
                if len(external_q) == count:
                    break
            if len(external_q) == count:
                break
    else:
        external_q = list(range(nq))

    raw_ranks, merged_counts = [], []
    ranks = {str(t): [] for t in tolerances}
    eig_spectra = []
    for qe in external_q:
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
        "external_q_indices": external_q,
        "full_external_q_axis": len(external_q) == nq,
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
    parser.add_argument("--external-q-count", type=int, default=0,
                        help="conservative sampled external-q screen; 0 means all")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    vf = load_decomposed(args.vertices) if args.vertices else None
    report = {}
    for spec in args.case:
        case = load_case(spec)
        report[case.name] = analyse_case(
            case, vf=vf, external_q_count=args.external_q_count)
        print(case.name, json.dumps(report[case.name], indent=2)[:5000], flush=True)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"WROTE {target}")


if __name__ == "__main__":
    main()
