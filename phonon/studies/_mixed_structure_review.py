"""Cross-structure gate for a mixed phonon--phonon SCBA representation.

This study does not modify the solver.  It combines only quantities that can
be measured before enabling an approximation:

* fixed-point/reference certification from a saved production run;
* adaptive piecewise-linear frequency reconstruction of saved G/current
  observables (a proxy for a multiresolution cell basis, not a bubble oracle);
* FC3 reach and exact sparsity from the finite-device block archive;
* optional real pole/source/rank gates from the passive auxiliary census.

The selector is deliberately material-blind.  The default manifest happens to
cover CNT, Si, and MoS2 because they exercise three different decisions.  A
production implementation must use the same indicators, not the case names.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class CaseSpec:
    name: str
    run: str
    spectral_run: str
    fc3: str
    green_band: int
    certified_exact_band: int = 1
    pole_states: str | None = None
    pole_evidence: str | None = None
    auxiliary_gate: str | None = None
    auxiliary_label: str | None = None
    known_min_gamma_over_h: float | None = None
    flat_factor_error: float | None = None
    sparse_kernel_time_ratio: float | None = None
    tensor_factor_validated: bool = False
    far_compression_gate: str = "unmeasured"


def default_cases(root: Path) -> list[CaseSpec]:
    """Return the repo-backed, cross-structure default manifest."""
    rel = lambda p: str(root / p)  # noqa: E731 - compact manifest helper
    return [
        CaseSpec(
            name="CNT-L16x2",
            run=rel("cluster/c16x2h/run.npz"),
            spectral_run=rel("cluster/c16x2h/run.npz"),
            fc3=rel("cluster/cnt_cal/fc3_blocks.hdf5"),
            green_band=3,
            certified_exact_band=2,
            known_min_gamma_over_h=1.57,
            flat_factor_error=0.0452,
            sparse_kernel_time_ratio=0.5207,
            far_compression_gate="fail_near_shells_high_rank",
        ),
        CaseSpec(
            name="Si-L8x2",
            run=rel("cluster/si-l8x2-final/run.npz"),
            spectral_run=rel("cluster/si-l8x2-final/run.npz"),
            fc3=rel("cluster/sifilm_nk9r/fc3_blocks.hdf5"),
            green_band=3,
            certified_exact_band=2,
            pole_states=rel("cluster/si-aux-l8b/poles.npz"),
            pole_evidence="frozen converged interacting state",
            auxiliary_gate=rel(
                "phonon/studies/out/si_auxiliary_scba_L8_worstq.json"),
            auxiliary_label="L8",
            tensor_factor_validated=True,
            far_compression_gate="fail_l8x2_hodlr",
        ),
        CaseSpec(
            name="MoS2-L3",
            run=rel("cluster/mos2f3nu/run.npz"),
            spectral_run=rel("cluster/mos2f3nu/run_ballistic.npz"),
            fc3=rel("cluster/mos2f3nu/fc3_blocks.hdf5"),
            green_band=1,
            pole_states=rel("cluster/mos2-aux-l3-uniform/poles.npz"),
            pole_evidence=("two-iteration interacting uniform-grid proxy; "
                           "the production nonuniform-grid state is uncertified"),
            far_compression_gate="not_needed_for_onsite_fc3_at_b1",
        ),
    ]


def _relative_balance(values: np.ndarray) -> float | None:
    values = np.asarray(values)
    if values.size != 2:
        return None
    den = float(np.sum(np.abs(values)))
    return float(abs(values[0] - values[1]) / den) if den else 0.0


def reference_gate(path: str) -> dict[str, Any]:
    p = np.load(path, allow_pickle=False)
    converged = bool(p["converged"]) if "converged" in p else False
    diverged = bool(p["diverged"]) if "diverged" in p else False
    finite = True
    for key in ("lead_current", "internal_spread"):
        if key in p:
            finite &= bool(np.all(np.isfinite(np.asarray(p[key]))))
    profile_finite_fraction = None
    if "final_heat" in p:
        profile = np.asarray(p["final_heat"])
        profile_finite_fraction = float(np.count_nonzero(np.isfinite(profile))
                                        / max(profile.size, 1))
    balance = (_relative_balance(p["final_bubble_balance"])
               if "final_bubble_balance" in p else None)
    certified = converged and not diverged and finite
    if balance is not None:
        certified &= balance <= 1e-2
    reasons = []
    if not converged:
        reasons.append("SCBA fixed point is not converged")
    if diverged:
        reasons.append("run is marked diverged")
    if not finite:
        reasons.append("non-finite observable")
    if balance is not None and balance > 1e-2:
        reasons.append("bubble balance exceeds 1e-2")
    return {
        "certified": bool(certified),
        "reasons": reasons,
        "converged": converged,
        "diverged": diverged,
        "n_iter": int(p["n_iter"]) if "n_iter" in p else None,
        "eta": float(p["eta"]) if "eta" in p else None,
        "lead_current": float(p["lead_current"])
        if "lead_current" in p else None,
        "internal_spread": float(p["internal_spread"])
        if "internal_spread" in p else None,
        "current_profile_finite_fraction": profile_finite_fraction,
        "bubble_balance_defect": balance,
    }


def _frequency_channels(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequency nodes, cell weights, and scaled observable channels."""
    p = np.load(path, allow_pickle=False)
    w = np.asarray(p["energies"], dtype=float)
    weights = (np.asarray(p["frequency_cell_widths"], dtype=float)
               if "frequency_cell_widths" in p else np.gradient(w))
    channels = []
    if "current_spectrum" in p:
        cur = np.asarray(p["current_spectrum"], dtype=float)
        channels.append(cur.reshape(w.size, -1))
    if "gr_diag_imag" in p:
        gr = np.asarray(p["gr_diag_imag"], dtype=float)
        # Preserve q dependence but reduce the local physical axis.  The trace
        # is a much more stable pilot observable than selecting one DOF.
        if gr.ndim >= 2:
            gr = np.sum(np.abs(gr), axis=-1)
        channels.append(gr.reshape(w.size, -1))
    if not channels:
        raise ValueError(f"{path} has neither current_spectrum nor gr_diag_imag")
    a = np.concatenate(channels, axis=1)
    # Some historical wide-support adapters intentionally omitted internal-cut
    # spectra while still reporting exact leads and a separately evaluated
    # current-spread scalar.  Drop only those unavailable pilot channels.
    a = a[:, np.all(np.isfinite(a), axis=0)]
    if a.shape[1] == 0:
        raise ValueError(f"{path} has no finite spectral pilot channel")
    norms = np.sqrt(np.sum(weights[:, None] * np.abs(a) ** 2, axis=0))
    keep = norms > max(float(np.max(norms)), 1e-300) * 1e-10
    a = a[:, keep]
    norms = norms[keep]
    if a.shape[1] == 0:
        raise ValueError(f"{path} has no nonzero spectral pilot channel")
    a = a / norms[None, :]
    return w, weights, a


def linear_reconstruction(x: np.ndarray, values: np.ndarray,
                          selected: np.ndarray) -> np.ndarray:
    selected = np.unique(np.asarray(selected, dtype=int))
    if selected.size < 2 or selected[0] != 0 or selected[-1] != x.size - 1:
        raise ValueError("selected knots must contain both endpoints")
    right_pos = np.searchsorted(selected, np.arange(x.size), side="left")
    right_pos = np.clip(right_pos, 1, selected.size - 1)
    left = selected[right_pos - 1]
    right = selected[right_pos]
    den = x[right] - x[left]
    t = np.divide(x - x[left], den, out=np.zeros_like(x), where=den != 0)
    return (1.0 - t[:, None]) * values[left] + t[:, None] * values[right]


def weighted_error(reference: np.ndarray, approximation: np.ndarray,
                   weights: np.ndarray) -> float:
    num = np.sum(weights[:, None] * np.abs(reference - approximation) ** 2)
    den = np.sum(weights[:, None] * np.abs(reference) ** 2)
    return float(np.sqrt(num / max(float(den), 1e-300)))


def adaptive_knots(x: np.ndarray, values: np.ndarray, weights: np.ndarray,
                   tolerances=(1e-2, 1e-3)) -> dict[str, Any]:
    """Greedy multiresolution proxy using one common vector-valued grid."""
    selected = [0, x.size - 1]
    targets = sorted(set(float(t) for t in tolerances), reverse=True)
    reached: dict[float, dict[str, float | int]] = {}
    history = []
    while len(selected) <= x.size:
        ids = np.array(sorted(selected), dtype=int)
        approx = linear_reconstruction(x, values, ids)
        err = weighted_error(values, approx, weights)
        history.append((len(ids), err))
        for tol in targets:
            if tol not in reached and err <= tol:
                reached[tol] = {"points": len(ids), "error": err}
        if len(reached) == len(targets) or len(ids) == x.size:
            break
        row_error = weights * np.sum(np.abs(values - approx) ** 2, axis=1)
        row_error[ids] = -1.0
        selected.append(int(np.argmax(row_error)))
    return {
        "at_tolerance": {str(t): reached[t] for t in targets},
        "history": history,
    }


def uniform_knots(x: np.ndarray, values: np.ndarray, weights: np.ndarray,
                  tolerances=(1e-2, 1e-3)) -> dict[str, Any]:
    targets = sorted(set(float(t) for t in tolerances), reverse=True)
    reached: dict[float, dict[str, float | int]] = {}
    history = []
    for count in range(2, x.size + 1):
        ids = np.unique(np.rint(np.linspace(0, x.size - 1, count)).astype(int))
        approx = linear_reconstruction(x, values, ids)
        err = weighted_error(values, approx, weights)
        history.append((len(ids), err))
        for tol in targets:
            if tol not in reached and err <= tol:
                reached[tol] = {"points": len(ids), "error": err}
        if len(reached) == len(targets):
            break
    for tol in targets:
        reached.setdefault(tol, {"points": x.size, "error": 0.0})
    return {"at_tolerance": {str(t): reached[t] for t in targets},
            "history": history}


def spectral_gate(path: str) -> dict[str, Any]:
    x, weights, values = _frequency_channels(path)
    adaptive = adaptive_knots(x, values, weights)
    uniform = uniform_knots(x, values, weights)
    comparison = {}
    for tol in (1e-2, 1e-3):
        a = int(adaptive["at_tolerance"][str(tol)]["points"])
        u = int(uniform["at_tolerance"][str(tol)]["points"])
        comparison[str(tol)] = {
            "adaptive_points": a,
            "uniform_points": u,
            "adaptive_fraction": a / x.size,
            "adaptive_over_uniform": a / max(u, 1),
        }
    return {
        "fine_points": int(x.size),
        "channels": int(values.shape[1]),
        "uniform_grid": bool(np.allclose(np.diff(x), np.diff(x)[0],
                                         rtol=1e-10, atol=1e-12)),
        "min_cell_width": float(np.min(weights)),
        "max_cell_width": float(np.max(weights)),
        "comparison": comparison,
    }


def fc3_gate(path: str) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        keys = np.asarray(f["meta/keys"], dtype=int)
        block_sizes = np.asarray(f["meta/block_sizes"], dtype=int)
        tensors = f["fc3_blocks"]
        total = 0
        nnz = 0
        for name in tensors:
            a = np.asarray(tensors[name])
            total += a.size
            nnz += int(np.count_nonzero(a))
    reach = int(max((np.max(row) - np.min(row) for row in keys), default=0))
    return {
        "n_device_blocks": int(block_sizes.size),
        "block_dof_min": int(np.min(block_sizes)),
        "block_dof_max": int(np.max(block_sizes)),
        "vertex_blocks": int(keys.shape[0]),
        "transport_reach": reach,
        "stored_density": nnz / max(total, 1),
        "stored_nnz": nnz,
        "stored_entries": total,
    }


def pole_gate(path: str | None) -> dict[str, Any] | None:
    if path is None or not Path(path).exists():
        return None
    p = np.load(path, allow_pickle=False)
    z = np.asarray(p["poles"], dtype=complex)
    gamma = -z.imag
    widths = np.asarray(p["local_frequency_weights"], dtype=float)
    h = float(np.median(widths))
    ratio = gamma[gamma > 0] / max(h, 1e-300)
    offsets = np.asarray(p["pole_offsets"], dtype=int)
    out = {
        "clusters": int(offsets.size - 1),
        "poles": int(z.size),
        "cell_width": h,
        "gamma_over_h": {
            "min": float(np.min(ratio)) if ratio.size else None,
            "p10": float(np.quantile(ratio, 0.1)) if ratio.size else None,
            "median": float(np.median(ratio)) if ratio.size else None,
            "p90": float(np.quantile(ratio, 0.9)) if ratio.size else None,
        },
    }
    if "source_fit" in p:
        fit = np.asarray(p["source_fit"], dtype=float)
        fit = fit[np.isfinite(fit)]
        if fit.size:
            out["source_variation"] = {
                "median": float(np.median(fit)),
                "p90": float(np.quantile(fit, 0.9)),
                "max": float(np.max(fit)),
                "fraction_at_or_below_0.1": float(np.mean(fit <= 0.1)),
            }
    return out


def auxiliary_gate(path: str | None, label: str | None) -> dict[str, Any] | None:
    if path is None or not Path(path).exists():
        return None
    row = json.loads(Path(path).read_text())
    if label is not None:
        row = row[label]
    return {
        "constant_source_error_median": row[
            "passive_constant_source_congruence_error"]["median"],
        "output_rank_at_1e3": row["integrated_physical_rank"]["0.001"][
            "max"],
        "woodbury_over_reblock_at_1e3": row[
            "woodbury_over_two_cell_reblock_cost"]["0.001"]["max_rank"],
    }


def minimum_reblock_for_full_band(radius: int) -> int:
    """Non-overlapping BTD supercell size covering every primitive pair <= R.

    A size-c supercell BTD reaches distance 2c-1, but does not contain every
    pair at that distance.  The boundary pair (c-1, c-1+R) proves c >= R is
    necessary; c=R is sufficient.
    """
    return max(1, int(radius))


def choose_actions(spec: CaseSpec, reference: dict[str, Any],
                   spectral: dict[str, Any], fc3: dict[str, Any],
                   poles: dict[str, Any] | None,
                   auxiliary: dict[str, Any] | None) -> dict[str, Any]:
    reach = 2 * fc3["transport_reach"] + spec.green_band
    spread = reference["internal_spread"]
    if not reference["certified"]:
        spatial = "certify the uncompressed SCBA fixed point first"
    elif spread is not None and spread <= 1e-3:
        spatial = ("retain the current exact/reblocked near pattern; expand "
                   "only if a shell-current certificate fails")
    elif reach > spec.certified_exact_band:
        spatial = (f"complete exact shells {spec.certified_exact_band + 1}.."
                   f"{reach} with a block-banded selected solve; test a tail "
                   "only beyond")
    else:
        spatial = "spatial support is already exact; diagnose frequency/mixing"

    gamma_min = (poles["gamma_over_h"]["min"] if poles else
                 spec.known_min_gamma_over_h)
    if not reference["certified"]:
        frequency = "pilot only; do not train an SCBA representation"
    elif gamma_min is None or gamma_min >= 1.0:
        frequency = "adaptive polynomial cells; no rational promotion"
    elif auxiliary is None:
        frequency = ("selective passive clusters only after source/output-rank "
                     "gates; use local multiresolution cells as fallback")
    elif (auxiliary["constant_source_error_median"] <= 2e-3 and
          auxiliary["woodbury_over_reblock_at_1e3"] < 1.0):
        frequency = "passive rational promotion is cheaper than reblock"
    else:
        frequency = ("reject wholesale promotion; select FC3-important passive "
                     "clusters and locally refine every rejected cluster")

    if (spec.flat_factor_error is not None and spec.flat_factor_error > 1e-2
            and spec.sparse_kernel_time_ratio is not None
            and spec.sparse_kernel_time_ratio < 1.0):
        vertex = "exact atom-sparse FC3 action"
    elif spec.tensor_factor_validated:
        vertex = "validated tensor-factor FC3 action"
    elif fc3["stored_density"] <= 0.15:
        vertex = "benchmark exact sparse FC3; keep dense as oracle"
    else:
        vertex = "fit/validate structured tensor factors; keep dense as oracle"

    return {
        "reference_first": not reference["certified"],
        "frequency": frequency,
        "spatial": spatial,
        "vertex": vertex,
        "support_radius_upper_bound": reach,
        "minimum_single_reblock_for_full_band": minimum_reblock_for_full_band(
            reach),
        "multiresolution_points_at_1e3": spectral["comparison"]["0.001"][
            "adaptive_points"],
        "far_compression_gate": spec.far_compression_gate,
    }


def analyse_case(spec: CaseSpec) -> dict[str, Any]:
    reference = reference_gate(spec.run)
    spectral = spectral_gate(spec.spectral_run)
    fc3 = fc3_gate(spec.fc3)
    poles = pole_gate(spec.pole_states)
    if poles is None and spec.known_min_gamma_over_h is not None:
        poles = {"gamma_over_h": {"min": spec.known_min_gamma_over_h},
                 "source": "documented production census"}
    elif poles is not None and spec.pole_evidence is not None:
        poles["source"] = spec.pole_evidence
    auxiliary = auxiliary_gate(spec.auxiliary_gate, spec.auxiliary_label)
    return {
        "spec": _portable_spec(spec),
        "reference": reference,
        "spectral_pilot": spectral,
        "fc3": fc3,
        "poles": poles,
        "auxiliary": auxiliary,
        "decision": choose_actions(spec, reference, spectral, fc3, poles,
                                   auxiliary),
    }


def _load_manifest(path: str) -> list[CaseSpec]:
    raw = json.loads(Path(path).read_text())
    return [CaseSpec(**row) for row in raw]


def _portable_spec(spec: CaseSpec) -> dict[str, Any]:
    """Keep the recorded JSON independent of the local checkout path."""
    row = asdict(spec)
    root = Path(__file__).resolve().parents[2]
    for key in ("run", "spectral_run", "fc3", "pole_states",
                "auxiliary_gate"):
        value = row.get(key)
        if value is None:
            continue
        try:
            row[key] = str(Path(value).resolve().relative_to(root))
        except ValueError:
            pass
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cases = _load_manifest(args.manifest) if args.manifest else default_cases(root)
    report = {spec.name: analyse_case(spec) for spec in cases}
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    for name, row in report.items():
        d = row["decision"]
        print(f"{name}: certified={row['reference']['certified']}; "
              f"freq={d['frequency']}; spatial={d['spatial']}; "
              f"vertex={d['vertex']}")
    print(f"WROTE {target}")


if __name__ == "__main__":
    main()
