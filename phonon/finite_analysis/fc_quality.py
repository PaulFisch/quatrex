"""FC2/FC3 quality and convergence diagnostics.

Two modes:

  * **Single-reap quality** (default): on a single bundle, computes the
    distance-binned FC2/FC3 magnitudes (cf. ``analysis/compare_hiphive_vs_fd``)
    and a phonon dispersion along the periodic axis. This is the "is the
    FC sensible at all?" check.

  * **Convergence sweep** (optional): if the YAML config includes a
    ``convergence:`` section listing extra reap directories
    (``[{name, fc3_path}, ...]``), the routine loads each one and reports
    how the FC2/FC3 norms and the Γ-point dispersion change. This catches
    under-converged supercell sizes / displacement amplitudes / hiphive
    cutoffs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._utils import min_image_distance_matrix
from .constants import THZ_TO_CM1
from .loader import SystemBundle, load_system


# --------------------------------------------------------------------------- #
# Distance-binned FC2/FC3 norms                                               #
# --------------------------------------------------------------------------- #


def fc2_distance_bins(bundle: SystemBundle) -> dict[str, np.ndarray]:
    """Mean and count of ``\\|Φ₂_{ij}\\|_F`` per unique distance shell."""
    fc2 = bundle.fc2
    n = fc2.shape[0]
    norms = np.linalg.norm(fc2.reshape(n, n, 9), axis=2)
    d = min_image_distance_matrix(bundle.sc_positions, bundle.sc_cell)
    rd = np.round(d, 4)
    uniq = np.unique(rd)
    means = np.zeros_like(uniq)
    counts = np.zeros_like(uniq, dtype=int)
    for k, du in enumerate(uniq):
        m = rd == du
        means[k] = norms[m].mean()
        counts[k] = int(m.sum())
    return {"distance_A": uniq, "mean_norm": means, "count": counts}


def fc3_distance_bins(bundle: SystemBundle, n_bins: int = 30) -> dict[str, np.ndarray]:
    """Mean ``\\|Φ₃_{ijk}\\|_F`` binned by triplet diameter."""
    fc3 = bundle.fc3_raw
    if fc3.ndim != 6:
        raise ValueError(f"unexpected fc3 shape {fc3.shape}")
    norms = np.linalg.norm(fc3.reshape(*fc3.shape[:3], 27), axis=-1)
    d = min_image_distance_matrix(bundle.sc_positions, bundle.sc_cell)
    n_lead = norms.shape[0]
    if n_lead != norms.shape[1]:
        # Compact (nat_prim, n, n) — use primitive→supercell mapping.
        p2s = bundle.phonon.primitive.p2s_map.astype(int)
        i_idx = p2s
    else:
        i_idx = np.arange(n_lead)

    n = d.shape[0]
    diam = np.empty(len(i_idx) * n * n)
    val = np.empty_like(diam)
    flat = 0
    for a, ip in enumerate(i_idx):
        for j in range(n):
            for k in range(n):
                diam[flat] = max(d[ip, j], d[ip, k], d[j, k])
                val[flat] = norms[a, j, k]
                flat += 1

    edges = np.linspace(0.0, diam.max() + 1e-9, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = (diam >= edges[b]) & (diam < edges[b + 1])
        if sel.any():
            means[b] = val[sel].mean()
            counts[b] = int(sel.sum())
    return {"diameter_A": centers, "mean_norm": means, "count": counts}


# --------------------------------------------------------------------------- #
# Dispersion at a few high-symmetry q                                         #
# --------------------------------------------------------------------------- #


def _q_path_for_axis(transport_axis: int) -> dict[str, list[float]]:
    base = [0.0, 0.0, 0.0]
    end = [0.0, 0.0, 0.0]
    end[transport_axis] = 0.5
    return {"Gamma": base, "Z(0.25)": [0.0 if i != transport_axis else 0.25 for i in range(3)],
            "Z(0.5)": end}


def freqs_at_high_sym(
    bundle: SystemBundle, q_path: dict[str, list[float]] | None = None,
) -> dict[str, np.ndarray]:
    q_path = q_path or _q_path_for_axis(bundle.transport_axis)
    bundle.phonon.force_constants = bundle.fc2
    bundle.phonon.run_qpoints(list(q_path.values()))
    f = np.asarray(bundle.phonon.get_qpoints_dict()["frequencies"])  # (nq, nb)
    return {name: f[i] for i, name in enumerate(q_path.keys())}


def plot_dispersion_compare(
    bundles: dict[str, SystemBundle], out_path: Path, *, n_q: int = 21
) -> None:
    """Overlay Γ→Z bands across multiple bundles."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cmap = plt.colormaps["tab10"]
    for idx, (label, b) in enumerate(bundles.items()):
        b.phonon.force_constants = b.fc2
        qs = np.zeros((n_q, 3))
        qs[:, b.transport_axis] = np.linspace(0.0, 0.5, n_q)
        b.phonon.run_qpoints(qs.tolist())
        f = np.asarray(b.phonon.get_qpoints_dict()["frequencies"]) * THZ_TO_CM1
        for band in range(f.shape[1]):
            ax.plot(qs[:, b.transport_axis], f[:, band],
                    color=cmap(idx % 10), lw=0.5, alpha=0.5,
                    label=label if band == 0 else None)
    ax.set_xlabel(r"$q$ along periodic axis")
    ax.set_ylabel(r"$\omega$  [cm$^{-1}$]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Plotting helpers                                                            #
# --------------------------------------------------------------------------- #


def plot_fc2_distance(bins: dict[str, np.ndarray], out_path: Path, system_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.semilogy(bins["distance_A"], bins["mean_norm"], "o-", lw=1.3, ms=4)
    ax.set_xlabel("interatomic distance [Å]")
    ax.set_ylabel(r"mean $\|\Phi_{2,ij}\|_F$  [eV/Å²]")
    ax.set_title(f"FC2 magnitude vs distance — {system_name}")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


def plot_fc3_distance(bins: dict[str, np.ndarray], out_path: Path, system_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.semilogy(bins["diameter_A"], bins["mean_norm"], "o-", lw=1.3, ms=4)
    ax.set_xlabel("triplet diameter [Å]")
    ax.set_ylabel(r"mean $\|\Phi_{3,ijk}\|_F$  [eV/Å³]")
    ax.set_title(f"FC3 magnitude vs triplet diameter — {system_name}")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Convergence sweep                                                           #
# --------------------------------------------------------------------------- #


def within_cutoff_comparison(
    bundle_a: SystemBundle, bundle_b: SystemBundle,
    *,
    fc2_cutoff_A: float, fc3_cutoff_A: float,
) -> dict:
    """Within-cutoff RMS / rel-RMS comparison of two FC sources.

    Promoted from the legacy ``analysis/compare_hiphive_vs_fd.py``: only
    contributions inside the named cutoffs (FC2 pairwise / FC3 triplet
    diameter) are compared, which is the only fair comparison when one of
    the FC sources (e.g. hiphive) is structurally limited to a real-space
    cutoff while the other (FD) extracts at all representable distances.
    """
    fc2_a, fc2_b = bundle_a.fc2, bundle_b.fc2
    fc3_a, fc3_b = bundle_a.fc3_raw, bundle_b.fc3_raw
    if fc2_a.shape != fc2_b.shape or fc3_a.shape != fc3_b.shape:
        raise ValueError("Bundles must share supercell shape (FC2 and FC3).")

    n = fc2_a.shape[0]
    d = min_image_distance_matrix(bundle_a.sc_positions, bundle_a.sc_cell)

    within2 = d <= fc2_cutoff_A
    diff2 = (fc2_b - fc2_a)[within2]
    a2 = fc2_a[within2]
    fc2_rms = float(np.sqrt(np.mean(diff2 ** 2)))
    fc2_rel = float(fc2_rms / np.sqrt(np.mean(a2 ** 2))) if np.any(a2) else float("nan")

    diam = np.maximum(np.maximum(d[:, :, None], d[:, None, :]), d[None, :, :])
    mask3 = diam <= fc3_cutoff_A
    diff3 = (fc3_b - fc3_a)[mask3]
    a3 = fc3_a[mask3]
    fc3_rms = float(np.sqrt(np.mean(diff3 ** 2)))
    fc3_rel = float(fc3_rms / np.sqrt(np.mean(a3 ** 2))) if np.any(a3) else float("nan")

    return {
        "units": {"fc2_rms": "eV/Å²", "fc3_rms": "eV/Å³"},
        "fc2_cutoff_A": float(fc2_cutoff_A),
        "fc3_cutoff_A": float(fc3_cutoff_A),
        "fc2_n_within": int(within2.sum()),
        "fc2_rms": fc2_rms, "fc2_rel_rms": fc2_rel,
        "fc3_n_within": int(mask3.sum()),
        "fc3_rms": fc3_rms, "fc3_rel_rms": fc3_rel,
    }


def load_convergence_variants(
    config_path: Path, variants: Iterable[dict]
) -> dict[str, SystemBundle]:
    """Load each ``{name, fc3_path}`` entry into a separate bundle."""
    out = {}
    for v in variants:
        name = v["name"]
        fc3_path = v["fc3_path"]
        out[name] = load_system(config_path, name=name, fc3_path_override=fc3_path)
    return out


def convergence_table(bundles: dict[str, SystemBundle]) -> list[dict]:
    """Per-bundle FC2/FC3 norms and Γ acoustic-mode magnitude."""
    rows = []
    for name, b in bundles.items():
        f0 = freqs_at_high_sym(b, {"Gamma": [0.0, 0.0, 0.0]})["Gamma"]
        rows.append({
            "name": name,
            "n_super": b.n_super,
            "fc2_max": float(np.max(np.abs(b.fc2))),
            "fc2_frob": float(np.linalg.norm(b.fc2)),
            "fc3_max": float(np.max(np.abs(b.fc3_raw))),
            "fc3_frob": float(np.linalg.norm(b.fc3_raw)),
            "max_acoustic_at_gamma_thz": float(np.abs(f0[:3]).max()),
            "highest_optical_thz": float(f0.max()),
        })
    return rows


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def run_fc_quality(
    bundle: SystemBundle,
    out_dir: Path,
    *,
    convergence_variants: list[dict] | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fc2_bins = fc2_distance_bins(bundle)
    fc3_bins = fc3_distance_bins(bundle)
    plot_fc2_distance(fc2_bins, out_dir / "fc_quality_fc2_distance.png", bundle.name)
    plot_fc3_distance(fc3_bins, out_dir / "fc_quality_fc3_distance.png", bundle.name)

    high_sym = freqs_at_high_sym(bundle)
    units = {
        "fc2_max": "eV/Å²", "fc2_frob": "eV/Å²",
        "fc3_max": "eV/Å³", "fc3_frob": "eV/Å³",
        "high_sym_freqs": "THz",
    }
    summary = {
        "units": units,
        "fc2_max": float(np.max(np.abs(bundle.fc2))),
        "fc2_frob": float(np.linalg.norm(bundle.fc2)),
        "fc3_max": float(np.max(np.abs(bundle.fc3_raw))),
        "fc3_frob": float(np.linalg.norm(bundle.fc3_raw)),
        "high_sym_freqs": {k: v.tolist() for k, v in high_sym.items()},
    }

    if convergence_variants:
        bundles = load_convergence_variants(
            Path(bundle.meta["config_path"]), convergence_variants,
        )
        # Insert the original bundle as the reference.
        bundles = {bundle.name: bundle, **bundles}
        rows = convergence_table(bundles)
        plot_dispersion_compare(
            bundles, out_dir / "fc_quality_dispersion.png"
        )
        summary["convergence_rows"] = rows
    else:
        # Single-bundle dispersion plot for completeness.
        plot_dispersion_compare(
            {bundle.name: bundle}, out_dir / "fc_quality_dispersion.png"
        )

    (out_dir / "fc_quality.json").write_text(json.dumps(summary, indent=2))
    return summary
