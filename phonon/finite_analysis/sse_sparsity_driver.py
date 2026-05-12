"""High-level wiring: synthetic-GF SSE + cutoff sweep, with optional
quatrex SCBA cross-check.

Two driver functions are exported, both expected by the CLI:

  * :func:`run_sse_sparsity` — computes the bubble on a synthetic G,
    saves a per-(I,J) Frobenius CSV plus a heatmap. If ``run_quatrex``
    is True it also runs a short SCBA via the quatrex solver and saves
    a second heatmap; otherwise that figure is skipped with a notice.
  * :func:`run_cutoffs` — runs the standard cutoff grid against
    the synthetic-GF baseline, saves per-config Frobenius diffs, a bar
    chart, and (when transport_metrics is implemented) a
    transmission/heat-current overlay.

Both functions return a small dict suitable for ``summary.json``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .loader import SystemBundle
from .sse_cutoffs import (
    build_sse_inputs,
    compute_sse_with_cutoffs,
    run_sse_cutoffs,
    block_frob_diff,
    standard_cutoff_grid,
    SSEResult,
)


# --------------------------------------------------------------------------- #
# Plot helpers                                                                #
# --------------------------------------------------------------------------- #


def sigma_block_heatmap(
    block_frob: dict[tuple[int, int], tuple[float, float]],
    n_blocks: int,
    out_path: Path,
    title: str,
) -> None:
    """Render the per-(I,J) Σ^< Frobenius norm as a slab × slab heatmap.

    The grid is sized ``n_blocks × n_blocks`` and shows every entry present
    in ``block_frob`` — so off-tridiagonal blocks appear automatically when
    the upstream ``compute_sse_with_cutoffs`` was called with
    ``sigma_block_distance > 1``. Empty cells stay at the colormap floor.
    """
    grid = np.zeros((n_blocks, n_blocks))
    for (I, J), (lesser, _) in block_frob.items():
        if 0 <= I < n_blocks and 0 <= J < n_blocks:
            grid[I, J] = lesser
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    if (grid > 0).any():
        floor = max(grid[grid > 0].min(), 1e-30)
        im = ax.imshow(
            grid, origin="lower", cmap="magma",
            norm=matplotlib.colors.LogNorm(vmin=floor, vmax=grid.max()),
        )
    else:
        im = ax.imshow(grid, origin="lower", cmap="magma")
    ax.set_xlabel("J (slab)")
    ax.set_ylabel("I (slab)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=r"$\|\Sigma^<_{IJ}\|_F$  [THz²]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


def sigma_atomic_block_heatmap(
    sigma_blocks: dict[tuple[int, int], np.ndarray],
    block_sizes: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Per-atom-pair Frobenius norm of Σ blocks, like ``sparsity_fc2_heatmap``.

    For each Σ_{IJ} block of shape ``(n_freq, 3*nA, 3*nB)`` we reshape to
    ``(n_freq, nA, 3, nB, 3)`` and take the Frobenius norm over
    ``(n_freq, 3, 3)``, giving an ``nA × nB`` atom-pair norm. Concatenated
    along all (I, J) blocks this is the dense ``n_atoms × n_atoms``
    coordinate heatmap. Atoms are in z-sorted order (matching the DOF
    ordering inside the blocks).
    """
    block_sizes = np.asarray(block_sizes, dtype=int)
    n_blocks = block_sizes.size
    atom_sizes = block_sizes // 3
    offsets = np.concatenate(([0], np.cumsum(atom_sizes)))
    n_atoms = int(offsets[-1])

    norms = np.zeros((n_atoms, n_atoms))
    for (I, J), block in sigma_blocks.items():
        if not (0 <= I < n_blocks and 0 <= J < n_blocks):
            continue
        nA = int(atom_sizes[I])
        nB = int(atom_sizes[J])
        # block has shape (n_freq, 3*nA, 3*nB) → (n_freq, nA, 3, nB, 3)
        reshaped = block.reshape(block.shape[0], nA, 3, nB, 3)
        # Frobenius norm over (frequency, 3, 3): per atom-pair scalar.
        sub = np.sqrt(np.sum(np.abs(reshaped) ** 2, axis=(0, 2, 4)))
        norms[offsets[I]:offsets[I+1], offsets[J]:offsets[J+1]] = sub

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    if (norms > 0).any():
        floor = max(norms[norms > 0].min(), 1e-30)
        im = ax.imshow(
            norms, origin="lower", cmap="magma",
            norm=matplotlib.colors.LogNorm(vmin=floor, vmax=norms.max()),
        )
    else:
        im = ax.imshow(norms, origin="lower", cmap="magma")
    # Slab boundaries (dashed lines).
    for k in range(1, n_blocks):
        ax.axvline(offsets[k] - 0.5, color="white", lw=0.4, ls="--", alpha=0.5)
        ax.axhline(offsets[k] - 0.5, color="white", lw=0.4, ls="--", alpha=0.5)
    ax.set_xlabel("atom j (z-sorted)")
    ax.set_ylabel("atom i (z-sorted)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax,
                 label=r"$\|\Sigma^<_{ij}\|_F$  [THz², integrated over $\omega$]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


def _frob_bar_chart(
    diffs: dict[str, dict[tuple[int, int], float]],
    out_path: Path,
) -> None:
    """Mean + max relative Frobenius gap vs the baseline per cutoff config.

    Renders the mean as a coloured bar and the per-(I,J) maximum as an
    overlaid open marker so the user can see both the typical and the
    worst-case effect of each cutoff. Bars are coloured by cutoff family
    (diag-G, FC3 magnitude, FC3 distance).
    """
    labels: list[str] = list(diffs.keys())
    means: list[float] = []
    maxes: list[float] = []
    for lbl in labels:
        vals = [v for v in diffs[lbl].values() if np.isfinite(v)]
        means.append(float(np.mean(vals)) if vals else 0.0)
        maxes.append(float(np.max(vals)) if vals else 0.0)

    # Colour bars by cutoff family.
    family_palette = {
        "baseline": "#888888",
        "diag_G":   "#4477AA",   # Tol blue
        "mag":      "#CCBB44",   # Tol yellow
        "dist":     "#AA3377",   # Tol purple
        "other":    "#228833",   # Tol green
    }

    def _family(lbl: str) -> str:
        s = lbl.lower()
        if "baseline" in s:
            return "baseline"
        if "diag" in s and "g" in s:
            return "diag_G"
        if "mag" in s or "thresh" in s:
            return "mag"
        if "dist" in s:
            return "dist"
        return "other"

    colors = [family_palette[_family(lbl)] for lbl in labels]

    fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * len(labels) + 2), 4.5))
    x = np.arange(len(labels))
    ax.bar(
        x, [m if m > 0 else 1e-30 for m in means], color=colors,
        edgecolor="black", lw=0.5, alpha=0.85, label="mean per (I, J)",
    )
    ax.plot(
        x, [m if m > 0 else 1e-30 for m in maxes],
        "o", color="black", markersize=6, mfc="none", mew=1.4,
        label="max per (I, J)",
    )

    # Annotate each bar with the numeric value.
    for xi, m in zip(x, means, strict=False):
        if m > 0:
            ax.text(
                xi, m * 1.18, f"{m:.1e}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(
        r"$\|\Sigma^<_{\mathrm{cutoff}} - \Sigma^<_{\mathrm{baseline}}\|_F"
        r" / \|\Sigma^<_{\mathrm{baseline}}\|_F$"
    )
    ax.set_title("Cutoff sensitivity of phonon-phonon SSE")
    ax.grid(True, axis="y", which="both", alpha=0.3, lw=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Family legend (manual handles to avoid one entry per bar).
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=family_palette[k], edgecolor="black", lw=0.5,
              label=k) for k in family_palette
        if any(_family(lbl) == k for lbl in labels)
    ]
    handles.append(plt.Line2D(
        [], [], marker="o", color="black", mfc="none", mew=1.4,
        ls="", markersize=6, label="max per (I, J)",
    ))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(Path(out_path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _csv_block_frob(
    block_frob: dict[tuple[int, int], tuple[float, float]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["I", "J", "frob_lesser", "frob_greater"])
        for (I, J), (l, g) in sorted(block_frob.items()):
            w.writerow([I, J, f"{l:.6e}", f"{g:.6e}"])


# --------------------------------------------------------------------------- #
# Drivers                                                                     #
# --------------------------------------------------------------------------- #


def run_sse_sparsity(
    bundle: SystemBundle,
    out_dir: Path,
    *,
    n_freq_pos: int = 64,
    eta_thz: float | None = None,
    temperature_k: float = 300.0,
    run_quatrex: bool = False,
    sigma_block_distance: int | None = None,
) -> dict:
    """Σ block-norm heatmap from the synthetic GF, optionally cross-checked
    with the quatrex SCBA.

    ``sigma_block_distance`` controls the largest ``|I - J|`` block computed
    for the synthetic branch; default ``n_blocks - 1`` (the full matrix), so
    the heatmap visualises off-tridiagonal contributions. Pass ``1`` to
    recover the transport-solver-equivalent tridiagonal restriction.
    The quatrex SCBA cross-check is intrinsically tridiagonal.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phi_blocks, gl_blocks, gg_blocks, _freqs, dw, modes = build_sse_inputs(
        bundle, n_freq_pos=n_freq_pos, eta_thz=eta_thz,
        temperature_k=temperature_k,
    )
    n_blocks = int(np.asarray(bundle.block_sizes).size)
    if sigma_block_distance is None:
        d_sigma = n_blocks - 1
    else:
        d_sigma = max(1, min(int(sigma_block_distance), n_blocks - 1))

    res = compute_sse_with_cutoffs(
        phi_blocks, gl_blocks, gg_blocks, bundle.block_sizes, dw,
        sigma_block_distance=d_sigma,
    )
    _csv_block_frob(res.block_frob, out_dir / "sse_sigma_block_frob_synth.csv")
    sigma_block_heatmap(
        res.block_frob, bundle.n_slabs,
        out_dir / "sse_sigma_heatmap_synth.png",
        title=(f"Σ^< block Frobenius — synthetic GF — {bundle.name}  "
               f"(|I-J| ≤ {d_sigma})"),
    )
    sigma_atomic_block_heatmap(
        res.sigma_lesser, bundle.block_sizes,
        out_dir / "sse_sigma_atomic_heatmap_synth.png",
        title=(f"Σ^< atomic-block Frobenius — synthetic GF — {bundle.name}  "
               f"(|I-J| ≤ {d_sigma})"),
    )

    quatrex_status = "skipped"
    if run_quatrex:
        # The dense bubble cross-check allocates a (n_fft × n_dof³) complex
        # intermediate. n_dof is the FULL supercell DOF count, so for any
        # SiNW-scale system this is tens to hundreds of GiB and stalls the
        # analysis on a single core. Guard hard at 30 GiB; the user can
        # still force-run by passing run_quatrex=True directly to this
        # function from Python (the guard only fires when the projected
        # buffer is enormous).
        n_dof = int(np.sum(bundle.block_sizes))
        n_fft = 2 * (2 * n_freq_pos + 1) - 1
        gib = n_fft * n_dof ** 3 * 16 / (1024 ** 3)
        if gib > 30.0:
            quatrex_status = (
                f"skipped (dense-bubble intermediate would be "
                f"~{gib:.0f} GiB at n_dof={n_dof}, n_fft={n_fft}; cross-check "
                f"is a chain/primitive-cell regression test, not a "
                f"production tool — pass --with-quatrex-crosscheck only on "
                f"≤ 50-atom supercells)"
            )
            print(f"sse_sparsity: {quatrex_status}")
            return {
                "units": {"dw": "THz"},
                "n_phi_blocks": len(phi_blocks),
                "n_sigma_blocks": len(res.block_frob),
                "n_unstable_modes": modes.n_unstable,
                "dw": float(dw),
                "quatrex_status": quatrex_status,
            }
        from .sse_quatrex_run import run_dense_scba_crosscheck
        quatrex_blocks = run_dense_scba_crosscheck(
            bundle, n_freq_pos=n_freq_pos, eta_thz=eta_thz,
            temperature_k=temperature_k, n_iter=2,
        )
        _csv_block_frob(
            quatrex_blocks, out_dir / "sse_sigma_block_frob_quatrex.csv",
        )
        sigma_block_heatmap(
            quatrex_blocks, bundle.n_slabs,
            out_dir / "sse_sigma_heatmap_quatrex.png",
            title=f"Σ^< block Frobenius — quatrex SCBA — {bundle.name}",
        )
        # The quatrex SCBA returns the per-(I,J) Frobenius norms only, not the
        # full block tensors, so the atomic-coordinate heatmap is only produced
        # for the synthetic-GF branch above.
        quatrex_status = "ok"

    return {
        "units": {"dw": "THz"},
        "n_phi_blocks": len(phi_blocks),
        "n_sigma_blocks": len(res.block_frob),
        "n_unstable_modes": modes.n_unstable,
        "dw": float(dw),
        "quatrex_status": quatrex_status,
    }


def run_cutoffs(
    bundle: SystemBundle,
    out_dir: Path,
    *,
    n_freq_pos: int = 64,
    eta_thz: float | None = None,
    temperature_k: float = 300.0,
    gamma_lead_thz: float | None = None,
    T_L: float | None = None,
    T_R: float | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phi_blocks, gl_blocks, gg_blocks, freqs, dw, _ = build_sse_inputs(
        bundle, n_freq_pos=n_freq_pos, eta_thz=eta_thz,
        temperature_k=temperature_k,
    )

    grid = standard_cutoff_grid()
    results = run_sse_cutoffs(
        bundle, gl_blocks, gg_blocks, phi_blocks, dw, cutoff_grid=grid,
    )
    baseline = results["baseline"]
    diffs = {label: block_frob_diff(baseline, res) for label, res in results.items()}
    _frob_bar_chart(diffs, out_dir / "cutoffs_sigma_frob_vs_cutoff.png")

    with open(out_dir / "cutoffs_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "mean_rel_diff_lesser", "max_rel_diff_lesser"])
        for label, d in diffs.items():
            finite = [v for v in d.values() if np.isfinite(v)]
            mean = float(np.mean(finite)) if finite else 0.0
            mx = float(np.max(finite)) if finite else 0.0
            w.writerow([label, f"{mean:.6e}", f"{mx:.6e}"])

    # Wire transport observables: per-cutoff T(ω) and Q.
    from .constants import (
        DEFAULT_GAMMA_LEAD_THZ, DEFAULT_T_L_K, DEFAULT_T_R_K,
        DEFAULT_ETA_THZ_TRANSPORT,
    )
    from .transport_metrics import (
        transport_for_cutoff_sweep, plot_transport_compare,
    )
    if gamma_lead_thz is None:
        gamma_lead_thz = DEFAULT_GAMMA_LEAD_THZ
    if T_L is None:
        T_L = DEFAULT_T_L_K
    if T_R is None:
        T_R = DEFAULT_T_R_K
    traces = transport_for_cutoff_sweep(
        bundle, results, freqs,
        eta_thz=DEFAULT_ETA_THZ_TRANSPORT,
        gamma_lead_thz=gamma_lead_thz, T_L=T_L, T_R=T_R,
    )
    plot_transport_compare(
        traces, out_dir / "cutoffs_transport.png", system_name=bundle.name,
    )
    with open(out_dir / "cutoffs_transport.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "T_max", "Q_W"])
        for label, tr in traces.items():
            w.writerow([
                label, f"{float(np.max(tr.transmission)):.6e}",
                f"{tr.heat_current_W:.6e}",
            ])

    return {
        "units": {"mean_rel_diff": "dimensionless (relative)", "Q": "W"},
        "labels": [c.label for c in grid],
        "mean_rel_diff": {
            label: float(np.mean([v for v in d.values() if np.isfinite(v)]))
            if d else 0.0
            for label, d in diffs.items()
        },
        "heat_current_W": {label: tr.heat_current_W for label, tr in traces.items()},
    }


# --------------------------------------------------------------------------- #
# Off-tridiagonal Σ audit                                                     #
# --------------------------------------------------------------------------- #


def _sigma_block_frob_by_distance(
    result: SSEResult, n_blocks: int,
) -> dict[int, dict[str, float]]:
    """Aggregate ‖Σ^{<,>}_{IJ}‖_F by ``|I - J|``.

    Returns ``{d: {"sum_lesser": s_l, "max_lesser": m_l,
                   "sum_greater": s_g, "max_greater": m_g, "count": n}}``.
    """
    out: dict[int, dict[str, float]] = {}
    for (I, J), block in result.sigma_lesser.items():
        d = abs(I - J)
        e = out.setdefault(d, {"sum_lesser": 0.0, "max_lesser": 0.0,
                               "sum_greater": 0.0, "max_greater": 0.0,
                               "count": 0})
        norm = float(np.linalg.norm(block))
        e["sum_lesser"] += norm
        e["max_lesser"] = max(e["max_lesser"], norm)
        e["count"] += 1
    for (I, J), block in result.sigma_greater.items():
        d = abs(I - J)
        e = out.setdefault(d, {"sum_lesser": 0.0, "max_lesser": 0.0,
                               "sum_greater": 0.0, "max_greater": 0.0,
                               "count": 0})
        norm = float(np.linalg.norm(block))
        e["sum_greater"] += norm
        e["max_greater"] = max(e["max_greater"], norm)
    return out


def run_sigma_block_audit(
    bundle: SystemBundle,
    out_dir: Path,
    *,
    n_freq_pos: int = 64,
    eta_thz: float | None = None,
    temperature_k: float = 300.0,
    max_block_distance: int | None = None,
) -> dict:
    """Compute Σ on all block pairs up to ``max_block_distance`` and report
    how Frobenius weight decays with ``|I - J|``.

    Writes:
      - ``sigma_block_decay.csv`` — per-distance sum / max / fraction-of-total
      - ``sigma_block_decay.png`` — semilogy bar chart of max-norm vs distance

    Default ``max_block_distance = n_blocks - 1`` (full block matrix). Cost
    scales roughly cubically with the bandwidth, so use ``--sigma-max-dist``
    on the CLI to cap it if the wire is long.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phi_blocks, gl_blocks, gg_blocks, _freqs, dw, _ = build_sse_inputs(
        bundle, n_freq_pos=n_freq_pos, eta_thz=eta_thz,
        temperature_k=temperature_k,
    )
    n_blocks = int(np.asarray(bundle.block_sizes).size)
    d_max = n_blocks - 1 if max_block_distance is None else int(max_block_distance)
    d_max = max(1, min(d_max, n_blocks - 1))

    result = compute_sse_with_cutoffs(
        phi_blocks, gl_blocks, gg_blocks,
        bundle.block_sizes, dw,
        sigma_block_distance=d_max,
    )
    decay = _sigma_block_frob_by_distance(result, n_blocks)
    total_l = sum(e["sum_lesser"] for e in decay.values()) or 1.0
    total_g = sum(e["sum_greater"] for e in decay.values()) or 1.0

    with open(out_dir / "sigma_block_decay.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "block_distance", "count",
            "sum_frob_lesser", "max_frob_lesser", "frac_total_lesser",
            "sum_frob_greater", "max_frob_greater", "frac_total_greater",
        ])
        for d in sorted(decay):
            e = decay[d]
            w.writerow([
                d, e["count"],
                f"{e['sum_lesser']:.6e}", f"{e['max_lesser']:.6e}",
                f"{e['sum_lesser'] / total_l:.6e}",
                f"{e['sum_greater']:.6e}", f"{e['max_greater']:.6e}",
                f"{e['sum_greater'] / total_g:.6e}",
            ])

    distances = sorted(decay)
    max_l = [decay[d]["max_lesser"] for d in distances]
    max_g = [decay[d]["max_greater"] for d in distances]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(distances, max_l, "o-", label=r"max $\|\Sigma^<_{IJ}\|_F$")
    ax.semilogy(distances, max_g, "s--", label=r"max $\|\Sigma^>_{IJ}\|_F$")
    ax.axvline(1.0, color="grey", lw=0.8, ls=":",
               label="tridiagonal transport cutoff")
    ax.set_xlabel("block distance $|I - J|$")
    ax.set_ylabel("Frobenius norm of $\\Sigma_{IJ}$")
    ax.set_title(f"{bundle.name}: Σ block-norm decay (full bubble)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "sigma_block_decay.png", dpi=150)
    plt.close(fig)

    # Headline number: fraction of weight beyond the transport solver's cutoff.
    frac_l_kept = sum(decay[d]["sum_lesser"] for d in decay if d <= 1) / total_l
    frac_g_kept = sum(decay[d]["sum_greater"] for d in decay if d <= 1) / total_g
    return {
        "max_block_distance": d_max,
        "frac_lesser_in_tridiagonal": float(frac_l_kept),
        "frac_greater_in_tridiagonal": float(frac_g_kept),
        "max_frob_lesser_by_distance": {
            int(d): float(decay[d]["max_lesser"]) for d in distances
        },
        "max_frob_greater_by_distance": {
            int(d): float(decay[d]["max_greater"]) for d in distances
        },
    }
