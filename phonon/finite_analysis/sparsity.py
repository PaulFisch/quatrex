"""Sparsity diagnostics for FC2 and FC3 on a finite system.

Produces three artifacts the writeup needs:

  * FC2 atomic-block-Frobenius heatmap (log color),
  * FC3 1D Frobenius decay vs. triplet diameter,
  * FC3 3D scatter of nonzero (i, j, k) entries.

Plus a ``nnz_table`` quantifying how many entries survive a relative
magnitude threshold ε ∈ {10⁻², 10⁻³, 10⁻⁴, 10⁻⁵}. The 3D scatter is
thresholded so the total point count stays in matplotlib-friendly range
(≤ ~5×10⁴ points by default) — readers care about *where* the support is,
not seeing every speck.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._summary import make_summary
from .loader import SystemBundle


# --------------------------------------------------------------------------- #
# Distance helpers                                                            #
# --------------------------------------------------------------------------- #


from ._utils import min_image_distance_matrix, triplet_diameter



# --------------------------------------------------------------------------- #
# FC2 heatmap                                                                 #
# --------------------------------------------------------------------------- #


def fc2_atomic_block_norms(fc2: np.ndarray) -> np.ndarray:
    """Per-atom-pair Frobenius norm of FC2 blocks. Returns ``(n, n)``."""
    n = fc2.shape[0]
    return np.linalg.norm(fc2.reshape(n, n, 9), axis=2)


def plot_fc2_heatmap(
    bundle: SystemBundle, out_path: Path, *, log_color: bool = True
) -> None:
    """FC2 atomic-block Frobenius heatmap, sorted by transport-axis position."""
    norms = fc2_atomic_block_norms(bundle.fc2)
    perm = bundle.atom_perm
    norms = norms[np.ix_(perm, perm)]

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    if log_color:
        floor = max(norms[norms > 0].min() if (norms > 0).any() else 1e-12, 1e-12)
        im = ax.imshow(
            norms, origin="lower", cmap="viridis",
            norm=matplotlib.colors.LogNorm(vmin=floor, vmax=norms.max() or 1.0),
        )
    else:
        im = ax.imshow(norms, origin="lower", cmap="viridis")
    ax.set_xlabel("atom j (z-sorted)")
    ax.set_ylabel("atom i (z-sorted)")
    ax.set_title(f"FC2 atomic-block Frobenius norm — {bundle.name}")
    fig.colorbar(im, ax=ax, label=r"$\|\Phi_{2,ij}\|_F$  [eV/Å²]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# FC3 1D decay                                                                #
# --------------------------------------------------------------------------- #


def fc3_atomic_block_norms(fc3: np.ndarray) -> np.ndarray:
    """Per-atom-triplet Frobenius norm of FC3 blocks. Returns ``(n, n, n)``.

    Accepts either the compact ``(nat_prim, n, n, 3, 3, 3)`` or full
    ``(n, n, n, 3, 3, 3)`` phono3py layout. The compact layout is left
    as-is (returns ``(nat_prim, n, n)``); the caller must be aware.
    """
    if fc3.ndim != 6:
        raise ValueError(f"fc3 must be 6-D, got {fc3.shape}")
    leading = fc3.shape[:3]
    return np.linalg.norm(fc3.reshape(*leading, 27), axis=-1)


def fc3_decay_1d(
    bundle: SystemBundle, *, n_bins: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Mean ``\\|Phi_{ijk}\\|_F`` binned by triplet diameter (full FC3 only).

    Returns ``(bin_centers, mean_norm)`` in the ranges actually populated.
    Compact-layout FC3 (``nat_prim`` leading) is silently broadcast over the
    primitive index — the i-axis loops over primitive atoms only.
    """
    norms = fc3_atomic_block_norms(bundle.fc3_raw)
    d = min_image_distance_matrix(bundle.sc_positions, bundle.sc_cell)

    if norms.shape[0] != d.shape[0]:
        # Compact layout: leading axis is primitive, not supercell.
        p2s = bundle.phonon.primitive.p2s_map.astype(int)
        i_idx = p2s
    else:
        i_idx = np.arange(norms.shape[0])

    n_i = len(i_idx)
    n = d.shape[0]
    diameters = np.empty(n_i * n * n, dtype=np.float64)
    values = np.empty_like(diameters)
    flat = 0
    for a, ip in enumerate(i_idx):
        for j in range(n):
            for k in range(n):
                diameters[flat] = max(d[ip, j], d[ip, k], d[j, k])
                values[flat] = norms[a, j, k]
                flat += 1

    edges = np.linspace(0.0, diameters.max() + 1e-9, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.zeros(n_bins)
    for b in range(n_bins):
        sel = (diameters >= edges[b]) & (diameters < edges[b + 1])
        if sel.any():
            mean[b] = values[sel].mean()
    keep = mean > 0
    return centers[keep], mean[keep]


def plot_fc3_decay_1d(bundle: SystemBundle, out_path: Path) -> None:
    centers, mean = fc3_decay_1d(bundle)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.semilogy(centers, mean, "o-", lw=1.5, ms=4)
    ax.set_xlabel(r"triplet diameter $\max(d_{ij}, d_{ik}, d_{jk})$  [Å]")
    ax.set_ylabel(r"mean $\|\Phi_{3,ijk}\|_F$  [eV/Å³]")
    ax.set_title(f"FC3 magnitude decay vs. triplet diameter — {bundle.name}")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# FC3 3D scatter                                                              #
# --------------------------------------------------------------------------- #


def plot_fc3_scatter_3d(
    bundle: SystemBundle,
    out_path: Path,
    *,
    rel_threshold: float = 1e-3,
    max_points: int = 50_000,
) -> None:
    """3D scatter of FC3 atomic-triplet support + a 2D projection.

    Two panels side-by-side:
      * Left: the same 3D scatter as before, with a fixed elevation /
        azimuth so triplet bands along the wire axis are legible.
        Marker alpha is reduced so dense clouds don't pile into a
        single dark blob.
      * Right: 2D projection onto the (i, j) plane (max over k) so the
        reader can compare to the FC2 atomic-block heatmap without
        having to mentally collapse the 3D view.
    """
    norms = fc3_atomic_block_norms(bundle.fc3_raw)
    if norms.shape[0] != norms.shape[1]:
        p2s = bundle.phonon.primitive.p2s_map.astype(int)
        i_axis_label = "i (primitive atom → supercell idx)"
        x_coords = p2s
    else:
        i_axis_label = "i (supercell)"
        x_coords = np.arange(norms.shape[0])

    floor = rel_threshold * (norms.max() or 1.0)
    mask = norms > floor
    while mask.sum() > max_points:
        floor *= 2
        mask = norms > floor
    iI, jI, kI = np.nonzero(mask)
    vals = norms[iI, jI, kI]

    fig = plt.figure(figsize=(13.0, 6.0))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.view_init(elev=20, azim=35)
    p3 = ax3d.scatter(
        x_coords[iI], jI, kI,
        c=np.log10(vals), cmap="viridis", s=10, alpha=0.45,
        edgecolor="none",
    )
    ax3d.set_xlabel(i_axis_label)
    ax3d.set_ylabel("j (supercell)")
    ax3d.set_zlabel("k (supercell)")
    ax3d.set_title(
        f"FC3 support — {bundle.name}\n"
        f"({int(mask.sum()):,} pts above {floor:.2e} eV/Å³)"
    )

    # 2D projection: collapse over k by taking max so the strongest
    # triplet involving each (i, j) atom pair is shown.
    proj_ij = norms.max(axis=2)
    if proj_ij.shape[0] != proj_ij.shape[1]:
        # Compact layout — lift the i-axis to supercell indices so the
        # projection is square and the FC2 heatmap is directly comparable.
        proj_full = np.zeros((norms.shape[1], norms.shape[1]))
        proj_full[p2s, :] = proj_ij
        proj_ij = proj_full
    ax2d = fig.add_subplot(1, 2, 2)
    pos = proj_ij > 0
    floor_2d = max(proj_ij[pos].min() if pos.any() else floor, 1e-30)
    im = ax2d.imshow(
        proj_ij, origin="lower", cmap="viridis",
        norm=matplotlib.colors.LogNorm(vmin=floor_2d, vmax=proj_ij.max() or 1.0),
    )
    ax2d.set_xlabel("j (supercell)")
    ax2d.set_ylabel("i (supercell)")
    ax2d.set_title(r"$\max_k\,\|\Phi_{3,ijk}\|_F$  (projection)")
    fig.colorbar(p3, ax=ax3d, label=r"$\log_{10}\|\Phi_{3,ijk}\|_F$  [eV/Å³]",
                 shrink=0.7)
    fig.colorbar(im, ax=ax2d, label=r"$\max_k\,\|\Phi_{3,ijk}\|_F$  [eV/Å³]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# nnz table                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class NnzRow:
    eps: float
    fc2_nnz: int
    fc2_total: int
    fc3_nnz: int
    fc3_total: int


def nnz_table(
    bundle: SystemBundle,
    *,
    eps_list: Iterable[float] = (1e-2, 1e-3, 1e-4, 1e-5),
) -> list[NnzRow]:
    """Count entries of FC2/FC3 above ``eps * max(|.|)`` for several ε."""
    fc2 = bundle.fc2
    fc3 = bundle.fc3_raw
    fc2_max = float(np.max(np.abs(fc2))) or 1.0
    fc3_max = float(np.max(np.abs(fc3))) or 1.0
    rows: list[NnzRow] = []
    for eps in eps_list:
        rows.append(
            NnzRow(
                eps=float(eps),
                fc2_nnz=int(np.count_nonzero(np.abs(fc2) > eps * fc2_max)),
                fc2_total=int(fc2.size),
                fc3_nnz=int(np.count_nonzero(np.abs(fc3) > eps * fc3_max)),
                fc3_total=int(fc3.size),
            )
        )
    return rows


def write_nnz_table_csv(rows: Iterable[NnzRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("eps,fc2_nnz,fc2_total,fc2_density,fc3_nnz,fc3_total,fc3_density\n")
        for r in rows:
            f.write(
                f"{r.eps:.0e},{r.fc2_nnz},{r.fc2_total},"
                f"{r.fc2_nnz / r.fc2_total:.6e},"
                f"{r.fc3_nnz},{r.fc3_total},"
                f"{r.fc3_nnz / r.fc3_total:.6e}\n"
            )


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def run_sparsity(bundle: SystemBundle, out_dir: Path) -> dict:
    """Produce all sparsity figures + nnz table; return a small summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_fc2_heatmap(bundle, out_dir / "sparsity_fc2_heatmap.png")
    plot_fc3_decay_1d(bundle, out_dir / "sparsity_fc3_decay_1d.png")
    plot_fc3_scatter_3d(bundle, out_dir / "sparsity_fc3_scatter_3d.png")
    rows = nnz_table(bundle)
    write_nnz_table_csv(rows, out_dir / "sparsity_nnz_table.csv")
    return make_summary(
        units={"fc2_max": "eV/Å²", "fc3_max": "eV/Å³"},
        fc2_max=float(np.max(np.abs(bundle.fc2))),
        fc3_max=float(np.max(np.abs(bundle.fc3_raw))),
        nnz_eps_1e_3=next(r for r in rows if r.eps == 1e-3).__dict__,
    )
