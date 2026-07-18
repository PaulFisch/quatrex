"""Build a NON-UNIFORM phonon frequency grid for phonon.frequency_grid="file".

Flat phonon bands produce a comb of narrow, well-separated Lorentzians;
a uniform grid resolving the narrowest width Gamma over [0, fmax] wastes
~(1 - sum_s Gamma_s / fmax) of its points on empty gaps. This script
concentrates the points on the mode frequencies instead: the target
point DENSITY is a background floor plus a Lorentzian comb,

    rho(omega) = 1/max_spacing
               + sum_s (pts_per_line / pi) * w_s / ((omega-omega_s)^2 + w_s^2),

capped at 1/min_spacing, and the grid points equidistribute its CDF
(each line receives ~pts_per_line points; the spacing at a line centre
is ~pi*w_s/pts_per_line). The result is written as phonon_energies.npy
for the engine (pair with --freq-grid file and an auxiliary bubble grid
--aux-dw/--aux-fmax; the FFT convolution itself stays uniform).

Mode-frequency sources (exactly one):
  --modes FILE   .npy/.txt list of mode frequencies (THz)
  --npz FILE     previous run.npz: peaks of the DOS -Im Tr G^R
  --dyn FILE     device dynamical matrix (THz^2): eigenfrequencies
                 (best effort: .npy/.npz/.mat with a square array)

Example (a-posteriori refinement from a converged coarse run):
  python phonon/studies/make_grid.py --npz out/base/run.npz \
      --fmax 34 --width-thz 0.05 --pts-per-line 12 \
      --out work/phonon_energies.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _modes_from_file(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=float).ravel()
    return np.loadtxt(path, dtype=float).ravel()


def _modes_from_npz(path: Path, prominence: float) -> np.ndarray:
    """Peak-pick the DOS of a previous run (gr_diag_imag = -Im G^R diag)."""
    from scipy.signal import find_peaks

    d = np.load(path)
    w = np.asarray(d["energies"], dtype=float)
    dos = np.asarray(d["gr_diag_imag"], dtype=float)
    dos = dos.reshape(dos.shape[0], -1).sum(axis=1)
    peaks, _ = find_peaks(dos, prominence=prominence * float(dos.max()))
    if peaks.size == 0:
        raise SystemExit(
            f"No DOS peaks found in {path} at prominence "
            f"{prominence:g}*max; lower --peak-prominence."
        )
    return w[peaks]


def _modes_from_dyn(path: Path) -> np.ndarray:
    """Eigenfrequencies (THz) of a dense dynamical matrix (THz^2)."""
    arrs = []
    if path.suffix == ".npy":
        arrs = [np.load(path)]
    elif path.suffix == ".npz":
        arrs = list(np.load(path).values())
    else:
        from scipy.io import loadmat

        arrs = [v for k, v in loadmat(path).items()
                if not k.startswith("__")]
    sq = [np.asarray(a) for a in arrs
          if np.asarray(a).ndim == 2
          and a.shape[0] == a.shape[1] and a.shape[0] > 1]
    if not sq:
        raise SystemExit(f"No square matrix found in {path}.")
    dyn = max(sq, key=lambda a: a.shape[0]).astype(complex)
    dyn = 0.5 * (dyn + dyn.conj().T)
    lam = np.linalg.eigvalsh(dyn).real
    return np.sqrt(np.clip(lam, 0.0, None))


def build_grid(
    modes: np.ndarray,
    fmax: float,
    width: float | np.ndarray,
    pts_per_line: int,
    max_spacing: float,
    min_spacing: float,
) -> np.ndarray:
    """CDF-equidistributed grid on [0, fmax] for a Lorentzian-comb density."""
    modes = np.asarray(modes, dtype=float).ravel()
    # Broadcast BEFORE filtering so per-line --widths stay index-aligned
    # with their modes (--dyn sources always contain omega = 0 acoustics
    # that the filter drops).
    widths = np.array(
        np.broadcast_to(np.asarray(width, dtype=float), modes.shape))
    keep = (modes > 0.0) & (modes <= fmax)
    modes, widths = modes[keep], widths[keep]
    # Merge lines closer than their width: coincident flat-band modes need
    # points at one location, not multiples of them.
    order = np.argsort(modes)
    modes, widths = modes[order], widths[order]
    keep_m, keep_w = [], []
    for m, w in zip(modes, widths):
        if keep_m and (m - keep_m[-1]) < 0.5 * max(w, keep_w[-1]):
            continue
        keep_m.append(float(m))
        keep_w.append(float(w))
    modes, widths = np.asarray(keep_m), np.asarray(keep_w)

    probe = np.linspace(0.0, fmax, max(4 * int(fmax / min_spacing), 1000))
    rho = np.full(probe.shape, 1.0 / max_spacing)
    for m, w in zip(modes, widths):
        rho += (pts_per_line / np.pi) * w / ((probe - m) ** 2 + w**2)
    rho = np.minimum(rho, 1.0 / min_spacing)
    cdf = np.concatenate(
        ([0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(probe))))
    n_pts = int(np.round(cdf[-1])) + 1
    grid = np.interp(np.linspace(0.0, cdf[-1], n_pts), cdf, probe)
    grid[0], grid[-1] = 0.0, fmax
    # Strictly ascending (the density cap makes ties all but impossible;
    # guard anyway).
    keep = np.concatenate(([True], np.diff(grid) > 1e-12))
    return grid[keep]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--modes", type=Path,
                     help=".npy/.txt mode frequencies (THz)")
    src.add_argument("--npz", type=Path,
                     help="previous run.npz (DOS peak-picking)")
    src.add_argument("--dyn", type=Path,
                     help="dynamical matrix file (THz^2)")
    p.add_argument("--fmax", type=float, required=True,
                   help="primary-grid top (THz), just above omega_max; the "
                        "2*omega_max convolution support lives on the "
                        "auxiliary grid (--aux-fmax at config time)")
    p.add_argument("--width-thz", type=float, default=0.1,
                   help="Lorentzian half-width per line (THz): the expected "
                        "ANHARMONIC linewidth Gamma_anh (golden rule, or "
                        "measured from a previous run). This is a grid-"
                        "design resolution target, NOT a broadening -- "
                        "runs stay at eta = 0 [0.1]")
    p.add_argument("--widths", type=Path, default=None,
                   help=".npy per-line widths (overrides --width-thz)")
    p.add_argument("--pts-per-line", type=int, default=12,
                   help="grid points allotted per line (~8 across the "
                        "FWHM at 12) [12]")
    p.add_argument("--max-spacing", type=float, default=None,
                   help="background spacing bound (THz) [fmax/100]. The "
                        "comb covers only the FLAT bands; dispersive "
                        "(acoustic) branches are continua that the "
                        "background floor must resolve -- keep this at or "
                        "below the uniform spacing you would otherwise "
                        "use, or the heat-carrying acoustic region is "
                        "under-sampled.")
    p.add_argument("--min-spacing", type=float, default=None,
                   help="lower spacing bound (THz) [width/8]")
    p.add_argument("--peak-prominence", type=float, default=0.01,
                   help="--npz peak-picking prominence, rel. to max [0.01]")
    p.add_argument("--out", type=Path, default=Path("phonon_energies.npy"))
    a = p.parse_args()

    if a.modes is not None:
        modes = _modes_from_file(a.modes)
    elif a.npz is not None:
        modes = _modes_from_npz(a.npz, a.peak_prominence)
    else:
        modes = _modes_from_dyn(a.dyn)

    width = (np.asarray(np.load(a.widths), dtype=float)
             if a.widths is not None else a.width_thz)
    max_spacing = a.max_spacing if a.max_spacing is not None else a.fmax / 100
    min_spacing = (a.min_spacing if a.min_spacing is not None
                   else float(np.min(width)) / 8.0)

    grid = build_grid(modes, a.fmax, width, a.pts_per_line,
                      max_spacing, min_spacing)
    np.save(a.out, grid)
    sp = np.diff(grid)
    n_uni = int(np.ceil(a.fmax / sp.min())) + 1
    print(f"wrote {a.out}: {grid.size} pts on [0, {a.fmax:g}] THz, "
          f"spacing [{sp.min():.4g}, {sp.max():.4g}] THz "
          f"({np.count_nonzero((modes > 0) & (modes <= a.fmax))} raw lines; "
          f"a uniform grid at the finest spacing would need {n_uni} pts, "
          f"{n_uni / grid.size:.1f}x more)")
    print("engine: pass --freq-grid file (write_config.py) or QX_FREQGRID="
          "file, plus --aux-dw ~ finest spacing and --aux-fmax >= "
          "2*omega_max.")


if __name__ == "__main__":
    main()
