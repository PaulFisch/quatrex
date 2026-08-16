"""d5a eta=0 BARE-SSE grid ladder: resolution, spectral variance, alignment.

Four panels from phonon/studies/out/d5a_gridladder/nf{N}/ (driver:
phonon/studies/_run_d5a_gridladder.py; all rungs bare -- no IR taper):

  (a) rel Sigma^R residual (+ lead balance, thin) vs SCBA iteration for the
      RESOLUTION rungs nf {181, 361, 721, 1441}: does the limit-cycle floor
      drop once d_omega resolves the flat-band linewidths?
  (b) WHERE the iteration variance lives: per-omega relative std of
      max|Sigma^<(w)| over the last iterations (QX_DIAG_SPECTRAL arrays),
      one curve per rung, with the FLAT bands of the d5a dispersion
      (bandwidth < d_omega) marked -- the grid-hits hypothesis test.
  (c) summary vs d_omega: residual floor, best lead balance and best-iterate
      G*dw, with the transport-relevant Gamma_anh distribution
      (phonon/scripts/verify/d5a_gamma_anh.npz) as the predicted
      d_omega < Gamma transition band.
  (d) ALIGNMENT scan nf {181, 185, 189, 193} (~constant resolution, bins
      shifted relative to the bands): residual floor vs the flat-band/grid
      alignment metric min_b dist(omega_b, grid)/d_omega.

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/d5a_grid_ladder.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style  # noqa: E402

DATA = ROOT / "phonon/studies/out/d5a_gridladder"
GAMMA_NPZ = ROOT / "phonon/scripts/verify/d5a_gamma_anh.npz"
FIGDIR = ROOT / "document/fig/transport_sweeps"
FMAX = 66.0
RES_RUNGS = [181, 361, 721, 1441, 2881]
ALIGN_RUNGS = [181, 185, 189, 193]

RESID_RE = re.compile(r"rel Sigma\^R residual ([0-9.eE+-]+); lead balance "
                      r"([0-9.eE+-]+)")


def trace(nf: int):
    log = DATA / f"nf{nf}.log"
    if not log.exists():
        return None, None
    m = RESID_RE.findall(log.read_text(errors="ignore"))
    if not m:
        return None, None
    r = np.array([float(a) for a, _ in m])
    lb = np.array([float(b) for _, b in m])
    return r, lb


def bands(nf: int, nk: int = 600):
    """d5a dispersion from the rung's dynamical matrix; returns (bands
    (nk, nmodes) THz, flat-band centers with bandwidth < d_omega)."""
    M = sio.loadmat(str(DATA / f"nf{nf}/dynamical_matrix.mat"))
    H0 = np.asarray(M["[0, 0, 0]"])
    H1 = np.asarray(M["[0, 0, 1]"])
    Hm = np.asarray(M["[0, 0, -1]"])
    ks = np.linspace(1e-4, np.pi, nk)
    bb = np.empty((nk, H0.shape[0]))
    for i, k in enumerate(ks):
        Dk = H0 + H1 * np.exp(1j * k) + Hm * np.exp(-1j * k)
        Dk = 0.5 * (Dk + Dk.conj().T)
        w2 = np.linalg.eigvalsh(Dk)
        bb[i] = np.sign(w2) * np.sqrt(np.abs(w2))
    dw = FMAX / (nf - 1)
    width = bb.max(0) - bb.min(0)
    centers = 0.5 * (bb.max(0) + bb.min(0))
    flat = centers[(width < dw) & (centers > 0.1)]
    return bb, flat


def alignment_metric(flat: np.ndarray, dw: float) -> float:
    """min over flat bands of the distance to the nearest grid point, in
    units of d_omega (0 = a band sits exactly ON a bin, 0.5 = maximally
    between bins)."""
    if flat.size == 0:
        return np.nan
    frac = np.abs(flat / dw - np.round(flat / dw))
    return float(frac.min())


def floor_of(r: np.ndarray) -> float:
    tail = r[-max(10, r.size // 5):]
    return float(np.median(tail))


def main() -> None:
    have = {nf: trace(nf) for nf in sorted(set(RES_RUNGS + ALIGN_RUNGS))}
    have = {nf: v for nf, v in have.items() if v[0] is not None}
    if not have:
        print("[skip] no ladder data yet")
        return
    print(f"rungs with data: {sorted(have)}")

    fig, axes = style.figure(ncols=2, nrows=2, width=8.4, height=6.4)
    axes = np.asarray(axes).ravel()

    # (a) residual vs iteration, resolution rungs -----------------------
    ax = axes[0]
    colors = {181: "0.6", 361: "C1", 721: "C3", 1441: "C0", 2881: "C2"}
    for nf in [n for n in RES_RUNGS if n in have]:
        r, lb = have[nf]
        c = colors.get(nf, "k")
        ax.semilogy(np.arange(1, r.size + 1), r, "-", color=c, lw=1.3,
                    label=rf"nf={nf} ($d\omega$={FMAX / (nf - 1):.3f})")
        ax.semilogy(np.arange(1, lb.size + 1), np.maximum(lb, 1e-17), ":",
                    color=c, lw=0.7)
    ax.set_title("(a) residual (lead balance dotted) vs iteration",
                 fontsize=9)
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel(r"rel $\Sigma^R$ residual")
    ax.legend(fontsize=6.5, loc="upper right")

    # (b) per-omega iteration variance + flat bands ----------------------
    ax = axes[1]
    off = 1.0
    for nf in [n for n in RES_RUNGS if n in have]:
        npz = DATA / f"nf{nf}/run.npz"
        if not npz.exists():
            continue
        z = np.load(npz, allow_pickle=True)
        if "iter_sigL_w" not in z.files:
            continue
        s = np.asarray(z["iter_sigL_w"], float)      # (n_iter, ne)
        tail = s[-max(10, s.shape[0] // 5):]
        mu = tail.mean(0)
        rel = tail.std(0) / np.maximum(np.abs(mu), 1e-300)
        w = np.linspace(0, FMAX, s.shape[1])
        ax.semilogy(w, np.maximum(rel, 1e-8) * off, "-",
                    color=colors.get(nf, "k"), lw=0.9,
                    label=f"nf={nf}" + (f" (x{off:g})" if off != 1 else ""))
        off *= 100.0
    nf0 = max(n for n in have if n in RES_RUNGS)
    _bb, flat = bands(nf0)
    for wb in flat:
        ax.axvline(wb, color="0.85", lw=0.6, zorder=0)
    ax.set_title(r"(b) per-$\omega$ iteration variance of "
                 r"$|\Sigma^<(\omega)|$; grey = flat bands", fontsize=9)
    ax.set_xlabel(r"$\omega$ (THz)")
    ax.set_ylabel("rel. std over last iterations (staggered)")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_xlim(0, 25)

    # (c) summary vs d_omega + Gamma_anh threshold -----------------------
    ax = axes[2]
    dws, floors, lbs = [], [], []
    for nf in [n for n in RES_RUNGS if n in have]:
        r, lb = have[nf]
        dws.append(FMAX / (nf - 1))
        floors.append(floor_of(r))
        lbs.append(float(np.min(lb)))
    ax.loglog(dws, floors, "o-", color="C0", label="residual floor")
    ax.loglog(dws, lbs, "s--", color="C3", label="best lead balance")
    if GAMMA_NPZ.exists():
        g = np.load(GAMMA_NPZ, allow_pickle=True)
        gam = np.asarray(g["gam"], float)
        hf = np.asarray(g["hfrac"], float)
        fr = np.asarray(g["FR"], float)
        sel = (hf < 0.5) & (fr > 0.1) & (fr < 20.0) & (gam > 0)
        if sel.any():
            lo, hi = np.percentile(gam[sel], [25, 75])
            ax.axvspan(lo, hi, color="C2", alpha=0.18,
                       label=r"$\Gamma_{\rm anh}$ IQR (Si modes < 20 THz)")
    ax.set_title(r"(c) floor vs $d\omega$ ($\Gamma$-resolution test)",
                 fontsize=9)
    ax.set_xlabel(r"$d\omega$ (THz)")
    ax.set_ylabel("convergence measure")
    ax.legend(fontsize=6.5, loc="best")
    ax.invert_xaxis()

    # (d) alignment scan -------------------------------------------------
    ax = axes[3]
    pts = []
    for nf in [n for n in ALIGN_RUNGS if n in have]:
        r, _ = have[nf]
        dw = FMAX / (nf - 1)
        _, flat = bands(nf)
        pts.append((alignment_metric(flat, dw), floor_of(r), nf))
    for x, y, nf in pts:
        ax.semilogy(x, y, "o", ms=9, color="C0")
        ax.annotate(f"nf={nf}", (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5)
    ax.set_title("(d) alignment scan: floor vs flat-band/grid distance",
                 fontsize=9)
    ax.set_xlabel(r"$\min_b\,\mathrm{dist}(\omega_b,\,\mathrm{grid})/d\omega$")
    ax.set_ylabel("residual floor")
    ax.set_xlim(-0.02, 0.52)

    style.save(fig, "d5a_grid_ladder", directory=FIGDIR)

    print("\n[summary]")
    for nf in sorted(have):
        r, lb = have[nf]
        dw = FMAX / (nf - 1)
        _, flat = bands(nf)
        am = alignment_metric(flat, dw)
        print(f"  nf={nf:5d} dw={dw:.4f}  floor={floor_of(r):.3e}  "
              f"best_lead={np.min(lb):.2e}  n_flat={flat.size}  align={am:.3f}")


if __name__ == "__main__":
    main()
