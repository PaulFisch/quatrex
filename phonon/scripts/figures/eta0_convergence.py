"""Figures + verified numbers for the eta->0 SCBA convergence write-up.

Single source of truth for every number that enters the LaTeX (printed to
stdout) and for the four figures, regenerated ONLY from VALID run data:

  cnt33 (SOLVABLE eta=0):  phonon/studies/out/conv1e10/cnt33_smooth_L2.{log,npz}
  cnt33 cutoff sweep:      the converged G*dw(omega_reg) table (from the runs;
                           mirrored from phonon/studies/_taper_plot.py)
  d5a (HARD eta=0):        sinw_d5a_L2_rpm_eta0.log (RPM, |lambda|),
                           sinw_d5a_L2_nf181_jfnk_eta0.log (JFNK k=25),
                           sinw_d5a_L2_nf181_jfnk_k50_eta0.log (JFNK k=50)
  d5a Gamma_anh anchor:    phonon/scripts/verify/d5a_gamma_anh.npz (NM=32,BW=0.2)

Figures -> document/fig/transport_sweeps/{eta0_convergence_methods,
eta0_cnt33_cutoff,eta0_cnt33_transmission,d5a_gamma_anh}.{pdf,png}

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/eta0_convergence.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style          # styled figure()/save()

CONV = ROOT / "phonon/studies/out/conv1e10"
FIGDIR = ROOT / "document/fig/transport_sweeps"
HBAR = 1.054571817e-34
KB = 1.380649e-23


def _bose(w_thz, T):
    w = np.asarray(w_thz, float)
    x = np.where(w > 1e-6, (w * 1e12 * 2 * np.pi * HBAR) / (KB * T), 1.0)
    return np.where(w > 1e-6, 1.0 / np.expm1(x), 0.0)


def parse_trace(log: Path):
    """rel Sigma^R residual + lead balance + bubble balance per iteration."""
    res, lead, bub = [], [], []
    pr = re.compile(r"rel Sigma\^R residual ([0-9.eE+-]+); lead balance ([0-9.eE+-]+)")
    pb = re.compile(r"Bubble energy balance: .*resid=([0-9.eE+-]+)")
    for ln in log.read_text(errors="ignore").splitlines():
        m = pr.search(ln)
        if m:
            res.append(float(m.group(1))); lead.append(float(m.group(2)))
        m = pb.search(ln)
        if m:
            bub.append(float(m.group(1)))
    return np.array(res), np.array(lead), np.array(bub)


def parse_rpm(log: Path):
    """RPM |lambda(H)|max + n_unstable lines."""
    out = []
    pr = re.compile(r"RPM it=(\d+): k=\d+ \|lam\(H\)\|max=([0-9.eE+-]+) n_unstable=(\d+)")
    for ln in log.read_text(errors="ignore").splitlines():
        m = pr.search(ln)
        if m:
            out.append((int(m.group(1)), float(m.group(2)), int(m.group(3))))
    return out


def parse_jfnk(log: Path):
    """JFNK per-Newton-step ||R|| and inner_res/||R||."""
    rnorm, inner = [], []
    pr = re.compile(r"JFNK newton#(\d+): gmres_m=\d+ inner_res/\|\|R\|\|=([0-9.eE+-]+) "
                    r"\|\|R\|\|=([0-9.eE+-]+)")
    for ln in log.read_text(errors="ignore").splitlines():
        m = pr.search(ln)
        if m:
            inner.append(float(m.group(2))); rnorm.append(float(m.group(3)))
    return np.array(rnorm), np.array(inner)


# ---- verified cnt33 cutoff sweep (converged 1e-10 fixed points; from the runs)
# (omega_reg THz, G*dw, converged?) and the grid check at omega_reg=1.83.
CNT_SWEEP = [(0.90, 17.091, False), (1.20, 17.131, False), (1.50, 17.001, False),
             (1.83, 17.047, True), (2.44, 16.807, True), (3.06, 16.429, True)]
CNT_GRID = [(181, 17.047), (361, 16.874)]


def fig_methods():
    """(a) cnt33 eta=0 converges; (b) d5a eta=0 root-finders do not."""
    res_c, lead_c, bub_c = parse_trace(CONV / "cnt33_smooth_L2.log")
    rpm_res, _, _ = parse_trace(CONV / "sinw_d5a_L2_rpm_eta0.log")
    j25_res, _, _ = parse_trace(CONV / "sinw_d5a_L2_nf181_jfnk_eta0.log")
    j50_res, _, _ = parse_trace(CONV / "sinw_d5a_L2_nf181_jfnk_k50_eta0.log")
    rpm_lam = parse_rpm(CONV / "sinw_d5a_L2_rpm_eta0.log")

    fig, axes = style.figure(ncols=2, width=4.6, height=3.5)
    ax = axes[0]
    it = np.arange(1, res_c.size + 1)
    ax.semilogy(it, res_c, "-", color="C0", lw=1.5, label=r"rel $\Sigma^R$ residual")
    ax.semilogy(it, np.maximum(lead_c, 1e-17), "-", color="C3", lw=1.3,
                label=r"lead balance $|J_L-J_R|/|J|$")
    if bub_c.size:
        ax.semilogy(np.arange(1, bub_c.size + 1), np.maximum(bub_c, 1e-18), "-",
                    color="C2", lw=1.0, label="bubble balance")
    ax.axhline(1e-10, color="k", ls="--", lw=0.7)
    ax.set_xlabel("SCBA iteration"); ax.set_ylabel("convergence measure")
    ax.set_title(r"cnt33 $\eta{=}0$: genuine fixed point")
    ax.legend(fontsize=7, loc="upper right"); ax.set_ylim(1e-17, 5)

    ax = axes[1]
    ax.semilogy(np.arange(1, rpm_res.size + 1), rpm_res, "-", color="C3", lw=1.3,
                label="RPM (diverges)")
    ax.semilogy(np.arange(1, j25_res.size + 1), j25_res, "-", color="C0", lw=1.3,
                label="JFNK $k{=}25$")
    ax.semilogy(np.arange(1, j50_res.size + 1), j50_res, "-", color="C1", lw=1.3,
                label="JFNK $k{=}50$")
    ax.axhline(1e-10, color="k", ls="--", lw=0.7)
    if rpm_lam:
        lam0 = rpm_lam[0][1]
        ax.annotate(rf"RPM $|\lambda|{{\approx}}{lam0:.0f}$ (nf181)$\to$199 (nf361)",
                    (0.5, 0.06), xycoords="axes fraction", fontsize=7, ha="center")
    ax.set_xlabel("SCBA iteration"); ax.set_ylabel(r"rel $\Sigma^R$ residual")
    ax.set_title(r"d5a $\eta{=}0$: no method lands it")
    ax.legend(fontsize=7, loc="upper right"); ax.set_ylim(5e-2, 5)
    style.save(fig, "eta0_convergence_methods", directory=FIGDIR)

    print("\n[cnt33 convergence]  final resid={:.3e}  final lead={:.3e}  "
          "final bubble={:.3e}  iters={}".format(
              res_c[-1], lead_c[-1], bub_c[-1] if bub_c.size else float("nan"),
              res_c.size))
    print("[d5a RPM]   |lambda|max,n_unstable per print:",
          [(it_, round(l_, 1), n_) for it_, l_, n_ in rpm_lam])
    for tag, r in [("JFNK k25", j25_res), ("JFNK k50", j50_res)]:
        print(f"[d5a {tag}]  min resid={r.min():.3e}  final resid={r[-1]:.3e}  "
              f"iters={r.size}")
    rn25, in25 = parse_jfnk(CONV / "sinw_d5a_L2_nf181_jfnk_eta0.log")
    rn50, in50 = parse_jfnk(CONV / "sinw_d5a_L2_nf181_jfnk_k50_eta0.log")
    print(f"[d5a JFNK k25] ||R|| floor={rn25.min():.2e}  inner_res/||R|| max={in25.max():.2f}")
    print(f"[d5a JFNK k50] ||R|| floor={rn50.min():.2e}  inner_res/||R|| max={in50.max():.2f}")


def fig_cutoff():
    """cnt33 G*dw vs IR-taper omega_reg (cutoff insensitivity) + grid check."""
    conv = [(w, g) for w, g, c in CNT_SWEEP if c]
    best = [(w, g) for w, g, c in CNT_SWEEP if not c]
    wc = np.array([w for w, _ in conv]); gc = np.array([g for _, g in conv])
    A = np.vstack([np.ones_like(wc), wc ** 2]).T
    (G0, b), *_ = np.linalg.lstsq(A, gc, rcond=None)

    fig, axes = style.figure(ncols=2, width=4.4, height=3.4)
    ax = axes[0]
    ax.plot(*zip(*conv), "o", ms=7, color="C0", label=r"converged ($10^{-10}$ f.p.)")
    if best:
        ax.plot(*zip(*best), "o", ms=7, mfc="none", color="C0",
                label="best-iterate")
    x = np.linspace(0, 3.3, 60)
    ax.plot(x, G0 - b * x ** 2, "k--", lw=1.0,
            label=rf"$G_0-b\,\omega_{{\rm reg}}^2$")
    ax.plot(0, G0, "k*", ms=13)
    ax.set_xlabel(r"IR regularisation $\omega_{\rm reg}=C\,d\omega$ (THz)")
    ax.set_ylabel(r"$G\cdot d\omega$ (d$\omega$-weighted heat)")
    ax.set_title(r"cnt33 $\eta{=}0$ cutoff sensitivity")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    ns = [n for n, _ in CNT_GRID]; gs = [g for _, g in CNT_GRID]
    ax.plot(ns, gs, "s-", color="C3", ms=8)
    for n, g in CNT_GRID:
        ax.annotate(f"{g:.2f}", (n, g), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel("nfreq"); ax.set_ylabel(r"$G\cdot d\omega$")
    ax.set_title(r"grid convergence ($\omega_{\rm reg}{=}1.83$)")
    ax.set_ylim(16.2, 17.4)
    style.save(fig, "eta0_cnt33_cutoff", directory=FIGDIR)

    spread = (gc.max() - gc.min()) / gc.mean() * 100
    print(f"\n[cnt33 cutoff] converged G*dw = {gc.tolist()}  "
          f"spread={spread:.1f}%  extrapolated G0(wreg->0)={G0:.2f}")
    print(f"[cnt33 grid]   nf181={CNT_GRID[0][1]}  nf361={CNT_GRID[1][1]}  "
          f"({abs(CNT_GRID[0][1]-CNT_GRID[1][1])/CNT_GRID[0][1]*100:.1f}%)")


def fig_transmission():
    """Physics of the converged cnt33 eta=0 run: per-omega heat-current spectrum,
    and the effective transmission T(w)=I(w)/Delta n(w) over the propagating band
    (the sub-1-THz bins are the IR-regularised acoustic region -- excluded from
    the transmission panel, where dividing by the vanishing Delta n is ill-posed)."""
    d = np.load(CONV / "cnt33_smooth_L2.npz", allow_pickle=True)
    w = d["energies"]; cs = d["current_spectrum"]   # (nw,3) per interface
    tL, tR = float(d["t_left"]), float(d["t_right"])
    dn = _bose(w, tL) - _bose(w, tR)
    # net forward heat-current spectrum (sign so a forward flow is positive)
    Inet = np.sign(np.nanmean(cs[w > 5, 0])) * cs[:, 0]
    band = w >= 2.0                                  # exclude IR-regularised bins
    with np.errstate(divide="ignore", invalid="ignore"):
        Teff = np.where(band & (dn > 1e-9), Inet / dn, np.nan)
    Tn = Teff / np.nanmax(Teff[np.isfinite(Teff)])

    fig, axes = style.figure(ncols=2, width=4.4, height=3.4)
    ax = axes[0]
    ax.plot(w, Inet, "-", color="C0", lw=1.2, label="hot-lead interface")
    ax.plot(w, np.sign(np.nanmean(cs[w > 5, 0])) * cs[:, -1], "-", color="C3",
            lw=1.0, alpha=0.7, label="cold-lead interface")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-0.25, 0.45)
    ax.set_xlabel("frequency (THz)")
    ax.set_ylabel(r"heat-current spectrum $I(\omega)$ (arb.)")
    ax.set_title(r"cnt33 $\eta{=}0$ converged: $I(\omega)$")
    ax.legend(fontsize=7, loc="upper right")
    ax = axes[1]
    ax.plot(w[band], Tn[band], "-", color="C2", lw=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("frequency (THz)")
    ax.set_ylabel(r"$T(\omega)=I(\omega)/\Delta n(\omega)$ (norm.)")
    ax.set_title(r"effective transmission ($\omega\!\geq\!2$ THz)")
    style.save(fig, "eta0_cnt33_transmission", directory=FIGDIR)
    dw = float(w[1] - w[0])
    print(f"\n[cnt33 physics] lead temps {tL:.1f}/{tR:.1f} K; dw={dw:.4f} THz; "
          f"raw lead_current(unweighted)={float(d['lead_current']):.3f} -> "
          f"G*dw={float(d['lead_current'])*dw:.2f}; converged={bool(d['converged'])}")


def fig_gamma():
    """d5a golden-rule Gamma_anh(omega) by H-character vs grid d-omega."""
    d = np.load(ROOT / "phonon/scripts/verify/d5a_gamma_anh.npz")
    FR, gam, hf = d["FR"].ravel(), d["gam"].ravel(), d["hfrac"].ravel()
    ok = np.isfinite(gam) & (FR > 0.3)
    FR, gam, hf = FR[ok], gam[ok], hf[ok]
    fmax = 66.0
    dw181, dw361 = fmax / 180, fmax / 360

    fig, axes = style.figure(ncols=1, width=5.2, height=3.6)
    ax = axes[0] if hasattr(axes, "__len__") else axes
    sc = ax.scatter(FR, gam, c=hf, s=14, cmap="viridis", vmin=0, vmax=1)
    ax.axhline(dw181, color="C3", ls="--", lw=0.9)
    ax.axhline(dw361, color="C1", ls="--", lw=0.9)
    ax.annotate(r"$d\omega$ (nf181)", (1.0, dw181), fontsize=7, color="C3",
                va="bottom")
    ax.annotate(r"$d\omega$ (nf361)", (1.0, dw361), fontsize=7, color="C1",
                va="bottom")
    ax.set_yscale("log"); ax.set_xlabel("mode frequency (THz)")
    ax.set_ylabel(r"$\Gamma_{\rm anh}$ (THz)")
    ax.set_title(r"d5a golden-rule linewidth vs grid $d\omega$")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("H-character", fontsize=8)
    style.save(fig, "d5a_gamma_anh", directory=FIGDIR)

    def band(lo, hi):
        m = (FR >= lo) & (FR < hi)
        return (int(m.sum()), float(np.median(gam[m])), float(gam[m].min()),
                float(gam[m].max()), float(hf[m].mean())) if m.any() else None
    print("\n[d5a Gamma_anh]  dw181={:.3f} dw361={:.3f}".format(dw181, dw361))
    for nm, lo, hi in [("Si ac+opt", 0.3, 15), ("Si-H bending", 15, 30),
                       ("Si-H stretch", 55, 66)]:
        b = band(lo, hi)
        if b:
            print(f"  {nm:14s} N={b[0]:4d}  Gamma med={b[1]:.3f} "
                  f"[{b[2]:.3f},{b[3]:.3f}] THz  Hchar={b[4]:.2f}")


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70 + "\nVERIFIED NUMBERS FOR THE eta->0 WRITE-UP\n" + "=" * 70)
    fig_methods()
    fig_cutoff()
    fig_transmission()
    fig_gamma()
    print("\nfigures ->", FIGDIR)
