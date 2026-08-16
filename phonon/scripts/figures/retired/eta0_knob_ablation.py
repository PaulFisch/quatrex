"""The eta=0 stabilisation-knob ablation + physics-sensitivity figures.

Companion to eta0_convergence.py (which shows THAT the eta=0 fixed point
exists); this pair shows WHAT each stabilisation knob does -- convergence
with/without every knob, and how much the converged PHYSICS depends on it.
Built ONLY from saved run traces (no new solver runs):

  (1) eta0_knob_ablation.{pdf,png} -- rel Sigma^R residual vs SCBA iteration,
      one panel per knob, each with the knob OFF and ON on the same axes:
      (a) causal retarded Sigma^R (retarded_method half -> fft), cnt33 L3:
          the half rule limit-cycles at O(0.1) under EVERY mixer; the causal
          KK real part restores contraction and the production run converges.
          [convergence/L3e0_A_half55_mix_*.log vs L3e0_Bc_fft55_mix_lin0.1.log
           + prod/cnt33_eta0/L3_anh.log]
      (b) infrared occupancy taper + smooth spectral window, cnt33 L2 (fft):
          bare fft floors at ~2e-4; ir_taper_cells=3 stalls, =4 reaches 1e-2,
          =6 + sse_smooth_window converges to a genuine 4e-11 fixed point.
          [prod/cnt33_eta0/L2_anh.log, conv1e10/L2_taper3.log, L2_taper4s.log,
           conv1e10/cnt33_smooth_L2.log]
      (c) mixer choice on the marginal eta=0 mode, cnt33 L3 (fft): Anderson
          and Broyden do NOT beat plain linear mixing (the marginal mode
          defeats secant models); linear + patience converges.
          [convergence/L3e0_B_fft55_mix_{lin,and,broy}0.1.log + Bc + prod]

  (2) eta0_knob_sensitivity.{pdf,png} -- converged G*dw of cnt33 L2 eta=0
      for each knob setting (npz last_heat * dw), showing which knobs move
      the PHYSICS and by how much: retarded half->fft -6.1%; the IR taper
      +25% (it REMOVES the unphysical IR-Bose contamination -- the tapered
      number is the cutoff-insensitive one, extrapolating to G0=17.4 as
      omega_reg->0); cutoff within the tapered family <=3.7%; grid 181->361
      1%; mixers 0% (identical fixed point).

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/eta0_knob_ablation.py
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
from phonon.studies import style  # noqa: E402

CONV10 = ROOT / "phonon/studies/out/conv1e10"
CONVAB = ROOT / "phonon/studies/out/convergence"
PROD = ROOT / "phonon/scripts/out/prod"
FIGDIR = ROOT / "document/fig/transport_sweeps"

DW = 55.0 / 180  # nf181, fmax 55 THz (all cnt33 runs used here)

# The verified cutoff-sweep table (single source: eta0_convergence.py).
CNT_SWEEP = [(0.90, 17.091, False), (1.20, 17.131, False), (1.50, 17.001, False),
             (1.83, 17.047, True), (2.44, 16.807, True), (3.06, 16.429, True)]
CNT_GRID = [(181, 17.047), (361, 16.874)]

RESID_RE = re.compile(r"rel Sigma\^R residual ([0-9.eE+-]+); lead balance "
                      r"([0-9.eE+-]+)")


def trace(path: Path) -> np.ndarray:
    res = [float(m.group(1)) for m in RESID_RE.finditer(path.read_text(
        errors="ignore"))]
    return np.asarray(res)


def g_dw(npz_path: Path) -> tuple[float, bool]:
    z = np.load(npz_path, allow_pickle=True)
    lh = np.asarray(z["last_heat"]).reshape(-1)
    return abs(float(lh[0])) * DW, bool(z.get("converged", False))


def fig_ablation():
    fig, axes = style.figure(ncols=3, width=6.6, height=3.2)

    # (a) retarded: half vs fft, cnt33 L3 -------------------------------
    ax = axes[0]
    for tag, lab, col, lw in [
            ("L3e0_A_half55_mix_lin0.1", "half, linear", "C3", 1.3),
            ("L3e0_A_half55_mix_and0.1", "half, Anderson", "C1", 0.9),
            ("L3e0_A_half55_mix_broy0.1", "half, Broyden", "0.6", 0.9)]:
        r = trace(CONVAB / f"{tag}.log")
        ax.semilogy(np.arange(1, r.size + 1), r, "-", color=col, lw=lw,
                    label=lab)
    r = trace(CONVAB / "L3e0_Bc_fft55_mix_lin0.1.log")
    ax.semilogy(np.arange(1, r.size + 1), r, "-", color="C0", lw=1.1,
                label="fft, linear (300 it)")
    r = trace(PROD / "cnt33_eta0/L3_anh.log")
    ax.semilogy(np.arange(1, r.size + 1), r, "-", color="C2", lw=1.5,
                label="fft, production (conv.)")
    ax.axhline(1e-3, color="k", ls="--", lw=0.6)
    ax.set_title(r"(a) causal Re$\,\Sigma^R$ (L3)", fontsize=9)
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel(r"rel $\Sigma^R$ residual")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(1e-4, 5)

    # (b) IR taper + smooth window, cnt33 L2 (all fft) -------------------
    ax = axes[1]
    for path, lab, col, lw in [
            (PROD / "cnt33_eta0/L2_anh.log", "no taper (floor $2{\\times}10^{-4}$)",
             "C3", 1.2),
            (CONV10 / "L2_taper3.log", r"taper $3\,d\omega$ (stalls)", "C1", 0.9),
            (CONV10 / "L2_taper4s.log", r"taper $4\,d\omega$", "0.6", 0.9),
            (CONV10 / "cnt33_smooth_L2.log",
             r"taper $6\,d\omega$ + smooth window", "C0", 1.6)]:
        r = trace(path)
        ax.semilogy(np.arange(1, r.size + 1), r, "-", color=col, lw=lw,
                    label=lab)
    ax.axhline(1e-10, color="k", ls="--", lw=0.6)
    ax.set_title(r"(b) IR taper + window (L2)", fontsize=9)
    ax.set_xlabel("SCBA iteration")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(1e-11, 5)

    # (c) mixers on the marginal mode, cnt33 L3 (fft) --------------------
    ax = axes[2]
    for tag, lab, col, lw in [
            ("L3e0_B_fft55_mix_lin0.1", "linear 0.1", "C0", 1.4),
            ("L3e0_B_fft55_mix_and0.1", "Anderson", "C1", 1.0),
            ("L3e0_B_fft55_mix_broy0.1", "Broyden", "C3", 1.0)]:
        r = trace(CONVAB / f"{tag}.log")
        ax.semilogy(np.arange(1, r.size + 1), r, "-", color=col, lw=lw,
                    label=lab)
    ax.annotate("no mixer beats plain linear;\nlinear + patience converges\n"
                "(panel a, green)", (0.97, 0.05), xycoords="axes fraction",
                fontsize=6.5, ha="right", va="bottom")
    ax.set_title("(c) mixer choice (L3, fft)", fontsize=9)
    ax.set_xlabel("SCBA iteration")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(3e-2, 5)

    style.save(fig, "eta0_knob_ablation", directory=FIGDIR)


def fig_sensitivity():
    g_half, c_half = g_dw(CONVAB / "L2e0_half_mix_lin0.2.npz")
    g_bare, c_bare = g_dw(PROD / "cnt33_eta0/L2_anh.npz")
    g_smooth, c_smooth = g_dw(CONV10 / "cnt33_smooth_L2.npz")

    conv = [(w, g) for w, g, c in CNT_SWEEP if c]
    wc = np.array([w for w, _ in conv]); gc = np.array([g for _, g in conv])
    A = np.vstack([np.ones_like(wc), wc ** 2]).T
    (G0, _b), *_ = np.linalg.lstsq(A, gc, rcond=None)

    rows = [
        # label, G*dw, filled(=converged 1e-10 family), note
        (r"half rule (drops Re$\,\Sigma^R$)", g_half, False,
         "non-conserving interior"),
        ("bare fft (no IR taper)", g_bare, False, r"floor $2{\times}10^{-4}$"),
        (r"fft + taper $6d\omega$ + window", g_smooth, True,
         r"$10^{-10}$ fixed point"),
        (r"same, $\omega_{\rm reg}=2.44$ THz", 16.807, True, "cutoff family"),
        (r"same, $\omega_{\rm reg}=3.06$ THz", 16.429, True, "cutoff family"),
        ("same, nfreq 361", 16.874, True, "grid check"),
    ]
    fig, axes = style.figure(ncols=1, width=4.6, height=3.2)
    ax = axes if not isinstance(axes, (list, np.ndarray)) else axes[0]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, g, filled, note) in zip(y, rows):
        ax.plot(g, yi, "o", ms=8, color="C0", mfc="C0" if filled else "none")
        right = g < 16.0  # keep labels off the right axis edge
        ax.annotate(f"{g:.2f}  ({(g - G0) / G0 * 100:+.1f}%)", (g, yi),
                    textcoords="offset points",
                    xytext=(8, -3) if right else (-8, -3),
                    ha="left" if right else "right", fontsize=7)
    ax.axvline(G0, color="k", ls="--", lw=0.8)
    ax.annotate(rf"$G_0={G0:.1f}$ ($\omega_{{\rm reg}}\to0$)", (G0, y[0] + 0.55),
                fontsize=7.5, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel(r"converged $G\cdot d\omega$ (cnt33 L2, $\eta=10^{-12}$)")
    ax.set_xlim(13.0, 18.4)
    style.save(fig, "eta0_knob_sensitivity", directory=FIGDIR)

    print(f"[knob sensitivity] G0(wreg->0)={G0:.2f}")
    print(f"  half:        {g_half:.3f} ({(g_half - G0) / G0 * 100:+.1f}%) "
          f"converged={c_half}")
    print(f"  bare fft:    {g_bare:.3f} ({(g_bare - G0) / G0 * 100:+.1f}%) "
          f"converged={c_bare}")
    print(f"  taper+smooth:{g_smooth:.3f} ({(g_smooth - G0) / G0 * 100:+.1f}%) "
          f"converged={c_smooth}")


if __name__ == "__main__":
    fig_ablation()
    fig_sensitivity()
