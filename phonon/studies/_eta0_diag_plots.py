"""Diagnostics for the sinw-d5a eta=0 SCBA fluctuation.

d5a (the soft-twist SiNW) does NOT reach a genuine eta=0 fixed point (RPM
diverges |lam| 28->200, JFNK stalls). These figures localise WHY / WHERE in
frequency the iteration fluctuates:

  bands        phonon dispersion Gamma->Z up to the band-top (which modes exist
               and where the heat-carrying / Si-H bands sit).
  convergence  the SCBA residual / lead-balance / bubble-balance history.
  spectral     per-frequency, per-iteration overlays of
                 - the Green's function fed INTO the bubble (-Im Tr G^R, a DOS),
                 - the RAW vs WINDOWED G^< actually convolved (what the eta=0
                   smooth window removes),
                 - the SSE magnitude |Sigma(omega)|,
                 - the G^R DOS resulting AFTER the Dyson re-solve (= next iter
                   input),
               for a spread of iterations -- this is what reveals the limit
               cycle and the omega-bins where it lives.

The `spectral` / `convergence` panels read the diagnostic npz + log produced by
the instrumented run (engine/run.py: iter_gin_dos / iter_graw_w / iter_gwin_w /
iter_sigR_w / iter_sigL_w + sse_window). `bands` is pure post-processing of the
canonical FC2 and needs no run.

Run:  OMP_NUM_THREADS=1 python -m phonon.studies._eta0_diag_plots [bands|diag|all]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.studies import style  # noqa: E402

FIGDIR = ROOT / "document/fig/transport_sweeps"
CONV = ROOT / "phonon/studies/out/conv1e10"
DIAG_NPZ = CONV / "sinw_d5a_L2_eta0_diag.npz"
DIAG_LOG = CONV / "sinw_d5a_L2_eta0_diag.log"

# Canonical d5a force constants (= production transport FCs).
D5A_YAML = ROOT / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml"
D5A_FC3 = ROOT / "phonon/configs/sinw/fc3_hiphive_sinw100_d5a_sc4_vasp/fc3.hdf5"
FMAX_WINDOW = 18.0   # the transport energy window used for the eta=0 runs


# ----------------------------------------------------------------------------
# (1) Band structure  Gamma -> Z
# ----------------------------------------------------------------------------
def _d5a_bands(nq: int = 101):
    """Signed harmonic frequencies (nq, n_modes) [THz] along Gamma->Z and the
    fractional q coordinate, from the canonical d5a FC2."""
    from phonon.finite_analysis.loader import load_system
    from phonon.postproc.spectral import (
        dynamical_matrix_qpath, frequencies_from_dynamical)

    bundle = load_system(D5A_YAML, name="sinw_d5a",
                         fc3_path_override=D5A_FC3, validate=False)
    qpath = np.zeros((nq, 3))
    qpath[:, 2] = np.linspace(0.0, 0.5, nq)   # transport axis = z
    dyn = dynamical_matrix_qpath(bundle.phonon, qpath)     # (nq, N, N) THz^2
    bands = frequencies_from_dynamical(dyn)                # (nq, N) signed THz
    return qpath[:, 2], bands


def fig_bands():
    qz, bands = _d5a_bands()
    qx = qz / 0.5  # 0 (Gamma) .. 1 (Z)
    band_top = float(np.nanmax(bands))
    soft = float(np.nanmin(bands))
    print(f"[bands] {bands.shape[1]} modes; band-top={band_top:.2f} THz; "
          f"lowest mode={soft:.4f} THz")

    fig, ax = style.figure(width=4.6, height=3.6)
    for m in range(bands.shape[1]):
        ax.plot(qx, bands[:, m], color="#0173b2", lw=0.8)
    ax.axhline(FMAX_WINDOW, color="#d55e00", ls="--", lw=1.1,
               label=f"transport window {FMAX_WINDOW:.0f} THz")
    ax.axhline(band_top, color="#949494", ls=":", lw=1.0,
               label=f"band top {band_top:.1f} THz")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels([r"$\Gamma$", "Z"])
    ax.set_xlim(0, 1)
    ax.set_ylabel("frequency (THz)")
    ax.set_xlabel("wavevector")
    ax.legend(fontsize=7, loc="center right")

    # inset: the near-Gamma soft/acoustic region (the d5a twist modes)
    axin = ax.inset_axes([0.10, 0.55, 0.36, 0.40])
    for m in range(bands.shape[1]):
        axin.plot(qx, bands[:, m], color="#0173b2", lw=0.8)
    axin.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    axin.set_xlim(0, 1); axin.set_ylim(-0.5, 4.0)
    axin.set_title("near $\\Gamma$", fontsize=7)
    axin.tick_params(labelsize=6)
    axin.set_xticks([0, 1]); axin.set_xticklabels([r"$\Gamma$", "Z"], fontsize=6)

    style.save(fig, "sinw_d5a_bands", directory=FIGDIR)
    return band_top


# Run selection: figures are named sinw_d5a_eta0[<suffix>]_*  so the taper
# baseline (suffix="") and the IR-occupation-subtraction run (suffix="_irsub")
# coexist. `TAPER_*` point at the committed taper baseline for overlays.
TAPER_NPZ = CONV / "sinw_d5a_L2_eta0_diag.npz"


def _name(base, suffix):
    return f"sinw_d5a_eta0{suffix}_{base}"


# ----------------------------------------------------------------------------
# (2) Convergence history
# ----------------------------------------------------------------------------
def fig_convergence(log: Path = DIAG_LOG, suffix=""):
    from phonon.studies.pipeline import parse_scba_trace
    tr = parse_scba_trace(Path(log))

    fig, ax = style.figure(width=4.6, height=3.4)
    res = np.asarray(tr.get("residual", []), float)
    lead = np.asarray(tr.get("lead_balance", []), float)
    bub = np.asarray(tr.get("bubble_balance", []), float)
    if res.size:
        ax.semilogy(np.arange(res.size), res, "-", color="#0173b2",
                    label=r"rel $\Sigma^R$ residual")
    if lead.size:
        ax.semilogy(np.arange(lead.size), np.abs(lead), "-", color="#de8f05",
                    label="lead balance")
    if bub.size:
        ax.semilogy(np.arange(bub.size), np.abs(bub), "-", color="#029e73",
                    label="bubble balance (conservation)")
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel("residual")
    ax.legend(fontsize=7)
    style.save(fig, _name("convergence", suffix), directory=FIGDIR)


# ----------------------------------------------------------------------------
# (2b) Heat-current convergence + lead deviation
# ----------------------------------------------------------------------------
def fig_heat_current(npz: Path = DIAG_NPZ, suffix=""):
    """Per-interface heat current vs iteration (left) and the lead-to-lead
    deviation |J_0 - J_last| (the heat-flow non-conservation) vs iteration
    (right) -- the physical convergence criterion."""
    d = np.load(npz, allow_pickle=True)
    ih = np.asarray(d["iter_heat"], float)        # (n_iter, n_interface)
    it = np.arange(ih.shape[0])
    fig, ax = style.figure(ncols=2, width=4.6, height=3.2)
    a = ax[0]
    for k in range(ih.shape[1]):
        a.plot(it, ih[:, k], "-", lw=1.1, label=f"interface {k}")
    a.set_xlabel("SCBA iteration"); a.set_ylabel("heat current (arb.)")
    a.legend(fontsize=7)
    a = ax[1]
    dev = np.abs(ih[:, 0] - ih[:, -1])            # lead-to-lead deviation
    a.semilogy(it, dev + 1e-30, "-", color="#d55e00", lw=1.3)
    a.set_xlabel("SCBA iteration")
    a.set_ylabel(r"lead deviation $|J_0-J_{\rm last}|$")
    print(f"[heat] final lead deviation = {dev[-1]:.3e}; "
          f"min over run = {dev.min():.3e}")
    style.save(fig, _name("heatcurrent", suffix), directory=FIGDIR)


# ----------------------------------------------------------------------------
# (2c) Spectral current + the omega->0 plateau (taper vs subtraction)
# ----------------------------------------------------------------------------
def fig_spectral_current(npz: Path = DIAG_NPZ, suffix="", overlay_taper=True):
    """Spectral current I(omega) and the energy-current density omega*I(omega):
    the omega->0 PLATEAU is the physical target (the omega^2 taper crushes it to
    zero; the IR occupation subtraction restores it). Overlays the taper
    baseline when available."""
    d = np.load(npz, allow_pickle=True)
    w = np.abs(np.asarray(d["energies"], float))
    I = np.asarray(d["current_spectrum"])[:, 0]
    I = np.sign(np.nanmean(I[(w > 5) & (w < 30)])) * I   # sign so forward>0
    fig, ax = style.figure(ncols=2, width=4.6, height=3.2)
    ax[0].plot(w, I, "-", color="#0173b2", lw=1.2, label="this run")
    ax[1].plot(w, w * I, "-o", color="#0173b2", ms=2.5, lw=1.2, label="this run")
    if overlay_taper and TAPER_NPZ.exists() and Path(npz) != TAPER_NPZ:
        dt = np.load(TAPER_NPZ, allow_pickle=True)
        wt = np.abs(np.asarray(dt["energies"], float))
        It = np.asarray(dt["current_spectrum"])[:, 0]
        It = np.sign(np.nanmean(It[(wt > 5) & (wt < 30)])) * It
        ax[0].plot(wt, It, "--", color="#949494", lw=1.0, label=r"$\omega^2$ taper")
        ax[1].plot(wt, wt * It, "--s", color="#949494", ms=2.5, lw=1.0,
                   label=r"$\omega^2$ taper")
    for a in ax:
        a.set_xlim(0, 18); a.set_xlabel("frequency (THz)"); a.legend(fontsize=7)
    ax[0].set_ylabel(r"spectral current $I(\omega)$")
    ax[1].set_ylabel(r"energy-current density $\omega\,I(\omega)$ (plateau)")
    ax[1].set_xlim(0, 6)
    style.save(fig, _name("spectral_current", suffix), directory=FIGDIR)


# ----------------------------------------------------------------------------
# (3) Per-omega, per-iteration spectral overlays
# ----------------------------------------------------------------------------
def _iter_selection(n: int):
    if n <= 6:
        return list(range(n))
    return sorted(set([0, 1, 2, n // 2, n - 2, n - 1]))


def fig_spectral_iters(npz: Path = DIAG_NPZ, suffix=""):
    d = np.load(npz, allow_pickle=True)
    w = np.abs(np.asarray(d["energies"], float))         # (ne,)
    gin = np.asarray(d["iter_gin_dos"])                  # (n_iter, ne)  -Im Tr G^R
    graw = np.asarray(d["iter_graw_w"])                  # (n_iter, ne)  raw max|G^<|
    gwin = np.asarray(d["iter_gwin_w"])                  # (n_iter, ne)  windowed max|G^<|
    sR = np.asarray(d["iter_sigR_w"])                    # (n_iter, ne)  max|Sigma^R|
    n_iter = gin.shape[0]
    sel = _iter_selection(n_iter)
    band_top = float(d["band_top"]) if "band_top" in d else FMAX_WINDOW
    print(f"[spectral] n_iter={n_iter}, selecting {sel}; ne={w.size}, "
          f"band_top={band_top:.1f} THz")

    # two columns: full 0..66 THz (the Si-H island) and a 0..fmax zoom.
    fig, ax = style.figure(ncols=2, nrows=4, width=4.4, height=2.2)
    cmap = style.plt.get_cmap("viridis")
    cols = [cmap(t) for t in np.linspace(0.0, 0.9, len(sel))]
    xlims = [(0, float(w.max())), (0, FMAX_WINDOW)]

    rows = [
        (gin, r"$-\mathrm{Im}\,\mathrm{Tr}\,G^R_{\mathrm{in}}$", "symlog"),
        (graw, r"raw $\max|G^<|$", "log"),
        (sR, r"$\max|\Sigma^R(\omega)|$", "log"),
        (gin, r"$-\mathrm{Im}\,\mathrm{Tr}\,G^R_{\mathrm{out}}$", "symlog"),
    ]
    for col in range(2):
        for r, (arr, ylab, yscale) in enumerate(rows):
            a = ax[r, col]
            for c, i in zip(cols, sel):
                if r == 1:  # G^< row: raw (solid) vs windowed (dashed)
                    a.plot(w, graw[i], "-", color=c, lw=1.0)
                    a.plot(w, gwin[i], "--", color=c, lw=0.9)
                elif r == 3:  # G_out = G_in of the next iteration
                    j = min(i + 1, n_iter - 1)
                    a.plot(w, gin[j], "-", color=c, lw=1.0, label=f"it {i}")
                else:
                    a.plot(w, arr[i], "-", color=c, lw=1.0, label=f"it {i}")
            if yscale == "log":
                a.set_yscale("log")
            elif yscale == "symlog":
                a.set_yscale("symlog", linthresh=1.0)
            a.axvline(FMAX_WINDOW, color="#949494", ls=":", lw=0.8)
            a.set_xlim(*xlims[col])
            if col == 0:
                a.set_ylabel(ylab, fontsize=8)
            if r == 3:
                a.set_xlabel("frequency (THz)")
    ax[0, 0].legend(fontsize=6, ncol=3, loc="upper right")
    ax[1, 0].plot([], [], "k-", label="raw"); ax[1, 0].plot([], [], "k--", label="windowed")
    ax[1, 0].legend(fontsize=6, loc="upper right")
    style.save(fig, _name("spectral_iters", suffix), directory=FIGDIR)


def fig_fluctuation(npz: Path = DIAG_NPZ, suffix=""):
    """WHERE it fluctuates: per-omega spread (std over iterations) of |Sigma^R|
    and |G^<|, normalised by the per-omega mean -> the relative oscillation
    amplitude vs frequency. Skip the first few iterations (transient)."""
    d = np.load(npz, allow_pickle=True)
    w = np.abs(np.asarray(d["energies"], float))
    sR = np.asarray(d["iter_sigR_w"])
    gl = np.asarray(d["iter_graw_w"])
    skip = min(5, sR.shape[0] // 4)
    sR, gl = sR[skip:], gl[skip:]

    def _spread(a):
        return a.std(axis=0)
    sR_std, gl_std = _spread(sR), _spread(gl)

    fig, ax = style.figure(ncols=2, width=4.4, height=3.2)
    a = ax[0]
    a.plot(w, sR_std, "-", color="#d55e00", lw=1.3, label=r"$\Sigma^R$")
    a.plot(w, gl_std, "-", color="#0173b2", lw=1.1, label=r"$G^<$")
    a.set_yscale("log")
    a.axvline(FMAX_WINDOW, color="#949494", ls=":", lw=0.9)
    a.set_xlabel("frequency (THz)"); a.set_xlim(0, float(w.max()))
    a.set_ylabel(r"iteration-to-iteration std")
    a.legend(fontsize=8)
    a = ax[1]
    a.plot(w, sR_std, "-", color="#d55e00", lw=1.4)
    a.plot(w, gl_std, "-", color="#0173b2", lw=1.1)
    a.set_yscale("log"); a.set_xlim(0, FMAX_WINDOW)
    a.axvline(FMAX_WINDOW, color="#949494", ls=":", lw=0.9)
    a.set_xlabel("frequency (THz)")
    # mark two well-separated fluctuation hot-spots: the global (near-DC) peak
    # and the strongest band-edge bin (in [8, FMAX_WINDOW]).
    i_dc = int(np.argmax(sR_std))
    edge = (w >= 8.0) & (w <= FMAX_WINDOW)
    i_edge = int(np.where(edge)[0][np.argmax(sR_std[edge])])
    for i in (i_dc, i_edge):
        a.annotate(f"{w[i]:.1f} THz", (w[i], sR_std[i]),
                   textcoords="offset points", xytext=(3, 4), fontsize=7,
                   color="#d55e00")
    print(f"[fluctuation] |Sigma^R| oscillation peaks: near-DC {w[i_dc]:.2f} THz, "
          f"band-edge {w[i_edge]:.2f} THz")
    style.save(fig, _name("fluctuation", suffix), directory=FIGDIR)


# ----------------------------------------------------------------------------
def main(which="all", tag="sinw_d5a_L2_eta0_diag", suffix=""):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    npz = CONV / f"{tag}.npz"
    log = CONV / f"{tag}.log"
    print("=" * 64 + f"\nsinw-d5a eta=0 DIAGNOSTICS  tag={tag} suffix='{suffix}'\n"
          + "=" * 64)
    if which in ("bands", "all"):
        fig_bands()
    if which in ("diag", "all"):
        if npz.exists():
            fig_convergence(log, suffix)
            fig_heat_current(npz, suffix)
            fig_spectral_current(npz, suffix)
            fig_spectral_iters(npz, suffix)
            fig_fluctuation(npz, suffix)
        else:
            print(f"[diag] {npz} not found yet -- run the diagnostic first.")
    print("\nfigures ->", FIGDIR)


if __name__ == "__main__":
    args = sys.argv[1:]
    _which = args[0] if args else "all"
    _tag = args[1] if len(args) > 1 else "sinw_d5a_L2_eta0_diag"
    _suffix = args[2] if len(args) > 2 else ""
    main(_which, _tag, _suffix)
