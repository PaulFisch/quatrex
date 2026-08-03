"""Resonance / loop-gain verification figures (thesis eq:resolvent_gain).

  resonance_response  (a) mode-projected spectral combs A_s(omega) at the
                      CNT(3,3) L2 converged fixed point with their
                      omega^2-Lorentzian fits (grid spacing marked);
                      (b) one-shot distortion response of the kicked
                      resonance against the theory ratio
                      |dSigma|/(2 Omega_s Gamma_s), peak-local (filled)
                      and l2-norm over the comb (open), for width-like
                      (-i) and pole-shift-like (+1) kicks.
  loop_gain           (a) channel-fraction heatmap
                      F[s'', s] = sum_{s'} Gamma^{(s,s')}_{s''}/Gamma_{s''}
                      over all device modes (ordered by Omega), with row
                      sums; (b) predicted spectral radius of the
                      link-gain matrix vs the measured power-iteration
                      |lambda| for L2 fp / L2 stall / L4 stall.

Data: phonon/scripts/data/resonance_gain_distilled.npz (committed-size
distillate). Regenerate from the full study output
(phonon/studies/out/resonance_gain/<state>.npz, produced on tortin by
phonon/studies/_resonance_gain_study.py) with --distill.

Run:  python phonon/scripts/figures/resonance_gain.py [--distill]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.studies import style  # noqa: E402

DATA = ROOT / "phonon/scripts/data"
FULL = ROOT / "phonon/studies/out/resonance_gain"
FIGDIR = ROOT / "document/fig/transport_sweeps"
DIST = DATA / "resonance_gain_distilled.npz"

STATES = ["L2_fp", "L2_andstall", "L4_stall"]


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------


def _near_dc_share(f, s, n_bins=5):
    """Share of the skew channel-(s,s) convolution integrand at Omega_s
    carried by configurations with one leg within n_bins of DC."""
    w = f["w"]
    dw = float(f["dw"])
    gl = f["ch_g_modes_lesser"].astype(complex)
    gg = f["ch_g_modes_greater"].astype(complex)
    Om = f["spec_Omega_harm"]
    nf = gl.shape[1]
    mid = nf // 2
    wf = np.concatenate([-w[1:][::-1], w])
    iw = int(round(Om[s] / dw))
    m = np.arange(nf)
    r = mid + iw - (m - mid)
    ok = (r >= 0) & (r < nf)
    m, r = m[ok], r[ok]
    diff = gl[s][m] * gl[s][r] - gg[s][m] * gg[s][r]
    near = (np.abs(wf[m]) <= n_bins * dw + 1e-9) \
        | (np.abs(wf[r]) <= n_bins * dw + 1e-9)
    return float(np.abs(diff[near]).sum() / np.abs(diff).sum())


def distill() -> None:
    out = {}
    for st in STATES:
        f = np.load(FULL / f"{st}.npz", allow_pickle=False)
        pre = f"{st}__"
        for k in ("w", "dw", "measured_lambda", "spec_Omega_harm",
                  "spec_Gamma_anh", "spec_Gamma_lead", "spec_Gamma_tot",
                  "spec_fit_Omega", "spec_fit_Gamma", "spec_fit_Z",
                  "spec_fit_c", "spec_fit_ok", "spec_fit_resid",
                  "spec_fit_nwin", "spec_peak_w",
                  "ch_fp_sign", "ch_fp_err_lesser", "ch_fp_err_greater",
                  "ch_wiring_gate", "ch_closure_lesser", "ch_closure_greater",
                  "ch_row_norm_ref", "ch_row_norm_snap", "ch_Gamma_ref_at",
                  "gain_rho_phys_all", "gain_rho_grid_all"):
            if k in f:
                out[pre + k] = f[k]
        Om = f["spec_Omega_harm"]
        Gch = f["ch_Gch"]
        Gref = f["ch_Gamma_ref_at"]
        Gt = f["spec_Gamma_tot"]
        dw = float(f["dw"])
        # channel-fraction matrix over all valid modes, ordered by Omega
        sel = np.where((Om > f["w"][1]) & np.isfinite(Gt) & (Gt > 0))[0]
        sel = sel[np.argsort(Om[sel])]
        with np.errstate(divide="ignore", invalid="ignore"):
            F2 = (Gch.sum(axis=2) * 2.0) / Gref[:, None]
        out[pre + "heat_sel"] = sel
        out[pre + "heat_F2"] = F2[np.ix_(sel, sel)]
        out[pre + "heat_rowsum"] = (2.0 * Gch.sum(axis=(1, 2)) / Gref)[sel]
        # one-leg link-gain spectral radius (the theory's literal reading,
        # without the bilinear two-leg sensitivity factor)
        M1 = np.zeros_like(f["gain_M_phys"])
        with np.errstate(divide="ignore", invalid="ignore"):
            M1 = (Om[:, None] * Gch.sum(axis=2)) / (Om[None, :] * Gt[None, :])
        M1[~np.isfinite(M1)] = 0.0
        idx = np.where(f["gain_sel_valid"])[0]
        out[pre + "rho_oneleg"] = np.max(np.abs(
            np.linalg.eigvals(M1[np.ix_(idx, idx)])))
        # dominant-cycle diagnostics: top-eigvec support of M_phys
        M = f["gain_M_phys"][np.ix_(idx, idx)]
        ev, V = np.linalg.eig(M)
        k = int(np.argmax(np.abs(ev)))
        v = np.abs(V[:, k])
        top = idx[np.argsort(v)[::-1][:6]]
        out[pre + "cycle_modes"] = top
        out[pre + "cycle_Omega"] = Om[top]
        out[pre + "cycle_Gamma"] = Gt[top]
        # width-sensitivity bracket: the theory's Gamma_s is the fitted
        # Lorentzian half-width; the fits run 0.3-1x the Sigma-diagonal
        # projection (mode hybridisation), which rescales the link-gain
        # denominator. Bracket rho by the median fit/Sigma width ratio.
        pw = f["spec_peak_w"]
        okf = (f["spec_fit_ok"].astype(bool) & (f["spec_fit_resid"] < 0.15)
               & (np.abs(pw - Om) < 5 * dw)
               & (f["spec_fit_Gamma"] > dw / 4) & (f["spec_fit_Gamma"] < 5))
        ratio = float(np.median(f["spec_fit_Gamma"][okf] / Gt[okf]))
        out[pre + "fit_over_sigma_ratio"] = ratio
        out[pre + "rho_fitwidth_bracket"] = np.array(
            [float(f["gain_rho_phys_all"]),
             float(f["gain_rho_phys_all"]) / ratio])
        if st == "L2_fp":
            out[pre + "near_dc_share"] = np.array(
                [_near_dc_share(f, s) for s in top[:3]])
            for kfile in ("L2_fp_kicks3.npz", "L2_fp_kicks2.npz"):
                if (FULL / kfile).exists():
                    fk = np.load(FULL / kfile, allow_pickle=False)
                    break
            else:
                fk = f
            out[pre + "kick_table"] = fk["kick_table"]
            out[pre + "kick_columns"] = fk["kick_columns"]
            # representative combs: clean quasiparticle fits (peak at the
            # kicked resonance, physical width, positive weight) spanning
            # sharp -> broad
            resid = f["spec_fit_resid"]
            Gfit = f["spec_fit_Gamma"]
            Zfit = f["spec_fit_Z"]
            pw = f["spec_peak_w"]
            ok = (f["spec_fit_ok"].astype(bool) & (resid < 0.15)
                  & np.isfinite(Gfit) & (Gfit > dw / 4) & (Gfit < 5.0)
                  & (Zfit > 0.2) & (Zfit < 3.0)
                  & (np.abs(pw - Om) < 5 * dw) & (Om > 5.0))
            cand = np.where(ok)[0]
            cand = cand[np.argsort(Gfit[cand])]
            reps = [cand[0], cand[len(cand) // 2], cand[-1]]
            reps = sorted(set(int(r) for r in reps), key=lambda s: Gfit[s])
            out[pre + "rep_modes"] = np.array(reps)
            out[pre + "rep_A"] = f["spec_A_modes"][reps]
    np.savez_compressed(DIST, **out)
    print(f"wrote {DIST} ({DIST.stat().st_size/1e6:.2f} MB)")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def lorentz_w2(w, Om, Ga, Z, c):
    return 2.0 * Z * (2.0 * Om * Ga) / ((w**2 - Om**2)**2
                                        + (2.0 * Om * Ga)**2) + c


def fig_resonance_response(d) -> None:
    st = "L2_fp"
    w = d[f"{st}__w"]
    dw = float(d[f"{st}__dw"])
    reps = d[f"{st}__rep_modes"]
    repA = d[f"{st}__rep_A"]
    Om_f = d[f"{st}__spec_fit_Omega"]
    Ga_f = d[f"{st}__spec_fit_Gamma"]
    Z_f = d[f"{st}__spec_fit_Z"]
    c_f = d[f"{st}__spec_fit_c"]
    Gam_tot = d[f"{st}__spec_Gamma_tot"]

    fig, (ax_a, ax_b) = style.figure(ncols=2, width=4.4, height=3.4)
    colors = style.RC["axes.prop_cycle"].by_key()["color"]

    wf = np.linspace(w[0], w[-1], 6001)
    for i, (s, A) in enumerate(zip(reps, repA)):
        c = colors[i]
        p = int(np.argmax(A))
        x0 = w[p]
        sl = slice(max(p - 8, 0), min(p + 9, len(w)))
        ax_a.plot(w[sl] - x0, A[sl] / A[p], "o", ms=3.5, color=c)
        yf = lorentz_w2(wf, Om_f[s], Ga_f[s], Z_f[s], c_f[s])
        m = (wf >= w[sl.start]) & (wf <= w[sl.stop - 1])
        ax_a.plot(wf[m] - x0, yf[m] / A[p], "-", lw=1.2, color=c,
                  label=rf"$\Gamma_s$ = {Ga_f[s]:.2f} THz")
    ax_a.axvspan(-dw / 2, dw / 2, color="0.85", zorder=0)
    ax_a.annotate(r"$\Delta\omega$", (0.02, 0.985), xycoords="axes fraction",
                  ha="left", va="top", fontsize=8, color="0.35")
    ax_a.set_xlabel(r"$\omega - \omega_{\mathrm{peak}}$ (THz)")
    ax_a.set_ylabel(r"$A_s(\omega)\,/\,A_s^{\mathrm{peak}}$")
    ax_a.legend(loc="upper right", handlelength=1.2)
    ax_a.set_xlim(-2.6, 2.6)

    # (b) kick scatter. Modes whose projected comb peaks away from the
    # kicked resonance (strong hybridisation: the |s>-projection carries
    # its weight at a partner frequency) respond at rounding level in the
    # peak window; they are excluded here and counted in the claim trail.
    kt = d[f"{st}__kick_table"]
    cols = [str(c) for c in d[f"{st}__kick_columns"]]
    x = kt[:, cols.index("x_phys")]
    y = kt[:, cols.index("y")]
    yp = kt[:, cols.index("y_peak")]
    real = kt[:, cols.index("is_real")] > 0.5
    smode = kt[:, cols.index("mode")].astype(int)
    pw = d[f"{st}__spec_peak_w"]
    Om_h = d[f"{st}__spec_Omega_harm"]
    onpeak = np.abs(pw[smode] - Om_h[smode]) < 3 * dw
    n_excl = len(set(smode[~onpeak]))
    ax_b.plot(x[~real & onpeak], yp[~real & onpeak], "o", ms=4.5,
              color=colors[0], label=r"width kick ($-i$), peak")
    ax_b.plot(x[real & onpeak], yp[real & onpeak], "s", ms=4,
              color=colors[1], label=r"pole kick ($+1$), peak")
    ax_b.plot(x[~real & onpeak], y[~real & onpeak], "o", ms=3,
              color=colors[0], mfc="none", alpha=0.45,
              label=r"$\ell_2$ over comb")
    ax_b.plot(x[real & onpeak], y[real & onpeak], "s", ms=3,
              color=colors[1], mfc="none", alpha=0.45)
    lim = [np.nanmin(x[x > 0]) / 3, np.nanmax(x) * 3]
    ax_b.plot(lim, lim, "-", color="0.4", lw=1.0, label="theory $y=x$")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_ylim(bottom=1e-7)
    ax_b.set_xlabel(r"$|\delta\Sigma|/(2\Omega_s\Gamma_s)$")
    ax_b.set_ylabel(r"$|\delta A_s|/|A_s|$")
    ax_b.legend(loc="lower right", fontsize=7)

    style.save(fig, "resonance_response", directory=FIGDIR)

    for tag, sel in (("width", ~real & onpeak), ("pole", real & onpeak)):
        r = yp[sel] / x[sel]
        print(f"[claim] {tag} kicks (on-peak modes): y_peak/x median "
              f"{np.median(r):.3f}, "
              f"IQR [{np.quantile(r, .25):.3f}, {np.quantile(r, .75):.3f}], "
              f"range [{r.min():.2e}, {r.max():.2e}] (n={sel.sum()})")
    print(f"[claim] kicks: {n_excl} of {len(set(smode))} kicked modes "
          f"excluded (projected comb peaks > 3 dw from Omega_s: "
          f"hybridised modes, peak response at rounding level)")
    print(f"[claim] L2 fp grid: dw = {dw:.4f} THz; Gamma_tot range "
          f"[{np.nanmin(Gam_tot):.3f}, {np.nanmax(Gam_tot):.3f}] THz -> "
          f"{int(np.nansum(Gam_tot < dw))} of "
          f"{int(np.sum(np.isfinite(Gam_tot)))} modes grid-limited")


def fig_loop_gain(d) -> None:
    st = "L2_fp"
    sel = d[f"{st}__heat_sel"]
    F2 = d[f"{st}__heat_F2"]
    rows = d[f"{st}__heat_rowsum"]
    Om = d[f"{st}__spec_Omega_harm"][sel]

    fig, (ax_a, ax_b) = style.figure(ncols=2, width=4.7, height=3.5)
    colors = style.RC["axes.prop_cycle"].by_key()["color"]

    n = len(sel)
    im = ax_a.imshow(F2, origin="lower", aspect="auto", cmap="Blues",
                     vmin=0.0, vmax=float(np.nanquantile(F2, 0.995)),
                     extent=(-0.5, n - 0.5, -0.5, n - 0.5))
    cb = fig.colorbar(im, ax=ax_a, location="top", pad=0.02, shrink=0.85)
    cb.set_label(r"$2\sum_{s'}\Gamma^{(s,s')}_{s''}/\Gamma_{s''}$",
                 fontsize=8)
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.xaxis.set_label_position("top")
    tick = np.arange(0, n, 10)
    ax_a.set_xticks(tick, [f"{v:.0f}" for v in Om[tick]])
    ax_a.set_yticks(tick, [f"{v:.0f}" for v in Om[tick]])
    ax_a.set_xlabel(r"source mode $\Omega_s$ (THz)")
    ax_a.set_ylabel(r"receiving mode $\Omega_{s''}$ (THz)")
    ax_a.grid(False)
    ax2 = ax_a.inset_axes([1.06, 0.0, 0.16, 1.0], sharey=ax_a)
    ax2.barh(np.arange(n), rows, color=colors[0], height=0.9)
    ax2.axvline(1.0, color="0.4", lw=0.9, ls="--")
    ax2.set_xlabel("row sum", fontsize=8)
    ax2.tick_params(labelleft=False, labelsize=7)
    ax2.grid(False)

    # (b) predicted vs measured
    marker = {"L2_fp": "o", "L2_andstall": "s", "L4_stall": "D"}
    label = {"L2_fp": "L2 fixed point", "L2_andstall": "L2 stall",
             "L4_stall": "L4 stall"}
    off = {"L2_fp": (6, -3), "L2_andstall": (-8, 8), "L4_stall": (6, -3)}
    for st_ in STATES:
        lam = np.max(np.abs(d[f"{st_}__measured_lambda"]))
        r_two = float(d[f"{st_}__gain_rho_phys_all"])
        r_one = float(d[f"{st_}__rho_oneleg"])
        ax_b.plot([lam], [r_two], marker[st_], ms=6.5, color=colors[0])
        ax_b.plot([lam], [r_one], marker[st_], ms=6.5, color=colors[1],
                  mfc="none")
        ax_b.annotate(label[st_], (lam, r_two), textcoords="offset points",
                      xytext=off[st_], fontsize=7.5,
                      ha="left" if off[st_][0] > 0 else "right")
    lim = [0.0, 6.2]
    ax_b.plot(lim, lim, "-", color="0.4", lw=1.0)
    ax_b.annotate("perfect\nprediction", (5.0, 5.0),
                  textcoords="offset points", xytext=(-34, 10), fontsize=8,
                  color="0.45")
    ax_b.set_xlim(lim)
    ax_b.set_ylim(0.0, 6.2)
    ax_b.set_xlabel(r"measured $|\lambda|$ (power iteration)")
    ax_b.set_ylabel(r"predicted $\rho$ (link-gain matrix)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=colors[0],
                      label=r"$\rho$, two-leg (bilinear)"),
               Line2D([], [], marker="o", ls="", color=colors[1], mfc="none",
                      label=r"$\rho$, one-leg (eq. gain)")]
    handles += [Line2D([], [], marker=marker[s], ls="", color="0.3",
                       label=label[s]) for s in STATES]
    ax_b.legend(handles=handles, loc="upper left", fontsize=7)

    style.save(fig, "loop_gain", directory=FIGDIR)

    for st_ in STATES:
        print(f"[claim] {st_}: measured |lambda| = "
              f"{np.abs(d[f'{st_}__measured_lambda'])}; predicted rho "
              f"two-leg {float(d[f'{st_}__gain_rho_phys_all']):.3f}, "
              f"one-leg {float(d[f'{st_}__rho_oneleg']):.3f}, "
              f"grid-enhanced {float(d[f'{st_}__gain_rho_grid_all']):.3f} "
              f"(== two-leg: every width resolved)")
        print(f"[claim] {st_}: wiring gate "
              f"{float(d[f'{st_}__ch_wiring_gate']):.2e}; kernel-vs-snapshot "
              f"{float(d[f'{st_}__ch_fp_err_lesser']):.2e} (sign "
              f"{float(d[f'{st_}__ch_fp_sign']):+.0f}); closure "
              f"{float(d[f'{st_}__ch_closure_lesser']):.1%}; row-sum "
              f"median {np.nanmedian(d[f'{st_}__ch_row_norm_ref']):.3f}")
        print(f"[claim] {st_}: dominant-cycle modes Omega = "
              f"{np.round(d[f'{st_}__cycle_Omega'], 2)} THz, Gamma = "
              f"{np.round(d[f'{st_}__cycle_Gamma'], 3)} THz")
        br = d[f"{st_}__rho_fitwidth_bracket"]
        print(f"[claim] {st_}: rho width-sensitivity bracket "
              f"[{br[0]:.2f}, {br[1]:.2f}] (fitted widths are "
              f"{float(d[f'{st_}__fit_over_sigma_ratio']):.2f}x the "
              f"Sigma-diagonal projection, median over clean fits)")
    print(f"[claim] L2_fp near-DC (|w'|<=5dw) share of the dominant "
          f"self-link skew integrands: "
          f"{np.round(d['L2_fp__near_dc_share'], 3)}")


def print_fit_table(d) -> None:
    st = "L2_fp"
    Om = d[f"{st}__spec_Omega_harm"]
    Ga_anh = d[f"{st}__spec_Gamma_anh"]
    Ga_lead = d[f"{st}__spec_Gamma_lead"]
    fOm = d[f"{st}__spec_fit_Omega"]
    fGa = d[f"{st}__spec_fit_Gamma"]
    res = d[f"{st}__spec_fit_resid"]
    ok = d[f"{st}__spec_fit_ok"]
    dw = float(d[f"{st}__dw"])
    print("\n[claim] L2 fp (Omega, Gamma) table (THz): harmonic Omega_s; "
          "fitted Omega/Gamma (omega^2-Lorentzian); Sigma-derived "
          "Gamma_anh (anharmonic) and Gamma_lead (contact); all "
          f"Gamma > dw = {dw:.3f} -> resolved")
    print("   s  Om_harm  Om_fit  Gam_fit  fit_res  Gam_anh  Gam_lead")
    for s in range(len(Om)):
        if Om[s] < 0.3:
            continue
        print(f"  {s:3d} {Om[s]:8.3f} {fOm[s]:7.2f} {fGa[s]:8.3f} "
              f"{res[s]:8.3f} {Ga_anh[s]:8.3f} {Ga_lead[s]:9.3f}"
              f"{'' if ok[s] else '   [fit rejected]'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distill", action="store_true",
                    help="rebuild the distillate from the full study NPZs")
    args = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    if args.distill or not DIST.exists():
        distill()
    d = dict(np.load(DIST, allow_pickle=False))
    fig_resonance_response(d)
    fig_loop_gain(d)
    print_fit_table(d)


if __name__ == "__main__":
    main()
