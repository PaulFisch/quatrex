"""Build the physics atlas for the two-cell-block CNT length ladder.

Run with ``--distill`` after pulling new result files, then without arguments
to redraw the seven PNG and PDF figure pairs from the compact distillate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "phonon"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phonon.postproc.cnt_ladder_physics import (  # noqa: E402
    apparent_mfp,
    bose,
    effective_transmission,
    lead_spectrum,
    linewidth_sector_matrix,
    mean_local_spectrum,
    modal_bubble_properties,
    spectral_quantiles,
)
from phonon.postproc.units import run_npz_conductance  # noqa: E402
from phonon.studies import style  # noqa: E402

OUT = ROOT / "phonon/studies/out/fig/cnt_reblocked_ladder_physics"
DIST = ROOT / "phonon/scripts/data/cnt_reblocked_ladder_physics.npz"
RESONANCE = ROOT / "phonon/scripts/data/resonance_gain_distilled.npz"
CELL_NM = 0.24595
INTERNAL_TO_W = 1.054571817e-34 * 2.0 * np.pi * 1e24
LENGTHS = np.array([16, 24, 32, 48, 64, 128])
DEEP_COMPLETE = np.array([16, 24, 32, 64, 128])
REPRESENTATIVE = np.array([16, 64, 128])

BALLISTIC = ROOT / "cluster/c16-ball/run.npz"
SHALLOW = {
    16: ROOT / "cluster/cnt-l16x2-conv-current/run.npz",
    24: ROOT / "cluster/cnt-l24x2-conv-current/run.npz",
    32: ROOT / "cluster/cnt-l32x2-conv-current-r2/run.npz",
    48: ROOT / "cluster/cnt-l48x2-conv-current/run.npz",
    64: ROOT / "cluster/cnt-l64x2-conv-current/run.npz",
    128: ROOT / "cluster/cnt-l128x2-conv-current-r2/run.npz",
}
DEEP = {
    16: ROOT / "cluster/cnt-l16x2-deep300-r2/run.npz",
    24: ROOT / "cluster/cnt-l24x2-deep300-r2/run.npz",
    32: ROOT / "cluster/cnt-l32x2-deep300-r3/run.npz",
    64: ROOT / "cluster/cnt-l64x2-deep300-r3/run.npz",
    128: ROOT / "cluster/cnt-l128x2-deep300-r3/run.npz",
}
DEEP_LOG = {
    16: (300, 6.4948e-8, 2.4107e-4, 11.555744267),
    24: (300, 2.2637e-7, 5.0995e-4, 10.535912177),
    32: (300, 7.3790e-8, 6.7475e-4, 9.802467362),
    48: (222, 1.7565e-6, 2.3864e-3, 8.790528),
    64: (300, 1.7526e-6, 3.4825e-3, 8.069135792),
    128: (300, 9.0109e-6, 6.4644e-3, 6.991447792),
}


def _load(path):
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; pull the result before --distill")
    return np.load(path, allow_pickle=True)


def _transport(path):
    run = _load(path)
    spectrum = lead_spectrum(run["current_spectrum"])
    summary = run_npz_conductance(path)
    return run, spectrum, float(summary["G_WK"])


def _causal_snapshot(path):
    snap = _load(path)
    dyn = snap["device_D"] if "device_D" in snap.files else snap["D"]
    omega, shift, width = modal_bubble_properties(
        snap["freqs"], dyn, snap["sigma_b"])
    return omega, shift, width


def distill():
    ball, ball_spectrum, ball_g = _transport(BALLISTIC)
    freqs = np.asarray(ball["energies"], dtype=float)
    weights = np.asarray(ball["frequency_cell_widths"], dtype=float)
    ball_integral = np.sum(weights * freqs * ball_spectrum)
    out = {
        "lengths": LENGTHS,
        "length_nm": LENGTHS * CELL_NM,
        "freqs": freqs,
        "weights": weights,
        "ball_spectrum": ball_spectrum,
        "ball_G": ball_g,
        "ball_integral": ball_integral,
        "t_left": float(ball["t_left"]),
        "t_right": float(ball["t_right"]),
    }

    shallow_g, shallow_spectrum, shallow_iter = [], [], []
    for length in LENGTHS:
        run, spectrum, conductance = _transport(SHALLOW[int(length)])
        shallow_g.append(conductance)
        shallow_spectrum.append(spectrum)
        shallow_iter.append(int(run["n_iter"]))
    out["shallow_G"] = np.array(shallow_g)
    out["shallow_spectrum"] = np.array(shallow_spectrum)
    out["shallow_iter"] = np.array(shallow_iter)

    deep_g = np.full(LENGTHS.size, np.nan)
    deep_spectrum = np.full((LENGTHS.size, freqs.size), np.nan)
    deep_source_commit = np.full(LENGTHS.size, "live-checkpoint", dtype="U40")
    for length, path in DEEP.items():
        run, spectrum, conductance = _transport(path)
        index = int(np.where(LENGTHS == length)[0][0])
        deep_g[index] = conductance
        deep_spectrum[index] = spectrum
        if "source_commit" in run.files:
            deep_source_commit[index] = str(run["source_commit"])
    live = int(np.where(LENGTHS == 48)[0][0])
    deep_g[live] = ball_g * DEEP_LOG[48][3] / ball_integral
    out["deep_G"] = deep_g
    out["deep_spectrum"] = deep_spectrum
    out["deep_result"] = np.isfinite(deep_spectrum).all(axis=1)
    out["deep_source_commit"] = deep_source_commit
    out["deep_iter"] = np.array([DEEP_LOG[int(n)][0] for n in LENGTHS])
    out["deep_residual"] = np.array([DEEP_LOG[int(n)][1] for n in LENGTHS])
    out["deep_balance"] = np.array([DEEP_LOG[int(n)][2] for n in LENGTHS])
    out["deep_log_current"] = np.array([DEEP_LOG[int(n)][3] for n in LENGTHS])

    local_ldos, local_occupation, local_temperature = [], [], []
    for length in REPRESENTATIVE:
        run = _load(DEEP[int(length)])
        ldos, occupation, temperature = mean_local_spectrum(
            run["energies"], run["gr_diag_imag"], run["gl_diag_imag"])
        local_ldos.append(ldos)
        local_occupation.append(occupation)
        local_temperature.append(temperature)
    ball_ldos, ball_occ, ball_temp = mean_local_spectrum(
        ball["energies"], ball["gr_diag_imag"], ball["gl_diag_imag"])
    out["representative"] = REPRESENTATIVE
    out["local_ldos"] = np.array(local_ldos)
    out["local_occupation"] = np.array(local_occupation)
    out["local_temperature"] = np.array(local_temperature)
    out["ball_ldos"] = ball_ldos
    out["ball_occupation"] = ball_occ
    out["ball_temperature"] = ball_temp

    resonance = _load(RESONANCE)
    for key in ("spec_Omega_harm", "spec_Gamma_anh", "spec_Gamma_lead",
                "spec_Gamma_tot", "spec_fit_Gamma", "spec_fit_ok",
                "spec_fit_resid", "dw"):
        out[f"linewidth_{key}"] = resonance[f"L4_stall__{key}"]
    coupling_sel = resonance["L2_fp__heat_sel"]
    out["coupling_freqs"] = resonance["L2_fp__spec_Omega_harm"][coupling_sel]
    out["coupling_matrix"] = resonance["L2_fp__heat_F2"]
    out["coupling_rowsum"] = resonance["L2_fp__heat_rowsum"]

    causal = ROOT / "cluster/snapshots/study_cnt33_T300_bubble.npz"
    causal_data = _load(causal)
    out["causal_freqs"] = causal_data["freqs"]
    out["causal_J_anh"] = causal_data["J_anh"]
    out["causal_J_ball"] = causal_data["J_ball"]
    out["causal_ratio"] = float(causal_data["Ga_over_Gb"])
    out["causal_omega"], out["causal_shift"], out["causal_width"] = \
        _causal_snapshot(causal)

    temperatures, ratios, residuals, converged = [], [], [], []
    for path in sorted((ROOT / "cluster/snapshots").glob("study_cnt33_T*_bubble.npz")):
        snap = _load(path)
        temperatures.append(float(snap["temp"]))
        ratios.append(float(snap["Ga_over_Gb"]))
        residuals.append(float(snap["resid"]))
        converged.append(bool(snap["converged"]))
    order = np.argsort(temperatures)
    out["causal_temperatures"] = np.asarray(temperatures)[order]
    out["causal_temperature_ratio"] = np.asarray(ratios)[order]
    out["causal_temperature_residual"] = np.asarray(residuals)[order]
    out["causal_temperature_converged"] = np.asarray(converged)[order]

    DIST.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DIST, **out)
    print(f"wrote {DIST} ({DIST.stat().st_size / 1e6:.2f} MB)")


def _data():
    if not DIST.exists():
        raise FileNotFoundError(f"missing {DIST}; run with --distill")
    return dict(np.load(DIST, allow_pickle=False))


def _save(fig, name):
    axes = [axis for axis in fig.axes if axis.get_label() != "<colorbar>"]
    style.panel_labels(axes)
    return style.save(fig, name, directory=OUT)


def plot_ladder(data):
    length = data["length_nm"]
    ball_g = float(data["ball_G"])
    shallow = data["shallow_G"]
    deep = data["deep_G"]
    ratio = deep / ball_g
    fig, axes = style.figure(ncols=2, nrows=2, width=4.5, height=3.35)
    ax = axes[0, 0]
    ax.axhline(ball_g * 1e9, color=style.C_BALLISTIC, ls="--",
               label="ballistic")
    ax.plot(length, shallow * 1e9, "o-", color="0.45",
            label=r"first $10^{-3}$ stop")
    ax.plot(length, deep * 1e9, "s-", color=style.C_ANHARMONIC,
            mfc="none", label="deep / live state")
    ax.plot(length[3], deep[3] * 1e9, "x", color=style.C_ANHARMONIC, ms=8,
            label="L48 has no final file")
    ax.set(xlabel="device length (nm)", ylabel="G (nW/K per tube)")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(length, shallow / ball_g, "o-", color="0.45", label="shallow")
    ax.plot(length, ratio, "s-", color=style.C_ANHARMONIC, mfc="none",
            label="deep")
    ax.set(xlabel="device length (nm)",
           ylabel=r"$G_{\rm anh}/G_{\rm ball}$", ylim=(0.0, 1.03))
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    mfp = apparent_mfp(length, ratio)
    ax.plot(length, mfp, "o-", color=style.C_THIRD)
    ax.set(xlabel="device length (nm)",
           ylabel=r"apparent $\lambda=L/(G_{\rm ball}/G-1)$ (nm)")
    ax.annotate("not constant: no single MFP", (0.04, 0.9),
                xycoords="axes fraction", fontsize=8)

    ax = axes[1, 1]
    ax.semilogy(length, data["deep_residual"], "o-",
                color=style.C_ANHARMONIC, label=r"relative $\Sigma^R$ residual")
    ax.semilogy(length, data["deep_balance"], "s-", color=style.C_THIRD,
                label="lead balance")
    ax.axhline(1e-8, color=style.C_ANHARMONIC, ls=":", lw=0.8)
    ax.axhline(1e-2, color=style.C_THIRD, ls=":", lw=0.8)
    ax.set(xlabel="device length (nm)", ylabel="endpoint diagnostic")
    ax.legend(fontsize=7)
    _save(fig, "ladder_transport")


def plot_spectral_transport(data):
    f = data["freqs"]
    w = data["weights"]
    tl, tr = float(data["t_left"]), float(data["t_right"])
    ball = data["ball_spectrum"]
    ball_t = effective_transmission(ball, f, tl, tr)
    reps = data["representative"]
    colors = ["#cc78bc", "#029e73", "#d55e00"]
    fig, axes = style.figure(ncols=2, nrows=2, width=4.6, height=3.35)

    axes[0, 0].plot(f, ball_t, color=style.C_BALLISTIC, label="ballistic")
    axes[0, 1].plot(f, INTERNAL_TO_W * f * ball * 1e12,
                    color=style.C_BALLISTIC, label="ballistic")
    ball_cum = np.cumsum(w * f * ball)
    axes[1, 1].plot(f, ball_cum / ball_cum[-1], color=style.C_BALLISTIC,
                    label="ballistic")
    for length, color in zip(reps, colors):
        index = int(np.where(data["lengths"] == length)[0][0])
        spectrum = data["deep_spectrum"][index]
        transmission = effective_transmission(spectrum, f, tl, tr)
        label = f"L{length} ({length * CELL_NM:.1f} nm)"
        axes[0, 0].plot(f, transmission, color=color, label=label)
        axes[0, 1].plot(f, INTERNAL_TO_W * f * spectrum * 1e12,
                        color=color, label=label)
        deficit = 1.0 - np.divide(transmission, ball_t,
                                  out=np.full_like(f, np.nan),
                                  where=ball_t > 0.05)
        axes[1, 0].plot(f, deficit, color=color, label=label)
        cumulative = np.cumsum(w * f * spectrum)
        axes[1, 1].plot(f, cumulative / cumulative[-1], color=color,
                        label=label)
    axes[0, 0].set(xlabel="frequency (THz)", ylabel="effective transmission")
    axes[0, 0].legend(fontsize=6.5)
    axes[0, 1].set(xlabel="frequency (THz)", ylabel=r"dJ/d$f$ (pW/THz)")
    axes[0, 1].legend(fontsize=6.5)
    axes[1, 0].set(xlabel="frequency (THz)",
                   ylabel=r"suppression $1-T_{\rm anh}/T_{\rm ball}$",
                   ylim=(-0.15, 1.05))
    axes[1, 1].set(xlabel="frequency (THz)",
                   ylabel="cumulative heat-current fraction", ylim=(0, 1.02))
    axes[1, 1].legend(fontsize=6.5)
    for ax in axes.ravel():
        ax.set_xlim(0, 50)
    _save(fig, "spectral_transport")


def plot_spectral_ladder(data):
    f = data["freqs"]
    w = data["weights"]
    tl, tr = float(data["t_left"]), float(data["t_right"])
    ball_t = effective_transmission(data["ball_spectrum"], f, tl, tr)
    deep_rows = np.where(data["deep_result"])[0]
    valid_f = (f >= 1.0) & (f <= 50.0) & (ball_t > 0.05)

    def ratios(spectra, rows):
        values = []
        for row in rows:
            transmission = effective_transmission(spectra[row], f, tl, tr)
            values.append(np.divide(transmission, ball_t,
                                    out=np.full_like(f, np.nan),
                                    where=ball_t > 0.05))
        return np.array(values)[:, valid_f]

    fig, axes = style.figure(ncols=2, nrows=2, width=4.6, height=3.35)
    shallow_ratio = ratios(data["shallow_spectrum"], np.arange(LENGTHS.size))
    deep_ratio = ratios(data["deep_spectrum"], deep_rows)
    for ax, matrix, labels, title in (
        (axes[0, 0], shallow_ratio, LENGTHS, r"first $10^{-3}$ stop"),
        (axes[0, 1], deep_ratio, LENGTHS[deep_rows], "deep endpoints"),
    ):
        image = ax.imshow(np.clip(matrix, 0.0, 1.2), origin="lower",
                          aspect="auto", vmin=0.0, vmax=1.0, cmap="magma",
                          extent=(f[valid_f][0], f[valid_f][-1], -0.5,
                                  len(labels) - 0.5), rasterized=True)
        ax.set_yticks(np.arange(len(labels)), [f"L{n}" for n in labels])
        ax.set(xlabel="frequency (THz)", ylabel="device", title=title)
    fig.colorbar(image, ax=axes[0].tolist(),
                 label=r"$T_{\rm anh}(f)/T_{\rm ball}(f)$")

    edges = np.array([0.0, 7.0, 20.0, 35.0, 55.1])
    labels = ["0-7", "7-20", "20-35", "35-55"]
    fractions = []
    q50, q90 = [], []
    for row in deep_rows:
        density = f * data["deep_spectrum"][row]
        total = np.sum(w * density)
        fractions.append([
            np.sum(w[(f >= lo) & (f < hi)] * density[(f >= lo) & (f < hi)]) / total
            for lo, hi in zip(edges[:-1], edges[1:])
        ])
        q = spectral_quantiles(f, density, w)
        q50.append(q[0])
        q90.append(q[1])
    fractions = np.asarray(fractions)
    for band, label, color in zip(fractions.T, labels,
                                  ["#0173b2", "#cc78bc", "#029e73", "#d55e00"]):
        axes[1, 0].plot(data["length_nm"][deep_rows], band, "o-",
                        color=color, label=label + " THz")
    axes[1, 0].set(xlabel="device length (nm)",
                   ylabel="fraction of heat current", ylim=(0, 0.65))
    axes[1, 0].legend(fontsize=6.5, ncols=2)
    axes[1, 1].plot(data["length_nm"][deep_rows], q50, "o-",
                    color=style.C_ANHARMONIC, label="50% below")
    axes[1, 1].plot(data["length_nm"][deep_rows], q90, "s-",
                    color=style.C_THIRD, label="90% below")
    axes[1, 1].set(xlabel="device length (nm)",
                   ylabel="heat-current quantile (THz)")
    axes[1, 1].legend(fontsize=7)
    _save(fig, "spectral_length_evolution")


def plot_local_spectra(data):
    f = data["freqs"]
    reps = data["representative"]
    colors = ["#cc78bc", "#029e73", "#d55e00"]
    fig, axes = style.figure(ncols=2, nrows=2, width=4.6, height=3.35)

    curves = [("ballistic L16", data["ball_ldos"], style.C_BALLISTIC)]
    curves += [(f"L{n}", data["local_ldos"][i], color)
               for i, (n, color) in enumerate(zip(reps, colors))]
    for label, ldos, color in curves:
        norm = np.trapezoid(ldos, f)
        density = ldos / norm if norm > 0.0 else ldos
        axes[0, 0].plot(f, density, color=color, label=label)
        axes[0, 1].plot(f, np.cumsum(data["weights"] * density), color=color,
                        label=label)
    axes[0, 0].set(xlabel="frequency (THz)",
                   ylabel="normalized LDOS per stored DOF")
    axes[0, 0].legend(fontsize=6.5)
    axes[0, 1].set(xlabel="frequency (THz)", ylabel="cumulative LDOS")

    n300 = bose(f, 300.0)
    for i, (length, color) in enumerate(zip(reps, colors)):
        ldos = data["local_ldos"][i]
        keep = (f > 1.0) & (ldos > 2e-3 * np.nanmax(ldos))
        temp = data["local_temperature"][i]
        ratio = np.divide(data["local_occupation"][i], n300,
                          out=np.full_like(f, np.nan), where=n300 > 1e-12)
        axes[1, 0].plot(f[keep], temp[keep], ".", ms=2.2, color=color,
                        label=f"L{length}")
        axes[1, 1].plot(f[keep], ratio[keep], ".", ms=2.2, color=color,
                        label=f"L{length}")
    axes[1, 0].axhspan(295, 305, color="0.85", zorder=0)
    axes[1, 0].set(xlabel="frequency (THz)",
                   ylabel=r"spectral $T_{\rm eff}$ (K)", ylim=(285, 315))
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].axhline(1.0, color="0.5", ls="--", lw=0.8)
    axes[1, 1].set(xlabel="frequency (THz)",
                   ylabel=r"occupation $n/n_B(300\,\mathrm{K})$", ylim=(0.8, 1.2))
    axes[1, 1].legend(fontsize=7)
    for ax in axes.ravel():
        ax.set_xlim(0, 50)
    _save(fig, "local_spectral_state")


def plot_linewidths(data):
    omega = data["linewidth_spec_Omega_harm"]
    anh = data["linewidth_spec_Gamma_anh"]
    lead = data["linewidth_spec_Gamma_lead"]
    total = data["linewidth_spec_Gamma_tot"]
    spacing = float(data["linewidth_dw"])
    valid = np.isfinite(total) & (total > 0.0) & (omega > 0.3)
    ratio = total[valid] / spacing
    fig, axes = style.figure(ncols=2, nrows=2, width=4.6, height=3.35)

    axes[0, 0].scatter(omega[valid], anh[valid], s=10, alpha=0.55,
                       color=style.C_ANHARMONIC, label="anharmonic")
    axes[0, 0].scatter(omega[valid], lead[valid], s=10, alpha=0.45,
                       color=style.C_BALLISTIC, label="contacts")
    axes[0, 0].scatter(omega[valid], total[valid], s=9, facecolors="none",
                       edgecolors="0.25", label="total")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(xlabel="harmonic mode frequency (THz)",
                   ylabel="HWHM (THz)")
    axes[0, 0].legend(fontsize=6.5)

    x = np.sort(ratio)
    axes[0, 1].plot(x, np.arange(1, x.size + 1) / x.size,
                    color=style.C_ANHARMONIC)
    axes[0, 1].axvline(1.0, color="0.45", ls="--", lw=0.8)
    axes[0, 1].axvline(2.0, color="0.65", ls=":", lw=0.8)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set(xlabel=r"$\Gamma_{\rm tot}/\Delta f$",
                   ylabel="cumulative mode fraction", ylim=(0, 1.02))

    fraction = np.divide(anh, total, out=np.full_like(total, np.nan),
                         where=total > 0.0)
    axes[1, 0].scatter(omega[valid], fraction[valid], s=11, alpha=0.55,
                       color=style.C_THIRD)
    axes[1, 0].axhline(0.5, color="0.6", ls="--", lw=0.8)
    axes[1, 0].set(xlabel="harmonic mode frequency (THz)",
                   ylabel=r"anharmonic share $\Gamma_{\rm anh}/\Gamma_{\rm tot}$",
                   ylim=(-0.03, 1.03))

    fit = data["linewidth_spec_fit_Gamma"]
    fit_ok = data["linewidth_spec_fit_ok"].astype(bool) \
        & (data["linewidth_spec_fit_resid"] < 0.15) & valid
    lo = min(np.nanmin(total[fit_ok]), np.nanmin(fit[fit_ok]))
    hi = max(np.nanmax(total[fit_ok]), np.nanmax(fit[fit_ok]))
    axes[1, 1].loglog(total[fit_ok], fit[fit_ok], "o", ms=3.5,
                      color="#cc78bc", alpha=0.65)
    axes[1, 1].plot([lo, hi], [lo, hi], color="0.4", ls="--", lw=0.8)
    axes[1, 1].set(xlabel=r"projected $\Gamma_{\rm tot}$ (THz)",
                   ylabel="spectral-fit HWHM (THz)")
    _save(fig, "mode_linewidths")


def plot_mode_coupling(data):
    omega = data["coupling_freqs"]
    coupling = data["coupling_matrix"]
    rowsum = data["coupling_rowsum"]
    edges = np.array([7.0, 20.0, 35.0, 50.0])
    sectors = linewidth_sector_matrix(coupling, omega, edges)
    sector_labels = ["7-20", "20-35", "35-50"]
    vmax = float(np.nanquantile(coupling, 0.99))
    fig, axes = style.figure(ncols=2, nrows=2, width=4.6, height=3.45)

    image = axes[0, 0].imshow(coupling, origin="lower", aspect="auto",
                              cmap="Blues", vmin=0.0, vmax=vmax,
                              extent=(omega[0], omega[-1], omega[0], omega[-1]),
                              rasterized=True)
    fig.colorbar(image, ax=axes[0, 0], label="partial linewidth fraction")
    axes[0, 0].set(xlabel="source mode frequency (THz)",
                   ylabel="receiving mode frequency (THz)")

    image = axes[0, 1].imshow(sectors, origin="lower", vmin=0.0, vmax=1.0,
                              cmap="magma")
    axes[0, 1].set_xticks(range(3), sector_labels, rotation=25)
    axes[0, 1].set_yticks(range(3), sector_labels)
    axes[0, 1].set(xlabel="source-frequency sector (THz)",
                   ylabel="receiving sector (THz)")
    for row in range(3):
        for col in range(3):
            axes[0, 1].text(col, row, f"{sectors[row, col]:.2f}",
                            ha="center", va="center",
                            color="white" if sectors[row, col] > 0.45 else "black",
                            fontsize=8)
    fig.colorbar(image, ax=axes[0, 1], label="row-normalized share")

    source = coupling.sum(axis=0)
    axes[1, 0].plot(omega, source, "o", ms=3, color=style.C_ANHARMONIC,
                    label="source-mode sum")
    axes[1, 0].plot(omega, rowsum, "s", ms=3, color=style.C_THIRD,
                    label="receiving-mode row sum")
    axes[1, 0].set(xlabel="mode frequency (THz)", ylabel="summed coupling")
    axes[1, 0].legend(fontsize=7)

    threshold = np.nanquantile(coupling, 0.985)
    receiver, source_mode = np.where(coupling >= threshold)
    size = 12.0 + 60.0 * coupling[receiver, source_mode] / coupling.max()
    scatter = axes[1, 1].scatter(
        omega[source_mode], omega[receiver], s=size,
        c=coupling[receiver, source_mode], cmap="viridis", alpha=0.7)
    fig.colorbar(scatter, ax=axes[1, 1], label="coupling strength")
    axes[1, 1].set(xlabel="source mode frequency (THz)",
                   ylabel="receiving mode frequency (THz)")
    _save(fig, "mode_to_mode_scattering")


def plot_causal_reference(data):
    f = data["causal_freqs"]
    ball = data["causal_J_ball"]
    anh = data["causal_J_anh"]
    norm = np.trapezoid(ball, f)
    fig, axes = style.figure(ncols=2, nrows=2, width=4.6, height=3.35)

    axes[0, 0].plot(f, ball / norm, color=style.C_BALLISTIC,
                    label="ballistic")
    axes[0, 0].plot(f, anh / norm, color=style.C_ANHARMONIC,
                    label="anharmonic FFT")
    axes[0, 0].set(xlabel="frequency (THz)",
                   ylabel="spectral heat current / ballistic integral",
                   xlim=(0, 55))
    axes[0, 0].legend(fontsize=7)

    scatter = axes[0, 1].scatter(
        data["causal_omega"], data["causal_width"],
        c=data["causal_shift"], cmap="coolwarm", s=22)
    axes[0, 1].set_yscale("log")
    fig.colorbar(scatter, ax=axes[0, 1], label="frequency shift (THz)")
    axes[0, 1].set(xlabel="harmonic mode frequency (THz)",
                   ylabel="bubble HWHM (THz)")

    axes[1, 0].plot(data["causal_omega"], data["causal_shift"], "o",
                    ms=4, color="#cc78bc")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.8)
    axes[1, 0].set(xlabel="harmonic mode frequency (THz)",
                   ylabel="on-shell frequency shift (THz)")

    temperature = data["causal_temperatures"]
    ratio = data["causal_temperature_ratio"]
    converged = data["causal_temperature_converged"].astype(bool)
    axes[1, 1].plot(temperature, ratio, "-", color=style.C_ANHARMONIC)
    axes[1, 1].plot(temperature[converged], ratio[converged], "o",
                    color=style.C_ANHARMONIC, label="converged")
    if np.any(~converged):
        axes[1, 1].plot(temperature[~converged], ratio[~converged], "o",
                        color=style.C_ANHARMONIC, mfc="none",
                        label="not converged")
    axes[1, 1].set(xlabel="temperature (K)",
                   ylabel=r"$G_{\rm anh}/G_{\rm ball}$", ylim=(0, 1.03))
    axes[1, 1].legend(fontsize=7)
    _save(fig, "causal_fft_reference")


def write_summary(data):
    rows = []
    for index, length in enumerate(data["lengths"]):
        rows.append({
            "cells": int(length),
            "length_nm": float(data["length_nm"][index]),
            "shallow_G_nW_per_K": float(data["shallow_G"][index] * 1e9),
            "deep_G_nW_per_K": float(data["deep_G"][index] * 1e9),
            "deep_G_over_Gball": float(data["deep_G"][index] / data["ball_G"]),
            "deep_iteration": int(data["deep_iter"][index]),
            "deep_residual": float(data["deep_residual"][index]),
            "deep_lead_balance": float(data["deep_balance"][index]),
            "has_final_result": bool(data["deep_result"][index]),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ladder_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "ballistic_G_nW_per_K": float(data["ball_G"] * 1e9),
        "ladder": rows,
        "limitations": [
            "deep ladder uses non-causal half-retarded reconstruction",
            "no deep endpoint reaches the 1e-8 self-energy gate",
            "L48 is a live iteration-222 value without run.npz",
            "distributed local spectra cover rank-0-owned DOF only",
        ],
    }
    (OUT / "physics_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT / 'ladder_summary.csv'} and physics_summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill", action="store_true")
    args = parser.parse_args()
    if args.distill or not DIST.exists():
        distill()
    data = _data()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_ladder(data)
    plot_spectral_transport(data)
    plot_spectral_ladder(data)
    plot_local_spectra(data)
    plot_linewidths(data)
    plot_mode_coupling(data)
    plot_causal_reference(data)
    write_summary(data)
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
