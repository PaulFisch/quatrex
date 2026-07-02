"""Anharmonic phonon linewidth: QuatRex NEGF self-energy vs a phono3py reference.

phono3py reference (created here, cached to fig/phono3py_ref_*.npz):
  mode linewidths gamma(q,s) [THz, HWHM] from FC2+FC3 via RTA imag-self-energy
  at 300 K, on a [1,1,M] wire mesh.

QuatRex NEGF (se_study bubble Sigma_B(omega), 300 K):
  per-mode linewidth  Gamma_s = |<e_s|Im Sigma(omega_s)|e_s>| / (2 omega_s)
  with (omega_s^2, e_s) the Gamma-point modes of the device dynamical matrix.

Both are the anharmonic broadening of the Gamma (q=0) modes -> directly comparable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO, REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
from phonon.finite_analysis.loader import load_system  # noqa: E402

SE = REPO / "cluster" / "tortin3-tmp" / "se_study"
C = {"p3p": "#BBBBBB", "p3pG": "#4477AA", "negf": "#AA3377"}

SYS = {
    "cnt33": dict(cfg="phonon/configs/cnt/cnt33_vasp.yaml",
                  fc3="cluster/tortin3-tmp/cnt_fc4/fc3.hdf5", mesh=21, label="CNT(3,3)"),
    "d5a": dict(cfg="phonon/configs/sinw/sinw100_d5a_vasp_sc4_fc4.yaml",
                fc3="cluster/tortin3-tmp/d5a_fc3_ardr_backup.hdf5", mesh=21, label="SiNW d5a"),
}

plt.rcParams.update({"font.size": 10, "legend.fontsize": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})


def _cnt33_phono3py():
    """Build CNT(3,3) phono3py directly: the local FC3 is a [1,1,3] (36-atom)
    cell that mismatches the config reap metadata, so bypass load_system."""
    import h5py, yaml
    from phonopy.structure.atoms import PhonopyAtoms
    from phono3py import Phono3py
    cfg = yaml.safe_load((REPO / SYS["cnt33"]["cfg"]).read_text())["structure"]
    cell = PhonopyAtoms(symbols=cfg["symbols"], cell=np.array(cfg["lattice"], float),
                        scaled_positions=np.array(cfg["scaled_positions"], float))
    with h5py.File(REPO / SYS["cnt33"]["fc3"], "r") as f:
        fc2, fc3 = f["fc2"][:], f["fc3"][:]
    ph3 = Phono3py(unitcell=cell, supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 3]],
                   primitive_matrix=np.eye(3), log_level=0)
    ph3.fc2, ph3.fc3 = fc2, fc3
    return ph3


def phono3py_ref(system):
    cache = HERE / "fig" / f"phono3py_ref_{system}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["w_all"], d["g_all"], d["w_G"], d["g_G"]
    from phono3py import Phono3py
    s = SYS[system]
    if system == "cnt33":
        ph3 = _cnt33_phono3py()
    else:
        b = load_system(str(REPO / s["cfg"]), validate=False, transport_axis=2,
                        fc3_path_override=str(REPO / s["fc3"]))
        ph = b.phonon
        ph3 = Phono3py(unitcell=ph.unitcell, supercell_matrix=ph.supercell_matrix,
                       primitive_matrix=ph.primitive_matrix, log_level=0)
        ph3.fc2 = b.fc2; ph3.fc3 = b.fc3_raw
    ph3.mesh_numbers = [1, 1, s["mesh"]]
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=[300.0], boundary_mfp=1e6)
    tc = ph3.thermal_conductivity
    g = np.array(tc.gamma)[0, 0]       # (n_grid, n_band) HWHM [THz] (drop temp,sigma)
    fr = np.array(tc.frequencies)      # (n_grid, n_band)
    # grid point 0 is Gamma (q=0)
    w_G, g_G = fr[0], g[0]
    np.savez(cache, w_all=fr.ravel(), g_all=g.ravel(), w_G=w_G, g_G=g_G)
    return fr.ravel(), g.ravel(), w_G, g_G


def _hwhm(fr, prof, ip):
    """Interpolated half-width-half-max of prof around peak index ip."""
    half = prof[ip] / 2.0
    l = ip
    while l > 0 and prof[l] > half:
        l -= 1
    wl = (fr[l] + (half - prof[l]) / (prof[l + 1] - prof[l]) * (fr[l + 1] - fr[l])
          if prof[l + 1] != prof[l] else fr[l])
    r = ip
    while r < len(fr) - 1 and prof[r] > half:
        r += 1
    wr = (fr[r - 1] + (half - prof[r - 1]) / (prof[r] - prof[r - 1]) * (fr[r] - fr[r - 1])
          if prof[r] != prof[r - 1] else fr[r])
    return 0.5 * (wr - wl), wl, wr


def negf_linewidths(system, eta=0.02):
    """Per Gamma-mode: on-shell |ImSigma|/2w, genuine spectral-function HWHM,
    and a 'resolved' flag (peak full-width >= 3 grid points). Also returns one
    well-resolved example (freqs, A_s, peak, half-max crossings) for an inset."""
    z = np.load(SE / f"study_{system}_T300_bubble.npz", allow_pickle=True)
    sb = np.asarray(z["sigma_b"]); fr = np.asarray(z["freqs"])
    D = np.asarray(z["device_D"]).real; D = 0.5 * (D + D.T)
    w2, E = np.linalg.eigh(D)
    ws = np.sign(w2) * np.sqrt(np.abs(w2))
    dw = float(fr[1] - fr[0]); N = D.shape[0]; I = np.eye(N)

    # mode-projected spectral function A_s(w) = -2 Im <e_s|G_R|e_s>
    A = np.empty((len(fr), N))
    for iw, w in enumerate(fr):
        GR = np.linalg.inv(((w + 1j * eta) ** 2) * I - D - sb[iw])
        A[iw] = -2.0 * np.imag(np.einsum("is,ij,js->s", E.conj(), GR, E))

    g_on = np.full(N, np.nan); g_sp = np.full(N, np.nan); resolved = np.zeros(N, bool)
    for s in range(N):
        if ws[s] < 0.5:
            continue
        iw0 = int(np.argmin(np.abs(fr - ws[s])))
        g_on[s] = abs(float(E[:, s].conj() @ sb[iw0].imag @ E[:, s])) / (2 * ws[s])
        win = np.abs(fr - ws[s]) < max(3.0, 0.4 * ws[s])
        if win.sum() < 4:
            continue
        idx = np.where(win)[0]
        ip = idx[np.argmax(A[idx, s])]
        if ip == idx[0] or ip == idx[-1] or A[ip, s] <= 0:
            continue
        hw, wl, wr = _hwhm(fr, A[:, s], ip)
        g_sp[s] = hw
        resolved[s] = (wr - wl) >= 3 * dw

    # an example: the most-resolved mid-band mode for the inset
    cand = [s for s in range(N) if resolved[s] and 3 < ws[s] < 0.7 * fr[-1]]
    ex = None
    if cand:
        s = max(cand, key=lambda s: A[int(np.argmin(np.abs(fr - ws[s]))), s])
        ip = int(np.argmin(np.abs(fr - ws[s])))
        ip = np.where(np.abs(fr - ws[s]) < max(3.0, 0.4 * ws[s]))[0]
        ip = ip[np.argmax(A[ip, s])]
        hw, wl, wr = _hwhm(fr, A[:, s], ip)
        m = np.abs(fr - fr[ip]) < 6 * hw + 1
        ex = dict(s=s, ws=ws[s], fr=fr[m], A=A[m, s], peak=fr[ip],
                  half=A[ip, s] / 2, wl=wl, wr=wr, hw=hw, g_on=g_on[s])
    return ws, g_on, g_sp, resolved, ex


def main():
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ok = []
    for system in ["cnt33", "d5a"]:
        try:
            ok.append((system, phono3py_ref(system), negf_linewidths(system)))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {system}: {type(e).__name__}: {str(e)[:80]}")
    fig, axes = plt.subplots(1, len(ok), figsize=(5.0 * len(ok), 3.8), squeeze=False)
    axes = axes[0]
    for ax, (system, (w_all, g_all, w_G, g_G), (ws, g_on, g_sp, res, ex)) in zip(axes, ok):
        m = (w_all > 0.2) & (g_all > 1e-5)
        ax.scatter(w_all[m], g_all[m], s=9, color=C["p3p"], alpha=0.55, label="phono3py (all $q$)")
        mg = w_G > 0.2
        ax.scatter(w_G[mg], g_G[mg], s=24, color=C["p3pG"], edgecolor="k",
                   linewidth=0.3, label=r"phono3py ($\Gamma$)", zorder=4)
        # NEGF: spectral-fit HWHM where resolved, on-shell fallback otherwise
        ax.scatter(ws[res], g_sp[res], s=34, marker="D", color=C["negf"],
                   edgecolor="k", linewidth=0.3, zorder=6, label="NEGF spectral-fit")
        unr = np.isfinite(g_on) & ~res
        ax.scatter(ws[unr], g_on[unr], s=26, marker="d", facecolor="none",
                   edgecolor=C["negf"], linewidth=0.9, zorder=5,
                   label=r"NEGF on-shell (unresolved, $\Gamma\!<\!3\,d\omega$)")
        ax.set_yscale("log")
        ax.set_xlabel(r"mode frequency $\omega$ [THz]")
        ax.set_ylabel(r"linewidth $\Gamma$ [THz, HWHM]")
        ax.set_title(SYS[system]["label"], fontsize=10)
        ax.legend(loc="lower right", fontsize=6.6)

        if ex is not None:  # inset: example A_s(omega) with HWHM marked
            ai = inset_axes(ax, width="38%", height="34%", loc="upper left", borderpad=1.1)
            ai.plot(ex["fr"], ex["A"], color=C["negf"], lw=1.3)
            ai.axhline(ex["half"], ls=":", color="0.5", lw=0.8)
            ai.axvspan(ex["wl"], ex["wr"], color=C["negf"], alpha=0.12)
            ai.set_title(rf"$A_s(\omega)$, $\omega_s{{=}}{ex['ws']:.1f}$  HWHM={ex['hw']:.2f}",
                         fontsize=6)
            ai.tick_params(labelsize=5); ai.set_xlabel(r"$\omega$ [THz]", fontsize=6)
    fig.tight_layout()
    out = HERE / "fig" / "linewidth_vs_phono3py.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
