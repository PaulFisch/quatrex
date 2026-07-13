"""Theory-chapter observables for every run of the anderson_test campaign.

Generalizes the single-run CNT L2 analysis to all systems/lengths/schemes:
for each registered run it computes what the snapshot allows —

  full runs (gr/gl diagonals + spectra):
    LDOS/DOS + sum rule, occupation n_i(w), T_eff per slab, per-atom MSD
    (+ isolated-device mode-sum reference), ballistic Caroli + Fisher-Lee
    (from the run's dynamical_matrix.mat), T_eff(w), spectral heat current
    + IR plateau, G_anh / G_ball / r + G_ball(T), conservation ledger
    (D(w), slab P_abs, telescoped spread);
  spectra-only runs (campaign schemes, QX_SAVE_DIAG_G=0):
    the transmission/heat/conductance/conservation subset.

Writes campaign_report/<run>_obs.npz + a combined summary.json.

Usage:
    python phonon/studies/_campaign_observables.py [--only run1,run2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.solver import leads, observables as obs  # noqa: E402
from phonon.solver.grids import bose_full_axis  # noqa: E402
from phonon.solver.static_se import (  # noqa: E402
    UU_PREFACTOR, equilibrium_uu_modesum)

BASE = ROOT / "phonon/studies/out/anderson_test"
OUT = BASE / "campaign_report"

H_SI = 6.62607015e-34
KB_SI = 1.380649e-23

#: name -> (npz, inputs dir with dynamical_matrix.mat, label, caveat)
RUNS: dict[str, dict] = {
    "cnt_L2": dict(npz="cnt33_linear/run.npz", inputs="inputs_L2",
                   label="CNT(3,3) L2 linear 0.2", caveat=""),
    "cnt_L3": dict(npz="cnt33_L3_linear/run.npz", inputs="inputs_L2",
                   label="CNT(3,3) L3 linear 0.2", caveat=""),
    "cnt_L4": dict(npz="cnt33_L4_linear/run.npz", inputs="inputs_L2",
                   label="CNT(3,3) L4 linear 0.2",
                   caveat="UNCONVERGED (best-conserved iterate, cons 1.5e-7)"),
    "d5a_lin": dict(npz="d5a_linear/run.npz", inputs="d5a_linear",
                    label="d5a SiNW linear 0.1 (+IR floor)",
                    caveat="residual-converged; lead balance floor ~2-4% "
                           "from eta_ir_floor"),
    "d5a_and": dict(npz="d5a_anderson/run.npz", inputs="d5a_linear",
                    label="d5a SiNW Anderson d8 (+IR floor)",
                    caveat="stagnated at 8e-3 residual"),
    # spectra-only scheme runs used for fixed-point comparisons
    "d5a_guard": dict(npz="mixer_campaign_d5a_v2/and_d8_r1e4_guard/run.npz",
                      inputs=None, label="d5a Anderson+ridge+guards",
                      caveat="converged (49 it)"),
    "d5a_rre8": dict(npz="mixer_campaign_d5a_v2/rre_c8/run.npz",
                     inputs=None, label="d5a RRE c8",
                     caveat="residual-converged"),
}
# every converged L2 scheme run gets a J-comparison entry (spectra only)
for _tag, _p in [
    ("L2_lin02", "mixer_campaign_L2/lin02/run.npz"),
    ("L2_irfloor", "mixer_campaign_L2/lin02_irfloor/run.npz"),
    ("L2_rre8", "mixer_campaign_L2/rre_c8/run.npz"),
    ("L2_rre12", "mixer_campaign_L2/rre_c12/run.npz"),
    ("L2_broyden", "mixer_campaign_L2/broyden/run.npz"),
    ("L2_rre10", "rre_sweep_L2/rre_c10/run.npz"),
    ("L2_rre16", "rre_sweep_L2/rre_c16/run.npz"),
    ("L2_rre20", "rre_sweep_L2/rre_c20/run.npz"),
    ("L2_rre12b03", "rre_sweep_L2/rre_c12_b0.3/run.npz"),
    ("L2_rre16b03", "rre_sweep_L2/rre_c16_b0.3/run.npz"),
    ("L2_rre12r8", "rre_sweep_L2/rre_c12_r1e-8/run.npz"),
    ("L2_rre12r3", "rre_sweep_L2/rre_c12_r1e-3/run.npz"),
]:
    RUNS[_tag] = dict(npz=_p, inputs=None, label=_tag, caveat="")

_ball_cache: dict = {}


def ballistic(matfile: Path, n_slabs: int, f: np.ndarray):
    """Caroli transmission, mode-resolved channels, ballistic LDOS."""
    key = (str(matfile), n_slabs)
    if key in _ball_cache:
        return _ball_cache[key]
    mat = sio.loadmat(matfile)
    H00 = np.asarray(mat["[0, 0, 0]"], complex)
    H01 = np.asarray(mat["[0, 0, 1]"], complex)
    b = H00.shape[0]
    N_D = b * n_slabs
    H_D = leads.build_device_hamiltonian(H00, H01, n_slabs)
    ne = f.size
    z2 = (f[1:] + 1j * 1e-8) ** 2
    gL, okL = leads.sancho_rubio_batch(z2, H00, H01)
    gR, okR = leads.sancho_rubio_batch(z2, H00, H01.conj().T)
    for iw in np.where(~(okL & okR))[0]:
        gL[iw] = leads.sancho_rubio(z2[iw], H00, H01)
        gR[iw] = leads.sancho_rubio(z2[iw], H00, H01.conj().T)
    H10 = H01.conj().T
    SL = np.einsum("ij,wjk,kl->wil", H10, gL, H01)
    SR = np.einsum("ij,wjk,kl->wil", H01, gR, H10)
    GamL = 1j * (SL - SL.conj().swapaxes(-1, -2))
    GamR = 1j * (SR - SR.conj().swapaxes(-1, -2))
    sl0, slN = slice(0, b), slice(N_D - b, N_D)
    eye = np.eye(N_D)
    T_ball = np.zeros(ne)
    ldos_ball = np.zeros((ne, N_D))
    # display-broadened LDOS pass (eta = dw/2)
    df = float(f[1] - f[0])
    z2d = (f[1:] + 1j * 0.5 * df) ** 2
    gLd, _ = leads.sancho_rubio_batch(z2d, H00, H01)
    gRd, _ = leads.sancho_rubio_batch(z2d, H00, H01.conj().T)
    SLd = np.einsum("ij,wjk,kl->wil", H10, gLd, H01)
    SRd = np.einsum("ij,wjk,kl->wil", H01, gRd, H10)
    for k in range(ne - 1):
        Aw = z2[k] * eye - H_D
        Aw[sl0, sl0] -= SL[k]
        Aw[slN, slN] -= SR[k]
        GR_b = np.linalg.inv(Aw)
        G0N = GR_b[sl0, slN]
        T_ball[k + 1] = float(np.einsum(
            "ij,jk,kl,il->", GamL[k], G0N, GamR[k], G0N.conj()).real)
        Ad = z2d[k] * eye - H_D
        Ad[sl0, sl0] -= SLd[k]
        Ad[slN, slN] -= SRd[k]
        ldos_ball[k + 1] = (2.0 * f[k + 1] / np.pi) * (
            -np.diag(np.linalg.inv(Ad)).imag)
    _ball_cache[key] = (T_ball, ldos_ball, H_D)
    return _ball_cache[key]


def analyze(name: str, spec_only: bool = False) -> dict:
    meta = RUNS[name]
    d = np.load(BASE / meta["npz"])
    f = np.asarray(d["energies"], float)
    df = float(f[1] - f[0])
    ne = f.size
    n_slabs = int(d["nblocks"])
    T_L, T_R = float(d["t_left"]), float(d["t_right"])
    T0, dT = 0.5 * (T_L + T_R), T_L - T_R
    spec = np.asarray(d["current_spectrum"], float)
    fh = np.asarray(d["final_heat"], float)
    C_W = H_SI * 1e24 * df
    n_L, n_R = obs.bose(f, T_L), obs.bose(f, T_R)
    dn = n_L - n_R

    out = dict(f=f, final_heat=fh, J_W=C_W * fh, t_left=T_L, t_right=T_R,
               n_iter=int(d["n_iter"]), converged=bool(d["converged"]))
    # effective transmission + heat spectra (always available)
    T_eff_w = np.zeros((ne, 2))
    for j, col in enumerate((0, spec.shape[1] - 1)):
        T_eff_w[1:, j] = spec[1:, col] / dn[1:]
    out["T_eff_w"] = T_eff_w
    HBAR = H_SI / (2 * np.pi)
    out["j_W_per_THz"] = HBAR * (2 * np.pi * f[:, None] * 1e12) * spec * 1e12
    out["G_anh"] = float(obs.thermal_conductance(T_eff_w[:, 0], f, T0))
    # conservation ledger
    if "slab_absorption" in d.files:
        pa = np.asarray(d["slab_absorption"])[0].real
        out["slab_pa"] = pa
        out["telescoped"] = obs.telescoped_spread(fh, pa)
        out["raw_spread"] = float(fh.max() - fh.min())
    if "bubble_balance_spectrum" in d.files:
        bb = np.asarray(d["bubble_balance_spectrum"])
        out["D_w"] = (f * spec[:, 0] - f * spec[:, -1]
                      - (bb[1].real - bb[0].real))
    out["lead_bal"] = float(abs(fh[0] - fh[-1]) / max(abs(fh[0]), 1e-300))
    if "iter_heat" in d.files:
        out["iter_heat"] = np.asarray(d["iter_heat"])
    if "iter_sigma_max" in d.files:
        out["iter_sigma_max"] = np.asarray(d["iter_sigma_max"])
    for k in ("iter_mixer_fnorm", "iter_mixer_cond", "iter_mixer_gnorm",
              "iter_mixer_kind"):
        if k in d.files:
            out[k] = np.asarray(d[k])

    if meta["inputs"] and not spec_only:
        T_ball, ldos_ball, H_D = ballistic(
            BASE / meta["inputs"] / "dynamical_matrix.mat", n_slabs, f)
        out["T_ball"] = T_ball
        out["ldos_ball"] = ldos_ball
        out["G_ball"] = float(obs.thermal_conductance(T_ball, f, T0))
        out["r"] = out["G_anh"] / out["G_ball"]
        Ts = np.linspace(50, 600, 56)
        out["Ts"] = Ts
        out["G_ball_T"] = np.array(
            [obs.thermal_conductance(T_ball, f, t) for t in Ts])
        out["plateau_W_per_THz"] = KB_SI * dT * T_ball[1] * 1e12

    if "gr_diag_imag" in d.files and not spec_only:
        gr = np.asarray(d["gr_diag_imag"], float)
        gl = np.asarray(d["gl_diag_imag"], float)
        n_dof = gr.shape[1] // n_slabs
        out["n_dof"] = n_dof
        out["ldos"] = (2.0 * f[:, None] / np.pi) * gr
        out["sum_rule"] = out["ldos"].sum(axis=0) * df
        A = 2.0 * gr
        iGl_s = gl.reshape(ne, n_slabs, n_dof).sum(axis=-1)
        A_s = A.reshape(ne, n_slabs, n_dof).sum(axis=-1)
        out["occ_slab"] = iGl_s / np.where(np.abs(A_s) > 1e-30, A_s, 1e-30)
        out["n_bose_mid"] = obs.bose(f, T0)
        out["T_eff_slab"] = obs.effective_temperature(f, iGl_s, A_s)
        uu = UU_PREFACTOR * (df / (2 * np.pi)) * (
            2 * gl[1:] + 2 * gr[1:]).sum(axis=0)
        out["msd_atom"] = uu.reshape(-1, 3).sum(axis=1)
        if meta["inputs"]:
            _, _, H_D = ballistic(
                BASE / meta["inputs"] / "dynamical_matrix.mat", n_slabs, f)
            out["msd_eq_atom"] = np.diag(equilibrium_uu_modesum(
                H_D, T0)).real.reshape(-1, 3).sum(axis=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    names = [n.strip() for n in args.only.split(",") if n.strip()] \
        or list(RUNS)
    summary = {}
    for name in names:
        try:
            res = analyze(name)
        except FileNotFoundError as exc:
            print(f"{name}: SKIP ({exc})")
            continue
        np.savez(OUT / f"{name}_obs.npz", **{
            k: v for k, v in res.items() if isinstance(v, np.ndarray)})
        summary[name] = {
            "label": RUNS[name]["label"], "caveat": RUNS[name]["caveat"],
            **{k: (float(v) if isinstance(v, (int, float, np.floating))
                   else v) for k, v in res.items()
               if isinstance(v, (int, float, bool, np.floating))},
            "J_L_int": float(res["final_heat"][0]),
            "J_L_nW": float(res["J_W"][0] * 1e9),
        }
        line = (f"{name:12s} J_L={res['final_heat'][0]:.6f} "
                f"({res['J_W'][0]*1e9:.4f} nW) G_anh="
                f"{res['G_anh']*1e9:.4f} nW/K leadbal={res['lead_bal']:.2e}")
        if "r" in res:
            line += f" G_ball={res['G_ball']*1e9:.4f} r={res['r']:.4f}"
        if "telescoped" in res:
            line += (f" tel={res['telescoped']:.2e}"
                     f"/{res['raw_spread']:.2e}")
        print(line)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"saved {OUT}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
