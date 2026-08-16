"""Grid RESOLUTION: does |lambda| really track Delta_omega/Gamma?

Run:  QTX_ARRAY_MODULE=numpy OMP_NUM_THREADS=4         python phonon/studies/_grid_stability_law.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "phonon"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import phonon.studies._toy_grid_study as T  # noqa: E402
from phonon.studies._toy_grid_e7 import (  # noqa: E402
    arnoldi_spectrum, make_F)

OUT = ROOT / "phonon/studies/out/grid_audit"

OMEGA_A, OMEGA_FLAT = 8.0, 10.0
FMAX = 2.2 * 2 * OMEGA_A


def part_a():
    """#2: discrete Lorentzian weight vs the closed form, across a cell."""
    print("[#2] discrete pole weight W(d) vs (dw/pi) G/(d^2+G^2)")
    print(f"  {'dw/Gamma':>9} {'max ratio':>11} {'min ratio':>11} "
          f"{'swing':>9} {'pred swing':>11}")
    rows = []
    for ratio in (0.25, 1.0, 4.0, 16.0, 64.0):
        gam = 0.2
        dw = ratio * gam
        # exact Lorentzian A(w) = (1/pi) G/((w-w0)^2+G^2), unit weight
        offs = np.linspace(0.0, 1.0, 41)[:-1]          # sub-cell offsets
        got, pred = [], []
        for fr in offs:
            w0 = 50.0 + fr * dw
            grid = np.arange(0, 100.0, dw)
            a = (1.0 / np.pi) * gam / ((grid - w0) ** 2 + gam ** 2)
            got.append(float(a.max() * dw))            # peak-bin weight
            d = np.min(np.abs(grid - w0))
            pred.append(float((dw / np.pi) * gam / (d ** 2 + gam ** 2)))
        got, pred = np.array(got), np.array(pred)
        rel = got / pred
        rows.append(dict(dw_over_gamma=ratio, max_ratio=float(rel.max()),
                         min_ratio=float(rel.min()),
                         swing=float(got.max() / got.min())))
        # the claim's predicted swing over a cell: ~ (dw/G) at dw >> G
        pswing = max(1.0, (dw / gam) ** 2 / 4.0 + 1.0) if ratio > 1 else 1.0
        print(f"  {ratio:9.2f} {rel.max():11.4f} {rel.min():11.4f} "
              f"{got.max()/got.min():9.3f} {pswing:11.3f}")
    return rows


def part_b(gamma_targets=(0.02, 0.2), nfs=(30, 60, 120, 240, 480)):
    """#1/#3: rho(J) over a dw/Gamma ladder at the fixed point."""
    print("\n[#1/#3] rho(J) = |lambda|_max vs dw/Gamma at the fixed point")
    g0 = 1e19
    h00, h01, phi = T.flatband_chain(OMEGA_A, OMEGA_FLAT, g0)
    gam0 = T.first_born_gamma(h00, h01, phi, FMAX, omega_flat=OMEGA_FLAT)
    rows = []
    print(f"  {'Gamma':>7} {'nf':>5} {'dw':>8} {'dw/Gam':>8} {'conv':>6} "
          f"{'rho(J)':>9} {'pred dw/Gam':>12}")
    for t in gamma_targets:
        g = float(g0 * np.sqrt(t / gam0))
        h00c, h01c, phic = T.flatband_chain(OMEGA_A, OMEGA_FLAT, g)
        for nf in nfs:
            res = T.run_case(h00c, h01c, phic, nf, FMAX, max_iter=400,
                             tol=1e-7)
            dw = res["dw"]
            F, freqs, dw2, N_D, nfreq = make_F(h00c, h01c, phic, nf, FMAX)
            ev = arnoldi_spectrum(F, np.asarray(res["Sigma_l"])[0],
                                  np.asarray(res["Sigma_g"])[0],
                                  nfreq, N_D, m=16)
            rho = float(abs(ev[0]))
            rows.append(dict(gamma=t, nf=nf, dw=float(dw),
                             dw_over_gamma=float(dw / t),
                             converged=bool(res["converged"]), rho=rho))
            print(f"  {t:7.3f} {nf:5d} {dw:8.4f} {dw/t:8.2f} "
                  f"{str(res['converged']):>6} {rho:9.4f} {dw/t:12.2f}")
    return rows


def part_c(nf=60, gamma_target=0.2, n_off=9):
    """#3/#4: rho(J) at FIXED dw as the pole crosses one cell."""
    print(f"\n[#3/#4] rho(J) vs sub-cell pole alignment (nf={nf}, "
          f"Gamma={gamma_target})")
    g0 = 1e19
    h00, h01, phi = T.flatband_chain(OMEGA_A, OMEGA_FLAT, g0)
    gam0 = T.first_born_gamma(h00, h01, phi, FMAX, omega_flat=OMEGA_FLAT)
    g = float(g0 * np.sqrt(gamma_target / gam0))
    dw = FMAX / nf
    rows = []
    print(f"  dw = {dw:.4f}, dw/Gamma = {dw/gamma_target:.2f}")
    print(f"  {'offset':>8} {'omega_flat':>11} {'conv':>6} {'n_it':>5} "
          f"{'rho(J)':>9} {'Gamma_em':>9}")
    for fr in np.linspace(0.0, 1.0, n_off):
        wf = OMEGA_FLAT + fr * dw
        h00c, h01c, phic = T.flatband_chain(OMEGA_A, wf, g)
        res = T.run_case(h00c, h01c, phic, nf, FMAX, max_iter=400, tol=1e-7)
        F, freqs, dw2, N_D, nfreq = make_F(h00c, h01c, phic, nf, FMAX)
        ev = arnoldi_spectrum(F, np.asarray(res["Sigma_l"])[0],
                              np.asarray(res["Sigma_g"])[0], nfreq, N_D,
                              m=16)
        rho = float(abs(ev[0]))
        gem = float(T.emergent_gamma(res, wf, 1, 2))
        rows.append(dict(offset=float(fr), omega_flat=float(wf),
                         converged=bool(res["converged"]),
                         n_it=int(len(res["convergence_history"])),
                         rho=rho, gamma_em=gem))
        print(f"  {fr:8.3f} {wf:11.4f} {str(res['converged']):>6} "
              f"{len(res['convergence_history']):5d} {rho:9.4f} "
              f"{gem:9.5f}")
    if rows:
        rr = np.array([r["rho"] for r in rows])
        print(f"  rho swing across one cell: {rr.min():.4f} -> "
              f"{rr.max():.4f}  (ratio {rr.max()/max(rr.min(),1e-30):.3f}); "
              f"claim #3 predicts ~dw/Gamma = {dw/gamma_target:.2f}")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"omega_a": OMEGA_A, "omega_flat": OMEGA_FLAT, "fmax": FMAX}
    rep["pole_weight"] = part_a()
    rep["rho_ladder"] = part_b()
    rep["alignment"] = part_c()
    (OUT / "stability_law.json").write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT / 'stability_law.json'}")


if __name__ == "__main__":
    main()
