"""CNT (3,3) length series in thermal conductance quanta
(fig:res_cnt_length; replaces the generator-less campaign_length.pdf).

Data:
  Data. Absolute anchor: the committed eta=0 production set
  phonon/scripts/out/prod/cnt33_eta0/summary.json (L2/L3 converged
  pairs; conversion G_wire = G_per_area * A_c). The L2-L7 anharmonic
  series is the exact-kernel (sse_g_band=2) campaign of
  fig:res_gband_series -- its run.npz files (phonon/studies/out/
  anderson_test/cnt33_L*_linear/run_gband2.npz, tortin campaign) are
  not committed, so the ratio series enters as literals here:
  r = 0.569, 0.500, 0.453, 0.417, 0.396, 0.362 at L = 2..7. The
  absolute series is r * G_ball with the (L-independent) ballistic
  anchor. Note the two independent L2 measurements: the production set
  gives r(L2) = 0.574 (181-pt grid), the campaign 0.569 (its own grid)
  -- a 0.9% two-run spread, both converged; the figure uses each
  series' own value.

Run:  python phonon/scripts/figures/cnt33_gq_length.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.studies import style

SUMMARY = ROOT / "phonon/scripts/out/prod/cnt33_eta0/summary.json"
FIGDIR = ROOT / "document/fig/transport_sweeps"

K_B = 1.380649e-23
H = 6.62607015e-34

# exact-kernel (g_band=2) campaign ratio series, L2..L7
L_SERIES = np.arange(2, 8)
R_SERIES = np.array([0.569, 0.500, 0.453, 0.417, 0.396, 0.362])
CELL_NM = 0.246  # CNT(3,3) transport cell length


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    gq = np.pi**2 * K_B**2 * 300.0 / (3 * H)

    rows = {r["tag"]: r for r in json.loads(SUMMARY.read_text())
            if r["tag"].startswith("L")}
    gball = rows["L2"]["G_ball_W_per_m2_K"] * rows["L2"]["A_c"]
    gball_gq = gball / gq
    prod = {int(t[1:]): (r["G_anh_W_per_m2_K"] * r["A_c"] / gq)
            for t, r in rows.items()
            if r.get("anh_converged") and r.get("G_anh_W_per_m2_K")}

    ganh_gq = R_SERIES * gball_gq

    fig, (ax_a, ax_b) = style.figure(ncols=2, width=4.3, height=3.3)
    colors = style.RC["axes.prop_cycle"].by_key()["color"]

    ax_a.axhline(gball_gq, color=colors[0], lw=1.4)
    ax_a.annotate(r"ballistic", (6.4, gball_gq),
                  textcoords="offset points", xytext=(0, -12), fontsize=8.5,
                  color=colors[0])
    ax_a.plot(L_SERIES, ganh_gq, "o-", color=colors[1],
              label="anharmonic (exact kernel)")
    ax_a.plot(sorted(prod), [prod[L] for L in sorted(prod)], "s",
              color=colors[3], markersize=6, label="production set")
    ax_a.set_xlabel("device length (transport cells)")
    ax_a.set_ylabel(r"$G / g_Q(300\,\mathrm{K})$")
    ax_a.set_ylim(0, 6.2)
    ax_a.legend(fontsize=8)

    gl = ganh_gq * L_SERIES * CELL_NM
    ax_b.plot(L_SERIES, gl, "o-", color=colors[1])
    ax_b.set_xlabel("device length (transport cells)")
    ax_b.set_ylabel(r"$G\cdot L$ ($g_Q\,$nm)")
    ax2 = ax_b.twinx()
    ax2.plot(L_SERIES, R_SERIES, "^--", color=colors[2])
    ax2.set_ylabel(r"$r = G_\mathrm{anh}/G_\mathrm{ball}$",
                   color=colors[2])
    ax2.grid(False)

    style.save(fig, "cnt33_gq_length", directory=FIGDIR)

    print(f"g_Q(300 K) = {gq:.4e} W/K")
    print(f"ballistic: {gball:.3e} W/K = {gball_gq:.3f} g_Q "
          "(L-independent across the production pairs)")
    print("anharmonic series (g_Q):",
          ", ".join(f"L{L} {g:.2f}" for L, g in zip(L_SERIES, ganh_gq)))
    print("production-set direct points (g_Q):",
          {f"L{L}": round(v, 3) for L, v in sorted(prod.items())})
    print("G*L (g_Q nm):",
          ", ".join(f"{v:.2f}" for v in gl),
          "-- sublinear growth, ballistic-to-diffusive onset")


if __name__ == "__main__":
    main()
