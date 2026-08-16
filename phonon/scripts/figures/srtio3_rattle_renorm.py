"""SrTiO3 rattle-amplitude renormalisation of the FC2 dispersion
(fig:res_srtio3).

Run:  python phonon/scripts/figures/srtio3_rattle_renorm.py
Figure -> document/fig/transport_sweeps/srtio3_rattle_renorm.{png,pdf}
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
warnings.filterwarnings("ignore")
from phonon.finite_analysis.loader import load_system  # noqa: E402
from phonon.studies import style  # noqa: E402
from postproc.spectral import (  # noqa: E402
    dynamical_matrix_qpath,
    frequencies_from_dynamical,
)

FIGDIR = ROOT / "document/fig/transport_sweeps"
THZ_TO_CM1 = 33.35641  # 1 THz in cm^-1

FITS = [
    # (label, yaml, color)  -- colors per the report caption (small=red, large=blue)
    (r"small rattle (0.025 $\mathrm{\AA}$)",
     "phonon/configs/perovskite/srtio3_small_vasp.yaml", "#d55e00"),
    (r"large rattle (0.08 $\mathrm{\AA}$)",
     "phonon/configs/perovskite/srtio3_vasp.yaml", "#0173b2"),
]

# cubic Pm-3m high-symmetry path
NODES = [("$\\Gamma$", (0.0, 0.0, 0.0)), ("X", (0.0, 0.5, 0.0)),
         ("M", (0.5, 0.5, 0.0)), ("R", (0.5, 0.5, 0.5)),
         ("$\\Gamma$", (0.0, 0.0, 0.0))]
PTS_PER_HALF = 60  # points per |dq|=0.5 segment


def build_path():
    """Fractional q-points, cumulative distance, node positions/labels."""
    qs, dist = [], []
    ticks = [0.0]
    s = 0.0
    for (_, q0), (_, q1) in zip(NODES[:-1], NODES[1:]):
        q0, q1 = np.asarray(q0), np.asarray(q1)
        seg = float(np.linalg.norm(q1 - q0))
        n = max(2, int(round(PTS_PER_HALF * seg / 0.5)))
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        qs.append(q0 + t[:, None] * (q1 - q0))
        dist.append(s + t * seg)
        s += seg
        ticks.append(s)
    qs.append(np.asarray(NODES[-1][1])[None, :])
    dist.append(np.array([s]))
    return (np.concatenate(qs), np.concatenate(dist), ticks,
            [lab for lab, _ in NODES])


def main() -> None:
    qpath, dist, ticks, labels = build_path()
    i_R = int(np.argmin(np.abs(dist - ticks[3])))  # the R node sample

    fig, ax = style.doc_figure(frac=0.7, aspect=0.69)
    x_mid = 0.5 * (ticks[2] + ticks[3])  # middle of the flat M-R AFD branch
    afd = {}
    for label, yaml_rel, color in FITS:
        bundle = load_system(str(ROOT / yaml_rel), validate=False,
                             transport_axis=2)
        dyn = dynamical_matrix_qpath(bundle.phonon, qpath)
        bands = frequencies_from_dynamical(dyn)  # (nq, N), signed THz
        for n in range(bands.shape[1]):
            ax.plot(dist, bands[:, n], color=color, lw=1.0,
                    label=label if n == 0 else None)
        w_R = float(bands[i_R].min())
        afd[label] = w_R
        cm = abs(w_R) * THZ_TO_CM1
        if w_R < 0:  # AFD unstable: annotate below the flat M-R branch
            txt = f"AFD $\\omega_R = {abs(w_R):.2f}i$ THz (${cm:.0f}i$ cm$^{{-1}}$)"
            xy_txt = (x_mid, -3.4)
        else:  # AFD stabilised: annotate in the gap above the branch
            txt = (f"AFD $\\omega_R = +{w_R:.2f}$ THz\n"
                   f"($+{cm:.0f}$ cm$^{{-1}}$)")
            xy_txt = (x_mid, 2.5)
        ax.annotate(txt, (ticks[3], w_R), xytext=xy_txt, textcoords="data",
                    fontsize=7.5, color=color, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7,
                                    shrinkA=8, shrinkB=2))

    # high-symmetry separators + tick labels (review fix: no floating letters)
    for t in ticks[1:-1]:
        ax.axvline(t, color="0.75", lw=0.7)
    ax.axhline(0.0, color="0.55", lw=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(-4.4, 26.0)
    ax.set_ylabel(r"$\omega$ (THz)")
    ax.grid(axis="x", visible=False)
    # caption support: below zero = imaginary (unstable) modes
    ax.annotate("imaginary ($\\omega^2<0$)", (0.03, -1.4), fontsize=7.5,
                color="0.35", ha="left", va="top")
    ax.legend(fontsize=8, loc="upper left")
    style.save(fig, "srtio3_rattle_renorm", directory=FIGDIR)

    print("=" * 64)
    print("AFD octahedral-rotation mode at R = (1/2,1/2,1/2)")
    print("=" * 64)
    for label, w in afd.items():
        cm = abs(w) * THZ_TO_CM1
        kind = "imaginary/unstable" if w < 0 else "real/stable"
        print(f"{label:32s}: omega_R = {w:+.3f} THz = "
              f"{'-' if w < 0 else '+'}{cm:.1f}{'i' if w < 0 else ''} cm^-1 "
              f"({kind})")
    print("report claims: small fit 55i-92i cm^-1 (converging up; PBEsol ref "
          "76i), large fit +90 cm^-1")


if __name__ == "__main__":
    main()
