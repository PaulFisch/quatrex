"""Bulk-Si phonon dispersion: DFPT (QE ph.x) vs finite-displacement FC2.

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/fc_method_dispersion_si.py
Figure -> document/fig/transport_sweeps/fc_method_dispersion_si.{pdf,png}
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style  # noqa: E402

import phonopy  # noqa: E402
from phonopy.structure.atoms import PhonopyAtoms  # noqa: E402
from phonopy.phonon.band_structure import (  # noqa: E402
    get_band_qpoints_and_path_connections,
)

FC2 = ROOT / "phonon/reaps/si_primitive_work/fc2.hdf5"
PH_OUT = ROOT / "phonon/configs/si_primitive/dfpt/ph.out"
FIGDIR = ROOT / "document/fig/transport_sweeps"

# ph.out q's are cartesian in 2pi/alat with alat = a/sqrt(2) (QE ibrav=0 FCC
# primitive cell): Gamma=(0,0,0), L=(+-0.354)^3, X=(0.707,0,0)-type.
Q_GAMMA = (0.0, 0.0, 0.0)


def parse_dfpt(path: Path) -> dict[str, np.ndarray]:
    """{point-name: sorted DFPT freqs (THz)} parsed from ph.out."""
    dfpt: dict[tuple, list[float]] = {}
    qcur = None
    for ln in path.read_text().splitlines():
        m = re.search(r"q = \(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)", ln)
        if m:
            qcur = tuple(round(float(x), 3) for x in m.groups())
            dfpt.setdefault(qcur, [])
        m2 = re.search(r"freq \(.*\) =\s*([-\d.]+)\s*\[THz\]", ln)
        if m2 and qcur is not None:
            dfpt[qcur].append(float(m2.group(1)))
    out: dict[str, np.ndarray] = {}
    for q, fr in dfpt.items():
        if not fr:
            continue
        mags = sorted(round(abs(x), 3) for x in q)
        if q == Q_GAMMA:
            out["G"] = np.sort(fr)
        elif len({abs(x) for x in q}) == 1 and abs(q[0]) > 0.1:   # (t,t,t) -> L
            out["L"] = np.sort(fr)
        elif mags[0] == mags[1] == 0.0 and mags[2] > 0.5:         # (t,0,0) -> X
            out["X"] = np.sort(fr)
    return out


def fd_band_structure():
    """FD phonopy band structure on the Gamma-X-K-Gamma-L path + point freqs."""
    prim = PhonopyAtoms(
        symbols=["Si", "Si"],
        cell=[[0.0, 2.7331, 2.7331], [2.7331, 0.0, 2.7331],
              [2.7331, 2.7331, 0.0]],
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    with h5py.File(FC2, "r") as f:
        fc2 = np.array(f["force_constants"])
    ph = phonopy.Phonopy(prim, supercell_matrix=np.diag([2, 2, 2]),
                         primitive_matrix="auto")
    ph.force_constants = fc2
    band_path = [[[0, 0, 0], [0.5, 0, 0.5], [0.375, 0.375, 0.75]],
                 [[0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
    qpts, conn = get_band_qpoints_and_path_connections(band_path, npoints=61)
    ph.run_band_structure(qpts, path_connections=conn,
                          labels=["$\\Gamma$", "X", "K", "$\\Gamma$", "L"])
    bs = ph.get_band_structure_dict()
    ph.run_qpoints([[0, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0.5]])
    fd_pts = {k: np.sort(v) for k, v in
              zip("GXL", ph.get_qpoints_dict()["frequencies"])}
    return bs, fd_pts


def main():
    dfpt = parse_dfpt(PH_OUT)
    bs, fd_pts = fd_band_structure()

    # high-symmetry x positions on the path: G, X, K, G, L
    d = bs["distances"]
    xG1, xX, xK, xG2, xL = d[0][0], d[0][-1], d[1][-1], d[2][-1], d[3][-1]

    # ---- table + derived numbers ----
    g_dfpt_opt = dfpt["G"][dfpt["G"] > 1.0]
    g_fd_opt = fd_pts["G"][fd_pts["G"] > 1.0]
    go_d, go_f = g_dfpt_opt.mean(), g_fd_opt.mean()
    print("=" * 64)
    print("BULK Si PHONONS (THz): DFPT (ph.x) vs FD (phono3py+symfc)")
    print("=" * 64)
    for name in ("G", "X", "L"):
        print(f"{name}:  DFPT {np.round(dfpt[name], 2)}")
        print(f"    FD   {np.round(fd_pts[name], 2)}")
    print(f"Gamma optical: DFPT {go_d:.2f}  FD {go_f:.2f}  "
          f"diff {100 * abs(go_d - go_f) / go_f:.1f}%")
    print(f"DFPT Gamma acoustic (ASR artifact, dropped from plot): "
          f"{dfpt['G'][dfpt['G'] < 1.0].mean():.2f} THz")

    # ---- figure: FD lines + DFPT markers, inset on the Gamma opticals ----
    fig, ax = style.doc_figure(frac=0.6, aspect=0.73)
    for i, (seg_d, seg_f) in enumerate(zip(bs["distances"], bs["frequencies"])):
        for b in range(seg_f.shape[1]):
            ax.plot(seg_d, seg_f[:, b], color="C0", lw=1.2,
                    label="FD (phono3py+symfc)" if (i == 0 and b == 0) else None)
    first = True
    for x, name in ((xG1, "G"), (xX, "X"), (xG2, "G"), (xL, "L")):
        fr = dfpt[name]
        fr = fr[fr > 0.0] if name == "G" else fr   # drop ASR-artifact zeros
        ax.plot([x] * len(fr), fr, "o", color="C3", ms=5, mfc="none", mew=1.4,
                label="DFPT (ph.x)" if first else None, zorder=5)
        first = False
    for x in (xX, xK, xG2):
        ax.axvline(x, color="0.8", lw=0.7, zorder=0)
    ax.set_xticks([xG1, xX, xK, xG2, xL])
    ax.set_xticklabels(["$\\Gamma$", "X", "K", "$\\Gamma$", "L"])
    ax.set_ylabel("frequency (THz)")
    ax.set_xlim(xG1, xL)
    ax.set_ylim(0, 17.4)
    ax.legend(loc="upper center", ncols=2, fontsize=7.5)
    ax.annotate("DFPT $\\Gamma$-acoustic modes at $-0.18$ THz\n"
                "(ph.x ASR artifact) omitted",
                xy=(0.02, 0.025), xycoords="axes fraction", fontsize=6.5,
                color="0.35")

    # inset: Gamma optical cluster (15.12 DFPT vs 15.37 FD, 1.6%)
    axin = ax.inset_axes([0.40, 0.10, 0.26, 0.34])
    span = 0.10 * (xL - xG1)
    for seg_d, seg_f in zip(bs["distances"], bs["frequencies"]):
        for b in range(seg_f.shape[1]):
            axin.plot(seg_d, seg_f[:, b], color="C0", lw=1.2)
    axin.plot([xG2] * len(g_dfpt_opt), g_dfpt_opt, "o", color="C3", ms=5,
              mfc="none", mew=1.4, zorder=5)
    axin.set_xlim(xG2 - span, xG2 + span)
    axin.set_ylim(14.6, 15.8)
    axin.set_xticks([xG2])
    axin.set_xticklabels(["$\\Gamma$"], fontsize=7)
    axin.tick_params(labelsize=6.5)
    axin.grid(alpha=0.3, lw=0.5)
    axin.annotate(f"FD {go_f:.2f}", (xG2 + 0.25 * span, go_f + 0.06),
                  fontsize=6.5, color="C0")
    axin.annotate(f"DFPT {go_d:.2f}", (xG2 + 0.25 * span, go_d - 0.16),
                  fontsize=6.5, color="C3")
    axin.set_title(f"$\\Gamma$ optical: {100 * abs(go_d - go_f) / go_f:.1f}%",
                   fontsize=7)
    _, connectors = ax.indicate_inset_zoom(axin, edgecolor="0.5", lw=0.7)
    for c in connectors:   # keep the source rectangle, drop the long connector lines
        c.set_visible(False)

    style.save(fig, "fc_method_dispersion_si", directory=FIGDIR)


if __name__ == "__main__":
    main()
