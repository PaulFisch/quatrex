"""Static-correction magnitude study figure (results/50_static_scp).

Data:
  One merged 2x2 figure (static_se_study) replacing the old
  static_se_study + static_se_tadpole_breakdown pair, from the regenerated
  snapshots (phonon/scripts/out/snapshots/study_*.npz, produced by
  phonon/scripts/verify/_static_se_sweep.sh; the original snapshots were purged).

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/static_se_figs.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

SRC = ROOT / "phonon/scripts/out/snapshots"
COMPACT = ROOT / "phonon/scripts/data/static_se_summary.npz"
FIGDIR = ROOT / "document/fig/transport_sweeps"
MODES = ["bubble", "loop", "tadpole", "loop_tadpole"]
MCOL = {"bubble": "C0", "loop": "C2", "tadpole": "C3", "loop_tadpole": "C4"}
TITLES = {"cnt33": "(3,3) CNT", "d5a": "d5a SiNW"}
SCALARS = ("sigma_static_norm", "conservation", "converged", "Ga_over_Gb",
           "soft_bare", "soft_ren", "reB", "imB", "resid")


def load():
    """Prefer the full snapshots; reduce them to the committed compact
    summary (the full set is ~430 MB and stays local). Fall back to the
    compact file so the figure regenerates from the repo alone."""
    d = {}
    for f in glob.glob(str(SRC / "study_*.npz")):
        z = np.load(f, allow_pickle=True)
        rec = {k: float(z[k]) if k != "converged" else bool(z[k])
               for k in SCALARS if k in z.files}
        rec["sigma_static"] = np.asarray(z["sigma_static"])
        d[(str(z["struct"]), float(z["temp"]), str(z["mode"]))] = rec
    if d:
        # coupled-tadpole norm needs the matrices; store it, drop the matrices
        payload = {}
        for (s, T, m), rec in d.items():
            zl = d.get((s, T, "loop"))
            if m == "loop_tadpole" and zl is not None:
                rec["coupled_tadpole_norm"] = float(np.linalg.norm(
                    rec["sigma_static"] - zl["sigma_static"]))
        for (s, T, m), rec in d.items():
            key = f"{s}|{T:g}|{m}"
            for k, v in rec.items():
                if k != "sigma_static":
                    payload[f"{key}|{k}"] = v
        COMPACT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(COMPACT, **payload)
        print(f"reduced {len(d)} snapshots -> {COMPACT}")
        for rec in d.values():
            rec.pop("sigma_static", None)
        return d
    if COMPACT.exists():
        z = np.load(COMPACT, allow_pickle=True)
        for name in z.files:
            s, T, m, k = name.split("|")
            rec = d.setdefault((s, float(T), m), {})
            rec[k] = bool(z[name]) if k == "converged" else float(z[name])
        print(f"loaded compact summary {COMPACT} ({len(d)} points)")
        return d
    raise SystemExit("no snapshots and no compact summary found")


def main():
    data = load()
    structs = [s for s in ("cnt33", "d5a") if any(k[0] == s for k in data)]
    fig, axes = style.figure(ncols=len(structs), nrows=2, width=4.0, height=3.0)
    axes = np.atleast_2d(axes)

    for c, s in enumerate(structs):
        temps = sorted({k[1] for k in data if k[0] == s})
        a = axes[0][c]
        sigT_iso, sigT_cpl, sigL, reB, w2lo = [], [], [], [], []
        for T in temps:
            zt = data.get((s, T, "tadpole"))
            zl = data.get((s, T, "loop"))
            zlt = data.get((s, T, "loop_tadpole"))
            zb = data.get((s, T, "bubble"))
            sigT_iso.append(zt["sigma_static_norm"] if zt else np.nan)
            sigL.append(zl["sigma_static_norm"] if zl else np.nan)
            sigT_cpl.append(zlt.get("coupled_tadpole_norm", np.nan)
                            if zlt else np.nan)
            reB.append(zb["reB"] if zb else np.nan)
            sb = zt["soft_bare"] if zt else np.nan
            w2lo.append(sb * sb)
        a.semilogy(temps, sigT_iso, "s-", color="C3",
                   label=r"$\|\Sigma_T\|$ (isolated)")
        a.semilogy(temps, sigT_cpl, "s--", color="C3", mfc="none",
                   label=r"$\|\Sigma_T\|$ (coupled)")
        a.semilogy(temps, sigL, "o-", color="C2", label=r"$\|\Sigma_{\rm loop}\|$")
        a.semilogy(temps, reB, "^-", color="C0", label=r"$\max|\mathrm{Re}\,\Sigma_B|$")
        a.semilogy(temps, w2lo, "k--", lw=0.9, label=r"$\omega^2_{\rm low}$ (validity)")
        for T, y in zip(temps, sigT_iso):
            zt = data.get((s, T, "tadpole"))
            if zt is not None and (float(zt["soft_ren"]) < 0 or not bool(zt["converged"])):
                a.scatter([T], [y], s=140, facecolors="none",
                          edgecolors="C3", linewidths=1.6, zorder=5)
        a.set_title(TITLES.get(s, s), fontsize=9)
        a.set_xlabel("T (K)")
        a.set_ylabel(r"magnitude (THz$^2$)")
        if c == 0:
            a.legend(fontsize=6, loc="lower right")

        a = axes[1][c]
        for m in MODES:
            cons, conv = [], []
            for T in temps:
                z = data.get((s, T, m))
                cons.append(float(z["conservation"]) if z is not None else np.nan)
                conv.append(bool(z["converged"]) if z is not None else False)
            a.semilogy(temps, cons, "o-", color=MCOL[m], label=m, ms=4)
            for T, y, cv in zip(temps, cons, conv):
                if not cv and np.isfinite(y):
                    a.scatter([T], [y], marker="x", color=MCOL[m], s=60, zorder=5)
        a.set_xlabel("T (K)")
        a.set_ylabel("heat-flow conservation")
        if c == 0:
            a.legend(fontsize=6, loc="upper left")

        # numbers for the text
        for T in temps:
            row = {m: data.get((s, T, m)) for m in MODES}
            msg = " ".join(
                f"{m}:R={float(z['Ga_over_Gb']):.3f}/c={float(z['conservation']):.2e}"
                f"{'' if bool(z['converged']) else '(x)'}"
                for m, z in row.items() if z is not None)
            print(f"[{s} T={T:.0f}] {msg}")
        i300 = temps.index(300.0) if 300.0 in temps else None
        if i300 is not None:
            print(f"[{s} 300K] |ReB|:{reB[i300]:.0f} loop:{sigL[i300]:.0f} "
                  f"T_iso:{sigT_iso[i300]:.0f} T_cpl:{sigT_cpl[i300]:.0f}")

    style.save(fig, "static_se_study", directory=FIGDIR)


if __name__ == "__main__":
    main()
