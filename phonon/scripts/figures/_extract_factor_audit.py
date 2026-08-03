"""Distil the vertex-factorisation audit into a committed dataset.

Sources (uncommitted run artifacts, pulled from tortin):
  cluster/mos2decomp3/run.log   MoS2 film INDSCAL fit ladder with the
                                norm-weighted aggregate gate and the
                                per-offset-class diagnostics (post
                                mass-weighted-ASR, post min-image)
  cluster/mos2decomp2/run.log   same ladder under the OLD
                                max-single-block gate (the r64
                                "phase-convention mismatch" false
                                alarm) -- only the rel_err series is
                                taken from here for low ranks
  cluster/sifilmdecomp/run.log  Si film ladder (homogeneous weights)
  cluster/sifilm_nk9r/run_{dense,r8,r32,r128}.npz
                                post-min-image conservation ladder
                                (3-iteration bubble-balance audit)

Writes phonon/scripts/data/factor_audit.npz.

Run:  python phonon/scripts/figures/_extract_factor_audit.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "phonon/scripts/data/factor_audit.npz"

RE_FIT = re.compile(r"INDSCAL R=(\d+): rel_err=([\d.]+)")
RE_AGG = re.compile(r"\[decompose r(\d+)\] fit rel_err=([\d.]+); sample "
                    r"aggregate rel err=([\d.]+); per-offset (\{.*\})")
RE_OLD = re.compile(r"\[decompose r(\d+)\] fit rel_err=([\d.]+); sample-block "
                    r"reconstruction max rel err=([\d.e+-]+)")


def main() -> None:
    data: dict = {}

    txt3 = (ROOT / "cluster/mos2decomp3/run.log").read_text(errors="ignore")
    txt2 = (ROOT / "cluster/mos2decomp2/run.log").read_text(errors="ignore")
    fit = {int(r): float(e) for r, e in RE_FIT.findall(txt2 + txt3)}
    data["mos2_ranks"] = np.array(sorted(fit))
    data["mos2_fit"] = np.array([fit[r] for r in sorted(fit)])

    agg_rows = RE_AGG.findall(txt3)
    for r, fe, ae, per in agg_rows:
        d = ast.literal_eval(per)
        offs = sorted(d)
        data[f"mos2_r{r}_agg"] = np.array([float(fe), float(ae)])
        data[f"mos2_r{r}_offsets"] = np.array(offs)
        data[f"mos2_r{r}_offerr"] = np.array([d[o] for o in offs])

    txt_si = (ROOT / "cluster/sifilmdecomp/run.log").read_text(errors="ignore")
    si = {int(r): (float(fe), float(me))
          for r, fe, me in RE_OLD.findall(txt_si)}
    data["si_ranks"] = np.array(sorted(si))
    data["si_fit"] = np.array([si[r][0] for r in sorted(si)])
    data["si_maxblock"] = np.array([si[r][1] for r in sorted(si)])

    # post-min-image conservation ladder (Si film, nk9)
    tags = ["dense", "r8", "r32", "r128"]
    bal, heat = [], []
    for tag in tags:
        d = np.load(ROOT / f"cluster/sifilm_nk9r/run_{tag}.npz")
        bb = d["final_bubble_balance"]
        bal.append(abs(bb[0] - bb[1]) / abs(bb[0]))
        h = d["final_heat"].sum(axis=-1)
        heat.append(h[0, -1])
    data["cons_tags"] = np.array(tags)
    data["cons_balance"] = np.array(bal)
    data["cons_heat"] = np.array(heat)

    np.savez_compressed(OUT, **data)
    print(f"wrote {OUT}")
    for k, v in data.items():
        print(f"  {k}: {np.asarray(v).shape} {np.asarray(v).ravel()[:6]}")


if __name__ == "__main__":
    main()
