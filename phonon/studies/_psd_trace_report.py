"""Iteration-resolved positivity trace from a run log.

Usage::
    python -m phonon.studies._psd_trace_report cluster/mos2psd10/slurm-*.out ...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ITER = re.compile(r"^Iteration (\d+)")
PSD = re.compile(
    r"positivity\s+(\S+)\s+worst=([-+0-9.eE]+)\s+at w\[(-?\d+)\]\s+(\S+)"
)
RES = re.compile(
    r"rel Sigma\^R residual ([-+0-9.eE]+); lead balance ([-+0-9.eE]+)"
)
TARGETS = ("sigma_lesser", "sigma_greater", "g_lesser", "g_greater")


def parse(path: Path):
    rows, cur = [], None
    for line in path.read_text(errors="replace").splitlines():
        m = ITER.match(line.strip())
        if m:
            cur = {"iter": int(m.group(1)), "psd": {}, "res": None, "bal": None}
            rows.append(cur)
            continue
        if cur is None:
            continue
        m = PSD.search(line)
        if m:
            cur["psd"][m.group(1)] = (float(m.group(2)), int(m.group(3)),
                                      m.group(4))
            continue
        m = RES.search(line)
        if m:
            cur["res"], cur["bal"] = float(m.group(1)), float(m.group(2))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("logs", nargs="+", type=Path)
    args = p.parse_args(argv)

    for path in args.logs:
        rows = parse(path)
        have = any(r["psd"] for r in rows)
        print(f"\n=== {path}  ({len(rows)} iterations, "
              f"positivity gate {'ON' if have else 'OFF -- no gate lines'})")
        if not have:
            continue
        head = f"{'it':>3}  {'residual':>10} {'lead bal':>9}"
        for t in TARGETS:
            head += f"  {t:>16}"
        print(head)
        for r in rows:
            # A missing residual means the iteration was cut off before it
            # printed one (a walltime kill), NOT a NaN result. Rendering it as
            # nan reads as a numeric blow-up that did not happen.
            res = f"{r['res']:>10.3e}" if r["res"] is not None else f"{'--':>10}"
            bal = f"{r['bal']:>9.3e}" if r["bal"] is not None else f"{'--':>9}"
            line = f"{r['iter']:>3}  {res} {bal}"
            for t in TARGETS:
                if t not in r["psd"]:
                    line += f"  {'-':>16}"
                    continue
                worst, wi, flag = r["psd"][t]
                mark = "!" if flag == "VIOLATION" else " "
                # w[-1] is the gate's sentinel for "no negative eigenvalue
                # anywhere on the local grid", not a frequency index.
                where = "--" if wi < 0 else str(wi)
                line += f"  {worst:>+11.3e}@{where:<3}{mark}"
            print(line)
        first = next((r for r in rows
                      if any(v[2] == "VIOLATION" for v in r["psd"].values())),
                     None)
        if first is None:
            print("  no VIOLATION at any iteration")
        else:
            bad = [k for k, v in first["psd"].items() if v[2] == "VIOLATION"]
            print(f"  first VIOLATION at iteration {first['iter']}: "
                  f"{', '.join(sorted(bad))}  "
                  f"(residual {first['res']}, lead balance {first['bal']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
