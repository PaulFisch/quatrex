"""Iteration-resolved positivity trace from a run log.

``phonon/solver.py::_check_positivity`` (behind ``pole_sector.psd_check``, env
``QX_POLE_PSD=1``) prints, once per SCBA iteration and per target,

    positivity sigma_lesser    worst=-1.234e-03 at w[17]  VIOLATION

for ``sigma_lesser``, ``sigma_greater``, ``g_lesser`` and ``g_greater``. Sigma is
the ROOT check: ``G^< = G^R Sigma^< G^A`` is a congruence, so a PSD Sigma cannot
produce a non-PSD G, and if G fails then Sigma failed first.

This reads one or more logs and lays the trace out per iteration next to the
SCBA residual, so "does positivity break before the blow-up, and at which
omega" is answered by looking at one table.

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
            line = (f"{r['iter']:>3}  "
                    f"{(r['res'] if r['res'] is not None else float('nan')):>10.3e} "
                    f"{(r['bal'] if r['bal'] is not None else float('nan')):>9.3e}")
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
