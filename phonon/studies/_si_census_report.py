#!/usr/bin/env python
"""Aggregate the per-q pole census across a run, and compare two stages.

Run:
    python phonon/studies/_si_census_report.py cluster/sicensus/log_cold.txt         cluster/sicensus/log_warm.txt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

Q_RE = re.compile(r"^\s*q \(([^)]*)\):\s*$")
HEAD_RE = re.compile(
    r"pole census: (\d+) candidates; (\d+) under-resolved .*?, "
    r"(\d+) isolated \(gamma/sep < 0\.5\)(?:; (\d+) CONTINUATION FAILURES)?")
ROW_RE = re.compile(r"^\s{4}(\S.*?)\s+min/p25/med/p75/max\s+(.*)$")
OUT_RE = re.compile(r"^\s*outcome: (.*)$")


def _floats(s):
    out = []
    for tok in s.split():
        try:
            out.append(float(tok))
        except ValueError:
            out.append(float("nan"))
    return out


def parse(path: Path) -> list[dict]:
    """Every per-q census block in the log, in order."""
    blocks, cur, q = [], None, None
    for line in path.read_text(errors="replace").splitlines():
        m = Q_RE.match(line)
        if m:
            q = m.group(1)
            continue
        m = HEAD_RE.search(line)
        if m:
            cur = {"q": q, "n": int(m.group(1)),
                   "unresolved": int(m.group(2)), "isolated": int(m.group(3)),
                   "continuation_failures": int(m.group(4) or 0),
                   "rows": {}, "outcome": {}}
            blocks.append(cur)
            continue
        if cur is None:
            continue
        m = ROW_RE.match(line)
        if m:
            cur["rows"][m.group(1).strip()] = _floats(m.group(2))
            continue
        m = OUT_RE.match(line)
        if m:
            for part in m.group(1).split(","):
                part = part.strip()
                if " x" in part:
                    key, _, cnt = part.rpartition(" x")
                    try:
                        cur["outcome"][key.strip()] = int(cnt)
                    except ValueError:
                        pass
            cur = None
    return blocks


def summarise(blocks: list[dict], label: str) -> dict:
    if not blocks:
        return {"label": label, "n_q": 0}
    tot = lambda k: sum(b[k] for b in blocks)                       # noqa: E731
    acc = sum(b["outcome"].get("accepted", 0) for b in blocks)
    why: dict[str, int] = {}
    for b in blocks:
        for k, v in b["outcome"].items():
            if k != "accepted":
                why[k] = why.get(k, 0) + v
    med = {}
    for key in ("gamma [THz]", "q_omega", "gamma/sep", "E_leg^max",
                "E_finite", "eps_z", "gamma_sens/gamma"):
        vals = [b["rows"][key][2] for b in blocks
                if key in b["rows"] and len(b["rows"][key]) > 2
                and np.isfinite(b["rows"][key][2])]
        med[key] = np.array(vals) if vals else np.array([])
    return {"label": label, "n_q": len(blocks), "candidates": tot("n"),
            "unresolved": tot("unresolved"), "isolated": tot("isolated"),
            "continuation_failures": tot("continuation_failures"),
            "accepted": acc, "refusals": why, "medians": med}


def report(s: dict) -> None:
    print(f"\n=== {s['label']} ===")
    if not s["n_q"]:
        print("  no census blocks found")
        return
    print(f"  q points censused      {s['n_q']}")
    print(f"  candidates             {s['candidates']}")
    print(f"  under-resolved         {s['unresolved']} "
          f"({100 * s['unresolved'] / max(s['candidates'], 1):.1f} %)")
    print(f"  isolated               {s['isolated']} "
          f"({100 * s['isolated'] / max(s['candidates'], 1):.1f} %)")
    print(f"  ACCEPTED               {s['accepted']} "
          f"({100 * s['accepted'] / max(s['candidates'], 1):.1f} %)")
    if s["continuation_failures"]:
        print(f"  continuation failures  {s['continuation_failures']} "
              "(roots in the UPPER half plane)")
    if s["refusals"]:
        top = sorted(s["refusals"].items(), key=lambda kv: -kv[1])[:5]
        print("  refused because: "
              + ", ".join(f"{k} x{v}" for k, v in top))
    # The shipped promotion gate is the crude ratio: leg_weight_tol defaults
    # to 0, so `accepted` is decided by q_omega < q_in and NOT by the exact
    # line-weight error. Sec. 1.6 measured the gap -- the ratio rule called 140
    # of 144 CNT modes under-resolved where the exact worst case was 1.3e-04.
    # So report what the exact gate would say, separately.
    eleg = s["medians"].get("E_leg^max")
    if eleg is not None and eleg.size:
        print("  EXACT gate (per-q median E_leg^max), q points whose median "
              "line is carried to:")
        for tol in (0.01, 0.05, 0.10):
            n = int(np.sum(eleg <= tol))
            print(f"    better than {100 * tol:4.0f} %   {n:>3}/{eleg.size} q")
        print(f"    worse than 100 %    {int(np.sum(eleg > 1.0)):>3}/{eleg.size} q"
              "   <- the regime the method exists for")

    print("  per-q MEDIANS, summarised across q (min/med/max):")
    for key, vals in s["medians"].items():
        if vals.size:
            print(f"    {key:<18} {vals.min():.3g}  {np.median(vals):.3g}  "
                  f"{vals.max():.3g}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--last-only", action="store_true",
                    help="keep only the final census sweep in each log "
                         "(the converged one), not every SCBA iteration")
    args = ap.parse_args()

    summaries = []
    for path in args.logs:
        if not path.exists():
            print(f"missing: {path}")
            continue
        blocks = parse(path)
        if args.last_only and blocks:
            # A sweep restarts when a q repeats; keep the trailing sweep.
            seen, start = set(), 0
            for i, b in enumerate(blocks):
                if b["q"] in seen:
                    seen, start = {b["q"]}, i
                else:
                    seen.add(b["q"])
            blocks = blocks[start:]
        s = summarise(blocks, path.stem)
        summaries.append(s)
        report(s)

    if len(summaries) >= 2 and all(x["n_q"] for x in summaries[:2]):
        a, b = summaries[0], summaries[1]
        print(f"\n=== {a['label']} -> {b['label']} ===")
        for key in ("candidates", "unresolved", "isolated", "accepted"):
            d = b[key] - a[key]
            frac = f"{b[key] / a[key]:.2f}x" if a[key] else "n/a"
            print(f"  {key:<16} {a[key]:>6} -> {b[key]:>6}  "
                  f"({d:+d}, {frac})")
        ga, gb = a["medians"].get("gamma [THz]"), b["medians"].get("gamma [THz]")
        if ga is not None and gb is not None and ga.size and gb.size:
            print(f"  median per-q gamma  {np.median(ga):.4g} -> "
                  f"{np.median(gb):.4g} THz "
                  f"({np.median(gb) / np.median(ga):.2f}x)")
        print("\n  The accepted count is the answer: it is the population that "
              "passed every\n  gate at once. Growing or holding means the bed "
              "supports the method;\n  collapsing means it does not (audit "
              "Sec. 51).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
