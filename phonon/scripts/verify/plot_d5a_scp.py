#!/usr/bin/env python
"""Plot the d5a soft-mode SCP investigation: SCBA residual vs iteration for the
key configs, showing (b) the Kramers-Kronig real part destabilises the soft mode
(full-KK 'fft' diverges; 'half' converges) and (a) a self-consistent tadpole
re-stabilises the rigorous full-KK calculation.

Parses the run logs in /tmp/claude/{d5a_scp,d5a_sc,d5a_fft_tad}.log.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = "/tmp/claude"
FIG = "/usr/scratch/mont-fort11/pfischill/quatrex/document/fig/transport_sweeps"
Path(FIG).mkdir(parents=True, exist_ok=True)

ITER = re.compile(r"SCBA iter\s+(\d+):.*?resid\s*=\s*([\d.eE+-]+)")
SUMMARY = re.compile(r"^\s+(.+?)\s*: conv=(\w+)\s+it=\s*(\d+)\s+resid=([\d.eE+-]+)")


def parse(logfile):
    """Return list of (label, conv, iters_array, resid_array) per config block."""
    blocks, cur = [], []
    out = []
    for line in Path(logfile).read_text().splitlines():
        m = ITER.search(line)
        if m:
            cur.append((int(m.group(1)), float(m.group(2))))
            continue
        s = SUMMARY.match(line)
        if s:
            label, conv = s.group(1), s.group(2)
            arr = np.array(cur) if cur else np.empty((0, 2))
            out.append((label, conv == "True", arr))
            cur = []
    return out


# Pull the key configs from the right logs
want = {
    "fft (full KK), no tadpole": ("d5a_fft_tad.log", 0),
    "half (KK real dropped), no tadpole": ("d5a_sc.log", 0),
    "fft + self-consistent tadpole": ("d5a_fft_tad.log", 1),
    "fixed-<uu> tadpole (fft)": ("d5a_scp.log", 2),
}
parsed = {f: parse(f"{LOGS}/{f}") for f in {v[0] for v in want.values()}}

colors = {"fft (full KK), no tadpole": "C3",
          "half (KK real dropped), no tadpole": "C0",
          "fft + self-consistent tadpole": "C2",
          "fixed-<uu> tadpole (fft)": "C1"}
styles = {"fft (full KK), no tadpole": "--",
          "half (KK real dropped), no tadpole": "-",
          "fft + self-consistent tadpole": "-",
          "fixed-<uu> tadpole (fft)": ":"}

fig, ax = plt.subplots(figsize=(8.2, 5.0))
for label, (logf, idx) in want.items():
    blocks = parsed[logf]
    if idx >= len(blocks):
        continue
    _, conv, arr = blocks[idx]
    if arr.size == 0:
        continue
    tag = "converged" if conv else "did NOT converge"
    ax.semilogy(arr[:, 0], arr[:, 1], styles[label], color=colors[label], lw=2,
                label=f"{label}  [{tag}]")
ax.axhline(1e-3, ls=":", color="gray", lw=1)
ax.text(2, 1.15e-3, "scba_tol = 1e-3", color="gray", fontsize=8)
ax.set_xlabel("SCBA iteration")
ax.set_ylabel(r"residual $\Vert\Delta\Sigma\Vert/\Vert\Sigma\Vert$")
ax.set_title("d5a SiNW soft-mode SCBA: Kramers-Kronig + self-consistent tadpole")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIG}/d5a_scp_convergence.pdf")
fig.savefig(f"{FIG}/d5a_scp_convergence.png", dpi=130)
print("wrote d5a_scp_convergence.pdf")
for label, (logf, idx) in want.items():
    b = parsed[logf]
    if idx < len(b):
        lab, conv, arr = b[idx]
        last = arr[-1, 1] if arr.size else float("nan")
        print(f"  {label:38s}: conv={conv}  npts={len(arr)}  final_resid={last:.2e}")
