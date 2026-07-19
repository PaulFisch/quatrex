"""Master orchestrator: run every queued phonon experiment back-to-back in
ONE cluster session, no intervention.

Each sub-study loops its own rungs and SKIPS any whose run.npz already
exists, so this whole script is idempotent and resumable: if the session
is interrupted (node reclaimed), relaunching it continues where it left
off. Launch it once via tortin.py and walk away:

    python phonon/scripts/tortin.py launch --name all -- \
        python phonon/studies/_run_all_cluster.py

Order (highest priority first):
  1. CNT (3,3) L8, eta=0, g_band=2: support-complete Kramers-Kronig x
     contact-dressing (trunc_bare / kk_bare / kk_dressed) -- does the
     complete KK + the GW-style dressed contact converge the run that
     diverged at iteration 63?
  2. CNT (3,3) L4 nu2: the acoustic-floor-corrected non-uniform grid
     (the uni/nu rungs are already done and are skipped).

A diverged rung is a RESULT (its residual trace + snapshot are saved);
the orchestrator moves on to the next rung either way.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

STUDIES = [
    "phonon/studies/_run_cnt33_L8_kk.py",
    "phonon/studies/_run_cnt33_L4_nugrid.py",
]


def main() -> int:
    for s in STUDIES:
        print(f"\n===== ORCHESTRATOR: running {s} =====", flush=True)
        rc = subprocess.run([sys.executable, str(REPO / s)]).returncode
        print(f"===== ORCHESTRATOR: {s} exited rc={rc} =====", flush=True)
    print("\n===== ORCHESTRATOR: ALL QUEUED EXPERIMENTS COMPLETE =====",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
