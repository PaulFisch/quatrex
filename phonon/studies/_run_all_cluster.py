"""Master orchestrator: run every queued phonon experiment back-to-back in ONE
cluster session, no intervention.

Each sub-study loops its own rungs and SKIPS any whose run.npz already
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
