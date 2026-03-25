#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    shared_script = Path(__file__).resolve().with_name("plot_phonon_band.py")
    sys.argv = [str(shared_script), "--material", "si", *sys.argv[1:]]
    runpy.run_path(str(shared_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
