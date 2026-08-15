"""CLI dispatch: ``python -m phonon.studies <investigation> {run,plot} [...]``."""

import argparse
import importlib
import sys

# "sse_verify" and "bte_linewidths" were registered here but never committed
# (no blob for either path anywhere in history), so both verbs raised
# ModuleNotFoundError on dispatch. Removed 2026-08-15 rather than left
# advertising modules that do not exist.
INVESTIGATIONS = ("conservation", "linewidths", "ballistic",
                  "convergence", "transport")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phonon.studies",
        description=__doc__,
    )
    parser.add_argument("investigation", choices=INVESTIGATIONS)
    parser.add_argument("action", choices=("run", "plot"))
    args, rest = parser.parse_known_args()

    module = importlib.import_module(f"phonon.studies.{args.investigation}")
    return getattr(module, args.action)(rest) or 0


if __name__ == "__main__":
    sys.exit(main())
