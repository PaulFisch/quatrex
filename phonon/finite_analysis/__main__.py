"""Entry point: ``python -m phonon_inputs.finite_analysis ...``."""
import sys
from pathlib import Path


def _bootstrap_quatrex_path() -> None:
    """Inject ``<repo_root>/src`` if the ``quatrex`` package is not installed.

    This package is normally consumed after ``pip install -e .`` from the repo
    root, but on shared clusters that often isn't done. Walk up from this file
    looking for a ``src/quatrex`` directory and prepend it to ``sys.path``.
    """
    try:
        import quatrex  # noqa: F401
        return
    except ImportError:
        pass
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src" / "quatrex"
        if candidate.is_dir():
            sys.path.insert(0, str(parent / "src"))
            return


_bootstrap_quatrex_path()
from .cli import main  # noqa: E402  (path bootstrap must precede import)

if __name__ == "__main__":
    raise SystemExit(main())
