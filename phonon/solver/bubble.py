"""Re-export of the canonical bubble kernel from the installed package.

The implementation lives in :mod:`quatrex.phonon.bubble` so that the
production block-sparse solver (``quatrex.phonon.sse_phonon_phonon``)
can import it without depending on the local ``phonon/`` directory
being on ``sys.path``. This module provides the
``from phonon.solver.bubble import bubble_dense`` convenience used by
the dense reference solver and the local examples.
"""

from quatrex.phonon.bubble import bubble_dense

__all__ = ["bubble_dense"]
