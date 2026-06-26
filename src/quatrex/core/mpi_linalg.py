# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Shared MPI linear-algebra primitives for the row-partitioned SCBA mixers.

The self-energy iterate ``x = [Sigma^<, Sigma^>, Sigma^R]`` is row-partitioned
across ``MPI.COMM_WORLD`` (disjoint slices; the concatenation is the global
vector). Any quantity that contracts the distributed dimension -- an inner
product ``u^H v``, a norm, a Gram ``U^H V``, a trust-region cap -- is a local
contraction followed by ``Allreduce(SUM)``. Every quasi-Newton / extrapolation
mixer (RRE, Broyden, RPM, JFNK) needs the identical primitive, so it lives here
once instead of each mixer re-opening ``MPI.COMM_WORLD`` and writing its own
reduction.
"""

import numpy as np


def get_comm():
    """Return ``(COMM_WORLD, SUM)``, or ``(None, None)`` without mpi4py."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD, MPI.SUM
    except Exception:  # pragma: no cover - no-MPI fallback
        return None, None


def allreduce_sum(comm, op_sum, arr: np.ndarray) -> np.ndarray:
    """``Allreduce(SUM)`` of a contiguous array (identity when serial)."""
    if comm is None or comm.size <= 1:
        return arr
    out = np.empty_like(arr)
    comm.Allreduce(np.ascontiguousarray(arr), out, op=op_sum)
    return out


def global_dot(comm, op_sum, u: np.ndarray, v: np.ndarray) -> float:
    """Global real inner product ``Re(u^H v)`` over row-partitioned vectors."""
    s = np.array([float(np.vdot(u, v).real)], dtype=np.float64)
    return float(allreduce_sum(comm, op_sum, s)[0])


def global_norm(comm, op_sum, v: np.ndarray) -> float:
    """Global L2 norm ``sqrt(Re(v^H v))`` of a row-partitioned vector."""
    return float(np.sqrt(max(global_dot(comm, op_sum, v, v), 0.0)))


def global_gram(comm, op_sum, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Global ``U^H V`` (contracts the distributed rows) via ``Allreduce(SUM)``."""
    return allreduce_sum(comm, op_sum, np.ascontiguousarray(u.conj().T @ v))


def trust_cap(comm, op_sum, step: np.ndarray, x: np.ndarray,
              trust: float) -> np.ndarray:
    """Trust-region cap: limit ``||step||`` to ``trust * ||x||`` (global norms).

    ``trust <= 0`` disables the cap. Both squared norms are reduced together.
    """
    if trust <= 0.0:
        return step
    buf = np.array([float(np.vdot(step, step).real),
                    float(np.vdot(x, x).real)], dtype=np.float64)
    buf = allreduce_sum(comm, op_sum, buf)
    sn, xn = np.sqrt(max(buf[0], 0.0)), np.sqrt(max(buf[1], 0.0))
    if sn > trust * xn and sn > 0.0:
        return step * (trust * xn / sn)
    return step
