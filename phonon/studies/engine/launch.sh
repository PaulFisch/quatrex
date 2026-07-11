#!/usr/bin/env bash
# Thread-pinned launcher for a production phonon-transport run.
#
# One rank = one core (1 thread/rank): the only valid setup for a strong-scaling
# sweep on the shared node. Pins every intra-rank threadpool to 1 and binds
# ranks to cores. The phph self-energy parallelizes over MPI RANKS (not a thread
# pool), so do NOT set QUATREX_PHPH_THREADS (that is the dense reference only).
#
# Env:
#   QX_CONFIG  (required) path to the quatrex_config.toml
#   NRANKS     (default 1) mpirun -np
#   QX_BALLISTIC=1         zero the vertex (G_ball baseline)
#   QX_NPZ                 snapshot path
#   PROFILE_SYNC=1         add the MPI-barrier comm-wait timing
#   plus any QX_* overrides consumed by run.py (QX_BCS/QX_QCS/QX_ETA/...).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${QX_CONFIG:?set QX_CONFIG to the toml path}"
NRANKS="${NRANKS:-1}"

# BLAS stays SINGLE-threaded; the 3-phonon bubble (~99% of a step) instead
# parallelises its omega/tau batch over a thread pool inside ring_contract
# (QUATREX_PHPH_RING_THREADS). So per rank: 1 BLAS thread x QX_RING_THREADS
# pool threads. Default: a single rank FILLS the node (~min(64,nproc)); MPI
# runs keep 1 (ranks own cores). Budget for sweeps:
# NRANKS * QX_RING_THREADS * concurrent <= cores.
#
# NOTE: the core count MUST be read before OMP_NUM_THREADS is exported --
# coreutils `nproc` honors OMP_NUM_THREADS, so the old order silently gave
# _ncpu=1 and serialized the bubble on every launch (found 2026-07-11; the
# historic "56x/1232 GF/s" pool numbers in this header predated that and are
# superseded by phonon/studies/_bench_sse_stages.py).
_ncpu="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 8)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMP_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if [ -n "${QX_RING_THREADS:-}" ]; then
  export QUATREX_PHPH_RING_THREADS="$QX_RING_THREADS"
elif [ "$NRANKS" -gt 1 ]; then
  export QUATREX_PHPH_RING_THREADS=1
else
  export QUATREX_PHPH_RING_THREADS="$(( _ncpu < 64 ? _ncpu : 64 ))"
fi
echo "launch: ncpu=$_ncpu ring_threads=$QUATREX_PHPH_RING_THREADS"
export QTX_PROFILE_LEVEL="${QTX_PROFILE_LEVEL:-default}"
[ "${PROFILE_SYNC:-0}" = "1" ] && export QTX_PROFILE_COMM_SYNC=1

echo "launch: NRANKS=$NRANKS QX_CONFIG=$QX_CONFIG BCS=${QX_BCS:-cfg} QCS=${QX_QCS:-cfg} ballistic=${QX_BALLISTIC:-0}"
if [ "$NRANKS" -gt 1 ]; then
  # Default: 1 rank/core across both sockets. Override QX_MPI_BIND for hybrid
  # MPI x ring-pool layouts (e.g. "--map-by numa --bind-to numa" = 1 rank/socket
  # each running a ring pool) or SMT oversubscription ("--map-by hwthread
  # --bind-to hwthread --oversubscribe" for NRANKS>128).
  _bind="${QX_MPI_BIND:---bind-to core --map-by core}"
  exec mpirun $_bind -np "$NRANKS" python -u "$HERE/run.py"
elif [ "${QX_INTERLEAVE:-0}" = "1" ] && command -v numactl >/dev/null 2>&1; then
  # Single rank spanning BOTH NUMA sockets: interleave the bubble's buffers
  # across nodes so a >64-thread pool isn't first-touch-bound to one socket.
  # Marginal (~1.15x) and noisy -- a single cell is ~capped at one socket; the
  # node-fill lever is one cell PER socket (launch_cells_concurrent), not this.
  exec numactl --interleave=all python -u "$HERE/run.py"
else
  exec python -u "$HERE/run.py"
fi
