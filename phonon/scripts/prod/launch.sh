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
# (QUATREX_PHPH_RING_THREADS) -- the per-omega matmuls are too small for BLAS
# threading (~1.5x@8) but the batch scales near-linearly (~15x@32, bit-exact).
# So per rank: 1 BLAS thread x QX_RING_THREADS pool threads. Budget
# NRANKS * QX_RING_THREADS * concurrent_cells <= cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMP_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export QUATREX_PHPH_RING_THREADS="${QX_RING_THREADS:-1}"
export QTX_PROFILE_LEVEL="${QTX_PROFILE_LEVEL:-default}"
[ "${PROFILE_SYNC:-0}" = "1" ] && export QTX_PROFILE_COMM_SYNC=1

echo "launch: NRANKS=$NRANKS QX_CONFIG=$QX_CONFIG BCS=${QX_BCS:-cfg} QCS=${QX_QCS:-cfg} ballistic=${QX_BALLISTIC:-0}"
if [ "$NRANKS" -gt 1 ]; then
  exec mpirun --bind-to core --map-by core -np "$NRANKS" python -u "$HERE/run.py"
else
  exec python -u "$HERE/run.py"
fi
