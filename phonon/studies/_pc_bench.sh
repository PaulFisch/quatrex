#!/usr/bin/env bash
# Preconditioner A/B/C benchmark: identical Newton solves from the
# archived L4 stall snapshot (the stiff mid-residual region), differing
# only in the inner-GMRES deflation:
#   none    -- baseline exact-JVP Newton
#   recycle -- harmonic-Ritz pairs recycled from the previous step's
#              Arnoldi relation (zero extra JVPs)
#   fresh   -- the literal low-rank Schur surrogate (rank JVPs per step)
# Metric: per-step gmres_m (in the newton# log lines) and the residual
# trajectory per cumulative JVP. Sequential on one node.
set -uo pipefail
REPO=/usr/scratch/mont-fort11/pfischill/quatrex
CFG=$REPO/phonon/studies/out/anderson_test/cnt33_L4_linear/quatrex_config.toml
SNAP=$REPO/phonon/studies/out/anderson_test/jprobe_snaps/L4_stall.npz
OUT=$REPO/phonon/studies/out/newton_pc_bench
mkdir -p "$OUT"
for ARM in none recycle fresh; do
  echo "=== ARM precond=$ARM ==="
  env QX_CONFIG="$CFG" QX_GBAND=2 QX_MIXMETHOD=newton \
      QX_SIGMA_INIT="$SNAP" QX_MIX=0.2 QX_SIGMATOL=1e-8 QX_MAXIT=12 \
      QX_NEWTON_WARMUP=1 QX_NEWTON_SWITCH=10 QX_NEWTON_KRYLOV=30 \
      QX_NEWTON_TRUSTMAX=2.0 QX_NEWTON_PRECOND=$ARM \
      QX_NEWTON_PRECOND_RANK=12 \
      QX_NPZ=$OUT/run_$ARM.npz \
      bash "$REPO/phonon/studies/engine/launch.sh"
  echo "=== ARM $ARM done (exit $?) ==="
done
echo "=== PC BENCH COMPLETE ==="
