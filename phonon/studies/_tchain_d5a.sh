#!/usr/bin/env bash
# d5a L2 temperature continuation with the exact-Jacobian Newton mixer.
#
# The old d5a temperature sweep (cluster/d5a_tempsweep, L1, Anderson)
# died in the 150 -> 200 K band. This chain follows the physical branch
# upward from the last good rung: each rung warm-starts from the
# previous converged Sigma (QX_SIGMA_INIT) and runs the two-phase
# Picard -> exact-Newton mixer. Where the branch ends, the Newton logs
# (GMRES stagnation / backtracking) bracket the endpoint instead of a
# diverging iteration.
#
# Single rank, one node (the bubble thread-pools over the ring).
set -uo pipefail
REPO=/usr/scratch/mont-fort11/pfischill/quatrex
CFG=$REPO/phonon/studies/out/newton_tchain_d5a/quatrex_config.toml
OUT=$REPO/phonon/studies/out/newton_tchain_d5a
mkdir -p "$OUT"
PREV=
for T in 150 160 170 180 190 200; do
  TL=$((T + 5)); TR=$((T - 5))
  echo "=== RUNG T=$T (TL=$TL TR=$TR) warm=${PREV:-cold} ==="
  env QX_CONFIG="$CFG" QX_TLEFT=$TL QX_TRIGHT=$TR \
      QX_MIXMETHOD=newton QX_MIX=0.2 QX_SIGMATOL=1e-8 QX_MAXIT=250 \
      QX_NEWTON_SWITCH=10 QX_NEWTON_KRYLOV=30 QX_NEWTON_TRUSTMAX=2.0 \
      ${PREV:+QX_SIGMA_INIT="$PREV"} \
      QX_SAVE_SIGMA="$OUT/sigma_T$T.npz" QX_NPZ="$OUT/run_T$T.npz" \
      bash "$REPO/phonon/studies/engine/launch.sh"
  echo "=== RUNG T=$T done (exit $?) ==="
  if [ -f "$OUT/sigma_T$T.npz" ]; then PREV="$OUT/sigma_T$T.npz"; fi
done
echo "=== CHAIN COMPLETE ==="
