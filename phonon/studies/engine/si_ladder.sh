#!/bin/bash
# Si: does the pole sector reproduce the fine-grid answer from a coarse grid?
#
# The question the method exists to answer. Three arms, all at wmax = 35 THz
# (past twice the 15 THz band top) so the Kramers-Kronig integral is not
# truncated and the only differences are the grid and the sector:
#
#   base  h = 0.25   ne = 141   the ring alone on the coarse grid
#   pole  h = 0.25   ne = 141   the same grid with leg="congruence"
#   ref   h = 0.125  ne = 281   the answer the coarse arms must reproduce
#
# pole must land on ref, not on base. base-vs-ref is the error the sector is
# supposed to remove; pole-vs-ref is what is left of it.
#
# This became affordable when the sector was batched (pole_solve_batching.md):
# the pole arm was ~190 s/iteration and is now ~18 s, so 150 iterations fit in
# well under an hour instead of eight. sipole2.sh was the pre-batching version
# and could not reach ref at all.
#
# eta = 0 on every arm. Do not add broadening -- see CLAUDE.md.
set -u
REPO=/capstor/scratch/cscs/pfischil/quatrex
OUT=${QX_OUT:-$REPO/cluster/siladder}
mkdir -p "$OUT"
export QX_CONFIG=$REPO/cluster/sichk_base/quatrex_config.toml
export QX_RETARDED=fft QX_WMAX=35.0 QX_MIX=0.1 QX_BBCHECK=1

run () {
    echo "==================== $1  (ne=$2, maxit=$3) ===================="
    env QX_NE=$2 QX_MAXIT=$3 $4 QX_NPZ=$OUT/run_$1.npz \
        python $REPO/phonon/studies/engine/run.py 2>&1 | tee $OUT/log_$1.txt \
        | grep -E "rel Sigma|SIGN|converged|SAVED|q solved|Traceback|Error" \
        | tail -6
    echo "---- $1 last residual / heat ----"
    grep -E "rel Sigma" $OUT/log_$1.txt | tail -1
    grep -E "SAVED" $OUT/log_$1.txt | tail -1
}

run base 141 150 "QX_POLE=0"
run pole 141 150 "QX_POLE=1 QX_POLE_NP=32 QX_POLE_WMIN=0.3 QX_POLE_WMAX=15.0 QX_POLE_LEG=congruence"
run ref  281 150 "QX_POLE=0"
echo "==================== all arms done ===================="
