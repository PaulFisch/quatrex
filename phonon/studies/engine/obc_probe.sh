#!/bin/bash
# Which OBC algorithm survives a fine frequency grid?
#
# At ne = 8001 on si4x2 the spectral OBC reports
#
#   High error at rank 1 for left of OBCSolver Spectral:
#   Relative recursion error: 6.507e-01
#
# i.e. the surface Green's function does not satisfy its own recursion to
# better than 65 %. That is a candidate cause for the fine-grid instability
# that has been read as physics: at h = 0.0044 THz far more grid points land
# near band edges, where a contour-based NEVP has to resolve roots crowding
# the contour.
#
# Three arms separate the two suspects:
#
#   spec  spectral + beyn   the default, and the one that warns
#   full  spectral + full   dense linearised eigensolver, no contour to tune
#   sr    sancho-rubio      iterative, no NEVP at all
#
# If `full` and `sr` are both clean, the contour is the problem. If only `sr`
# is clean, the spectral construction is. If all three warn, it is not the OBC.
#
# eta = 0 on every arm. Do not add broadening -- see CLAUDE.md.
set -u
REPO=/capstor/scratch/cscs/pfischil/quatrex
OUT=${QX_OUT:-$REPO/cluster/obcprobe}
mkdir -p "$OUT"
export QX_CONFIG=$REPO/cluster/si4x2/quatrex_config.toml
export QX_RETARDED=fft QX_WMAX=35.0 QX_MIX=0.1 QX_BBCHECK=1
export QX_NE=${QX_NE:-8001} QX_MAXIT=${QX_MAXIT:-3}

arm () {   # name, extra env
    echo "==================== $1 ===================="
    env $2 QX_NPZ=$OUT/run_$1.npz \
        python $REPO/phonon/studies/engine/run.py 2>&1 | tee $OUT/log_$1.txt \
        | grep -E "rel Sigma|SAVED|Traceback|Error" | tail -4
    echo "---- $1: OBC recursion warnings ----"
    echo -n "  count: "; grep -c "High error" $OUT/log_$1.txt || true
    grep -A2 "High error" $OUT/log_$1.txt | grep -oE "Relative recursion error: [0-9.e+-]+" \
        | sort -u | tail -3
}

arm spec "QX_OBC_ALG=spectral QX_NEVP=beyn"
arm full "QX_OBC_ALG=spectral QX_NEVP=full"
arm sr   "QX_OBC_ALG=sancho-rubio"
echo "==================== obc probe done ===================="
