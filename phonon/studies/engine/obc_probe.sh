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

# srun runs THIS SCRIPT on every rank, so a shared `tee` target would have
# four writers opening the same path with O_TRUNC and the log would be
# shredded. Per-rank files, and only rank 0 narrates.
R=${SLURM_PROCID:-0}
say () { [ "$R" = "0" ] && echo "$@"; }

arm () {   # name, extra env
    say "==================== $1 ===================="
    env $2 QX_NPZ=$OUT/run_$1.npz \
        python $REPO/phonon/studies/engine/run.py 2>&1 | tee $OUT/log_$1.r$R.txt \
        | { [ "$R" = "0" ] && grep -E "rel Sigma|SAVED|Traceback|Error" | tail -4 \
            || cat > /dev/null; }
    # Every rank counts its OWN warnings: the recursion error is reported per
    # rank and per contact, and "which rank" is part of the answer.
    n=$(grep -c "High error" $OUT/log_$1.r$R.txt || true)
    worst=$(grep -oE "Relative recursion error: [0-9.e+-]+" $OUT/log_$1.r$R.txt \
            | grep -oE "[0-9.e+-]+$" | sort -g | tail -1)
    echo "  OBC[$1] rank $R: $n warning(s), worst relative recursion error ${worst:-none}"
}

arm spec "QX_OBC_ALG=spectral QX_NEVP=beyn"
arm full "QX_OBC_ALG=spectral QX_NEVP=full"
arm sr   "QX_OBC_ALG=sancho-rubio"
say "==================== obc probe done ===================="
