#!/bin/bash
# Si: does the narrow, isolated pole population survive anharmonic
# self-consistency?
#
# The gating measurement of the 2026-08-14 pole/QNM audit (Sec. 38, and its
# decision tree Sec. 51). pole_sector_observations.md Sec. 1.8 found that Si
# carries modes that are simultaneously under-resolved and isolated -- the
# first bed that does -- but that census was taken at ITERATION 1, where the
# anharmonic width is not built yet, so its linewidths are a lower bound. Its
# own closing line: "A converged Si census is the number that would settle it."
#
# Three stages on one bed, one grid, one wmax, so the only thing that differs
# between the first and the last is how developed the scattering is:
#
#   cold  iteration 1, no warm start      the Sec. 1.8 picture, reproduced here
#   base  150 iterations, pole OFF        converge and freeze (QX_SAVE_SIGMA)
#   warm  the census on that frozen state the number Sec. 38 asks for
#
# extraction_only reports the candidates and hands the ring an EMPTY pole set,
# so every stage is bit-identical to the pole-free baseline and the census
# costs only the solve. That is also why `base` and `cold` may be compared:
# nothing the census does feeds back.
#
# wmax = 35 THz with retarded=fft, unlike Sec. 1.8, which ran wmax = 15 and was
# therefore Kramers-Kronig truncated (Sec. 1.9). gamma is a physical quantity
# and comparable across the two; the h-dependent columns (q_omega, E_leg,
# E_finite) are at h = 35/140 = 0.25 THz here against 0.125 there, so read
# those from THIS run only.
#
# eta = 0 on every stage. Do not add broadening -- see CLAUDE.md.
set -u
REPO=/capstor/scratch/cscs/pfischil/quatrex
OUT=${QX_OUT:-$REPO/cluster/sicensus}
mkdir -p "$OUT"

export QX_CONFIG=$REPO/cluster/sichk_base/quatrex_config.toml
export QX_RETARDED=fft QX_WMAX=35.0 QX_MIX=0.1 QX_BBCHECK=1 QX_NE=141

CENSUS="QX_POLE=1 QX_POLE_EXTRACT=1 QX_POLE_WMIN=0.3 QX_POLE_WMAX=15.0 QX_POLE_NP=32"

run () {
    echo "==================== $1  (maxit=$2) ===================="
    env QX_MAXIT=$2 $3 QX_NPZ=$OUT/run_$1.npz \
        python $REPO/phonon/studies/engine/run.py 2>&1 | tee $OUT/log_$1.txt \
        | grep -E "rel Sigma|pole census|CONTINUATION|WARM START|SAVED SIGMA|Traceback|Error" \
        | tail -12
    echo "---- $1: last residual, and the census population ----"
    grep -E "rel Sigma" "$OUT/log_$1.txt" | tail -1
    grep -c "pole census" "$OUT/log_$1.txt" || true
}

run cold 3   "$CENSUS"
run base 150 "QX_POLE=0 QX_SAVE_SIGMA=$OUT/sig_base.npz"
run warm 3   "$CENSUS QX_SIGMA_INIT=$OUT/sig_base.npz"

echo "==================== census done ===================="
