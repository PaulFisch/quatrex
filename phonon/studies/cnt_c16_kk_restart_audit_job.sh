#!/bin/bash
set -eu

repo=/capstor/scratch/cscs/pfischil/quatrex
base="$repo/cluster"

export QX_MINIT=3
export QX_MIXMETHOD=linear
export QX_SIGMATOL=0.00001
export QX_HEATTOL=0.01
export QX_GBAND=3
export QX_COMM_BACKEND=device_mpi
export QX_POLE_PSD=1
export QX_BBCHECK=1
export QX_SIGMA_BEST_LIVE=1
export QX_SIGMA_BEST_LIVE_STRIDE=25
export QX_REQUIRE_RESTARTABLE=1

run_stage() {
    name=$1
    config=$2
    restart=$3
    mixing=$4
    iterations=$5
    diagnostic=$6
    output="$base/$name"
    mkdir -p "$output"
    export QX_CONFIG="$base/$config/quatrex_config.toml"
    export QX_NPZ="$output/run.npz"
    export QX_MIX="$mixing"
    export QX_MAXIT="$iterations"
    export QX_SAVE_SIGMA="$output/sigma_final.npz"
    export QX_SAVE_SIGMA_BEST="$output/sigma_best.npz"
    if [ "$restart" = none ]; then
        unset QX_SIGMA_INIT
    else
        export QX_SIGMA_INIT="$base/$restart"
    fi
    if [ "$diagnostic" = yes ]; then
        export QX_DIAG_SPECTRAL=1
    else
        unset QX_DIAG_SPECTRAL
    fi
    echo "START ${name} MIX=${mixing} MAXIT=${iterations} INIT=${restart}"
    srun --exclusive --exact \
        --nodes=1 --ntasks=4 --ntasks-per-node=4 \
        --gpus-per-task=1 --cpu-bind=cores bash -c \
        'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; exec python \
            /capstor/scratch/cscs/pfischil/quatrex/phonon/studies/engine/run.py'
}

run_stage c16-kk-cont-b005 c16-kk c16-kk/sigma_final.npz 0.05 500 yes
run_stage c16-half-seed-tight c16-half none 0.1 200 no
run_stage c16-kk-from-half-b005 c16-kk \
    c16-half-seed-tight/sigma_final.npz 0.05 500 yes
