#!/bin/bash
set -eu

repo=/capstor/scratch/cscs/pfischil/quatrex
base="$repo/cluster"
restart="$base/cnt-l48x2-deep300-r3/sigma_best.npz"

export QX_MAXIT=150
export QX_MINIT=3
export QX_MIX=0.1
export QX_MIXMETHOD=linear
export QX_SIGMATOL=0.00000001
export QX_HEATTOL=0.01
export QX_GBAND=3
export QX_BCS=2
export QX_COMM_BACKEND=device_mpi
export QX_OBC_MEMO=cache
export QX_MAXBATCH=100000
export QX_BBCHECK=1
export QX_G_FROM_L=1
export CUPY_CACHE_IN_MEMORY=1
export QX_SIGMA_BEST_LIVE=1
export QX_SIGMA_BEST_LIVE_STRIDE=10
export QX_REQUIRE_RESTARTABLE=1

for chunk in 1 2 3 4; do
    output="$base/cnt-l48x2-deep150-r4-${chunk}"
    mkdir -p "$output"
    export QX_CONFIG="$base/cnt-l48x2-current/quatrex_config.toml"
    export QX_NPZ="$output/run.npz"
    export QX_SIGMA_INIT="$restart"
    export QX_SAVE_SIGMA="$output/sigma_final.npz"
    export QX_SAVE_SIGMA_BEST="$output/sigma_best.npz"
    echo "START CNT L48 CHUNK ${chunk}/4 FROM ${restart}"
    srun --exclusive --exact \
        --nodes=4 --ntasks=16 --ntasks-per-node=4 \
        --gpus-per-task=1 --cpu-bind=cores bash -c \
        'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; exec python \
            /capstor/scratch/cscs/pfischil/quatrex/phonon/studies/engine/run.py'
    restart="$output/sigma_final.npz"
    if python - "$output/run.npz" <<'PY'
import sys

import numpy as np

with np.load(sys.argv[1]) as run:
    converged = bool(run["converged"])
sys.exit(0 if converged else 1)
PY
    then
        echo "L48 CONVERGED IN CHUNK ${chunk}"
        break
    fi
done
