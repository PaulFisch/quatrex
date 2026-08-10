#!/bin/bash
# daint debug job: compile (sm_90) + parity check + bench for the fused
# three-phonon ring kernel. Launched via phonon/scripts/daint.py; the
# sbatch wrapper provides the uenv (nvcc), venv python and PYTHONPATH.
set -euo pipefail
STUDIES=/capstor/scratch/cscs/pfischil/quatrex/phonon/studies
cd "$STUDIES"
echo "== toolchain =="
which nvcc && nvcc --version | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "== build (sm_90) =="
python _fused_ring_bench.py build --arch sm_90
echo "== check =="
python _fused_ring_bench.py check
echo "== bench =="
python _fused_ring_bench.py bench --json "${SLURM_SUBMIT_DIR:-$PWD}/bench.json"
echo "== done =="
