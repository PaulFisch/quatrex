#!/bin/bash
# ASR projection of the MoS2 film 3-phonon vertices -- daint scratch tree.
#
# Runs _asr_project_film.py project on the production SCP build
#   cluster/mos2f3scp/{fc3_blocks.hdf5,qfold_vertices.npz}
# writing the projected pair + logs into cluster/mos2f3-asr/.
#
# DRAFT -- do not launch automatically. Intended invocation (single token,
# from a login/compute shell on daint):
#
#   bash /capstor/scratch/cscs/pfischil/quatrex/phonon/studies/_asr_project_film_job.sh
#
# Resources: single node, CPU-only, ~4 GB RAM (the ~1 GB npz is inverse-
# folded per-triplet), a few minutes. No GPU, no MPI.
# The projection aborts (nonzero exit, nothing written) if the inverse-fold
# round-trip exceeds 1e-12 or the correction exceeds --max-corr 0.05.

set -euo pipefail

REPO=/capstor/scratch/cscs/pfischil/quatrex
SRC=$REPO/cluster/mos2f3scp
OUT=$REPO/cluster/mos2f3-asr
PY=${QX_PYTHON:-python3}

export PYTHONPATH=$REPO/src:$REPO/phonon:$REPO

mkdir -p $OUT

echo asr-project-film: selftest
$PY $REPO/phonon/studies/_asr_project_film.py selftest --workdir $OUT/selftest_tmp > $OUT/selftest.log 2>&1
tail -n 3 $OUT/selftest.log

echo asr-project-film: pre-audit
$PY $REPO/phonon/studies/_asr_project_film.py audit --fc3 $SRC/fc3_blocks.hdf5 --qfold $SRC/qfold_vertices.npz > $OUT/audit_pre.log 2>&1

echo asr-project-film: project
$PY $REPO/phonon/studies/_asr_project_film.py project --fc3 $SRC/fc3_blocks.hdf5 --qfold $SRC/qfold_vertices.npz --out $OUT > $OUT/project.log 2>&1

grep -E "correction|round-trip|converged|interior: max|edge: max|ABORT" $OUT/project.log
echo asr-project-film: done, outputs in $OUT
