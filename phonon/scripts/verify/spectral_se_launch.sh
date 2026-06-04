#!/usr/bin/env bash
# Fan out the spectral-function self-energy grid across the node.
# structures x temperatures x self-energy modes, each a single-core SCBA.
set -u
cd /usr/scratch/mont-fort11/pfischill/quatrex
OUT=/tmp/claude/spectral_se
mkdir -p "$OUT"
WORKER=phonon/scripts/verify/spectral_se_worker.py
PAR=${PAR:-24}

JOBS="$OUT/jobs.txt"
: > "$JOBS"
for s in cnt33 d5a; do
  for T in 30 100 300 600; do
    for m in bubble_half bubble_fft scp_fft; do
      echo "$s $T $m" >> "$JOBS"
    done
  done
done
echo "launching $(wc -l < "$JOBS") jobs at concurrency $PAR"

run_one() {
  s=$1; T=$2; m=$3
  out="$OUT/${s}_T${T}_${m}.npz"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 QUATREX_PHPH_THREADS=4 \
    QUATREX_PHPH_MEMORY_GB=12 \
    python -u phonon/scripts/verify/spectral_se_worker.py \
      --struct "$s" --temp "$T" --mode "$m" --out "$out" --nfreq 81
}
export -f run_one
export OUT

xargs -a "$JOBS" -P "$PAR" -L1 bash -c 'run_one "$@"' _
echo "ALL_DONE"
