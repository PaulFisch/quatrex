#!/usr/bin/env bash
# Fan out the spectral-function self-energy grid across the node.
# structures x temperatures x self-energy modes, each a single-core SCBA.
set -u
cd /usr/scratch/mont-fort11/pfischill/quatrex
OUT=/tmp/claude/spectral_se
mkdir -p "$OUT"
PAR=${PAR:-24}
STRUCTS=${STRUCTS:-"cnt33 d5a"}
TEMPS=${TEMPS:-"30 100 300 600"}
MODES=${MODES:-"bubble_half bubble_fft scp_fft"}
NFREQ=${NFREQ:-81}
MEMGB=${MEMGB:-12}
TAG=${TAG:-jobs}

JOBS="$OUT/$TAG.txt"
: > "$JOBS"
for s in $STRUCTS; do
  for T in $TEMPS; do
    for m in $MODES; do
      echo "$s $T $m" >> "$JOBS"
    done
  done
done
echo "launching $(wc -l < "$JOBS") jobs at concurrency $PAR (nfreq=$NFREQ memgb=$MEMGB)"

export OUT NFREQ MEMGB
run_one() {
  s=$1; T=$2; m=$3
  out="$OUT/${s}_T${T}_${m}.npz"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 QUATREX_PHPH_THREADS=4 \
    QUATREX_PHPH_MEMORY_GB="$MEMGB" \
    python -u phonon/scripts/verify/spectral_se_worker.py \
      --struct "$s" --temp "$T" --mode "$m" --out "$out" --nfreq "$NFREQ"
}
export -f run_one

xargs -a "$JOBS" -P "$PAR" -L1 bash -c 'run_one "$@"' _
echo "ALL_DONE"
