#!/usr/bin/env bash
# Fan out the static-correction magnitude study: (T x mode) for one structure.
set -u
cd /usr/scratch/mont-fort11/pfischill/quatrex
OUT=/tmp/claude/se_study; mkdir -p "$OUT"
STRUCT=${STRUCT:?set STRUCT}
FC3=${FC3:-}
NFREQ=${NFREQ:-61}
TEMPS=${TEMPS:-"30 100 300 600"}
MODES=${MODES:-"bubble loop tadpole loop_tadpole"}
PAR=${PAR:-12}

JOBS="$OUT/jobs_${STRUCT}.txt"; : > "$JOBS"
for T in $TEMPS; do for m in $MODES; do echo "$T $m" >> "$JOBS"; done; done
echo "STRUCT=$STRUCT FC3=${FC3:-default} nfreq=$NFREQ : $(wc -l < "$JOBS") jobs, PAR=$PAR"

export OUT STRUCT FC3 NFREQ
run_one() {
  T=$1; m=$2
  fc3arg=""; [ -n "$FC3" ] && fc3arg="--fc3 $FC3"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 QUATREX_PHPH_THREADS=4 \
    QUATREX_PHPH_MEMORY_GB=18 \
    python -u phonon/scripts/verify/static_se_study.py \
      --struct "$STRUCT" --temp "$T" --mode "$m" --solver linear \
      --nfreq "$NFREQ" $fc3arg --out "$OUT/study_${STRUCT}_T${T}_${m}.npz"
}
export -f run_one
xargs -a "$JOBS" -P "$PAR" -L1 bash -c 'run_one "$@"' _
echo "STUDY_${STRUCT}_DONE"
