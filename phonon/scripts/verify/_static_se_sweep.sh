#!/bin/bash
# Regenerate the static-correction magnitude study snapshots (the originals in
# scripts/out/snapshots were purged). 40 cheap dense points (n_slabs=1, nf=61);
# 4 concurrent workers x 8 BLAS threads, sized to coexist with a running
# 8x16 production probe. Snapshots -> phonon/scripts/out/snapshots/study_*.npz
set -u
cd "$(dirname "$0")/../../.."   # repo root
OUT=phonon/scripts/out/snapshots
mkdir -p "$OUT"
CNT_FC4=phonon/configs/cnt/fc3_hiphive_cnt33_fc4_vasp/fc3.hdf5

run() {  # struct T mode extra...
  local s=$1 t=$2 m=$3; shift 3
  local f="$OUT/study_${s}_T${t}_${m}.npz"
  [ -f "$f" ] && { echo "skip $f"; return; }
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=1 QUATREX_PHPH_THREADS=2 QUATREX_PHPH_MEMORY_GB=30 \
  python phonon/scripts/verify/static_se_study.py \
    --struct "$s" --temp "$t" --mode "$m" --out "$f" "$@" \
    > "$OUT/study_${s}_T${t}_${m}.log" 2>&1
  echo "done $f rc=$?"
}

njobs() { jobs -rp | wc -l; }
throttle() { while [ "$(njobs)" -ge 4 ]; do sleep 20; done; }

for T in 100 200 300 400 500 600; do
  for M in bubble loop tadpole loop_tadpole; do
    throttle; run cnt33 "$T" "$M" --fc3 "$CNT_FC4" &
  done
done
for T in 30 100 300 600; do
  for M in bubble loop tadpole loop_tadpole; do
    throttle; run d5a "$T" "$M" &
  done
done
wait
echo "SWEEP COMPLETE $(ls "$OUT"/study_*.npz 2>/dev/null | wc -l) snapshots"
