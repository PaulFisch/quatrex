#!/usr/bin/env bash
# Retry the static-SE (tadpole / loop) runs with the regularised mean_displacement
# fix + linear mixing on the soft-mode wires (Anderson diverges there). Loop is
# only available for d5a (the only structure with FC4).
set -u
cd /usr/scratch/mont-fort11/pfischill/quatrex
OUT=/tmp/claude/spectral_se; mkdir -p "$OUT"
W=phonon/scripts/verify/spectral_se_worker.py

# struct temp mode solver nfreq
JOBS=$(cat <<'EOF'
d5a 300 scp_fft linear 61
d5a 300 loop_fft linear 61
d5a 300 loop_tadpole_fft linear 61
cnt33 300 scp_fft linear 81
cnt33 600 scp_fft linear 81
d11a 300 scp_fft linear 31
EOF
)
echo "$JOBS" | while read s T m solver nf; do
  [ -z "$s" ] && continue
  out="$OUT/retry_${s}_T${T}_${m}.npz"
  echo "LAUNCH $s $T $m solver=$solver nfreq=$nf"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 QUATREX_PHPH_THREADS=4 \
    QUATREX_PHPH_MEMORY_GB=30 \
    python -u "$W" --struct "$s" --temp "$T" --mode "$m" --solver "$solver" \
      --mixing 0.3 --max-iter 200 --nfreq "$nf" --out "$out" 2>&1 \
      | grep -vE "memory cap" &
done
wait
echo "RETRY_ALL_DONE"
