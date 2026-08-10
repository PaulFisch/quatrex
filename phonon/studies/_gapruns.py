"""Report-overhaul gap runs (sequential; respects run_one's live-rank guard).

1) d11a L2 eta=0 soft-mode PROBE, relaunched at 8 ranks x 16 ring threads.
   The 128-rank/ring-1 launch memory-thrashed (iteration-0 bubble 98379 s vs
   ~9 min/iter for the LARGER cnt80 L3): 128 x per-rank bubble workspace on the
   dense 315 MB d11a FC3 exceeds RAM. The d11a block size (135) is squarely in
   the ring-pool's validated regime, so few ranks x many threads is the right
   shape. Same physics recipe as the first probe (gentle linear + smooth
   window + grid eta-floor).

2) cnt80 L3 eta=0 FULL run with the probe-validated soft-mode kit (the 60-iter
   probe descended 1.0 -> 3.8e-3, best lead-cons 1.8e-3): same recipe, full
   iteration budget.

Run (background, after node hygiene):
  nohup python phonon/studies/_gapruns.py > phonon/studies/out/conv1e10/_gapruns.log 2>&1 &
"""
from phonon.studies import _conv1e10 as cv


def d11a_probe2():
    nf, fmax = 161, 18.0
    return cv.run_one(
        "d11a_L2_eta0_probe2", 2, nf, 0.1, 0.0, 0.05, 1e-10, 80, 0.0, 8,
        mixing_method="linear", system="sinw_d11a", fmax=fmax,
        band_support_margin=1.5, smooth_window=True, support_taper_cells=4.0,
        eta_floor_cells=1.0, ring_threads=16)


def cnt80_L3_full():
    nf, fmax = 121, 50.0
    C = 2.0 * (nf - 1) / fmax  # overridden by ir_subtraction (kept for parity)
    return cv.run_one(
        "cnt80_L3_eta0_full", 3, nf, 0.1, 0.0, 0.1, 1e-10, 300, 0.0, 120,
        mixing_method="jfnk", ir_taper_cells=C, system="cnt80", fmax=fmax,
        band_support_margin=1.5, sse_freeze_occupation=1e-3, smooth_window=True,
        support_taper_cells=4.0, ir_subtraction=True, eta_ir_floor=2.0,
        jfnk_warmup=50, jfnk_trust=0.05, jfnk_trust_max=0.4, jfnk_max_krylov=50)


print("GAPRUN d11a probe2 ->", d11a_probe2(), flush=True)
print("GAPRUN cnt80 L3 full ->", cnt80_L3_full(), flush=True)
