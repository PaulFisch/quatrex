"""Report-overhaul gap runs (sequential; respects run_one's live-rank guard).

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
