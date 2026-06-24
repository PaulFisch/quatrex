"""Capped d5a eta=0 diagnostic run -- plain-linear mixing (to EXPOSE the natural
fluctuation) with the per-omega/per-iteration spectral capture on
(QX_DIAG_SPECTRAL via diag_spectral=True). Everything else is the canonical d5a
eta=0 recipe (_sinw_taper.py): nf181, fmax=66 (Si-H island on-grid), smooth
window + IR taper + band support + eta-floor=dw.

Run (background, after node hygiene):
  python phonon/studies/_eta0_diag_run.py <max_iter> <tag>
"""
import sys

from phonon.studies import _conv1e10 as cv

NF, FMAX, WREG = 181, 66.0, 2.16
C = WREG * (NF - 1) / FMAX            # ir_taper_cells (omega_reg = WREG THz)

max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 60
tag = sys.argv[2] if len(sys.argv) > 2 else "sinw_d5a_L2_eta0_diag"
# eta_floor_cells: 0 -> LITERAL eta=0 (eta=1e-12); >0 -> grid-consistent floor
# c*dw. Default 0 = literal eta=0 (most faithful to "eta=0", most revealing of
# the fluctuation). The smooth window + IR taper + band support stay on.
eta_floor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
# arg 4: ir_subtraction (1 -> full physical occupation, no omega^2 taper)
ir_sub = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
# arg 5: eta_ir_floor_cells (sub-grid soft-mode broadening stabiliser; 0 = off)
eta_ir_floor = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
# arg 6: mixing method (linear|rre|anderson) ; arg 7: mixing factor
mixing = sys.argv[6] if len(sys.argv) > 6 else "linear"
mixf = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
# arg 8: eta_ir_floor_final (anneal target) ; arg 9: eta_ir_floor_ramp_iterations
floor_final = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
floor_ramp = int(sys.argv[9]) if len(sys.argv) > 9 else 0

r = cv.run_one(
    tag, 2, NF, 0.1, 0.0, mixf, 1e-10, max_iter, 0.0, 128,
    mixing_method=mixing, ir_taper_cells=C,
    system="sinw_d5a", fmax=FMAX, band_support_margin=1.5,
    sse_freeze_occupation=1e-3, smooth_window=True,
    support_taper_cells=4.0, eta_floor_cells=eta_floor,
    diag_spectral=True, ir_subtraction=ir_sub, eta_ir_floor=eta_ir_floor,
    eta_ir_floor_final=floor_final, eta_ir_floor_ramp=floor_ramp,
)
print("DIAG_RUN_DONE", r, flush=True)
