"""Short eta=0 soft-mode convergence PROBES for the cells whose default --eta0
recipe (plain linear beta=0.2) diverged: cnt80 L3 (ultra-soft 0.0014 THz twist)
and d11a L2 (band-edge marginal mode). Goal: confirm a DESCENDING basin (vs the
blow-up) over ~60-80 iters before committing the multi-day full runs. Reuses the
d5a-proven soft-mode kit via _conv1e10.run_one (ir-subtraction + eta_ir_floor +
smooth window + trust-grow JFNK for the ultra-soft tube; gentler linear + window
+ grid eta-floor for the band-edge-marginal wire).

Run (background, after node hygiene; full-node 128-rank stack-parallel):
  python phonon/studies/_softmode_probe.py {cnt80|d11a|all}
"""
import sys

from phonon.studies import _conv1e10 as cv

which = sys.argv[1] if len(sys.argv) > 1 else "all"


def cnt80_probe():
    # ultra-soft twist 0.0014 THz -> the d5a kit: IR-subtraction + eta_ir_floor=2
    # + smooth window + trust-grow JFNK (warmup 50 picard then Newton).
    nf, fmax = 121, 50.0
    C = 2.0 * (nf - 1) / fmax  # ir_taper_cells (overridden by ir_subtraction)
    return cv.run_one(
        "cnt80_L3_eta0_probe", 3, nf, 0.1, 0.0, 0.1, 1e-10, 60, 0.0, 120,
        mixing_method="jfnk", ir_taper_cells=C, system="cnt80", fmax=fmax,
        band_support_margin=1.5, sse_freeze_occupation=1e-3, smooth_window=True,
        support_taper_cells=4.0, ir_subtraction=True, eta_ir_floor=2.0,
        jfnk_warmup=50, jfnk_trust=0.05, jfnk_trust_max=0.4, jfnk_max_krylov=50)


def d11a_probe():
    # band-edge marginal mode (lowest 1.07 THz, NOT IR-soft): the default beta=0.2
    # got to resid 0.105 then blew up. Gentler linear + smooth window + grid
    # eta-floor (c*dw raises the band-edge pole linewidth to the grid).
    nf, fmax = 161, 18.0
    return cv.run_one(
        "d11a_L2_eta0_probe", 2, nf, 0.1, 0.0, 0.05, 1e-10, 80, 0.0, 128,
        mixing_method="linear", system="sinw_d11a", fmax=fmax,
        band_support_margin=1.5, smooth_window=True, support_taper_cells=4.0,
        eta_floor_cells=1.0)


if which in ("cnt80", "all"):
    print("PROBE_RESULT", cnt80_probe(), flush=True)
if which in ("d11a", "all"):
    print("PROBE_RESULT", d11a_probe(), flush=True)
