"""Ballistic (Landauer, no SCBA) d5a current: omega^2 taper vs IR occupation
subtraction. Ballistic isolates the occupation treatment from the SCBA
convergence -- the taper multiplies the clean current by t(omega)=omega^2/
(omega^2+omega_reg^2), so the low-omega drop-to-zero shows up directly.

Reuses the two config work dirs already written by the anharmonic smokes:
  sinw_d5a_L2_eta0_diag   -> ir_taper_cells=5.89 (omega_reg=2.16 THz)  [taper]
  sinw_d5a_L2_irsub2_smoke-> sse_ir_subtraction=true, ir_taper_cells=0  [occ-sub]

Run (background): python phonon/studies/_ball_compare.py
"""
from phonon.studies import pipeline, _conv1e10 as cv

env = {"QX_MPI_BIND": "--bind-to core --map-by core",
       "LD_LIBRARY_PATH": cv._ld_env()}

for tag in ("sinw_d5a_L2_eta0_diag", "sinw_d5a_L2_irsub2_smoke"):
    cfg = cv.OUT / "work" / tag / "quatrex_config.toml"
    npz = cv.OUT / f"{tag}_ball.npz"
    log = cv.OUT / f"{tag}_ball.log"
    rc = pipeline.launch_cell(cfg, npz, log, nranks=128, ring_threads=1,
                              ballistic=True, check_idle=False, env=env)
    print(f"BALL {tag} rc={rc}", flush=True)
print("BALL_DONE", flush=True)
