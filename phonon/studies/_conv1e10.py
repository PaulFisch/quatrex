"""(b) Converge cnt33 L_n eta=0 to a GENUINE 1e-10 Sigma^R fixed point.

Pure linear + gentle low-omega mixing. The mixing (1-a)*Sigma_prev + a*Sigma_new
equals Sigma_prev at the fixed point for ANY a -> damping only the IR bins is
PROVABLY fixed-point-preserving (changes the path, not the solution). The IR
marginal mode is a period-2 limit cycle (lambda ~ -1, residual 0.04<->0.54), so
its iteration eigenvalue under low-omega mixing is |1 + a_low*(lambda-1)| =
|1 - 2*a_low|, MINIMIZED near a_low ~ 0.4-0.5 (NOT the very-gentle 0.02, which
gives 0.96 -> glacial). retarded=fft (causal eta=0 partner, REQUIRED),
band_limit, exact conserving fold (bcs=1). QX_DIAG_OMEGA localizes the residual
per-omega. Reaching 1e-10 with the first bin no longer dominating, bubble
balance ~1e-16 and lead balance ~1e-3 CERTIFIES the marginal mode is tamed.

Usage:
  python phonon/studies/_conv1e10.py TAG L NFREQ A_LOW MIX_THZ MIX SIGMA_TOL \
      MAX_ITER CUTOFF NRANKS [MIXING_METHOD]
"""
import os
import shutil
import sys
import time

from phonon.studies import pipeline

OUT = pipeline.OUT / "conv1e10"

# mpi4py's ABI probe dlopens libmpi.so by name -> the openmpi lib must be on
# LD_LIBRARY_PATH for every rank (foreground resolves it via launcher
# discovery, but mpirun-spawned ranks do not). Derive it from `mpirun`.
_mpirun = shutil.which("mpirun")
_OMPI_LIB = (os.path.join(os.path.dirname(os.path.dirname(_mpirun)), "lib")
             if _mpirun else "")


def _ld_env():
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _OMPI_LIB and _OMPI_LIB not in ld.split(":"):
        ld = f"{_OMPI_LIB}:{ld}" if ld else _OMPI_LIB
    return ld


def _active_ranks():
    """Count live solver RANKS (the CPU consumers). Idle leftover prterun
    orchestrators are harmless (0% CPU) and a fresh mpirun runs fine alongside
    them, so we gate only on engine/run.py -- assert_node_idle's prterun match
    refuses launches over those un-killable idle orchestrators."""
    import glob
    n = 0
    me = os.getpid()
    for d in glob.glob("/proc/[0-9]*"):
        pid = int(d.rsplit("/", 1)[-1])
        if pid == me:
            continue
        try:
            with open(f"{d}/cmdline", "rb") as f:
                parts = [p.decode(errors="ignore")
                         for p in f.read().split(b"\x00") if p]
        except OSError:
            continue
        # A real solver rank has argv[0] = python interpreter AND run.py as a
        # script arg. Requiring python as argv[0] excludes shells/pgrep/grep
        # whose command line merely *contains* the string "engine/run.py".
        if (parts and os.path.basename(parts[0]).startswith("python")
                and any("engine/run.py" in p for p in parts)):
            n += 1
    return n


def run_one(tag, L, nf, a_low, mix_thz, mix, sigma_tol, max_iter, cutoff,
            nranks, mixing_method="linear", ir_taper_cells=0.0,
            system="cnt33", fmax=55.0, eta_obc=0.0, band_support_margin=0.0,
            sse_freeze_occupation=0.0, smooth_window=False,
            support_taper_cells=4.0, eta_floor_cells=0.0, broyden_warmup=0,
            rpm_max_subspace=6, jfnk_warmup=10, jfnk_max_krylov=30,
            jfnk_inner_tol=0.1, jfnk_forcing="ew", jfnk_max_newton=60,
            jfnk_eps=1e-7, jfnk_trust=0.5, jfnk_trust_max=0.0,
            jfnk_newton_damp=1.0, jfnk_ptc=0.0,
            diag_spectral=False, ir_subtraction=False, eta_ir_floor=0.0,
            eta_ir_floor_final=0.0, eta_ir_floor_ramp=0, ring_threads=1):
    work = OUT / "work" / tag
    work.mkdir(parents=True, exist_ok=True)
    geom = pipeline.GEOM / f"{system}_L{L}"
    for f in geom.iterdir():
        d = work / f.name
        if f.is_file() and not d.exists():
            os.link(f, d)
    # eta-floor = c*dw (grid-consistent resolvent broadening; raises every
    # sub-grid-sharp pole's linewidth to the grid -> kills the 1/Gamma^2 eta=0
    # Jacobian blow-up at band edges/sharp modes; -> 0 as dw->0). 0 = pure eta=0.
    dw = fmax / (nf - 1)
    eta_val = eta_floor_cells * dw if eta_floor_cells > 0 else 1e-12
    # The exact Bose-pole subtraction REPLACES the omega^2 IR taper.
    if ir_subtraction:
        ir_taper_cells = 0.0
    pipeline.write_config(
        system, work, ncells=L, nfreq=nf, fmax=fmax,
        temperature=300.0, dt=10.0, eta=eta_val, eta_obc=eta_obc,
        retarded="fft", band_limit=True, mixing_method=mixing_method,
        mix=mix, max_iter=max_iter, bcs=1, sse_cutoff=cutoff,
        low_freq_mix_thz=mix_thz, low_freq_mix_factor=a_low,
        sigma_tol=sigma_tol, ir_taper_cells=ir_taper_cells,
        sse_ir_subtraction=ir_subtraction, eta_ir_floor_cells=eta_ir_floor,
        eta_ir_floor_final_cells=eta_ir_floor_final,
        eta_ir_floor_ramp_iterations=eta_ir_floor_ramp,
        band_support_margin=band_support_margin,
        sse_freeze_occupation=sse_freeze_occupation,
        sse_smooth_window=smooth_window,
        support_taper_cells=support_taper_cells,
        broyden_warmup_iters=broyden_warmup, rpm_max_subspace=rpm_max_subspace,
        jfnk_warmup_iters=jfnk_warmup, jfnk_max_krylov=jfnk_max_krylov,
        jfnk_inner_tol=jfnk_inner_tol, jfnk_forcing=jfnk_forcing,
        jfnk_max_newton=jfnk_max_newton, jfnk_eps=jfnk_eps,
        jfnk_trust=jfnk_trust, jfnk_trust_max=jfnk_trust_max,
        jfnk_newton_damp=jfnk_newton_damp, jfnk_ptc=jfnk_ptc)
    # stack_comm_size = nranks (bcs=qcs=1); the solver requires it <= nfreq.
    nranks = min(int(nranks), nf - 1)
    log = OUT / f"{tag}.log"
    npz = OUT / f"{tag}.npz"
    nr = _active_ranks()
    if nr > 0:
        raise SystemExit(f"refuse: {nr} live engine/run.py ranks (real load)")
    t0 = time.perf_counter()
    _env = {"QX_MPI_BIND": "--bind-to core --map-by core", "QX_DIAG_OMEGA": "1",
            "LD_LIBRARY_PATH": _ld_env()}
    if diag_spectral:
        _env["QX_DIAG_SPECTRAL"] = "1"
    rc = pipeline.launch_cell(
        work / "quatrex_config.toml", npz, log, nranks=nranks,
        ring_threads=ring_threads, check_idle=False, env=_env)
    wall = (time.perf_counter() - t0) / 60.0
    tr = pipeline.parse_scba_trace(log)
    res, lead, bub = tr["residual"], tr["lead_balance"], tr["bubble_balance"]
    print(f"\n=== {tag}  rc={rc}  wall={wall:.1f}min  iters={len(res)} ===",
          flush=True)
    if len(res):
        fb = bub[-1] if len(bub) else float("nan")
        print(f"  final resid={res[-1]:.3e}  min resid={res.min():.3e}  "
              f"final lead_bal={lead[-1]:.2e}  final bubble_bal={fb:.2e}",
              flush=True)
        step = max(1, len(res) // 30)
        for i in range(0, len(res), step):
            print(f"   it{i:4d} resid={res[i]:.3e} lead={lead[i]:.2e}",
                  flush=True)
        print(f"   it{len(res)-1:4d} resid={res[-1]:.3e} lead={lead[-1]:.2e}",
              flush=True)
    G = float("nan")
    try:
        lh = pipeline.lead_heat(npz)
        G = lh["mean"]
        print("  leadG=", lh, flush=True)
    except Exception as e:
        print("  leadG: NA", e, flush=True)
    fr = float(res[-1]) if len(res) else float("nan")
    lb = float(lead[-1]) if len(lead) else float("nan")
    conv = "YES" if (len(res) and res[-1] < sigma_tol) else "NO"
    print(f"CONV_DONE {tag} converged={conv} final={fr} G={G:.4f}", flush=True)
    return {"tag": tag, "nf": nf, "C": ir_taper_cells, "G": G,
            "converged": conv, "final_resid": fr, "lead_bal": lb,
            "dw": fmax / (nf - 1), "wreg": ir_taper_cells * fmax / (nf - 1)}


if __name__ == "__main__":
    a = sys.argv[1:]
    mm = a[10] if len(a) > 10 else "linear"
    taper = float(a[11]) if len(a) > 11 else 0.0
    run_one(a[0], int(a[1]), int(a[2]), float(a[3]), float(a[4]), float(a[5]),
            float(a[6]), int(a[7]), float(a[8]), int(a[9]), mm, taper)
