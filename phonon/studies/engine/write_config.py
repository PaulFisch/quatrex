"""Write a PRODUCTION quatrex phonon-transport config TOML.

Committed, parameterized port of the /tmp config writers, emitting the
converged anharmonic-SCBA recipe (``retarded_method="half"``, gentle linear
mixing, heat-flow convergence) plus a
``[compute]`` / ``[compute.comm]`` block so a distributed run honors the
rank grid (``block_comm_size`` x ``q_comm_size``) from the config -- this is
what the scaling sweep drives.

Asserts the comm-grid constraints: ``block*q`` must divide the rank count
(checked at launch), ``block<=L`` (CNT band halo), and the film keeps
``block==1`` (the coupled-q SSE forbids nq>1 with block>1).

Usage (CNT):
    python write_config.py --system cnt33 --work DIR -L 4 \
        --nfreq 161 --fmax 55 [--bcs 1 --qcs 1 --numba-threads 1]
Usage (film):
    python write_config.py --system sifilm --work DIR --nslabs 5 --nk 8 \
        --tdir x --shift <kshift> --nfreq 121 --fmax 15 [--qcs 8]
"""
import argparse
from pathlib import Path


def _orbital_block(work):
    """``[device.num_orbitals_per_atom]`` with ``X = 3`` per species (3 phonon
    DOF/atom), species read IN ORDER from the builder's ``structure.xyz``
    (extended XYZ: line1 count, line2 Lattice header, then ``Symbol x y z``).
    CNT -> ``C = 3`` (unchanged); SiNW -> ``Si = 3`` + ``H = 3``. The builder
    always runs before this writer, so structure.xyz exists and is exactly the
    cell the solver consumes."""
    xyz = Path(work) / "structure.xyz"
    syms = []
    with open(xyz) as f:
        f.readline()  # atom count
        f.readline()  # Lattice header
        for line in f:
            tok = line.split()
            if tok:
                syms.append(tok[0])
    body = "\n".join(f"{s} = 3" for s in dict.fromkeys(syms))
    return f"[device.num_orbitals_per_atom]\n{body}"


def tail_block(a):
    """The shared [outputs] + [compute] + [compute.comm] tail."""
    tail = f"""
[outputs]
save_profiling_results = {str(a.profile).lower()}
profiling_save_format = "json"

[compute]
numba_num_threads = {a.numba_threads}
blas_num_threads = {a.blas_threads}

[compute.comm]
block_comm_size = {a.bcs}
q_comm_size = {a.qcs}
"""
    if getattr(a, "comm_backend", None):
        # One backend for every per-op comm selector (GPU runs); unset
        # keeps the emitted TOML byte-identical to the legacy output.
        ops = ["all_to_all", "all_gather", "all_reduce", "bcast"]
        lines = [f'{axis}_{op} = "{a.comm_backend}"'
                 for axis in ("block", "stack", "q") for op in ops]
        lines += [f'{axis}_send_recv = "{a.comm_backend}"'
                  for axis in ("block", "stack")]
        tail += "\n".join(lines) + "\n"
    return tail


def cnt_config(a):
    if a.bcs > a.ncells:
        raise SystemExit(f"block_comm_size {a.bcs} > num_transport_cells {a.ncells} "
                         "(each block-rank needs >=1 BTD block)")
    if a.bcs > 1 and a.ncells < 2 * a.bcs:
        # Runtime enforces ncells >= (sse_g_band+1)*bcs (band halo +
        # distributed RGF off-diagonal post-pass); warn early for the
        # weakest case g_band=1.
        raise SystemExit(f"block_comm_size {a.bcs} needs >= 2 blocks per "
                         f"block-rank (ncells >= {2 * a.bcs}); with "
                         "sse_g_band=g the runtime requires ncells >= "
                         "(g+1)*bcs.")
    if a.qcs != 1:
        raise SystemExit("Gamma-only system (k==1); q_comm_size must be 1")
    orb = _orbital_block(a.work)
    return f"""simulation_dir = "{a.work}"
input_dir = "{a.work}"
output_dir = "{a.work}/out"
formalism = "negf"
simulation_type = "phonon"

[device]
transport_direction = "{a.tdir}"
construct_from_unit_cell = true
num_transport_cells = {a.ncells}
neighbor_cell_cutoff = [0, 0, 1]
kpoint_grid = [1, 1, 1]
kpoint_shift = [0, 0, 0]
{orb}

[scba]
max_iterations = {a.max_iter}
min_iterations = 3
mixing_factor = {a.mix}
mixing_method = "{a.mixing_method}"
anderson_depth = {a.anderson_depth}
anderson_period = {a.anderson_period}
anderson_warmup_iters = {a.anderson_warmup}
anderson_restart = {a.anderson_restart}
anderson_ridge = {a.anderson_ridge}
phonon = true

[scba.experimental_mixer]
rre_cycle = {a.rre_cycle}
broyden_warmup_iters = {a.broyden_warmup}
broyden_ridge = {a.broyden_ridge}
broyden_trust = {a.broyden_trust}
rpm_max_subspace = {a.rpm_max_subspace}
jfnk_warmup_iters = {a.jfnk_warmup}
jfnk_max_krylov = {a.jfnk_max_krylov}
jfnk_inner_tol = {a.jfnk_inner_tol}
jfnk_forcing = "{a.jfnk_forcing}"
jfnk_max_newton = {a.jfnk_max_newton}
jfnk_eps = {a.jfnk_eps}
jfnk_trust = {a.jfnk_trust}
jfnk_trust_max = {a.jfnk_trust_max}
jfnk_newton_damp = {a.jfnk_newton_damp}
jfnk_ptc = {a.jfnk_ptc}

[electron]
energy_window_min = {a.emin}
energy_window_max = {a.fmax}
energy_window_num = {a.nfreq}

[phonon]
eta = {a.eta}
eta_obc = {a.eta_obc}
left_temperature = {a.tL}
right_temperature = {a.tR}
model = "negf"
fc3_path = "{a.work}/fc3_blocks.hdf5"
retarded_method = "{a.retarded}"
scp_tadpole = {str(a.tadpole).lower()}
sse_ramp_iterations = {a.ramp}
sse_vertex_scale = {a.vertex_scale}
eta_ramp_iterations = {a.eta_ramp_iters}
eta_final = {a.eta_final}
eta_obc_ramp_iterations = {a.eta_obc_ramp_iters}
eta_obc_final = {a.eta_obc_final}
eta_ir_floor_cells = {a.eta_ir_floor_cells}
eta_ir_floor_final_cells = {a.eta_ir_floor_final_cells}
eta_ir_floor_ramp_iterations = {a.eta_ir_floor_ramp_iterations}
low_freq_mixing_thz = {a.low_freq_mix_thz}
low_freq_mixing_factor = {a.low_freq_mix_factor}
sigma_convergence_tol = {a.sigma_tol}
heat_flow_conservation_tol = 1e-2
frequency_grid = "{a.freq_grid}"
sse_aux_grid_dw_thz = {a.aux_dw}
sse_aux_grid_fmax_thz = {a.aux_fmax}
[phonon.solver]
compute_current = true
max_batch_size = {a.max_batch}
algorithm = "{a.algorithm}"
[phonon.obc]
algorithm = "{a.obc}"
nevp_solver = "full"
block_sections = 1
{tail_block(a)}"""  # noqa: E501  (left/right temperature set below from --temperature/--dt)


def _vertex_source(a):
    """[phonon] coupled-q vertex source: dense qfold (default) or the
    tensor-decomposed factors (mutually exclusive in the solver config)."""
    if getattr(a, "decomposed_vertices", None):
        return (f'decomposed_vertices_path = "{a.decomposed_vertices}"\n'
                f"sse_vertex_rank = {a.vertex_rank}")
    return f'qfold_path = "{a.work}/qfold_vertices.npz"'


def film_config(a):
    if a.bcs != 1:
        raise SystemExit("film (k>1 coupled-q) requires block_comm_size==1 "
                         "(the SSE forbids nq>1 with block>1); scale on q x stack")
    tidx = "xyz".index(a.tdir)
    kg = [1, 1, 1]
    ks = [0.0, 0.0, 0.0]
    for ax in range(3):
        if ax != tidx:
            kg[ax] = a.nk
            ks[ax] = float(a.shift)
    ncc = [1 if ax == tidx else a.nk // 2 for ax in range(3)]
    if a.qcs > a.nk * a.nk:
        raise SystemExit(f"q_comm_size {a.qcs} > nk*nk {a.nk * a.nk}")
    return f"""simulation_dir = "{a.work}"
input_dir = "{a.work}"
output_dir = "{a.work}/out"
formalism = "negf"
simulation_type = "phonon"

[device]
transport_direction = "{a.tdir}"
construct_from_unit_cell = true
num_transport_cells = {a.nslabs}
neighbor_cell_cutoff = [{ncc[0]}, {ncc[1]}, {ncc[2]}]
kpoint_grid = [{kg[0]}, {kg[1]}, {kg[2]}]
kpoint_shift = [{ks[0]}, {ks[1]}, {ks[2]}]
[device.num_orbitals_per_atom]
Si = 3

[scba]
max_iterations = {a.max_iter}
min_iterations = 3
mixing_factor = {a.mix}
mixing_method = "{a.mixing_method}"
anderson_depth = {a.anderson_depth}
anderson_period = {a.anderson_period}
anderson_warmup_iters = {a.anderson_warmup}
anderson_restart = {a.anderson_restart}
anderson_ridge = {a.anderson_ridge}
phonon = true

[scba.experimental_mixer]
rre_cycle = {a.rre_cycle}
broyden_warmup_iters = {a.broyden_warmup}
broyden_ridge = {a.broyden_ridge}
broyden_trust = {a.broyden_trust}
rpm_max_subspace = {a.rpm_max_subspace}
jfnk_warmup_iters = {a.jfnk_warmup}
jfnk_max_krylov = {a.jfnk_max_krylov}
jfnk_inner_tol = {a.jfnk_inner_tol}
jfnk_forcing = "{a.jfnk_forcing}"
jfnk_max_newton = {a.jfnk_max_newton}
jfnk_eps = {a.jfnk_eps}
jfnk_trust = {a.jfnk_trust}
jfnk_trust_max = {a.jfnk_trust_max}
jfnk_newton_damp = {a.jfnk_newton_damp}
jfnk_ptc = {a.jfnk_ptc}

[electron]
energy_window_min = {a.emin}
energy_window_max = {a.fmax}
energy_window_num = {a.nfreq}

[phonon]
eta = {a.eta}
eta_obc = {a.eta_obc}
left_temperature = {a.tL}
right_temperature = {a.tR}
model = "negf"
fc3_path = "{a.work}/fc3_blocks.hdf5"
{_vertex_source(a)}
retarded_method = "{a.retarded}"
scp_tadpole = {str(a.tadpole).lower()}
sse_ramp_iterations = {a.ramp}
sse_vertex_scale = {a.vertex_scale}
eta_ramp_iterations = {a.eta_ramp_iters}
eta_final = {a.eta_final}
eta_obc_ramp_iterations = {a.eta_obc_ramp_iters}
eta_obc_final = {a.eta_obc_final}
eta_ir_floor_cells = {a.eta_ir_floor_cells}
eta_ir_floor_final_cells = {a.eta_ir_floor_final_cells}
eta_ir_floor_ramp_iterations = {a.eta_ir_floor_ramp_iterations}
low_freq_mixing_thz = {a.low_freq_mix_thz}
low_freq_mixing_factor = {a.low_freq_mix_factor}
sigma_convergence_tol = {a.sigma_tol}
heat_flow_conservation_tol = 1e-2
frequency_grid = "{a.freq_grid}"
sse_aux_grid_dw_thz = {a.aux_dw}
sse_aux_grid_fmax_thz = {a.aux_fmax}
[phonon.solver]
compute_current = true
max_batch_size = {a.max_batch}
algorithm = "{a.algorithm}"
[phonon.obc]
algorithm = "{a.obc}"
nevp_solver = "full"
block_sections = 1
{tail_block(a)}"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", required=True,
                   choices=["cnt33", "cnt80", "sinw_d5a", "sinw_d11a", "srtio3", "sifilm"])
    p.add_argument("--work", required=True)
    p.add_argument("-L", "--ncells", type=int, default=2)
    p.add_argument("--nslabs", type=int, default=5)
    p.add_argument("--nk", type=int, default=8)
    p.add_argument("--tdir", default=None)
    p.add_argument("--shift", type=float, default=0.0, help="film kpoint_shift (from kshift.npy)")
    p.add_argument("--decomposed-vertices", default=None,
                   help="path to the tensor-decomposed vertex factors "
                        "(.npz from vertex_factors.save_decomposed); emits "
                        "decomposed_vertices_path INSTEAD of qfold_path")
    p.add_argument("--vertex-rank", type=int, default=0,
                   help="sse_vertex_rank truncation (0 = full stored rank); "
                        "only with --decomposed-vertices")
    p.add_argument("--temperature", type=float, default=300.0,
                   help="mean device temperature T (K); leads at T +/- dt/2")
    p.add_argument("--dt", type=float, default=10.0, help="lead temperature drop (K)")
    p.add_argument("--eta", type=float, default=None,
                   help="THz [0 = NO artificial broadening, the default -- do NOT add smearing, see CLAUDE.md]")
    p.add_argument("--eta-obc", type=float, default=0.0)
    p.add_argument("--emin", type=float, default=0.0)
    p.add_argument("--nfreq", type=int, default=None)
    p.add_argument("--fmax", type=float, default=None)
    p.add_argument("--freq-grid", dest="freq_grid", default="window",
                   choices=["window", "file"],
                   help="phonon frequency grid source: the uniform window "
                        "(legacy) or phonon_energies.npy verbatim (may be "
                        "non-uniform; pair with --aux-dw). With 'file' an "
                        "EXISTING phonon_energies.npy (e.g. from "
                        "make_grid.py) is kept, not overwritten.")
    p.add_argument("--aux-dw", dest="aux_dw", type=float, default=0.0,
                   help="auxiliary uniform bubble-grid spacing (THz); "
                        "0 = legacy (bubble on the primary grid)")
    p.add_argument("--aux-fmax", dest="aux_fmax", type=float, default=0.0,
                   help="auxiliary bubble-grid upper edge (THz), set >= "
                        "2*omega_max for a support-complete KK; 0 = span "
                        "the primary grid")
    p.add_argument("--mix", type=float, default=0.1)
    p.add_argument("--mixing-method", default="linear",
                   choices=["linear", "anderson", "broyden", "rre", "rpm", "jfnk"],
                   help="linear | anderson | broyden | rre | rpm | jfnk. broyden "
                        "(type-I good Broyden root finder), rpm (Recursive "
                        "Projection Method) and jfnk (Jacobian-free Newton-Krylov, "
                        "GMRES on the matrix-free Newton system -- lands the "
                        "STRONGLY-unstable d5a eta=0 saddle where rpm fails) LAND "
                        "the UNSTABLE eta=0 fixed point that damped/Anderson/rre "
                        "mixing cannot reach")
    p.add_argument("--rre-cycle", type=int, default=8,
                   help="rre: restart cycle length (iterates per extrapolation)")
    p.add_argument("--broyden-warmup-iters", dest="broyden_warmup", type=int,
                   default=0,
                   help="broyden/rpm: damped-LINEAR mixing for the first N iters "
                        "(park in the bounded limit-cycle neighbourhood) then "
                        "engage the quasi-Newton/projection root finder")
    p.add_argument("--broyden-ridge", type=float, default=1e-8,
                   help="broyden/rpm: tiny Tikhonov ridge on the small multisecant/"
                        "restricted-Jacobian solve (keep small so it does not damp "
                        "the marginal-mode Newton correction)")
    p.add_argument("--broyden-trust", type=float, default=0.3,
                   help="broyden/rpm: trust-region step cap -- limit "
                        "||Sigma_new-Sigma|| to broyden_trust*||Sigma|| (tames the "
                        "far-from-root quasi-Newton overshoot; 0 disables)")
    p.add_argument("--rpm-max-subspace", dest="rpm_max_subspace", type=int,
                   default=6,
                   help="rpm: cap on the unstable-subspace dimension k (Newton on "
                        "k modes, Picard on the complement; band-edge pair -> k~2)")
    # --- jfnk (Jacobian-free Newton-Krylov) ------------------------------------
    p.add_argument("--jfnk-warmup-iters", dest="jfnk_warmup", type=int, default=10,
                   help="jfnk: damped-linear steps before engaging Newton-Krylov "
                        "(basin capture)")
    p.add_argument("--jfnk-max-krylov", dest="jfnk_max_krylov", type=int,
                   default=30, help="jfnk: max GMRES dim per Newton step = max map "
                                    "evals per Newton step (n_unstable + a few)")
    p.add_argument("--jfnk-inner-tol", dest="jfnk_inner_tol", type=float,
                   default=0.1, help="jfnk: base relative GMRES inner tolerance")
    p.add_argument("--jfnk-forcing", dest="jfnk_forcing", default="ew",
                   choices=["ew", "fixed"],
                   help="jfnk: inner-tol forcing (ew=Eisenstat-Walker | fixed)")
    p.add_argument("--jfnk-max-newton", dest="jfnk_max_newton", type=int,
                   default=60, help="jfnk: cap on outer Newton steps")
    p.add_argument("--jfnk-eps", dest="jfnk_eps", type=float, default=1e-7,
                   help="jfnk: relative FD step for J*v, eps*(1+||Sigma||)")
    p.add_argument("--jfnk-trust", dest="jfnk_trust", type=float, default=0.5,
                   help="jfnk: INITIAL trust-region cap ||delta||<=trust*||Sigma|| (0 off)")
    p.add_argument("--jfnk-trust-max", dest="jfnk_trust_max", type=float,
                   default=0.0, help="jfnk: MAX trust radius the adaptive growth "
                        "may reach (<=0 -> = jfnk_trust, no growth). Lets the radius "
                        "breathe up as the residual descends (d5a marginal-mode crawl)")
    p.add_argument("--jfnk-newton-damp", dest="jfnk_newton_damp", type=float,
                   default=1.0, help="jfnk: damping of the trust-capped Newton step")
    p.add_argument("--jfnk-ptc", dest="jfnk_ptc", type=float, default=0.0,
                   help="jfnk: pseudo-transient/LM shift mu0 for (J+mu I)delta=-R, "
                        "mu=ptc*||R||/||R0|| -> 0 at root (lifts marginal modes off "
                        "the origin so inner GMRES does not stall; 0=pure Newton)")
    p.add_argument("--anderson-depth", type=int, default=5)
    p.add_argument("--anderson-period", type=int, default=1,
                   help="periodic-Pulay stride: extrapolate every Nth iter, "
                        "damped linear between (breaks marginal-mode limit cycles)")
    p.add_argument("--anderson-warmup-iters", dest="anderson_warmup", type=int,
                   default=0,
                   help="run LINEAR mixing for the first N SCBA iters then switch "
                        "to Anderson (cold-start warmup for the eta=0 causal map)")
    p.add_argument("--anderson-restart", type=int, default=0,
                   help="forget Anderson history every N steps (escape the "
                        "marginal-mode limit cycle at eta=0)")
    p.add_argument("--anderson-ridge", type=float, default=0.0,
                   help="Tikhonov regularisation of the Anderson lstsq "
                        "(suppress the overshoot spikes)")
    p.add_argument("--max-iter", type=int, default=50,
                   help="SCBA cap; the conductance (best-iterate) converges well "
                        "before the Sigma residual (F30), so 50 bounds wall-time")
    p.add_argument("--retarded", default="half", choices=["half", "fft"])
    p.add_argument("--eta-ir-floor-cells", dest="eta_ir_floor_cells", type=float,
                   default=0.0,
                   help="sub-grid soft-mode broadening floor (grid cells); a "
                        "DC-concentrated constant broadening that stabilises the "
                        "eta=0 SCBA without crushing the low-omega current")
    p.add_argument("--eta-ir-floor-final-cells", dest="eta_ir_floor_final_cells",
                   type=float, default=0.0,
                   help="anneal target for eta_ir_floor_cells")
    p.add_argument("--eta-ir-floor-ramp-iterations",
                   dest="eta_ir_floor_ramp_iterations", type=int, default=0,
                   help="solves over which to anneal the soft-mode floor down")
    p.add_argument("--low-freq-mix-thz", dest="low_freq_mix_thz", type=float,
                   default=0.0,
                   help="frequency-dependent mixing: bins below this THz get "
                        "--low-freq-mix-factor (damps the IR Bose marginal mode "
                        "at eta=0 without removing low-omega scattering; 0=off)")
    p.add_argument("--low-freq-mix-factor", dest="low_freq_mix_factor",
                   type=float, default=0.02,
                   help="gentle mixing factor for the low-omega bins")
    p.add_argument("--sigma-tol", type=float, default=1e-3,
                   help="relative Sigma^R residual tolerance")
    p.add_argument("--vertex-scale", type=float, default=1.0,
                   help="3-phonon vertex scale lambda (Sigma ~ lambda^2)")
    p.add_argument("--ramp", type=int, default=0,
                   help="adiabatic bubble switch-on over N SCBA iterations")
    p.add_argument("--eta-ramp-iters", type=int, default=0,
                   help="anneal eta DOWN over N SCBA iterations (0=off; the "
                        "anharmonic Sigma^R takes over the broadening)")
    p.add_argument("--eta-final", type=float, default=0.0,
                   help="target eta (THz) at the end of the eta ramp")
    p.add_argument("--eta-obc-ramp-iters", dest="eta_obc_ramp_iters", type=int,
                   default=0,
                   help="anneal eta_obc (contact broadening, THz^2) DOWN over N SCBA "
                        "iters then hold (in-run eta_obc continuation for eta=0 on "
                        "long cells; 0=off)")
    p.add_argument("--eta-obc-final", type=float, default=0.0,
                   help="target eta_obc (THz^2) at the end of the eta_obc ramp")
    p.add_argument("--tadpole", action="store_true",
                   help="enable the self-consistent SCP cubic tadpole static SE")
    p.add_argument("--obc", default="spectral", choices=["spectral", "sancho-rubio"],
                   help="contact solver; spectral(NEVP-full) is robust on soft "
                        "modes where sancho-rubio stalls (d5a)")
    p.add_argument("--bcs", type=int, default=1, help="block_comm_size")
    p.add_argument("--qcs", type=int, default=1, help="q_comm_size")
    p.add_argument("--comm-backend", default=None,
                   choices=("host_mpi", "device_mpi", "nccl"),
                   help="set every per-op [compute.comm] backend selector "
                        "(GPU runs); default omits them (legacy TOML)")
    p.add_argument("--numba-threads", type=int, default=1)
    p.add_argument("--blas-threads", type=int, default=1)
    p.add_argument("--algorithm", default="rgf", choices=["rgf","inv"])
    p.add_argument("--max-batch", dest="max_batch", type=int, default=100000)
    p.add_argument("--profile", action="store_true", help="enable per-phase profiler JSON dump")
    a = p.parse_args()
    a.tL = a.temperature + a.dt / 2.0
    a.tR = a.temperature - a.dt / 2.0

    if a.system in ("cnt33", "cnt80"):
        a.tdir = a.tdir or "z"
        a.eta = a.eta if a.eta is not None else 0.0
        a.nfreq = a.nfreq or 161
        a.fmax = a.fmax or 55.0
        cfg = cnt_config(a)
    elif a.system in ("sinw_d5a", "sinw_d11a"):
        a.tdir = a.tdir or "z"
        a.eta = a.eta if a.eta is not None else 0.0
        a.nfreq = a.nfreq or 101
        a.fmax = a.fmax or 18.0
        cfg = cnt_config(a)
    elif a.system == "srtio3":
        a.tdir = a.tdir or "z"
        a.eta = a.eta if a.eta is not None else 0.0
        a.nfreq = a.nfreq or 121
        a.fmax = a.fmax or 26.0
        cfg = cnt_config(a)
    else:
        a.tdir = a.tdir or "x"
        a.eta = a.eta if a.eta is not None else 0.0
        a.nfreq = a.nfreq or 121
        a.fmax = a.fmax or 15.0
        cfg = film_config(a)

    # Validate BEFORE writing anything: --freq-grid file must not leave a
    # TOML behind that references a grid file which does not exist.
    import numpy as np
    ep = Path(a.work) / "phonon_energies.npy"
    if a.freq_grid == "file" and not (ep.exists() and not ep.is_symlink()):
        raise SystemExit(
            "--freq-grid file requires an existing (non-symlink) "
            f"phonon_energies.npy in {a.work} (run make_grid.py first).")

    path = Path(a.work) / "quatrex_config.toml"
    path.write_text(cfg)
    # Keep the energy-grid input consistent with the configured window: the
    # geometry build may have used a different default nfreq, and the SSE
    # grid spacing scales Sigma (2026-06-10 audit). Replace symlinked inputs
    # with a real file (never write through the shared geometry symlink).
    if a.freq_grid == "file":
        # The file IS the (possibly non-uniform) grid -- e.g. written by
        # make_grid.py before this call. Keep it.
        g = np.load(ep)
        print(f"kept phonon_energies.npy: {g.size} pts on "
              f"[{g[0]:.4g}, {g[-1]:.4g}] THz")
    else:
        if ep.is_symlink() or ep.exists():
            ep.unlink()  # break sym/hard links to the shared geometry copy
        np.save(ep, np.linspace(a.emin, a.fmax, a.nfreq))
    print(f"wrote {path}  system={a.system} T={a.temperature}(dT={a.dt}) "
          f"eta={a.eta} retarded={a.retarded} "
          f"mix={a.mix}/{a.mixing_method} bcs={a.bcs} qcs={a.qcs} "
          f"nfreq={a.nfreq} fmax={a.fmax}")


if __name__ == "__main__":
    main()
