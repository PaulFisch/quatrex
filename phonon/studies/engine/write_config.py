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
    python write_config.py --system cnt33 --work DIR -L 4 --eta 0.7 \
        --nfreq 161 --fmax 55 [--bcs 1 --qcs 1 --numba-threads 1]
Usage (film):
    python write_config.py --system sifilm --work DIR --nslabs 5 --nk 8 \
        --tdir x --shift <kshift> --eta 0.4 --nfreq 121 --fmax 15 [--qcs 8]
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
    return f"""
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


def cnt_config(a):
    if a.bcs > a.ncells:
        raise SystemExit(f"block_comm_size {a.bcs} > num_transport_cells {a.ncells} "
                         "(each block-rank needs >=1 BTD block)")
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
phonon = true

[electron]
energy_window_min = {a.emin}
energy_window_max = {a.fmax}
energy_window_num = {a.nfreq}
fermi_level = 0.0

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
sse_low_freq_cutoff_thz = {a.sse_cutoff}
sse_cutoff_zero_g = {str(a.sse_zero_g).lower()}
sigma_convergence_tol = {a.sigma_tol}
heat_flow_conservation_tol = 1e-2
[phonon.solver]
compute_current = true
max_batch_size = {a.max_batch}
algorithm = "{a.algorithm}"
[phonon.obc]
algorithm = "{a.obc}"
nevp_solver = "full"
block_sections = 1
{tail_block(a)}"""  # noqa: E501  (left/right temperature set below from --temperature/--dt)


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
phonon = true

[electron]
energy_window_min = {a.emin}
energy_window_max = {a.fmax}
energy_window_num = {a.nfreq}
fermi_level = 0.0

[phonon]
eta = {a.eta}
eta_obc = {a.eta_obc}
left_temperature = {a.tL}
right_temperature = {a.tR}
model = "negf"
fc3_path = "{a.work}/fc3_blocks.hdf5"
qfold_path = "{a.work}/qfold_vertices.npz"
retarded_method = "{a.retarded}"
scp_tadpole = {str(a.tadpole).lower()}
sse_ramp_iterations = {a.ramp}
sse_vertex_scale = {a.vertex_scale}
sse_low_freq_cutoff_thz = {a.sse_cutoff}
sse_cutoff_zero_g = {str(a.sse_zero_g).lower()}
sigma_convergence_tol = {a.sigma_tol}
heat_flow_conservation_tol = 1e-2
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
    p.add_argument("--temperature", type=float, default=300.0,
                   help="mean device temperature T (K); leads at T +/- dt/2")
    p.add_argument("--dt", type=float, default=10.0, help="lead temperature drop (K)")
    p.add_argument("--eta", type=float, default=None, help="THz (CNT 0.7, film 0.4)")
    p.add_argument("--eta-obc", type=float, default=0.0)
    p.add_argument("--emin", type=float, default=0.0)
    p.add_argument("--nfreq", type=int, default=None)
    p.add_argument("--fmax", type=float, default=None)
    p.add_argument("--mix", type=float, default=0.1)
    p.add_argument("--mixing-method", default="linear",
                   choices=["linear", "anderson"],
                   help="plain Anderson(m) acceleration (Walker & Ni 2011)")
    p.add_argument("--anderson-depth", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=50,
                   help="SCBA cap; the conductance (best-iterate) converges well "
                        "before the Sigma residual (F30), so 50 bounds wall-time")
    p.add_argument("--retarded", default="half", choices=["half", "fft"])
    p.add_argument("--sse-zero-g", action="store_true",
                   help="hard cutoff: zero lead injection below --sse-cutoff")
    p.add_argument("--sse-cutoff", type=float, default=0.0,
                   help="low-frequency 3ph-SSE cutoff in THz (ballistic below)")
    p.add_argument("--sigma-tol", type=float, default=1e-3,
                   help="relative Sigma^R residual tolerance")
    p.add_argument("--vertex-scale", type=float, default=1.0,
                   help="3-phonon vertex scale lambda (Sigma ~ lambda^2)")
    p.add_argument("--ramp", type=int, default=0,
                   help="adiabatic bubble switch-on over N SCBA iterations")
    p.add_argument("--tadpole", action="store_true",
                   help="enable the self-consistent SCP cubic tadpole static SE")
    p.add_argument("--obc", default="spectral", choices=["spectral", "sancho-rubio"],
                   help="contact solver; spectral(NEVP-full) is robust on soft "
                        "modes where sancho-rubio stalls (d5a)")
    p.add_argument("--bcs", type=int, default=1, help="block_comm_size")
    p.add_argument("--qcs", type=int, default=1, help="q_comm_size")
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
        a.eta = a.eta if a.eta is not None else 0.7
        a.nfreq = a.nfreq or 161
        a.fmax = a.fmax or 55.0
        cfg = cnt_config(a)
    elif a.system in ("sinw_d5a", "sinw_d11a"):
        a.tdir = a.tdir or "z"
        a.eta = a.eta if a.eta is not None else 0.11  # physical eta_w (F10/F14)
        a.nfreq = a.nfreq or 101
        a.fmax = a.fmax or 18.0
        cfg = cnt_config(a)
    elif a.system == "srtio3":
        a.tdir = a.tdir or "z"
        a.eta = a.eta if a.eta is not None else 0.3  # strongly anharmonic -> broader
        a.nfreq = a.nfreq or 121
        a.fmax = a.fmax or 26.0
        cfg = cnt_config(a)
    else:
        a.tdir = a.tdir or "x"
        a.eta = a.eta if a.eta is not None else 0.4
        a.nfreq = a.nfreq or 121
        a.fmax = a.fmax or 15.0
        cfg = film_config(a)

    path = Path(a.work) / "quatrex_config.toml"
    path.write_text(cfg)
    # Keep the energy-grid input consistent with the configured window: the
    # geometry build may have used a different default nfreq, and the SSE
    # grid spacing scales Sigma (2026-06-10 audit). Replace symlinked inputs
    # with a real file (never write through the shared geometry symlink).
    import numpy as np
    ep = Path(a.work) / "phonon_energies.npy"
    if ep.is_symlink() or ep.exists():
        ep.unlink()  # break sym/hard links to the shared geometry copy
    np.save(ep, np.linspace(a.emin, a.fmax, a.nfreq))
    print(f"wrote {path}  system={a.system} T={a.temperature}(dT={a.dt}) "
          f"eta={a.eta} retarded={a.retarded} "
          f"mix={a.mix}/{a.mixing_method} bcs={a.bcs} qcs={a.qcs} "
          f"nfreq={a.nfreq} fmax={a.fmax}")


if __name__ == "__main__":
    main()
