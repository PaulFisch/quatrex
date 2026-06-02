"""Si thin-film cross-plane thermal conductivity vs thickness (replicate
Guo, Bescond & Zhang, PRB 102, 195412 (2020), Table 4).

The film = bulk Si truncated to N layers along the transport axis with in-plane
(2D) periodicity -> a transverse q_perp problem solved by the q-resolved anharmonic
NEGF (`solver.transmission_q`, coupled-q 3-phonon self-energy `se_q`), reusing the
BULK Si force constants (FD FC2+FC3 in reaps/si_primitive_work, kappa_bulk ~ 110-115).

Conductance from transmission_q is per unit area, G [W/m^2/K] = J / (A_perp * dT).
Cross-plane conductivity of a film of physical thickness L:  kappa(L) = G(L) * L.
As L grows the ballistic kappa rises ~linearly (few scatterings) and the anharmonic
kappa saturates toward kappa_bulk -- the ballistic->diffusive crossover of their Table 4.

Usage:
  python si_film_kappa.py --smoke                 # load + (1,1)==finite regression
  python si_film_kappa.py --nk 4 --nfreq 101 --n-slabs 2 3 5 8 --max-iter 40
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_W = Path("/usr/scratch/mont-fort11/pfischill/quatrex/phonon")
for p in (_W.parent, _W):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phonon.solver import transmission_q


def load_bulk_si(fc3_subdir="reaps/si_primitive_work"):
    """Bulk-Si 2-atom FCC primitive with FD FC2 (from fc3.hdf5) + FC3 (phono3py)."""
    import h5py
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phono3py import load as phono3py_load
    from phonon_inputs.force_constants import load_fc3_phono3py

    d = _W / fc3_subdir
    fc3_h5 = d / "fc3.hdf5"
    yaml_path = d / "phono3py.yaml"
    with h5py.File(d / "fc2.hdf5", "r") as f:
        fc2 = f["force_constants"][:]
    ph3 = phono3py_load(phono3py_yaml=str(yaml_path), produce_fc=False, log_level=0)
    cell = PhonopyAtoms(symbols=ph3.unitcell.symbols, cell=ph3.unitcell.cell,
                        scaled_positions=ph3.unitcell.scaled_positions)
    phonon = Phonopy(cell, supercell_matrix=ph3.supercell_matrix,
                     primitive_matrix=np.eye(3))
    phonon.force_constants = fc2
    return phonon, str(fc3_h5)


def build_nn_masked_M(phonon, fc3_path, transport_direction, nn_cutoff):
    """Mass-weighted FC3 (M_stacked) with the 3rd-order vertex restricted to atom triplets
    (i,j,k) where j and k are within nn_cutoff of i -- Guo et al.'s nearest-neighbour
    approximation (II) for the anharmonic force constants. nn_cutoff=0 -> no restriction."""
    import h5py
    from phonon_inputs.separable import build_supercell_mapping, build_realspace_fc3_matrices
    prim_indices, cell_frac, slab_indices, ref_sc = build_supercell_mapping(
        phonon, transport_direction)
    masses_super = phonon.supercell.masses
    nat = len(phonon.primitive.masses)
    with h5py.File(fc3_path, "r") as f:
        fc3 = f["fc3"][:]                       # (n_super, n_super, n_super, 3,3,3)
    if nn_cutoff and nn_cutoff > 0:
        sc = phonon.supercell
        pos = sc.scaled_positions @ sc.cell
        inv = np.linalg.inv(sc.cell)
        n = len(pos)
        D = np.zeros((n, n))
        for i in range(n):
            d = pos[i] - pos
            fr = d @ inv
            fr -= np.round(fr)
            D[i] = np.linalg.norm(fr @ sc.cell, axis=1)
        near = D <= nn_cutoff                    # (n,n) bool; diagonal True (dist 0)
        mask = near[:, :, None] & near[:, None, :]   # [i,j,k]: j and k both near i
        fc3 = fc3 * mask[:, :, :, None, None, None]
        kept = float(mask.sum()) / mask.size
        print(f"  FC3 1st-NN mask (cutoff {nn_cutoff} A): kept {kept*100:.1f}% of triplets",
              flush=True)
    return build_realspace_fc3_matrices(fc3, nat, masses_super, ref_sc)


def layer_spacing_m(phonon, transport_direction):
    """Physical inter-layer repeat distance along the transport axis (meters)."""
    tidx = "xyz".index(transport_direction)
    a_t = phonon.primitive.cell[tidx]
    return float(np.linalg.norm(a_t)) * 1e-10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nk", type=int, default=4)
    ap.add_argument("--nfreq", type=int, default=101)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--n-slabs", type=int, nargs="+", default=[2, 3, 5, 8])
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--delta-T", type=float, default=10.0)
    ap.add_argument("--eta-factor", type=float, default=0.5)
    ap.add_argument("--mixing", type=float, default=0.3)
    ap.add_argument("--transport-dir", default="x")
    ap.add_argument("--fc3-nn-cutoff", type=float, default=0.0,
                    help="restrict FC3 vertex to atom triplets within this distance (A) of the "
                         "central atom (Guo approx. II, 1st-NN ~ 2.6); 0 = full FD FC3")
    ap.add_argument("--vertex-scale", type=float, default=1.0,
                    help="multiply the FC3 vertex M by this factor (Sigma ~ M^2); for LOA-Pade "
                         "coupling sweeps. The Guo factor-of-4 self-energy convention is now the "
                         "DEFAULT prefactor (constants.PHPH_SYMMETRY_FACTOR=0.25), not this knob.")
    ap.add_argument("--legacy-luisier-prefactor", action="store_true",
                    help="restore the historical 4x-too-large (Luisier) self-energy prefactor")
    ap.add_argument("--fc3-subdir", default="reaps/si_primitive_work",
                    help="material FC directory (fc2.hdf5/fc3.hdf5/phono3py.yaml); "
                         "e.g. reaps/ge_primitive_work for Ge")
    ap.add_argument("--out", default=str(_W / "scripts/out/si_film/si_film_kappa.json"))
    args = ap.parse_args()

    print(f"Loading bulk material from {args.fc3_subdir} (FD FC2+FC3)...", flush=True)
    phonon, fc3_path = load_bulk_si(fc3_subdir=args.fc3_subdir)
    d_layer = layer_spacing_m(phonon, args.transport_dir)
    print(f"  layer spacing along {args.transport_dir} = {d_layer*1e10:.4f} Ang", flush=True)
    M_override = None
    if (args.fc3_nn_cutoff and args.fc3_nn_cutoff > 0) or args.vertex_scale != 1.0:
        M_override = build_nn_masked_M(phonon, fc3_path, args.transport_dir, args.fc3_nn_cutoff)
        if args.vertex_scale != 1.0:
            M_override = M_override * args.vertex_scale
            print(f"  vertex scaled by {args.vertex_scale} -> self-energy x {args.vertex_scale**2}",
                  flush=True)

    if args.smoke:
        # q_mesh=(1,1) triggers transmission_q's built-in == transmission_finite check
        t0 = time.time()
        r = transmission_q(phonon, fc3_path, q_mesh_transverse=(1, 1),
                           freq_range_thz=(0.0, 15.0, 61), transport_direction=args.transport_dir,
                           eta_factor=args.eta_factor, temperature=args.temperature,
                           delta_T=args.delta_T, max_scba_iter=3, n_slabs=1, verbose=True, M_stacked_override=M_override,
                           legacy_prefactor=args.legacy_luisier_prefactor)
        print(f"[smoke] G_ball={r['thermal_conductance_ballistic']/1e6:.2f} "
              f"G_anh={r['thermal_conductance_anharmonic']/1e6:.2f} MW/m2K  "
              f"({time.time()-t0:.1f}s) -- (1,1)==finite regression passed if no assertion",
              flush=True)
        return

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n_slabs in args.n_slabs:
        L = n_slabs * d_layer
        print(f"\n{'='*60}\nThickness {n_slabs} layers = {L*1e9:.3f} nm\n{'='*60}", flush=True)
        t0 = time.time()
        r = transmission_q(phonon, fc3_path, q_mesh_transverse=(args.nk, args.nk),
                           freq_range_thz=(0.0, 15.0, args.nfreq),
                           transport_direction=args.transport_dir,
                           eta_factor=args.eta_factor, temperature=args.temperature,
                           delta_T=args.delta_T, max_scba_iter=args.max_iter,
                           scba_tol=0.005, mixing=args.mixing, n_slabs=n_slabs, verbose=True,
                           M_stacked_override=M_override,
                           legacy_prefactor=args.legacy_luisier_prefactor)
        Gb = r["thermal_conductance_ballistic"]; Ga = r["thermal_conductance_anharmonic"]
        row = dict(n_slabs=n_slabs, L_nm=L*1e9, G_ball=Gb, G_anh=Ga,
                   kappa_ball=Gb*L, kappa_anh=Ga*L,
                   conservation=float(r["heat_flow_conservation"]),
                   wall_s=time.time()-t0)
        rows.append(row)
        print(f"  -> kappa_ball={row['kappa_ball']:.2f}  kappa_anh={row['kappa_anh']:.2f} W/mK "
              f"(cons={row['conservation']:.1e}, {row['wall_s']:.0f}s)", flush=True)
        json.dump(dict(rows=rows, d_layer_ang=d_layer*1e10, args=vars(args)),
                  open(out, "w"), indent=2)
        print(f"  [checkpoint] {out}", flush=True)

    print("\n=== Si film cross-plane kappa(thickness) ===")
    for row in rows:
        print(f"  {row['n_slabs']:2d} layers ({row['L_nm']:.2f} nm): "
              f"kappa_ball={row['kappa_ball']:6.2f}  kappa_anh={row['kappa_anh']:6.2f} W/mK")


if __name__ == "__main__":
    main()
