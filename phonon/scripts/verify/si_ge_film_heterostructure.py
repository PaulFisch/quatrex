"""Si/Ge heterostructure thin film (Guo-Bescond-Zhang PRB 102 195412): a Ge-layer barrier
embedded in a Si film, treated as Si force constants + Ge mass (mass-mismatch), Si contacts.

Reproduces Guo's heterostructure result that phonon-phonon scattering enhances the thermal
resistance. Uses the per-slab MASS profile in transmission_q (device dynamical matrix re-weighted
by slab mass; Si leads; uniform-Si FC3 vertex -- the F25-justified model). The corrected 1/4
prefactor is the default.

Compares, at the same total thickness/mesh:
  pure Si film       (mass_profile all Si)   -- regression: must equal plain transmission_q
  Si | Ge-barrier | Si                       -- the heterostructure
for both ballistic and anharmonic, and reports the thermal-resistance enhancement.

Usage:
  python si_ge_film_heterostructure.py --smoke                    # all-Si == plain transmission_q
  python si_ge_film_heterostructure.py --nk 8 --nfreq 121 --eta-factor 0.1 \
      --n-left 2 --n-barrier 2 --n-right 2
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
from phonon.scripts.verify.si_film_kappa import load_bulk_si, layer_spacing_m

M_SI = 28.0855
M_GE = 72.630


def run(phonon, fc3_path, nk, nfreq, eta_factor, tdir, n_slabs, mass_profile,
        max_iter, temperature=300.0, delta_T=10.0):
    return transmission_q(
        phonon, fc3_path, q_mesh_transverse=(nk, nk),
        freq_range_thz=(0.0, 15.0, nfreq), transport_direction=tdir,
        eta_factor=eta_factor, temperature=temperature, delta_T=delta_T,
        max_scba_iter=max_iter, scba_tol=0.005, mixing=0.3, n_slabs=n_slabs,
        verbose=False, mass_profile=mass_profile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nk", type=int, default=8)
    ap.add_argument("--nfreq", type=int, default=121)
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--n-left", type=int, default=2)
    ap.add_argument("--n-barrier", type=int, default=2)
    ap.add_argument("--n-right", type=int, default=2)
    ap.add_argument("--eta-factor", type=float, default=0.1)
    ap.add_argument("--transport-dir", default="x")
    ap.add_argument("--out", default=str(_W / "scripts/out/si_film/si_ge_heterostructure.json"))
    args = ap.parse_args()

    phonon, fc3_path = load_bulk_si()
    d = layer_spacing_m(phonon, args.transport_dir)

    if args.smoke:
        # all-Si mass profile must reproduce plain transmission_q (mass_profile=None)
        ns = 3
        t0 = time.time()
        r_none = run(phonon, fc3_path, 1, 61, args.eta_factor, args.transport_dir,
                     ns, None, 3)
        r_si = run(phonon, fc3_path, 1, 61, args.eta_factor, args.transport_dir,
                   ns, [M_SI] * ns, 3)
        gb0 = r_none["thermal_conductance_ballistic"]; gb1 = r_si["thermal_conductance_ballistic"]
        ga0 = r_none["thermal_conductance_anharmonic"]; ga1 = r_si["thermal_conductance_anharmonic"]
        eb = abs(gb1 - gb0) / abs(gb0); ea = abs(ga1 - ga0) / abs(ga0)
        print(f"[smoke] all-Si mass_profile vs mass_profile=None: "
              f"G_ball rel {eb:.1e}, G_anh rel {ea:.1e}  ({time.time()-t0:.1f}s) -> "
              f"{'PASS' if eb < 1e-10 and ea < 1e-10 else 'FAIL'}")
        return

    n_slabs = args.n_left + args.n_barrier + args.n_right
    L = n_slabs * d
    prof_si = [M_SI] * n_slabs
    prof_het = [M_SI] * args.n_left + [M_GE] * args.n_barrier + [M_SI] * args.n_right
    print(f"Si/Ge heterostructure film: {n_slabs} slabs = {L*1e9:.2f} nm "
          f"(Si{args.n_left}|Ge{args.n_barrier}|Si{args.n_right}), nk={args.nk}, nfreq={args.nfreq}",
          flush=True)
    rows = {}
    for label, prof in [("pure_Si", prof_si), ("Si_Ge_Si", prof_het)]:
        t0 = time.time()
        r = run(phonon, fc3_path, args.nk, args.nfreq, args.eta_factor, args.transport_dir,
                n_slabs, prof, args.max_iter)
        Gb = r["thermal_conductance_ballistic"]; Ga = r["thermal_conductance_anharmonic"]
        rows[label] = dict(G_ball=Gb, G_anh=Ga, R_ball=1.0 / Gb, R_anh=1.0 / Ga,
                           conservation=float(r["heat_flow_conservation"]), wall_s=time.time() - t0)
        print(f"  {label:9s}: G_ball={Gb/1e6:7.1f}  G_anh={Ga/1e6:7.1f} MW/m2K  "
              f"(R_anh={1e9/Ga:.3f} nK·m2/W, cons={rows[label]['conservation']:.1e}, "
              f"{rows[label]['wall_s']:.0f}s)", flush=True)
    # resistance enhancement from the Ge barrier + from phph
    het, si = rows["Si_Ge_Si"], rows["pure_Si"]
    print(f"\n  barrier raises ballistic R by {100*(het['R_ball']/si['R_ball']-1):+.1f}%, "
          f"anharmonic R by {100*(het['R_anh']/si['R_anh']-1):+.1f}% vs pure Si")
    print(f"  phph raises R: pure Si {100*(si['R_anh']/si['R_ball']-1):+.1f}%, "
          f"heterostructure {100*(het['R_anh']/het['R_ball']-1):+.1f}%  (Guo: phph enhances R)")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(rows=rows, L_nm=L*1e9, profile_si=prof_si, profile_het=prof_het,
                   args=vars(args)), open(out, "w"), indent=2)
    print(f"  [saved] {out}", flush=True)


if __name__ == "__main__":
    main()
