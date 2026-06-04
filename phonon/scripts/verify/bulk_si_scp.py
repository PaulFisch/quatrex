#!/usr/bin/env python
"""Phase 4 -- bulk-Si end-to-end SCP (loop + tadpole) + band renormalisation.

Drives the full pipeline on a hiPhive-fit bulk Si reap that carries FC2 + FC3 +
FC4 (the ``fc3.hdf5`` written by ``fc3-hiphive-reap`` with a length-3 cutoff;
see ``configs/si_primitive/hiphive_fc4.yaml``):

  1. baseline transmission (harmonic + cubic bubble only),
  2. self-consistent-phonon run (loop ``Sigma_L`` + tadpole ``Sigma_T``) -> the
     renormalised ``Phi_eff`` and the anharmonic conductance,
  3. tadpole-consistency diagnostic: ``||Sigma_T||`` should be ~0 for the
     symmetry-fixed diamond Si (brief §3.6) -- a large value flags an
     inconsistency (wrong relaxation T / mismatched <uu>),
  4. the spectral function ``A(q, omega)`` + decomposed bands
     (bare -> +loop -> +loop+tadpole) along a high-symmetry q-path.

This driver needs a bulk-Si FC4 reap (not on the dev box). Compute it with:

    cd phonon
    python -m phonon_inputs fc3-hiphive-sow  --config configs/si_primitive/hiphive_fc4.yaml --from-yaml
    python -m phonon_inputs fc3-hiphive-run  --config configs/si_primitive/hiphive_fc4.yaml
    python -m phonon_inputs fc3-hiphive-reap --config configs/si_primitive/hiphive_fc4.yaml
    # -> fc3_hiphive_si_fc4/fc3.hdf5 (with fc4_atoms / fc4_values) + hiphive_meta.json

Then:
    python phonon/scripts/verify/bulk_si_scp.py --reap-dir phonon/fc3_hiphive_si_fc4 \
        --temperature 300 --out /tmp/claude/si_scp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np


def load_hiphive_phonon(reap_dir: Path):
    """Build the Phonopy object + fc3/fc4 path from a hiPhive reap dir."""
    import h5py
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    meta = json.loads((reap_dir / "hiphive_meta.json").read_text())
    pm = meta["primitive"]
    primitive = PhonopyAtoms(symbols=pm["symbols"],
                             cell=np.asarray(pm["cell"]),
                             scaled_positions=np.asarray(pm["scaled_positions"]))
    sc = np.diag(np.asarray(meta["supercell"], dtype=int))
    phonon = Phonopy(primitive, supercell_matrix=sc, primitive_matrix=np.eye(3))
    fc_path = reap_dir / "fc3.hdf5"
    with h5py.File(fc_path, "r") as f:
        phonon.force_constants = f["fc2"][:]
        has_fc4 = "fc4_atoms" in f
    return phonon, str(fc_path), meta, has_fc4


def fcc_qpath(n_per_seg=40):
    """Gamma -> X -> K -> Gamma -> L path in primitive fractional coords."""
    pts = {"G": (0.0, 0.0, 0.0), "X": (0.0, 0.5, 0.5),
           "K": (0.375, 0.375, 0.75), "L": (0.5, 0.5, 0.5)}
    seq = ["G", "X", "K", "G", "L"]
    q, dist, ticks = [], [], [0.0]
    d0 = 0.0
    for a, b in zip(seq[:-1], seq[1:]):
        pa, pb = np.array(pts[a]), np.array(pts[b])
        seg = np.linspace(0, 1, n_per_seg, endpoint=False)
        for t in seg:
            qv = pa + t * (pb - pa)
            q.append(qv)
            dist.append(d0 + t * np.linalg.norm(pb - pa))
        d0 += np.linalg.norm(pb - pa)
        ticks.append(d0)
    q.append(np.array(pts[seq[-1]]))
    dist.append(d0)
    return np.array(q), np.array(dist), ticks, seq


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reap-dir", required=True,
                    help="hiPhive reap dir (fc3.hdf5 with fc4 + hiphive_meta.json)")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--delta-t", type=float, default=10.0)
    ap.add_argument("--transport-direction", default="x")
    ap.add_argument("--freq-range", type=float, nargs=3, default=[0.01, 18.0, 101],
                    metavar=("FMIN", "FMAX", "NPOS"))
    ap.add_argument("--eta-factor", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--no-loop", action="store_true", help="tadpole only")
    ap.add_argument("--no-tadpole", action="store_true", help="loop only")
    ap.add_argument("--n-qpath", type=int, default=40)
    ap.add_argument("--out", default="/tmp/claude/si_scp")
    args = ap.parse_args()

    from solver.dense import transmission_finite
    from postproc.spectral import (
        band_renormalization_bundle,
        dynamical_matrix_qpath,
    )
    from postproc.io import save_spectral, write_reference_plot_script

    reap = Path(args.reap_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    phonon, fc_path, meta, has_fc4 = load_hiphive_phonon(reap)
    loop = (not args.no_loop) and has_fc4
    tadpole = not args.no_tadpole
    if not has_fc4 and not args.no_loop:
        print("  WARNING: no fc4 in the reap -> loop disabled (tadpole only). "
              "Run the FC4 reap first (see module docstring).")

    # <uu> for the static loop/tadpole is now sourced self-consistently from the
    # device G^< every SCBA iteration (the fixed bulk-equilibrium override was
    # removed as physically wrong -- it is non-self-consistent). NB for a bulk
    # property the open single-cell device G^< over-counts the low-omega <uu>;
    # this script is kept as the SCP machinery exerciser, not the bulk tadpole-
    # zero benchmark (which would need a genuine equilibrium, contact-free <uu>).
    common = dict(
        fc3_hdf5=fc_path,
        transport_direction=args.transport_direction,
        freq_range_thz=tuple(args.freq_range), eta_factor=args.eta_factor,
        temperature=args.temperature, delta_T=args.delta_t,
        max_scba_iter=args.max_iter,
        enforce_asr=True, verbose=True,
    )

    print("== baseline (harmonic + bubble) ==")
    base = transmission_finite(phonon, **common)

    print("== SCP (loop + tadpole) ==")
    scp = transmission_finite(
        phonon, loop=loop, tadpole=tadpole, fc4_hdf5=fc_path if loop else None,
        stage_loop_first=True, **common)
    sigma_static = scp.get("sigma_static")

    # tadpole-consistency diagnostic: ||Sigma_T|| alone (~0 for symmetric Si).
    print("== tadpole-only (consistency diagnostic) ==")
    tad = transmission_finite(phonon, loop=False, tadpole=True, **common)
    sig_t = tad.get("sigma_static")
    norm_t = float(np.linalg.norm(sig_t)) if sig_t is not None else float("nan")

    summary = {
        "temperature": args.temperature,
        "G_ball": base["thermal_conductance_ballistic"],
        "G_anh_baseline": base["thermal_conductance_anharmonic"],
        "G_anh_scp": scp["thermal_conductance_anharmonic"],
        "sigma_static_norm": (None if sigma_static is None
                              else float(np.linalg.norm(sigma_static))),
        "sigma_tadpole_norm": norm_t,
        "loop": loop, "tadpole": tadpole, "has_fc4": has_fc4,
    }
    print("\n== summary ==")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("  NOTE: sigma_tadpole_norm should be ~0 for symmetry-fixed diamond "
          "Si; a large value flags a structure/temperature/<uu> inconsistency.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # --- spectral function + decomposed bands along the q-path --------------
    print("\n== spectral A(q, omega) + decomposed bands ==")
    q_frac, q_dist, ticks, labels = fcc_qpath(args.n_qpath)
    D_q = dynamical_matrix_qpath(phonon, q_frac)
    fmax = args.freq_range[1]
    grid = np.linspace(0.05, fmax, 600)
    eta_w = args.eta_factor * (fmax / args.freq_range[2])
    # split sigma_static into loop / tadpole pieces for the decomposition:
    # loop-only and tadpole-only runs give the two static contributions.
    loop_only = (None if not loop else transmission_finite(
        phonon, loop=True, tadpole=False, fc4_hdf5=fc_path,
        stage_loop_first=True, **common).get("sigma_static"))
    bundle = band_renormalization_bundle(
        D_q, grid, eta_w_thz=eta_w, q_distance=q_dist,
        sigma_loop=loop_only, sigma_tadpole=sig_t)
    npz = save_spectral(out / "spectral.npz", **bundle,
                        tick_positions=ticks, tick_labels=labels)
    plot = write_reference_plot_script(out / "plot_spectral.py")
    print(f"  wrote {npz}\n  wrote {plot}")
    try:
        import subprocess
        subprocess.run([sys.executable, plot, str(npz),
                        str(out / "spectral.pdf")], check=True)
        print(f"  wrote {out / 'spectral.pdf'}")
    except Exception as exc:
        print(f"  (skipped PDF render: {exc})")


if __name__ == "__main__":
    main()
