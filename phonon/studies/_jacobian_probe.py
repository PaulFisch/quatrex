"""Power-iteration probe of the SCBA fixed-point Jacobian J = dF/dSigma.

Usage:
    python phonon/studies/_jacobian_probe.py         --config <quatrex_config.toml> --sigma <sigma_snapshot.npz>         [--steps 15] [--eps-rel 1e-6] [--top 2] [--workdir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

KEYS = ("sigma_lesser", "sigma_greater", "sigma_retarded")


def _flat(d):
    return np.concatenate([np.asarray(d[k]).ravel() for k in KEYS])


def _unflat(v, shapes):
    out, o = {}, 0
    for k, sh in shapes.items():
        n = int(np.prod(sh))
        out[k] = v[o:o + n].reshape(sh)
        o += n
    return out


def apply_map(engine, config, sigma_npz: Path, out_npz: Path,
              log: Path) -> None:
    """One raw SCBA map application F(Sigma_in) -> out_npz."""
    env = os.environ.copy()
    env.update(
        QX_CONFIG=str(config), QX_MAXIT="1", QX_MINIT="1", QX_MIX="1.0",
        QX_MIXMETHOD="linear", QX_SIGMA_INIT=str(sigma_npz),
        QX_SAVE_SIGMA=str(out_npz), QX_NPZ=str(out_npz) + ".run.npz",
        QX_SAVE_DIAG_G="0",
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
    )
    env.setdefault("QUATREX_PHPH_RING_THREADS", "64")
    proc = subprocess.run([sys.executable, str(engine)], env=env,
                          capture_output=True, text=True)
    log.write_text(proc.stdout + proc.stderr)
    if not out_npz.exists():
        print(proc.stdout[-2000:])
        sys.exit(f"map application failed (see {log})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--sigma", required=True, type=Path,
                    help="Sigma* snapshot (QX_SAVE_SIGMA npz)")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--eps-rel", type=float, default=1e-6)
    ap.add_argument("--top", type=int, default=1,
                    help="number of dominant eigenvalues (deflation)")
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    engine = Path(__file__).resolve().parents[1] / "studies/engine/run.py"
    work = args.workdir or (args.sigma.parent / "jprobe")
    work.mkdir(parents=True, exist_ok=True)

    ref = np.load(args.sigma)
    shapes = {k: ref[k].shape for k in KEYS}
    x_star = _flat(ref)
    xnorm = float(np.linalg.norm(x_star))
    eps = args.eps_rel * xnorm
    print(f"|Sigma*| = {xnorm:.4e}, eps = {eps:.4e}, "
          f"n = {x_star.size}", flush=True)

    # F(Sigma*): the fixed-point defect tells how close the reference is.
    f_star_npz = work / "F_star.npz"
    if not f_star_npz.exists():
        apply_map(engine, args.config, args.sigma, f_star_npz,
                  work / "F_star.log")
    f_star = _flat(np.load(f_star_npz))
    defect = float(np.linalg.norm(f_star - x_star)) / xnorm
    print(f"fixed-point defect |F(S*)-S*|/|S*| = {defect:.3e}", flush=True)

    rng = np.random.default_rng(args.seed)
    basis: list[np.ndarray] = []          # converged eigenvectors (deflation)
    results = []
    for ev in range(args.top):
        delta = rng.standard_normal(x_star.size) \
            + 1j * rng.standard_normal(x_star.size)
        for b in basis:
            delta -= np.vdot(b, delta) * b
        delta /= np.linalg.norm(delta)
        lam = None
        for step in range(args.steps):
            pert = _unflat(x_star + eps * delta, shapes)
            pert_npz = work / f"pert_{ev}_{step}.npz"
            np.savez(pert_npz, **pert)
            out_npz = work / f"Fpert_{ev}_{step}.npz"
            apply_map(engine, args.config, pert_npz, out_npz,
                      work / f"Fpert_{ev}_{step}.log")
            jd = (_flat(np.load(out_npz)) - f_star) / eps
            lam = complex(np.vdot(delta, jd))
            ratio = float(np.linalg.norm(jd))
            print(f"  ev{ev} step {step:2d}: |J d|/|d| = {ratio:.4f}   "
                  f"rayleigh = {lam.real:+.4f}{lam.imag:+.4f}j "
                  f"(|.| = {abs(lam):.4f})", flush=True)
            for b in basis:
                jd -= np.vdot(b, jd) * b
            nrm = np.linalg.norm(jd)
            if nrm < 1e-300:
                break
            delta = jd / nrm
            pert_npz.unlink()
            out_npz.unlink()
        # omega support of the converged eigenvector (axis 0 = omega)
        prof = np.zeros(shapes["sigma_lesser"][0])
        v = _unflat(delta, shapes)
        for k in KEYS:
            a = np.abs(np.asarray(v[k]))
            prof += (a ** 2).reshape(a.shape[0], -1).sum(axis=1)
        top_bins = np.argsort(prof)[::-1][:5]
        results.append({"lambda": [lam.real, lam.imag], "abs": abs(lam),
                        "top_omega_bins": sorted(int(i) for i in top_bins)})
        print(f"ev{ev}: |lambda| = {abs(lam):.4f}  "
              f"omega-support bins {sorted(int(i) for i in top_bins)}",
              flush=True)
        basis.append(delta.copy())

    (work / "result.json").write_text(json.dumps(
        {"defect": defect, "eigs": results}, indent=1))
    print(f"saved {work / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
