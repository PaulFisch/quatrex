"""Head-to-head SCBA mixer/scheme comparison on one testbed config.

Clones the base config once per scheme (rewriting the [scba]/[phonon]
mixing knobs in place), runs each through the production engine to a fixed
iteration budget, and reports a ranked table: best residual, iterations to
tolerance, trend at the budget, wall s/it, final heat currents and lead
balance. Every run's npz keeps the full per-iteration record
(iter_sigma_max, iter_mixer_* diagnostics when enabled).

Usage:
    python phonon/studies/_mixer_campaign.py \
        --base .../cnt33_linear/quatrex_config.toml \
        --budget 300 [--schemes lin02,and_d8,and_d8_r1e4,rre_c8,...] \
        [--workdir .../mixer_campaign_L2] [--save-sigma-at N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DIAG = ["mixer_diagnostics = true"]

#: scheme name -> {"scba": [...], "phonon": [...]} config-line patches
SCHEMES: dict[str, dict[str, list[str]]] = {
    "lin02": {"scba": ['mixing_method = "linear"', "mixing_factor = 0.2"]},
    "lin01": {"scba": ['mixing_method = "linear"', "mixing_factor = 0.1"]},
    "lin02_lowfreq": {
        "scba": ['mixing_method = "linear"', "mixing_factor = 0.2"],
        "phonon": ["low_freq_mixing_thz = 1.5",
                   "low_freq_mixing_factor = 0.02"]},
    "lin02_irfloor": {
        "scba": ['mixing_method = "linear"', "mixing_factor = 0.2"],
        "phonon": ["eta_ir_floor_cells = 2.0"]},
    # Anderson family (beta = mixing_factor of the base run kept at 0.2)
    "and_d8": {"scba": ['mixing_method = "anderson"', "mixing_factor = 0.2",
                        "anderson_depth = 8", "anderson_warmup_iters = 20",
                        "anderson_ridge = 0.0"] + DIAG},
    "and_d8_r1e8": {"scba": ['mixing_method = "anderson"',
                             "mixing_factor = 0.2", "anderson_depth = 8",
                             "anderson_warmup_iters = 20",
                             "anderson_ridge = 1e-8"] + DIAG},
    "and_d8_r1e4": {"scba": ['mixing_method = "anderson"',
                             "mixing_factor = 0.2", "anderson_depth = 8",
                             "anderson_warmup_iters = 20",
                             "anderson_ridge = 1e-4"] + DIAG},
    "and_d8_r1e2": {"scba": ['mixing_method = "anderson"',
                             "mixing_factor = 0.2", "anderson_depth = 8",
                             "anderson_warmup_iters = 20",
                             "anderson_ridge = 1e-2"] + DIAG},
    "and_d4_r1e4": {"scba": ['mixing_method = "anderson"',
                             "mixing_factor = 0.2", "anderson_depth = 4",
                             "anderson_warmup_iters = 20",
                             "anderson_ridge = 1e-4"] + DIAG},
    "and_d8_p3_r1e4": {"scba": ['mixing_method = "anderson"',
                                "mixing_factor = 0.2", "anderson_depth = 8",
                                "anderson_warmup_iters = 20",
                                "anderson_period = 3",
                                "anderson_ridge = 1e-4"] + DIAG},
    "and_d8_r1e4_guard": {"scba": [
        'mixing_method = "anderson"', "mixing_factor = 0.2",
        "anderson_depth = 8", "anderson_warmup_iters = 20",
        "anderson_ridge = 1e-4", "anderson_step_cap = 25.0",
        "anderson_revert_factor = 5.0",
        "anderson_stagnation_restart = 5"] + DIAG},
    # Root-finder family (experimental_mixer defaults unless noted)
    "rre_c5": {"scba": ['mixing_method = "rre"', "mixing_factor = 0.2",
                        "anderson_ridge = 1e-6"],
               "scba.experimental_mixer": ["rre_cycle = 5"]},
    "rre_c8": {"scba": ['mixing_method = "rre"', "mixing_factor = 0.2",
                        "anderson_ridge = 1e-6"],
               "scba.experimental_mixer": ["rre_cycle = 8"]},
    "rre_c12": {"scba": ['mixing_method = "rre"', "mixing_factor = 0.2",
                         "anderson_ridge = 1e-6"],
                "scba.experimental_mixer": ["rre_cycle = 12"]},
    "broyden": {"scba": ['mixing_method = "broyden"', "mixing_factor = 0.2",
                         "anderson_depth = 8"]},
    "rpm": {"scba": ['mixing_method = "rpm"', "mixing_factor = 0.2"],
            "scba.experimental_mixer": ["rpm_max_subspace = 6"]},
    "jfnk": {"scba": ['mixing_method = "jfnk"', "mixing_factor = 0.2"]},
}

_MIX_KEYS = re.compile(
    r"^\s*(mixing_method|mixing_factor|anderson_\w+|low_freq_mixing_\w+|"
    r"eta_ir_floor_cells|mixer_diagnostics|rre_cycle|rpm_max_subspace|"
    r"broyden_\w+|jfnk_\w+)\s*=")


def make_config(base: Path, workdir: Path, name: str,
                patches: dict[str, list[str]]) -> Path:
    text = base.read_text()
    vdir = workdir / name
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "out").mkdir(exist_ok=True)
    for f in ("dynamical_matrix.mat", "fc3_blocks.hdf5", "structure.xyz",
              "phonon_energies.npy"):
        if not (vdir / f).exists():
            (vdir / f).symlink_to(base.parent / f)
    text = text.replace(str(base.parent), str(vdir))
    # strip every existing mixing-related line, then insert the scheme's
    lines = [ln for ln in text.splitlines() if not _MIX_KEYS.match(ln)]
    text = "\n".join(lines) + "\n"
    for section, adds in patches.items():
        header = f"[{section}]"
        if header in text:
            text = text.replace(header, header + "\n" + "\n".join(adds), 1)
        else:
            text += f"\n{header}\n" + "\n".join(adds) + "\n"
    cfg = vdir / "quatrex_config.toml"
    cfg.write_text(text)
    return cfg


def parse_residuals(log_text: str) -> np.ndarray:
    return np.array([float(m) for m in re.findall(
        r"rel Sigma\^R residual ([0-9.e+-]+)", log_text)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--schemes", default=",".join(SCHEMES))
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="override mixing_factor in every scheme (0 = keep "
                         "the scheme table's 0.2)")
    ap.add_argument("--save-sigma-at", type=int, default=0,
                    help="additionally save the final Sigma snapshot of "
                         "every scheme (for the Jacobian probe)")
    args = ap.parse_args()

    engine = Path(__file__).resolve().parents[1] / "studies/engine/run.py"
    workdir = args.workdir or (args.base.parent.parent / "mixer_campaign")
    rows = []
    for name in args.schemes.split(","):
        name = name.strip()
        if name not in SCHEMES:
            sys.exit(f"unknown scheme {name!r}; known: {sorted(SCHEMES)}")
        patches = SCHEMES[name]
        if args.beta > 0.0:
            patches = {sec: [re.sub(r"mixing_factor = [0-9.]+",
                                    f"mixing_factor = {args.beta}", ln)
                             for ln in lines]
                       for sec, lines in patches.items()}
        cfg = make_config(args.base, workdir, name, patches)
        env = os.environ.copy()
        env.update(QX_CONFIG=str(cfg), QX_MAXIT=str(args.budget),
                   QX_MINIT="3", QX_NPZ=str(cfg.parent / "run.npz"),
                   QX_SAVE_DIAG_G="0",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        env.setdefault("QUATREX_PHPH_RING_THREADS", "64")
        if args.save_sigma_at:
            env["QX_SAVE_SIGMA"] = str(cfg.parent / "sigma_final.npz")
        t0 = time.perf_counter()
        print(f"=== {name}: budget {args.budget}", flush=True)
        proc = subprocess.run([sys.executable, str(engine)], env=env,
                              capture_output=True, text=True)
        wall = time.perf_counter() - t0
        log = proc.stdout + proc.stderr
        (cfg.parent / "run.log").write_text(log)
        r = parse_residuals(log)
        row = {"scheme": name, "n_it": int(r.size),
               "s_per_it": wall / max(r.size, 1)}
        if r.size:
            row["best"] = float(r.min())
            row["best_at"] = int(r.argmin())
            row["last"] = float(r[-1])
            hit = np.nonzero(r < 1e-3)[0]
            row["it_to_1e-3"] = int(hit[0]) if hit.size else -1
            n = min(40, r.size)
            row["slope"] = float(np.polyfit(
                np.arange(n), np.log(np.maximum(r[-n:], 1e-300)), 1)[0])
        npz = cfg.parent / "run.npz"
        if npz.exists():
            d = np.load(npz)
            row["converged"] = bool(d["converged"])
            if "final_heat" in d.files:
                fh = np.asarray(d["final_heat"], float)
                row["J_L"] = float(fh[0])
                row["lead_bal"] = float(
                    abs(fh[0] - fh[-1]) / max(abs(fh[0]), 1e-300))
        rows.append(row)
        print("   ", json.dumps(row), flush=True)

    (workdir / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"\n{'scheme':>18} {'conv':>5} {'it@1e-3':>8} {'best':>10} "
          f"{'last':>10} {'slope':>8} {'s/it':>6} {'J_L':>9} {'leadbal':>9}")
    for row in sorted(rows, key=lambda r: (
            not r.get("converged", False),
            r["it_to_1e-3"] if r.get("it_to_1e-3", -1) >= 0 else 10**9,
            r.get("best", np.inf))):
        print(f"{row['scheme']:>18} {str(row.get('converged', '?')):>5} "
              f"{row.get('it_to_1e-3', -1):>8} {row.get('best', np.nan):>10.2e} "
              f"{row.get('last', np.nan):>10.2e} {row.get('slope', np.nan):>8.4f} "
              f"{row['s_per_it']:>6.1f} {row.get('J_L', np.nan):>9.3f} "
              f"{row.get('lead_bal', np.nan):>9.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
