"""MoS2 cross-plane R(t) ladder: contact resistance and bulk kappa.

Reads a set of converged eta=0 SCBA runs, converts each run's raw lead current
into a thermal boundary resistance and fits

    R(t) = R_c + t / kappa_bulk

so contact and bulk separate. A single length cannot do this: at 2 layers the
contact term is about 90 % of R, so kappa_z,eff from one thickness is mostly the
interface.

Bridge (verified against the two-point fit of 2026-08-10, R2 = 13.816):

    G [W/m^2/K] = h * 1e24 * J_raw * df_THz / (A_c * dT * N_q)

with ``A_c`` the transverse cell area in m^2, ``N_q`` the transverse q-count and
``dT`` the applied bias. Every one of those constants is identical across the
ladder, so they cancel in the SLOPE and only shift R_c.

Each run must have the box mask inactive -- ``interaction_cutoff`` greater than
the device's transport-direction span -- or the self-energy is not PSD and the
current is not a physical number (``bubble_positivity.md`` Sec. 6.10c). This
script refuses to fit a run that fails that test.

Usage::

    python -m phonon.studies._kappa_z_ladder cluster/cvM2b cluster/cvM4e cluster/cvM6b
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Offline analysis -- it must never need a GPU, and it must give the same
# numbers on every machine. quatrex picks its array module at qttools import
# time from this variable, so it has to be set before the first
# quatrex/qttools import (all of which happen lazily inside the functions
# below). Under the default cupy backend the mask audit dies inside
# compute_sparsity_pattern, an xp routine handed a host grid.
os.environ.setdefault("QTX_ARRAY_MODULE", "numpy")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

H_PLANCK = 6.62607015e-34  # J s
CURRENT = re.compile(r"lead_current=([-+0-9.eE]+)")
CONVERGED = re.compile(r"SCBA converged after (\d+) iterations")
NE = re.compile(r"\bne=(\d+)\b")
# Per-iteration lead current (scba.py, from 716107fa). Older logs do not have
# it; the run is then quotable only if it converged.
ITER_CURRENT = re.compile(r"lead current ([-+0-9.eE]+)")
ITER_RESIDUAL = re.compile(r"rel Sigma\^R residual ([-+0-9.eE]+)")
TAIL = 5  # iterations whose spread becomes the quoted uncertainty


def read_run(run_dir: Path) -> dict:
    from phonon.studies._cutoff_mask_audit import audit

    cfg_path = run_dir / "quatrex_config.toml"
    logs = sorted(p for p in run_dir.glob("slurm-*.out")
                  if "times" not in p.name)
    if not logs:
        raise FileNotFoundError(f"no slurm log in {run_dir}")
    text = logs[-1].read_text(errors="replace")

    cur = CURRENT.findall(text)
    if not cur:
        raise ValueError(f"{run_dir.name}: no lead_current in {logs[-1].name}")
    conv = CONVERGED.search(text)
    ne = NE.search(text)

    mask = audit(cfg_path)
    from quatrex.core.config import parse_config
    cfg = parse_config(cfg_path)

    # thickness = the full transport period times the cell count (the device
    # length), not the atom span the mask test uses.
    struct = Path(cfg.input_dir) / "structure.xyz"
    if not struct.exists():
        for cand in (REPO / "cluster" / Path(cfg.input_dir).name, run_dir):
            if (cand / "structure.xyz").exists():
                struct = cand / "structure.xyz"
                break
    header = struct.read_text().splitlines()[1]
    lat = [float(x) for x in re.search(r'Lattice="([^"]+)"', header).group(1).split()]
    a1, a2, a3 = np.array(lat[0:3]), np.array(lat[3:6]), np.array(lat[6:9])
    axis = "xyz".index(cfg.device.transport_direction)
    period = abs([a1, a2, a3][axis][axis])
    n_cells = int(cfg.device.num_transport_cells)
    area_ang2 = float(np.linalg.norm(np.cross(a1, a2)))

    n_freq = int(ne.group(1)) if ne else int(cfg.electron.energy_window_num)
    df = (float(cfg.electron.energy_window_max)
          - float(cfg.electron.energy_window_min)) / (n_freq - 1)
    n_q = int(np.prod(cfg.device.kpoint_grid))
    d_t = float(cfg.phonon.left_temperature) - float(cfg.phonon.right_temperature)

    g = (H_PLANCK * 1e24 * float(cur[-1]) * df
         / (area_ang2 * 1e-20 * d_t * n_q))

    # A run that stopped at max_iterations still reports its LAST iterate
    # (run.py says so explicitly), so the number is quotable -- but only with
    # a bar on it. The per-iteration currents give one directly: the half
    # range over the last TAIL iterations is how much the observable was
    # still moving when the loop stopped. On a converged run this is a
    # consistency check, not the error bar.
    iter_j = [float(x) for x in ITER_CURRENT.findall(text)]
    iter_res = [float(x) for x in ITER_RESIDUAL.findall(text)]
    tail = iter_j[-TAIL:]
    j_halfrange = (max(tail) - min(tail)) / 2.0 if len(tail) > 1 else None

    return {
        "name": run_dir.name,
        "t_nm": n_cells * period / 10.0,
        "j_raw": float(cur[-1]),
        "iters": int(conv.group(1)) if conv else None,
        "converged": conv is not None,
        "n_iter_logged": len(iter_j),
        "j_halfrange": j_halfrange,
        "j_rel_halfrange": (j_halfrange / abs(float(cur[-1]))
                            if j_halfrange is not None else None),
        "last_residual": iter_res[-1] if iter_res else None,
        "ne": n_freq, "df": df, "n_q": n_q, "dT": d_t,
        "area_ang2": area_ang2,
        "G": g, "R_m2KGW": 1e9 / g,
        "mask_active": mask["active"], "fill": mask["fill"],
        "cutoff": mask["cutoff"], "span": mask["span"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--allow-masked", action="store_true",
                   help="fit even runs whose box mask is active (they are not "
                        "physical -- see bubble_positivity.md Sec. 6.10c)")
    p.add_argument("--allow-unconverged", action="store_true",
                   help="fit runs that stopped at max_iterations, using the "
                        "last iterate and the spread of the last "
                        f"{TAIL} per-iteration currents as its uncertainty. "
                        "A separate switch from --allow-masked on purpose: "
                        "a mask-active run is unphysical at any residual, "
                        "while an unconverged one is a physical number with "
                        "a bar on it, and one flag for both would hide the "
                        "unphysical case behind the tolerable one.")
    args = p.parse_args(argv)

    rows = [read_run(r) for r in args.runs]
    rows.sort(key=lambda r: r["t_nm"])

    print(f"{'run':<8} {'t [nm]':>7} {'J_raw':>12} {'it':>4} "
          f"{'cutoff/span':>13} {'fill':>6} {'R [m2K/GW]':>11} {'+-':>8}")
    for r in rows:
        bar = ("" if r["j_rel_halfrange"] is None
               else f"{r['j_rel_halfrange']:>7.2%}")
        print(f"{r['name']:<8} {r['t_nm']:>7.4f} {r['j_raw']:>12.4f} "
              f"{(r['iters'] if r['iters'] is not None else -1):>4} "
              f"{r['cutoff']:>6.1f}/{r['span']:<6.2f} {r['fill']:>6.3f} "
              f"{r['R_m2KGW']:>11.4f} {bar:>8}")
    for r in rows:
        if not r["converged"]:
            print(f"  {r['name']}: NOT converged -- last iterate of "
                  f"{r['n_iter_logged']} logged, residual "
                  f"{r['last_residual']:.3e}"
                  + ("" if r["j_rel_halfrange"] is None else
                     f", current moving {r['j_rel_halfrange']:.2%} over the "
                     f"last {TAIL} iterations"))

    masked = [r for r in rows if r["mask_active"]]
    unconv = [r for r in rows if not r["converged"]]
    if masked and not args.allow_masked:
        for r in masked:
            print(f"  refusing to fit {r['name']}: box mask ACTIVE")
        return 1
    if unconv and not args.allow_unconverged:
        for r in unconv:
            print(f"  refusing to fit {r['name']}: did not converge "
                  f"(pass --allow-unconverged to fit it with a bar)")
        return 1
    if unconv and any(r["j_rel_halfrange"] is None for r in unconv):
        for r in unconv:
            if r["j_rel_halfrange"] is None:
                print(f"  refusing to fit {r['name']}: unconverged AND its "
                      f"log predates the per-iteration lead-current print, "
                      f"so there is nothing to put a bar on")
        return 1

    consts = {(r["ne"], round(r["df"], 9), r["n_q"], r["dT"],
               round(r["area_ang2"], 6)) for r in rows}
    if len(consts) != 1:
        print("\n  WARNING: the runs do not share (ne, df, N_q, dT, area); "
              "the bridge constants no longer cancel in the ratio:")
        for c in sorted(consts):
            print(f"    {c}")

    t = np.array([r["t_nm"] for r in rows]) * 1e-9
    R = np.array([r["R_m2KGW"] for r in rows]) * 1e-9
    if len(rows) < 2:
        return 0
    if len(set(np.round(t, 15))) < 2:
        # Two runs at the SAME thickness are an A/B at fixed length, not a
        # ladder: R(t) = R_c + t/kappa has no slope to fit and polyfit will
        # happily return a number anyway.
        print("\n  same-thickness runs -- this is a fixed-length A/B, not a "
              "ladder; no R(t) fit. Ratio of resistances:")
        ref = rows[0]
        for r in rows[1:]:
            d = (r["R_m2KGW"] - ref["R_m2KGW"]) / ref["R_m2KGW"]
            print(f"    {r['name']} / {ref['name']}: "
                  f"{r['R_m2KGW']:.4f} vs {ref['R_m2KGW']:.4f} m2K/GW "
                  f"({d:+.2%})")
        return 0
    slope, intercept = np.polyfit(t, R, 1)
    kappa, r_c = 1.0 / slope, intercept
    print(f"\nfit over {len(rows)} points: R(t) = R_c + t/kappa_bulk")
    print(f"  kappa_bulk = {kappa:.4f} W/m/K")
    print(f"  R_c        = {r_c * 1e9:.4f} m2K/GW")

    # Propagate the per-point bars. R = 1e9/G and G is linear in J, so a
    # relative half-range on J is the same relative half-range on R. Two
    # points leave no residual to estimate scatter from, so the bar has to
    # come from the runs themselves; a parametric bootstrap carries it
    # through the reciprocal of the slope without a linearisation.
    sig = np.array([(r["j_rel_halfrange"] or 0.0) * abs(r["R_m2KGW"]) * 1e-9
                    for r in rows])
    if np.any(sig > 0):
        rng = np.random.default_rng(0)
        draws = R[None, :] + rng.normal(0.0, 1.0, (4000, len(R))) * sig[None, :]
        ks, rcs = [], []
        for row in draws:
            sl, ic = np.polyfit(t, row, 1)
            if sl > 0:                       # a negative slope is unphysical
                ks.append(1.0 / sl)
                rcs.append(ic * 1e9)
        if len(ks) > 100:
            lo, hi = np.percentile(ks, [16, 84])
            rlo, rhi = np.percentile(rcs, [16, 84])
            spreads = ", ".join(
                "{} {:.2%}".format(r["name"], r["j_rel_halfrange"] or 0.0)
                for r in rows)
            print(f"  bar from the per-run current spread ({spreads}):")
            print(f"    kappa_bulk in [{lo:.4f}, {hi:.4f}] W/m/K (68%), "
                  f"{len(ks)}/{len(draws)} draws physical")
            print(f"    R_c        in [{rlo:.4f}, {rhi:.4f}] m2K/GW")
        else:
            print("  bar: the current spread is large enough that most "
                  "bootstrap draws give an unphysical (negative) slope -- "
                  "this ladder does not determine kappa_bulk")
    pred = slope * t + intercept
    resid = (R - pred) / R
    print(f"  residuals  = {', '.join(f'{x:+.3%}' for x in resid)}")
    if len(rows) > 2:
        print(f"  max |residual| = {np.abs(resid).max():.3%} over "
              f"{len(rows) - 2} degree(s) of freedom -- this is the first "
              f"test of linearity, not just a fit through the points.")
        pair = np.polyfit(t[:2], R[:2], 1)
        print(f"  two-point (shortest pair) would give "
              f"kappa_bulk = {1 / pair[0]:.4f} W/m/K, "
              f"R_c = {pair[1] * 1e9:.4f} m2K/GW")
    for r in rows:
        print(f"  {r['name']}: R_c is {r_c / (r['R_m2KGW'] * 1e-9):.1%} of R, "
              f"kappa_z,eff = {r['t_nm'] * 1e-9 / (r['R_m2KGW'] * 1e-9):.4f} W/m/K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
