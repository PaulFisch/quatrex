#!/usr/bin/env python
"""Read the CNT campaign arms and report every gate in one table.

Each arm changes ONE variable against ``c16-kk``.  This reduces them to the
quantities the gates are stated in, so the comparison is done on numbers
rather than on logs:

  gate (b)  effective ``sse_g_band`` (the env value, clamped to n_blocks-1)
  gate (c)  the ``_check_kk_grid_support`` percentage the run itself logged
            (sse_phonon_phonon.py:1903).  Must be silent, i.e. below 1 %.
  gate (a)  cells per block, from the block DOF
  mask      ``interaction_cutoff`` against the BLOCK BAND.  The device-span
            fill is the wrong diagnostic: the mask is a box on |dz| and the
            solver is block-tridiagonal, so only the band it retains can be
            cut.  Measured: cutoff 10 and 40 are bit-identical on CNT at one
            cell per block (c16-kk vs c16-cut40, 2026-08-28).
  outcome   iterations, final residual, lead balance, interior heat spread

Rank-locality: neither ``iter_heat``/``iter_sigma_max`` (rank-local in the
stack axis) nor ``gr_diag_imag``/``gl_diag_imag`` (rank-local in the BLOCK
axis once ``bcs > 1``) is safe to read here.  Cells per block comes from
``structure.xyz``.

Conductance comes from ``current_spectrum`` via
``phonon.postproc.units.heat_current_watts``, and kappa_eff uses the pi*d*h
shell cross-section, the same convention as cnt33_observables_atlas.py.

Deliberately does NOT read ``iter_heat`` / ``iter_sigma_max`` /
``bubble_balance_spectrum``: those are rank-0-local frequency slices and the
campaign runs on 4 ranks.  Per-iteration history comes from the log.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phonon"))

from phonon.postproc.units import heat_current_watts  # noqa: E402

CL = ROOT / "cluster"
CELL_A = 2.4595                       # (3,3) transport period
D33, SHELL = 4.07e-10, 3.35e-10
AREA = np.pi * D33 * SHELL            # pi*d*h tube shell, m^2
G_Q = 2.8393e-10                      # pi^2 kB^2 T / 3h at 300 K, W/K

ARM_ORDER = ["c16-half", "c16-fft", "c16-kk", "c16-cut40", "c16-kk-lfm",
             "c16-ball", "c16-g1", "c16-g2", "c16-ne241", "c16-ne361", "c16x2"]

RE_KK = re.compile(r"still carries ([\d.]+)% of its peak weight")
RE_CONV = re.compile(r"SCBA converged after (\d+) iterations")
RE_RESID = re.compile(r"rel Sigma\^R residual ([\d.eE+-]+); "
                      r"lead balance ([\d.eE+-]+); "
                      r"internal spread ([\d.eE+-]+)")
RE_RUN = re.compile(r"RUN config=(\S+) .*?retarded=(\S+) nblk=(\d+) ne=(\d+) "
                    r"fgrid=(\S+) bcs=(\d+) qcs=(\d+) nranks=(\d+)")
RE_ENV = re.compile(r"^RUN env (.*)$", re.M)
RE_CUT = re.compile(r"Max Interaction Cutoff: ([\d.]+)")


def device_span(n_cells: int) -> float:
    """Orbital extent along transport.  Reported for the record only -- it is
    NOT what decides whether the mask bites (see block_band)."""
    return (n_cells - 1) * CELL_A + CELL_A / 2


def mask_fill(n_cells: int, cutoff: float) -> float:
    """Fraction of orbital pairs the box mask keeps over the whole device."""
    z = np.repeat(np.concatenate(
        [np.array([0.0, CELL_A / 2]) + k * CELL_A for k in range(n_cells)]), 6)
    return float((np.abs(z[:, None] - z[None, :]) <= cutoff).mean())


def block_band(cells_per_block: int) -> float:
    """Largest |dz| the block-tridiagonal structure retains: an orbital in
    block I against one in block I+1.  A cutoff above this cannot change the
    answer, which is why cutoff 10 and 40 are bit-identical on CNT."""
    z = np.concatenate([np.array([0.0, CELL_A / 2]) + k * CELL_A
                        for k in range(cells_per_block)])
    return float(np.abs(z[:, None] - (z[None, :] + cells_per_block * CELL_A)).max())


def read_log(d: Path) -> dict:
    logs = sorted(d.glob("slurm-*.out"), key=lambda p: p.stat().st_mtime)
    logs = [p for p in logs if "_quatrex_times" not in p.name]
    if not logs:
        return {}
    txt = logs[-1].read_text(errors="replace")
    kk = [float(x) for x in RE_KK.findall(txt)]
    resid = RE_RESID.findall(txt)
    run = RE_RUN.search(txt)
    env = RE_ENV.findall(txt)
    conv = RE_CONV.search(txt)
    out = {"log": logs[-1].name, "kk": max(kk) if kk else None,
           "n_warn": len(kk), "iters": int(conv.group(1)) if conv else None,
           "env": env[-1] if env else "", "cut": None}
    m = RE_CUT.search(txt)
    if m:
        out["cut"] = float(m.group(1))
    if run:
        out.update(retarded=run.group(2), nblk=int(run.group(3)),
                   ne=int(run.group(4)), bcs=int(run.group(6)),
                   nranks=int(run.group(8)))
    if resid:
        out.update(resid=float(resid[-1][0]), lead_bal=float(resid[-1][1]),
                   spread=float(resid[-1][2]))
    for tag in ("DIVERGED", "NOT CONVERGED", "oom-kill", "TIME LIMIT"):
        if tag in txt:
            out.setdefault("flag", tag)
    return out


def read_npz(d: Path) -> dict:
    f = d / "run.npz"
    if not f.exists():
        return {}
    z = np.load(f, allow_pickle=True)
    out = {"conv": bool(z["converged"]), "n_iter": int(z["n_iter"]),
           "ballistic": bool(z["ballistic"]) if "ballistic" in z.files else False,
           "nblk": int(z["nblocks"])}
    if "lead_current" in z.files:
        out["lead_current"] = float(z["lead_current"])
    if "current_spectrum" in z.files:
        en = np.abs(np.asarray(z["energies"], float))
        cs = np.real(np.asarray(z["current_spectrum"]))
        while cs.ndim > 2:
            cs = cs.mean(axis=1)
        w = np.asarray(z["frequency_cell_widths"], float)
        tl, tr = float(z["t_left"]), float(z["t_right"])
        J = heat_current_watts(en, cs, w)
        out["G_WK"] = 0.5 * (abs(J[0]) + abs(J[-1])) / (tl - tr)
    # I_int is the archive's convention (gband_ladder.npz, written by
    # _extract_gband_ladder.py): with uniform_frequency_grid the heat keys are
    # UNWEIGHTED sums, so the integral is lead_current * dw.  Checked against
    # the committed l16f-g3 row: 38.361263551244 * 0.34375 = 13.18668.
    if "lead_current" in z.files and "frequency_cell_widths" in z.files:
        dw = float(np.asarray(z["frequency_cell_widths"], float)[1])
        uni = bool(z["uniform_frequency_grid"]) if "uniform_frequency_grid" in z.files else True
        out["I_int"] = float(z["lead_current"]) * (dw if uni else 1.0)
    # NOT from gr_diag_imag: that array is gathered over the STACK axis but
    # stays rank-local in the BLOCK axis, so at bcs=2 it covers only half the
    # device (c16x2h: 288 of 576 DOF).  The geometry is authoritative.
    xyz = f.parent / "structure.xyz"
    if xyz.exists():
        out["dof_blk"] = 3 * int(xyz.read_text().split("\n", 1)[0])
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="*", default=ARM_ORDER)
    a = p.parse_args()

    rows = []
    for name in a.arms:
        d = CL / name
        if not d.is_dir():
            continue
        r = {"name": name, **read_log(d), **read_npz(d)}
        rows.append(r)
    if not rows:
        print("no arms pulled yet -- python phonon/scripts/daint.py pull --name <arm>")
        return 1

    ball = next((r for r in rows if r.get("ballistic")), None)

    print(f"{'arm':<11} {'blk':>4} {'c/blk':>5} {'ne':>4} {'ret':>4} {'cut':>6} "
          f"{'band':>6} {'mask':>6} {'KK%':>6} {'it':>4} {'resid':>9} {'leadbal':>9} "
          f"{'spread':>7} {'I_int':>8} {'G/gQ':>7} {'k_eff':>7}")
    print("-" * 140)
    for r in rows:
        nblk = r.get("nblk")
        cpb = (r.get("dof_blk", 36) // 36) if nblk else None
        ncell = nblk * cpb if nblk and cpb else None
        cut = r.get("cut")
        band = block_band(cpb) if cpb else None
        bite = ("BITES" if (band and cut and band > cut) else "inert") if cut else "-"
        kk = f"{r['kk']:.1f}" if r.get("kk") is not None else ("ok" if r.get("log") else "-")
        G = r.get("G_WK")
        kap = (G * ncell * CELL_A * 1e-10 / AREA) if (G and ncell) else None
        st = "BALL" if r.get("ballistic") else ("" if r.get("conv") else r.get("flag", "?"))
        print(f"{r['name']:<11} {nblk or '-':>4} {cpb or '-':>5} {r.get('ne','-'):>4} "
              f"{r.get('retarded','-'):>4} {cut or '-':>6} "
              f"{(f'{band:.2f}' if band else '-'):>6} {bite:>6} {kk:>6} "
              f"{r.get('n_iter', r.get('iters','-')):>4} "
              f"{r.get('resid', float('nan')):>9.2e} "
              f"{r.get('lead_bal', float('nan')):>9.2e} "
              f"{r.get('spread', float('nan')):>7.2%} "
              f"{r.get('I_int', float('nan')):>8.3f} "
              f"{(G/G_Q) if G else float('nan'):>7.3f} "
              f"{kap if kap else float('nan'):>7.1f} {st}")

    if ball and ball.get("G_WK"):
        print(f"\nballistic reference: G = {ball['G_WK']/G_Q:.3f} g_Q "
              f"(report: 5.54 g_Q at 300 K)")
        for r in rows:
            if r.get("G_WK") and not r.get("ballistic"):
                print(f"  {r['name']:<11} G/G_ball = {r['G_WK']/ball['G_WK']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
