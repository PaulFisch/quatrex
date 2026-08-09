"""Why does the Si film stop converging when the grid is refined?

The Si film converged to 9.6e-08 on a 121-point grid and ABORTS at
iteration 4 with residual 2.4e+06 on a 15001-point one. Si is the system
whose interaction mask is the identity by accident (its transport axis is
x, on which the fcc cell has a 1.37 A extent, so the 10 A box cutoff never
truncates -- `bubble_positivity.md` 6.11), so the MoS2 mechanism cannot be
the cause here and something else is.

This reads the saved run.npz of a resolution ladder and reports, per rung:

* **negative occupation** -- ``min_i (-i G^<)_ii`` relative to the largest
  positive diagonal. This is the observable signature of a broken
  ``-i Sigma^<_tot >= 0``: since ``G^< = G^R Sigma^< G^A`` is a congruence,
  a negative occupation PROVES the total lesser self-energy has a negative
  eigenvalue (bubble_positivity.md section 3).
* **where** it sits in frequency, and whether it tracks the band edge.
* **registration** -- the per-orbital spectral sum rule
  ``S_i = int 2w (-1/pi) Im G^R_ii dw``, which must tend to 1 once the grid
  resolves the resonances. The grid audit's blind -> transition -> resolved
  ladder predicts that a coarse grid reports trivial stability precisely
  because S_B is small: the map is nearly ballistic and contracting while
  the answer is wrong.
* **lead balance and the iterate trace**, to separate "diverged" from
  "not yet converged".

Usage:
    QTX_ARRAY_MODULE=numpy python phonon/studies/_si_divergence.py \
        cluster/sichk_base cluster/sires501 cluster/sires1001 ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phonon/studies/out/si_divergence"


def _occupation(d) -> dict:
    """Negative-occupation metrics from the saved diagonal of G^<.

    ``gl_diag_imag`` is ``-Im G^<`` on the production sign convention, i.e.
    the occupation density; it must be >= 0 everywhere for a PSD Sigma^<.
    Bins with no spectral weight are excluded -- at eta = 0 the comb has
    nulls where every ratio is denormal noise.
    """
    w = np.asarray(d["energies"], float)
    gl = np.asarray(d["gl_diag_imag"], float)
    gl = gl.reshape(gl.shape[0], -1)                 # (ne, q*dof)
    scale = float(np.abs(gl).max())
    if scale <= 0:
        return {"worst_rel": 0.0, "omega": 0.0, "frac_bins": 0.0}
    live = np.abs(gl).max(axis=1) > 1e-6 * scale
    worst = float(gl[live].min()) if live.any() else 0.0
    iw = int(np.argmin(np.where(live[:, None], gl, np.inf).min(axis=1)))
    # fraction of live bins carrying ANY negative occupation
    frac = float((gl[live].min(axis=1) < -1e-9 * scale).mean()) if live.any() else 0.0
    return {"worst_rel": worst / scale, "omega": float(w[iw]),
            "frac_bins": frac, "scale": scale}


def _sum_rule(d) -> dict:
    """Per-orbital spectral sum rule S_i = int 2w (-1/pi) Im G^R_ii dw.

    Saturates at 1 once the grid registers the resonances; the grid audit
    uses exactly this to separate the blind regime from the resolved one.
    """
    w = np.asarray(d["energies"], float)
    gr = np.asarray(d["gr_diag_imag"], float)
    gr = gr.reshape(gr.shape[0], -1)
    cw = np.asarray(d["frequency_cell_widths"], float) if \
        "frequency_cell_widths" in d else np.gradient(w)
    s = ((2.0 * w * (-1.0 / np.pi))[:, None] * gr * cw[:, None]).sum(axis=0)
    return {"S_median": float(np.median(s)), "S_min": float(s.min()),
            "S_max": float(s.max())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dirs = [Path(a) for a in sys.argv[1:]]
    if not dirs:
        raise SystemExit(__doc__)
    rep = {}
    print(f"{'run':16s} {'nf':>6s} {'J*dw':>10s} {'conv':>5s} "
          f"{'neg occ':>10s} {'@THz':>7s} {'bins<0':>7s} "
          f"{'S_med':>7s} {'balance':>10s}")
    for d0 in dirs:
        p = d0 if d0.suffix == ".npz" else d0 / "run.npz"
        if not p.exists():
            print(f"{d0.name:16s}  (no run.npz)")
            continue
        d = np.load(p)
        w = np.asarray(d["energies"], float)
        nf = w.size
        dw = float(w[1] - w[0])
        lc = float(d["lead_current"]) if "lead_current" in d else float("nan")
        occ = _occupation(d)
        sr = _sum_rule(d)
        bal = (float(np.real(np.asarray(d["final_bubble_balance"])).max())
               if "final_bubble_balance" in d else float("nan"))
        conv = bool(d["converged"]) if "converged" in d else False
        rep[d0.name] = {"nf": nf, "J_integral": lc * dw, "converged": conv,
                        **occ, **sr, "balance": bal}
        print(f"{d0.name:16s} {nf:6d} {lc*dw:10.4f} {str(conv):>5s} "
              f"{occ['worst_rel']:10.2e} {occ['omega']:7.3f} "
              f"{100*occ['frac_bins']:6.1f}% {sr['S_median']:7.3f} "
              f"{bal:10.2e}")
    (OUT / "si_divergence.json").write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {OUT / 'si_divergence.json'}")
    print("neg occ < 0 proves -i Sigma^<_tot has a negative eigenvalue "
          "(G^< = G^R Sigma^< G^A is a congruence).")
    print("S_med -> 1 means the grid registers the resonances; a small S_med "
          "with a converged run is the BLIND regime, not a result.")


if __name__ == "__main__":
    main()
