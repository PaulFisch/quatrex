"""Measure the near-DC singularity class of the device Green's functions
from recorded engine data (QX_SAVE_DIAG_G arrays).

Establishes the empirical exponents the IR derivation must reproduce
(phonon/docs/ir_residue_derivation.md, D1). On the ballistic MoS2 film
(cluster/mos2f3nu/run_ballistic.npz, 2026-08-04 measurement):

    |G^<|  ~ omega^-2   (log-log slopes -1.999, -1.982, -1.951 at q=Gamma)
    |G^R|  ~ omega^-1   (slopes -0.986, -0.960, -0.920)
    gapped q != Gamma bins carry ~1e-21 -- the singular channel lives
    entirely in the q=Gamma acoustic subspace.

i.e. G^{<,>} = -i C2/w^2 -/+ i C1/w + O(1): an EVEN double pole from
n(w)~kT/hw times A(w)~1/w, with an odd 1/w subleading part. This
falsifies all three prior recorded assumptions (bounded legs with
A(0)=0 in the purged infrared appendix; simple 1/w contact pole in
_ir_bubble_probe.py; the |G^<|~w^-0.5 note of commit b009ebcc).

Interacting iterates (e.g. cluster/mos2f3long/run.npz) are resonance-
dominated near DC and NOT power-law fittable -- exponent fits on
diverging iterates measure the disease, not the physics; only
ballistic data is fit here.

Run:  python phonon/studies/_ir_exponents.py [path/to/run_ballistic.npz]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "cluster/mos2f3nu/run_ballistic.npz"
OUT = ROOT / "phonon/studies/out/ir_residue"


def fit_slopes(w: np.ndarray, v: np.ndarray, nfit: int = 7) -> np.ndarray:
    """Consecutive log-log slopes over the first nfit finite bins."""
    m = (w > 0) & (v > 0)
    w, v = w[m][:nfit], v[m][:nfit]
    return np.diff(np.log(v)) / np.diff(np.log(w))


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    d = np.load(path)
    en = d["energies"]
    report = {"file": str(path), "first_bins_thz": en[:6].tolist(),
              "eta": float(d["eta"]) if "eta" in d else None,
              "t_left": float(d["t_left"]) if "t_left" in d else None,
              "t_right": float(d["t_right"]) if "t_right" in d else None}

    for key, label in (("gl_diag_imag", "G^<"), ("gr_diag_imag", "G^R")):
        a = d[key]  # (nw, nq1, nq2, ndof)
        m00 = np.abs(a[:, 0, 0, :]).max(axis=-1)
        slopes = fit_slopes(en, m00)
        # largest magnitude among the gapped (q != Gamma) points, first bins
        gap = np.abs(a[:6, 1:, :, :]).max()
        gap = max(gap, np.abs(a[:6, :1, 1:, :]).max())
        report[label] = {
            "q0_first_vals": m00[:5].tolist(),
            "q0_loglog_slopes": np.round(slopes, 4).tolist(),
            "gapped_q_max_first_bins": float(gap),
        }
        print(f"{label}: q=Gamma first vals {np.round(m00[1:5], 3).tolist()}"
              f"  slopes {np.round(slopes, 3).tolist()}"
              f"  gapped-q max {gap:.3e}")

    # C2 plateau: w^2 |G^<| should approach a constant as w -> 0
    gl = np.abs(d["gl_diag_imag"][:, 0, 0, :]).max(axis=-1)
    m = en > 0
    c2 = (en[m][:6] ** 2 * gl[m][:6])
    report["C2_plateau_w2_gl"] = c2.tolist()
    print(f"C2 plateau (w^2 |G^<|, first bins): {np.round(c2, 4).tolist()}")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"exponents_{path.parent.name}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
