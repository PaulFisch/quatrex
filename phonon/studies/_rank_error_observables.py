"""Self-consistent observable error of the decomposed SSE vs the full-vertex SSE.

Companion to ``_rank_error_sse.py``. That script measures the ONE-SHOT error (both
self-energies evaluated on the same ballistic G, so the vertex error alone). This
one runs the FULL SCBA at each rank and compares every observable against the
dense q-folded-vertex run, so it includes the self-consistent feedback.

The question it answers: the CP fit residual ``eps_R`` is an error on the FC3
TENSOR. The self-energy contracts that tensor against two Green's functions, and
the observables integrate the result. Neither step need inherit ``eps_R`` --- the
components the fit discards may be the ones the bubble weights least. The
amplification table at the bottom is the answer.

Usage:
    python phonon/studies/_rank_error_observables.py --runs DIR [--dense dense]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Physically meaningful quantities, in reporting order.
OBSERVABLES = (
    ("last_heat", "heat J"),
    ("final_heat", "J(q)"),
    ("current_spectrum", "J(w,q)"),
    ("lead_current", "G_th"),
    ("slab_absorption", "P_abs"),
    ("gr_diag_imag", "LDOS"),
    ("gl_diag_imag", "G< diag"),
    ("occupation", "n(w)"),
    ("bubble_balance_spectrum", "bubble(w)"),
    ("iter_sigma_max", "|Sigma|"),
)


def _rel(a, b):
    """Relative error in the max norm, NaN-aware."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or b.size == 0:
        return float("nan")
    a, b = np.abs(a).astype(float) if np.iscomplexobj(a) else a.astype(float), (
        np.abs(b).astype(float) if np.iscomplexobj(b) else b.astype(float)
    )
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return float("nan")
    a, b = a[finite], b[finite]
    denom = np.max(np.abs(b))
    return float("nan") if denom == 0 else float(np.max(np.abs(a - b)) / denom)


def _occupation(npz):
    """n(w) ~ G^< / (-Im G^R): the local occupation, up to the spectral norm."""
    gl = np.asarray(npz["gl_diag_imag"], dtype=float)
    gr = np.asarray(npz["gr_diag_imag"], dtype=float)
    weight = np.abs(gr) > 1e-3 * np.max(np.abs(gr))  # only where there is spectrum
    out = np.full_like(gl, np.nan)
    out[weight] = gl[weight] / gr[weight]
    return out


def _get(npz, key):
    if key == "occupation":
        if "gl_diag_imag" not in npz.files or "gr_diag_imag" not in npz.files:
            return None
        return _occupation(npz)
    return npz[key] if key in npz.files else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, help="dir holding <dense|rR>/run.npz")
    p.add_argument("--dense", default="dense")
    p.add_argument("--eps", default="8:0.1554,16:0.0834,32:0.0071,64:0.0006,128:0.0004",
                   help="rank:FC3 fit residual (from the INDSCAL fit log)")
    args = p.parse_args()

    root = Path(args.runs)
    eps = {int(k): float(v) for k, v in
           (t.split(":") for t in args.eps.split(",") if t)}

    ref = np.load(root / args.dense / "run.npz")
    ranks = sorted(
        int(d.name[1:]) for d in root.iterdir()
        if d.is_dir() and d.name.startswith("r") and (d / "run.npz").exists()
    )
    if not ranks:
        raise SystemExit(f"no rank runs under {root}")

    present = [(k, t) for k, t in OBSERVABLES if _get(ref, k) is not None]

    rows = []
    for rank in ranks:
        got = np.load(root / f"r{rank}" / "run.npz")
        row = {"R": rank, "eps": eps.get(rank, float("nan"))}
        for key, tag in present:
            row[tag] = _rel(_get(got, key), _get(ref, key))
        rows.append(row)

    tags = [t for _, t in present]
    print("Observable error of the rank-R decomposed SSE against the FULL "
          "(dense q-folded) vertex.")
    print("Both runs are the same device, same settings, same number of SCBA "
          "iterations.\n")

    head = f"{'R':>4} {'eps_R (FC3)':>12} |" + "".join(f"{t:>11}" for t in tags)
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['R']:>4} {r['eps']:>11.2%} |"
              + "".join(f"{r.get(t, float('nan')):>11.1e}" for t in tags))

    print("\nAMPLIFICATION -- observable error / FC3 fit residual:")
    head2 = f"{'R':>4} |" + "".join(f"{t:>11}" for t in tags)
    print(head2)
    print("-" * len(head2))
    for r in rows:
        e = r["eps"]
        print(f"{r['R']:>4} |" + "".join(
            f"{r.get(t, float('nan')) / e:>11.3f}" if e else f"{'--':>11}"
            for t in tags))

    print("\n< 1 means the PHYSICS is less sensitive than the tensor residual:")
    print("the components the CP fit discards are the ones the bubble weights least.")
    print("The theory chapter's bound is 2*eps_R, i.e. an amplification of 2.")


if __name__ == "__main__":
    main()
