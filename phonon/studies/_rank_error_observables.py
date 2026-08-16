"""Self-consistent observable error of the decomposed SSE vs the full-vertex
SSE.

Usage:
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Sigma is bilinear in the vertex; the two vertices differ by 6.8e-5 in amplitude.
FLOOR = 1.4e-4

# (key, label) -- physically meaningful, in reporting order.
OBSERVABLES = (
    ("last_heat", "heat J"),
    ("lead_current", "G_th"),
    ("final_heat", "J(q)"),
    ("current_spectrum", "J(w,q)"),
    ("slab_absorption", "P_abs(x)"),
    ("gr_diag_imag", "LDOS"),
    ("gl_diag_imag", "G< diag"),
    ("occupation", "n(w)"),
    ("bubble_balance_spectrum", "bubble(w)"),
    ("internal_spread", "spread"),
    ("iter_sigma_max", "|Sigma|"),
    ("iter_heat", "J(iter)"),
)

# Run metadata -- asserted equal, never compared as an observable.
METADATA = ("energies", "eta", "retarded", "nblocks", "phonon", "ballistic",
            "t_left", "t_right", "nranks", "q_comm_size", "block_comm_size")


def _real(a):
    a = np.asarray(a)
    return np.abs(a) if np.iscomplexobj(a) else a.astype(float)


def _rel(a, b, norm="max"):
    """Relative error, NaN-aware. norm='max' (sup) or 'l2'."""
    if a is None or b is None:
        return float("nan")
    a, b = _real(a), _real(b)
    if a.shape != b.shape or b.size == 0:
        # iter_* legitimately differ in length; compare the common prefix.
        if a.ndim == 1 and b.ndim == 1 and a.size and b.size:
            n = min(a.size, b.size)
            a, b = a[:n], b[:n]
        else:
            return float("nan")
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return float("nan")
    a, b = a[finite], b[finite]
    if norm == "l2":
        denom = float(np.sqrt(np.sum(b ** 2)))
        return float("nan") if denom == 0 else float(np.sqrt(np.sum((a - b) ** 2)) / denom)
    denom = float(np.max(np.abs(b)))
    return float("nan") if denom == 0 else float(np.max(np.abs(a - b)) / denom)


def _occupation(npz):
    """n(w) ~ G^< / (-2 Im G^R): the local occupation, where there is spectrum."""
    gl, gr = _real(npz["gl_diag_imag"]), _real(npz["gr_diag_imag"])
    live = np.abs(gr) > 1e-3 * np.max(np.abs(gr))
    out = np.full_like(gl, np.nan)
    out[live] = gl[live] / (2.0 * gr[live])
    return out


def _get(npz, key):
    if key == "occupation":
        if not {"gl_diag_imag", "gr_diag_imag"} <= set(npz.files):
            return None
        return _occupation(npz)
    if key == "slab_absorption" and key in npz.files:
        # slot [1] is the experimental diagonal-only attribution -- not an observable
        return _real(np.asarray(npz[key])[0])
    return npz[key] if key in npz.files else None


def _bubble_residual(npz):
    """|P_in - P_out| / (|P_in| + |P_out|): the Phi-derivability (energy conservation)
    test of the vertex. An ABSOLUTE quality metric of each run, not an error vs dense."""
    if "final_bubble_balance" not in npz.files:
        return float("nan")
    pin, pout = (float(x.real) for x in
                 np.asarray(npz["final_bubble_balance"]).ravel()[:2])
    denom = abs(pin) + abs(pout)
    return float("nan") if denom == 0 else abs(pin - pout) / denom


def _guard(ref, got, rank, warn):
    for k in METADATA:
        if k not in ref.files or k not in got.files:
            continue
        a, b = np.asarray(ref[k]), np.asarray(got[k])
        if a.shape != b.shape:
            same = False
        elif a.dtype.kind in "OUS" or b.dtype.kind in "OUS":  # strings / objects
            same = bool(np.array_equal(a, b))
        else:
            same = bool(np.allclose(_real(a), _real(b), rtol=1e-12, atol=0))
        if not same:
            warn.append(f"  r{rank}: metadata '{k}' differs from the reference "
                        f"-- the runs are NOT comparable")
    for k in ("slab_absorption", "gl_diag_imag"):
        v = _get(got, k)
        if v is not None and np.max(np.abs(v)) == 0:
            warn.append(f"  r{rank}: '{k}' is identically zero (keep_g regression?)")


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

    ref = np.load(root / args.dense / "run.npz", allow_pickle=True)
    ranks = sorted(
        int(d.name[1:]) for d in root.iterdir()
        if d.is_dir() and d.name.startswith("r") and (d / "run.npz").exists()
    )
    if not ranks:
        raise SystemExit(f"no rank runs under {root}")

    runs = {r: np.load(root / f"r{r}" / "run.npz", allow_pickle=True) for r in ranks}
    present = [(k, t) for k, t in OBSERVABLES if _get(ref, k) is not None]
    tags = [t for _, t in present]

    warn: list[str] = []
    for r in ranks:
        _guard(ref, runs[r], r, warn)

    # ---- per-run absolute diagnostics -------------------------------------
    print("PER-RUN DIAGNOSTICS (absolute, not a comparison)\n")
    print(f"{'run':>6} {'iters':>6} {'conv':>6} {'heat J':>12} "
          f"{'bubble resid':>13}   (P_in vs P_out: the Phi-derivability test)")
    print("-" * 78)

    def _diag(name, z):
        heat = _get(z, "lead_current")
        heat = float(np.asarray(heat).ravel()[0]) if heat is not None else float("nan")
        it = int(np.asarray(z["n_iter"])) if "n_iter" in z.files else -1
        cv = bool(np.asarray(z["converged"])) if "converged" in z.files else False
        print(f"{name:>6} {it:>6} {str(cv):>6} {heat:>12.5f} {_bubble_residual(z):>13.2e}")

    _diag("dense", ref)
    for r in ranks:
        _diag(f"r{r}", runs[r])

    # ---- error vs the dense reference -------------------------------------
    print("\n\nOBSERVABLE ERROR vs the FULL (dense q-folded) vertex "
          "-- relative, sup-norm\n")
    head = f"{'R':>4} {'eps_R (FC3)':>12} |" + "".join(f"{t:>11}" for t in tags)
    print(head)
    print("-" * len(head))
    rows = []
    for r in ranks:
        row = {t: _rel(_get(runs[r], k), _get(ref, k)) for k, t in present}
        row["R"], row["eps"] = r, eps.get(r, float("nan"))
        rows.append(row)
        print(f"{r:>4} {row['eps']:>11.2%} |"
              + "".join(f"{row[t]:>11.1e}" for t in tags))

    print(f"\n[comparison floor ~{FLOOR:.0e}: the dense and factored vertices carry "
          f"different FC3 block support\n (7 vs 25 offset pairs, ~6.8e-5 of the "
          f"amplitude). Errors at or below it measure the\n REFERENCE, not the rank.]")

    # ---- amplification ----------------------------------------------------
    print("\n\nAMPLIFICATION -- observable error / FC3 fit residual\n")
    head2 = f"{'R':>4} |" + "".join(f"{t:>11}" for t in tags)
    print(head2)
    print("-" * len(head2))
    for row in rows:
        e = row["eps"]
        print(f"{row['R']:>4} |" + "".join(
            f"{row[t] / e:>11.3f}" if e and np.isfinite(row[t]) else f"{'--':>11}"
            for t in tags))

    print("\n< 1 means the PHYSICS is less sensitive than the tensor residual: the")
    print("components the CP fit discards are the ones the bubble weights least.")
    print("The theory chapter's bound is 2*eps_R, i.e. an amplification of 2.")

    # ---- resolved breakdown of the spectral current ------------------------
    cs_ref = _get(ref, "current_spectrum")
    if cs_ref is not None and cs_ref.ndim >= 2:
        print("\n\nRESOLVED ERROR of the spectral current J(w,q)")
        print("(sup-norm after integrating out the OTHER axis -- shows where the rank")
        print(" error actually lives)\n")
        print(f"{'R':>4} | {'err after q-sum: J(w)':>22} | {'err after w-sum: J(q)':>22}")
        print("-" * 56)
        qax = tuple(range(1, cs_ref.ndim))
        for r in ranks:
            cs = _get(runs[r], "current_spectrum")
            if cs is None or cs.shape != cs_ref.shape:
                continue
            a, b = _real(cs), _real(cs_ref)
            e_w = _rel(a.sum(axis=qax), b.sum(axis=qax))
            e_q = _rel(a.sum(axis=0), b.sum(axis=0))
            print(f"{r:>4} | {e_w:>22.2e} | {e_q:>22.2e}")

    if warn:
        print("\n\nWARNINGS")
        print("\n".join(warn))


if __name__ == "__main__":
    main()
