"""Reduce the decomposed-SSE campaign run.npz set to a committable artifact.

The campaign writes ~10 MB per leg (14 legs); the report's figure contract
(PLOTS.md) requires generators to read only COMMITTED data. This script keeps
the scalars and the q-summed / q-averaged spectra -- a few hundred kB -- and
throws away the (ne, nkx, nky, N_D) fields nobody plots raw.

  in:   <runs>/{L3,L10}/{dense,ball,r8,...,r128}/run.npz   (pulled from tortin)
  out:  phonon/scripts/data/decomposed_sse.csv             (one row per leg)
        phonon/scripts/data/decomposed_sse_spectra.npz     (spectra + traces)

Run once, then commit both outputs.

    python phonon/scripts/verify/_extract_decomposed_sse.py --runs cluster
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "phonon/scripts/data"

# INDSCAL fit residuals + factor file sizes (build log, cluster/sifilm-L10-geom).
# The fit is on the BULK FC3, so these are identical at every device length.
EPS_R = {8: 0.1554, 16: 0.0834, 32: 0.0071, 64: 0.0006, 128: 0.0004}
FACTOR_BYTES = {8: 249_693, 16: 589_654, 32: 1_320_840, 64: 2_822_910,
                128: 6_300_893}
QFOLD_BYTES = {3: 365_517_812, 10: 1_559_369_198}   # the dense q-folded vertex


def _real(a):
    a = np.asarray(a)
    return a.real if np.iscomplexobj(a) else a


def _qsum(a, keep_last=True):
    """Sum out the transverse-q axes of an (ne, nkx, nky, X) field."""
    a = _real(np.asarray(a))
    while a.ndim > (2 if keep_last else 1):
        a = np.nansum(a, axis=1)
    return a


def _qmean(a):
    a = _real(np.asarray(a))
    while a.ndim > 2:
        a = np.nanmean(a, axis=1)
    return a


def _bubble_resid(z):
    if "final_bubble_balance" not in z.files:
        return float("nan")
    pin, pout = (float(x.real) for x in
                 np.asarray(z["final_bubble_balance"]).ravel()[:2])
    d = abs(pin) + abs(pout)
    return float("nan") if d == 0 else abs(pin - pout) / d


def _rel(a, b):
    """Relative sup-norm error, NaN-aware."""
    if a is None or b is None:
        return float("nan")
    a, b = np.abs(_real(a)).astype(float), np.abs(_real(b)).astype(float)
    if a.shape != b.shape or b.size == 0:
        return float("nan")
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return float("nan")
    a, b = a[ok], b[ok]
    den = float(np.max(np.abs(b)))
    return float("nan") if den == 0 else float(np.max(np.abs(a - b)) / den)


_MARK = re.compile(r"^=== (SCBA|done|BALLISTIC|done ballistic)\s+"
                   r"(?:L\d+\s+)?(r\d+|dense|ball|L3|L10)?\s*=== (.+)$")

# The per-iteration Sigma residual and lead balance the SCBA prints.
_RES = re.compile(r"rel Sigma\^R residual ([0-9.eE+-]+); lead balance ([0-9.eE+-]+)")
_BAL = re.compile(r"Bubble energy balance: .*resid=([0-9.eE+-]+)")


def _split_log(log: Path) -> tuple[dict[str, float], dict[str, dict]]:
    """Split a campaign log into per-leg wall times and per-iteration traces.

    The SCBA's own stdout is the only usable source for the residual and lead
    balance: run.npz's `iter_heat` / `iter_sigma_max` are the RANK-0-LOCAL
    frequency slice, and at 121 ranks rank 0 owns a single frequency (omega=0),
    where the heat current is identically zero. `iter_bubble_balance` is fine --
    it is all-reduced -- and is taken from the npz.
    """
    import datetime as dt
    if not log.exists():
        return {}, {}
    wall: dict[str, float] = {}
    traces: dict[str, dict] = {}
    start: dict[str, dt.datetime] = {}
    cur: str | None = None
    res: list[float] = []
    lead: list[float] = []

    def _flush():
        if cur and (res or lead):
            traces[cur] = {"residual": np.array(res), "lead_balance": np.array(lead)}

    for line in log.read_text(errors="ignore").splitlines():
        m = _MARK.match(line.strip())
        if m:
            kind, leg, when = m.groups()
            leg = (leg or "").strip()
            if kind == "BALLISTIC":
                leg = "ball"
            try:
                t = dt.datetime.strptime(when.strip(), "%a %b %d %H:%M:%S %Z %Y")
            except ValueError:
                t = None
            if kind in ("SCBA", "BALLISTIC"):
                _flush()
                cur, res, lead = leg, [], []
                if t:
                    start[leg] = t
            else:                                   # done / done ballistic
                if kind == "done ballistic":
                    leg = "ball"
                if t and leg in start:
                    wall[leg] = (t - start[leg]).total_seconds()
                _flush()
                cur = None
            continue
        m = _RES.search(line)
        if m and cur:
            res.append(float(m.group(1)))
            lead.append(float(m.group(2)))
    _flush()
    return wall, traces


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="cluster",
                   help="dir holding sifilm-L3/ and sifilm-L10/")
    args = p.parse_args()
    root = ROOT / args.runs if not Path(args.runs).is_absolute() else Path(args.runs)

    DATA.mkdir(parents=True, exist_ok=True)
    rows, spectra = [], {}

    for length, nslabs in (("L3", 3), ("L10", 10)):
        base = root / f"sifilm-{length}"
        if not base.is_dir():
            print(f"  [skip] {base} absent")
            continue
        wall, traces = {}, {}
        for logdir in (f"sifilm-{length}-campaign", f"sifilm-{length}-sweep"):
            w, t = _split_log(root / logdir / "run.log")
            wall |= w
            traces |= t

        legs = {d.name: d / "run.npz" for d in sorted(base.iterdir())
                if d.is_dir() and (d / "run.npz").exists()}
        ref = np.load(legs["dense"], allow_pickle=True) if "dense" in legs else None

        for leg, path in legs.items():
            z = np.load(path, allow_pickle=True)
            rank = int(leg[1:]) if re.fullmatch(r"r\d+", leg) else 0
            tag = f"{length}_{leg}"

            row = {
                "length": length, "nslabs": nslabs, "leg": leg, "rank": rank,
                "ballistic": bool(np.asarray(z["ballistic"])),
                "eps_R": EPS_R.get(rank, ""),
                "factor_bytes": FACTOR_BYTES.get(rank, ""),
                "qfold_bytes": QFOLD_BYTES.get(nslabs, "") if leg == "dense" else "",
                "wall_s": round(wall.get(leg, float("nan")), 1),
                "n_iter": int(np.asarray(z["n_iter"])),
                "converged": bool(np.asarray(z["converged"])),
                "lead_current": float(np.asarray(z["lead_current"])),
                "internal_spread": float(np.asarray(z["internal_spread"])),
                "bubble_resid": _bubble_resid(z),
            }
            # observable errors vs the dense reference of the SAME length
            if ref is not None and leg not in ("dense",):
                for key, name in (("last_heat", "err_heat"),
                                  ("final_heat", "err_heat_q"),
                                  ("current_spectrum", "err_j_wq"),
                                  ("gr_diag_imag", "err_ldos"),
                                  ("gl_diag_imag", "err_gl"),
                                  ("slab_absorption", "err_pabs")):
                    a = z[key] if key in z.files else None
                    b = ref[key] if key in ref.files else None
                    row[name] = _rel(a, b)
                row["err_G"] = _rel(z["lead_current"], ref["lead_current"])
            rows.append(row)

            # --- spectra, reduced ---------------------------------------
            spectra[f"{tag}/energies"] = np.asarray(z["energies"], dtype=float)
            spectra[f"{tag}/current_spectrum"] = _qsum(z["current_spectrum"])
            for k in ("gr_diag_imag", "gl_diag_imag"):
                if k in z.files:
                    spectra[f"{tag}/{k}"] = _qmean(z[k])
            # iter_heat / iter_sigma_max are deliberately NOT kept: they are the
            # rank-0-local frequency slice, and at 121 ranks that is omega=0
            # alone. The residual and lead-balance traces come from the log.
            for k in ("slab_absorption", "bubble_balance_spectrum",
                      "iter_bubble_balance", "final_heat", "last_heat"):
                if k in z.files:
                    spectra[f"{tag}/{k}"] = np.asarray(_real(z[k]), dtype=float)
            for k, v in traces.get(leg, {}).items():
                spectra[f"{tag}/trace_{k}"] = np.asarray(v, dtype=float)
            for k in ("t_left", "t_right", "eta"):
                spectra[f"{tag}/{k}"] = float(np.asarray(z[k]))

    if not rows:
        raise SystemExit("no runs found")

    cols = sorted({k for r in rows for k in r},
                  key=lambda c: (c not in ("length", "nslabs", "leg", "rank"), c))
    csv_path = DATA / "decomposed_sse.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    npz_path = DATA / "decomposed_sse_spectra.npz"
    np.savez_compressed(npz_path, **spectra)

    print(f"{csv_path.relative_to(ROOT)}  ({len(rows)} legs)")
    print(f"{npz_path.relative_to(ROOT)}  "
          f"({npz_path.stat().st_size / 1e6:.2f} MB, {len(spectra)} arrays)")
    print()
    hdr = f"{'leg':>10} {'iters':>6} {'conv':>6} {'wall s':>9} {'J':>11} {'bubble':>10}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['length']+'/'+r['leg']:>10} {r['n_iter']:>6} "
              f"{str(r['converged']):>6} {r['wall_s']:>9} "
              f"{r['lead_current']:>11.5f} {r['bubble_resid']:>10.2e}")


if __name__ == "__main__":
    main()
