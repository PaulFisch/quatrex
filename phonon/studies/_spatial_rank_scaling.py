"""How the spatial rank scales -- with frequency, with device length, with DOF.

The three numbers the matrix-free programme turns on, measured on one bed each
time so they are comparable.

**Joint spatial-frequency rank.** The proposal wants
``DeltaSigma(omega) = sum_a F_a d_a(omega)`` with the spatial operators ``F_a``
frequency INDEPENDENT, so the Hilbert transform acts only on the small
coefficient functions and the retarded reconstruction stays causal. The rank of
that decomposition is the rank of ``DeltaSigma`` arranged as (spatial-operator
element x frequency). Measured on a dispersive chain it is about half the
frequency count and the singular spectrum is flat -- but a dispersive chain is
the WRONG bed for the question, because its modal range varies by two orders
across the band and the spatial structure turns over within a few samples. A
sharp line is the case where one basis might serve many frequencies, and that is
what ``gapped_flatband_chain`` and ``gapped_sharp_pair_chain`` are for.

**Off-diagonal (semiseparable) rank.** The rank a bidirectional generator
representation would need: the largest rank of the corner block
``M[i >= k, j < k]`` over the interior splits ``k``. That is the quasiseparable
rank by definition, and it is what sets the augmented Dyson block size
``d + r+ + r-`` -- which has to beat a 2-cell reblock's ``2d`` for the whole
architecture to be worth building.

**Both against device length and against DOF per block.** ``G^R`` and ``G^<``
already saturate with length on a 1-DOF chain; ``Sigma`` grows and 20 cells
cannot distinguish saturation from slow linear growth. Nothing at all is known
about the DOF scaling, and that is the one the cost case rests on.

Every rank is reported with the cap it could not exceed beside it: a Hankel or
corner-block rank equal to its own matrix dimension is a lower bound, not a
measurement.

Run:
    QTX_ARRAY_MODULE=numpy python phonon/studies/_spatial_rank_scaling.py \
        --bed flatband --cells 20 --cubic 3e17
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QTX_ARRAY_MODULE", "numpy")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "phonon")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):        # pragma: no cover
    pass

from studies._spatial_bed import FrozenBed, OUT                  # noqa: E402

TOLS = (1e-2, 1e-3, 1e-4)


def numerical_rank(mat, tol: float) -> int:
    """Energy-based rank: smallest ``r`` with tail energy below ``tol^2``."""
    if min(np.shape(mat)) == 0:
        return 0
    sv = np.linalg.svd(mat, compute_uv=False)
    if sv[0] <= 0.0:
        return 0
    sv = sv / sv[0]
    tail = np.cumsum(sv[::-1] ** 2)[::-1]
    total = float(np.sum(sv ** 2))
    for r in range(sv.size + 1):
        if (float(tail[r]) if r < sv.size else 0.0) / total < tol ** 2:
            return r
    return int(sv.size)


def live_frequencies(mat, pos_mask, rel: float = 1e-4):
    """Positive-frequency samples that carry weight.

    A sample four orders below the peak contributes nothing to any observable
    but occupies a column of the spatial-frequency matrix, so the rank has to
    be quoted over the samples that matter.
    """
    mag = np.abs(mat).max(axis=tuple(range(1, mat.ndim)))
    mag = np.where(np.asarray(pos_mask), mag, 0.0)
    if mag.max() <= 0.0:
        return np.zeros(0, dtype=int)
    return np.where(mag > rel * mag.max())[0]


def joint_rank(mat, bed: FrozenBed, cols=None, tail_only: bool = False,
               m_edge: int = 2, r0: int | None = None):
    r"""Rank of ``mat`` as (spatial-operator element x frequency).

    ``tail_only`` restricts the spatial axis to interior blocks in the pure-tail
    region, which is the part a modal representation would actually carry.
    """
    nd, n, p = bed.n_dof, bed.n_slabs, bed.p
    r0 = (4 + 2 * p) if r0 is None else int(r0)
    rows = []
    for i in range(n):
        for j in range(n):
            if tail_only:
                if abs(i - j) < r0:
                    continue
                if not (p + m_edge <= min(i, j)
                        and max(i, j) <= n - 1 - p - m_edge):
                    continue
            rows.append(mat[:, i * nd:(i + 1) * nd,
                            j * nd:(j + 1) * nd].reshape(mat.shape[0], -1))
    if not rows:
        return None, None, (0, 0)
    x = np.concatenate(rows, axis=1).T
    if cols is not None:
        x = x[:, cols]
    sv = np.linalg.svd(x, compute_uv=False)
    sv = sv / (sv[0] if sv[0] > 0 else 1.0)
    return {t: numerical_rank(x, t) for t in TOLS}, sv, x.shape


def offdiag_rank(mat, bed: FrozenBed, iws, m_edge: int = 2):
    r"""Quasiseparable rank: ``max_k rank(M[i >= k, j < k])`` over interior
    splits, median over the given frequencies. Returns ``(ladder, cap)``."""
    nd, n, p = bed.n_dof, bed.n_slabs, bed.p
    lo, hi = p + m_edge, n - p - m_edge
    if hi - lo < 2:
        return None, 0
    cap = min(hi, n - lo) * nd
    out = {t: [] for t in TOLS}
    for iw in iws:
        a = mat[iw]
        for t in TOLS:
            out[t].append(max(numerical_rank(a[k * nd:, :k * nd], t)
                              for k in range(lo, hi)))
    return {t: int(np.median(v)) for t, v in out.items()}, cap


def subspace_turn(x, k: int = 3, step: int = 3):
    """Principal angle between the dominant ``k``-dim spatial subspace at
    frequencies ``step`` apart -- why one basis does or does not serve many."""
    ang = []

    def sub(c):
        u, _, _ = np.linalg.svd(x[:, max(0, c - 2):c + 3], full_matrices=False)
        return u[:, :k]

    for c in range(4, x.shape[1] - 4, max(1, x.shape[1] // 20)):
        s = np.linalg.svd(sub(c).conj().T @ sub(c + step), compute_uv=False)
        ang.append(float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0)))))
    return (float(np.median(ang)), float(max(ang))) if ang else (np.nan, np.nan)


def windowed_totals(x, n_windows):
    """Total state count if a separate basis is fitted per frequency window.

    The proposal's fallback when one global basis is too large. If the total
    rises with the window count the fallback is a repackaging, not a saving.
    """
    out = {}
    for nw in n_windows:
        chunks = [c for c in np.array_split(np.arange(x.shape[1]), nw)
                  if c.size > 2]
        if not chunks:
            continue
        out[nw] = sum(numerical_rank(x[:, c], 1e-3) for c in chunks)
    return out


def measure(bed: FrozenBed, *, m_edge: int = 2, n_threads=None,
            n_freq_samples: int = 12):
    from studies._spatial_tail_tails import sigma_dense

    sl, sg, _ = sigma_dense(bed, sigma_cutoff=None, g_cutoff=None,
                            n_threads=n_threads)
    d_sigma = sg - sl
    live = live_frequencies(d_sigma, bed.pos_mask)
    if live.size < 4:
        raise RuntimeError(f"{bed.name}: only {live.size} frequencies carry "
                           "weight; the bed produced no self-energy")
    iws = live[:: max(1, live.size // n_freq_samples)]

    jr_all, sv_all, shape_all = joint_rank(d_sigma, bed, cols=live)
    jr_tail, _, shape_tail = joint_rank(d_sigma, bed, cols=live, tail_only=True,
                                        m_edge=m_edge)
    rows = [d_sigma[:, i * bed.n_dof:(i + 1) * bed.n_dof,
                    j * bed.n_dof:(j + 1) * bed.n_dof].reshape(
                        d_sigma.shape[0], -1)
            for i in range(bed.n_slabs) for j in range(bed.n_slabs)]
    x = np.concatenate(rows, axis=1).T[:, live]

    od = {}
    for name, mat in (("G^R", bed.g_retarded), ("G^<", bed.g_lesser),
                      ("Sigma^<", sl), ("Delta", d_sigma)):
        od[name] = offdiag_rank(mat, bed, iws, m_edge=m_edge)
    return {
        "live": live, "n_freq": int(bed.freqs_thz.size),
        "joint_all": jr_all, "joint_tail": jr_tail,
        "shape_all": shape_all, "shape_tail": shape_tail,
        "spectrum": sv_all[:8], "turn": subspace_turn(x),
        "windows": windowed_totals(x, (1, 2, 4, 8, 16, 32)),
        "offdiag": od, "m_edge": m_edge,
    }


def report(bed: FrozenBed, res: dict) -> None:
    m = bed.meta
    print(f"\n{bed.name}: {bed.n_slabs} cells x {bed.n_dof} dof, p={bed.p}, "
          f"{res['n_freq']} freqs ({res['live'].size} carry weight)")
    print(f"  frozen state: converged={m.get('converged')} "
          f"resid={m.get('scba_residual', float('nan')):.2e} "
          f"conservation={m.get('conservation_err', float('nan')):.2e} "
          f"cubic={m.get('cubic')} vertex={m.get('vertex')}")

    nl = res["live"].size
    print("\n  JOINT spatial-frequency rank of Delta -- how many FREQUENCY-"
          "INDEPENDENT\n  spatial operators a common basis would need "
          f"(the frequency count is {nl})")
    for lbl, lad, shp in (("whole matrix", res["joint_all"], res["shape_all"]),
                          ("pure tail   ", res["joint_tail"], res["shape_tail"])):
        if lad is None:
            print(f"    {lbl}: no admissible blocks")
            continue
        frac = " ".join(f"{lad[t]:4d} ({100 * lad[t] / max(nl, 1):5.1f} %)"
                        for t in TOLS)
        print(f"    {lbl} {str(shp):>14s} | " + frac)
    print("    singular values: "
          + " ".join(f"{s:.3f}" for s in res["spectrum"]))
    med, mx = res["turn"]
    print(f"    dominant 3-dim subspace turns {med:.1f} deg (max {mx:.1f}) "
          "between frequencies 3 samples apart")
    if res["windows"]:
        one = res["windows"].get(1, 0)
        print("    windowed fallback, TOTAL states at 1e-3: "
              + "  ".join(f"{nw}w:{v}" for nw, v in sorted(res["windows"].items()))
              + (f"   (one basis: {one}; rising means the fallback is a "
                 "repackaging)" if one else ""))

    print("\n  OFF-DIAGONAL (semiseparable) rank, median over frequency")
    print("    object    r@1e-2 r@1e-3 r@1e-4 | cap | augmented block d+2r@1e-2 "
          "| 2-cell reblock")
    for name, (lad, cap) in res["offdiag"].items():
        if lad is None:
            print(f"    {name}: device too short for an interior split")
            continue
        aug = (f"{bed.n_dof + 2 * lad[1e-2]:24d} | {2 * bed.n_dof:14d}"
               if name == "Sigma^<" else " " * 24 + " | " + " " * 14)
        print(f"    {name:9s} " + " ".join(f"{lad[t]:6d}" for t in TOLS)
              + f" | {cap:3d} | {aug}")
    print()


# `eps_flat` is the flat band's coupling to the lead along transport. At
# exactly zero the sharp-pair bed is INERT: its two flat bands couple only to
# each other through the cubic vertex and to nothing else at all, so neither
# has a source, Sigma is identically zero, and the SCBA "converges" at
# iteration one with residual 0.0. A small width is what lets the leads
# populate the pair; it is the docstring's own contact-broadened control, and
# it still leaves a line far narrower than the dispersive band.
BEDS = {
    "flatband": ("gapped_flatband_chain", {}),
    "sharppair": ("gapped_sharp_pair_chain", {"eps_flat": 0.05}),
}


def build(bed_name: str, n_dof: int, cells: int, cubic: float,
          eps_flat: float | None = None, **kw):
    from solver import toy_models as tm
    from studies._spatial_bed import build_frozen_chain

    if bed_name == "multi":
        return build_frozen_chain(tm.gapped_multi_chain(n_dof), cells,
                                  cubic=cubic, vertex="random", **kw)
    if bed_name == "chain":
        return build_frozen_chain(tm.gapped_chain(), cells, cubic=cubic,
                                  vertex="random", **kw)
    # The sharp-line models carry the vertex STRUCTURE -- which modes couple --
    # and that is the whole point of them, so the model is built at unit
    # amplitude and `build_frozen_chain` supplies the scale. Building it at the
    # constructor's default of 0.0 and then multiplying gives a vertex that is
    # identically zero, a self-energy that is exactly zero, and an SCBA that
    # "converges" at iteration one with residual 0.0.
    fn, extra = BEDS[bed_name]
    extra = dict(extra)
    if eps_flat is not None:
        extra["eps_flat"] = eps_flat
    model = getattr(tm, fn)(cubic=1.0, **extra)
    return build_frozen_chain(model, cells, cubic=cubic, vertex="model", **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bed", default="flatband",
                    choices=["chain", "flatband", "sharppair", "multi"])
    ap.add_argument("--dof", type=int, default=2,
                    help="DOF per cell; only used by --bed multi")
    ap.add_argument("--cells", type=int, default=20)
    ap.add_argument("--cubic", type=float, default=1e17)
    ap.add_argument("--eps-flat", type=float, default=None,
                    help="flat-band coupling to the lead along transport; at "
                         "exactly 0 the sharp-pair bed is inert")
    ap.add_argument("--nfreq-pos", type=int, default=120)
    ap.add_argument("--max-iter", type=int, default=600)
    ap.add_argument("--mixing", type=float, default=0.2)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--m-edge", type=int, default=2)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--load", type=Path, default=None,
                    help="measure an existing frozen bed instead of building")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    if a.load is not None:
        bed = FrozenBed.load(a.load)
    else:
        bed = build(a.bed, a.dof, a.cells, a.cubic, eps_flat=a.eps_flat,
                    nfreq_pos=a.nfreq_pos,
                    max_scba_iter=a.max_iter, solver="linear",
                    anderson_mixing=False, mixing=a.mixing, scba_tol=a.tol,
                    divergence_guard=False, n_threads=a.threads, verbose=True,
                    name=(f"{a.bed}{a.dof if a.bed == 'multi' else ''}"
                          f"_L{a.cells}"))
        if a.save is not None:
            bed.save(a.save)
            print(f"  saved bed to {a.save}")

    res = measure(bed, m_edge=a.m_edge, n_threads=a.threads)
    report(bed, res)

    out = a.out or (OUT.parent / "spatial_rank" / f"{bed.name}_rank.npz")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, name=bed.name, n_slabs=bed.n_slabs, n_dof=bed.n_dof, p=bed.p,
        n_freq=res["n_freq"], n_live=res["live"].size,
        joint_all=np.array([res["joint_all"][t] for t in TOLS]),
        joint_tail=np.array([res["joint_tail"][t] for t in TOLS]
                            if res["joint_tail"] else [-1, -1, -1]),
        spectrum=res["spectrum"], turn=np.array(res["turn"]),
        windows=np.array(sorted(res["windows"].items())),
        offdiag=np.array([[res["offdiag"][k][0][t] for t in TOLS]
                          if res["offdiag"][k][0] else [-1, -1, -1]
                          for k in res["offdiag"]]),
        offdiag_names=np.array(list(res["offdiag"])),
        offdiag_caps=np.array([res["offdiag"][k][1] for k in res["offdiag"]]),
        meta=np.array(repr(bed.meta)))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
