"""Is the KELDYSH Green function low rank in the same spatial basis as G^R?

The second gate, and the proposal's own "most important missing test" (its
Sec. 7). The spatial complex-band pencil represents ``G^R``; the bubble uses
``G^{<,>}``, which is not the resolvent of anything. If the Keldysh object needs
a rank that grows with the device, there is nothing to compress and the
programme ends whatever the tail attribution said.

The measurement is SOURCE-RESOLVED rather than a rank taken on the total, which
costs almost nothing because ``G^R`` is already in hand:

    arm L : Sigma^{<,>} = the left contact only
    arm R : the right contact only
    arm S : the frozen anharmonic Sigma_s^{<,>} only

Two batched matmuls per arm, and by linearity ``L + R + S`` must reproduce the
frozen ``G^<`` to roundoff -- which is the arm's own correctness assert, not a
separate test. Then the pencil runs along three DIRECTIONS per arm. For a source
at ``s``,

    G^<_{IJ} = sum_{mn} U_m lambda_m^{I-s} S_mn (lambda_n^*)^{J-s} U_n^dagger ,

so along ``J`` at fixed ``I`` the exponents are the ADVANCED conjugates
``lambda_n^*``, along ``I`` at fixed ``J`` the retarded ``lambda_m``, and along
the diagonal the products ``lambda_m lambda_n^*``. Three unambiguous answers
where a single Hankel rank on the total gives one ambiguous one -- and the
residue ``(lambda_m lambda_n^*)^I`` is what makes ``Sigma`` semiseparable rather
than Toeplitz, which decides the class of the eventual state-space fit.

The quantity to watch is ``min_{mn} |1 - lambda_m lambda_n^*|``. Where it goes
to zero the geometric sum over sources degenerates into a term linear in device
length -- polynomial times geometric, a rank doubling -- and that is exactly the
weakly damped regime the whole programme is aimed at.

Also reported: the Hankel spectra of ``G^R``, ``G^<``, ``G^>`` and the
positivity factor ``Y = G^R L`` with ``-i Sigma^{<,>}_tot = L L^dagger``, so a
large rank can be attributed to a source rather than merely noted.

Run:
    QTX_ARRAY_MODULE=numpy python phonon/studies/_spatial_tail_rank.py \
        --bed phonon/studies/out/spatial_bed/si16_lin.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
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


def psd_factor(mat, *, floor: float = 0.0, sign: float = +1.0):
    r"""``L`` with ``L L^dagger = i M``, from an eigendecomposition.

    **Sign.** The proposal writes ``-i Sigma^{<,>} = L L^dagger``; this tree's
    convention is the opposite one. ``grids.boson_contact_self_energies_from_gamma``
    sets ``Sigma^< = -i n Gamma``, so ``i Sigma^<`` is the positive object, and
    measured on a frozen chain the negative spectral weight of ``i G^<`` is
    5.6e-04 against 9.994e-01 for ``-i G^<``. Getting this backwards does not
    fail loudly -- it clips almost the entire spectrum and returns a factor of
    the wrong object.

    A Cholesky would refuse a source that is only numerically semi-definite,
    which every frozen SCBA source is: the anharmonic ``Sigma_s^<`` carries real
    negative weight (11.7 % on that same chain, and the bubble's positivity is a
    known open question). The clipped fraction is returned so a caller reports
    it rather than assuming it away.
    """
    herm = sign * 1j * np.asarray(mat)
    herm = 0.5 * (herm + herm.conj().swapaxes(-1, -2))
    ev, vec = np.linalg.eigh(herm)
    clipped = float(np.sum(np.abs(ev[ev < floor])))
    total = float(np.sum(np.abs(ev)))
    ev = np.clip(ev, floor, None)
    return vec * np.sqrt(ev)[..., None, :], clipped / (total + 1e-300)


def source_arms(bed: FrozenBed):
    """``{'L','R','S'}`` -> ``(G^<, G^>)`` from that source alone, plus a check."""
    from solver.leads import solve_green_batch

    z2 = bed.freqs_thz.astype(complex) ** 2
    zero = np.zeros_like(bed.sigma_lesser)
    empty = {k: np.zeros_like(v) for k, v in bed.obc.items()}

    def solve(sig_l, sig_g):
        _, g_l, g_g = solve_green_batch(
            z2, bed.h_d, {**empty, "Sigma_L_R": bed.obc["Sigma_L_R"],
                          "Sigma_R_R": bed.obc["Sigma_R_R"]},
            bed.sigma_retarded, sig_l, sig_g)
        return g_l, g_g

    arms = {
        "L": solve(bed.obc["Sigma_L_lesser"], bed.obc["Sigma_L_greater"]),
        "R": solve(bed.obc["Sigma_R_lesser"], bed.obc["Sigma_R_greater"]),
        "S": solve(bed.sigma_lesser, bed.sigma_greater),
    }
    total_l = sum(a[0] for a in arms.values())
    resid = (np.abs(total_l - bed.g_lesser).max()
             / (np.abs(bed.g_lesser).max() + 1e-300))
    return arms, float(resid), zero


def _seq(mat, bed, anchor, span, *, along, column_is_cell: bool = True):
    """Block sequence from a dense one-frequency matrix, in one direction.

    ``column_is_cell=False`` keeps the whole column width, which is what
    ``Y = G^R L`` needs: ``L``'s columns index the SOURCE eigenbasis, not a
    transport cell, so slicing them as cells and asking for a ``J`` direction
    measures nothing. The proposal's claim about ``Y`` is that it is modal in
    ``I`` for each source column, which is the ``I`` sequence at full width.
    """
    nd = bed.n_dof

    def blk(i, j):
        return mat[i * nd:(i + 1) * nd, j * nd:(j + 1) * nd]

    if not column_is_cell:
        if along != "I":
            return None
        return [mat[(anchor + r) * nd:(anchor + r + 1) * nd, :]
                for r in range(span)]
    if along == "J":
        return [blk(anchor, anchor + r) for r in range(span)]
    if along == "I":
        return [blk(anchor + r, anchor) for r in range(span)]
    return [blk(anchor + r, anchor + r) for r in range(span)]


def run(bed: FrozenBed, *, iw: int, eps: float = 1e-6, m_edge: int = 2,
        rank: int | None = None, sigma_range_tol: float = 1e-6):
    from quatrex.phonon.spatial_hankel import (cluster_exponents,
                                               matrix_pencil, numerical_rank)
    from quatrex.phonon.spatial_modes import bloch_modes, bloch_modes_poly

    arms, resid, _ = source_arms(bed)
    anchor = bed.p + m_edge
    span = max(3, bed.n_slabs - 2 * anchor)
    # A pencil needs more samples than it has exponents to find, by a margin.
    # A rank-r estimate from an r+1 point sequence fits noise and reports it as
    # physics, which is the failure mode this whole file exists to avoid.
    if span < 8:
        warnings.warn(
            f"{bed.name}: the interior span is {span} blocks at m_edge="
            f"{m_edge}; a pencil estimate needs roughly 2r+2 samples and this "
            f"bed supports at most r ~ {max(1, (span - 2) // 2)}. The sizing "
            f"law N >= R + 2(p + m_edge) + 1 wants a longer device.",
            stacklevel=2)

    # The operator's own bands at this frequency, for the comparison -- at the
    # degree the operator ACTUALLY has. Sigma^R reaches 2p+b once the output pin
    # is off, so the spatial recurrence is sum_{n=-M}^{M} a_n lambda^n = 0 and a
    # quadratic pencil is a different operator. Comparing data-derived exponents
    # against the wrong operator is a false negative, not a null result.
    nd = bed.n_dof
    w = float(bed.freqs_thz[iw])
    sig_r = bed.sigma_retarded[iw]

    def sblk(i, j):
        return sig_r[i * nd:(i + 1) * nd, j * nd:(j + 1) * nd]

    peak = np.abs(sig_r).max()
    m_sig = 0
    for d in range(1, bed.n_slabs):
        ok = [i for i in range(bed.n_slabs - d)
              if np.abs(sblk(i, i + d)).max() > sigma_range_tol * peak]
        if ok:
            m_sig = d
    m_pencil = max(1, m_sig)

    a_blocks = []
    for n in range(-m_pencil, m_pencil + 1):
        i, j = (anchor, anchor + n) if n >= 0 else (anchor - n, anchor)
        block = -sblk(anchor, anchor + n) if n >= 0 else -sblk(anchor + (-n) * 0,
                                                               anchor)
        # Sigma^R at separation n, taken from the anchor row.
        block = -sblk(anchor, anchor + n) if 0 <= anchor + n < bed.n_slabs             else np.zeros((nd, nd), complex)
        if n == 0:
            block = block + (w * w) * np.eye(nd) - bed.d00
        elif n == 1:
            block = block - bed.d01
        elif n == -1:
            block = block - bed.d10
        a_blocks.append(block)
    modes = bloch_modes_poly(a_blocks, residual=True)
    lam = np.asarray(modes.lam)
    good = modes.converged(tol=1e-6) & np.isfinite(lam)
    dec = lam[good & (np.abs(lam) < 1.0)]

    # the quadratic reading, kept beside it so the difference is visible
    modes2 = bloch_modes((w * w) * np.eye(nd) - bed.d00 - sblk(anchor, anchor),
                         -bed.d01 - sblk(anchor, anchor + 1),
                         -bed.d10 - sblk(anchor, anchor + 1).conj().T,
                         residual=True)
    lam2 = np.asarray(modes2.lam)
    dec2 = lam2[np.abs(lam2) < 1.0]

    prod = np.abs(1.0 - np.outer(dec, np.conj(dec)))
    closeness = float(prod.min()) if prod.size else float("nan")

    out = {"w": w, "iw": iw, "linearity_residual": resid, "anchor": anchor,
           "span": span, "lam": lam, "lam_decaying": dec,
           "lam_quadratic": dec2, "m_sigma": m_sig, "m_pencil": m_pencil,
           "min_one_minus_prod": closeness, "arms": {}}

    seqs = {"G^R": bed.g_retarded[iw], "G^<": bed.g_lesser[iw],
            "G^>": bed.g_greater[iw]}
    sig_tot_l = (bed.obc["Sigma_L_lesser"][iw] + bed.obc["Sigma_R_lesser"][iw]
                 + bed.sigma_lesser[iw])
    l_fac, clipped = psd_factor(sig_tot_l)
    seqs["Y=G^R L"] = bed.g_retarded[iw] @ l_fac
    out["psd_clipped"] = clipped
    for tag, (g_l, _g) in arms.items():
        seqs[f"G^<[{tag}]"] = g_l[iw]

    for name, mat in seqs.items():
        cell_cols = not name.startswith("Y")
        entry = {}
        for along in ("J", "I", "diag"):
            sq = _seq(mat, bed, anchor, span, along=along,
                      column_is_cell=cell_cols)
            if sq is None:
                continue
            r_eps = numerical_rank(sq, eps)
            est = matrix_pencil(sq, rank=rank, eps=eps)
            uniq, mult = cluster_exponents(est.xi, 1e-4)
            # A rank equal to the Hankel's own size is "at least", not "equal
            # to": the matrix cannot express more, so the number is a property
            # of the sequence LENGTH and not of the physics. Reported at a
            # ladder of tolerances with the cap beside it, because a single
            # tolerance on a short sequence saturates and reads as a result.
            cap = est.spectrum.size
            entry[along] = {"r_eps": r_eps, "xi": uniq, "mult": mult,
                            "spectrum": est.spectrum, "cap": cap,
                            "ladder": {t: numerical_rank(sq, t)
                                       for t in (1e-2, 1e-3, 1e-4, 1e-6)},
                            "saturated": r_eps >= cap,
                            "recon": float(est.rel_error(sq).max())}
        out["arms"][name] = entry
    return out


def report(bed: FrozenBed, res: dict) -> None:
    lam = res["lam_decaying"]
    print(f"\n{bed.name} at omega = {res['w']:.4f} THz "
          f"(sample {res['iw']} of {bed.freqs_thz.size})")
    print(f"  frozen state: converged={bed.meta.get('converged')} "
          f"resid={bed.meta.get('scba_residual'):.2e}")
    print(f"  source arms L+R+S reproduce the frozen G^<: "
          f"{res['linearity_residual']:.2e}  "
          f"{'OK' if res['linearity_residual'] < 1e-10 else 'FAILED -- the arms are not the decomposition'}")
    print(f"  anchor cell {res['anchor']}, span {res['span']} blocks "
          f"(p={bed.p})")
    print(f"  Sigma^R reaches |I-J| <= {res['m_sigma']}, so the spatial pencil "
          f"is degree {2 * res['m_pencil']} (not 2)")
    if lam.size:
        print(f"  operator bands: {lam.size} decaying of {res['lam'].size} "
              f"(residual-converged), |lambda| in "
              f"[{np.abs(lam).min():.4f}, {np.abs(lam).max():.4f}]")
        q = res["lam_quadratic"]
        if q.size:
            print(f"  the QUADRATIC reading would give {q.size} decaying, "
                  f"|lambda| in [{np.abs(q).min():.4f}, {np.abs(q).max():.4f}]")
    else:
        print("  operator bands: none decaying")
    print(f"  min |1 - lambda_m lambda_n^*| = {res['min_one_minus_prod']:.3e} "
          "(-> 0 is where the source sum degenerates and the rank doubles)")
    print(f"  PSD factor L (i*Sigma convention): clipped negative weight "
          f"{res['psd_clipped']:.2e}  -- how far the frozen source is from PSD")

    print("\n  numerical rank r_eps by direction, and the exponents recovered")
    print("    object          | r(J) r(I) r(d) | recon | |xi| (largest first)")
    for name, entry in res["arms"].items():
        ref_dir = "J" if "J" in entry else "I"
        e = entry[ref_dir]
        cols = " ".join(f"{entry[d]['r_eps']:4d}" if d in entry else "   -"
                        for d in ("J", "I", "diag"))
        mods = np.sort(np.abs(e["xi"]))[::-1][:4]
        print(f"    {name:15s} | {cols} | {e['recon']:.1e} | "
              + " ".join(f"{m:.4f}" for m in mods))

    if lam.size:
        print("\n  are the operator's bands present in what the pencil found?")
        print("    (per band: distance to the NEAREST recovered exponent -- the "
              "other\n     direction, max over recovered, counts spurious roots "
              "and says nothing)")
        adv = np.conj(lam)
        for name in ("G^R", "G^<", "G^>"):
            e = res["arms"].get(name)
            if e is None:
                continue
            for along, ref, lbl in (("J", adv, "lambda*"), ("I", lam, "lambda")):
                if along not in e:
                    continue
                xi = e[along]["xi"]
                if xi.size == 0 or ref.size == 0:
                    continue
                per_band = [float(np.min(np.abs(xi - z))) for z in ref]
                covered = sum(1 for d in per_band if d < 1e-2)
                print(f"    {name:6s} along {along} vs {lbl:8s}: "
                      f"{covered}/{ref.size} bands within 1e-2, worst "
                      f"{max(per_band):.3e}")
    print()


def sweep(bed: FrozenBed, iws, **kw):
    """Run at several frequencies and summarise, because one is not a result.

    A single frequency near the band bottom is the LEAST damped point on the
    grid and is where a rank is largest and an exponent least determined; the
    quantity the gate needs is how the rank behaves across the band.
    """
    rows = []
    for iw in iws:
        try:
            r = run(bed, iw=int(iw), **kw)
        except np.linalg.LinAlgError:
            continue
        rows.append(r)
    return rows


def sweep_report(bed: FrozenBed, rows) -> None:
    if not rows:
        print("  no frequency produced a usable estimate")
        return
    names = list(rows[0]["arms"])
    print(f"\n{bed.name}: rank across {len(rows)} frequencies "
          f"({rows[0]['w']:.3f} .. {rows[-1]['w']:.3f} THz), "
          f"span {rows[0]['span']} blocks")
    print("  linearity of the source arms: worst "
          f"{max(r['linearity_residual'] for r in rows):.2e}")
    xi_g = []
    for r in rows:
        d = r["lam_decaying"]
        if d.size:
            xi_g.append(float(np.max(-1.0 / np.log(np.abs(d)))))
    if xi_g:
        print(f"  longest modal range xi over the sweep: "
              f"min {min(xi_g):.2f}  median {np.median(xi_g):.2f}  "
              f"max {max(xi_g):.2f} cells "
              f"(the device is {bed.n_slabs} cells; a range longer than that "
              "means the bed cannot show a tail)")
    print(f"  min |1 - lambda_m lambda_n^*| over the sweep: "
          f"{min(r['min_one_minus_prod'] for r in rows):.3e}")

    cap = rows[0]["arms"][names[0]].get(
        "J", rows[0]["arms"][names[0]]["I"])["cap"]
    print(f"\n  numerical rank, median over frequency, at four tolerances "
          f"(Hankel cap {cap})")
    print("    object           1e-2 1e-3 1e-4 1e-6 | saturated at 1e-6 | "
          "worst recon")
    for name in names:
        e = [r["arms"][name].get("J", r["arms"][name].get("I")) for r in rows]
        med = [int(np.median([x["ladder"][t] for x in e]))
               for t in (1e-2, 1e-3, 1e-4, 1e-6)]
        sat = sum(1 for x in e if x["saturated"])
        lbl = name if "J" in rows[0]["arms"][name] else f"{name} (I)"
        print(f"    {lbl:16s} " + " ".join(f"{m:4d}" for m in med)
              + f" | {sat:2d}/{len(e)} frequencies | "
                f"{max(x['recon'] for x in e):.1e}")
    print("    A rank at the cap is a lower bound, not a measurement: the "
          "Hankel\n    matrix cannot express more than its own size.")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bed", type=Path, required=True)
    ap.add_argument("--iw", type=int, default=None,
                    help="frequency sample; default is the largest |G^<|")
    ap.add_argument("--n-freqs", type=int, default=9,
                    help="sweep this many frequencies spread over the "
                         "positive axis where G^< carries weight; 1 uses --iw")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--m-edge", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    bed = FrozenBed.load(a.bed)
    weight = np.abs(bed.g_lesser).max(axis=(1, 2))
    weight[~bed.pos_mask] = 0.0
    iw = a.iw if a.iw is not None else int(np.argmax(weight))

    res = run(bed, iw=iw, eps=a.eps, m_edge=a.m_edge, rank=a.rank)
    report(bed, res)

    if a.n_freqs > 1:
        live = np.where(weight > 1e-6 * weight.max())[0]
        iws = np.unique(np.linspace(live[0], live[-1], a.n_freqs).astype(int))
        sweep_report(bed, sweep(bed, iws, eps=a.eps, m_edge=a.m_edge,
                                rank=a.rank))

    out = a.out or (OUT.parent / "spatial_tail" / f"{bed.name}_rank.npz")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, name=bed.name, w=res["w"], iw=res["iw"],
        linearity_residual=res["linearity_residual"],
        lam=res["lam"], min_one_minus_prod=res["min_one_minus_prod"],
        psd_clipped=res["psd_clipped"],
        ranks=np.array([[res["arms"][k][d]["r_eps"] if d in res["arms"][k]
                         else -1 for d in ("J", "I", "diag")]
                        for k in res["arms"]]),
        objects=np.array(list(res["arms"]), dtype=object).astype(str),
        meta=np.array(repr(bed.meta)))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
