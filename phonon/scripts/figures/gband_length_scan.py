"""Compare g_band = 1, 2, 3 on the longer CNT (3,3) chains (L8, L10).

Data:
  python phonon/scripts/figures/gband_length_scan.py         --root phonon/studies/out/cnt33_gband_length         --lengths 8 10 --gbands 1 2 3         --out phonon/studies/out/cnt33_gband_length/gband_scan.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = np.load(path)
    w = np.asarray(d["energies"], float)
    cw = np.asarray(d.get("frequency_cell_widths", np.gradient(w)), float)
    wt = np.abs(w) * cw
    out = dict(
        w=w, wt=wt,
        converged=bool(d.get("converged", False)),
        diverged=bool(d.get("diverged", False)),
        n_iter=int(d.get("n_iter", -1)),
        lead_current=float(d["lead_current"]) if "lead_current" in d else np.nan,
    )
    cs = d.get("current_spectrum")
    if cs is not None:
        cs = np.asarray(cs, float)
        cs = cs.reshape(cs.shape[0], -1, cs.shape[-1]).sum(axis=1)  # sum q
        I_bond = np.sum(wt[:, None] * cs, axis=0)
        out["cs"] = cs
        out["I_bond"] = I_bond
        out["dJ"] = float(I_bond[-1] - I_bond[0])
        out["scale"] = max(abs(I_bond[0]), abs(I_bond[-1]), 1e-300)
    else:
        out["cs"] = out["I_bond"] = None
        out["dJ"] = np.nan
        out["scale"] = np.nan
    bb = d.get("bubble_balance_spectrum")
    if bb is not None:
        # NOTE: the npz spectrum is the RANK-0-LOCAL frequency slice; the
        # authoritative global balance is iter_bubble_balance (all-reduced
        # P_in, P_out per iteration). Js here is the slice sum -- indicative,
        # not the global number.
        P_in, P_out = np.asarray(bb, float)
        out["Js_spec"] = P_out - P_in
        out["Js"] = float(np.sum(P_out - P_in))
    else:
        out["Js_spec"] = None
        out["Js"] = np.nan
    ibb = d.get("iter_bubble_balance")
    if ibb is not None:
        # (P_in, P_out, resid) per iteration -- resid is RELATIVE to the
        # bubble power, the honest conservation figure of merit.
        out["bb_resid"] = float(np.asarray(ibb)[-1, 2])
    else:
        out["bb_resid"] = np.nan
    return out


def _status(r: dict) -> str:
    return ("converged" if r["converged"]
            else "DIVERGED" if r["diverged"] else "NOT CONV")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path,
                   default=Path("phonon/studies/out/cnt33_gband_length"))
    p.add_argument("--lengths", type=int, nargs="+", default=[8, 10])
    p.add_argument("--gbands", nargs="+", default=["1", "2", "3", "1t"],
                   help="g_band rung tags; a 't' suffix = Bartlett-tapered "
                        "(e.g. 1 2 3 1t)")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    data: dict[tuple[int, str], dict] = {}
    hdr = (f"{'L':>3} {'g_band':>6} {'status':>10} {'n_iter':>6} "
           f"{'lead_current':>13} {'|dJ|/|I|':>11} {'|J_s|/|I|':>11} "
           f"{'bb_resid':>10}")
    print(hdr)
    print("-" * len(hdr))
    for L in a.lengths:
        for g in a.gbands:
            r = _load(a.root / f"L{L}_g{g}" / "run.npz")
            if r is None:
                print(f"{L:>3} {g:>6} {'(no run)':>10}")
                continue
            data[(L, g)] = r
            sc = r["scale"]
            dJrel = abs(r["dJ"]) / sc if np.isfinite(sc) else np.nan
            Jsrel = abs(r["Js"]) / sc if np.isfinite(sc) else np.nan
            print(f"{L:>3} {g:>6} {_status(r):>10} {r['n_iter']:>6} "
                  f"{r['lead_current']:>13.5g} {dJrel:>11.3e} {Jsrel:>11.3e} "
                  f"{r['bb_resid']:>10.2e}")

    if a.out is None or not data:
        return 0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nL = len(a.lengths)
    fig, ax = plt.subplots(nL, 3, figsize=(15, 4 * nL), squeeze=False)
    for row, L in enumerate(a.lengths):
        gs = [g for g in a.gbands if (L, g) in data]
        lc = [data[(L, g)]["lead_current"] for g in gs]
        dJ = [abs(data[(L, g)]["dJ"]) / data[(L, g)]["scale"] for g in gs]
        Js = [abs(data[(L, g)]["Js"]) / data[(L, g)]["scale"] for g in gs]
        conv = [data[(L, g)]["converged"] for g in gs]

        ax[row, 0].plot(gs, lc, "o-", color="tab:blue")
        for g, y, c in zip(gs, lc, conv):
            ax[row, 0].annotate("converged" if c else _status(data[(L, g)]),
                                (g, y), fontsize=7,
                                color="green" if c else "red")
        ax[row, 0].set(title=f"L{L}: lead current vs g_band",
                       xlabel="g_band", ylabel="0.5(|J_L|+|J_R|)")
        ax[row, 0].set_xticks(gs)

        ax[row, 1].semilogy(gs, dJ, "s-", color="tab:red",
                            label="|dJ|/|I| (continuity)")
        ax[row, 1].semilogy(gs, Js, "^--", color="tab:purple",
                            label="|J_s|/|I| (bubble)")
        ax[row, 1].set(title=f"L{L}: conservation breaks vs g_band",
                       xlabel="g_band", ylabel="relative break")
        ax[row, 1].set_xticks(gs)
        ax[row, 1].legend(fontsize=8)

        for g in gs:
            r = data[(L, g)]
            if r["cs"] is not None:
                ax[row, 2].plot(r["w"], r["cs"][:, 0], label=f"g_band={g}")
        ax[row, 2].set(title=f"L{L}: left-lead current spectrum J_L(w)",
                       xlabel="omega [THz]", ylabel="number-current density")
        ax[row, 2].legend(fontsize=8)
        for c in range(3):
            ax[row, c].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
