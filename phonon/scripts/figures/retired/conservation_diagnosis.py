"""Diagnose a lead-balance failure by decomposing the conservation triad.

Data:
  Inputs: two run.npz (bare, dressed) with current_spectrum,
  python phonon/scripts/figures/conservation_diagnosis.py         --bare  out/cnt33_L4_conservation/bare/run.npz         --dressed out/cnt33_L4_conservation/dressed/run.npz         --out   out/cnt33_L4_conservation/conservation.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load(path):
    d = np.load(path)
    # POST-HOC PAIRING TRAP: bubble_balance_spectrum / slab_absorption are
    # produced in engine/run.py AFTER scba.run() returns, when sigma holds
    # the MIXED iterate -- not the pair (Sigma = bubble[G], G) the identity
    # is about. The post-hoc number is then the SCBA residual in energy
    # units, not a conservation defect: measured on a CONVERGED CNT run,
    # pre-mixing 5e-16 vs post-hoc 2.8e-8 (phonon/docs/
    # mos2_conservation_audit.md). Only iter_bubble_balance (QX_BBCHECK=1,
    # scba.py:884, between the SSE evaluation and the mixing step) tests
    # the bubble itself.
    if d.get("iter_bubble_balance") is None:
        print(f"WARNING [{Path(path).name}]: no iter_bubble_balance in this "
              "run -- identity (C) below is computed from the POST-HOC "
              "(mixed-sigma) arrays and mostly measures non-self-"
              "consistency. Rerun with QX_BBCHECK=1 before attributing a "
              "break to the bubble/vertex.")
    w = np.asarray(d["energies"], float)
    cw = np.asarray(d.get("frequency_cell_widths", np.gradient(w)), float)
    wt = np.abs(w) * cw                    # hbar-omega * quadrature weight
    cs = np.asarray(d["current_spectrum"], float)      # (ne, *nk, n_iface)
    cs = cs.reshape(cs.shape[0], -1, cs.shape[-1]).sum(axis=1)  # sum q
    # Integrated bond currents per interface I_{i,i+1}.
    I_bond = np.sum(wt[:, None] * cs, axis=0)
    # Bubble balance spectrum: [P_in(w), P_out(w)], already |omega|-weighted.
    bb = d.get("bubble_balance_spectrum")
    if bb is not None:
        P_in, P_out = np.asarray(bb, float)
        Js_spec = P_out - P_in
    else:
        P_in = P_out = Js_spec = np.zeros_like(w)
    Js = float(np.sum(Js_spec))
    # Slab absorption (row-binned), sum_i P_abs(i).
    sa = d.get("slab_absorption")
    P_abs = np.asarray(sa, float)[0].real if sa is not None else None
    return dict(
        w=w, wt=wt, cs=cs, I_bond=I_bond,
        dJ=float(I_bond[-1] - I_bond[0]),
        P_in=P_in, P_out=P_out, Js_spec=Js_spec, Js=Js,
        P_abs=P_abs, sumPabs=(float(np.nansum(P_abs)) if P_abs is not None else np.nan),
        iter_heat=d.get("iter_heat"), iter_bb=d.get("iter_bubble_balance"),
        converged=bool(d.get("converged", False)),
        n_iter=int(d.get("n_iter", -1)),
    )


def _report(tag, r):
    print(f"\n=== {tag}  (converged={r['converged']}, n_iter={r['n_iter']}) ===")
    Ib = r["I_bond"]
    print(f"  bond currents I_i:      {np.array2string(Ib, precision=3)}")
    print(f"  lead imbalance  dJ = I_N - I_0        = {r['dJ']:+.4e}")
    print(f"  scattering sum  sum_i P_abs(i)        = {r['sumPabs']:+.4e}")
    print(f"  bubble residual J_s = sum(P_out-P_in) = {r['Js']:+.4e}")
    scale = max(abs(Ib[0]), abs(Ib[-1]), 1e-300)
    print(f"  --- identity checks (relative to |I_lead| = {scale:.3e}) ---")
    print(f"  (A) continuity  |dJ - sum P_abs|/|I| = {abs(r['dJ']-r['sumPabs'])/scale:.3e}"
          "   [large => contact/Sigma^R (D) broken]")
    print(f"  (B) slab<->bub  |sum P_abs - J_s|/|I| = {abs(r['sumPabs']-r['Js'])/scale:.3e}"
          "   [construction identity; ~0 expected]")
    print(f"  (C) bubble cons |J_s|/|I|            = {abs(r['Js'])/scale:.3e}"
          "   [large => bubble/vertex/g_band]")
    print(f"  observed lead balance |dJ|/|I|       = {abs(r['dJ'])/scale:.3e}")


def _plot(bare, dressed, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for r, c, lab in ((bare, "tab:blue", "bare"),
                      (dressed, "tab:red", "dressed")):
        w = r["w"]
        # (0,0) per-omega bond-current spectra at the two leads
        ax[0, 0].plot(w, r["wt"] * 0 + r["cs"][:, 0], c=c, ls="-",
                      label=f"{lab} J_L(w)")
        ax[0, 0].plot(w, r["cs"][:, -1], c=c, ls="--", label=f"{lab} J_R(w)")
        # (0,1) per-omega bubble balance P_in, P_out and J_s(w)
        ax[0, 1].plot(w, r["Js_spec"], c=c, label=f"{lab} J_s(w)=P_out-P_in")
        # (1,0) running slab ledger: bond current vs telescoped
        if r["P_abs"] is not None:
            nb = r["P_abs"].size
            xi = np.arange(nb)
            ax[1, 0].plot(xi, r["I_bond"][:nb], c=c, marker="o", ls="-",
                          label=f"{lab} bond I_i")
            tele = r["I_bond"][0] + np.cumsum(r["P_abs"])
            ax[1, 0].plot(xi, tele, c=c, marker="x", ls=":",
                          label=f"{lab} I_0+sum P_abs")
        # (1,1) per-iteration triad
        ih, ibb = r["iter_heat"], r["iter_bb"]
        if ih is not None and ibb is not None:
            ih = np.asarray(ih); ibb = np.asarray(ibb)
            n = min(len(ih), len(ibb))
            dJ_it = np.abs(ih[:n, -1] - ih[:n, 0])
            ax[1, 1].semilogy(range(n), dJ_it, c=c, ls="-",
                              label=f"{lab} |dJ| (lead imbalance)")
            ax[1, 1].semilogy(range(n), np.abs(ibb[:n, 0] - ibb[:n, 1]),
                              c=c, ls="--", label=f"{lab} |J_s| bubble")
    ax[0, 0].set(title="Per-omega lead bond currents", xlabel="omega [THz]",
                 ylabel="MW current density")
    ax[0, 1].set(title="Per-omega bubble balance J_s(w)", xlabel="omega [THz]")
    ax[1, 0].set(title="Slab ledger: bond current vs I_0+cumsum P_abs",
                 xlabel="interface i", ylabel="integrated current")
    ax[1, 1].set(title="Per-iteration |lead imbalance| and |J_s|",
                 xlabel="SCBA iteration")
    for a in ax.ravel():
        a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bare", type=Path, required=True)
    p.add_argument("--dressed", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("conservation.png"))
    a = p.parse_args()
    bare, dressed = _load(a.bare), _load(a.dressed)
    _report("BARE (obc_scattering=False)", bare)
    _report("DRESSED (obc_scattering=True)", dressed)
    print("\n=== DIAGNOSIS ===")
    for tag, r in (("bare", bare), ("dressed", dressed)):
        sc = max(abs(r["I_bond"][0]), abs(r["I_bond"][-1]), 1e-300)
        contact = abs(r["dJ"] - r["sumPabs"]) / sc
        bubble = abs(r["Js"]) / sc
        verdict = ("CONTACT/Sigma^R (identity D)" if contact > 5 * max(bubble, 1e-12)
                   and contact > 1e-2 else
                   "BUBBLE/g_band (identity J_s)" if bubble > 1e-2 else
                   "conserving (both identities hold)")
        print(f"  {tag}: continuity-break {contact:.2e}, bubble-break "
              f"{bubble:.2e}  ->  {verdict}")
    _plot(bare, dressed, a.out)


if __name__ == "__main__":
    main()
