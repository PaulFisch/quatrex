"""Vertex-factorisation audit figures (fig:res_factor_audit,
fig:res_decomp_cons_postfix).

  factor_audit          (a) INDSCAL fit error vs rank for the MoS2
                        film and the Si film vertices, with the MoS2
                        norm-weighted aggregate reconstruction points
                        (aggregate ~ fit error = the convention chain
                        is correct); (b) MoS2 per-offset-class
                        reconstruction error at R=64/128: the
                        transport-diagonal class follows the fit, the
                        six cross-slab (vdW) classes stay at
                        O(1) -- fit-noise, the weak-block failure.
  decomp_cons_postfix   post-min-image conservation audit of the
                        factored coupled-q SSE on the Si film (nk9,
                        3-iteration bubble-balance identity, eta=0):
                        balance residual and lead-to-lead heat vs
                        rank against the dense reference.

Data: phonon/scripts/data/factor_audit.npz, distilled by
_extract_factor_audit.py from cluster/mos2decomp{2,3}/run.log,
cluster/sifilmdecomp/run.log, cluster/sifilm_nk9r/run_*.npz.

Run:  python phonon/scripts/figures/factor_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.studies import style

DATA = ROOT / "phonon/scripts/data/factor_audit.npz"
FIGDIR = ROOT / "document/fig/transport_sweeps"


def fig_audit(d) -> None:
    fig, (ax_a, ax_b) = style.figure(ncols=2, width=4.4, height=3.3)
    colors = style.RC["axes.prop_cycle"].by_key()["color"]

    ax_a.loglog(d["mos2_ranks"], d["mos2_fit"], "o-", color=colors[0],
                label="MoS$_2$ film, fit")
    for r, ci in ((64, 0), (128, 0)):
        fe, ae = d[f"mos2_r{r}_agg"]
        ax_a.plot([r], [ae], "s", color=colors[3], markersize=6, zorder=5)
    ax_a.plot([], [], "s", color=colors[3], label="MoS$_2$ aggregate recon.")
    ax_a.loglog(d["si_ranks"], d["si_fit"], "o-", color=colors[2],
                label="Si film, fit")
    ax_a.loglog(d["si_ranks"], d["si_maxblock"], "^--", color=colors[2],
                alpha=0.6, label="Si film, max block recon.")
    ax_a.set_xlabel("rank $R$")
    ax_a.set_ylabel("relative error")
    ax_a.set_xticks(d["mos2_ranks"], [str(int(r)) for r in d["mos2_ranks"]])
    ax_a.minorticks_off()
    ax_a.legend(fontsize=7.5)

    offs = d["mos2_r64_offsets"]
    labels = [f"({int(a)},{int(b)})" for a, b in offs]
    x = np.arange(len(labels))
    w = 0.38
    ax_b.bar(x - w / 2, d["mos2_r64_offerr"], w, color=colors[0],
             label="$R=64$")
    ax_b.bar(x + w / 2, d["mos2_r128_offerr"], w, color=colors[1],
             label="$R=128$")
    ax_b.set_xticks(x, labels, rotation=45, fontsize=7.5)
    ax_b.set_xlabel(r"offset class $(\Delta J, \Delta K)$")
    ax_b.set_ylabel("class reconstruction error")
    ax_b.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax_b.legend()

    style.save(fig, "factor_audit", directory=FIGDIR)

    for r in (64, 128):
        fe, ae = d[f"mos2_r{r}_agg"]
        err = d[f"mos2_r{r}_offerr"]
        offs_r = d[f"mos2_r{r}_offsets"]
        diag = err[[tuple(o) == (0, 0) for o in offs_r.tolist()]][0]
        cross = err[[tuple(o) != (0, 0) for o in offs_r.tolist()]]
        print(f"MoS2 R={r}: fit {fe:.4f}, aggregate {ae:.4f} "
              f"(ratio {ae / fe:.2f}); diagonal class {diag:.3f}, "
              f"cross-slab classes {cross.min():.2f}-{cross.max():.2f}")
    print(f"Si film: fit {d['si_fit'][-1]:.2e}, max-block "
          f"{d['si_maxblock'][-1]:.2e} at R={int(d['si_ranks'][-1])}")


def fig_cons(d) -> None:
    fig, ax = style.figure(width=4.8, height=3.2)
    colors = style.RC["axes.prop_cycle"].by_key()["color"]

    tags = [str(t) for t in d["cons_tags"]]
    x = np.arange(len(tags))
    ax.semilogy(x, d["cons_balance"], "o-", color=colors[0])
    ax.set_xticks(x, tags)
    ax.set_ylabel("bubble-balance residual", color=colors[0])
    ax.set_xlabel("vertex")
    ax2 = ax.twinx()
    dh = np.abs(d["cons_heat"] / d["cons_heat"][0] - 1.0)
    ax2.semilogy(x, np.maximum(dh, 1e-8), "s--", color=colors[1])
    ax2.set_ylabel("lead heat, rel. deviation from dense",
                   color=colors[1])
    ax2.grid(False)

    style.save(fig, "decomp_cons_postfix", directory=FIGDIR)

    for t, b, h in zip(tags, d["cons_balance"], d["cons_heat"]):
        print(f"{t:6s} balance {b:.2e}  heat[L->R] {h:.5f} "
              f"({h / d['cons_heat'][0] - 1:+.2e} vs dense)")


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA)
    fig_audit(d)
    fig_cons(d)


if __name__ == "__main__":
    main()
