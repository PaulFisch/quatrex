"""Production coupled-q silicon-film figures
(appendices/production_coupled_q).

Run:  python phonon/scripts/figures/prod_qfilm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

FIGDIR = ROOT / "document/fig/transport_sweeps"

# --- tab:prod_film (production film, coupled-q, eta=0.4 THz, zero-mode
#     projection, n_slabs=3): rows n_k = 3/5/7 ------------------------------
NK = [3, 5, 7]
G_BALL_PER_NQ = [3.29, 3.02, 3.05]      # G_ball/N_q (arbitrary, common units)
RATIO = [0.861, 0.871, 0.867]           # G_anh/G_ball
NONCONS = [0.097, 0.095, 0.096]         # lead non-conservation 9.7/9.5/9.6 %
# tab:prod_film last row: no zero-mode projection, n_k=3, n_slabs=3
NONCONS_NO_ZEROMODE = 0.22              # 22 %
# tab:prod_dense: dense si_film at the matched mesh (n_k=3, n_slabs=3),
# eta_w ~ 0.4 (matched to production)
NONCONS_DENSE_MATCHED = 0.17            # 17 %
# sec:prod_dense / fig:prod_qfilm_conservation caption: the dense solver
# reaches ~1e-5 at small broadening on the converged n_k>=8 mesh
NONCONS_DENSE_SMALL_ETA = 1e-5


def fig_qconv():
    """G_ball/N_q + G_anh/N_q vs transverse mesh (left) and ratio (right)."""
    fig, ax = style.figure(width=4.0, height=3.0)
    g_anh = [r * g for r, g in zip(RATIO, G_BALL_PER_NQ)]  # from table cols
    ax.plot(NK, G_BALL_PER_NQ, "o-", color="C0", label=r"$G_\mathrm{ball}/N_q$")
    ax.plot(NK, g_anh, "s-", color="C1", label=r"$G_\mathrm{anh}/N_q$")
    ax.set_xlabel(r"transverse mesh $n_k$ ($n_k\times n_k$)")
    ax.set_ylabel(r"$G/N_q$ (arb. units)")
    ax.set_xticks(NK)
    ax.set_xlim(2.6, 7.4)
    ax.set_ylim(2.3, 3.55)

    ax2 = ax.twinx()
    ax2.plot(NK, RATIO, "d--", color="C2", label=r"$G_\mathrm{anh}/G_\mathrm{ball}$")
    for nk, r in zip(NK, RATIO):
        ax2.annotate(f"{r:.3f}", (nk, r + 0.004), fontsize=7, ha="center",
                     color="C2")
    ax2.set_ylabel(r"$G_\mathrm{anh}/G_\mathrm{ball}$", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax2.set_ylim(0.80, 0.90)
    ax2.grid(False)

    # one merged legend, lower left where no data sits
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", framealpha=0.9)
    style.save(fig, "prod_qfilm_qconv", directory=FIGDIR)


def fig_conservation():
    """Lead non-conservation, log-y 1e-6..1 so the dense small-eta gap shows."""
    fig, ax = style.figure(width=4.0, height=3.0)
    ax.plot(NK, NONCONS, "o-", color="C0",
            label=r"production, $\eta=0.4$, zero-mode proj.")
    # the two nk=3 reference points, x-offset slightly for legibility only
    ax.plot([2.92], [NONCONS_NO_ZEROMODE], "s", color="C3",
            label=r"production, no zero-mode proj.")
    ax.plot([3.08], [NONCONS_DENSE_MATCHED], "^", color="C1",
            label=r"dense, $\eta_w\approx0.4$ (matched mesh)")
    ax.axhline(NONCONS_DENSE_SMALL_ETA, ls="--", color="C2", lw=1.2)
    ax.annotate(r"dense, small $\eta$ (converged mesh)",
                (2.85, NONCONS_DENSE_SMALL_ETA * 1.6), fontsize=8, color="C2")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel(r"transverse mesh $n_k$ ($n_k\times n_k$)")
    ax.set_ylabel(r"lead non-conservation $|J_0-J_{-1}|/|J_0|$")
    ax.set_xticks(NK)
    ax.set_xlim(2.6, 7.4)
    ax.legend(loc="center right", framealpha=0.9)
    style.save(fig, "prod_qfilm_conservation", directory=FIGDIR)


if __name__ == "__main__":
    fig_qconv()
    fig_conservation()
