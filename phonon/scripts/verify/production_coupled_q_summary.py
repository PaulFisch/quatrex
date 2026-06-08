"""Consolidate + plot the production transverse-q (coupled-q) anharmonic
transport runs (2026-06-08 session). Reads the run snapshots, copies them
into the repo, writes summary.json, and renders the figures used in the
LaTeX appendix. Pure reporting of what was run -- no new physics.

Sources (scratch run dirs from this session):
  /tmp/sifilm_zmp/   eta=0.4 + zero-mode-projection film, q-mesh sweep
  /tmp/sifilm_run/   eta=0.4, NO zero-mode film (k3 only)
  /tmp/cnt_ladder/   CNT L=2/4/6 length ladder (eta=0.45)
  /tmp/dense_cmp_*.json  dense si_film at matched/small eta (cross-check)
"""
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/usr/scratch/mont-fort11/pfischill/quatrex/phonon/scripts/out/production_coupled_q")
FIG = Path("/usr/scratch/mont-fort11/pfischill/quatrex/document/fig/transport_sweeps")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SNAP = OUT / "snapshots"
SNAP.mkdir(exist_ok=True)


def lead_heat(npz):
    """q-summed heat current per device interface; lead value = index 0."""
    d = np.load(npz, allow_pickle=True)
    fh = d.get("final_heat")
    if fh is None:
        return None
    fh = np.asarray(fh)
    js = np.nansum(fh.reshape(-1, fh.shape[-1]), axis=0)
    return dict(per_interface=js.tolist(), lead0=float(js[0]), lead1=float(js[-1]),
                finite=bool(np.isfinite(fh).all()))


def rung(dirpath, tag, nk=None):
    bp, ap = Path(dirpath) / f"{tag}_ball.npz", Path(dirpath) / f"{tag}_anh.npz"
    if not (bp.exists() and ap.exists()):
        return None
    for p in (bp, ap):
        shutil.copy(p, SNAP / p.name)
    b, a = lead_heat(bp), lead_heat(ap)
    r = dict(tag=tag, G_ball=b["lead0"], G_anh=a["lead0"],
             ratio=a["lead0"] / b["lead0"],
             lead_conservation=abs(a["lead0"] - a["lead1"]) / abs(a["lead0"]),
             anh_finite=a["finite"])
    if nk is not None:
        r["nk"] = nk
        r["G_ball_per_q"] = b["lead0"] / nk ** 2
        r["G_anh_per_q"] = a["lead0"] / nk ** 2
    return r


summary = {"validation": {
    "coupled_q_sse_vs_oracle_relerr": 6.4e-16,
    "production_Hq_vs_dense_getbtdblocks_relerr": 1.08e-10,
    "dynmat_idft_roundtrip_relerr": 5.0e-16,
    "note": "from test_compute_coupled_q_matches_reference, verify_sifilm_hq.py, "
            "build_sifilm_inputs.py self-check (this session)",
}}

# --- Si film, zero-mode projection, q-mesh sweep (eta=0.4) ---
film_zmp = [rung("/tmp/sifilm_zmp", t, nk) for t, nk in
            [("k3_n3", 3), ("k5_n3", 5), ("k7_n3", 7), ("k5_n5", 5)]]
summary["si_film_zmp_eta0p4"] = [r for r in film_zmp if r]

# --- Si film, NO zero-mode projection (k3) ---
film_nozmp = [rung("/tmp/sifilm_run", t, nk) for t, nk in [("k3_n3", 3), ("k3_n5", 3)]]
summary["si_film_nozmp_eta0p4"] = [r for r in film_nozmp if r]

# --- CNT length ladder (eta=0.45) ---
cnt = []
for L, tag in [(2, "L2_e0.45_n121"), (4, "L4_e0.45_n121"), (6, "L6_e0.45_n121")]:
    r = rung("/tmp/cnt_ladder", tag)
    if r:
        ad = np.load(Path("/tmp/cnt_ladder") / f"{tag}_anh.npz", allow_pickle=True)
        bh = ad.get("best_heat")
        if bh is not None:
            bh = np.asarray(bh)
            r["G_anh_best"] = float(bh[0])
            r["best_conservation"] = float(ad.get("best_cons", np.nan))
            r["ratio_best"] = float(bh[0]) / r["G_ball"]
        r["L"] = L
        cnt.append(r)
summary["cnt_ladder_eta0p45"] = cnt

# --- dense cross-check (matched + small eta) ---
dense = {}
for tag, f in [("eta_w~0.4", "/tmp/dense_cmp_bigeta.json"),
               ("eta_w~0.05", "/tmp/dense_cmp_smalleta.json")]:
    try:
        d = json.load(open(f)); row = d["rows"][0]
        dense[tag] = dict(G_ball=row["G_ball"], G_anh=row["G_anh"],
                          ratio=row["G_anh"] / row["G_ball"],
                          conservation=row["conservation"])
        shutil.copy(f, SNAP / Path(f).name)
    except Exception as e:
        dense[tag] = f"unavailable: {e}"
summary["dense_si_film_nk3_nslabs3"] = dense

json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))

# ============================ FIGURES ============================
# Fig 1: Si film q-convergence (per-q conductance + ratio vs nk, n_slabs=3)
qrows = [r for r in summary["si_film_zmp_eta0p4"] if r.get("nk") and r["tag"].endswith("n3")]
qrows.sort(key=lambda r: r["nk"])
nks = [r["nk"] for r in qrows]
fig, ax1 = plt.subplots(figsize=(5.2, 3.8))
ax1.plot(nks, [r["G_ball_per_q"] for r in qrows], "o-", color="C0", label=r"$G_{\rm ball}/N_q$")
ax1.plot(nks, [r["G_anh_per_q"] for r in qrows], "s-", color="C1", label=r"$G_{\rm anh}/N_q$")
ax1.set_xlabel(r"transverse mesh $n_k$ ($n_k\times n_k$)")
ax1.set_ylabel(r"per-$q$ lead heat current (arb. units)")
ax1.set_xticks(nks)
ax2 = ax1.twinx()
ax2.plot(nks, [r["ratio"] for r in qrows], "^--", color="C3", label=r"$G_{\rm anh}/G_{\rm ball}$")
ax2.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$", color="C3")
ax2.tick_params(axis="y", labelcolor="C3")
ax2.set_ylim(0.5, 1.0)
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
ax1.set_title("Production Si film, coupled-$q$ (η=0.4, zero-mode proj.)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "prod_qfilm_qconv.pdf"); plt.close(fig)

# Fig 2: conservation -- zmp vs no-zmp, and vs nk (shows it is NOT q-limited)
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(nks, [100 * r["lead_conservation"] for r in qrows], "^-", color="C2",
        label="η=0.4 + zero-mode proj. (n_slabs=3)")
nz = {r["tag"]: r for r in summary["si_film_nozmp_eta0p4"]}
if "k3_n3" in nz:
    ax.plot([3], [100 * nz["k3_n3"]["lead_conservation"]], "vr", ms=9,
            label="η=0.4, NO zero-mode (nk=3)")
ax.axhline(100 * summary["dense_si_film_nk3_nslabs3"].get("eta_w~0.4", {}).get("conservation", np.nan)
           if isinstance(summary["dense_si_film_nk3_nslabs3"].get("eta_w~0.4"), dict) else np.nan,
           ls=":", color="C4", label="dense, η≈0.4, nk=3")
ax.set_xlabel(r"transverse mesh $n_k$"); ax.set_ylabel("lead-to-lead heat non-conservation (%)")
ax.set_xticks(nks); ax.set_ylim(0, 25)
ax.legend(fontsize=8); ax.set_title("Heat-flow conservation (open issue)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "prod_qfilm_conservation.pdf"); plt.close(fig)

# Fig 3: CNT length ladder (ratio vs L) -- whatever rungs are done
if cnt:
    Ls = [r["L"] for r in cnt]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(Ls, [r.get("ratio_best", r["ratio"]) for r in cnt], "o-", color="C0")
    for r in cnt:
        ax.annotate(f"{r.get('best_conservation', float('nan'))*100:.1f}% cons",
                    (r["L"], r.get("ratio_best", r["ratio"])), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("device length $L$ (transport cells)")
    ax.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$ (best iterate)")
    ax.set_xticks(Ls); ax.set_title("Production CNT(3,3) length ladder (η=0.45)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "prod_cnt_ladder.pdf"); plt.close(fig)

print("\nFIGURES ->", FIG)
print("SNAPSHOTS + summary.json ->", OUT)
