"""Energy-conservation audit of the decomposed three-phonon SSE (decomp_conservation).

The bubble is Phi-derivable, so P_in = P_out is an IDENTITY of the exact vertex --
it holds at ANY G, not just at self-consistency. It holds precisely when the cubic
vertex is totally symmetric (S3). On the transversely-periodic Si film (sifilm, L3,
nk=9, eta=1e-12) the bubble misses it by ~2e-6, which the ratio |P|/J ~ 6.6e3
amplifies into a 1-3% violation of the terminal heat balance. This figure asks
whether the low-rank INDSCAL vertex is to blame. It is not.

  left   : the vertex's S3 defect vs rank, for the two q-fold phase conventions.
           The PRODUCTION fold puts the transverse Bloch phase on RAW supercell
           cell indices (0..4) instead of minimum-image ones (-2..2) --
           phonon/phonon_inputs/separable.py:102 (cell_frac[s] = R_int), consumed
           at phonon/solver/se_q.py:41-42 and fc3_factor_device.py:122. The FC3
           supercell is 5x5x5, so exp(-2i.pi.5q) = 1 only when 5 | nk. At the
           production mesh nk=9 that alias breaks S3 by ~0.65 -- ALREADY IN THE
           EXACT DENSE VERTEX (dashed line), independent of rank. Re-folded with
           minimum-image cells the exact vertex is S3-clean to 3e-16, and what
           remains is INDSCAL's own defect, which falls with rank as the
           lam_r d_r (x) u_r (x) u_r ansatz predicts.
  middle : the measured imbalance. Circles: the bubble residual at ITERATION 0,
           where G is the ballistic G and therefore IDENTICAL for every leg (all
           legs log the same lead balance 2.16e-15) -- so only the vertex differs.
           The exact dense vertex gives the SAME residual as R=64 (1.961e-6),
           which refutes truncation as the leading cause. Squares: the same defect
           at the fixed point, amplified by |P|/J into the reported heat-balance
           violation. Open squares = the run never converged (hit the 450-iteration
           cap), because the gate demands a balance the bubble cannot reach.
  right  : where the energy is lost. The per-frequency imbalance d(w) = P_out(w) -
           P_in(w) (normalised by J, so it sums to the middle panel's squares) sits
           on the optical/zone-boundary band, NOT in the infrared -- even though the
           IR is where the bubble throughput |P_out(w)| (grey) actually lives. The
           eta -> 0 infrared singularity is therefore not the culprit either.

Data: phonon/scripts/data/decomp_conservation.npz -- extracted read-only from the
      cluster campaign cluster/eta0-L3/{ball,dense,r8,r16,r32,r64} (stdout.log +
      run.npz) and from a symmetry probe over the shipped vertices
      (cluster/prod/geom/sifilm_L3_nk9/qfold_vertices.npz, decomposed_vertices_r*.npz)
      and the cached real-space INDSCAL factors (phonon/reaps/si_big_hiphive/
      fc3_factors_indscal_r*.npz). The S2/S3 defects are relative Frobenius norms
          S2: Phi^{q1,q2}_{I,K,K'}[a,b,c] - Phi^{q2,q1}_{I,K',K}[a,c,b]
          S3: Phi^{q1,q2}_{I,K,K'}[a,b,c] - Phi^{q3,q2}_{K,I,K'}[b,a,c], q3=-(q1+q2)
      over all 15 device blocks and 61 sampled (q1,q2) pairs (9 pairs for the
      re-folded dense controls, which cost a full fold each). S2 swaps the two
      CONTRACTED legs -- the one permutation INDSCAL enforces by construction and
      the one the conservation proof cannot use. S3 moves the EXTERNAL leg.
      The probe re-folds the vertex from the raw FC3 and reproduces the shipped
      qfold_vertices.npz to 0.0 (bit-identical), which is what pins "the solver
      really did consume raw cell indices".

Run:  python phonon/scripts/figures/decomposed_sse_conservation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

NPZ = ROOT / "phonon/scripts/data/decomp_conservation.npz"
FIGDIR = ROOT / "document/fig/transport_sweeps"

RANKS = [8, 16, 32, 64, 128]
MEASURED = [8, 16, 32, 64]          # ranks with an SCBA run
C_BALL = "#0173b2"                  # C0 -- ballistic / harmonic
C_ANH = "#d55e00"                   # C3 -- anharmonic
C_MIN = "#029e73"                   # min-image control
C_AMP = "#cc78bc"                   # the amplified (reported) imbalance
IR_MAX = 2.0                        # THz -- "infrared" band
OPT = (8.0, 13.0)                   # THz -- where the imbalance sits


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ, allow_pickle=True)
    runs = json.loads(str(z["runs_json"]))
    s3 = json.loads(str(z["s3_json"]))

    ind = s3["indscal"]
    d_raw = np.array([ind[str(r)]["raw"]["S3"] for r in RANKS])
    d_min = np.array([ind[str(r)]["minimage"]["S3"] for r in RANKS])
    s2_raw = np.array([ind[str(r)]["raw"]["S2"] for r in RANKS])
    eps_R = np.array([ind[str(r)]["rel_err"] for r in RANKS])
    dense_raw = s3["dense_shipped_nk9"]["S3"]
    dense_min = s3["dense_minimage_nk9"]["S3"]
    dense_s2 = s3["dense_shipped_nk9"]["S2"]

    # ---- claim trail -------------------------------------------------------
    print("=" * 78)
    print("CONSERVATION AUDIT -- sifilm L3, nk=9, eta=1e-12 (cluster/eta0-L3)")
    print("=" * 78)
    rr = [int(round(v)) for v in s3["cell_frac_raw_range"]]
    rm = [int(round(v)) for v in s3["cell_frac_min_range"]]
    print(f"\nq-fold: FC3 supercell 5x5x5, n_super={s3['n_super']}; production "
          f"cell_frac (transverse) in [{rr[0]},{rr[1]}], min-image in "
          f"[{rm[0]},{rm[1]}]; "
          f"{s3['n_cells_changed']}/{s3['n_super']} cells differ.")
    print(f"device blocks: {s3['n_blocks']}, S3-closed under (I,K,K')->(K,I,K'): "
          f"{s3['blocks_s3_closed']}  (so the nn_only truncation cannot break S3)")
    print(f"refold with RAW cells vs shipped qfold_vertices.npz: "
          f"{s3['raw_refold_vs_shipped']:.2e}  <- the solver DID consume raw cells")
    print(f"||V_raw - V_minimage|| / ||V_minimage|| = "
          f"{s3['vertex_raw_vs_minimage_rel']:.3f}  <- the production vertex is "
          f"{100 * s3['vertex_raw_vs_minimage_rel']:.0f}% wrong, conservation aside")

    print("\n-- VERTEX S3 DEFECT (external leg; the permutation Phi-derivability needs)")
    print(f"  {'vertex':22s} {'S3 (raw/production)':>20s} {'S3 (min-image)':>16s}"
          f" {'S2':>10s}")
    print(f"  {'dense (exact FC3)':22s} {dense_raw:20.4e} {dense_min:16.4e}"
          f" {dense_s2:10.2e}")
    for i, r in enumerate(RANKS):
        print(f"  {'INDSCAL R=%d' % r:22s} {d_raw[i]:20.4e} {d_min[i]:16.4e}"
              f" {s2_raw[i]:10.2e}   (fit rel_err {eps_R[i]:.4f})")
    print(f"  commensurability: dense raw nk=5 S3={s3['dense_raw_nk5']['S3']:.3e} "
          f"(5|5, clean)   nk=7 S3={s3['dense_raw_nk7']['S3']:.3e} (broken)")
    print("  => S2 ~ 1e-16 at EVERY rank: INDSCAL enforces the one permutation")
    print("     the proof cannot use. The S3 defect is alias-dominated and FLAT")
    print("     in rank; min-imaging the cells exposes INDSCAL's own (sub-leading)")
    print("     defect, which falls from 1.4e-1 (R=8) to 4.5e-4 (R=128).")

    # ---- measured ----------------------------------------------------------
    # The fixed-point imbalance is taken from the FULL-PRECISION spectra in
    # run.npz (P = sum_w spec), not from the 7-digit P_in/P_out in the log:
    # |P_out - P_in| ~ 8 against |P| ~ 1.4e6, so differencing the logged
    # mantissas loses ~5 digits. It also makes the right panel's d(w)/J
    # integrate exactly to the squares in the middle panel.
    print("\n-- MEASURED BUBBLE IMBALANCE")
    print(f"  {'leg':8s} {'iter0 resid':>12s} {'final resid':>12s} {'|dP|/J':>10s}"
          f" {'J':>9s} {'iters':>6s} {'conv':>6s} {'amp=|P|_1/J':>12s}")
    iter0, final_amp, amp = {}, {}, {}
    for leg in ["ball", "dense", "r8", "r16", "r32", "r64"]:
        r = runs[leg]
        iter0[leg] = r["iter0_resid"]
        J = r["J"]
        if f"{leg}/spec_in" not in z.files or J != J:
            final_amp[leg] = amp[leg] = float("nan")
            print(f"  {leg:8s} {r['iter0_resid']:12.4e} {r['final_resid']:12.4e}"
                  f" {'--':>10s} {'--':>9s} {r['n_iter_log']:6d}"
                  f" {str(r['converged']):>6s} {'--':>12s}")
            continue
        p_in = float(z[f"{leg}/spec_in"].sum())
        p_out = float(z[f"{leg}/spec_out"].sum())
        # amplification of a relative bubble error into a heat-balance error
        amp[leg] = (abs(p_in) + abs(p_out)) / J
        final_amp[leg] = abs(p_out - p_in) / J
        print(f"  {leg:8s} {r['iter0_resid']:12.4e} {r['final_resid']:12.4e}"
              f" {final_amp[leg]:10.3e} {J:9.2f} {r['n_iter_log']:6d}"
              f" {str(r['converged']):>6s} {amp[leg]:12.1f}")
    # cross-check: |dP|/J must equal resid * (|P_in|+|P_out|) / J
    chk = runs["r8"]["final_resid"] * amp["r8"]
    print(f"  cross-check (r8): resid x amp = {chk:.4e} vs |dP|/J = "
          f"{final_amp['r8']:.4e}; logged lead balance = "
          f"{runs['r8']['final_lead_balance']:.4e}  -- the bubble imbalance IS "
          f"the lead imbalance")
    print(f"  iter-0 lead balance (ballistic G, identical for every leg): "
          f"{runs['r8']['iter0_lead_balance']:.4e}")
    print("  => at iteration 0 the G is the BALLISTIC G for every leg, so only the")
    print("     vertex differs. The EXACT DENSE vertex gives resid="
          f"{iter0['dense']:.3e},")
    print(f"     identical to R=64 ({iter0['r64']:.3e}). Truncation is NOT the")
    print("     leading cause -- the defect is already in the exact vertex.")
    print(f"  amplification (|P_in|+|P_out|)/J = {amp['r8']:.3e}: a 2.7e-6 relative")
    print(f"     bubble error becomes a {100 * final_amp['r8']:.1f}% heat imbalance.")

    # ---- per-frequency -----------------------------------------------------
    print("\n-- WHERE THE ENERGY IS LOST (per-frequency imbalance)")
    spec = {}
    for leg in ["r8", "r16", "r32", "r64"]:
        e = z[f"{leg}/energies"]
        d = z[f"{leg}/spec_out"] - z[f"{leg}/spec_in"]
        spec[leg] = (e, d)
        tot = np.abs(d).sum()
        ir = np.abs(d[e <= IR_MAX]).sum() / tot
        opt = np.abs(d[(e >= OPT[0]) & (e <= OPT[1])]).sum() / tot
        pw = np.abs(z[f"{leg}/spec_out"])
        ir_p = pw[e <= IR_MAX].sum() / pw.sum()
        print(f"  {leg:5s} |d| in IR (<{IR_MAX:.0f} THz): {100 * ir:5.1f}% | "
              f"in {OPT[0]:.0f}-{OPT[1]:.0f} THz: {100 * opt:5.1f}%  "
              f"(throughput |P_out| in IR: {100 * ir_p:5.1f}%)")
    print("  => the IR carries the bubble's POWER but almost none of its IMBALANCE:")
    print("     the eta->0 infrared singularity is not the source either.")
    print(f"\nharmonic control: lead balance "
          f"{runs['ball']['final_lead_balance']:.1e}, J={runs['ball']['J']:.2f} "
          f"-- leads/OBC/Meir-Wingreen are exact; they never touch cell_frac.")
    print()

    # ---- figure ------------------------------------------------------------
    fig, axes = style.figure(ncols=3, width=4.0, height=3.2)
    ax0, ax1, ax2 = axes

    # (a) vertex S3 defect vs rank
    ax0.axhline(dense_raw, color=C_ANH, ls="--", lw=1.1)
    ax0.text(8.4, dense_raw * 2.4, "dense (exact FC3), raw cells",
             color=C_ANH, fontsize=7.2)
    ax0.axhline(dense_min, color=C_MIN, ls="--", lw=1.1)
    ax0.text(8.4, dense_min * 2.4, "dense, min-image cells", color=C_MIN,
             fontsize=7.2)
    ax0.plot(RANKS, d_raw, "o-", color=C_ANH,
             label="INDSCAL, production fold (raw cells)")
    ax0.plot(RANKS, d_min, "s-", color=C_MIN,
             label="INDSCAL, min-image fold")
    ax0.plot(RANKS, s2_raw, "^:", color="#949494",
             label=r"$S_2$ (contracted legs) -- enforced")
    ax0.set_xscale("log", base=2)
    ax0.set_yscale("log")
    ax0.set_xticks(RANKS)
    ax0.set_xticklabels([str(r) for r in RANKS])
    ax0.xaxis.set_minor_formatter(NullFormatter())
    ax0.yaxis.set_minor_formatter(NullFormatter())
    ax0.set_xlabel("CP/INDSCAL rank $R$")
    ax0.set_ylabel(r"vertex permutation defect  $\|\Phi - P\Phi\| / \|\Phi\|$")
    ax0.set_ylim(2e-17, 6.0)
    ax0.legend(loc="center left", fontsize=7, framealpha=0.95)

    # (b) measured imbalance vs rank
    ax1.axhline(iter0["dense"], color="#404040", ls="--", lw=1.1)
    ax1.text(8.4, iter0["dense"] * 0.30,
             f"dense (exact FC3) vertex: {iter0['dense']:.2e}",
             color="#404040", fontsize=7.2)
    ax1.plot(MEASURED, [iter0[f"r{r}"] for r in MEASURED], "o-", color=C_ANH,
             label=r"bubble residual, iter 0 (ballistic $G$)")
    conv = [runs[f"r{r}"]["converged"] for r in MEASURED]
    y = [final_amp[f"r{r}"] for r in MEASURED]
    ax1.plot(MEASURED, y, "-", color=C_AMP, zorder=2)
    for r, yy, c in zip(MEASURED, y, conv):
        ax1.plot([r], [yy], "s", color=C_AMP, mfc=C_AMP if c else "white",
                 mec=C_AMP, ms=6, zorder=3)
    ax1.plot([], [], "s", color=C_AMP, label=r"$|P_{out}-P_{in}| / J$ (fixed pt.)")
    ax1.plot([], [], "s", color=C_AMP, mfc="white", label="   open: hit iter. cap")
    ax1.axhline(abs(runs["ball"]["final_lead_balance"]), color=C_BALL, ls=":",
                lw=1.2, label="harmonic leg (exact)")
    ax1.annotate("", xy=(24, y[1]), xytext=(24, iter0["r16"]),
                 arrowprops=dict(arrowstyle="<->", color="#404040", lw=0.9))
    ax1.text(26, 3e-4, r"$\times\,(|P_{in}|{+}|P_{out}|)/J$"
             "\n" fr"$\approx {amp['r8'] / 1e3:.1f}{{\times}}10^{{3}}$",
             fontsize=7.2, color="#404040")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(MEASURED)
    ax1.set_xticklabels([str(r) for r in MEASURED])
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.set_xlabel("CP/INDSCAL rank $R$")
    ax1.set_ylabel("bubble energy imbalance")
    ax1.set_ylim(3e-16, 3.0)
    ax1.legend(loc="lower right", fontsize=7, framealpha=0.95)

    # (c) per-frequency imbalance
    e, _ = spec["r8"]
    pw = np.abs(z["r8/spec_out"])
    axb = ax2.twinx()
    axb.fill_between(e, pw / pw.max(), color="#949494", alpha=0.18, lw=0)
    axb.set_ylabel(r"bubble throughput $|P_{\rm out}(\omega)|$ (norm.)",
                   color="#707070", fontsize=8.5)
    axb.tick_params(axis="y", colors="#707070")
    axb.set_ylim(0, 1.05)
    axb.grid(False)
    axb.set_zorder(0)
    ax2.set_zorder(1)
    ax2.patch.set_visible(False)

    ax2.axvspan(*OPT, color=C_ANH, alpha=0.07, lw=0)
    ax2.axhline(0.0, color=C_BALL, lw=1.0, ls=":")
    # rank colours here are NOT the panel-(a) roles: C_MIN/green means
    # "min-image" on the left, so the ranks get their own ramp.
    for leg, col, lab in [("r8", C_ANH, "$R=8$"), ("r16", "#de8f05", "$R=16$"),
                          ("r32", "#cc78bc", "$R=32$"),
                          ("r64", "#ca9161", "$R=64$")]:
        ee, d = spec[leg]
        ax2.plot(ee, d / runs[leg]["J"], lw=1.1, color=col, label=lab)
    ax2.set_xlabel(r"$\omega$ (THz)")
    ax2.set_ylabel(r"per-frequency imbalance  $d(\omega)/J$")
    ax2.set_xlim(0, e.max())
    ax2.legend(loc="upper left", fontsize=7.5)

    style.save(fig, "decomp_conservation", directory=FIGDIR)


if __name__ == "__main__":
    main()
