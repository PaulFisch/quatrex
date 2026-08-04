"""MoS2 film instability figures (fig:res_mos2_spiral, fig:res_mos2_stab).

  mos2_spiral       the 95-iteration record of the eta=0 cubic-only
                    film SCBA (SCP(300 K) fc2, linear 0.1 mixing,
                    nu grid): (a) rel Sigma^R residual with the
                    metastable orbit and the terminal burst; (b) the
                    bounded, non-converging orbit in the leading two
                    principal components of the per-iteration heat
                    matrix; (c) max|Sigma^<| per (iteration, energy)
                    on the low-frequency slice -- the burst locus
                    sits on the soft interlayer modes.
  mos2_stabilisers  (a) residual traces of the FULL-vertex stabiliser
                    probes on the film fixed point: the 95-it linear
                    record, the orbit-mean restart, the SCP tadpole,
                    and the two quartic-loop attempts -- none
                    descends below 0.62 and both loop probes
                    diverge. (b) the vertex-ablation control: on the
                    accidental diagonal-only (no cross-slab FC3)
                    build the same iteration descends -- probe c
                    monotonically to 0.087 before its 55-it cap
                    (unrecorded per-run overrides), the
                    current-code defaults continuation to 0.646
                    before a late divergence at 66, and the
                    full-provenance heavy-damped rerun (alpha=0.05,
                    400-it budget, explicit job.sh env record) to
                    0.355 before diverging at 47; the in-code
                    ablation on the correct build + resolved grid
                    (xs0, sse_cross_slab_scale=0) dips to 0.375 and
                    diverges at 38. The ablated model is gentler but
                    nowhere convergent -- the cross-slab channel
                    destabilises, and probe c's monotone descent is
                    unreproduced under recorded conditions.

Data: phonon/scripts/data/mos2_spiral.npz, distilled by
_extract_mos2_spiral.py (see its docstring for the full vertex
provenance of every run). All runs eta=0.

Run:  python phonon/scripts/figures/mos2_spiral.py
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

DATA = ROOT / "phonon/scripts/data/mos2_spiral.npz"
FIGDIR = ROOT / "document/fig/transport_sweeps"


def orbit_periods(x: np.ndarray, n_top: int = 3) -> list[float]:
    """Leading periods (iterations) from the PC-trajectory spectrum."""
    y = x - x.mean()
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    ks = np.argsort(spec[1:])[::-1][:n_top] + 1
    return [len(y) / int(k) for k in ks]


def fig_spiral(d) -> None:
    res = d["res_long"][:, 0]
    heat = d["heat_long"]
    sig = d["sigmax_long"]
    en = d["energies_lo"]

    fig, (ax_a, ax_b, ax_c) = style.figure(ncols=3, width=3.5, height=3.1)
    colors = style.RC["axes.prop_cycle"].by_key()["color"]

    it = np.arange(1, len(res) + 1)
    ax_a.semilogy(it, res, color=colors[0], lw=1.2)
    ax_a.set_xlabel("iteration")
    ax_a.set_ylabel(r"rel $\Sigma^R$ residual")
    ax_a.axvspan(30, 88, color=colors[2], alpha=0.12, linewidth=0)
    ax_a.annotate("metastable orbit", (58, res[30:88].max() * 2),
                  ha="center", fontsize=8)
    ax_a.annotate("burst", (len(res) - 1, res[-1]),
                  textcoords="offset points", xytext=(-28, -2), fontsize=8)

    # PCA of the heat-matrix trajectory over the orbit window
    w0, w1 = 10, 88
    flat = heat[w0:w1].reshape(w1 - w0, -1)
    flat = flat - flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    pc = flat @ vt[:2].T
    ax_b.plot(pc[:, 0], pc[:, 1], "-", color="0.75", lw=0.8, zorder=1)
    sc = ax_b.scatter(pc[:, 0], pc[:, 1], c=np.arange(w0, w1), s=9,
                      cmap="viridis", zorder=2)
    fig.colorbar(sc, ax=ax_b, label="iteration", shrink=0.85)
    ax_b.set_xlabel("heat-matrix PC 1")
    ax_b.set_ylabel("heat-matrix PC 2")

    pers = orbit_periods(pc[:, 0])

    it_sig = np.arange(1, sig.shape[0] + 1)
    m = ax_c.pcolormesh(it_sig, en, np.log10(np.maximum(sig, 1e-3)).T,
                        cmap="magma", shading="nearest", rasterized=True)
    fig.colorbar(m, ax=ax_c, label=r"$\log_{10}\max|\Sigma^<|$", shrink=0.85)
    ax_c.set_xlabel("iteration")
    ax_c.set_ylabel(r"$\omega$ (THz)")
    ax_c.set_ylim(0, en.max())

    style.save(fig, "mos2_spiral", directory=FIGDIR)

    print(f"long record: {len(res)} residual points, min {res.min():.3e} "
          f"at it {res.argmin() + 1}, final {res[-1]:.3e}")
    print("orbit recurrence, leading PC1 periods (iterations): "
          + ", ".join(f"{p:.0f}" for p in pers)
          + "  (slow envelope + subharmonics; no single clean period)")
    p90 = np.percentile(sig[-1], 90)
    locus = en[sig[-1] > p90]
    print(f"burst locus (last iteration, top decile of max|Sigma^<|): "
          f"{locus.min():.2f}-{locus.max():.2f} THz on the "
          f"{en.min():.2f}-{en.max():.2f} THz slice")


def fig_stabilisers(d) -> None:
    fig, (ax_a, ax_b) = style.figure(ncols=2, width=4.4, height=3.4)
    colors = style.RC["axes.prop_cycle"].by_key()["color"]

    long_res = d["res_long"][:, 0]
    ax_a.semilogy(np.arange(1, len(long_res) + 1), long_res, color="0.6",
                  lw=1.1, label="linear 0.1 (95-it record)")
    named = [("orbit_mean", "orbit-mean restart", 2),
             ("tadpole", "SCP tadpole", 3),
             ("loop3", "tadpole + quartic loop", 4),
             ("loop4", "quartic loop only", 5)]
    for key, lab, ci in named:
        r = d[f"res_{key}"][:, 0]
        ax_a.semilogy(np.arange(1, len(r) + 1), r, color=colors[ci],
                      lw=1.3, label=lab)
    ax_a.set_xlabel("iteration")
    ax_a.set_ylabel(r"rel $\Sigma^R$ residual")
    ax_a.legend(fontsize=7, loc="lower right")

    ax_b.semilogy(np.arange(1, len(long_res) + 1), long_res, color="0.6",
                  lw=1.1, label="full vertex (record)")
    for key, lab, ci, lw in (
            ("abl_c", "ablated, probe c", 0, 1.4),
            ("abl_a", "ablated, probe a", 0, 0.8),
            ("abl_cont", "ablated, defaults (250-it cont.)", 1, 1.2),
            ("ablcoarse", r"ablated, $\alpha=0.05$ rerun", 2, 1.2),
            ("xs0", "in-code ablation, resolved grid", 4, 1.2)):
        r = d[f"res_{key}"][:, 0]
        ax_b.semilogy(np.arange(1, len(r) + 1), r, color=colors[ci],
                      lw=lw, alpha=1.0 if lw > 1 else 0.5,
                      label=lab)
    ax_b.set_xlabel("iteration")
    ax_b.legend(fontsize=7, loc="lower left")

    style.save(fig, "mos2_stabilisers", directory=FIGDIR)

    print("FULL-vertex probes, min residual (iterations):")
    for key in ("orbit_mean", "tadpole", "loop3", "loop4"):
        r = d[f"res_{key}"][:, 0]
        print(f"  {key:11s} {r.min():.3e} ({len(r)} it, last {r[-1]:.3e})")
    print(f"  long record {long_res.min():.3e} ({len(long_res)} it)")
    print("ABLATED-vertex (diagonal-only) probes:")
    for key in ("abl_a", "abl_b", "abl_c", "abl_floor", "abl_cont",
                "ablcoarse", "xs0"):
        r = d[f"res_{key}"][:, 0]
        print(f"  {key:11s} {r.min():.3e} ({len(r)} it, last {r[-1]:.3e})")
    print("grid cells (full vertex): u121 / aux13 / u2001:")
    for key in ("u121", "aux13", "u2001"):
        r = d[f"res_{key}"][:, 0]
        print(f"  {key:11s} {r.min():.3e} ({len(r)} it, last {r[-1]:.3e})")


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA)
    fig_spiral(d)
    fig_stabilisers(d)


if __name__ == "__main__":
    main()
