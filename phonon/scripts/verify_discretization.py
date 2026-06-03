#!/usr/bin/env python
"""Verify discretization convergence of the dense SCBA solver.

Verification work-stream Part 5. On a fixed physical toy (a monatomic
chain whose acoustic band gives a multi-mode device) it refines each
grid parameter and characterises convergence:

  * frequency-grid density ``ne`` -- the heat-flow conservation error
    must decrease monotonically as the grid is refined. Conservation
    error is the cleanest discretization metric: it vanishes in the
    continuum limit and exposes the grid error directly.
  * frequency range ``fmax`` -- the heat current must saturate once the
    range covers the 3-phonon convolution support (~2 x omega_max).
  * broadening ``eta`` -- the central finding: the heat-current *value*
    of a resonant device is only trustworthy when eta resolves the
    device resonances (eta >~ d_omega) AND is small enough not to
    over-broaden them. Reported, with the eta/d_omega guidance.
  * device length ``n_slabs`` -- informational.

A subtle point this surfaces: a small conservation error is necessary
but NOT sufficient for a converged value -- an under-resolved grid can
over-count J_L and J_R symmetrically, so conservation looks good while
the value is wrong. Always check value stability under refinement too.

The end-to-end converged defaults for the real d5a SiNW come from
running this kind of sweep through ``transmission_finite`` on the
cluster; this script establishes the methodology and guidance locally.

Run::

    /home/paul/miniconda3/envs/quatrex-dev/bin/python \\
        phonon/scripts/verify_discretization.py

Exits non-zero if a convergence check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PHONON_DIR = _REPO_ROOT / "phonon"
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _PHONON_DIR, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from solver.toy_models import monatomic_chain  # noqa: E402
from verify_scba_convergence import (  # noqa: E402
    _build_toy_device,
    _calibrate_phi_scale,
    _run_scba,
)

_N_SLABS = 10
_TEMP = 300.0
_DELTA_T = 20.0


def _heat_current(dev, run) -> float:
    res = run["result"]
    j_anh = 0.5 * (np.asarray(res["spectral_J_L"])
                   + np.asarray(res["spectral_J_R"]))
    return float(np.sum(j_anh[dev["pos_mask"]]) * dev["dw"] * 1e12)


def _calibrated_phi(toy, n_slabs, target_fraction):
    """Calibrate the cubic vertex once; the FC3 is grid-independent."""
    ref = _build_toy_device(toy, n_slabs, (0.01, 20.0, 90),
                            eta_factor=2.0, temperature=_TEMP,
                            delta_T=_DELTA_T)
    scale = _calibrate_phi_scale(toy, ref, target_fraction)
    return {(i, i, i): (toy.phi * scale).astype(complex)
            for i in range(n_slabs)}


def _run(dev, phi_dev):
    return _run_scba(dev, phi_dev, mixing=0.4, anderson=False, depth=5,
                     max_iter=80, scba_tol=1e-4, conservation_tol=5e-3,
                     dc_handling="interpolate")


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def sweep_ne(report, plotdata, toy, phi_dev):
    """Refine the frequency grid at fixed eta; conservation must converge."""
    fmax, eta_w = 20.0, 0.45
    ne_values = [30, 45, 60, 90, 135]
    js, cons = [], []
    for ne in ne_values:
        dw = fmax / ne
        dev = _build_toy_device(toy, _N_SLABS, (0.01, fmax, ne),
                                eta_factor=eta_w / dw, temperature=_TEMP,
                                delta_T=_DELTA_T)
        run = _run(dev, phi_dev)
        js.append(_heat_current(dev, run))
        cons.append(run["conservation"])
    plotdata["ne"] = (ne_values, js, cons)

    # Conservation error must fall monotonically and reach a small value.
    monotone = all(cons[i + 1] <= cons[i] * 1.05 for i in range(len(cons) - 1))
    converged = cons[-1] < 5e-3 and cons[-1] < 0.2 * cons[0]
    report("conservation error converges with frequency-grid density",
           monotone and converged,
           f"conservation {cons[0]:.1e} -> {cons[-1]:.1e} over "
           f"ne={ne_values[0]}..{ne_values[-1]}")

    # Value drift is informational: the resonant toy integral is grid
    # sensitive; the real d5 (many bands) is smoother.
    drift = abs(js[-1] - js[-2]) / abs(js[-1]) if js[-1] else float("inf")
    report("heat-current value drift with ne (informational)", True,
           f"|dJ/J| between the two finest grids = {drift:.2e}")


def sweep_range(report, plotdata, toy, phi_dev):
    """Frequency range must cover the 3-phonon support (~2 omega_max)."""
    omega_max = float(np.sqrt(np.max(np.linalg.eigvalsh(
        toy.device_dynamical_matrix(_N_SLABS)))))
    dw_target = 0.25
    fmax_values = [10.0, 13.0, 16.0, 20.0, 26.0]
    js = []
    for fmax in fmax_values:
        ne = max(8, int(round(fmax / dw_target)))
        dev = _build_toy_device(toy, _N_SLABS, (0.01, fmax, ne),
                                eta_factor=2.0, temperature=_TEMP,
                                delta_T=_DELTA_T)
        run = _run(dev, phi_dev)
        js.append(_heat_current(dev, run))
    plotdata["range"] = (fmax_values, js, omega_max)

    covered = [j for fmax, j in zip(fmax_values, js)
               if fmax >= 2.0 * omega_max]
    spread = ((max(covered) - min(covered)) / abs(np.mean(covered))
              if len(covered) >= 2 and np.mean(covered) != 0
              else float("inf"))
    report("heat current saturates once fmax covers 2 x omega_max",
           spread < 0.05,
           f"omega_max = {omega_max:.2f} THz; spread above 2 omega_max = "
           f"{spread:.2e}")


def sweep_eta(report, plotdata, toy, phi_dev):
    """Characterise the eta sensitivity, honestly separating converged runs.

    A run that does not meet the SCBA convergence/conservation gate
    returns a meaningless heat current; lumping those into an eta
    "spread" would fake a huge sensitivity. Only converged runs are
    compared here.
    """
    fmax, ne = 20.0, 90
    dw = fmax / ne
    eta_factors = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    js, cons, converged = [], [], []
    for ef in eta_factors:
        dev = _build_toy_device(toy, _N_SLABS, (0.01, fmax, ne),
                                eta_factor=ef, temperature=_TEMP,
                                delta_T=_DELTA_T)
        run = _run(dev, phi_dev)
        js.append(_heat_current(dev, run))
        cons.append(run["conservation"])
        converged.append(run["converged"])
    plotdata["eta"] = (eta_factors, js, cons, dw, converged)

    conv_js = [j for j, c in zip(js, converged) if c]
    conv_ef = [e for e, c in zip(eta_factors, converged) if c]
    failed_ef = [e for e, c in zip(eta_factors, converged) if not c]
    if len(conv_js) >= 2:
        ratio = max(np.abs(conv_js)) / max(min(np.abs(conv_js)), 1e-300)
    else:
        ratio = float("nan")
    # This is an honest finding, not a pass/fail: for a resonant toy
    # device the heat-current value is genuinely eta-sensitive even
    # among converged runs, and its eta->0 limit must be established
    # by extrapolation. The real d5 system has to be checked directly.
    report("eta sensitivity characterised (informational)", True,
           f"d_omega={dw:.3f} THz; converged at eta/d_omega in "
           f"{conv_ef}, NOT converged at {failed_ef}; among converged "
           f"runs the heat current still varies x{ratio:.1f} -- the "
           f"value is resonance-resolution limited and its eta->0 "
           f"limit needs extrapolation (verify on d5 directly)")


def sweep_n_slabs(report, plotdata, toy):
    """Device-length dependence -- informational, not a grid artefact."""
    fmax, ne = 20.0, 90
    n_slabs_values = [4, 6, 8, 10, 12]
    js = []
    for n_slabs in n_slabs_values:
        phi_dev = _calibrated_phi(toy, n_slabs, target_fraction=0.1)
        dev = _build_toy_device(toy, n_slabs, (0.01, fmax, ne),
                                eta_factor=2.0, temperature=_TEMP,
                                delta_T=_DELTA_T)
        run = _run(dev, phi_dev)
        js.append(_heat_current(dev, run))
    plotdata["n_slabs"] = (n_slabs_values, js)
    report("device-length (n_slabs) dependence recorded (informational)",
           True,
           "J(n_slabs) = [" + ", ".join(f"{j:.2e}" for j in js)
           + "] pW -- a physical length dependence, not a grid artefact")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _make_plots(plotdata, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    if "ne" in plotdata:
        ne, js, cons = plotdata["ne"]
        ax.semilogy(ne, cons, "o-", color="crimson")
        ax.set_xlabel("frequency points  ne")
        ax.set_ylabel("heat-flow conservation error")
        ax.set_title("Conservation converges with grid density")

    ax = axes[0, 1]
    if "range" in plotdata:
        fmax, js, omega_max = plotdata["range"]
        ax.plot(fmax, js, "^-")
        ax.axvline(2.0 * omega_max, color="k", ls=":",
                   label=r"$2\,\omega_{\max}$")
        ax.set_xlabel("frequency range  fmax (THz)")
        ax.set_ylabel("anharmonic heat current (pW)")
        ax.set_title("Convergence vs frequency range")
        ax.legend()

    ax = axes[1, 0]
    if "eta" in plotdata:
        ef, js, cons, dw, converged = plotdata["eta"]
        ax.loglog(ef, np.abs(js), "s-", label="|heat current|")
        ax.loglog(ef, cons, "o--", color="crimson",
                  label="conservation error")
        ax.axvspan(1.0, 2.0, color="green", alpha=0.12,
                   label=r"$\eta/d\omega\sim 1$-$2$")
        ax.set_xlabel(r"$\eta / d\omega$")
        ax.set_title(f"eta resolution trade-off (d_omega={dw:.3f} THz)")
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    if "n_slabs" in plotdata:
        ns, js = plotdata["n_slabs"]
        ax.plot(ns, js, "D-")
        ax.set_xlabel("device length  n_slabs")
        ax.set_ylabel("anharmonic heat current (pW)")
        ax.set_title("Length dependence (informational)")

    fig.tight_layout()
    out_path = out_dir / "verify_discretization.pdf"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    def report(name, passed, detail):
        results.append((name, bool(passed), detail))
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {name}" + (f"  ({detail})" if detail else ""))

    print("=== verify_discretization: grid-convergence audit ===\n")

    toy = monatomic_chain(omega_max_thz=8.0)
    phi_dev = _calibrated_phi(toy, _N_SLABS, target_fraction=0.1)
    plotdata: dict = {}

    print("-- frequency-grid density --")
    sweep_ne(report, plotdata, toy, phi_dev)
    print("\n-- frequency range --")
    sweep_range(report, plotdata, toy, phi_dev)
    print("\n-- broadening eta --")
    sweep_eta(report, plotdata, toy, phi_dev)
    print("\n-- device length --")
    sweep_n_slabs(report, plotdata, toy)

    out_dir = _REPO_ROOT / "phonon/scripts/out/verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = _make_plots(plotdata, out_dir)
    print(f"\n  diagnostic plot: {plot_path}")

    print("\n-- recommended grid defaults --")
    print("    * fmax >= 2 x the maximum phonon frequency, so the 3-phonon "
          "convolution support is fully sampled.")
    print("    * eta/d_omega ~ 1-2: large enough to resolve device "
          "resonances on the grid, small enough to avoid over-broadening. "
          "The production default eta_factor=0.05 is far below this and "
          "under-resolves isolated modes (cf. Part 2).")
    print("    * refine ne until the conservation error is small AND the "
          "heat-current value is stable -- conservation alone can look "
          "good on an under-resolved grid that over-counts symmetrically.")

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print("FAILED: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
