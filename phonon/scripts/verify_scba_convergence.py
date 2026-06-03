#!/usr/bin/env python
"""Verify and characterise the SCBA self-consistency loop of the dense solver.

Verification work-stream Part 4. Runs the *production* SCBA fixed-point
loop (:func:`phonon.solver.dense.scba_loop_dev`) on the analytic toy
systems, with the 3-phonon vertex auto-scaled so the scattering
self-energy is a controlled fraction of the device Hamiltonian
("weak / medium / strong" anharmonicity). For each anharmonic strength
it sweeps the mixing scheme and records the per-iteration convergence
history (dJ/J, dSigma/Sigma, conservation, ||Sigma^R||).

It answers "how do we get the solver to converge well?" and audits
three loop-design concerns flagged in the static review:

  (a) Anderson mixing flattens the residual with no norm scaling across
      Sigma's dynamic range -- checked empirically against linear mixing.
  (b) Sigma^R is rebuilt *after* mixing -- checked for consistency
      (Sigma^R must equal build_retarded of the stored Sigma^{<,>}).
  (c) symmetrize_lesser_greater runs every iteration -- its correction
      magnitude is measured on the converged self-energy.

Outputs a recommended-settings table and convergence plots in
``phonon/scripts/out/verify/``.

Run::

    /home/paul/miniconda3/envs/quatrex-dev/bin/python \\
        phonon/scripts/verify_scba_convergence.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PHONON_DIR = _REPO_ROOT / "phonon"
for _p in (_REPO_ROOT, _PHONON_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phonon_inputs.constants import THZ_TO_RAD  # noqa: E402
from solver.dense import scba_loop_dev  # noqa: E402
from solver.diagnostics import symmetrize_lesser_greater  # noqa: E402
from solver.grids import build_frequency_grid  # noqa: E402
from solver.leads import (  # noqa: E402
    build_device_hamiltonian,
    compute_obc_batch,
    solve_green_batch,
)
from solver.retarded import build_retarded  # noqa: E402
from solver.se_finite import (  # noqa: E402
    compute_phph_self_energy_finite_multi_slab,
)
from solver.toy_models import diatomic_chain  # noqa: E402
from solver.zero_modes import (  # noqa: E402
    build_translation_projector,
    translation_leakage,
)


# ---------------------------------------------------------------------------
# Toy device + SCBA harness
# ---------------------------------------------------------------------------


def _g_dict_from_dense(g_dense, n_slabs, n_dof):
    out = {}
    for k in range(n_slabs):
        s_k = slice(k * n_dof, (k + 1) * n_dof)
        for kp in range(n_slabs):
            s_kp = slice(kp * n_dof, (kp + 1) * n_dof)
            out[(k, kp)] = g_dense[:, s_k, s_kp]
    return out


def _build_toy_device(toy, n_slabs, freq_range, eta_factor,
                       temperature, delta_T):
    """Assemble the leads + device matrices for a toy SCBA run."""
    freqs, dw, eta_w, z2, pos_mask, mid = build_frequency_grid(
        freq_range, eta_factor=eta_factor)
    h00 = toy.h00.astype(complex)
    h01 = toy.h01.astype(complex)
    h_d = build_device_hamiltonian(h00, h01, n_slabs)
    t_l = temperature + delta_T / 2.0
    t_r = temperature - delta_T / 2.0
    obc = compute_obc_batch(z2, h00, h01, freqs, t_l, t_r, n_slabs=n_slabs)
    return {
        "freqs": freqs, "dw": dw, "z2": z2, "pos_mask": pos_mask,
        "h00": h00, "h01": h01, "h_d": h_d, "obc": obc,
        "t_l": t_l, "t_r": t_r, "n_dof": toy.n_dof,
        "n_slabs": n_slabs, "N_D": n_slabs * toy.n_dof,
    }


def _make_se_kernel(phi_dev, dev, dc_handling="interpolate"):
    """Build the se_kernel closure scba_loop_dev expects."""
    n_slabs, n_dof, N_D = dev["n_slabs"], dev["n_dof"], dev["N_D"]
    freqs, dw = dev["freqs"], dev["dw"]
    nfreq = len(freqs)

    def se_kernel(g_less_dev_q, g_great_dev_q):
        sig_l = np.zeros((1, nfreq, N_D, N_D), dtype=complex)
        sig_g = np.zeros_like(sig_l)
        gl = _g_dict_from_dense(g_less_dev_q[0], n_slabs, n_dof)
        gg = _g_dict_from_dense(g_great_dev_q[0], n_slabs, n_dof)
        sl_blocks, sg_blocks = compute_phph_self_energy_finite_multi_slab(
            gl, gg, phi_dev, n_slabs, freqs, dw,
            sigma_cutoff=None, g_cutoff=None, dc_handling=dc_handling,
            n_threads=1,
        )
        for (i, j), blk in sl_blocks.items():
            s_i = slice(i * n_dof, (i + 1) * n_dof)
            s_j = slice(j * n_dof, (j + 1) * n_dof)
            sig_l[0, :, s_i, s_j] = blk
        for (i, j), blk in sg_blocks.items():
            s_i = slice(i * n_dof, (i + 1) * n_dof)
            s_j = slice(j * n_dof, (j + 1) * n_dof)
            sig_g[0, :, s_i, s_j] = blk
        return sig_l, sig_g

    return se_kernel


def _calibrate_phi_scale(toy, dev, target_fraction):
    """Scale the cubic vertex so max|Sigma^R| ~ target_fraction * max|H_D|.

    Runs one bubble on the ballistic Green's function; Sigma scales as
    phi**2, so a single rescale hits the target.
    """
    n_slabs, N_D = dev["n_slabs"], dev["N_D"]
    phi_dev = {(i, i, i): toy.phi.astype(complex)
               for i in range(n_slabs)}
    se_kernel = _make_se_kernel(phi_dev, dev)
    zero = np.zeros((len(dev["freqs"]), N_D, N_D), dtype=complex)
    g_ret, g_l, g_g = solve_green_batch(
        dev["z2"], dev["h_d"], dev["obc"], zero, zero, zero)
    sig_l, sig_g = se_kernel(g_l[None], g_g[None])
    sig_r = build_retarded(sig_l[0], sig_g[0], dev["freqs"], method="fft")
    sig_mag = float(np.max(np.abs(sig_r)))
    h_mag = float(np.max(np.abs(dev["h_d"])))
    if sig_mag <= 0:
        return 1.0
    return float(np.sqrt(target_fraction * h_mag / sig_mag))


_RE = {
    "dJ": re.compile(r"dJ/J = ([0-9.eE+-]+)"),
    "dS": re.compile(r"resid = ([0-9.eE+-]+)"),
    "cons": re.compile(r"conservation = ([0-9.eE+-]+)"),
    "sigR": re.compile(r"max\|Sigma\^R\| = ([0-9.eE+-]+)"),
}


def _run_scba(dev, phi_dev, *, mixing, anderson, depth,
              max_iter=60, scba_tol=1e-4, conservation_tol=5e-3,
              dc_handling="interpolate",
              solver=None, anderson_safeguard=True,
              zero_mode_projection=False, gate_on_conservation=False,
              divergence_guard=True, masses=None):
    """Run scba_loop_dev, capturing the per-iteration convergence trace."""
    se_kernel = _make_se_kernel(phi_dev, dev, dc_handling=dc_handling)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = scba_loop_dev(
            z2_arr=dev["z2"], freqs_thz=dev["freqs"], dw_thz=dev["dw"],
            omega_rad=dev["freqs"] * THZ_TO_RAD, pos_mask=dev["pos_mask"],
            n_slabs=dev["n_slabs"], n_dof=dev["n_dof"], N_D=dev["N_D"],
            H_D_list=[dev["h_d"]], obc_list=[dev["obc"]],
            btd_blocks_list=[(dev["h00"], dev["h01"])], n_kpts=1,
            se_kernel=se_kernel, T_L=dev["t_l"], T_R=dev["t_r"],
            max_scba_iter=max_iter, scba_tol=scba_tol,
            conservation_tol=conservation_tol,
            mixing=mixing, anderson_mixing=anderson, anderson_depth=depth,
            scattering_contacts=False, retarded="fft", verbose=True,
            solver=solver, anderson_safeguard=anderson_safeguard,
            zero_mode_projection=zero_mode_projection,
            gate_on_conservation=gate_on_conservation,
            divergence_guard=divergence_guard, masses_primitive=masses,
        )
    text = buf.getvalue()
    trace = {k: [] for k in _RE}
    for line in text.splitlines():
        for key, rx in _RE.items():
            m = rx.search(line)
            if m:
                trace[key].append(float(m.group(1)))
    converged = bool(result.get("converged", "Converged after" in text))
    n_iter = len(result["convergence_history"]) + 1
    return {
        "trace": {k: np.array(v) for k, v in trace.items()},
        "converged": converged,
        "n_iter": n_iter,
        "conservation": result["conservation_err"],
        "residual": float(result.get("scba_residual", float("nan"))),
        "result": result,
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_baseline_converges(report, store):
    """The loop converges on a medium-anharmonicity toy at default mixing."""
    toy = diatomic_chain()
    dev = _build_toy_device(toy, n_slabs=3, freq_range=(0.01, 20.0, 60),
                            eta_factor=1.0, temperature=300.0, delta_T=20.0)
    scale = _calibrate_phi_scale(toy, dev, target_fraction=0.1)
    phi_dev = {(i, i, i): (toy.phi * scale).astype(complex)
               for i in range(dev["n_slabs"])}
    run = _run_scba(dev, phi_dev, mixing=0.5, anderson=False, depth=5)
    store["dev"] = dev
    store["phi_dev"] = phi_dev
    store["baseline"] = run
    report("SCBA loop converges (medium anharmonicity, linear mix=0.5)",
           run["converged"],
           f"converged in {run['n_iter']} iters, "
           f"conservation = {run['conservation']:.2e}")


def check_mixing_study(report, store, plotdata):
    """Sweep the mixing scheme; the loop must converge for a sane choice."""
    dev = store["dev"]
    phi_dev = store["phi_dev"]
    rows = []
    for alpha in (0.2, 0.4, 0.6, 0.8, 1.0):
        run = _run_scba(dev, phi_dev, mixing=alpha, anderson=False, depth=5)
        rows.append(("linear", alpha, None, run))
    for depth in (3, 5):
        run = _run_scba(dev, phi_dev, mixing=0.5, anderson=True, depth=depth)
        rows.append(("anderson", 0.5, depth, run))
    plotdata["mixing_rows"] = rows

    any_converged = any(r[3]["converged"] for r in rows)
    report("at least one mixing setting converges the loop", any_converged,
           "")

    # Anderson is the stabilized scheme (restart + step cap + stronger
    # regularization). On these toys it converges; the bare Anderson it
    # replaced diverges on the strongly anharmonic d5a SiNW (see the
    # convergence writeup -- conservation blows up to 100 %).
    lin = [r for r in rows if r[0] == "linear" and r[3]["converged"]]
    ander = [r for r in rows if r[0] == "anderson" and r[3]["converged"]]
    best_lin = min((r[3]["n_iter"] for r in lin), default=10**9)
    best_and = min((r[3]["n_iter"] for r in ander), default=10**9)
    report("stabilized Anderson mixing converges the toy systems",
           len(ander) > 0,
           f"best Anderson = {best_and} iters, best linear = {best_lin} iters")
    store["mixing_rows"] = rows


def check_sigma_r_consistency(report, store):
    """Issue (b): Sigma^R at loop exit equals build_retarded(Sigma^<,>)."""
    run = store["baseline"]["result"]
    sig_r = run["Sigma_R"]
    rebuilt = build_retarded(run["Sigma_l"], run["Sigma_g"],
                             store["dev"]["freqs"], method="fft")
    rel = float(np.max(np.abs(sig_r - rebuilt))
                / (np.max(np.abs(sig_r)) + 1e-300))
    report("Sigma^R is consistent with the stored Sigma^{<,>} (no lag bug)",
           rel < 1e-12, f"max rel mismatch = {rel:.2e}")


def check_symmetrization_correction(report, store):
    """Issue (c): the per-iteration symmetrization is a small correction."""
    run = store["baseline"]["result"]
    sl = run["Sigma_l"].copy()
    sg = run["Sigma_g"].copy()
    sl0, sg0 = sl.copy(), sg.copy()
    symmetrize_lesser_greater(sl, sg)
    scale = float(np.max(np.abs(sl0)) + np.max(np.abs(sg0)) + 1e-300)
    corr = float(np.max(np.abs(sl - sl0)) + np.max(np.abs(sg - sg0))) / scale
    report("symmetrization is a small correction at convergence "
           "(not masking a bug)", corr < 1e-2,
           f"relative correction = {corr:.2e}")


def check_strength_dependence(report, plotdata):
    """Across anharmonic strengths, find the mixing that converges.

    The physical regime for a Si nanowire at 300 K is weak (Sigma is a
    ~1 % perturbation). "strong" (|Sigma|/|H| ~ 0.45) is an unphysical
    stress test that probes the SCBA fixed-point convergence radius.
    """
    toy = diatomic_chain()
    summary = []
    for label, frac in (("weak", 0.03), ("medium", 0.15), ("strong", 0.45)):
        dev = _build_toy_device(toy, n_slabs=3, freq_range=(0.01, 20.0, 60),
                                eta_factor=1.0, temperature=300.0,
                                delta_T=20.0)
        scale = _calibrate_phi_scale(toy, dev, target_fraction=frac)
        phi_dev = {(i, i, i): (toy.phi * scale).astype(complex)
                   for i in range(dev["n_slabs"])}
        best = None
        # linear sweep; for the stress case also probe heavy damping.
        alphas = ((0.1, 0.15, 0.2, 0.3, 0.5)
                  if label == "strong" else (0.2, 0.3, 0.5, 0.7))
        for alpha in alphas:
            run = _run_scba(dev, phi_dev, mixing=alpha, anderson=False,
                            depth=5)
            if run["converged"] and (best is None
                                     or run["n_iter"] < best[1]):
                best = (f"linear {alpha}", run["n_iter"])
        for depth in (3, 5):
            run = _run_scba(dev, phi_dev, mixing=0.3, anderson=True,
                            depth=depth)
            if run["converged"] and (best is None
                                     or run["n_iter"] < best[1]):
                best = (f"Anderson d{depth}", run["n_iter"])
        summary.append((label, frac, best))
    plotdata["strength_summary"] = summary

    # Hard requirement: the physical regime (weak, medium) must converge.
    phys_ok = all(b[2] is not None for b in summary
                  if b[0] in ("weak", "medium"))
    detail = "; ".join(
        f"{lab}: " + (f"{b[0]} -> {b[1]} it" if b else "no convergence")
        for lab, frac, b in summary)
    report("physical regime (weak, medium anharmonicity) converges",
           phys_ok, detail)

    # The stress case is informational: it maps the convergence radius.
    strong = next(s for s in summary if s[0] == "strong")
    if strong[2] is None:
        msg = ("|Sigma|/|H|~0.45 did not converge with linear mixing "
               "0.1-0.5 or Anderson -- beyond the SCBA fixed-point radius "
               "(far stronger than any physical SiNW)")
    else:
        msg = (f"|Sigma|/|H|~0.45 converges with {strong[2][0]} "
               f"in {strong[2][1]} iters (needs heavy damping)")
    report("strong-anharmonicity convergence radius (informational)",
           True, msg)


def check_d5_convergence(report):
    """Opt-in: run the real transmission_finite SCBA on the d5a SiNW.

    The toy systems above are weak/medium; the actual d5a wire is
    strongly anharmonic (|Sigma| ~ |H|). This exercises the shipped
    code path on the real system with both converging schemes.
    """
    cfg = _REPO_ROOT / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml"
    if not cfg.exists():
        report("d5a SiNW SCBA convergence (skipped -- config absent)",
               True, "")
        return
    from finite_analysis.loader import load_system
    from solver.dense import transmission_finite

    bundle = load_system(str(cfg), validate=False, transport_axis=2)
    fc3 = str(bundle.meta["fc3_path"])
    rows = []
    for label, kw in (
        ("linear mix=0.3", dict(anderson_mixing=False, mixing=0.3)),
        ("stabilized Anderson",
         dict(anderson_mixing=True, mixing=0.5, anderson_depth=8)),
    ):
        res = transmission_finite(
            bundle.phonon, fc3_hdf5=fc3, freq_range_thz=(0.01, 18.0, 21),
            transport_direction="z", eta_factor=0.05, temperature=300.0,
            delta_T=20.0, max_scba_iter=90, scba_tol=1e-3,
            conservation_tol=2e-2, n_slabs=1, verbose=False, **kw)
        n_it = res["n_scba_iterations"]
        gb = res["thermal_conductance_ballistic"]
        ga = res["thermal_conductance_anharmonic"]
        rows.append((label, n_it, res["heat_flow_conservation"], ga / gb))
    detail = "; ".join(
        f"{lab}: {n} iters, cons={c:.1e}, G_anh/G_ball={r:.3f}"
        for lab, n, c, r in rows)
    both_conv = all(n < 90 for _, n, _, _ in rows)
    ratios = [r for *_, r in rows]
    agree = abs(ratios[0] - ratios[1]) < 0.02
    report("d5a SiNW SCBA converges (linear 0.3 and stabilized Anderson, "
           "consistent G_anh)", both_conv and agree, detail)


def check_safeguarded_anderson(report, plotdata):
    """Safeguarded Anderson converges a strongly-anharmonic multi-slab
    toy and is no worse than the legacy hard-restart scheme."""
    toy = diatomic_chain()
    dev = _build_toy_device(toy, n_slabs=4, freq_range=(0.01, 20.0, 60),
                            eta_factor=1.0, temperature=300.0, delta_T=20.0)
    scale = _calibrate_phi_scale(toy, dev, target_fraction=0.4)
    phi_dev = {(i, i, i): (toy.phi * scale).astype(complex)
               for i in range(dev["n_slabs"])}
    legacy = _run_scba(dev, phi_dev, mixing=0.3, anderson=True, depth=8,
                       max_iter=120, anderson_safeguard=False)
    safe = _run_scba(dev, phi_dev, mixing=0.3, anderson=True, depth=8,
                     max_iter=120, anderson_safeguard=True)
    plotdata["safeguard_rows"] = [("legacy", legacy), ("safeguarded", safe)]
    report("safeguarded Anderson converges the strong multi-slab toy",
           safe["converged"],
           f"safeguarded resid={safe['residual']:.2e} ({safe['n_iter']} it); "
           f"legacy resid={legacy['residual']:.2e} ({legacy['n_iter']} it)")
    # On the toys both schemes converge; the safeguarded one's advantage
    # shows on the strongly-anharmonic d5a SiNW (--with-d5).
    report("safeguarded and legacy Anderson both reach the SCF tolerance",
           safe["converged"] and legacy["converged"], "")


def check_solver_modes(report):
    """All four solver modes drive a medium toy to the SCF tolerance."""
    toy = diatomic_chain()
    dev = _build_toy_device(toy, n_slabs=3, freq_range=(0.01, 20.0, 60),
                            eta_factor=1.0, temperature=300.0, delta_T=20.0)
    scale = _calibrate_phi_scale(toy, dev, target_fraction=0.15)
    phi_dev = {(i, i, i): (toy.phi * scale).astype(complex)
               for i in range(dev["n_slabs"])}
    detail = []
    all_ok = True
    for mode in ("linear", "anderson", "jfnk", "anderson+jfnk"):
        run = _run_scba(dev, phi_dev, mixing=0.3, anderson=False, depth=8,
                        max_iter=80, solver=mode)
        all_ok = all_ok and run["converged"]
        detail.append(f"{mode}: {'ok' if run['converged'] else 'FAIL'} "
                      f"(resid {run['residual']:.1e})")
    report("all solver modes (linear/anderson/jfnk/anderson+jfnk) converge",
           all_ok, "; ".join(detail))


def check_zero_mode_projection(report):
    """Translation-projected Sigma^R is acoustic-sum-rule clean, and the
    loop still converges with the projection on.

    Uses the diatomic chain (acoustic + optical branches): the rigid
    translation is a small subspace, as for a real multi-atom device.
    The monatomic chain is degenerate here -- its single branch *is*
    largely the translation, so projecting it changes the system.
    """
    toy = diatomic_chain()
    dev = _build_toy_device(toy, n_slabs=4, freq_range=(0.01, 20.0, 60),
                            eta_factor=1.0, temperature=300.0, delta_T=20.0)
    scale = _calibrate_phi_scale(toy, dev, target_fraction=0.15)
    phi_dev = {(i, i, i): (toy.phi * scale).astype(complex)
               for i in range(dev["n_slabs"])}
    n_cart = dev["n_dof"] // len(toy.masses)
    Q = build_translation_projector(toy.masses, dev["n_slabs"], n_cart=n_cart)

    off = _run_scba(dev, phi_dev, mixing=0.3, anderson=True, depth=8,
                    max_iter=80, zero_mode_projection=False)
    on = _run_scba(dev, phi_dev, mixing=0.3, anderson=True, depth=8,
                   max_iter=80, zero_mode_projection=True, masses=toy.masses)
    leak_off = translation_leakage(off["result"]["Sigma_R"][0], Q)
    leak_on = translation_leakage(on["result"]["Sigma_R"][0], Q)
    report("zero-mode projection makes Sigma^R translation-clean",
           leak_on < 1e-8,
           f"translational leakage: projected={leak_on:.2e}, "
           f"unprojected={leak_off:.2e}")
    report("SCBA still converges with zero-mode projection on",
           on["converged"],
           f"converged in {on['n_iter']} iters, resid={on['residual']:.2e}")


def check_divergence_handling(report):
    """A run past the SCBA fixed-point radius is handled gracefully:
    the loop aborts (divergence guard) and returns a finite best
    iterate instead of the last, NaN-poisoned one."""
    toy = diatomic_chain()
    dev = _build_toy_device(toy, n_slabs=3, freq_range=(0.01, 20.0, 60),
                            eta_factor=1.0, temperature=300.0, delta_T=20.0)
    scale = _calibrate_phi_scale(toy, dev, target_fraction=3.0)
    phi_dev = {(i, i, i): (toy.phi * scale).astype(complex)
               for i in range(dev["n_slabs"])}
    run = _run_scba(dev, phi_dev, mixing=1.0, anderson=False, depth=8,
                    max_iter=120, solver="linear", divergence_guard=True)
    hist = np.asarray(run["result"]["convergence_history"], dtype=float)
    best_le_hist = (hist.size == 0
                    or run["residual"] <= float(np.nanmax(hist)) + 1e-9)
    finite_best = bool(np.isfinite(run["residual"]))
    report("divergence is handled gracefully (finite best iterate, "
           "not converged)",
           (not run["converged"]) and finite_best and best_le_hist,
           f"n_iter={run['n_iter']}/120, best resid={run['residual']:.2e}")


# ---------------------------------------------------------------------------
# Plotting + recommendation
# ---------------------------------------------------------------------------


def _make_plots(plotdata, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    for scheme, alpha, depth, run in plotdata.get("mixing_rows", []):
        ds = run["trace"]["dS"]
        if ds.size == 0:
            continue
        iters = np.arange(2, 2 + ds.size)
        if scheme == "linear":
            ax.semilogy(iters, ds, "o-", ms=3,
                        label=f"linear mix={alpha}")
        else:
            ax.semilogy(iters, ds, "s--", ms=3,
                        label=f"Anderson depth={depth}")
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel(r"SCF residual $\|G(\Sigma)-\Sigma\|/\|\Sigma\|$")
    ax.set_title("Convergence vs mixing scheme (medium anharmonicity)")
    ax.legend(fontsize=8)

    ax = axes[1]
    summ = plotdata.get("strength_summary", [])
    if summ:
        labels = [s[0] for s in summ]
        n_iters = [s[2][1] if s[2] else np.nan for s in summ]
        best_mix = [s[2][0] if s[2] else np.nan for s in summ]
        x = np.arange(len(labels))
        ax.bar(x, n_iters, color="steelblue")
        for xi, (ni, bm) in enumerate(zip(n_iters, best_mix)):
            if not np.isnan(ni):
                ax.text(xi, ni, f"mix={bm}", ha="center", va="bottom",
                        fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lab}\n(|S|/|H|={s[1]})"
                            for lab, s in zip(labels, summ)])
        ax.set_ylabel("iterations to converge (best mixing)")
        ax.set_title("Convergence vs anharmonic strength")

    fig.tight_layout()
    out_path = out_dir / "verify_scba_convergence.pdf"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _print_recommendation(plotdata):
    print("\n-- recommended SCBA settings --")
    rows = plotdata.get("mixing_rows", [])
    conv = [(r[0], r[1], r[2], r[3]["n_iter"]) for r in rows
            if r[3]["converged"]]
    if conv:
        best = min(conv, key=lambda r: r[3])
        print(f"    fastest scheme on the medium toy: {best[0]} "
              + (f"mix={best[1]}" if best[0] == "linear"
                 else f"(depth={best[2]})")
              + f"  -> {best[3]} iterations")
    print("    guidance (toys; the real d5a SiNW is verified separately):")
    print("      * solver='anderson' (safeguarded) is the robust default; "
          "'anderson+jfnk' adds a Newton-Krylov fallback for the hardest "
          "(linearly-unstable) fixed points.")
    print("      * zero_mode_projection=True strips the rigid-translation "
          "component of Sigma so the bubble cannot push acoustic modes to "
          "negative omega^2 -- the multi-slab d5a instability.")
    print("      * the loop stops on the SCF residual alone; heat-flow "
          "conservation is reported as a diagnostic (it is grid-limited, "
          "not an SCF residual -- gating on it iterates a converged "
          "self-energy and can trip the divergence guard).")
    print("      * eta_factor ~ 1 (eta ~ d_omega) resolves the propagator "
          "on the grid; recover eta -> 0 by extrapolation "
          "(phonon/scripts/extrapolate_eta.py).")
    print("      * scba_tol 1e-3 with max_scba_iter >= 80 is a safe "
          "envelope.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--with-d5", action="store_true",
        help="also run the real transmission_finite SCBA on the d5a "
             "SiNW (slow: ~3 min; needs the d5a fc3.hdf5)")
    args = parser.parse_args()

    results: list[tuple[str, bool, str]] = []

    def report(name, passed, detail):
        results.append((name, bool(passed), detail))
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {name}" + (f"  ({detail})" if detail else ""))

    print("=== verify_scba_convergence: SCBA fixed-point loop audit ===\n")

    store: dict = {}
    plotdata: dict = {}

    print("-- baseline convergence --")
    check_baseline_converges(report, store)
    print("\n-- mixing study --")
    check_mixing_study(report, store, plotdata)
    print("\n-- loop-design audit --")
    check_sigma_r_consistency(report, store)
    check_symmetrization_correction(report, store)
    print("\n-- anharmonic-strength dependence --")
    check_strength_dependence(report, plotdata)
    print("\n-- safeguarded solver (new mixing / JFNK / zero-mode) --")
    check_safeguarded_anderson(report, plotdata)
    check_solver_modes(report)
    check_zero_mode_projection(report)
    check_divergence_handling(report)
    if args.with_d5:
        print("\n-- d5a SiNW (real transmission_finite path) --")
        check_d5_convergence(report)

    out_dir = _REPO_ROOT / "phonon/scripts/out/verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = _make_plots(plotdata, out_dir)
    print(f"\n  diagnostic plot: {plot_path}")

    _print_recommendation(plotdata)

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
