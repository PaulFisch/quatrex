"""E6 v3: contact-broadening continuation of the sharp-sharp branch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _toy_grid_study as T  # noqa: E402


def main() -> int:
    out = Path(__file__).resolve().parent / "out/toy_grid"
    out.mkdir(parents=True, exist_ok=True)
    omega_a = 8.0
    fmax = 2.2 * 2 * omega_a
    rows = []

    g0p = 1e19
    h00, h01, phi = T.sharp_pair_chain(omega_a, 5.0, 10.0, g0p,
                                       eps_flat=0.02)
    gam0p = T.first_born_gamma(h00, h01, phi, fmax, omega_flat=10.0, idx=2,
                               nfreq_pos=8000, eta_w=5e-3)
    t6 = 0.2
    gp6 = float(g0p * np.sqrt(t6 / gam0p))
    h00, h01, phi = T.sharp_pair_chain(omega_a, 5.0, 10.0, gp6,
                                       eps_flat=0.02)
    print(f"calibration: g={gp6:.3e} -> Gamma_FB target {t6}", flush=True)

    def report(tag, res, nf, eta):
        gem2 = T.emergent_gamma(res, 10.0, 2, 3)
        gem1 = T.emergent_gamma(res, 5.0, 1, 3)
        st = T.rate_stats(res["convergence_history"], 0.2)
        rows.append(dict(exp="E6v2", tag=tag, nfreq=nf, eta_w=eta,
                         gamma_t=t6, gamma_em_B2=gem2, gamma_em_B1=gem1,
                         J=T.heat_current(res),
                         converged=bool(res["converged"]),
                         n_it=len(res["convergence_history"]), **st))
        print(f"  {tag:22s} nf={nf:4d} eta={eta:g}: "
              f"G_em(B2)={gem2:.4g} G_em(B1)={gem1:.4g} "
              f"conv={res['converged']} "
              f"n_it={len(res['convergence_history'])} "
              f"best={st['best']:.1e} rate={st['rate']:.3f} "
              f"jit={st['jitter']:.2f}", flush=True)
        return res

    nf_ref = 480
    print("== stage 1+2: eps_flat continuation chain (nf=480) ==",
          flush=True)
    cont = None
    for eps in (1.0, 0.3, 0.1, 0.02):
        h00e, h01e, phie = T.sharp_pair_chain(omega_a, 5.0, 10.0, gp6,
                                              eps_flat=eps)
        si = (None if cont is None
              else (cont["Sigma_l"], cont["Sigma_g"]))
        cont = report(f"eps={eps}", T.run_case(
            h00e, h01e, phie, nf_ref, fmax, eta_w=1e-6, sigma_init=si),
            nf_ref, 1e-6)

    print("== stage 3: cold control at the true system ==", flush=True)
    report("cold-control", T.run_case(h00, h01, phi, nf_ref, fmax,
                                      eta_w=1e-6), nf_ref, 1e-6)

    print("== stage 4: warm ladder to coarser grids ==", flush=True)
    seed = cont
    for nf in (240, 120, 60, 30):
        # freqs identical layout, just coarser: interpolate the seed Sigma
        probe = T.run_case(h00, h01, phi, nf, fmax, eta_w=1e-6, max_iter=1)
        fr_c = np.asarray(probe["freqs"])
        si = (T.interp_sigma(seed["Sigma_l"], seed["freqs"], fr_c),
              T.interp_sigma(seed["Sigma_g"], seed["freqs"], fr_c))
        report("warm-ladder", T.run_case(h00, h01, phi, nf, fmax,
                                         eta_w=1e-6, sigma_init=si), nf,
               1e-6)

    (out / "results_e6.json").write_text(json.dumps(
        dict(rows=rows), indent=1,
        default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"saved {out}/results_e6.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
