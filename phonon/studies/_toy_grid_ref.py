"""Fine-grid reference points for the E1 ladder (grid-converged Gamma_em)."""
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
    omega_a, omega_flat = 8.0, 10.0
    fmax = 2.2 * 2 * omega_a
    rows = []
    # same calibrated couplings as the main study
    g0 = 1e19
    h00, h01, phi = T.flatband_chain(omega_a, omega_flat, g0)
    gam0 = T.first_born_gamma(h00, h01, phi, fmax, omega_flat=omega_flat)
    G_LIST = tuple(float(g0 * np.sqrt(t / gam0)) for t in (0.02, 0.2, 2.0))
    cases = [(G_LIST[0], 960), (G_LIST[0], 1920), (G_LIST[0], 3840),
             (G_LIST[1], 960), (G_LIST[1], 1920),
             (G_LIST[2], 960)]
    for g, nf in cases:
        h00, h01, phi = T.flatband_chain(omega_a, omega_flat, g)
        res = T.run_case(h00, h01, phi, nf, fmax)
        st = T.rate_stats(res["convergence_history"], 0.2)
        gem = T.emergent_gamma(res, omega_flat, 1, 2)
        rows.append(dict(exp="E1ref", g=g, nfreq=nf, dw=res["dw"],
                         gamma_em=gem, J=T.heat_current(res),
                         converged=bool(res["converged"]),
                         n_it=len(res["convergence_history"]), **st))
        print(f"  g={g:.2e} nf={nf}: G_em={gem:.4f} "
              f"conv={res['converged']} n_it={len(res['convergence_history'])} "
              f"best={st['best']:.1e} rate={st['rate']:.3f} "
              f"jit={st['jitter']:.2f}", flush=True)
    (out / "results_ref.json").write_text(json.dumps(
        dict(rows=rows), indent=1,
        default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"saved {out}/results_ref.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
