"""Compare two engine run.npz snapshots on the physics observables.

Compare two engine run.npz snapshots on the physics observables.
Usage: python parity_check.py ref.npz other.npz [--rtol 1e-8] [--atol 0]
"""
import argparse
import sys

import numpy as np

KEYS = (
    "energies",
    "final_heat",
    "last_heat",
    "lead_current",
    "iter_heat",
    "iter_sigma_max",
    "iter_bubble_balance",
    "final_bubble_balance",
    "slab_absorption",
    "gr_diag_imag",
    "gl_diag_imag",
    "bubble_balance_spectrum",
    "current_spectrum",
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("other")
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--atol", type=float, default=0.0)
    a = p.parse_args()

    ref = np.load(a.ref, allow_pickle=True)
    oth = np.load(a.other, allow_pickle=True)

    n_iter_ref = int(ref["n_iter"]) if "n_iter" in ref else -1
    n_iter_oth = int(oth["n_iter"]) if "n_iter" in oth else -1
    if n_iter_ref != n_iter_oth:
        print(f"NOTE: n_iter differs ({n_iter_ref} vs {n_iter_oth}); "
              "per-iteration keys are compared on the common prefix.")

    ok = True
    for key in KEYS:
        if key not in ref and key not in oth:
            continue
        if (key in ref) != (key in oth):
            print(f"{key:26s} MISSING in {'other' if key in ref else 'ref'}")
            ok = False
            continue
        x = np.asarray(ref[key])
        y = np.asarray(oth[key])
        if key.startswith("iter_") and x.shape != y.shape:
            n = min(x.shape[0], y.shape[0])
            x, y = x[:n], y[:n]
        if x.shape != y.shape:
            print(f"{key:26s} SHAPE {x.shape} vs {y.shape}")
            ok = False
            continue
        if x.size == 0:
            continue
        note = ""
        if key in ("gr_diag_imag", "gl_diag_imag") and "energies" in ref:
            # At eta=0 the omega=0 bin of G^R is the inverse of a singular
            # matrix -- backend-dependent and physically masked. Exclude.
            en = np.asarray(ref["energies"]).ravel()
            if en.size == x.shape[0] and abs(en[0]) < 1e-12:
                x, y = x[1:], y[1:]
                note = " (dc bin excluded)"
        # Scale-normalized gate: max|x-y| <= rtol * max|x| + atol. A
        # per-element rtol would fail on physically-zero entries whose
        # absolute deviation is roundoff of the array scale.
        denom = np.max(np.abs(x))
        err = float(np.max(np.abs(x - y)))
        rel = err / denom if denom > 0 else err
        close = err <= a.rtol * denom + a.atol
        print(f"{key:26s} {'ok  ' if close else 'FAIL'} "
              f"max_rel={rel:.3e}{note}")
        ok = ok and close
    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
