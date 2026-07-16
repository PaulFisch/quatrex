"""Unit test for NewtonKrylovMixer: the synchronous exact-JVP Newton loop.

Simulates the SCBA driver contract (one map eval per mixer call) on
synthetic maps with an EXACT analytic JVP context standing in for
PhononJVP (same prepare()/apply() protocol). Checks:

  linear      g(x) = A x + b with unstable outliers: one exact Newton
              step after warmup lands x* = (I-A)^{-1} b.
  two-phase   contractive map + weak conjugating nonlinearity: damped
              Picard descends into the basin (switch on the relative
              residual), then Newton finishes with a quadratic tail to
              1e-12 in a handful of steps -- the production pattern.
  unstable    unstable outliers + weak nonlinearity (Picard diverges,
              d5a-like): Newton with trust region + backtracking lands
              the fixed point that no contraction scheme can reach.
  overshoot   strong cubic map where full Newton steps overshoot: the
              accept/reject test must recover (backtracking + trust
              shrink + Picard fallback from the base point) and still
              converge.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from quatrex.core.newton import NewtonKrylovMixer  # noqa: E402


class ToyJVP:
    """PhononJVP-protocol stand-in with the exact derivative of g."""

    def __init__(self, dg):
        self._dg = dg          # dg(x, v) -> exact R-linear derivative
        self._x = None

    def bind(self, get_x):
        self._get_x = get_x

    def prepare(self):
        self._x = self._get_x().copy()
        return 0.0

    def apply(self, v):
        return self._dg(self._x, v)


def drive(g, x0, mixer, jvp, tol=1e-11, max_calls=200):
    x = x0.copy()
    jvp.bind(lambda: x)
    hist = []
    for _ in range(max_calls):
        gx = g(x)
        r = np.linalg.norm(gx - x)
        hist.append(r)
        if r < tol:
            return x, hist, True
        x = mixer.step(x, gx)
    return x, hist, False


def test_linear():
    rng = np.random.default_rng(0)
    n = 60
    U, _ = np.linalg.qr(rng.standard_normal((n, n))
                        + 1j * rng.standard_normal((n, n)))
    lam = np.concatenate([[30.0, -18.0, 8 + 6j, 8 - 6j],
                          rng.uniform(-0.7, 0.7, n - 4)]).astype(complex)
    A = U @ np.diag(lam) @ U.conj().T
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    xstar = np.linalg.solve(np.eye(n) - A, b)

    def g(x):
        return A @ x + b

    jvp = ToyJVP(lambda x, v: A @ v)
    mixer = NewtonKrylovMixer(
        jvp_factory=lambda: jvp, warmup=2, switch_tol=1e9, beta=0.02,
        max_krylov=2 * n + 5, inner_tol=1e-12, forcing="fixed",
        trust=0.0, backtrack=0, verbose=False)
    x, hist, ok = drive(g, np.zeros(n, complex), mixer, jvp)
    err = np.linalg.norm(x - xstar) / np.linalg.norm(xstar)
    print(f"[linear]    calls={len(hist)} ||R||={hist[-1]:.2e} "
          f"x-x* rel={err:.2e} {'PASS' if ok and err < 1e-9 else 'FAIL'}")
    return ok and err < 1e-9


def test_two_phase():
    rng = np.random.default_rng(1)
    n = 40
    U, _ = np.linalg.qr(rng.standard_normal((n, n))
                        + 1j * rng.standard_normal((n, n)))
    lam = rng.uniform(-0.8, 0.8, n).astype(complex)
    A = U @ np.diag(lam) @ U.conj().T
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    c = 0.02

    def g(x):
        return A @ x + b + c * np.conj(x) ** 2

    def dg(x, v):
        return A @ v + 2 * c * np.conj(x) * np.conj(v)

    jvp = ToyJVP(dg)
    mixer = NewtonKrylovMixer(
        jvp_factory=lambda: jvp, warmup=3, switch_tol=0.3, beta=0.5,
        max_krylov=2 * n + 5, inner_tol=0.1, forcing="ew",
        trust=0.5, trust_max=1.0, backtrack=3, verbose=False)
    x, hist, ok = drive(g, np.zeros(n, complex), mixer, jvp, tol=1e-12)
    res = np.linalg.norm(g(x) - x)
    print(f"[two-phase] calls={len(hist)} ||R||={hist[-1]:.2e} "
          f"tail={['%.1e' % h for h in hist[-4:]]} "
          f"{'PASS' if ok and res < 1e-11 else 'FAIL'}")
    return ok and res < 1e-11


def test_unstable():
    rng = np.random.default_rng(3)
    n = 40
    U, _ = np.linalg.qr(rng.standard_normal((n, n))
                        + 1j * rng.standard_normal((n, n)))
    lam = np.concatenate([[4.0, -3.0], rng.uniform(-0.6, 0.6, n - 2)]
                         ).astype(complex)
    A = U @ np.diag(lam) @ U.conj().T
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    c = 0.01

    def g(x):
        return A @ x + b + c * np.conj(x) ** 2

    def dg(x, v):
        return A @ v + 2 * c * np.conj(x) * np.conj(v)

    jvp = ToyJVP(dg)
    mixer = NewtonKrylovMixer(
        jvp_factory=lambda: jvp, warmup=2, switch_tol=1e9, beta=0.05,
        max_krylov=2 * n + 5, inner_tol=0.05, forcing="ew",
        trust=0.5, trust_max=2.0, backtrack=4, verbose=False)
    x, hist, ok = drive(g, np.zeros(n, complex), mixer, jvp,
                        tol=1e-11, max_calls=300)
    res = np.linalg.norm(g(x) - x)
    print(f"[unstable]  calls={len(hist)} ||R||={hist[-1]:.2e} "
          f"{'PASS' if ok and res < 1e-10 else 'FAIL'}")
    return ok and res < 1e-10


def test_overshoot():
    # The canonical Newton-overshoot family: R(x) = tanh(x - t) per
    # component (monotone -- unique root x* = t, NO false ||R|| minima),
    # but a full Newton step from |x - t| > ~1.1 overshoots by cosh^2.
    # The SATURATING residual makes ||R||-backtracking alone blind (an
    # escaped iterate can show a smaller ||R||), so this exercises the
    # trust cap + halvings + Picard fallback together -- the production
    # configuration.
    rng = np.random.default_rng(4)
    n = 16
    t = 3.0 * rng.standard_normal(n).astype(complex)

    def g(x):
        return x - np.tanh(x - t)

    def dg(x, v):
        return v - v / np.cosh(x - t) ** 2

    jvp = ToyJVP(dg)
    mixer = NewtonKrylovMixer(
        jvp_factory=lambda: jvp, warmup=2, switch_tol=1e9, beta=0.5,
        max_krylov=40, inner_tol=1e-8, forcing="fixed",
        trust=0.3, trust_max=1.0, backtrack=8, verbose=False)
    x, hist, ok = drive(g, np.zeros(n, complex), mixer, jvp,
                        tol=1e-11, max_calls=400)
    err = np.linalg.norm(x - t)
    print(f"[overshoot] calls={len(hist)} ||R||={hist[-1]:.2e} "
          f"x-x*={err:.2e} {'PASS' if ok and err < 1e-9 else 'FAIL'}")
    return ok and err < 1e-9


def test_pathological_bounded():
    # Cubic map with genuine ||R|| local minima (per-component turning
    # points): line-search Newton CANNOT converge here by construction.
    # The mixer must degrade gracefully -- finite accepted iterates, no
    # blow-up of the base sequence, some initial progress -- rather than
    # diverge or crash. (Production analogue: a fixed point that has
    # ceased to exist; the diagnosis lives in the GMRES/backtrack logs.)
    rng = np.random.default_rng(2)
    n = 12
    A = np.diag(rng.uniform(-0.5, 0.5, n)).astype(complex)
    b = rng.standard_normal(n) + 0j
    c = 2.0

    def g(x):
        return A @ x + b + c * x**3

    def dg(x, v):
        return A @ v + 3 * c * x**2 * v

    jvp = ToyJVP(dg)
    mixer = NewtonKrylovMixer(
        jvp_factory=lambda: jvp, warmup=2, switch_tol=1e9, beta=0.05,
        max_krylov=30, inner_tol=1e-8, forcing="fixed",
        trust=0.3, trust_max=0.5, backtrack=6, verbose=False)
    x, hist, ok = drive(g, np.zeros(n, complex), mixer, jvp,
                        tol=1e-11, max_calls=200)
    h = np.asarray(hist)
    finite = h[np.isfinite(h)]
    progressed = finite.min() < 0.8 * h[0]
    # Trials that overflow the map show up as non-finite history entries;
    # the mixer must keep retrying from finite base points (most calls
    # finite) without crashing.
    graceful = finite.size > 0.5 * h.size
    print(f"[pathological] calls={len(hist)} min||R||={finite.min():.2e} "
          f"(start {h[0]:.2e}) "
          f"{'PASS' if progressed and graceful else 'FAIL'}")
    return progressed and graceful


if __name__ == "__main__":
    results = [test_linear(), test_two_phase(), test_unstable(),
               test_overshoot(), test_pathological_bounded()]
    print("OVERALL:", "PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)
