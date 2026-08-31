"""Unit test for JFNKMixer: land an iteration-UNSTABLE fixed point.

Simulates the SCBA driver contract (one map eval per mixer call) on synthetic
maps whose Jacobian has a few |lambda|>1 outliers + a contractive bulk -- the
d5a eta=0 spectrum in miniature -- where Picard/Anderson/RRE diverge. Checks
JFNK drives the residual to ~1e-12 and lands the exact fixed point x* of g.

  Linear:    g(x) = A x + b,  x* = (I-A)^{-1} b   (GMRES exact in <= dim steps)
  Nonlinear: g(x) = A x + b + c*(x.real**2 ...)   (real-linear conjugating part,
             tests the real-embedding J*v and the Newton outer loop)
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from quatrex.experimental.mixers.jfnk import JFNKMixer


def drive(g, x0, mixer, tol=1e-11, max_calls=400):
    """Replay the SCBA loop: feed x, get gx=g(x), call mixer.step -> next x."""
    x = x0.copy()
    hist = []
    for k in range(max_calls):
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
    # Eigenvalues: 4 unstable outliers (|lam|=5..30, incl a complex pair) + bulk<1.
    U, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    lam = np.concatenate([[30.0, -18.0, 8 + 6j, 8 - 6j],
                          rng.uniform(-0.7, 0.7, n - 4)]).astype(complex)
    A = U @ np.diag(lam) @ U.conj().T
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    xstar = np.linalg.solve(np.eye(n) - A, b)

    def g(x):
        return A @ x + b

    mixer = JFNKMixer(warmup=3, beta=0.05, max_krylov=40, inner_tol=1e-3,
                      forcing="fixed", eps=1e-7, trust=0.0, newton_damp=1.0,
                      verbose=False)
    x, hist, ok = drive(g, np.zeros(n, complex), mixer)
    err = np.linalg.norm(x - xstar) / np.linalg.norm(xstar)
    rho = np.max(np.abs(lam))
    print(f"[linear]  spectral radius |lam|max={rho:.1f}  converged={ok}  "
          f"calls={len(hist)}  final_resid={hist[-1]:.2e}  rel_err_vs_xstar={err:.2e}")
    assert ok and err < 1e-8, "linear JFNK failed to land the unstable fixed point"


def test_nonlinear():
    rng = np.random.default_rng(1)
    n = 50
    U, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    lam = np.concatenate([[12.0, -7.0, 5 + 4j, 5 - 4j],
                          rng.uniform(-0.6, 0.6, n - 4)]).astype(complex)
    A = U @ np.diag(lam) @ U.conj().T
    b = 0.3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    kappa = 0.02

    def g(x):                       # real-linear conjugating nonlinearity
        return A @ x + b + kappa * (x.real ** 2 - 1j * x.imag * np.abs(x))

    # reference fixed point by a stabilised solve (Newton in numpy, dense)
    def R(x):
        return g(x) - x
    xs = np.zeros(n, complex)
    for _ in range(50):             # dense FD-Newton to get ground truth
        r0 = R(xs)
        if np.linalg.norm(r0) < 1e-13:
            break
        J = np.zeros((2 * n, 2 * n))
        e = 1e-7
        x2 = np.concatenate([xs.real, xs.imag])
        for i in range(2 * n):
            xp = x2.copy(); xp[i] += e
            xc = xp[:n] + 1j * xp[n:]
            dr = (R(xc) - r0) / e
            J[:, i] = np.concatenate([dr.real, dr.imag])
        d = np.linalg.solve(J, -np.concatenate([r0.real, r0.imag]))
        xs = xs + (d[:n] + 1j * d[n:])

    mixer = JFNKMixer(warmup=5, beta=0.03, max_krylov=40, inner_tol=1e-2,
                      forcing="ew", eps=1e-7, trust=0.5, newton_damp=1.0,
                      verbose=False)
    x, hist, ok = drive(g, np.zeros(n, complex), mixer)
    err = np.linalg.norm(x - xs) / max(np.linalg.norm(xs), 1e-30)
    print(f"[nonlin]  |lam|max={np.max(np.abs(lam)):.1f}  converged={ok}  "
          f"calls={len(hist)}  final_resid={hist[-1]:.2e}  rel_err_vs_newton={err:.2e}")
    assert ok and err < 1e-6, "nonlinear JFNK failed"


def test_picard_diverges():
    """Control: plain Picard on the same unstable map MUST diverge."""
    rng = np.random.default_rng(0)
    n = 60
    U, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    lam = np.concatenate([[30.0, -18.0, 8 + 6j, 8 - 6j],
                          rng.uniform(-0.7, 0.7, n - 4)]).astype(complex)
    A = U @ np.diag(lam) @ U.conj().T
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x = np.zeros(n, complex)
    for _ in range(60):
        x = 0.1 * (A @ x + b) + 0.9 * x
    print(f"[picard]  damped Picard final ||x||={np.linalg.norm(x):.2e} (diverges)")
    assert np.linalg.norm(x) > 1e3


def test_marginal_bounded():
    """The d5a stall in miniature: J = J_F - I is INDEFINITE and near-singular
    -- a few |lambda(J_F)|>>1 unstable AND a cluster lambda(J_F)~1 MARGINAL (J
    eigenvalues ~0), plus contractive modes (J eigenvalues in [-2,0)). On such a
    map the inner GMRES stalls on the near-null-space and pure Newton descends
    only slowly -- but the TRUST REGION must keep the iterate BOUNDED (no blow-up;
    this is the property RPM lost). NB: a positive LM/PTC shift is the WRONG tool
    here -- it pushes a negative eigenvalue of the indefinite J through zero and
    diverges -- so ptc stays 0 and the trust region is the globalisation."""
    rng = np.random.default_rng(2)
    n = 80
    U, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    marg = 1.0 + 1e-3 * rng.standard_normal(12)        # near-marginal cluster
    lam = np.concatenate([[25.0, -15.0, 9 + 5j, 9 - 5j], marg,
                          rng.uniform(-0.6, 0.6, n - 16)]).astype(complex)
    A = U @ np.diag(lam) @ U.conj().T
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def g(x):
        return A @ x + b

    m = JFNKMixer(warmup=3, beta=0.02, max_krylov=40, inner_tol=1e-2,
                  forcing="ew", eps=1e-7, trust=0.3, newton_damp=1.0,
                  max_newton=60, ptc=0.0, verbose=False)
    x, h, ok = drive(g, np.zeros(n, complex), m, tol=1e-9, max_calls=600)
    cond = np.max(np.abs(lam - 1)) / np.min(np.abs(lam - 1))
    print(f"[marginal] cond(J)~{cond:.0f}  trust-region JFNK: bounded={np.isfinite(h[-1])} "
          f"final_resid={h[-1]:.2e}  min_resid={min(h):.2e}  (descends, may not reach 1e-9)")
    assert np.isfinite(h[-1]) and h[-1] < h[0], \
        "trust-region JFNK must stay bounded and descend on the near-singular map"


if __name__ == "__main__":
    test_picard_diverges()
    test_linear()
    test_nonlinear()
    test_marginal_bounded()
    print("ALL JFNK UNIT TESTS PASSED")
