"""Reduced reference study for a hybrid grid--rational phonon bubble.

This is deliberately not a production pole sector.  It isolates the numerical
question that the production experiments could not answer cleanly: can one
coarse, smooth cell basis and a coherent passive pole basis be passed through
one bilinear bubble without sampling a narrow input or output line?

The representation used here is

    A(w) = U (w I - K)^-1 Q (w I - K)^-H U^H,

with all eigenvalues of ``K`` in the lower half plane and ``Q >= 0``.  The
cluster therefore remains positive by congruence.  A scalar affine source
factor is allowed; it models the 2 %/THz source-variation case in the pole
campaign without destroying the exact partial-fraction representation.

Mixed cluster--cell terms are product integrated.  The zeroth and first cell
moments of the rational cluster depend only on ``output_index-input_index``,
so their application is Toeplitz and is evaluated by FFT.  Cluster--cluster
terms are closed under convolution: output poles are pairwise sums of poles in
the same half plane.  The returned :class:`RationalBubble` is the object a
future Dyson solve would carry as an auxiliary state instead of sampling it on
the coarse grid.

Run the reduced campaign used by ``phonon/docs/phph_acceleration_review.md``::

    QTX_ARRAY_MODULE=numpy python phonon/studies/_hybrid_frequency_review.py

Use ``--full`` for every pre-registered separation and source-slope case.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.signal import fftconvolve


TWO_PI = 2.0 * np.pi


def _hermitian(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


@dataclass(frozen=True)
class RationalSpectrum:
    """Matrix-valued sum ``sum_p residues[p] / (w - poles[p])``."""

    poles: np.ndarray
    residues: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.poles, complex)
        r = np.asarray(self.residues, complex)
        if p.ndim != 1 or r.ndim != 3 or r.shape[0] != p.size:
            raise ValueError("RationalSpectrum expects poles (r,) and residues (r,d,d)")
        if r.shape[1] != r.shape[2]:
            raise ValueError("RationalSpectrum residues must be square")
        object.__setattr__(self, "poles", p)
        object.__setattr__(self, "residues", r)

    @property
    def n_dof(self) -> int:
        return int(self.residues.shape[1])

    def eval(self, omega: np.ndarray | float) -> np.ndarray:
        w = np.asarray(omega, complex)
        flat = w.reshape(-1)
        out = np.einsum(
            "wp,pij->wij", 1.0 / (flat[:, None] - self.poles[None, :]),
            self.residues, optimize=True)
        return out.reshape(w.shape + (self.n_dof, self.n_dof))

    def cell_moments(self, centres: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``int A(c+t) dt`` and ``int t A(c+t) dt`` per cell."""
        c = np.asarray(centres, float)[:, None]
        alpha = self.poles[None, :] - c
        lo = c - 0.5 * float(h) - self.poles[None, :]
        hi = c + 0.5 * float(h) - self.poles[None, :]
        log_moment = np.log(hi) - np.log(lo)
        first_moment = alpha * log_moment + float(h)
        m0 = np.einsum("cp,pij->cij", log_moment, self.residues, optimize=True)
        m1 = np.einsum("cp,pij->cij", first_moment, self.residues, optimize=True)
        return m0, m1


@dataclass(frozen=True)
class PassiveCluster:
    """Coherent lower-half-plane cluster with a PSD Keldysh source."""

    z: np.ndarray
    u: np.ndarray
    q: np.ndarray
    source_centre: float = 0.0
    source_slope: float = 0.0

    def __post_init__(self) -> None:
        z = np.asarray(self.z, complex)
        u = np.asarray(self.u, complex)
        q = _hermitian(np.asarray(self.q, complex))
        if z.ndim != 1 or u.ndim != 2 or u.shape[1] != z.size:
            raise ValueError("PassiveCluster expects z (r,) and u (d,r)")
        if q.shape != (z.size, z.size):
            raise ValueError("PassiveCluster q must be (r,r)")
        if np.any(np.imag(z) >= 0.0):
            raise ValueError("PassiveCluster poles must lie strictly below the real axis")
        scale = max(float(np.linalg.norm(q, 2)), 1.0)
        if np.linalg.eigvalsh(q).min() < -1e-12 * scale:
            raise ValueError("PassiveCluster q must be positive semidefinite")
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "q", q)

    @property
    def n_dof(self) -> int:
        return int(self.u.shape[0])

    @property
    def n_poles(self) -> int:
        return int(self.z.size)

    def source_factor(self, omega: np.ndarray | float) -> np.ndarray:
        return 1.0 + self.source_slope * (np.asarray(omega) - self.source_centre)

    def spectrum(self, omega: np.ndarray | float) -> np.ndarray:
        """Hermitian spectral/Keldysh weight; PSD where the source factor is nonnegative."""
        w = np.asarray(omega, complex)
        flat = w.reshape(-1)
        d = 1.0 / (flat[:, None] - self.z[None, :])
        modal = np.einsum("wa,ab,wb->wab", d, self.q, d.conj(), optimize=True)
        out = np.einsum("ia,wab,jb->wij", self.u, modal, self.u.conj(), optimize=True)
        out *= self.source_factor(flat)[:, None, None]
        return out.reshape(w.shape + (self.n_dof, self.n_dof))

    def keldysh(self, omega: np.ndarray | float) -> np.ndarray:
        """Anti-Hermitian wrapper used only for the structure diagnostics."""
        # Quatrex uses G^< = -i n A, so +i G^< is the positive carrier.
        return -1j * self.spectrum(omega)

    def partial_fractions(self) -> RationalSpectrum:
        """Exact poles/residues, including an affine scalar source factor.

        Multiplication by ``1+s(w-c)`` only evaluates that factor at each pole:
        the apparent constant terms cancel because the congruence decays as
        ``1/w^2`` and hence the sum of its residues is zero.
        """
        r, d = self.n_poles, self.n_dof
        lower = np.zeros((r, d, d), complex)
        upper = np.zeros_like(lower)
        for a in range(r):
            for b in range(r):
                base = self.q[a, b] * np.outer(self.u[:, a], self.u[:, b].conj())
                gap = self.z[a] - self.z[b].conjugate()
                lower[a] += base / gap
                upper[b] -= base / gap
        lower *= self.source_factor(self.z)[:, None, None]
        upper *= self.source_factor(self.z.conjugate())[:, None, None]
        return RationalSpectrum(np.concatenate((self.z, self.z.conjugate())),
                                np.concatenate((lower, upper), axis=0))


@dataclass(frozen=True)
class RationalBubble(RationalSpectrum):
    """Rational output of a cluster--cluster bubble."""

    @property
    def auxiliary_dimension(self) -> int:
        return int(sum(np.linalg.matrix_rank(a) for a in self.residues))


def convolve_rational(a: RationalSpectrum, b: RationalSpectrum) -> RationalBubble:
    r"""Infinite-line ``int A(u) B(Omega-u) du/(2 pi)`` by residues."""
    if a.n_dof != b.n_dof:
        raise ValueError("rational convolution requires matching matrix dimensions")
    poles: list[complex] = []
    residues: list[np.ndarray] = []
    for p, cp in zip(a.poles, a.residues):
        for q, cq in zip(b.poles, b.residues):
            if np.imag(p) < 0.0 and np.imag(q) < 0.0:
                poles.append(p + q)
                residues.append(-1j * (cp @ cq))
            elif np.imag(p) > 0.0 and np.imag(q) > 0.0:
                poles.append(p + q)
                residues.append(+1j * (cp @ cq))
    if not poles:
        d = a.n_dof
        return RationalBubble(np.zeros(0, complex), np.zeros((0, d, d), complex))
    return RationalBubble(np.asarray(poles), np.asarray(residues))


def cluster_bubble(a: PassiveCluster, b: PassiveCluster | None = None) -> RationalBubble:
    return convolve_rational(a.partial_fractions(),
                             (a if b is None else b).partial_fractions())


def _fft_lag_left(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``out[m] = sum_j weights[m-j] @ values[j]`` by scalar FFTs."""
    n, d = values.shape[0], values.shape[1]
    if weights.shape != (2 * n - 1, d, d):
        raise ValueError("lag weights must have shape (2*n-1,d,d)")
    out = np.zeros_like(values, dtype=complex)
    take = slice(n - 1, 2 * n - 1)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                out[:, i, j] += fftconvolve(
                    weights[:, i, k], values[:, k, j], mode="full")[take]
    return out


def _fft_lag_right(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``out[m] = sum_j values[j] @ weights[m-j]`` by scalar FFTs."""
    n, d = values.shape[0], values.shape[1]
    if weights.shape != (2 * n - 1, d, d):
        raise ValueError("lag weights must have shape (2*n-1,d,d)")
    out = np.zeros_like(values, dtype=complex)
    take = slice(n - 1, 2 * n - 1)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                out[:, i, j] += fftconvolve(
                    weights[:, k, j], values[:, i, k], mode="full")[take]
    return out


def mixed_product_integration(
    cluster: PassiveCluster,
    grid: np.ndarray,
    smooth: np.ndarray,
    smooth_derivative: np.ndarray,
) -> np.ndarray:
    r"""Both mixed sectors on ``grid`` using linear cell product integration.

    The background is reconstructed as ``B_j + B'_j t`` in each cell.  The
    returned value is ``int [A(u)B(w-u)+B(u)A(w-u)] du/(2 pi)``.
    """
    grid = np.asarray(grid, float)
    smooth = np.asarray(smooth, complex)
    deriv = np.asarray(smooth_derivative, complex)
    n = grid.size
    if n % 2 != 1 or not np.allclose(np.diff(grid), np.diff(grid)[0]):
        raise ValueError("mixed_product_integration needs an odd uniform grid")
    if not np.isclose(grid[n // 2], 0.0):
        raise ValueError("mixed_product_integration grid must contain zero at its centre")
    if smooth.shape != deriv.shape or smooth.shape[0] != n:
        raise ValueError("smooth values and derivatives must share the grid axis")
    h = float(grid[1] - grid[0])
    lags = np.arange(-(n - 1), n)
    m0, m1 = cluster.partial_fractions().cell_moments(lags * h, h)
    return (_fft_lag_left(m0, smooth) - _fft_lag_left(m1, deriv)
            + _fft_lag_right(smooth, m0) - _fft_lag_right(deriv, m1)) / TWO_PI


def cell_cell_rectangle(grid: np.ndarray, smooth: np.ndarray) -> np.ndarray:
    """The current FFT/rectangle cell--cell sector, cropped to ``grid``."""
    grid = np.asarray(grid, float)
    smooth = np.asarray(smooth, complex)
    n, d = smooth.shape[:2]
    mid = n // 2
    lag = np.zeros((2 * n - 1, d, d), complex)
    lag[n - 1 - mid:n - 1 + (n - mid)] = smooth
    return float(grid[1] - grid[0]) * _fft_lag_right(smooth, lag) / TWO_PI


def hybrid_bubble(
    cluster: PassiveCluster,
    grid: np.ndarray,
    smooth: np.ndarray,
    smooth_derivative: np.ndarray,
) -> tuple[np.ndarray, RationalBubble]:
    """Grid background plus the unsampled rational output component."""
    rational = cluster_bubble(cluster)
    grid_part = (cell_cell_rectangle(grid, smooth)
                 + mixed_product_integration(cluster, grid, smooth,
                                             smooth_derivative))
    return grid_part, rational


def _matrix_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _smooth_model(d: int = 2):
    base = np.array([[1.0, 0.18 + 0.07j], [0.18 - 0.07j, 0.72]], complex)[:d, :d]

    def scalar(w):
        w = np.asarray(w)
        return np.exp(-0.5 * (w / 4.5) ** 2) * (1.0 + 0.01 * w)

    def scalar_prime(w):
        w = np.asarray(w)
        f = np.exp(-0.5 * (w / 4.5) ** 2)
        return f * (0.01 - (w / 4.5**2) * (1.0 + 0.01 * w))

    return (lambda w: scalar(w)[..., None, None] * base,
            lambda w: scalar_prime(w)[..., None, None] * base)


def proxy_cluster(gamma_over_h: float, offset: float, separation: float,
                  slope: float, h: float = 0.25) -> PassiveCluster:
    """Two-mode coherent proxy matching the recorded narrow overlapping Si tail."""
    width = float(gamma_over_h) * h
    centre = 1.0 + float(offset) * h
    gap = float(separation) * (2.0 * width)
    z = np.array([centre - 0.5 * gap - 1j * width,
                  centre + 0.5 * gap - 1.15j * width])
    u = np.array([[1.0, 0.38 + 0.12j], [0.24 - 0.08j, 0.91]], complex)
    l = np.array([[1.0, 0.28j], [0.22, 0.73]], complex)
    q = l @ l.conj().T
    return PassiveCluster(z, u, q, source_centre=centre, source_slope=slope)


def adaptive_reference(cluster: PassiveCluster, smooth_fn, omega: float,
                       limit: float = 18.0) -> np.ndarray:
    """Dense oracle for the mixed sectors; rational--rational stays analytic."""
    d = cluster.n_dof

    def integrand(u):
        a_u = cluster.spectrum(u)
        a_v = cluster.spectrum(omega - u)
        b_u = smooth_fn(u)
        b_v = smooth_fn(omega - u)
        return ((a_u @ b_v + b_u @ a_v) / TWO_PI).reshape(d * d)

    points = [float(x.real) for x in cluster.z]
    points += [float(omega - x.real) for x in cluster.z]
    points = sorted(x for x in points if -limit < x < limit)
    got, _ = quad_vec(integrand, -limit, limit, epsabs=2e-11, epsrel=2e-11,
                      points=points, limit=800)
    return np.asarray(got).reshape(d, d)


def run_sweep(full: bool = False) -> dict:
    h = 0.25
    grid = np.arange(-72, 73) * h
    smooth_fn, deriv_fn = _smooth_model()
    smooth, deriv = smooth_fn(grid), deriv_fn(grid)
    gammas = (1.0, 0.2, 0.04, 0.008, 0.001)
    offsets = (0.0, 0.25, 0.49)
    separations = (0.5, 1.0, 2.0, 5.0) if full else (0.5, 2.0)
    slopes = (0.0, 0.02) if full else (0.0, 0.02)
    rows = []
    for gamma in gammas:
        for offset in offsets:
            for separation in separations:
                for slope in slopes:
                    cluster = proxy_cluster(gamma, offset, separation, slope, h)
                    mixed = mixed_product_integration(cluster, grid, smooth, deriv)
                    # Test the two output regions where the narrow cluster matters:
                    # one mixed peak and the cluster--cluster combination peak.
                    targets = (float(np.mean(cluster.z.real)),
                               float(2.0 * np.mean(cluster.z.real)))
                    errs = []
                    for target in targets:
                        iw = int(np.argmin(np.abs(grid - target)))
                        ref = adaptive_reference(cluster, smooth_fn, float(grid[iw]))
                        errs.append(_matrix_rel(mixed[iw], ref))
                    w_probe = np.linspace(-4.0, 4.0, 33)
                    spec = cluster.spectrum(w_probe)
                    min_eig = min(float(np.linalg.eigvalsh(_hermitian(a)).min())
                                  for a in spec)
                    kel = cluster.keldysh(w_probe)
                    anti = float(np.linalg.norm(kel + kel.conj().transpose(0, 2, 1))
                                 / max(np.linalg.norm(kel), 1e-300))
                    rb = cluster_bubble(cluster)
                    rows.append({
                        "gamma_over_h": gamma, "offset_over_h": offset,
                        "separation_over_width": separation,
                        "source_slope_per_thz": slope,
                        "mixed_rel_error": max(errs),
                        "min_spectral_eigenvalue": min_eig,
                        "keldysh_antihermiticity": anti,
                        "input_poles": cluster.n_poles,
                        "output_terms": int(rb.poles.size),
                        "auxiliary_dimension": rb.auxiliary_dimension,
                    })
    by_gamma = {}
    for gamma in gammas:
        vals = [r["mixed_rel_error"] for r in rows if r["gamma_over_h"] == gamma]
        by_gamma[str(gamma)] = {"max": max(vals), "median": float(np.median(vals))}
    runtime_by_gamma = {}
    # Time only the proposed algorithm, never the adaptive oracle.  Repeating
    # the same fixed-size application makes the width-independence measurable
    # without letting sub-millisecond timer noise dominate.
    for gamma in gammas:
        cluster = proxy_cluster(gamma, 0.25, 0.5, 0.02, h)
        samples = []
        for _ in range(12):
            t0 = time.perf_counter()
            mixed_product_integration(cluster, grid, smooth, deriv)
            samples.append(time.perf_counter() - t0)
        runtime_by_gamma[str(gamma)] = float(np.median(samples))
    return {
        "grid_points": int(grid.size), "h_thz": h, "full": bool(full),
        "cases": len(rows), "mixed_error_by_gamma": by_gamma,
        "mixed_runtime_seconds_by_gamma": runtime_by_gamma,
        "max_mixed_error": max(r["mixed_rel_error"] for r in rows),
        "min_spectral_eigenvalue": min(r["min_spectral_eigenvalue"] for r in rows),
        "max_keldysh_antihermiticity": max(r["keldysh_antihermiticity"] for r in rows),
        "rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    result = run_sweep(full=a.full)
    print("gamma/h | max mixed error | median mixed error")
    for gamma, row in result["mixed_error_by_gamma"].items():
        print(f"{gamma:>7s} | {row['max']:.3e} | {row['median']:.3e}")
    print(f"cases={result['cases']} max_error={result['max_mixed_error']:.3e} "
          f"min_eig={result['min_spectral_eigenvalue']:.3e} "
          f"anti={result['max_keldysh_antihermiticity']:.3e}")
    if a.json is not None:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
