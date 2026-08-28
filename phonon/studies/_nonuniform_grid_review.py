"""Decisive reduced study for Quatrex's dual frequency grid.

The production nonuniform path has two independent discretisations:

* Green functions and self-energies live on a possibly nonuniform *primary*
  grid;
* the SCBA convolution still lives on a zero-anchored uniform *auxiliary*
  grid, reached by piecewise-linear interpolation.

Earlier tests established that this bridge is implemented consistently.  This
study asks the missing complexity question: if a narrow line is resolved only
locally on the primary grid, may the auxiliary grid stay coarse?  A positive
Lorentzian mixture is useful here because its infinite-line bubble is known
analytically.  The primary grid resolves every input and combination line with
a fixed number of local points, while the auxiliary spacing is varied
independently.

Run the table recorded in ``phonon/docs/nonuniform_grid_review.md`` with::

    PYTHONPATH=phonon python phonon/studies/_nonuniform_grid_review.py \
      --json phonon/studies/out/nonuniform_grid_review.json

This is a private numerical study; it does not alter production behaviour.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class LorentzianMixture:
    """Positive scalar spectrum ``sum_j a_j L(w;c_j,g_j)``.

    ``L`` is a unit-area Cauchy/Lorentz density.  Its convolution is closed:
    ``L(c,g) * L(d,e) = L(c+d,g+e)``.
    """

    centres: np.ndarray
    widths: np.ndarray
    amplitudes: np.ndarray

    def __post_init__(self) -> None:
        c = np.asarray(self.centres, float)
        g = np.asarray(self.widths, float)
        a = np.asarray(self.amplitudes, float)
        if c.ndim != 1 or c.shape != g.shape or c.shape != a.shape:
            raise ValueError("centres, widths and amplitudes must be equal 1D arrays")
        if np.any(g <= 0.0) or np.any(a < 0.0):
            raise ValueError("widths must be positive and amplitudes nonnegative")
        object.__setattr__(self, "centres", c)
        object.__setattr__(self, "widths", g)
        object.__setattr__(self, "amplitudes", a)

    @staticmethod
    def _line(w: np.ndarray, centre: float, width: float) -> np.ndarray:
        return width / (np.pi * ((w - centre) ** 2 + width**2))

    def eval(self, omega: np.ndarray) -> np.ndarray:
        w = np.asarray(omega, float)
        out = np.zeros_like(w)
        for c, g, a in zip(self.centres, self.widths, self.amplitudes):
            out += a * self._line(w, c, g)
        return out

    def bubble(self, omega: np.ndarray) -> np.ndarray:
        """Exact infinite-line self-convolution."""
        w = np.asarray(omega, float)
        out = np.zeros_like(w)
        for c, g, a in zip(self.centres, self.widths, self.amplitudes):
            for d, e, b in zip(self.centres, self.widths, self.amplitudes):
                out += a * b * self._line(w, c + d, g + e)
        return out

    @property
    def output_centres(self) -> np.ndarray:
        return np.unique(np.add.outer(self.centres, self.centres).ravel())

    @property
    def output_widths(self) -> np.ndarray:
        # For coincident combination centres retain the narrowest local scale.
        cs = np.add.outer(self.centres, self.centres).ravel()
        gs = np.add.outer(self.widths, self.widths).ravel()
        uc = np.unique(cs)
        return np.asarray([np.min(gs[np.isclose(cs, c)]) for c in uc])


def cell_widths(grid: np.ndarray) -> np.ndarray:
    """Voronoi/midpoint cell widths, matching production edge convention."""
    w = np.asarray(grid, float)
    if w.ndim != 1 or w.size < 2 or np.any(np.diff(w) <= 0.0):
        raise ValueError("grid must be a strictly increasing 1D array")
    out = np.empty_like(w)
    out[1:-1] = 0.5 * (w[2:] - w[:-2])
    out[0] = w[1] - w[0]
    out[-1] = w[-1] - w[-2]
    return out


def adaptive_grid(model: LorentzianMixture, fmax: float, background_h: float,
                  points_per_hwhm: int = 8, radii: float = 8.0) -> np.ndarray:
    """Background grid plus graded local patches at input/output lines.

    Combination-frequency patches are included because primary Sigma and G
    share one grid in the production solver.  A fixed-width patch is *not*
    enough: linear interpolation would connect its still-large Lorentzian tail
    to a distant background point and create a spurious triangle.  The
    ``gamma*sinh(t)`` nodes retain ``gamma/points_per_hwhm`` spacing at the
    peak and grade geometrically into the background.  Consequently the point
    count grows only logarithmically, rather than as ``1/gamma``.
    """
    pieces = [np.arange(-fmax, fmax + 0.5 * background_h, background_h)]
    centres = np.concatenate((model.centres, model.output_centres))
    widths = np.concatenate((model.widths, model.output_widths))
    del radii  # kept in the signature for compatibility with the first study draft
    dt = 1.0 / float(points_per_hwhm)
    for centre, width in zip(centres, widths):
        reach = max(fmax + centre, fmax - centre)
        tmax = np.arcsinh(reach / width)
        t = np.arange(0.0, tmax + dt, dt)
        x = width * np.sinh(t)
        pieces.append(centre + x)
        pieces.append(centre - x)
    grid = np.unique(np.concatenate(pieces))
    grid = grid[(grid >= -fmax) & (grid <= fmax)]
    # Pin endpoints even when arange roundoff missed one.
    return np.unique(np.concatenate(([-fmax], grid, [fmax])))


def adaptive_output_grid(model: LorentzianMixture, fmax: float,
                         background_h: float,
                         points_per_hwhm: int = 8) -> np.ndarray:
    """Graded grid for the full ``[-2*fmax, 2*fmax]`` bubble support."""
    limit = 2.0 * fmax
    pieces = [np.arange(-limit, limit + 0.5 * background_h, background_h)]
    dt = 1.0 / float(points_per_hwhm)
    for centre, width in zip(model.output_centres, model.output_widths):
        reach = max(limit + centre, limit - centre)
        t = np.arange(0.0, np.arcsinh(reach / width) + dt, dt)
        x = width * np.sinh(t)
        pieces.extend((centre + x, centre - x))
    grid = np.unique(np.concatenate(pieces))
    grid = grid[(grid >= -limit) & (grid <= limit)]
    return np.unique(np.concatenate(([-limit], grid, [limit])))


def uniform_grid(fmax: float, spacing: float) -> np.ndarray:
    n = int(np.ceil(2.0 * fmax / float(spacing))) + 1
    return np.linspace(-fmax, fmax, n)


def interpolate_primary(primary: np.ndarray, values: np.ndarray,
                        auxiliary: np.ndarray) -> np.ndarray:
    """Production P operator on a full-line test grid (zero outside)."""
    return np.interp(auxiliary, primary, values, left=0.0, right=0.0)


def fft_bubble(auxiliary: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rectangle-rule convolution and its natural ``[-2F,2F]`` grid."""
    h = float(auxiliary[1] - auxiliary[0])
    omega = auxiliary[0] + auxiliary[0] + np.arange(2 * auxiliary.size - 1) * h
    return omega, h * fftconvolve(values, values, mode="full")


def p1_product_integration(primary: np.ndarray, values: np.ndarray,
                           output: np.ndarray) -> np.ndarray:
    """Convolve one compactly supported nonuniform P1 interpolant exactly.

    For fixed output ``z``, the breakpoints of
    ``g_h(x) g_h(z-x)`` are the primary knots and their reflections about
    ``z``.  Both factors are linear between consecutive breakpoints, hence
    their product is quadratic and the interval integral below is exact.

    This is the reference nonuniform collision discretisation, not a proposed
    production algorithm: evaluating all output knots costs ``O(N_p**2)``.
    It cleanly separates the accuracy of a nonuniform basis from the accuracy
    and inverse-linewidth cost of uniform auxiliary gridding.
    """
    p = np.asarray(primary, float)
    v = np.asarray(values)
    z_values = np.asarray(output, float)
    if p.ndim != 1 or v.shape != p.shape or np.any(np.diff(p) <= 0.0):
        raise ValueError("primary must be increasing and values must match it")
    result = np.zeros(z_values.shape, dtype=np.result_type(v, float))
    left, right = float(p[0]), float(p[-1])
    for k, z in np.ndenumerate(z_values):
        lo = max(left, float(z) - right)
        hi = min(right, float(z) - left)
        if hi <= lo:
            continue
        direct = p[(p > lo) & (p < hi)]
        reflected = float(z) - p
        reflected = reflected[(reflected > lo) & (reflected < hi)]
        knots = np.unique(np.concatenate(([lo], direct, reflected, [hi])))
        x0, x1 = knots[:-1], knots[1:]
        f0 = np.interp(x0, p, v)
        f1 = np.interp(x1, p, v)
        g0 = np.interp(float(z) - x0, p, v)
        g1 = np.interp(float(z) - x1, p, v)
        result[k] = np.sum((x1 - x0) * (
            2.0 * f0 * g0 + f0 * g1 + f1 * g0 + 2.0 * f1 * g1) / 6.0)
    return result


def interpolation_plan(primary: np.ndarray, auxiliary: np.ndarray):
    """Return the two-sparse linear P plan used by production."""
    hi = np.clip(np.searchsorted(primary, auxiliary, side="left"),
                 1, primary.size - 1)
    lo = hi - 1
    weight = np.clip((auxiliary - primary[lo]) /
                     (primary[hi] - primary[lo]), 0.0, 1.0)
    valid = ((auxiliary >= primary[0] - 1e-14) &
             (auxiliary <= primary[-1] + 1e-14))
    return lo, hi, weight, valid


def energy_adjoint(primary: np.ndarray, auxiliary: np.ndarray,
                   data: np.ndarray) -> np.ndarray:
    """Production-style ``(W omega)^-1 P^T(h omega)`` restriction.

    This helper uses positive grids because the heat pairing is the production
    invariant.  The zero-frequency row has zero measure and is returned as 0.
    """
    lo, hi, weight, valid = interpolation_plan(primary, auxiliary)
    h = float(auxiliary[1] - auxiliary[0])
    rhs = np.zeros(primary.size, dtype=np.result_type(data, complex))
    col = h * auxiliary * valid * data
    np.add.at(rhs, lo, (1.0 - weight) * col)
    np.add.at(rhs, hi, weight * col)
    denom = cell_widths(primary) * primary
    return np.divide(rhs, denom, out=np.zeros_like(rhs), where=denom > 0.0)


def energy_pairing_defect(primary: np.ndarray, auxiliary: np.ndarray,
                          seed: int = 7) -> tuple[float, float]:
    """Weighted-adjoint and point-sampling pairing defects."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=primary.size) + 1j * rng.normal(size=primary.size)
    g[0] = 0.0
    sigma = rng.normal(size=auxiliary.size) + 1j * rng.normal(size=auxiliary.size)
    pg = interpolate_primary(primary, g, auxiliary)
    h = float(auxiliary[1] - auxiliary[0])
    rhs = h * np.sum(auxiliary * sigma * pg)
    weights = cell_widths(primary) * primary
    adj = np.sum(weights * energy_adjoint(primary, auxiliary, sigma) * g)
    sampled = np.interp(primary, auxiliary, sigma.real) + 1j * np.interp(
        primary, auxiliary, sigma.imag)
    sam = np.sum(weights * sampled * g)
    scale = max(abs(rhs), 1e-300)
    return float(abs(adj - rhs) / scale), float(abs(sam - rhs) / scale)


def _relative_l2(got: np.ndarray, want: np.ndarray, spacing: float) -> float:
    num = np.sqrt(spacing * np.sum(np.abs(got - want) ** 2))
    den = np.sqrt(spacing * np.sum(np.abs(want) ** 2))
    return float(num / max(den, 1e-300))


def _p1_relative_l2(grid: np.ndarray, values: np.ndarray,
                    exact, interval: tuple[float, float]) -> float:
    """Gauss-integrated L2 error of a P1 output reconstruction."""
    nodes, weights = np.polynomial.legendre.leggauss(8)
    lo_limit, hi_limit = interval
    numerator = 0.0
    denominator = 0.0
    for x0, x1, y0, y1 in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        lo, hi = max(float(x0), lo_limit), min(float(x1), hi_limit)
        if hi <= lo:
            continue
        x = 0.5 * ((hi - lo) * nodes + hi + lo)
        t = (x - x0) / (x1 - x0)
        got = (1.0 - t) * y0 + t * y1
        want = exact(x)
        scale = 0.5 * (hi - lo)
        numerator += scale * float(np.sum(weights * np.abs(got - want) ** 2))
        denominator += scale * float(np.sum(weights * np.abs(want) ** 2))
    return float(np.sqrt(numerator / max(denominator, 1e-300)))


def _p1_peak_area_error(grid: np.ndarray, values: np.ndarray,
                        model: LorentzianMixture) -> float:
    """Peak-window integral error of the nonuniform P1 output."""
    errs = []
    for centre, width in zip(model.output_centres, model.output_widths):
        probe = np.linspace(centre - 8.0 * width,
                            centre + 8.0 * width, 2001)
        got_area = float(np.trapezoid(np.interp(probe, grid, values), probe))
        ref_area = float(np.trapezoid(model.bubble(probe), probe))
        errs.append(abs(got_area - ref_area) / max(abs(ref_area), 1e-300))
    return float(max(errs))


def _peak_area_error(omega: np.ndarray, got: np.ndarray,
                     model: LorentzianMixture) -> float:
    errs = []
    exact = model.bubble(omega)
    for centre, width in zip(model.output_centres, model.output_widths):
        mask = np.abs(omega - centre) <= 8.0 * width
        if np.count_nonzero(mask) < 2:
            # An unresolved interval is itself a failed quadrature, not zero
            # error.  Compare the whole nearest numerical cell to the exact
            # analytic mass in the physical peak window.
            i = int(np.argmin(np.abs(omega - centre)))
            h = float(omega[1] - omega[0])
            got_area = float(got[i] * h)
            probe = np.linspace(centre - 8 * width, centre + 8 * width, 2001)
            ref_area = float(np.trapezoid(model.bubble(probe), probe))
        else:
            got_area = float(np.trapezoid(got[mask], omega[mask]))
            ref_area = float(np.trapezoid(exact[mask], omega[mask]))
        errs.append(abs(got_area - ref_area) / max(abs(ref_area), 1e-300))
    return float(max(errs))


def run_sweep() -> dict:
    background_h = 0.25
    fmax = 4.0
    ratios = (1.0, 0.2, 0.04, 0.008, 0.001)
    offsets = (0.0, 0.25, 0.49)
    aux_rules = {
        "background": lambda gamma: background_h,
        "half_background": lambda gamma: 0.5 * background_h,
        "eight_per_hwhm": lambda gamma: gamma / 8.0,
    }
    rows = []
    direct_rows = []
    for ratio in ratios:
        gamma = ratio * background_h
        for offset in offsets:
            centre = 1.0 + offset * background_h
            model = LorentzianMixture(
                centres=np.array([centre - 1.4 * gamma, centre + 1.4 * gamma]),
                widths=np.array([gamma, 1.15 * gamma]),
                amplitudes=np.array([1.0, 0.63]),
            )
            primary = adaptive_grid(model, fmax, background_h)
            primary_values = model.eval(primary)
            direct_grid = adaptive_output_grid(model, fmax, background_h)
            direct = p1_product_integration(
                primary, primary_values, direct_grid)
            direct_rows.append({
                "gamma_over_background_h": ratio,
                "offset_over_background_h": offset,
                "primary_points": int(primary.size),
                "output_points": int(direct_grid.size),
                "bubble_relative_l2": _p1_relative_l2(
                    direct_grid, direct, model.bubble, (-3.5, 3.5)),
                "max_peak_area_error": _p1_peak_area_error(
                    direct_grid, direct, model),
                "pair_intervals_upper_bound": int(
                    primary.size * direct_grid.size),
            })
            for name, spacing_fn in aux_rules.items():
                requested_h = float(spacing_fn(gamma))
                auxiliary = uniform_grid(fmax, requested_h)
                actual_h = float(auxiliary[1] - auxiliary[0])
                exact_leg = model.eval(auxiliary)
                interp_leg = interpolate_primary(primary, primary_values, auxiliary)
                out_w, got = fft_bubble(auxiliary, interp_leg)
                exact = model.bubble(out_w)
                # Score only the physical central interval.  F=4 is at least
                # 11 HWHM from every input line even for gamma/h=1; the small
                # finite-domain tail error is reported as part of the method.
                live = np.abs(out_w) <= 3.5
                rows.append({
                    "gamma_over_background_h": ratio,
                    "offset_over_background_h": offset,
                    "aux_rule": name,
                    "primary_points": int(primary.size),
                    "auxiliary_points": int(auxiliary.size),
                    "aux_h_over_gamma": actual_h / gamma,
                    "leg_relative_l2": _relative_l2(
                        interp_leg, exact_leg, actual_h),
                    "bubble_relative_l2": _relative_l2(
                        got[live], exact[live], actual_h),
                    "max_peak_area_error": _peak_area_error(out_w, got, model),
                })

    by_rule_gamma = {}
    for rule in aux_rules:
        by_rule_gamma[rule] = {}
        for ratio in ratios:
            group = [r for r in rows if r["aux_rule"] == rule and
                     r["gamma_over_background_h"] == ratio]
            by_rule_gamma[rule][str(ratio)] = {
                "max_leg_relative_l2": max(r["leg_relative_l2"] for r in group),
                "max_bubble_relative_l2": max(r["bubble_relative_l2"] for r in group),
                "max_peak_area_error": max(r["max_peak_area_error"] for r in group),
                "primary_points": max(r["primary_points"] for r in group),
                "max_auxiliary_points": max(r["auxiliary_points"] for r in group),
            }

    direct_summary = {}
    for ratio in ratios:
        group = [r for r in direct_rows if
                 r["gamma_over_background_h"] == ratio]
        direct_summary[str(ratio)] = {
            "max_bubble_relative_l2": max(
                r["bubble_relative_l2"] for r in group),
            "max_peak_area_error": max(
                r["max_peak_area_error"] for r in group),
            "primary_points": max(r["primary_points"] for r in group),
            "output_points": max(r["output_points"] for r in group),
            "pair_intervals_upper_bound": max(
                r["pair_intervals_upper_bound"] for r in group),
        }

    # Independently pin the production energy-pairing distinction.
    positive_primary = np.unique(np.concatenate((
        np.linspace(0.0, 4.0, 25), np.linspace(0.8, 1.2, 31),
        np.linspace(1.8, 2.2, 31))))
    positive_aux = np.linspace(0.0, 4.0, 401)
    adj_defect, sample_defect = energy_pairing_defect(
        positive_primary, positive_aux)

    return {
        "background_h_thz": background_h,
        "fmax_thz": fmax,
        "cases": len(rows),
        "rows": rows,
        "summary": by_rule_gamma,
        "direct_nonuniform_rows": direct_rows,
        "direct_nonuniform_summary": direct_summary,
        "energy_pairing_adjoint_defect": adj_defect,
        "energy_pairing_sample_defect": sample_defect,
        "interpretation": {
            "primary_complexity": "logarithmic under linewidth refinement",
            "fine_auxiliary_complexity": "proportional to inverse linewidth",
            "direct_nonuniform_p1_complexity": "quadratic in primary points",
            "rational_cluster_complexity": "bounded by promoted cluster rank",
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    result = run_sweep()
    print("rule gamma/h | Nprimary Naux | leg-L2 bubble-L2 peak-area")
    for rule, by_gamma in result["summary"].items():
        for gamma, row in by_gamma.items():
            print(f"{rule:>15s} {gamma:>7s} | "
                  f"{row['primary_points']:8d} {row['max_auxiliary_points']:5d} | "
                  f"{row['max_leg_relative_l2']:.3e} "
                  f"{row['max_bubble_relative_l2']:.3e} "
                  f"{row['max_peak_area_error']:.3e}")
    print("direct nonuniform P1 gamma/h | Nprimary | bubble-L2 peak-area")
    for gamma, row in result["direct_nonuniform_summary"].items():
        print(f"{gamma:>7s} | {row['primary_points']:8d} | "
              f"{row['max_bubble_relative_l2']:.3e} "
              f"{row['max_peak_area_error']:.3e}")
    print("pairing defects: adjoint="
          f"{result['energy_pairing_adjoint_defect']:.3e}, sample="
          f"{result['energy_pairing_sample_defect']:.3e}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
