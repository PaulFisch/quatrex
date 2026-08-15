#!/usr/bin/env python
"""How many blocks the self-energy would need, measured on a real device.

Hypothesis: ``sse_g_band`` is far shorter than the distance an anharmonically
damped mode actually travels, and that is why the long-CNT band ladder brackets
its answer by a factor 2.2 instead of converging
(``document/src/results/64_gband.tex``).

The test is a number rather than an argument. A damped mode's range is
``xi = v_g / gamma`` in cells -- verified against the complex bands in
``tests/quatrex/phonon/test_spatial_modal.py`` -- so with the group velocity
from the harmonic dispersion and ``gamma`` from the pole census, the required
band follows with no new calculation. ``exp(-b/xi)`` is then the fraction of a
mode a boxcar of ``b`` blocks fails to remove.

Falsifier: if the measured ranges come out at or below the bands in use, the
truncation is a controlled approximation and the bracket has another cause.

The stored keys ``[nx, ny, nz]`` are REAL-SPACE cell offsets on all three
axes -- not a transport offset and two momentum indices, which is how the first
version of this script read them. ``cm_channel.py`` reads the same file and
sums over the transverse offsets to reach Gamma, which is what settles it. So a
transverse momentum needs a Fourier sum,

    D_n(q) = sum_{ny,nz} D[n, ny, nz] exp(i(q_y ny + q_z nz)),

before the transport dispersion D(k; q) = D_0(q) + D_+1(q) e^{ik} + h.c. can be
formed.

Run:
    python phonon/studies/_band_range_report.py \
        cluster/sichk_base/dynamical_matrix.mat --gamma 0.16 --band 3
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import scipy.io as sio

KEY = re.compile(r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]$")


def load_offsets(path: Path, transport_axis: int = 0):
    """``{(nx, ny, nz): block}`` -- real-space cell offsets on every axis."""
    raw = sio.loadmat(str(path))
    out: dict[tuple[int, int, int], np.ndarray] = {}
    for key, val in raw.items():
        m = KEY.match(key.strip())
        if not m:
            continue
        out[tuple(int(x) for x in m.groups())] = np.asarray(val)
    if not out:
        return out, ()
    axes = np.array(list(out))
    return out, tuple(sorted(set(axes[:, a])) for a in range(3))


def transport_layers(offsets, q_y: float, q_z: float, transport_axis: int = 0):
    """``{n_transport: D_n(q)}`` by Fourier summing the transverse offsets."""
    tr = [a for a in range(3) if a != transport_axis]
    layers: dict[int, np.ndarray] = {}
    for off, blk in offsets.items():
        n_t = off[transport_axis]
        phase = np.exp(1j * (q_y * off[tr[0]] + q_z * off[tr[1]]))
        if n_t not in layers:
            layers[n_t] = np.zeros(blk.shape, dtype=complex)
        layers[n_t] = layers[n_t] + blk * phase
    return layers


def dispersion(layers: dict[int, np.ndarray], n_k: int = 129):
    """``omega(k)`` and ``dw/dk`` for one transverse q, in THz and cells*THz."""
    d0, dp, dm = layers.get(0), layers.get(1), layers.get(-1)
    if d0 is None or dp is None or dm is None:
        return None, None
    k = np.linspace(0.0, np.pi, n_k)
    ph = np.exp(1j * k)[:, None, None]
    dk = d0[None] + dp[None] * ph + dm[None] * np.conj(ph)
    lam = np.linalg.eigvalsh(0.5 * (dk + np.conj(np.swapaxes(dk, -2, -1))))
    w = np.sqrt(np.maximum(lam.real, 0.0))            # (n_k, b)
    # central differences on the sorted branches; the endpoints are extrema
    # (v_g -> 0 there) so a one-sided difference would overstate them.
    v = np.gradient(w, k, axis=0)
    return w, np.abs(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", type=Path)
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.16],
                    help="HWHM in THz; the Si census median is 0.16")
    ap.add_argument("--band", type=int, nargs="+", default=[1, 2, 3],
                    help="sse_g_band values to compare against")
    ap.add_argument("--vmin", type=float, default=1e-3,
                    help="ignore near-flat branches below this cells*THz")
    args = ap.parse_args()

    offsets, axis_values = load_offsets(args.mat)
    if not offsets:
        print(f"no [nx, ny, nz] blocks in {args.mat}")
        return 1
    n_y, n_z = len(axis_values[1]), len(axis_values[2])
    q_mesh = [(2 * np.pi * i / n_y, 2 * np.pi * j / n_z)
              for i in range(n_y) for j in range(n_z)]
    layers = {(i, j): transport_layers(offsets, qy, qz)
              for (i, j), (qy, qz) in zip(
                  [(i, j) for i in range(n_y) for j in range(n_z)], q_mesh)}

    # Per (q, branch): the range that constrains a truncation is set by the
    # FASTEST point of a branch, not by an average over k. Averaging includes
    # the zone boundary, where every branch flattens by symmetry, and reports a
    # short range for a mode that travels far somewhere else in the zone.
    rows, decoupled = [], 0
    for q, lay in sorted(layers.items()):
        w, v = dispersion(lay)
        if w is None:
            continue
        coupling = np.linalg.norm(lay[1]) / max(np.linalg.norm(lay[0]), 1e-300)
        if coupling < 1e-12:
            # Some transverse channels carry no transport at all -- D_{+1} is
            # exactly zero. They have v_g == 0 by construction and say nothing
            # about a band, so counting them drags every statistic down.
            decoupled += 1
            continue
        for b in range(w.shape[1]):
            rows.append((q, b, float(v[:, b].max()), coupling))

    if not rows:
        print("every transverse channel is decoupled along transport")
        return 1
    v_max = np.array([r[2] for r in rows])
    live = v_max > args.vmin

    print(f"\n{args.mat}")
    print(f"  transverse q points   {len(layers)} "
          f"({decoupled} decoupled along transport, excluded)")
    print(f"  (q, branch) pairs     {len(rows)}, of which "
          f"{int(live.sum())} carry |v_g| > {args.vmin:g}")
    print(f"  branch-max |v_g| [cells*THz]  min/p25/med/p75/max  "
          + "  ".join(f"{np.percentile(v_max[live], p):.3g}"
                      for p in (0, 25, 50, 75, 100)))

    for gamma in args.gamma:
        xi = v_max[live] / gamma
        print(f"\n  gamma = {gamma:g} THz  ->  range xi = v_g/gamma [cells]")
        print("    min/p25/med/p75/max  "
              + "  ".join(f"{np.percentile(xi, p):.4g}"
                          for p in (0, 25, 50, 75, 100)))
        for b in args.band:
            over = float(np.mean(xi > b))
            kept = np.exp(-b / xi)
            print(f"    band {b}: {100 * over:5.1f} % of branches reach "
                  f"further than the band; the LONGEST-range branch keeps "
                  f"{np.exp(-b / xi.max()) * 100:5.1f} % of what the "
                  f"truncation should have removed "
                  f"(median branch {np.median(kept) * 100:.1f} %)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
