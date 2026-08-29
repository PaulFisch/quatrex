"""Fast projected convolution on locally refined dyadic P1 meshes.

This private prototype implements the level recurrences from

    W. Hackbusch, *Fast and Exact Projected Convolution of Piecewise Linear
    Functions on Non-equidistant Grids*, Computing 80 (2007), 137--168.

The important interface is :func:`projected_convolution`: the scalar product
inside each discrete convolution is a callback.  Replacing scalar
multiplication by Quatrex's bilinear FC3 ring therefore preserves the frequency
algorithm while reusing the production spatial contraction.

The representation is discontinuous, cell-local, orthonormal Legendre P1.
Continuous nodal P1 data are converted cell by cell; the final continuous
projection is a separate tridiagonal mass solve.  Keeping the discontinuous
form here makes the multilevel recurrences local and testable.

This is study code.  It does not change production solver behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.fft import next_fast_len
from scipy.linalg import solve_banded


Array = np.ndarray
Bilinear = Callable[[Array, Array], Array]


@dataclass(frozen=True)
class Sequence:
    """Finite part of an integer-indexed sequence.

    ``data[j]`` is coefficient ``start + j``.  Trailing dimensions are
    arbitrary and are passed through all scalar level recurrences.
    """

    start: int
    data: Array

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        if data.ndim < 1:
            raise ValueError("sequence data needs an index axis")
        object.__setattr__(self, "data", data)

    @property
    def stop(self) -> int:
        return self.start + self.data.shape[0]

    def sample(self, indices: Array) -> Array:
        idx = np.asarray(indices, dtype=int)
        out = np.zeros(idx.shape + self.data.shape[1:], dtype=self.data.dtype)
        live = (idx >= self.start) & (idx < self.stop)
        if np.any(live):
            out[live] = self.data[idx[live] - self.start]
        return out


def _trim(seq: Sequence) -> Sequence:
    if seq.data.shape[0] == 0:
        return seq
    flat = seq.data.reshape(seq.data.shape[0], -1)
    live = np.any(flat != 0, axis=1)
    if not np.any(live):
        return Sequence(0, seq.data[:0])
    lo, hi = np.flatnonzero(live)[[0, -1]]
    return Sequence(seq.start + int(lo), seq.data[lo:int(hi) + 1])


def _crop(seq: Sequence, start: int, stop: int) -> Sequence:
    """Restrict an integer sequence to ``[start, stop)``."""
    lo, hi = max(start, seq.start), min(stop, seq.stop)
    if hi <= lo:
        return Sequence(start, np.zeros((0,) + seq.data.shape[1:],
                                        dtype=seq.data.dtype))
    return Sequence(lo, seq.data[lo - seq.start:hi - seq.start])


def _nonzero_components(seq: Sequence) -> list[Sequence]:
    """Split disconnected active cells before an FFT densifies their gap."""
    if not seq.data.shape[0]:
        return []
    live = np.any(seq.data.reshape(seq.data.shape[0], -1) != 0, axis=1)
    edge = np.diff(np.pad(live.astype(np.int8), (1, 1)))
    starts, stops = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)
    return [Sequence(seq.start + int(a), seq.data[a:b])
            for a, b in zip(starts, stops)]


def _integer_runs(indices: Array) -> list[tuple[int, int]]:
    idx = np.unique(np.asarray(indices, dtype=int))
    if not idx.size:
        return []
    cuts = np.flatnonzero(np.diff(idx) != 1) + 1
    return [(int(part[0]), int(part[-1]) + 1)
            for part in np.split(idx, cuts)]


def _sum_sequences(parts: list[Sequence]) -> Sequence:
    parts = [p for p in parts if p.data.shape[0]]
    if not parts:
        return Sequence(0, np.zeros((0,), dtype=complex))
    start = min(p.start for p in parts)
    stop = max(p.stop for p in parts)
    tail = parts[0].data.shape[1:]
    dtype = np.result_type(*[p.data.dtype for p in parts])
    out = np.zeros((stop - start,) + tail, dtype=dtype)
    for part in parts:
        if part.data.shape[1:] != tail:
            raise ValueError("cannot add sequences with different trailing shapes")
        out[part.start - start:part.stop - start] += part.data
    return _trim(Sequence(start, out))


def fft_bilinear_convolution(a: Sequence, b: Sequence,
                             bilinear: Bilinear | None = None) -> Sequence:
    """Discrete sequence convolution through an arbitrary bilinear product."""
    if a.data.shape[0] == 0 or b.data.shape[0] == 0:
        return Sequence(a.start + b.start, np.zeros((0,), dtype=complex))
    n = a.data.shape[0] + b.data.shape[0] - 1
    nfft = next_fast_len(n)
    ah = np.fft.fft(a.data, n=nfft, axis=0)
    bh = np.fft.fft(b.data, n=nfft, axis=0)
    product = ah * bh if bilinear is None else bilinear(ah, bh)
    out = np.fft.ifft(product, axis=0)[:n]
    if (np.isrealobj(a.data) and np.isrealobj(b.data)
            and np.max(np.abs(out.imag), initial=0.0) < 1e-12):
        out = out.real
    return _trim(Sequence(a.start + b.start, out))


def fft_bilinear_convolutions(
        pairs: list[tuple[Sequence, Sequence]],
        bilinear: Bilinear | None = None) -> list[Sequence]:
    """Evaluate independent convolutions in batched Fourier-mode calls.

    The FC3 ring is pointwise in the transformed frequency index.  Independent
    FFTs with the same padded length can consequently be concatenated along
    that index, contracted in one GPU/BLAS launch, and split again.  This does
    not mix products and is algebraically identical to repeated calls to
    :func:`fft_bilinear_convolution`.

    Empty products are deliberately rejected: their result trailing shape
    cannot be inferred without evaluating ``bilinear``.  The multilevel
    callers remove them before reaching this routine.
    """
    if not pairs:
        return []
    if any(a.data.shape[0] == 0 or b.data.shape[0] == 0 for a, b in pairs):
        raise ValueError("batched convolution received an empty sequence")

    groups: dict[int, list[int]] = {}
    lengths = []
    for ip, (a, b) in enumerate(pairs):
        n = a.data.shape[0] + b.data.shape[0] - 1
        lengths.append(n)
        groups.setdefault(next_fast_len(n), []).append(ip)

    result: list[Sequence | None] = [None] * len(pairs)
    for nfft, members in groups.items():
        tail_a = pairs[members[0]][0].data.shape[1:]
        tail_b = pairs[members[0]][1].data.shape[1:]
        if any(pairs[ip][0].data.shape[1:] != tail_a
               or pairs[ip][1].data.shape[1:] != tail_b for ip in members):
            raise ValueError("one FFT batch needs common input trailing shapes")
        apad = np.zeros((len(members), nfft) + tail_a,
                        dtype=np.result_type(*[
                            pairs[ip][0].data for ip in members], complex))
        bpad = np.zeros((len(members), nfft) + tail_b,
                        dtype=np.result_type(*[
                            pairs[ip][1].data for ip in members], complex))
        for local, ip in enumerate(members):
            a, b = pairs[ip]
            apad[local, :a.data.shape[0]] = a.data
            bpad[local, :b.data.shape[0]] = b.data
        ah = np.fft.fft(apad, axis=1)
        bh = np.fft.fft(bpad, axis=1)
        in_tail_a, in_tail_b = ah.shape[2:], bh.shape[2:]
        flat_a = ah.reshape((len(members) * nfft,) + in_tail_a)
        flat_b = bh.reshape((len(members) * nfft,) + in_tail_b)
        product = flat_a * flat_b if bilinear is None else bilinear(
            flat_a, flat_b)
        product = np.asarray(product)
        if product.shape[0] != len(members) * nfft:
            raise ValueError("bilinear callback changed its mode axis")
        product = product.reshape(
            (len(members), nfft) + product.shape[1:])
        values = np.fft.ifft(product, axis=1)
        for local, ip in enumerate(members):
            a, b = pairs[ip]
            out = values[local, :lengths[ip]]
            if (np.isrealobj(a.data) and np.isrealobj(b.data)
                    and np.max(np.abs(out.imag), initial=0.0) < 1e-12):
                out = out.real
            result[ip] = _trim(Sequence(a.start + b.start, out))
    return [part for part in result if part is not None]


@dataclass(frozen=True)
class DyadicMesh:
    """Leaf cells ``(level, index)`` with width ``base_h / 2**level``."""

    base_h: float
    leaves: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.base_h <= 0.0 or not self.leaves:
            raise ValueError("mesh needs positive base_h and at least one leaf")
        intervals = sorted((i * self.base_h / 2**level,
                            (i + 1) * self.base_h / 2**level,
                            level, i) for level, i in self.leaves)
        for left, right, level, _ in intervals:
            if level < 0 or not right > left:
                raise ValueError("invalid dyadic leaf")
        for a, b in zip(intervals[:-1], intervals[1:]):
            if not np.isclose(a[1], b[0], rtol=0.0, atol=1e-12 * self.base_h):
                raise ValueError("leaves must be a gap-free non-overlapping partition")
        object.__setattr__(self, "leaves", tuple(
            (level, i) for _, _, level, i in intervals))

    @classmethod
    def refined(cls, base_h: float, first: int, stop: int,
                target_level: Callable[[float, float], int]) -> "DyadicMesh":
        leaves: list[tuple[int, int]] = []

        def visit(level: int, index: int) -> None:
            h = base_h / 2**level
            left, right = index * h, (index + 1) * h
            wanted = int(target_level(left, right))
            if level >= wanted:
                leaves.append((level, index))
            else:
                visit(level + 1, 2 * index)
                visit(level + 1, 2 * index + 1)

        for index in range(first, stop):
            visit(0, index)
        return cls(base_h, tuple(leaves))

    @property
    def levels(self) -> tuple[int, ...]:
        return tuple(sorted({level for level, _ in self.leaves}))

    @property
    def vertices(self) -> Array:
        h0 = self.base_h
        edges = []
        for level, index in self.leaves:
            h = h0 / 2**level
            edges.extend((index * h, (index + 1) * h))
        return np.unique(np.asarray(edges, float))

    def level_indices(self, level: int) -> Array:
        return np.asarray([i for lev, i in self.leaves if lev == level], int)


@dataclass(frozen=True)
class P1Field:
    mesh: DyadicMesh
    levels: dict[int, Sequence]

    @classmethod
    def from_callable(cls, mesh: DyadicMesh, fn: Callable[[Array], Array]) -> "P1Field":
        levels: dict[int, Sequence] = {}
        for level in mesh.levels:
            idx = mesh.level_indices(level)
            start, stop = int(idx.min()), int(idx.max()) + 1
            all_i = np.arange(start, stop)
            h = mesh.base_h / 2**level
            left = all_i * h
            right = left + h
            yl = np.asarray(fn(left))
            yr = np.asarray(fn(right))
            tail = np.broadcast_shapes(yl.shape[1:], yr.shape[1:])
            yl = np.broadcast_to(yl, (all_i.size,) + tail)
            yr = np.broadcast_to(yr, (all_i.size,) + tail)
            data = np.zeros((all_i.size, 2) + tail,
                            dtype=np.result_type(yl, yr))
            active = np.isin(all_i, idx)
            root_h = np.sqrt(h)
            data[active, 0] = 0.5 * root_h * (yl[active] + yr[active])
            data[active, 1] = root_h / np.sqrt(12.0) * (
                yr[active] - yl[active])
            levels[level] = _trim(Sequence(start, data))
        return cls(mesh, levels)

    @classmethod
    def from_vertices(cls, mesh: DyadicMesh, values: Array) -> "P1Field":
        """Build the continuous interpolant from values at mesh vertices."""
        vertices = mesh.vertices
        values = np.asarray(values)
        if values.shape[0] != vertices.size:
            raise ValueError("one value is required at every mesh vertex")
        levels: dict[int, Sequence] = {}
        for level in mesh.levels:
            idx = mesh.level_indices(level)
            start, stop = int(idx.min()), int(idx.max()) + 1
            all_i = np.arange(start, stop)
            h = mesh.base_h / 2**level
            left = all_i * h
            li = np.searchsorted(vertices, left)
            ri = np.searchsorted(vertices, left + h)
            live = np.isin(all_i, idx)
            data = np.zeros((all_i.size, 2) + values.shape[1:],
                            dtype=values.dtype)
            root_h = np.sqrt(h)
            data[live, 0] = 0.5 * root_h * (
                values[li[live]] + values[ri[live]])
            data[live, 1] = root_h / np.sqrt(12.0) * (
                values[ri[live]] - values[li[live]])
            levels[level] = _trim(Sequence(start, data))
        return cls(mesh, levels)


@dataclass(frozen=True)
class ContinuousP1Field:
    """Globally continuous P1 values on the adaptive mesh vertices."""

    mesh: DyadicMesh
    values: Array

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.shape[0] != self.mesh.vertices.size:
            raise ValueError("continuous P1 values must match mesh vertices")
        object.__setattr__(self, "values", values)


def project_continuous(field: P1Field) -> ContinuousP1Field:
    """Exact L2 projection of discontinuous P1 data into continuous P1.

    The adaptive nodal mass matrix is tridiagonal.  Constants and the global
    coordinate belong to this space, so this final solve retains both zeroth
    and first convolution moments to roundoff when the output support is
    covered.
    """
    vertices = field.mesh.vertices
    n = vertices.size
    tail = next(iter(field.levels.values())).data.shape[2:]
    dtype = np.result_type(*[seq.data for seq in field.levels.values()])
    rhs = np.zeros((n,) + tail, dtype=dtype)
    diag = np.zeros(n, float)
    upper = np.zeros(n - 1, float)
    for level, index in field.mesh.leaves:
        h = field.mesh.base_h / 2**level
        left = index * h
        iv = int(np.searchsorted(vertices, left))
        coeff = field.levels[level].sample(np.array([index]))[0]
        root_h = np.sqrt(h)
        rhs[iv] += root_h * (0.5 * coeff[0]
                             - coeff[1] / np.sqrt(12.0))
        rhs[iv + 1] += root_h * (0.5 * coeff[0]
                                 + coeff[1] / np.sqrt(12.0))
        diag[iv] += h / 3.0
        diag[iv + 1] += h / 3.0
        upper[iv] += h / 6.0
    ab = np.zeros((3, n), float)
    ab[0, 1:] = upper
    ab[1] = diag
    ab[2, :-1] = upper
    flat = rhs.reshape(n, -1)
    values = solve_banded((1, 1), ab, flat).reshape(rhs.shape)
    return ContinuousP1Field(field.mesh, values)


def p1_moments(field: P1Field) -> tuple[Array, Array]:
    """Return exact zeroth and first frequency moments of a P1 field."""
    m0 = m1 = None
    for level, index in field.mesh.leaves:
        h = field.mesh.base_h / 2**level
        centre = (index + 0.5) * h
        coeff = field.levels[level].sample(np.array([index]))[0]
        z = np.sqrt(h) * coeff[0]
        o = centre * z + h**1.5 / np.sqrt(12.0) * coeff[1]
        m0 = z if m0 is None else m0 + z
        m1 = o if m1 is None else m1 + o
    return np.asarray(m0), np.asarray(m1)


def refine_p1(seq: Sequence, levels: int = 1,
              window: tuple[int, int] | None = None) -> Sequence:
    out = seq
    for step in range(levels):
        if window is not None:
            remaining = levels - step
            scale = 2**remaining
            parent_start = int(np.floor(window[0] / scale))
            parent_stop = int(np.ceil(window[1] / scale))
            out = _crop(out, parent_start, parent_stop)
            if not out.data.shape[0]:
                return Sequence(window[0], np.zeros(
                    (0,) + seq.data.shape[1:], dtype=seq.data.dtype))
        data = np.zeros((2 * out.data.shape[0],) + out.data.shape[1:],
                        dtype=out.data.dtype)
        c0, c1 = out.data[:, 0], out.data[:, 1]
        data[0::2, 0] = c0 / np.sqrt(2.0) - np.sqrt(3.0 / 8.0) * c1
        data[1::2, 0] = c0 / np.sqrt(2.0) + np.sqrt(3.0 / 8.0) * c1
        data[0::2, 1] = c1 / (2.0 * np.sqrt(2.0))
        data[1::2, 1] = c1 / (2.0 * np.sqrt(2.0))
        out = _trim(Sequence(2 * out.start, data))
        if window is not None:
            remaining = levels - step - 1
            scale = 2**remaining
            child_start = int(np.floor(window[0] / scale))
            child_stop = int(np.ceil(window[1] / scale))
            out = _crop(out, child_start, child_stop)
    return out


def coarsen_p1(seq: Sequence, levels: int = 1) -> Sequence:
    out = seq
    for _ in range(levels):
        if out.data.shape[0] == 0:
            return out
        first = int(np.floor(out.start / 2))
        last = int(np.floor((out.stop - 1) / 2))
        i = np.arange(first, last + 1)
        left = out.sample(2 * i)
        right = out.sample(2 * i + 1)
        data = np.zeros((i.size,) + out.data.shape[1:], dtype=out.data.dtype)
        data[:, 0] = (left[:, 0] + right[:, 0]) / np.sqrt(2.0)
        data[:, 1] = (np.sqrt(3.0 / 8.0) *
                      (right[:, 0] - left[:, 0])
                      + (left[:, 1] + right[:, 1]) / (2.0 * np.sqrt(2.0)))
        out = _trim(Sequence(first, data))
    return out


def _gamma_same_level(g: Sequence, h: float) -> Sequence:
    first, last = g.start, g.stop
    i = np.arange(first, last + 1)
    gi = g.sample(i)
    gm = g.sample(i - 1)
    tail = g.data.shape[2:]
    out = np.zeros((i.size, 2, 2) + tail, dtype=g.data.dtype)
    out[:, 0, 0] = (0.5 * np.sqrt(h) * (gi[:, 0] + gm[:, 0])
                    + np.sqrt(h / 12.0) * (gm[:, 1] - gi[:, 1]))
    out[:, 1, 0] = np.sqrt(h / 12.0) * (gi[:, 0] - gm[:, 0])
    out[:, 0, 1] = np.sqrt(h / 12.0) * (gm[:, 0] - gi[:, 0])
    out[:, 1, 1] = np.sqrt(3.0 * h) / 5.0 * (gm[:, 1] - gi[:, 1])
    return _trim(Sequence(first, out))


def _coarsen_gamma(gamma: Sequence) -> Sequence:
    first = int(np.floor((gamma.start - 1) / 2))
    last = int(np.floor(gamma.stop / 2)) + 1
    i = np.arange(first, last + 1)
    gm = gamma.sample(2 * i - 1)
    gc = gamma.sample(2 * i)
    gp = gamma.sample(2 * i + 1)
    tail = gamma.data.shape[3:]
    out = np.zeros((i.size, 2, 2) + tail, dtype=gamma.data.dtype)
    out[:, 0, 0] = 0.5 * gm[:, 0, 0] + gc[:, 0, 0] + 0.5 * gp[:, 0, 0]
    out[:, 1, 0] = (
        np.sqrt(3.0) / 4.0 * (gp[:, 0, 0] - gm[:, 0, 0])
        + 0.25 * (gm[:, 1, 0] + gp[:, 1, 0]) + 0.5 * gc[:, 1, 0])
    out[:, 0, 1] = (
        np.sqrt(3.0) / 4.0 * (gm[:, 0, 0] - gp[:, 0, 0])
        + 0.25 * (gm[:, 0, 1] + gp[:, 0, 1]) + 0.5 * gc[:, 0, 1])
    out[:, 1, 1] = (
        -3.0 / 8.0 * gm[:, 0, 0] + 3.0 / 4.0 * gc[:, 0, 0]
        - 3.0 / 8.0 * gp[:, 0, 0]
        + 0.125 * (gm[:, 1, 1] + gp[:, 1, 1]) + 0.25 * gc[:, 1, 1]
        + np.sqrt(3.0) / 8.0 * (-gm[:, 0, 1] + gp[:, 0, 1]
                               + gm[:, 1, 0] - gp[:, 1, 0]))
    return _trim(Sequence(first, out))


def gamma_to_level(g: Sequence, source_level: int, target_level: int,
                   base_h: float) -> Sequence:
    if target_level > source_level:
        raise ValueError("Gamma projection level cannot exceed source level")
    out = _gamma_same_level(g, base_h / 2**source_level)
    for _ in range(source_level - target_level):
        out = _coarsen_gamma(out)
    return out


def _case_ab(f: Sequence, g: Sequence, lf: int, lg: int, lo: int,
             base_h: float, bilinear: Bilinear | None,
             target: tuple[int, int] | None = None) -> Sequence | None:
    if not lf <= lg or lo > lg:
        raise ValueError("case A/B requires lf <= lg and lo <= lg")
    if lo < lf:
        scale = 2**(lf - lo)
        fine_target = None if target is None else (
            target[0] * scale, target[1] * scale)
        at_lf = _case_ab(
            f, g, lf, lg, lf, base_h, bilinear, fine_target)
        if at_lf is None:
            return None
        out = coarsen_p1(at_lf, lf - lo)
        return out if target is None else _crop(out, *target)
    gamma = gamma_to_level(g, lg, lo, base_h)
    if target is None:
        f_at = refine_p1(f, lo - lf)
    else:
        # For c[k] = sum_j f[j] gamma[k-j], requested output k in
        # [t0,t1) can only see these f indices.
        f_window = (target[0] - (gamma.stop - 1),
                    target[1] - gamma.start)
        f_at = refine_p1(f, lo - lf, f_window)
    if not f_at.data.shape[0] or not gamma.data.shape[0]:
        return None
    return _convolve_p1_gamma(f_at, gamma, bilinear, target)


def _convolve_p1_gamma(
        f_at: Sequence, gamma: Sequence, bilinear: Bilinear | None,
        target: tuple[int, int] | None = None) -> Sequence | None:
    """Contract P1 coefficients with a projected two-index Gamma sequence."""
    return _convolve_p1_gamma_many(
        [(f_at, gamma, target)], bilinear)[0]


def _convolve_p1_gamma_many(
        tasks: list[tuple[Sequence, Sequence, tuple[int, int] | None]],
        bilinear: Bilinear | None) -> list[Sequence | None]:
    """Batch independent P1--Gamma contractions across mesh components."""
    live = [(it, task) for it, task in enumerate(tasks)
            if task[0].data.shape[0] and task[1].data.shape[0]]
    result: list[Sequence | None] = [None] * len(tasks)
    if not live:
        return result
    pairs = [
        (Sequence(f_at.start, f_at.data[:, beta]),
         Sequence(gamma.start, gamma.data[:, alpha, beta]))
        for _, (f_at, gamma, _) in live
        for alpha in range(2) for beta in range(2)
    ]
    terms = fft_bilinear_convolutions(pairs, bilinear)
    for local, (original, (_, _, target)) in enumerate(live):
        base = 4 * local
        parts = [_sum_sequences(terms[base + 2 * alpha:
                                      base + 2 * alpha + 2])
                 for alpha in range(2)]
        start = min(p.start for p in parts)
        stop = max(p.stop for p in parts)
        tail = parts[0].data.shape[1:]
        data = np.zeros((stop - start, 2) + tail,
                        dtype=np.result_type(*[p.data.dtype for p in parts]))
        for alpha, part in enumerate(parts):
            data[part.start - start:part.stop - start, alpha] += part.data
        out = _trim(Sequence(start, data))
        result[original] = out if target is None else _crop(out, *target)
    return result


def _point_gammas(g: Sequence, h: float) -> tuple[Sequence, Sequence, Sequence]:
    first, last = g.start, g.stop + 2
    i = np.arange(first, last + 1)
    gi = g.sample(i)
    gm = g.sample(i - 1)
    gmm = g.sample(i - 2)
    tail = g.data.shape[2:]
    zero = np.zeros((i.size, 2) + tail, dtype=g.data.dtype)
    plus = np.zeros_like(zero)
    minus = np.zeros_like(zero)
    zero[:, 0] = gm[:, 0]
    zero[:, 1] = -gm[:, 1]
    plus[:, 0] = (gi[:, 0] - gm[:, 0]
                  + np.sqrt(3.0) * (gm[:, 1] - gi[:, 1])) / h
    plus[:, 1] = (np.sqrt(3.0) * (gm[:, 0] - gi[:, 0])
                  + 3.0 * (gm[:, 1] + gi[:, 1])) / h
    minus[:, 0] = (gm[:, 0] - gmm[:, 0]
                   + np.sqrt(3.0) * (gm[:, 1] - gmm[:, 1])) / h
    minus[:, 1] = (np.sqrt(3.0) * (gm[:, 0] - gmm[:, 0])
                   - 3.0 * (gm[:, 1] + gmm[:, 1])) / h
    return (_trim(Sequence(first, zero)), _trim(Sequence(first, plus)),
            _trim(Sequence(first, minus)))


def _point_convolutions(f: Sequence, gammas: tuple[Sequence, ...],
                        bilinear: Bilinear | None) -> list[Sequence]:
    return _point_convolutions_many([(f, gammas)], bilinear)[0]


def _point_convolutions_many(
        tasks: list[tuple[Sequence, tuple[Sequence, ...]]],
        bilinear: Bilinear | None) -> list[list[Sequence]]:
    """Batch independent point/derivative convolution triples."""
    if not tasks:
        return []
    n_gamma = len(tasks[0][1])
    if any(len(gammas) != n_gamma for _, gammas in tasks):
        raise ValueError("point-convolution batch needs a common tuple size")
    pairs = [
        (Sequence(f.start, f.data[:, beta]),
         Sequence(gamma.start, gamma.data[:, beta]))
        for f, gammas in tasks for gamma in gammas for beta in range(2)
    ]
    terms = fft_bilinear_convolutions(pairs, bilinear)
    result = []
    width = 2 * n_gamma
    for it in range(len(tasks)):
        base = width * it
        result.append([
            _sum_sequences(terms[base + 2 * ig:base + 2 * ig + 2])
            for ig in range(n_gamma)
        ])
    return result


def _refine_cubic(delta: Sequence, plus: Sequence, minus: Sequence,
                  h: float) -> tuple[Sequence, Sequence, Sequence]:
    first = min(delta.start, plus.start, minus.start)
    last = max(delta.stop, plus.stop, minus.stop) - 1
    i = np.arange(first, last + 1)
    d0, d1 = delta.sample(i), delta.sample(i + 1)
    p0, m1 = plus.sample(i), minus.sample(i + 1)
    tail = np.broadcast_shapes(d0.shape[1:], p0.shape[1:], m1.shape[1:])
    data_d = np.zeros((2 * i.size + 1,) + tail,
                      dtype=np.result_type(d0, d1, p0, m1))
    data_p = np.zeros_like(data_d)
    data_m = np.zeros_like(data_d)
    data_d[0:2 * i.size:2] = d0
    data_d[1:2 * i.size:2] = 0.5 * (d0 + d1) + h / 8.0 * (p0 - m1)
    data_d[-1] = delta.sample(np.array([last + 1]))[0]
    data_p[0:2 * i.size:2] = p0
    data_m[0:2 * i.size:2] = minus.sample(i)
    mid_deriv = 3.0 / (2.0 * h) * (d1 - d0) - 0.25 * (p0 + m1)
    data_p[1:2 * i.size:2] = mid_deriv
    data_m[1:2 * i.size:2] = mid_deriv
    data_p[-1] = plus.sample(np.array([last + 1]))[0]
    data_m[-1] = minus.sample(np.array([last + 1]))[0]
    start = 2 * first
    return (_trim(Sequence(start, data_d)), _trim(Sequence(start, data_p)),
            _trim(Sequence(start, data_m)))


def _project_cubic(delta: Sequence, plus: Sequence, minus: Sequence,
                   h: float) -> Sequence:
    first = min(delta.start, plus.start, minus.start)
    last = max(delta.stop, plus.stop, minus.stop) - 2
    i = np.arange(first, last + 1)
    d0, d1 = delta.sample(i), delta.sample(i + 1)
    p0, m1 = plus.sample(i), minus.sample(i + 1)
    tail = np.broadcast_shapes(d0.shape[1:], p0.shape[1:], m1.shape[1:])
    data = np.zeros((i.size, 2) + tail,
                    dtype=np.result_type(d0, d1, p0, m1))
    data[:, 0] = (np.sqrt(h) / 2.0 * (d0 + d1)
                  + h**1.5 / 12.0 * (p0 - m1))
    data[:, 1] = (np.sqrt(3.0 * h) / 5.0 * (d1 - d0)
                  - np.sqrt(3.0) * h**1.5 / 60.0 * (p0 + m1))
    return _trim(Sequence(first, data))


def _case_c(f: Sequence, g: Sequence, lf: int, lg: int, lo: int,
            base_h: float, bilinear: Bilinear | None,
            target: tuple[int, int] | None = None) -> Sequence | None:
    if not lf <= lg < lo:
        raise ValueError("case C requires lf <= lg < lo")
    h = base_h / 2**lg
    gz, gp, gm = _point_gammas(g, h)
    if target is None:
        f_at = refine_p1(f, lg - lf)
    else:
        scale = 2**(lo - lg)
        # Cubic refinement of target cells needs the containing coarse
        # interval plus one grid point on either side.
        coarse_start = int(np.floor(target[0] / scale)) - 1
        coarse_stop = int(np.ceil(target[1] / scale)) + 2
        gamma_start = min(gz.start, gp.start, gm.start)
        gamma_stop = max(gz.stop, gp.stop, gm.stop)
        f_window = (coarse_start - (gamma_stop - 1),
                    coarse_stop - gamma_start)
        f_at = refine_p1(f, lg - lf, f_window)
    if not f_at.data.shape[0]:
        return None
    delta, plus, minus = _point_convolutions(
        f_at, (gz, gp, gm), bilinear)
    if target is not None:
        scale = 2**(lo - lg)
        point_window = (int(np.floor(target[0] / scale)) - 2,
                        int(np.ceil(target[1] / scale)) + 3)
        delta = _crop(delta, *point_window)
        plus = _crop(plus, *point_window)
        minus = _crop(minus, *point_window)
    for level in range(lg, lo):
        delta, plus, minus = _refine_cubic(
            delta, plus, minus, base_h / 2**level)
        if target is not None:
            scale = 2**(lo - level - 1)
            point_window = (int(np.floor(target[0] / scale)) - 2,
                            int(np.ceil(target[1] / scale)) + 3)
            delta = _crop(delta, *point_window)
            plus = _crop(plus, *point_window)
            minus = _crop(minus, *point_window)
    out = _project_cubic(delta, plus, minus, base_h / 2**lo)
    return out if target is None else _crop(out, *target)


def level_pair_projection(f: Sequence, g: Sequence, lf: int, lg: int,
                          lo: int, base_h: float,
                          bilinear: Bilinear | None = None,
                          target: tuple[int, int] | None = None
                          ) -> Sequence | None:
    """Projected convolution of two single-level fields onto level ``lo``."""
    if lf <= lg:
        if lo <= lg:
            return _case_ab(f, g, lf, lg, lo, base_h, bilinear, target)
        return _case_c(f, g, lf, lg, lo, base_h, bilinear, target)

    swapped = None if bilinear is None else lambda a, b: bilinear(b, a)
    if lo <= lf:
        return _case_ab(g, f, lg, lf, lo, base_h, swapped, target)
    return _case_c(g, f, lg, lf, lo, base_h, swapped, target)


def projected_convolution(f: P1Field, g: P1Field, output_mesh: DyadicMesh,
                          bilinear: Bilinear | None = None) -> P1Field:
    """Exact discontinuous-P1 projection on ``output_mesh``.

    The current implementation follows the exact level algebra but stores one
    dense bounding sequence per active level.  A subsequent optimisation
    splits disconnected same-level components before FFT; correctness is
    intentionally established first.
    """
    if not np.isclose(f.mesh.base_h, g.mesh.base_h) or not np.isclose(
            f.mesh.base_h, output_mesh.base_h):
        raise ValueError("all meshes must share one dyadic base spacing")
    result: dict[int, Sequence] = {}
    f_components = {level: _nonzero_components(seq)
                    for level, seq in f.levels.items()}
    g_components = {level: _nonzero_components(seq)
                    for level, seq in g.levels.items()}
    for lo in output_mesh.levels:
        idx = output_mesh.level_indices(lo)
        start, stop = int(idx.min()), int(idx.max()) + 1
        tail = None
        accum = None
        for lf, fparts in f_components.items():
            for lg, gparts in g_components.items():
                for fs in fparts:
                    for gs in gparts:
                        for target in _integer_runs(idx):
                            part = level_pair_projection(
                                fs, gs, lf, lg, lo, output_mesh.base_h,
                                bilinear, target)
                            if part is None:
                                continue
                            sampled = part.sample(np.arange(start, stop))
                            if accum is None:
                                tail = sampled.shape[1:]
                                # Different level pairs may return real arrays
                                # or retain harmless complex FFT roundoff.
                                accum = np.zeros(
                                    sampled.shape,
                                    dtype=np.result_type(sampled, complex))
                            if sampled.shape[1:] != tail:
                                raise ValueError(
                                    "bilinear callback changed output shape")
                            accum += sampled
        if accum is None:
            accum = np.zeros((stop - start, 2), dtype=complex)
        active = np.isin(np.arange(start, stop), idx)
        accum[~active] = 0.0
        result[lo] = _trim(Sequence(start, accum))
    return P1Field(output_mesh, result)


def _gamma_hierarchy(g: P1Field
                     ) -> tuple[dict[int, Sequence | None],
                                dict[int, Sequence | None]]:
    """Return Gamma sums from levels ``>= level`` and ``> level``."""
    top = max(g.mesh.levels)
    inclusive: dict[int, Sequence | None] = {}
    strict: dict[int, Sequence | None] = {}
    acc: Sequence | None = None
    for level in range(top, -1, -1):
        if acc is not None:
            acc = _coarsen_gamma(acc)
        strict[level] = acc
        own = g.levels.get(level)
        if own is not None and own.data.shape[0]:
            own_gamma = _gamma_same_level(
                own, g.mesh.base_h / 2**level)
            acc = own_gamma if acc is None else _sum_sequences(
                [acc, own_gamma])
        inclusive[level] = acc
    return inclusive, strict


def _aggregate_refined(field: P1Field, target_level: int,
                       source_levels: list[int],
                       window: tuple[int, int] | None) -> Sequence | None:
    """Represent a sum of native level fields on one finer target level."""
    parts: list[Sequence] = []
    for level in source_levels:
        if level > target_level:
            raise ValueError("aggregate_refined only refines")
        seq = field.levels.get(level)
        if seq is None:
            continue
        for component in _nonzero_components(seq):
            part = refine_p1(component, target_level - level, window)
            if part.data.shape[0]:
                parts.append(part)
    return _sum_sequences(parts) if parts else None


def _add_output_part(parts: dict[int, list[Sequence]], level: int,
                     part: Sequence | None,
                     target: tuple[int, int] | None = None) -> None:
    if part is None or not part.data.shape[0]:
        return
    if target is not None:
        part = _crop(part, *target)
    if part.data.shape[0]:
        parts.setdefault(level, []).append(part)


def _case_a_combined(first: P1Field, second: P1Field,
                     output_mesh: DyadicMesh, bilinear: Bilinear | None,
                     *, strict: bool) -> dict[int, list[Sequence]]:
    """All ordered Case-A products, ``lo <= first_level <= second_level``."""
    gamma_inclusive, gamma_strict = _gamma_hierarchy(second)
    gammas = gamma_strict if strict else gamma_inclusive
    top = min(max(first.mesh.levels), max(second.mesh.levels))
    wanted = set(output_mesh.levels)
    parts: dict[int, list[Sequence]] = {}
    omega: Sequence | None = None
    for level in range(top, -1, -1):
        if omega is not None:
            omega = coarsen_p1(omega)
        f_level, gamma = first.levels.get(level), gammas.get(level)
        if f_level is not None and gamma is not None:
            tasks = [(fs, gs, None)
                     for fs in _nonzero_components(f_level)
                     for gs in _nonzero_components(gamma)]
            terms = [term for term in _convolve_p1_gamma_many(
                tasks, bilinear) if term is not None]
            if terms:
                current = _sum_sequences(terms)
                omega = current if omega is None else _sum_sequences(
                    [omega, current])
        if level in wanted and omega is not None:
            _add_output_part(parts, level, omega)
    return parts


def _case_b_combined(first: P1Field, second: P1Field,
                     output_mesh: DyadicMesh,
                     bilinear: Bilinear | None) -> dict[int, list[Sequence]]:
    """All ordered Case-B products, ``first_level < lo <= second_level``."""
    gammas, _ = _gamma_hierarchy(second)
    parts: dict[int, list[Sequence]] = {}
    for level in output_mesh.levels:
        gamma = gammas.get(level)
        if gamma is None:
            continue
        source_levels = [lev for lev in first.mesh.levels if lev < level]
        if not source_levels:
            continue
        for target in _integer_runs(output_mesh.level_indices(level)):
            tasks = []
            for gs in _nonzero_components(gamma):
                f_window = (target[0] - (gs.stop - 1),
                            target[1] - gs.start)
                fhat = _aggregate_refined(
                    first, level, source_levels, f_window)
                if fhat is None:
                    continue
                tasks.append((fhat, gs, target))
            for term in _convolve_p1_gamma_many(tasks, bilinear):
                _add_output_part(parts, level, term, target)
    return parts


def _case_c_combined(first: P1Field, second: P1Field,
                     output_mesh: DyadicMesh, bilinear: Bilinear | None,
                     *, strict: bool) -> dict[int, list[Sequence]]:
    """All ordered Case-C products, ``first_level <= second_level < lo``."""
    parts: dict[int, list[Sequence]] = {}
    for lo in output_mesh.levels:
        for target in _integer_runs(output_mesh.level_indices(lo)):
            tasks = []
            for lg in (lev for lev in second.mesh.levels if lev < lo):
                source_levels = [
                    lev for lev in first.mesh.levels
                    if lev < lg or (lev == lg and not strict)
                ]
                if not source_levels:
                    continue
                g_level = second.levels[lg]
                for gs in _nonzero_components(g_level):
                    h = output_mesh.base_h / 2**lg
                    point_gammas = _point_gammas(gs, h)
                    scale = 2**(lo - lg)
                    coarse_start = int(np.floor(target[0] / scale)) - 1
                    coarse_stop = int(np.ceil(target[1] / scale)) + 2
                    gamma_start = min(x.start for x in point_gammas)
                    gamma_stop = max(x.stop for x in point_gammas)
                    f_window = (coarse_start - (gamma_stop - 1),
                                coarse_stop - gamma_start)
                    fhat = _aggregate_refined(
                        first, lg, source_levels, f_window)
                    if fhat is None:
                        continue
                    tasks.append((fhat, point_gammas, lg,
                                  (coarse_start - 1, coarse_stop + 1)))
            point_tasks = [(fhat, gammas)
                           for fhat, gammas, _, _ in tasks]
            for (_, _, lg, point_window), (delta, plus, minus) in zip(
                    tasks, _point_convolutions_many(point_tasks, bilinear)):
                delta = _crop(delta, *point_window)
                plus = _crop(plus, *point_window)
                minus = _crop(minus, *point_window)
                for level in range(lg, lo):
                    delta, plus, minus = _refine_cubic(
                        delta, plus, minus,
                        output_mesh.base_h / 2**level)
                    scale = 2**(lo - level - 1)
                    needed = (int(np.floor(target[0] / scale)) - 2,
                              int(np.ceil(target[1] / scale)) + 3)
                    delta = _crop(delta, *needed)
                    plus = _crop(plus, *needed)
                    minus = _crop(minus, *needed)
                term = _project_cubic(
                    delta, plus, minus, output_mesh.base_h / 2**lo)
                _add_output_part(parts, lo, term, target)
    return parts


def projected_convolution_combined(
        f: P1Field, g: P1Field, output_mesh: DyadicMesh,
        bilinear: Bilinear | None = None) -> P1Field:
    """Exact Hackbusch convolution with level sums combined before FFTs.

    This is the production-oriented form of :func:`projected_convolution`.
    The latter expands every native-level pair and remains useful as an
    independent oracle; here Cases A--C use the intertwined level recurrences
    of sections 5.1.4, 5.2.3 and 5.3.4.  The reverse ordered triangle uses a
    swapped bilinear callback and excludes its diagonal, so noncommutative FC3
    products are neither assumed symmetric nor double counted.
    """
    if not np.isclose(f.mesh.base_h, g.mesh.base_h) or not np.isclose(
            f.mesh.base_h, output_mesh.base_h):
        raise ValueError("all meshes must share one dyadic base spacing")
    swapped = None if bilinear is None else lambda a, b: bilinear(b, a)
    sectors = [
        _case_a_combined(f, g, output_mesh, bilinear, strict=False),
        _case_a_combined(g, f, output_mesh, swapped, strict=True),
        _case_b_combined(f, g, output_mesh, bilinear),
        _case_b_combined(g, f, output_mesh, swapped),
        _case_c_combined(f, g, output_mesh, bilinear, strict=False),
        _case_c_combined(g, f, output_mesh, swapped, strict=True),
    ]
    result: dict[int, Sequence] = {}
    for level in output_mesh.levels:
        idx = output_mesh.level_indices(level)
        start, stop = int(idx.min()), int(idx.max()) + 1
        level_parts = [part for sector in sectors
                       for part in sector.get(level, [])]
        if not level_parts:
            result[level] = Sequence(start, np.zeros((stop - start, 2)))
            continue
        total = _sum_sequences(level_parts)
        data = total.sample(np.arange(start, stop))
        active = np.isin(np.arange(start, stop), idx)
        data[~active] = 0.0
        result[level] = _trim(Sequence(start, data))
    return P1Field(output_mesh, result)


def evaluate(field: P1Field, x: Array) -> Array:
    """Evaluate a discontinuous P1 field (right-continuous at leaf edges)."""
    points = np.asarray(x, float)
    tail = next(iter(field.levels.values())).data.shape[2:]
    out = np.zeros(points.shape + tail,
                   dtype=np.result_type(*[s.data for s in field.levels.values()]))
    right_edge = field.mesh.vertices[-1]
    for level, index in field.mesh.leaves:
        h = field.mesh.base_h / 2**level
        left, right = index * h, (index + 1) * h
        live = (points >= left) & (points < right)
        # Include the global right endpoint.
        if np.isclose(right, right_edge):
            live |= np.isclose(points, right)
        if not np.any(live):
            continue
        coeff = field.levels[level].sample(np.array([index]))[0]
        xi = (points[live] - (left + right) / 2.0) / h
        out[live] = (coeff[0] / np.sqrt(h)
                     + np.sqrt(12.0 / h) * xi.reshape(
                         xi.shape + (1,) * len(tail)) * coeff[1])
    return out


def evaluate_continuous(field: ContinuousP1Field, x: Array) -> Array:
    """Evaluate continuous adaptive P1 data by its nodal interpolant."""
    points = np.asarray(x, float)
    vertices = field.mesh.vertices
    flat = points.reshape(-1)
    hi = np.searchsorted(vertices, flat, side="right")
    hi = np.clip(hi, 1, vertices.size - 1)
    lo = hi - 1
    weight = (flat - vertices[lo]) / (vertices[hi] - vertices[lo])
    tail = field.values.shape[1:]
    shape = (flat.size,) + (1,) * len(tail)
    out = ((1.0 - weight).reshape(shape) * field.values[lo]
           + weight.reshape(shape) * field.values[hi])
    outside = (flat < vertices[0]) | (flat > vertices[-1])
    if np.any(outside):
        out[outside] = 0.0
    return out.reshape(points.shape + tail)
