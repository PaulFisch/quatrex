"""Hierarchical near-band plus far-field reference study for phonon Sigma.

The existing spatial campaign measured one *global* quasiseparable rank.  This
module asks the missing algebraic question: after an exact block band is taken
out, what ranks are required by the off-diagonal blocks at each level of a
binary hierarchy?

It is a deliberately small HODLR reference, not a production HSS library:

* the near block band is stored exactly;
* sibling off-diagonal residual blocks are compressed by deterministic SVD;
* Hermitian inputs share the two directional factors exactly;
* an anti-Hermitian self-energy is compressed through ``H=+i Sigma``;
* matvecs never form the far field;
* a sparse extended system supplies a reference solve for the general
  non-Hermitian case.

The extended system is the standard low-rank Schur realization.  If
``A = B + U V^H``, then eliminating ``y`` from

    [ B   U ] [x] = [b]
    [ V^H -I ] [y]   [0]

gives ``A x=b``.  Stacking one auxiliary block per hierarchy node avoids the
single ``d+2*r_global`` width that made the global semiseparable arm expensive.

Run the reduced campaign used by ``phonon/docs/phph_acceleration_review.md``::

    QTX_ARRAY_MODULE=numpy PYTHONPATH=src:phonon \
      python phonon/studies/_spatial_hierarchy_review.py
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


TOLS = (1e-2, 1e-3, 1e-4)


def numerical_rank(a: np.ndarray, tol: float, max_rank: int | None = None) -> int:
    """Smallest Frobenius-tail rank below ``tol``."""
    if min(a.shape) == 0:
        return 0
    s = np.linalg.svd(a, compute_uv=False)
    if s.size == 0 or s[0] == 0.0:
        return 0
    tail = np.r_[np.cumsum((s[::-1] ** 2))[::-1], 0.0]
    total = max(float(tail[0]), 1e-300)
    rank = s.size
    for r in range(s.size + 1):
        if tail[r] / total <= float(tol) ** 2:
            rank = r
            break
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return int(rank)


def _factor(a: np.ndarray, tol: float, max_rank: int | None):
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    rank = numerical_rank(a, tol, max_rank)
    if rank == 0:
        return np.zeros((a.shape[0], 0), complex), np.zeros((0, a.shape[1]), complex)
    root = np.sqrt(s[:rank])
    return u[:, :rank] * root[None, :], root[:, None] * vh[:rank]


def block_band(a: np.ndarray, n_dof: int, half_band: int) -> np.ndarray:
    """Keep block entries ``|i-j| <= half_band`` exactly."""
    a = np.asarray(a, complex)
    if a.shape[0] != a.shape[1] or a.shape[0] % int(n_dof):
        raise ValueError("block_band needs a square matrix divisible by n_dof")
    n = a.shape[0] // int(n_dof)
    out = np.zeros_like(a)
    for i in range(n):
        lo, hi = max(0, i - half_band), min(n, i + half_band + 1)
        out[i*n_dof:(i+1)*n_dof, lo*n_dof:hi*n_dof] = \
            a[i*n_dof:(i+1)*n_dof, lo*n_dof:hi*n_dof]
    return out


@dataclass
class _Node:
    lo: int
    hi: int
    level: int
    leaf: np.ndarray | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    upper_u: np.ndarray | None = None
    upper_vh: np.ndarray | None = None
    lower_u: np.ndarray | None = None
    lower_vh: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        return self.leaf is not None

    def add_dense(self, out: np.ndarray) -> None:
        if self.is_leaf:
            out[self.lo:self.hi, self.lo:self.hi] += self.leaf
            return
        assert self.left is not None and self.right is not None
        self.left.add_dense(out)
        self.right.add_dense(out)
        mid = self.left.hi
        if self.upper_u is not None:
            out[self.lo:mid, mid:self.hi] += self.upper_u @ self.upper_vh
        if self.lower_u is not None:
            out[mid:self.hi, self.lo:mid] += self.lower_u @ self.lower_vh

    def add_apply(self, x: np.ndarray, out: np.ndarray) -> None:
        if self.is_leaf:
            out[self.lo:self.hi] += self.leaf @ x[self.lo:self.hi]
            return
        assert self.left is not None and self.right is not None
        self.left.add_apply(x, out)
        self.right.add_apply(x, out)
        mid = self.left.hi
        if self.upper_u is not None:
            out[self.lo:mid] += self.upper_u @ (self.upper_vh @ x[mid:self.hi])
        if self.lower_u is not None:
            out[mid:self.hi] += self.lower_u @ (self.lower_vh @ x[self.lo:mid])

    def factors(self, n: int):
        """Yield global ``U,Vh,level`` for every directed low-rank block."""
        if self.is_leaf:
            return
        assert self.left is not None and self.right is not None
        mid = self.left.hi
        for u, vh, rs, cs in (
            (self.upper_u, self.upper_vh, slice(self.lo, mid), slice(mid, self.hi)),
            (self.lower_u, self.lower_vh, slice(mid, self.hi), slice(self.lo, mid)),
        ):
            if u is not None and u.shape[1]:
                ug = np.zeros((n, u.shape[1]), complex)
                vg = np.zeros((u.shape[1], n), complex)
                ug[rs] = u
                vg[:, cs] = vh
                yield ug, vg, self.level
        yield from self.left.factors(n)
        yield from self.right.factors(n)

    def leaf_sparse(self, n: int) -> sparse.csr_matrix:
        if self.is_leaf:
            rows, cols = np.nonzero(self.leaf)
            return sparse.coo_matrix(
                (self.leaf[rows, cols], (rows + self.lo, cols + self.lo)),
                shape=(n, n)).tocsr()
        assert self.left is not None and self.right is not None
        return self.left.leaf_sparse(n) + self.right.leaf_sparse(n)

    def ranks_by_level(self, out: dict[int, list[int]]) -> None:
        if self.is_leaf:
            return
        ranks = []
        if self.upper_u is not None:
            ranks.append(int(self.upper_u.shape[1]))
        if self.lower_u is not None:
            ranks.append(int(self.lower_u.shape[1]))
        out.setdefault(self.level, []).extend(ranks)
        assert self.left is not None and self.right is not None
        self.left.ranks_by_level(out)
        self.right.ranks_by_level(out)

    def stored_scalars(self) -> int:
        if self.is_leaf:
            return int(self.leaf.size)
        total = 0
        for a in (self.upper_u, self.upper_vh, self.lower_u, self.lower_vh):
            if a is not None:
                total += a.size
        assert self.left is not None and self.right is not None
        return int(total + self.left.stored_scalars() + self.right.stored_scalars())


def _build_node(residual: np.ndarray, lo: int, hi: int, leaf_size: int,
                tol: float, max_rank: int | None, level: int,
                hermitian: bool) -> _Node:
    if hi - lo <= leaf_size:
        return _Node(lo, hi, level, leaf=np.array(residual[lo:hi, lo:hi], copy=True))
    mid = (lo + hi) // 2
    upper_u, upper_vh = _factor(residual[lo:mid, mid:hi], tol, max_rank)
    if hermitian:
        lower_u = upper_vh.conj().T
        lower_vh = upper_u.conj().T
    else:
        lower_u, lower_vh = _factor(residual[mid:hi, lo:mid], tol, max_rank)
    return _Node(
        lo, hi, level,
        left=_build_node(residual, lo, mid, leaf_size, tol, max_rank,
                         level + 1, hermitian),
        right=_build_node(residual, mid, hi, leaf_size, tol, max_rank,
                          level + 1, hermitian),
        upper_u=upper_u, upper_vh=upper_vh,
        lower_u=lower_u, lower_vh=lower_vh,
    )


@dataclass
class HODLROperator:
    """Exact block band plus a recursive low-rank residual."""

    band: sparse.csr_matrix
    root: _Node
    n_dof: int
    half_band: int
    tol: float
    hermitian: bool

    @classmethod
    def from_dense(cls, a: np.ndarray, n_dof: int, *, half_band: int = 1,
                   leaf_cells: int = 2, tol: float = 1e-3,
                   max_rank: int | None = None,
                   hermitian: bool = False) -> "HODLROperator":
        a = np.asarray(a, complex)
        if a.shape[0] != a.shape[1] or a.shape[0] % int(n_dof):
            raise ValueError("HODLR input must be square and block aligned")
        if hermitian:
            scale = max(float(np.linalg.norm(a)), 1e-300)
            if np.linalg.norm(a - a.conj().T) > 1e-10 * scale:
                raise ValueError("hermitian HODLR input is not Hermitian")
            a = 0.5 * (a + a.conj().T)
        near = block_band(a, n_dof, int(half_band))
        residual = a - near
        root = _build_node(residual, 0, a.shape[0], int(leaf_cells) * int(n_dof),
                           float(tol), max_rank, 0, bool(hermitian))
        # A low-rank approximation of an off-diagonal sibling rectangle can
        # spill back into the zeroed near-band corner of that rectangle.  The
        # band is meant to be exact, not merely excluded from the SVD error, so
        # cancel that spill in the explicitly stored band.  This keeps the
        # total represented operator equal to roundoff with ``a`` on every retained
        # block diagonal while leaving the far field low rank.
        approx_residual = np.zeros_like(a)
        root.add_dense(approx_residual)
        near -= block_band(approx_residual, n_dof, int(half_band))
        return cls(sparse.csr_matrix(near), root, int(n_dof), int(half_band),
                   float(tol), bool(hermitian))

    @classmethod
    def from_antihermitian(cls, sigma: np.ndarray, n_dof: int, **kw) -> "HODLROperator":
        sigma = np.asarray(sigma, complex)
        scale = max(float(np.linalg.norm(sigma)), 1e-300)
        if np.linalg.norm(sigma + sigma.conj().T) > 1e-9 * scale:
            raise ValueError("self-energy is not anti-Hermitian")
        # This repository uses Sigma^< = -i n Gamma: +i Sigma^< is the
        # positive/Hermitian carrier (spatial_representation.md Sec. 5).
        return cls.from_dense(1j * sigma, n_dof, hermitian=True, **kw)

    @property
    def shape(self) -> tuple[int, int]:
        return self.band.shape

    def to_dense(self) -> np.ndarray:
        out = self.band.toarray()
        self.root.add_dense(out)
        return out

    def antihermitian_dense(self) -> np.ndarray:
        if not self.hermitian:
            raise ValueError("antihermitian_dense needs a Hermitian carrier")
        return -1j * self.to_dense()

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, complex)
        out = self.band @ x
        self.root.add_apply(x, out)
        return out

    def ranks_by_level(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        self.root.ranks_by_level(out)
        return out

    @property
    def max_rank(self) -> int:
        ranks = [r for vals in self.ranks_by_level().values() for r in vals]
        return max(ranks, default=0)

    @property
    def auxiliary_size(self) -> int:
        return int(sum(r for vals in self.ranks_by_level().values() for r in vals))

    @property
    def stored_scalars(self) -> int:
        return int(self.band.nnz + self.root.stored_scalars())

    def extended_system(self) -> sparse.csc_matrix:
        """Sparse Schur realization of this approximate operator."""
        n = self.shape[0]
        base = self.band + self.root.leaf_sparse(n)
        factors = list(self.root.factors(n))
        if not factors:
            return base.tocsc()
        u = sparse.csc_matrix(np.concatenate([x[0] for x in factors], axis=1))
        vh = sparse.csc_matrix(np.concatenate([x[1] for x in factors], axis=0))
        eye = sparse.eye(u.shape[1], dtype=complex, format="csc")
        return sparse.bmat([[base, u], [vh, -eye]], format="csc")

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, complex)
        n = self.shape[0]
        ext = self.extended_system()
        if ext.shape[0] == n:
            return np.asarray(spsolve(ext, rhs))
        tail = np.zeros((ext.shape[0] - n,) + rhs.shape[1:], complex)
        full_rhs = np.concatenate((rhs, tail), axis=0)
        return np.asarray(spsolve(ext, full_rhs))[:n]


@dataclass(frozen=True)
class PositiveFactor:
    """Dense congruence control arm for a PSD matrix.

    This intentionally reports its storage.  It answers whether positivity can
    be restored by an output factor without pretending that the factor is
    automatically hierarchically compressible.
    """

    factor: np.ndarray

    @classmethod
    def from_operator(cls, op: HODLROperator, *, reference_tol: float = 1e-11):
        a = 0.5 * (op.to_dense() + op.to_dense().conj().T)
        vals, vecs = np.linalg.eigh(a)
        scale = max(float(np.max(np.abs(vals))), 1.0)
        if vals.min() < -float(reference_tol) * scale:
            raise ValueError(
                "PositiveFactor refuses a materially indefinite reference")
        keep = vals > float(reference_tol) * scale
        return cls(vecs[:, keep] * np.sqrt(vals[keep])[None, :])

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.factor @ (self.factor.conj().T @ x)

    def to_dense(self) -> np.ndarray:
        return self.factor @ self.factor.conj().T

    @property
    def stored_scalars(self) -> int:
        return int(self.factor.size)


def quasiseparable_rank(a: np.ndarray, n_dof: int, tol: float,
                        band: int = 0) -> int:
    n = a.shape[0] // int(n_dof)
    ranks = []
    for k in range(1, n):
        row = min(n, k + int(band)) * n_dof
        ranks.append(numerical_rank(a[row:, :k*n_dof], tol))
    return max(ranks, default=0)


def phase_demodulate(a: np.ndarray, n_dof: int, phases: np.ndarray) -> np.ndarray:
    """Unitary block phase extraction used to demonstrate rank invariance."""
    phases = np.asarray(phases, float)
    if phases.shape != (int(n_dof),):
        raise ValueError("one phase per cell DOF is required")
    n = a.shape[0] // int(n_dof)
    diag = np.concatenate([np.exp(1j * phases * i) for i in range(n)])
    return diag.conj()[:, None] * a * diag[None, :]


def propagating_proxy(n_cells: int, n_dof: int, n_modes: int | None = None,
                      seed: int = 7) -> np.ndarray:
    """PSD damped multi-mode covariance; ``-1j*A`` is a valid lesser leg."""
    rng = np.random.default_rng(seed + 31 * n_cells + n_dof)
    n_modes = int(n_modes or min(6, max(2, n_dof + 1)))
    phases = np.linspace(0.22, 2.45, n_modes)
    decays = np.geomspace(0.015, 0.22, n_modes)
    vecs = rng.normal(size=(n_modes, n_dof)) + 1j * rng.normal(
        size=(n_modes, n_dof))
    vecs /= np.linalg.norm(vecs, axis=1)[:, None]
    out = np.zeros((n_cells*n_dof, n_cells*n_dof), complex)
    for i in range(n_cells):
        for j in range(n_cells):
            dist = abs(i - j)
            block = np.zeros((n_dof, n_dof), complex)
            for k in range(n_modes):
                phase = np.exp(1j * phases[k] * (i - j))
                block += np.exp(-decays[k] * dist) * phase * np.outer(
                    vecs[k], vecs[k].conj())
            out[i*n_dof:(i+1)*n_dof, j*n_dof:(j+1)*n_dof] = block
    out += 0.15 * np.eye(out.shape[0])
    return 0.5 * (out + out.conj().T)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def measure_proxy() -> dict:
    rows = []
    for n_cells in (8, 12, 16, 20):
        for n_dof in (1, 2, 4, 8):
            a = propagating_proxy(n_cells, n_dof)
            sigma = -1j * a
            for tol in TOLS:
                op = HODLROperator.from_antihermitian(
                    sigma, n_dof, half_band=1, leaf_cells=2, tol=tol)
                approx = op.antihermitian_dense()
                rng = np.random.default_rng(100 + n_cells + n_dof)
                x = rng.normal(size=a.shape[0]) + 1j * rng.normal(size=a.shape[0])
                apply_err = _rel(op.apply(x), (1j * sigma) @ x)
                # A shifted system keeps the solve away from a planted pole.
                system = 1.8 * np.eye(a.shape[0]) + 0.07 * a
                sop = HODLROperator.from_dense(
                    system, n_dof, half_band=1, leaf_cells=2, tol=tol,
                    hermitian=True)
                rhs = rng.normal(size=a.shape[0]) + 1j * rng.normal(size=a.shape[0])
                t0 = time.perf_counter()
                got = sop.solve(rhs)
                solve_time = time.perf_counter() - t0
                want = np.linalg.solve(system, rhs)
                positive = PositiveFactor.from_operator(op)
                pa = positive.to_dense()
                phase = phase_demodulate(a, n_dof,
                                         np.linspace(0.2, 1.1, n_dof))
                phase_rank = quasiseparable_rank(phase, n_dof, tol, band=1)
                global_rank = quasiseparable_rank(a, n_dof, tol, band=1)
                rows.append({
                    "kind": "proxy", "n_cells": n_cells, "n_dof": n_dof,
                    "tol": tol, "operator_error": _rel(approx, sigma),
                    "apply_error": apply_err, "solve_error": _rel(got, want),
                    "solve_seconds": solve_time,
                    "global_quasiseparable_rank": global_rank,
                    "phase_quasiseparable_rank": phase_rank,
                    "max_hodlr_rank": op.max_rank,
                    "ranks_by_level": {str(k): v for k, v in op.ranks_by_level().items()},
                    "storage_ratio": op.stored_scalars / a.size,
                    "extended_dimension_ratio": sop.extended_system().shape[0] / a.shape[0],
                    "positive_factor_storage_ratio": positive.stored_scalars / a.size,
                    "positive_min_eigenvalue": float(np.linalg.eigvalsh(pa).min()),
                    "positive_operator_error": _rel(-1j * pa, sigma),
                })
    return {"cases": len(rows), "rows": rows}


def _select_frequency_indices(bed) -> list[int]:
    mag = np.linalg.norm(bed.sigma_lesser.reshape(bed.freqs_thz.size, -1), axis=1)
    live = np.where(np.asarray(bed.pos_mask) & (mag > 1e-5 * mag.max()))[0]
    if live.size == 0:
        return []
    return sorted(set([int(live[0]), int(live[len(live)//2]),
                       int(live[np.argmax(mag[live])])]))


def _retarded_builder():
    """Load the audited dense retarded helper without importing solver.__init__.

    The latter eagerly imports the phonopy input stack, which is intentionally
    absent from the lightweight study environment.  ``retarded.py`` itself is
    a standalone NumPy module.
    """
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "solver/retarded.py"
    spec = importlib.util.spec_from_file_location("_qx_review_retarded", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.build_retarded


def frozen_current(bed, sigma_lesser: np.ndarray,
                   sigma_greater: np.ndarray) -> float:
    """Dimensionless left Meir--Wingreen current for relative comparisons."""
    build_retarded = _retarded_builder()
    sigma_r = build_retarded(sigma_lesser, sigma_greater,
                             bed.freqs_thz, method="fft")
    n = bed.n_d
    total_r = (bed.obc["Sigma_L_R"] + bed.obc["Sigma_R_R"] + sigma_r)
    system = (bed.freqs_thz[:, None, None] ** 2 * np.eye(n)[None]
              - bed.h_d[None] - total_r)
    gr = np.linalg.inv(system)
    ga = gr.conj().transpose(0, 2, 1)
    total_l = (bed.obc["Sigma_L_lesser"] + bed.obc["Sigma_R_lesser"]
               + sigma_lesser)
    total_g = (bed.obc["Sigma_L_greater"] + bed.obc["Sigma_R_greater"]
               + sigma_greater)
    gl, gg = gr @ total_l @ ga, gr @ total_g @ ga
    sl = slice(0, bed.n_dof)
    trace = np.trace(
        bed.obc["Sigma_L_greater"][:, sl, sl] @ gl[:, sl, sl]
        - bed.obc["Sigma_L_lesser"][:, sl, sl] @ gg[:, sl, sl],
        axis1=-2, axis2=-1).real
    return float(np.sum(bed.freqs_thz[bed.pos_mask] * trace[bed.pos_mask])
                 * bed.dw_thz)


def _compress_frequency_stack(stack: np.ndarray, n_dof: int,
                              half_band: int, tol: float) -> np.ndarray:
    out = np.empty_like(stack)
    for iw, sigma in enumerate(stack):
        anti = 0.5 * (sigma - sigma.conj().T)
        out[iw] = HODLROperator.from_antihermitian(
            anti, n_dof, half_band=half_band, leaf_cells=2,
            tol=tol).antihermitian_dense()
    return out


def _reblock_mask(stack: np.ndarray, n_dof: int, cells_per_block: int) -> np.ndarray:
    n = stack.shape[-1] // int(n_dof)
    labels = np.arange(n) // int(cells_per_block)
    out = np.zeros_like(stack)
    for i in range(n):
        for j in range(n):
            if abs(labels[i] - labels[j]) <= 1:
                out[..., i*n_dof:(i+1)*n_dof,
                    j*n_dof:(j+1)*n_dof] = stack[..., i*n_dof:(i+1)*n_dof,
                                                      j*n_dof:(j+1)*n_dof]
    return out


def measure_frozen(paths: list[Path]) -> dict:
    from studies._spatial_bed import FrozenBed

    rows, currents = [], []
    for path in paths:
        bed = FrozenBed.load(path)
        for iw in _select_frequency_indices(bed):
            sigma = bed.sigma_lesser[iw]
            anti = 0.5 * (sigma - sigma.conj().T)
            for half_band in (1, 2, 3):
                for tol in TOLS:
                    op = HODLROperator.from_antihermitian(
                        anti, bed.n_dof, half_band=half_band,
                        leaf_cells=2, tol=tol)
                    approx = op.antihermitian_dense()
                    ref_neg = float(np.linalg.eigvalsh(1j * anti).min())
                    got_neg = float(np.linalg.eigvalsh(1j * approx).min())
                    rows.append({
                        "kind": "frozen", "bed": bed.name,
                        "n_cells": bed.n_slabs, "n_dof": bed.n_dof,
                        "frequency_thz": float(bed.freqs_thz[iw]),
                        "half_band": half_band, "tol": tol,
                        "operator_error": _rel(approx, anti),
                        "global_quasiseparable_rank": quasiseparable_rank(
                            1j * anti, bed.n_dof, tol, band=half_band),
                        "max_hodlr_rank": op.max_rank,
                        "ranks_by_level": {str(k): v for k, v in op.ranks_by_level().items()},
                        "storage_ratio": op.stored_scalars / anti.size,
                        "reference_min_eigenvalue": ref_neg,
                        "compressed_min_eigenvalue": got_neg,
                        "added_negativity": max(0.0, ref_neg - got_neg),
                    })
        reference_current = frozen_current(bed, bed.sigma_lesser,
                                           bed.sigma_greater)
        current_scale = max(abs(reference_current), 1e-300)
        for half_band in (1, 2, 3):
            sl = _compress_frequency_stack(bed.sigma_lesser, bed.n_dof,
                                           half_band, 1e-3)
            sg = _compress_frequency_stack(bed.sigma_greater, bed.n_dof,
                                           half_band, 1e-3)
            got = frozen_current(bed, sl, sg)
            currents.append({
                "bed": bed.name, "method": "hodlr", "half_band": half_band,
                "tol": 1e-3, "reference_current": reference_current,
                "current": got,
                "current_error": abs(got - reference_current) / current_scale,
            })
        for cells in (2, 3, 4):
            sl = _reblock_mask(bed.sigma_lesser, bed.n_dof, cells)
            sg = _reblock_mask(bed.sigma_greater, bed.n_dof, cells)
            got = frozen_current(bed, sl, sg)
            currents.append({
                "bed": bed.name, "method": "reblock", "cells": cells,
                "reference_current": reference_current, "current": got,
                "current_error": abs(got - reference_current) / current_scale,
            })
        for half_band in (1, 2, 3):
            sl = np.stack([block_band(a, bed.n_dof, half_band)
                           for a in bed.sigma_lesser])
            sg = np.stack([block_band(a, bed.n_dof, half_band)
                           for a in bed.sigma_greater])
            got = frozen_current(bed, sl, sg)
            currents.append({
                "bed": bed.name, "method": "hard_band",
                "half_band": half_band, "reference_current": reference_current,
                "current": got,
                "current_error": abs(got - reference_current) / current_scale,
            })
    return {"cases": len(rows), "rows": rows, "currents": currents}


def summarize(proxy: dict, frozen: dict) -> dict:
    rows = proxy["rows"]
    target = [r for r in rows if r["tol"] == 1e-3]
    phase_delta = max(abs(r["phase_quasiseparable_rank"]
                          - r["global_quasiseparable_rank"]) for r in rows)
    return {
        "proxy_cases": proxy["cases"], "frozen_cases": frozen["cases"],
        "target_max_operator_error": max(r["operator_error"] for r in target),
        "target_max_solve_error": max(r["solve_error"] for r in target),
        "target_max_hodlr_rank": max(r["max_hodlr_rank"] for r in target),
        "target_max_storage_ratio": max(r["storage_ratio"] for r in target),
        "phase_rank_max_delta": phase_delta,
        "frozen_max_operator_error": (max(r["operator_error"] for r in frozen["rows"])
                                      if frozen["rows"] else None),
        "frozen_target_max_current_error": (
            max(r["current_error"] for r in frozen.get("currents", [])
                if r["method"] == "hodlr")
            if frozen.get("currents") else None),
    }


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[2]
    default_paths = [root / "phonon/studies/out/spatial_bed/chain_L10.npz",
                     root / "phonon/studies/out/spatial_bed/chain_L12.npz",
                     root / "phonon/studies/out/spatial_bed/chain16_c2e+16.npz"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frozen", type=Path, action="append", default=None)
    ap.add_argument("--no-frozen", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    proxy = measure_proxy()
    paths = [] if a.no_frozen else (a.frozen or [p for p in default_paths if p.exists()])
    frozen = measure_frozen(paths)
    summary = summarize(proxy, frozen)
    result = {"summary": summary, "proxy": proxy, "frozen": frozen}
    print("metric | value")
    for key, value in summary.items():
        print(f"{key} | {value}")
    if a.json is not None:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
