"""Conserving near-band plus bidirectional state-space tail study.

This is a private reference prototype for the spatial part of a future
phonon--phonon NEGF solver.  It does not change the production path.

The represented matrix is

    A = B_near + C^+ s^+ + C^- s^-,

where the two states obey forward and backward recurrences.  Eliminating the
states gives the usual sequentially semiseparable (SSS) matrix.  Keeping the
states instead gives a sparse extended Dyson system; it does *not* require a
dense block of width ``d + r_plus + r_minus`` at every elimination step.

The module also contains three structural gates required before this can be
called a conserving SCBA approximation:

* a scalar bubble of two SSS tails closes exactly under a Kronecker-product
  transition;
* a compression basis enriched by a collision covector preserves that moment
  to roundoff;
* compression of a Keldysh factor followed by congruence remains positive.

Run the reduced study used by ``phonon/docs/phph_acceleration_review.md``::

    QTX_ARRAY_MODULE=numpy PYTHONPATH=src:phonon \
      python phonon/studies/_conserving_spatial_tail_review.py
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from quatrex.phonon.experimental.spatial.spatial_operator import SemiSepOperator
from studies._spatial_hierarchy_review import block_band


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BED = ROOT / "phonon/studies/out/spatial_bed/chain_L12.npz"
DEFAULT_OUT = ROOT / "phonon/studies/out/conserving_spatial_tail_review.json"


def relative_error(got: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(got - reference)
                 / max(float(np.linalg.norm(reference)), 1e-300))


def _block_diag(diag: np.ndarray) -> sparse.csr_matrix:
    """Sparse block diagonal from ``(N,d,d)`` blocks."""
    return sparse.block_diag(list(np.asarray(diag)), format="csr")


def _hermitian_tail_from_dense(a: np.ndarray, n_dof: int, *, tol: float,
                               rank: int | None = None) -> SemiSepOperator:
    """SSS realisation whose upper triangle is exactly the lower adjoint."""
    a = 0.5 * (np.asarray(a, complex) + np.asarray(a, complex).conj().T)
    lower = SemiSepOperator.from_dense(a, n_dof, tol=tol, rank=rank)
    return SemiSepOperator(
        n_cells=lower.n_cells,
        n_dof=lower.n_dof,
        diag=0.5 * (lower.diag
                    + lower.diag.conj().transpose(0, 2, 1)),
        a_plus=lower.a_plus,
        b_plus=lower.b_plus,
        c_plus=lower.c_plus,
        a_minus=lower.a_plus.conj().transpose(0, 2, 1),
        b_minus=lower.c_plus.conj().transpose(0, 2, 1),
        c_minus=lower.b_plus.conj().transpose(0, 2, 1),
    )


@dataclass
class BandSemiSepOperator:
    """Exact block band plus a bidirectional SSS approximation of the tail.

    ``base`` contains the requested exact band and cancels any low-rank spill
    of the tail approximation back into that band.  Therefore every retained
    near block is bitwise independent of the tail tolerance.
    """

    base: sparse.csr_matrix
    tail: SemiSepOperator
    half_band: int
    hermitian: bool = False

    @classmethod
    def from_dense(cls, a: np.ndarray, n_dof: int, *, half_band: int = 1,
                   tol: float = 1e-3, rank: int | None = None,
                   hermitian: bool = False) -> "BandSemiSepOperator":
        a = np.asarray(a, complex)
        if a.shape[0] != a.shape[1] or a.shape[0] % int(n_dof):
            raise ValueError("matrix must be square and block aligned")
        if hermitian:
            scale = max(float(np.linalg.norm(a)), 1e-300)
            if np.linalg.norm(a - a.conj().T) > 1e-10 * scale:
                raise ValueError("Hermitian construction received a non-Hermitian matrix")
            a = 0.5 * (a + a.conj().T)

        near = block_band(a, int(n_dof), int(half_band))
        residual = a - near
        if hermitian:
            tail = _hermitian_tail_from_dense(
                residual, int(n_dof), tol=float(tol), rank=rank)
        else:
            tail = SemiSepOperator.from_dense(
                residual, int(n_dof), tol=float(tol), rank=rank)

        # Truncated corner bases may reconstruct non-zero entries inside the
        # zeroed band.  Cancel only those entries in the explicit base.  The
        # represented near field is then exactly the input near field.
        spill = block_band(tail.to_dense(), int(n_dof), int(half_band))
        base = sparse.csr_matrix(near - spill)
        return cls(base, tail, int(half_band), bool(hermitian))

    @classmethod
    def from_antihermitian(cls, sigma: np.ndarray, n_dof: int, **kw
                           ) -> "BandSemiSepOperator":
        """Represent the Hermitian carrier ``+i Sigma``.

        The physical anti-Hermitian approximation is ``-i * to_dense()``.
        """
        sigma = np.asarray(sigma, complex)
        scale = max(float(np.linalg.norm(sigma)), 1e-300)
        if np.linalg.norm(sigma + sigma.conj().T) > 1e-9 * scale:
            raise ValueError("self-energy is not anti-Hermitian")
        return cls.from_dense(1j * sigma, n_dof, hermitian=True, **kw)

    @property
    def shape(self) -> tuple[int, int]:
        return self.base.shape

    @property
    def rank(self) -> tuple[int, int]:
        return self.tail.rank

    @property
    def stored_scalars(self) -> int:
        arrays = (
            self.tail.diag, self.tail.a_plus, self.tail.b_plus,
            self.tail.c_plus, self.tail.a_minus, self.tail.b_minus,
            self.tail.c_minus,
        )
        return int(self.base.nnz + sum(x.size for x in arrays))

    def to_dense(self) -> np.ndarray:
        return self.base.toarray() + self.tail.to_dense()

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.base @ x) + self.tail.apply(x)

    def extended_system(self) -> sparse.csc_matrix:
        """Sparse recurrence realisation whose Schur complement is ``self``.

        Unknowns are ``[x, s_plus, s_minus]``.  The state equations are

        ``s_i = A^+_{i-1} s_{i-1} + B^+_{i-1} x_{i-1}``, ``s_0=0``;

        ``t_i = A^-_{i+1} t_{i+1} + B^-_{i+1} x_{i+1}``, ``t_{N-1}=0``.
        """
        n = self.tail.n_cells
        d = self.tail.n_dof
        rp, rm = self.rank
        nd = n * d
        ns, nt = n * rp, n * rm
        total = nd + ns + nt
        out = sparse.lil_matrix((total, total), dtype=complex)

        # The explicit diagonal of the SSS tail is not generated by either
        # recurrence and therefore belongs in the physical block.
        out[:nd, :nd] = self.base + _block_diag(self.tail.diag)

        s0 = nd
        t0 = nd + ns
        eye_p = np.eye(rp, dtype=complex)
        eye_m = np.eye(rm, dtype=complex)
        for i in range(n):
            xi = slice(i*d, (i+1)*d)
            if rp:
                si = slice(s0 + i*rp, s0 + (i+1)*rp)
                out[xi, si] = self.tail.c_plus[i]
                out[si, si] = eye_p
                if i:
                    sp = slice(s0 + (i-1)*rp, s0 + i*rp)
                    xp = slice((i-1)*d, i*d)
                    out[si, sp] = -self.tail.a_plus[i-1]
                    out[si, xp] = -self.tail.b_plus[i-1]
            if rm:
                ti = slice(t0 + i*rm, t0 + (i+1)*rm)
                out[xi, ti] = self.tail.c_minus[i]
                out[ti, ti] = eye_m
                if i < n - 1:
                    tn = slice(t0 + (i+1)*rm, t0 + (i+2)*rm)
                    xn = slice((i+1)*d, (i+2)*d)
                    out[ti, tn] = -self.tail.a_minus[i+1]
                    out[ti, xn] = -self.tail.b_minus[i+1]
        return out.tocsc()

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, complex)
        flat = rhs.ndim == 1
        if flat:
            rhs = rhs[:, None]
        nd = self.shape[0]
        if rhs.shape[0] != nd:
            raise ValueError(f"rhs has {rhs.shape[0]} rows, expected {nd}")
        ext = self.extended_system()
        padded = np.zeros((ext.shape[0], rhs.shape[1]), complex)
        padded[:nd] = rhs
        answer = splu(ext).solve(padded)[:nd]
        return answer[:, 0] if flat else answer


def scalar_hadamard_product(a: SemiSepOperator,
                            b: SemiSepOperator) -> SemiSepOperator:
    """Exact SSS closure for the entrywise product of scalar operators.

    This is the fixed-frequency spatial algebra inside one bubble term.  In
    the block/FC3 case the Kronecker states are followed by vertex projection;
    the transition itself remains ``A_1 kron A_2``.
    """
    if a.n_dof != 1 or b.n_dof != 1 or a.n_cells != b.n_cells:
        raise ValueError("scalar_hadamard_product needs matching scalar operators")
    n = a.n_cells

    def kron_cells(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.stack([np.kron(x[i], y[i]) for i in range(n)])

    return SemiSepOperator(
        n_cells=n,
        n_dof=1,
        diag=a.diag * b.diag,
        a_plus=kron_cells(a.a_plus, b.a_plus),
        b_plus=kron_cells(a.b_plus, b.b_plus),
        c_plus=kron_cells(a.c_plus, b.c_plus),
        a_minus=kron_cells(a.a_minus, b.a_minus),
        b_minus=kron_cells(a.b_minus, b.b_minus),
        c_minus=kron_cells(a.c_minus, b.c_minus),
    )


def moment_preserving_projection(samples: np.ndarray, rank: int,
                                 covector: np.ndarray
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Project columns while retaining ``covector^H samples`` exactly.

    The requested SVD rank is enriched by at most one vector.  In SCBA the
    covector is the discretised collision/energy moment.  This is a frozen-
    basis identity; changing the basis during a nonlinear iteration would
    invalidate a conserving-functional argument.
    """
    x = np.asarray(samples, complex)
    c = np.asarray(covector, complex).reshape(-1)
    if x.ndim != 2 or x.shape[0] != c.size:
        raise ValueError("samples must be (space,samples) and match covector")
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    basis = u[:, :min(int(rank), u.shape[1])]
    c_norm = np.linalg.norm(c)
    if c_norm:
        q = c / c_norm
        q -= basis @ (basis.conj().T @ q)
        q_norm = np.linalg.norm(q)
        if q_norm > 50 * np.finfo(float).eps:
            basis = np.column_stack((basis, q / q_norm))
    return basis @ (basis.conj().T @ x), basis


def positive_factor_projection(factor: np.ndarray, rank: int
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Low-rank congruence ``Z_r Z_r^H`` of a positive Keldysh carrier."""
    u, s, _ = np.linalg.svd(np.asarray(factor, complex), full_matrices=False)
    r = min(int(rank), s.size)
    z = u[:, :r] * s[:r]
    return z, z @ z.conj().T


def _random_semisep(n: int, rank: int, seed: int) -> SemiSepOperator:
    rng = np.random.default_rng(seed)
    # Diagonal transitions are sufficient to plant an exact non-trivial SSS
    # tail while keeping the recurrence stable.
    lam_p = rng.uniform(0.25, 0.85, rank) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, rank))
    lam_m = rng.uniform(0.25, 0.85, rank) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, rank))
    ap = np.broadcast_to(np.diag(lam_p), (n, rank, rank)).copy()
    am = np.broadcast_to(np.diag(lam_m), (n, rank, rank)).copy()
    bp = rng.normal(size=(n, rank, 1)) + 1j*rng.normal(size=(n, rank, 1))
    bm = rng.normal(size=(n, rank, 1)) + 1j*rng.normal(size=(n, rank, 1))
    cp = rng.normal(size=(n, 1, rank)) + 1j*rng.normal(size=(n, 1, rank))
    cm = rng.normal(size=(n, 1, rank)) + 1j*rng.normal(size=(n, 1, rank))
    diag = (0.1 * rng.normal(size=(n, 1, 1))
            + 0.1j * rng.normal(size=(n, 1, 1)))
    return SemiSepOperator(n, 1, diag, ap, bp, cp, am, bm, cm)


def _random_block_semisep(n: int, n_dof: int, rank: int,
                          seed: int) -> SemiSepOperator:
    """Stable planted block SSS operator, as an analytic bubble would emit."""
    rng = np.random.default_rng(seed)
    lam_p = rng.uniform(0.25, 0.82, rank) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, rank))
    lam_m = rng.uniform(0.25, 0.82, rank) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, rank))
    ap = np.broadcast_to(np.diag(lam_p), (n, rank, rank)).copy()
    am = np.broadcast_to(np.diag(lam_m), (n, rank, rank)).copy()
    scale = 0.035 / np.sqrt(max(rank, 1))
    bp = scale * (rng.normal(size=(n, rank, n_dof))
                  + 1j*rng.normal(size=(n, rank, n_dof)))
    bm = scale * (rng.normal(size=(n, rank, n_dof))
                  + 1j*rng.normal(size=(n, rank, n_dof)))
    cp = scale * (rng.normal(size=(n, n_dof, rank))
                  + 1j*rng.normal(size=(n, n_dof, rank)))
    cm = scale * (rng.normal(size=(n, n_dof, rank))
                  + 1j*rng.normal(size=(n, n_dof, rank)))
    diag = np.zeros((n, n_dof, n_dof), complex)
    return SemiSepOperator(n, n_dof, diag, ap, bp, cp, am, bm, cm)


def analytic_tail_study() -> dict[str, float | int | list[int]]:
    """Crossover proxy when generators are produced directly by the bubble.

    Extracting a tail from a dense matrix has an interface-rank penalty.  The
    proposed solver would instead carry the generators from one SCBA step to
    the next.  This planted case measures that distinct regime.
    """
    rng = np.random.default_rng(31)
    n, d, r = 64, 8, 4
    tail = _random_block_semisep(n, d, r, 32)
    tail_dense = tail.to_dense()
    target_near = np.zeros((n*d, n*d), complex)
    for i in range(n):
        for j in range(max(0, i-1), min(n, i+2)):
            block = 0.04 * (rng.normal(size=(d, d))
                            + 1j*rng.normal(size=(d, d)))
            if i == j:
                block += 8.0 * np.eye(d)
            target_near[i*d:(i+1)*d, j*d:(j+1)*d] = block
    # The exact near band replaces, rather than adds to, the analytic tail in
    # that region.  The correction is local and does not change the tail rank.
    base = target_near - block_band(tail_dense, d, 1)
    op = BandSemiSepOperator(sparse.csr_matrix(base), tail, 1)
    represented = op.to_dense()
    rhs = rng.normal(size=n*d) + 1j*rng.normal(size=n*d)
    ext = op.extended_system()
    t0 = time.perf_counter()
    got = op.solve(rhs)
    sparse_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    want = np.linalg.solve(represented, rhs)
    dense_seconds = time.perf_counter() - t0
    return {
        "n_cells": n,
        "n_dof": d,
        "rank": list(op.rank),
        "near_error": relative_error(block_band(represented, d, 1),
                                     target_near),
        "solve_error_vs_represented": relative_error(got, want),
        "stored_over_dense": op.stored_scalars / represented.size,
        "extended_nnz_over_dense": ext.nnz / represented.size,
        "extended_dimension_over_physical": ext.shape[0] / represented.shape[0],
        "sparse_solve_seconds": sparse_seconds,
        "dense_solve_seconds": dense_seconds,
    }


def planted_study() -> dict[str, float | int | list[int]]:
    rng = np.random.default_rng(41)
    n, d, r = 24, 2, 3
    scalar_tail = _random_semisep(n, r, 42).to_dense()
    mix = rng.normal(size=(d, d)) + 1j*rng.normal(size=(d, d))
    far = np.kron(scalar_tail, mix)
    near = np.zeros((n*d, n*d), complex)
    for i in range(n):
        for j in range(max(0, i-1), min(n, i+2)):
            blk = rng.normal(size=(d, d)) + 1j*rng.normal(size=(d, d))
            if i == j:
                blk += 10.0 * np.eye(d)
            near[i*d:(i+1)*d, j*d:(j+1)*d] = blk
    a = near + far - block_band(far, d, 1)
    op = BandSemiSepOperator.from_dense(a, d, half_band=1, tol=1e-12)
    dense_op = op.to_dense()
    rhs = rng.normal(size=n*d) + 1j*rng.normal(size=n*d)
    ext = op.extended_system()

    t0 = time.perf_counter()
    got = op.solve(rhs)
    sparse_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    reference = np.linalg.solve(dense_op, rhs)
    dense_seconds = time.perf_counter() - t0
    return {
        "n_cells": n,
        "n_dof": d,
        "rank": list(op.rank),
        "operator_error": relative_error(dense_op, a),
        "near_error": relative_error(block_band(dense_op, d, 1),
                                     block_band(a, d, 1)),
        "solve_error_vs_represented": relative_error(got, reference),
        "stored_over_dense": op.stored_scalars / a.size,
        "extended_nnz_over_dense": ext.nnz / a.size,
        "extended_dimension_over_physical": ext.shape[0] / a.shape[0],
        "sparse_solve_seconds": sparse_seconds,
        "dense_solve_seconds": dense_seconds,
    }


def structural_study() -> dict[str, float | int | list[int]]:
    # Kronecker closure of the spatial bubble.
    a = _random_semisep(18, 2, 5)
    b = _random_semisep(18, 3, 6)
    product = scalar_hadamard_product(a, b)
    product_error = relative_error(product.to_dense(),
                                   a.to_dense() * b.to_dense())

    # Collision-moment preservation.
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 30)) + 1j*rng.normal(size=(64, 30))
    c = rng.normal(size=64) + 1j*rng.normal(size=64)
    plain_u = np.linalg.svd(x, full_matrices=False)[0][:, :8]
    plain = plain_u @ (plain_u.conj().T @ x)
    conservative, basis = moment_preserving_projection(x, 8, c)
    norm_moment = max(float(np.linalg.norm(c.conj() @ x)), 1e-300)
    plain_defect = float(np.linalg.norm(c.conj() @ (plain - x)) / norm_moment)
    conservative_defect = float(
        np.linalg.norm(c.conj() @ (conservative - x)) / norm_moment)

    # Positive congruence after factor compression.
    z0 = rng.normal(size=(48, 20)) + 1j*rng.normal(size=(48, 20))
    z, carrier = positive_factor_projection(z0, 7)
    min_eig = float(np.linalg.eigvalsh(carrier).min())
    return {
        "bubble_product_error": product_error,
        "bubble_input_ranks": [2, 3],
        "bubble_product_rank": list(product.rank),
        "plain_moment_defect": plain_defect,
        "conservative_moment_defect": conservative_defect,
        "plain_rank": 8,
        "conservative_rank": int(basis.shape[1]),
        "positive_factor_rank": int(z.shape[1]),
        "positive_carrier_min_eigenvalue": min_eig,
    }


def frozen_bed_study(path: Path, *, tol: float = 1e-3,
                     half_band: int = 1) -> dict[str, float | int | list[int] | str]:
    """Approximate one stable Dyson matrix from a frozen SCBA bed."""
    from studies._spatial_bed import FrozenBed

    bed = FrozenBed.load(path)
    candidates = np.flatnonzero(bed.pos_mask)
    scored = []
    for iw in candidates:
        gr = bed.g_retarded[iw]
        cond = float(np.linalg.cond(gr))
        if np.isfinite(cond) and cond < 1e10:
            scored.append((float(np.linalg.norm(bed.sigma_retarded[iw])), iw, cond))
    if not scored:
        raise RuntimeError(f"{bed.name}: no stable positive-frequency Dyson sample")
    _, iw, cond_g = max(scored)
    dyson = np.linalg.inv(bed.g_retarded[iw])
    op = BandSemiSepOperator.from_dense(
        dyson, bed.n_dof, half_band=half_band, tol=tol)
    approx = op.to_dense()

    ext = op.extended_system()
    eye = np.eye(bed.n_d, dtype=complex)
    t0 = time.perf_counter()
    g_approx = op.solve(eye)
    sparse_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    g_represented = np.linalg.inv(approx)
    dense_seconds = time.perf_counter() - t0
    gr = bed.g_retarded[iw]
    d = bed.n_dof
    selected = np.s_[0:d, -d:]

    # Keldysh symmetry gate on the same sample.  Frozen references have small
    # numerical defects, so compare with their explicitly anti-Hermitian part.
    sigma_l = 0.5 * (bed.sigma_lesser[iw]
                     - bed.sigma_lesser[iw].conj().T)
    carrier_op = BandSemiSepOperator.from_antihermitian(
        sigma_l, d, half_band=half_band, tol=tol)
    sigma_l_approx = -1j * carrier_op.to_dense()
    antiherm = relative_error(sigma_l_approx.conj().T, -sigma_l_approx)

    return {
        "bed": bed.name,
        "path": str(path),
        "n_cells": bed.n_slabs,
        "n_dof": d,
        "frequency_index": int(iw),
        "frequency_thz": float(bed.freqs_thz[iw]),
        "condition_g": cond_g,
        "tol": tol,
        "half_band": half_band,
        "rank": list(op.rank),
        "operator_error": relative_error(approx, dyson),
        "green_error_vs_original": relative_error(g_approx, gr),
        "green_error_sparse_vs_represented": relative_error(
            g_approx, g_represented),
        "selected_end_to_end_green_error": relative_error(
            g_approx[selected], gr[selected]),
        "near_error": relative_error(block_band(approx, d, half_band),
                                     block_band(dyson, d, half_band)),
        "stored_over_dense": op.stored_scalars / dyson.size,
        "extended_nnz_over_dense": ext.nnz / dyson.size,
        "extended_dimension_over_physical": ext.shape[0] / dyson.shape[0],
        "sparse_full_inverse_seconds": sparse_seconds,
        "dense_full_inverse_seconds": dense_seconds,
        "keldysh_carrier_rank": list(carrier_op.rank),
        "antihermiticity_error": antiherm,
    }


def run(bed: Path | None = DEFAULT_BED, *, tol: float = 1e-3,
        half_band: int = 1) -> dict:
    out = {
        "planted": planted_study(),
        "analytic_tail": analytic_tail_study(),
        "structure": structural_study(),
    }
    if bed is not None and Path(bed).exists():
        out["frozen"] = frozen_bed_study(
            Path(bed), tol=tol, half_band=half_band)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bed", type=Path, default=DEFAULT_BED)
    ap.add_argument("--no-bed", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--half-band", type=int, default=1)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    result = run(None if args.no_bed else args.bed,
                 tol=args.tol, half_band=args.half_band)
    print(json.dumps(result, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
