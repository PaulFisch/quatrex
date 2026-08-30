"""Unit tests for the FC3 tensor-decomposition fitters.

Covers the production-critical properties: exact-rank recovery, ASR
preservation of the reconstruction, structural S2 symmetry, determinism
(no global-RNG leaks), parameter-count formulas, the S3 supercell lift,
mSVD Eckart-Young monotonicity, and the production export contract.

Runs on a tiny mock-phonopy fixture (2-atom primitive, 2x2x2 supercell:
n_dof=6, dim_sc=48) in seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phonon_inputs import fc3_compression as fcc  # noqa: E402

torch = pytest.importorskip("torch", reason="fitters need torch")
pytest.importorskip("tensorly", reason="fitters need tensorly")


# ---------------------------------------------------------------------
# Mock phonopy fixture: 2-atom primitive, diagonal 2x2x2 supercell
# ---------------------------------------------------------------------

class _Cell:
    def __init__(self, masses, positions, cell=None, p2s_map=None):
        self.masses = np.asarray(masses, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        if cell is not None:
            self.cell = np.asarray(cell, dtype=float)
        if p2s_map is not None:
            self.p2s_map = np.asarray(p2s_map, dtype=int)


class _MockPhonon:
    def __init__(self):
        a = 2.0
        prim_cell = a * np.eye(3)
        prim_pos = np.array([[0.0, 0.0, 0.0], [0.6 * a, 0.55 * a, 0.5 * a]])
        sc_mat = np.diag([2, 2, 2])
        # supercell atoms: for each lattice translation, both primitive atoms.
        pos, masses = [], []
        prim_masses = [28.0, 70.0]
        for tx in range(2):
            for ty in range(2):
                for tz in range(2):
                    t = prim_cell.T @ np.array([tx, ty, tz])
                    for p_at in range(2):
                        pos.append(prim_pos[p_at] + t)
                        masses.append(prim_masses[p_at])
        pos = np.asarray(pos)
        # p2s_map: supercell indices of the primitive atoms (translation 0).
        p2s = np.array([0, 1])
        self.primitive = _Cell(prim_masses, prim_pos, prim_cell, p2s)
        self.supercell = _Cell(masses, pos)
        self.supercell_matrix = sc_mat


@pytest.fixture(scope="module")
def phonon():
    return _MockPhonon()


def _physical_fc3(phonon, rng):
    """Random FC3 with the exact physical symmetries: S3 over (atom, cart)
    pairs and lattice-translation invariance. Returned in FULL
    (n_super, n_super, n_super, 3, 3, 3) format."""
    from phonon_inputs.pcp import build_cell_mapping, build_cell_diff_table

    n_super = len(phonon.supercell.masses)
    F = rng.normal(size=(n_super, n_super, n_super, 3, 3, 3))
    # S3 symmetrise over joint (atom, cartesian) legs.
    F = (F
         + F.transpose(0, 2, 1, 3, 5, 4)
         + F.transpose(1, 0, 2, 4, 3, 5)
         + F.transpose(1, 2, 0, 4, 5, 3)
         + F.transpose(2, 0, 1, 5, 3, 4)
         + F.transpose(2, 1, 0, 5, 4, 3)) / 6.0
    # Translation-average: F(s1,s2,s3) <- mean over shifts of the triple.
    sc_to_cell, sc_to_prim, _, n_cells, sc_mat = build_cell_mapping(phonon)
    cell_diff, _ = build_cell_diff_table(n_cells, sc_mat)
    cell_atom = np.full((n_cells, len(phonon.primitive.masses)), -1, int)
    for s in range(n_super):
        cell_atom[sc_to_cell[s], sc_to_prim[s]] = s

    def shift(s, l):
        return cell_atom[cell_diff[sc_to_cell[s], l], sc_to_prim[s]]

    G = np.zeros_like(F)
    for l in range(n_cells):
        idx = np.array([shift(s, l) for s in range(n_super)])
        G += F[np.ix_(idx, idx, idx)]
    return G / n_cells


@pytest.fixture(scope="module")
def target(phonon):
    rng = np.random.default_rng(7)
    fc3 = _physical_fc3(phonon, rng)
    return fcc.build_fc3_target(fc3, phonon)


# ---------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------

def test_target_shapes_and_s2(target):
    assert target.T.shape == (6, 48, 48)
    assert target.s2_residual < 1e-12
    # physical FC3 => exact S2 in (j,k) of the compact rows
    assert np.allclose(target.T, target.T.transpose(0, 2, 1), atol=1e-12)


def test_s3_lift_correctness(target):
    L = target.T_lifted
    assert np.allclose(L, fcc._symmetrise_s3(L), atol=1e-10 * target.target_norm)
    back = fcc._slice_to_ndof(L, target.p2s_map)
    assert np.allclose(back, target.T, atol=1e-12)


# ---------------------------------------------------------------------
# Exact-rank recovery
# ---------------------------------------------------------------------

def _indscal_target(target, rank, rng):
    D = rng.normal(size=(target.n_dof, rank))
    V = fcc.asr_project_factor(rng.normal(size=(target.dim_sc, rank)),
                               target.n_super)
    T = np.einsum("mr,jr,kr->mjk", D, V, V)
    t = fcc.FC3Target(
        T=T, T_lifted=target.T_lifted, T_lifted_sym=target.T_lifted_sym,
        p2s_map=target.p2s_map, nat_prim=target.nat_prim,
        n_super=target.n_super, n_dof=target.n_dof, dim_sc=target.dim_sc,
        target_norm=float(np.linalg.norm(T)))
    return t, D, V


def test_msvd_exact_rank(target):
    rng = np.random.default_rng(3)
    R0 = 4
    U = rng.normal(size=(target.n_dof * target.dim_sc, R0))
    W = rng.normal(size=(R0, target.dim_sc))
    T = (U @ W).reshape(target.n_dof, target.dim_sc, target.dim_sc)
    t = fcc.FC3Target(
        T=T, T_lifted=target.T_lifted, T_lifted_sym=target.T_lifted_sym,
        p2s_map=target.p2s_map, nat_prim=target.nat_prim,
        n_super=target.n_super, n_dof=target.n_dof, dim_sc=target.dim_sc,
        target_norm=float(np.linalg.norm(T)))
    res = fcc.fit_msvd(t, rank=R0, enforce_asr=False)
    assert res.rel_err < 1e-12


def test_indscal_exact_rank(target):
    rng = np.random.default_rng(5)
    t, _, _ = _indscal_target(target, rank=3, rng=rng)
    res = fcc.fit_indscal(t, rank=3, n_restarts=4, seed=0)
    assert res.rel_err < 1e-6


def test_s2cp_pairing_is_exact_orthogonal_projection(target):
    rng = np.random.default_rng(15)
    rank = 3
    A = rng.normal(size=(target.n_dof, rank))
    B = fcc.asr_project_factor(
        rng.normal(size=(target.dim_sc, rank)), target.n_super
    )
    C = fcc.asr_project_factor(
        rng.normal(size=(target.dim_sc, rank)), target.n_super
    )
    lam = rng.normal(size=rank)
    cp = fcc.CompressionResult(
        name="CP", rank=rank,
        n_params=fcc.n_params_cp(rank, target.n_dof, target.dim_sc),
        rel_err=np.nan, fit_time_s=0.0,
        factors={"A": A, "B": B, "C": C, "lambdas": lam},
    )

    raw = fcc.reconstruct_cp(cp, target)
    paired = fcc.symmetrise_cp_result(cp, target)
    reconstructed = fcc.reconstruct(paired, target)

    np.testing.assert_allclose(
        reconstructed, 0.5 * (raw + raw.transpose(0, 2, 1)),
        rtol=0, atol=2e-13 * np.abs(raw).max(),
    )
    np.testing.assert_allclose(
        reconstructed, reconstructed.transpose(0, 2, 1),
        rtol=0, atol=2e-13 * np.abs(raw).max(),
    )
    assert paired.rank == 2 * rank
    assert paired.info["paired_base_rank"] == rank
    asr = fcc.asr_residual(reconstructed, target.n_super)
    assert asr["leg_j"] < 1e-12 * asr["norm"]
    assert asr["leg_k"] < 1e-12 * asr["norm"]

    fcc.annotate_result(paired, target)
    exported = fcc.export_production_factors(paired, target)
    assert exported["meta"]["method"] == "S2CP"
    assert exported["meta"]["paired_base_rank"] == rank
    np.testing.assert_allclose(
        np.einsum(
            "r,mr,jr,kr->mjk", exported["lambdas"], exported["A"],
            exported["B"], exported["C"], optimize=True,
        ),
        reconstructed,
        rtol=0, atol=2e-13 * np.abs(raw).max(),
    )


def test_s2cp_rejects_odd_final_rank(target):
    with pytest.raises(ValueError, match="positive even"):
        fcc.fit_s2cp(target, rank=3, n_restarts=1)


def test_msvd_monotone(target):
    errs = [fcc.fit_msvd(target, rank=R, enforce_asr=False).rel_err
            for R in (2, 4, 8)]
    assert errs[0] >= errs[1] >= errs[2]


# ---------------------------------------------------------------------
# ASR preservation
# ---------------------------------------------------------------------

@pytest.mark.parametrize("method,rank", [
    ("mSVD", 4), ("CP", 4), ("S2CP", 4), ("INDSCAL", 4),
    ("HOSVD", (3, 6)), ("Waring", 4),
])
def test_asr_preserved(target, phonon, method, rank):
    # break ASR on purpose in the input
    rng = np.random.default_rng(11)
    T_bad = target.T + 0.1 * rng.normal(size=target.T.shape)
    t = fcc.FC3Target(
        T=T_bad, T_lifted=target.T_lifted, T_lifted_sym=target.T_lifted_sym,
        p2s_map=target.p2s_map, nat_prim=target.nat_prim,
        n_super=target.n_super, n_dof=target.n_dof, dim_sc=target.dim_sc,
        target_norm=float(np.linalg.norm(T_bad)))
    fitter = fcc.FITTERS[method]
    kwargs = {"enforce_asr": True}
    if method == "HOSVD":
        res = fitter(t, R1=rank[0], R2=rank[1], **kwargs)
    elif method in ("CP", "S2CP", "INDSCAL"):
        res = fitter(t, rank=rank, n_restarts=1, seed=0, **kwargs)
    elif method == "Waring":
        res = fitter(t, rank=rank, n_restarts=1, seed=0, **kwargs)
    else:
        res = fitter(t, rank=rank, **kwargs)
    T_hat = fcc.reconstruct(res, t)
    asr = fcc.asr_residual(T_hat, t.n_super)
    assert asr["leg_j"] < 1e-10 * asr["norm"]
    assert asr["leg_k"] < 1e-10 * asr["norm"]


# ---------------------------------------------------------------------
# Structural S2 of the symmetric reconstructions
# ---------------------------------------------------------------------

@pytest.mark.parametrize("method,rank", [
    ("INDSCAL", 4), ("S2CP", 4), ("HOSVD", (3, 6)), ("Waring", 4),
])
def test_s2_structural(target, method, rank):
    fitter = fcc.FITTERS[method]
    if method == "HOSVD":
        res = fitter(target, R1=rank[0], R2=rank[1])
    else:
        res = fitter(target, rank=rank, n_restarts=1, seed=0)
    T_hat = fcc.reconstruct(res, target)
    assert np.allclose(T_hat, T_hat.transpose(0, 2, 1),
                       atol=1e-13 * (target.target_norm or 1.0))


# ---------------------------------------------------------------------
# Determinism (no global RNG leaks)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("method,rank", [
    ("CP", 3), ("S2CP", 4), ("INDSCAL", 3),
])
def test_determinism(target, method, rank):
    fitter = fcc.FITTERS[method]
    np.random.seed(12345)
    probe_before = np.random.random()
    np.random.seed(12345)
    r1 = fitter(target, rank=rank, n_restarts=2, seed=1)
    probe_after = np.random.random()
    r2 = fitter(target, rank=rank, n_restarts=2, seed=1)
    for k in r1.factors:
        np.testing.assert_array_equal(r1.factors[k], r2.factors[k])
    # the fitter must not consume/reset the global stream
    assert probe_before == probe_after


# ---------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------

def test_param_counts(target):
    R = 4
    res = fcc.fit_indscal(target, rank=R, n_restarts=1, seed=0)
    stored = sum(v.size for v in res.factors.values())
    assert res.n_params == stored
    res = fcc.fit_cp(target, rank=R, n_restarts=1, seed=0)
    stored = res.factors["A"].size + res.factors["B"].size + \
        res.factors["C"].size + res.factors["lambdas"].size
    assert res.n_params == stored


# ---------------------------------------------------------------------
# Production driver + export contract
# ---------------------------------------------------------------------

def test_fit_production_and_export(target):
    res = fcc.fit_production(target, rank=4, ansatz="INDSCAL",
                             n_restarts=2, seed=0)
    assert "asr" in res.info and "production" in res.info
    exp = fcc.export_production_factors(res, target)
    D, V = exp["D"], exp["V"]
    assert D.shape == (target.n_dof, 4) and V.shape == (target.dim_sc, 4)
    T_exp = np.einsum("mr,jr,kr->mjk", D, V, V)
    T_ref = fcc.reconstruct_indscal(res, target)
    np.testing.assert_allclose(T_exp, T_ref, atol=1e-13)
    # columns sorted by descending contribution
    contrib = np.linalg.norm(D, axis=0) * np.linalg.norm(V, axis=0) ** 2
    assert np.all(np.diff(contrib) <= 1e-12)


def test_s2cp_production_gate_and_export(target):
    res = fcc.fit_production(
        target, rank=4, ansatz="S2CP", n_restarts=1, seed=0,
        max_iter=30, lbfgs_iters=30,
    )
    assert res.info["s2_recon"] < 1e-12
    exp = fcc.export_production_factors(res, target)
    assert exp["meta"]["method"] == "S2CP"
    assert exp["meta"]["paired_base_rank"] == 2
    reconstructed = np.einsum(
        "r,mr,jr,kr->mjk", exp["lambdas"], exp["A"], exp["B"], exp["C"],
        optimize=True,
    )
    np.testing.assert_allclose(
        reconstructed, reconstructed.transpose(0, 2, 1),
        rtol=0, atol=1e-12 * np.abs(reconstructed).max(),
    )


def test_fit_production_gate(target):
    # sabotage: a fitter that ignores ASR must be caught by the gate
    rng = np.random.default_rng(2)
    T_bad = target.T + 0.5 * rng.normal(size=target.T.shape)
    t = fcc.FC3Target(
        T=T_bad, T_lifted=target.T_lifted, T_lifted_sym=target.T_lifted_sym,
        p2s_map=target.p2s_map, nat_prim=target.nat_prim,
        n_super=target.n_super, n_dof=target.n_dof, dim_sc=target.dim_sc,
        target_norm=float(np.linalg.norm(T_bad)))
    with pytest.raises(RuntimeError, match="not conserving"):
        fcc.fit_production(t, rank=2, ansatz="INDSCAL", n_restarts=1,
                           seed=0, enforce_asr=False)


# ---------------------------------------------------------------------
# Mass-weighted ASR (two-species): the physical sum rule on the
# mass-weighted target is sum_s sqrt(m_s) M[.., 3s+b, ..] = 0.
# ---------------------------------------------------------------------

def test_weighted_projector_reduces_to_plain_for_equal_masses():
    rng = np.random.default_rng(3)
    n_super = 8
    V = rng.normal(size=(3 * n_super, 5))
    plain = fcc.asr_project_factor(V, n_super)
    unif = fcc.asr_project_factor(V, n_super, weights=np.full(n_super, 2.7))
    np.testing.assert_allclose(unif, plain, rtol=0, atol=1e-14)


def test_weighted_projector_zeroes_weighted_sum_and_is_idempotent():
    rng = np.random.default_rng(4)
    n_super = 8
    w = np.sqrt(rng.uniform(20.0, 90.0, size=n_super))
    V = rng.normal(size=(3 * n_super, 5))
    P = fcc.asr_project_factor(V, n_super, weights=w)
    s = (w[:, None, None] * P.reshape(n_super, 3, -1)).sum(axis=0)
    assert np.abs(s).max() < 1e-12
    P2 = fcc.asr_project_factor(P, n_super, weights=w)
    np.testing.assert_allclose(P2, P, rtol=0, atol=1e-13)


@pytest.mark.parametrize("method", ["INDSCAL", "CP", "S2CP"])
def test_weighted_asr_preserved_two_species(target, method):
    # Two-species weights (the mock has masses 28/70); a fit with
    # asr_weights must satisfy the PHYSICAL (weighted) sum rule, which
    # differs measurably from the plain one.
    rng = np.random.default_rng(12)
    w = np.sqrt(np.tile([28.0, 70.0], target.n_super // 2))
    T_bad = target.T + 0.1 * rng.normal(size=target.T.shape)
    t = fcc.FC3Target(
        T=T_bad, T_lifted=target.T_lifted, T_lifted_sym=target.T_lifted_sym,
        p2s_map=target.p2s_map, nat_prim=target.nat_prim,
        n_super=target.n_super, n_dof=target.n_dof, dim_sc=target.dim_sc,
        target_norm=float(np.linalg.norm(T_bad)), asr_weights=w)
    res = fcc.FITTERS[method](t, rank=4, n_restarts=1, seed=0,
                              enforce_asr=True)
    T_hat = fcc.reconstruct(res, t)
    asr_w = fcc.asr_residual(T_hat, t.n_super, weights=w)
    assert asr_w["leg_j"] < 1e-10 * asr_w["norm"]
    assert asr_w["leg_k"] < 1e-10 * asr_w["norm"]
    # the plain (unweighted) sum must NOT vanish -- the two null spaces
    # genuinely differ for two species
    asr_plain = fcc.asr_residual(T_hat, t.n_super)
    assert asr_plain["leg_j"] > 1e-6 * asr_plain["norm"]
