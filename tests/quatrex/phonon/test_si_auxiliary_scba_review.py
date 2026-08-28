"""Reduced checks for the real-Si auxiliary rank gate."""

import numpy as np

from studies import _si_auxiliary_scba_review as S
from quatrex.phonon.vertex_factors import VertexFactors


def _case(source_sign=1.0):
    w = np.linspace(0.0, 3.0, 61)
    z = np.array([1.2 - 0.08j])
    u = np.array([[1.0], [0.2], [0.7], [0.4j]], complex)
    source = source_sign * 1j * np.ones((w.size, 1, 1))
    cluster = S.FrozenCluster(
        q=(), z=z, u=u, v=u.copy(), source_lesser=0.4 * source,
        source_greater=0.9 * source, label="proxy")
    return S.FrozenCase(
        name="proxy", frequencies=w, block_sizes=np.array([2, 2]),
        q_shape=(), clusters=[cluster])


def _vertex():
    eye = np.eye(2)
    return VertexFactors(
        D=eye, lambdas=np.ones(2), offsets=np.array([0]),
        UB=eye[None, None].astype(complex),
        UC=eye[None, None].astype(complex),
        q_diff_map=np.zeros((1, 1), dtype=int), nk_shape=(), ansatz="test",
        meta={"support_pairs": [(0, 0)]})


def test_si_gate_closes_one_passive_cluster_through_the_qfold():
    out = S.analyse_case(_case(), vf=_vertex())
    assert out["n_invalid_passive_clusters"] == 0
    assert out["qfold_status"] == "complete"
    assert out["raw_output_states_per_q"]["max"] == 1
    assert out["integrated_physical_rank"]["0.001"]["max"] <= 2
    assert out["input_mode_effective_cells"]["max"] > 1.0


def test_si_gate_refuses_a_materially_nonpassive_source():
    out = S.analyse_case(_case(source_sign=-1.0), vf=_vertex())
    assert out["n_invalid_passive_clusters"] == 1
    assert "refused" in out["qfold_status"]


def test_si_gate_can_screen_a_conservative_external_q_subset():
    base = _case()
    clusters = []
    for q in range(2):
        cl = base.clusters[0]
        clusters.append(S.FrozenCluster(
            q=(q,), z=cl.z, u=cl.u, v=cl.v,
            source_lesser=cl.source_lesser,
            source_greater=cl.source_greater, label=f"q{q}"))
    case = S.FrozenCase(
        name="qproxy", frequencies=base.frequencies,
        block_sizes=base.block_sizes, q_shape=(2,), clusters=clusters)
    vf = _vertex()
    vf.UB = np.repeat(vf.UB, 2, axis=1)
    vf.UC = np.repeat(vf.UC, 2, axis=1)
    vf.q_diff_map = np.array([[0, 1], [1, 0]])
    vf.nk_shape = (2,)
    out = S.analyse_case(case, vf=vf, external_q_count=1)
    assert out["qfold_status"] == "complete"
    assert out["full_external_q_axis"] is False
    assert len(out["external_q_indices"]) == 1
