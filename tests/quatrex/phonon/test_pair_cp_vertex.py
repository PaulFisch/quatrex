import importlib.util
from pathlib import Path

import numpy as np

from quatrex.phonon.vertex_factors import VertexFactors

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_pair_cp_vertex", ROOT / "phonon/studies/_pair_cp_vertex.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_pair_cp_vertex_is_exact_q_resolved_s2_projection():
    rng = np.random.default_rng(8)
    dof, rank, nq = 4, 3, 3
    offsets = np.array([-1, 0, 1])
    vertex = VertexFactors(
        D=rng.normal(size=(dof, rank)),
        lambdas=rng.normal(size=rank),
        offsets=offsets,
        UB=(rng.normal(size=(3, nq, dof, rank))
            + 1j * rng.normal(size=(3, nq, dof, rank))),
        UC=(rng.normal(size=(3, nq, dof, rank))
            + 1j * rng.normal(size=(3, nq, dof, rank))),
        q_diff_map=np.array(
            [[(i - j) % nq for j in range(nq)] for i in range(nq)]
        ),
        nk_shape=(nq,), ansatz="CP",
        meta={"rel_err": 0.2, "s2_recon": 0.3,
              "qfold_sample_aggregate_rel_err": 0.4},
    )

    paired = MODULE.pair_cp_vertex(vertex)
    direct = paired.reconstruct_block(1, 2, -1, 1)
    raw = vertex.reconstruct_block(1, 2, -1, 1)
    transposed = vertex.reconstruct_block(2, 1, 1, -1).transpose(0, 2, 1)

    np.testing.assert_allclose(direct, 0.5 * (raw + transposed))
    np.testing.assert_allclose(
        direct,
        paired.reconstruct_block(2, 1, 1, -1).transpose(0, 2, 1),
    )
    assert paired.rank == 2 * rank
    assert paired.meta["method"] == "S2CP"
    assert paired.meta["paired_cp_exact_projection"]
    assert "qfold_sample_aggregate_rel_err" not in paired.meta
    assert paired.meta["base_qfold_sample_diagnostics"] == {
        "qfold_sample_aggregate_rel_err": 0.4
    }
    np.testing.assert_allclose(
        paired.meta["rel_err"], np.sqrt(0.2**2 - 0.25 * 0.3**2)
    )
