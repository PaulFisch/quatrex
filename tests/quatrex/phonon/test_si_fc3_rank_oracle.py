import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_si_fc3_rank_oracle", ROOT / "phonon/studies/_si_fc3_rank_oracle.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_unfolding_bound_detects_planted_contracted_rank():
    rng = np.random.default_rng(4)
    left = rng.standard_normal((5, 2))
    right = rng.standard_normal((3, 7, 2))
    target = np.einsum("jr,mkr->mjk", left, right)

    singular_values, bounds = MODULE.unfolding_lower_bounds(
        target, (1, 2, 8)
    )

    assert singular_values[2] < 1e-12 * singular_values[0]
    assert bounds[1] > 1e-3
    assert bounds[2] < 1e-12
    assert bounds[8] < 1e-12
