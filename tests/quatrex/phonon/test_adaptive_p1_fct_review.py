import numpy as np

from studies import _adaptive_p1_fct_review as R


def test_adaptive_mesh_refines_curvature_not_linear_background():
    mesh = R.adaptive_mesh(
        lambda x: 0.2 + x + 3.0 / (1.0 + ((x - 1.0) / 0.03)**2),
        0.25, 0, 8, 1e-2, max_level=10)
    levels = np.array([level for level, _ in mesh.leaves])
    centres = np.array([(index + 0.5) * mesh.base_h / 2**level
                        for level, index in mesh.leaves])
    assert levels[np.argmin(np.abs(centres - 1.0))] >= 5
    assert max(levels[np.abs(centres - 1.0) > 0.7], default=0) <= 1

