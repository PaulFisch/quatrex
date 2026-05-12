"""Smoke test for the hiphive convergence sweep.

Builds a tiny synthetic Si-chain "DFT pool" from a known analytic FC2,
runs :func:`run_convergence_check` over two sample sizes, and asserts:

  * Both ``(size, fit_method)`` cells return finite RMSE.
  * The fitted dispersion-max-frequency is within ~30 % of the analytic
    chain band edge.
  * The on-disk artefacts (per-size FCP files, summary JSON, PDF plot)
    are produced.

Intentionally minimal: hiphive on a 1-D periodic chain with
``cutoffs=[3.0]`` (FC2 only) converges fast, so a 10-structure pool
with 4 / 8 subsample sizes is enough to exercise every branch of the
harness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PHONON = Path(__file__).resolve().parents[2] / "phonon"
if str(_PHONON) not in sys.path:
    sys.path.insert(0, str(_PHONON))


@pytest.fixture(scope="module")
def chain_rattled_pool():
    """Build a 10-structure rattled pool from an analytic 8-atom Si chain."""
    import ase
    from ase import Atoms
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    n_atoms = 8
    spacing = 2.35
    c_len = n_atoms * spacing
    cell = np.diag([15.0, 15.0, c_len])
    scaled = np.array([[0.5, 0.5, (i + 0.5) / n_atoms] for i in range(n_atoms)])
    ph_atoms = PhonopyAtoms(
        symbols=["Si"] * n_atoms, cell=cell, scaled_positions=scaled,
    )
    phonon = Phonopy(ph_atoms, supercell_matrix=np.eye(3, dtype=int))

    k = 6.0  # eV/Å² along the chain (PBC NN spring)
    fc2 = np.zeros((n_atoms, n_atoms, 3, 3))
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            d = (i - j) % n_atoms
            if d == 1 or d == n_atoms - 1:
                fc2[i, j] = -k * np.eye(3)
    for i in range(n_atoms):
        fc2[i, i] = -fc2[i, :, :, :].sum(axis=0)
    fc2 = 0.5 * (fc2 + fc2.transpose(1, 0, 3, 2))
    phonon.force_constants = fc2

    # The ideal supercell (= primitive in this fixture) for hiphive.
    ideal = Atoms(
        symbols=["Si"] * n_atoms,
        cell=cell,
        scaled_positions=scaled,
        pbc=True,
    )

    # Synthetic rattled pool: F_i = -Σ_j Φ_{ij} · u_j (harmonic).
    rng = np.random.default_rng(seed=7)
    pool = []
    for _ in range(10):
        du = rng.normal(scale=0.03, size=(n_atoms, 3))  # Å
        # Apply harmonic force law on the ideal positions.
        F = -np.einsum("ijab,jb->ia", fc2, du)  # eV/Å
        rat = Atoms(
            symbols=["Si"] * n_atoms,
            cell=cell,
            positions=ideal.positions + du,
            pbc=True,
        )
        rat.arrays["forces"] = F
        pool.append(rat)

    # Use the supercell PhonopyAtoms (= primitive here) as ``primitive``
    # input for the harness.
    return ph_atoms, ideal, pool


def test_run_convergence_check_smoke(chain_rattled_pool, tmp_path):
    """The harness produces FCPs + summary JSON + plot on the chain."""
    from phonon_inputs.config import ConvergenceConfig, HiphiveConfig
    from phonon_inputs.hiphive_convergence import run_convergence_check

    primitive, ideal, pool = chain_rattled_pool

    hh = HiphiveConfig(
        supercell=[1, 1, 1],
        n_structures=10,
        cutoffs=[3.0],  # FC2-only for speed
        fit_method="least-squares",
        rotational_sum_rule="off",
        convergence=ConvergenceConfig(
            sizes=[4, 8],
            pool_size=10,
            test_fraction=0.25,
            seed=1,
            fit_methods=("least-squares",),
            dispersion_q_mesh=[4, 4, 4],
        ),
    )

    out_dir = tmp_path / "convergence"
    results = run_convergence_check(pool, primitive, ideal, hh, out_dir)

    # Both cells produced.
    assert (4, "least-squares") in results
    assert (8, "least-squares") in results

    for r in results.values():
        assert r.error is None, f"Fit failed: {r.error}"
        assert np.isfinite(r.rmse_train)
        # FC2-only fit on harmonic data should reach numerical zero.
        assert r.rmse_train < 1e-6, (
            f"size={r.size} {r.fit_method}: rmse_train={r.rmse_train:.3e}"
        )
        assert r.dispersion_max_thz > 0
        # Analytic band edge for k=6 eV/Å², m=28.0855 amu, NN chain:
        # ω_max = sqrt(4k/m) ≈ 14.4 THz (with phonopy's VASP_TO_THZ).
        assert 5.0 < r.dispersion_max_thz < 25.0, (
            f"Unphysical max-freq {r.dispersion_max_thz:.2f} THz"
        )

    # On-disk artefacts.
    assert (out_dir / "fcp_n4_least-squares.fcp").exists()
    assert (out_dir / "fcp_n8_least-squares.fcp").exists()
    assert (out_dir / "convergence_summary.json").exists()
    summary = json.loads(
        (out_dir / "convergence_summary.json").read_text()
    )
    assert summary["hiphive_config"]["rotational_sum_rule"] == "off"
    assert len(summary["rows"]) == 2
    assert (out_dir / "convergence_vs_n_structures.png").exists()
    assert (out_dir / "convergence_vs_n_structures.pdf").exists()


def test_rotational_sum_rule_post_fit_reduces_residual(chain_rattled_pool, tmp_path):
    """``rotational_sum_rule='post_fit'`` must strictly decrease the
    rotational residual (or leave it at the noise floor on a chain that
    is already rotationally invariant)."""
    from phonon_inputs.config import ConvergenceConfig, HiphiveConfig
    from phonon_inputs.hiphive_convergence import run_convergence_check

    primitive, ideal, pool = chain_rattled_pool

    hh = HiphiveConfig(
        supercell=[1, 1, 1],
        n_structures=10,
        cutoffs=[3.0],
        fit_method="least-squares",
        rotational_sum_rule="post_fit",
        convergence=ConvergenceConfig(
            sizes=[8],
            pool_size=10,
            test_fraction=0.25,
            seed=1,
            fit_methods=("least-squares",),
            dispersion_q_mesh=[4, 4, 4],
        ),
    )
    results = run_convergence_check(
        pool, primitive, ideal, hh, tmp_path / "rot",
    )
    r = results[(8, "least-squares")]
    assert r.rotational_residual_after <= r.rotational_residual_before + 1e-10, (
        f"Rotational projection did not reduce residual: "
        f"{r.rotational_residual_before:.3e} -> "
        f"{r.rotational_residual_after:.3e}"
    )
