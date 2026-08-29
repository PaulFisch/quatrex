import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[3] / "phonon/studies/_si_conventional_fcp.py"
SPEC = importlib.util.spec_from_file_location("si_conventional_fcp", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CONVENTIONAL_TRANSFORM = MODULE.CONVENTIONAL_TRANSFORM
conventional_cell = MODULE.conventional_cell
folded_primitive_q_points = MODULE.folded_primitive_q_points
validate_gamma_folding = MODULE.validate_gamma_folding
materialise_dense = MODULE.materialise_dense


FCP = Path("phonon/reaps/hiphive_prim/fcp.fcp")


@pytest.fixture(scope="module")
def fcp():
    hiphive = pytest.importorskip("hiphive")
    return hiphive.ForceConstantPotential.read(str(FCP))


def test_primitive_to_conventional_geometry(fcp):
    primitive = fcp.primitive_structure
    conventional = conventional_cell(primitive)

    assert round(np.linalg.det(CONVENTIONAL_TRANSFORM)) == 4
    assert len(primitive) == 2
    assert len(conventional) == 8
    np.testing.assert_allclose(
        conventional.cell.lengths(), conventional.cell.lengths()[0], atol=1.0e-10
    )
    np.testing.assert_allclose(conventional.cell.angles(), 90.0, atol=1.0e-10)
    assert conventional.get_volume() == pytest.approx(
        4.0 * primitive.get_volume(), rel=1.0e-12
    )


def test_conventional_gamma_is_four_folded_primitive_points(fcp):
    primitive = fcp.primitive_structure
    conventional = conventional_cell(primitive)
    q_points = folded_primitive_q_points(primitive, conventional)
    expected = np.array(
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    )
    assert q_points.shape == (4, 3)
    for q_point in expected:
        assert np.any(np.all(np.isclose(q_points, q_point, atol=1.0e-10), axis=1))


def test_same_fcp_has_invariant_gamma_folding(fcp):
    conventional = conventional_cell(fcp.primitive_structure)
    result = validate_gamma_folding(
        fcp, conventional, primitive_repeat=3, conventional_repeat=2
    )
    assert result["maximum_frequency_difference_thz"] < 2.0e-6


def test_dense_fc2_relabelling_preserves_folded_spectrum(fcp, tmp_path):
    h5py = pytest.importorskip("h5py")
    primitive = fcp.primitive_structure
    _, source_supercell = MODULE._phonopy_supercell(primitive, 3)
    source_fc2 = fcp.get_force_constants(source_supercell).get_fc_array(order=2)
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("fc2", data=source_fc2)

    result = materialise_dense(
        source,
        tmp_path / "conventional",
        source_repeat=3,
        target_repeat=2,
        lattice_constant=primitive.cell.lengths()[0] * np.sqrt(2.0),
        orders=(2,),
        compression=None,
        validate_folding=True,
    )
    validation = result["materialisation"]["gamma_folding_validation"]
    assert validation["maximum_frequency_difference_thz"] < 2.0e-6
    assert (
        result["materialisation"]["diagnostics"]["fc2_asr_maximum_ev_a2"]
        < 1.0e-12
    )


def test_dense_fc3_relabelling_matches_direct_fcp(fcp, tmp_path):
    h5py = pytest.importorskip("h5py")
    primitive = fcp.primitive_structure
    _, source_supercell = MODULE._phonopy_supercell(primitive, 2)
    source_constants = fcp.get_force_constants(source_supercell)
    source = tmp_path / "source_fc3.hdf5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("fc2", data=source_constants.get_fc_array(order=2))
        handle.create_dataset("fc3", data=source_constants.get_fc_array(order=3))

    output = tmp_path / "conventional_fc3"
    materialise_dense(
        source,
        output,
        source_repeat=2,
        target_repeat=1,
        lattice_constant=primitive.cell.lengths()[0] * np.sqrt(2.0),
        orders=(2, 3),
        compression=None,
    )
    conventional = conventional_cell(primitive)
    _, target_supercell = MODULE._phonopy_supercell(conventional, 1)
    direct = fcp.get_force_constants(target_supercell)
    with h5py.File(output / "fc3.hdf5", "r") as remapped:
        np.testing.assert_allclose(
            remapped["fc2"][...], direct.get_fc_array(order=2), atol=1.0e-12
        )
        np.testing.assert_allclose(
            remapped["fc3"][...], direct.get_fc_array(order=3), atol=1.0e-12
        )
