# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from types import SimpleNamespace

import numpy as np
import pytest

import quatrex.core.scba as scba_module
import quatrex.phonon.microblocks as microblocks


class PatternBuilt(RuntimeError):
    pass


@pytest.mark.parametrize(("micro_dof", "expected_band"), [(0, 3), (6, 1)])
def test_phonon_sparsity_uses_complete_selected_band(
    monkeypatch, micro_dof, expected_band
):
    config = SimpleNamespace(
        simulation_type="phonon",
        device=SimpleNamespace(num_orbitals_per_atom={}, kpoint_grid=[1, 1, 1]),
        scba=SimpleNamespace(coulomb_screening=False, photon=False, phonon=True),
        phonon=SimpleNamespace(
            interaction_cutoff=0.01,
            sse_g_band=3,
            sse_microblock_dof=micro_dof,
        ),
    )
    monkeypatch.setattr(
        scba_module.Device,
        "load_structure",
        staticmethod(lambda _config: (np.zeros((4, 3)), None, ["X"] * 4)),
    )
    monkeypatch.setattr(
        scba_module, "get_block_sizes", lambda _config, _grid: np.ones(4, int)
    )
    monkeypatch.setattr(
        scba_module,
        "comm",
        SimpleNamespace(rank=0, block=SimpleNamespace(rank=0, size=1)),
    )
    monkeypatch.setattr(
        scba_module,
        "compute_sparsity_pattern",
        lambda *_args, **_kwargs: pytest.fail("phonon transport used a cutoff mask"),
    )
    seen = {}

    def capture(_sizes, band, start_block, end_block):
        seen.update(
            band=band, start_block=start_block, end_block=end_block
        )
        raise PatternBuilt

    monkeypatch.setattr(microblocks, "grouped_band_indices", capture)
    with pytest.raises(PatternBuilt):
        scba_module.SCBAData(config, np.array([0.0]))

    assert seen == {"band": expected_band, "start_block": 0, "end_block": 4}
