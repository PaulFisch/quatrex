# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""The pole sector's configuration gates.

Each gate is a refusal, not a warning: they cover the configurations in which
the sector would return a confidently wrong answer rather than a noisy one. The
first test is the one the project convention requires of every optional
numerics feature -- the default must be off and legacy.
"""
from pathlib import Path

import pytest
from pydantic import ValidationError

from quatrex.core.config import PhononConfig, PoleSectorConfig


def _phonon(**pole_kw) -> dict:
    """Minimal valid phonon block with the pole sector ENABLED.

    Always enabled: every gate below is conditioned on ``enabled``, so a helper
    that quietly left it off would make the refusal tests pass vacuously.
    """
    return dict(
        model="negf",
        fc3_path=Path("/nonexistent/fc3.hdf5"),
        pole_sector=PoleSectorConfig(enabled=True, **pole_kw),
    )


# --------------------------------------------------------------------------- #

def test_default_is_off_and_legacy():
    # Deliberately NOT via _phonon(): that helper enables the sector.
    cfg = PhononConfig(model="negf", fc3_path=Path("/nonexistent/fc3.hdf5"))
    assert cfg.pole_sector.enabled is False
    # Nothing else in the phonon block is perturbed by its presence.
    assert cfg.retarded_method == "fft"
    assert cfg.eta_ir_floor_cells == 0.0


def test_disabled_sector_skips_every_gate():
    """A disabled sector must not constrain the rest of the configuration."""
    cfg = PhononConfig(
        model="negf", fc3_path=Path("/nonexistent/fc3.hdf5"),
        retarded_method="half", eta_ir_floor_cells=2.0,
        pole_sector=PoleSectorConfig(enabled=False, q_in=2.0, q_out=1.0),
    )
    assert cfg.pole_sector.enabled is False


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        PoleSectorConfig(enabled=True, not_a_knob=1.0)


# --- cross-field gates ------------------------------------------------------ #

def test_half_retarded_is_refused():
    """Without the KK real part the operator is not causal, so its roots are not
    resonances."""
    with pytest.raises(ValidationError, match="retarded_method='fft'"):
        PhononConfig(retarded_method="half", **_phonon())


def test_ir_floor_is_refused():
    """The floor broadens exactly the modes the sector treats exactly."""
    with pytest.raises(ValidationError, match="eta_ir_floor_cells"):
        PhononConfig(eta_ir_floor_cells=2.0, **_phonon())


def test_cm_subtraction_requires_an_explicit_lower_edge():
    """The CM channel already owns omega -> 0; the two must stay disjoint."""
    with pytest.raises(ValidationError, match="omega_min_thz > 0"):
        PhononConfig(sse_cm_subtraction=True, **_phonon())
    # ... and is fine once the pole window is lifted clear of it.
    cfg = PhononConfig(sse_cm_subtraction=True, **_phonon(omega_min_thz=1.0))
    assert cfg.pole_sector.omega_min_thz == 1.0


def test_pole_window_must_sit_above_the_low_frequency_mask():
    with pytest.raises(ValidationError, match="ABOVE sse_low_freq_mask_thz"):
        PhononConfig(sse_low_freq_mask_thz=1.5, **_phonon(omega_min_thz=1.0))
    cfg = PhononConfig(sse_low_freq_mask_thz=1.5, **_phonon(omega_min_thz=2.0))
    assert cfg.pole_sector.omega_min_thz == 2.0


def test_outgoing_sheet_requires_the_spectral_obc():
    with pytest.raises(ValidationError, match="spectral"):
        PhononConfig(
            obc={"algorithm": "sancho-rubio"}, **_phonon(sheet="outgoing")
        )


def test_incomplete_sector_set_warns():
    """Dropping SR/RS drops real three-phonon processes; that must be loud.

    Only on the ``leg="keldysh"`` route, where the sectors are separate terms
    added beside the ring and switching one off really does remove a diagram.
    """
    with pytest.warns(UserWarning, match="DROPS physical"):
        PhononConfig(**_phonon(sectors="rr_ss", leg="keldysh"))


def test_sector_set_is_inert_on_the_congruence_route():
    """There, the pole enters as a cell-average correction to the ring's own
    leg, so there is no analytic sector to switch off. Silently accepting the
    setting would read as if there still were."""
    with pytest.warns(UserWarning, match="ignored when leg='congruence'"):
        PhononConfig(**_phonon(sectors="rr_ss"))


def test_complete_sector_set_is_silent():
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        PhononConfig(**_phonon(sectors="rr_ss_sr"))


# --- internal consistency --------------------------------------------------- #

def test_hysteresis_gap_is_required():
    with pytest.raises(ValidationError, match="q_out .* must exceed q_in"):
        PoleSectorConfig(enabled=True, q_in=2.0, q_out=1.0)


def test_condition_thresholds_must_be_ordered():
    with pytest.raises(ValidationError, match="condition_reject"):
        PoleSectorConfig(enabled=True, condition_max=1e5, condition_reject=1e3)


def test_fit_window_must_support_the_fit_order():
    with pytest.raises(ValidationError, match="too few for a degree"):
        PoleSectorConfig(enabled=True, delta_fit_order=8, delta_fit_window_cells=2)


def test_pole_window_must_be_ordered():
    with pytest.raises(ValidationError, match="omega_max_thz"):
        PoleSectorConfig(enabled=True, omega_min_thz=10.0, omega_max_thz=5.0)
