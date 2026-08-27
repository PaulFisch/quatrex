# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Tests for ``phonon/scripts/audit_runs.py``.

The audit decides which of ~330 run directories still satisfy the method's
correctness gates (``phonon/docs/run_audit_2026-08.md``). Three of its
resolution rules are where a mistake would be silent rather than loud, so
they are pinned here:

1. ``sse_g_band`` appears in no stored config and in no ``run.npz``, so the
   effective band is recovered from the environment, then from the run
   name, then from the code default for the run's date -- and is then
   clamped to ``n_blocks - 1`` the way the three solver call sites clamp
   it. Reading the unclamped value would call a 2-block device band-3.
2. cells-per-block is measured from the transport-direction lattice length
   against the shortest one seen for the same system. Getting the
   transport axis wrong silently reports a 2-cell block as a 1-cell block.
3. a gate whose input is missing must FAIL, not pass. An unknown
   ``interaction_cutoff`` on a MoS2 run is not evidence that the cutoff was
   fine.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "_audit_runs", ROOT / "phonon/scripts/audit_runs.py")


@pytest.fixture(scope="module")
def audit():
    m = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(m)
    return m


def _rec(audit, **kw):
    rec = {f: "" for f in audit.FIELDS}
    rec.update({"has_solver_log": True, "outcome": "converged",
                "system": "mos2_film", "g_band_eff": 3, "g_band_src": "env",
                "interaction_cutoff": 30.0, "cmsub": "1", "eta": "0.0",
                "kk_percent": "", "read_by_code": ""})
    rec.update(kw)
    return rec


# --- 1. the band ----------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("cnt-L4-gband2", 2),
    ("l16f-g3", 3),
    ("l16f-g1t", 1),
    ("cnt-L10-g2floor", 2),
    ("l10-g2-snap", 2),
    ("mos2f3", None),          # no band in the name
    ("cnt-L10-gband2", 2),
])
def test_band_recovered_from_the_run_name(audit, name, expected) -> None:
    """The tortin campaign recorded the band only in the directory name."""
    assert audit._gband_from_name(name) == expected


def test_name_probe_catches_every_way_a_path_is_spelled(audit) -> None:
    """CL / f"newton-pc-{arm}/run.log" leaves only the stem in the source."""
    probes = audit._name_probes("newton-pc-none-x")
    assert "newton-pc-none-x" in probes
    assert "newton-pc-{" in probes           # the f-string form
    assert "newton-pc-none-{" in probes
    # a short stem must not become a probe: "l4gpu" has no separator early
    # enough, so it can only match in full
    assert audit._name_probes("l4gpu") == ["l4gpu"]


# --- 2. the gates ---------------------------------------------------------

def test_a_clean_run_fails_only_the_gate_no_run_satisfies(audit) -> None:
    rec = audit.classify(_rec(audit), 2)
    assert rec["verdict"] == "keep-current"
    assert rec["reasons"] == ""


def test_missing_inputs_fail_rather_than_pass(audit) -> None:
    """cells_per_block=None is not evidence of a 2-cell block."""
    assert "cells_per_block=unknown" in audit.classify(_rec(audit), None)[
        "reasons"]
    rec = audit.classify(_rec(audit, interaction_cutoff=""), 2)
    assert "h6-cutoff=unknown" in rec["reasons"]
    rec = audit.classify(_rec(audit, g_band_eff="", g_band_src="unknown"), 2)
    assert "gband=unknown" in rec["reasons"]


def test_the_h6_cutoff_gate_is_mos2_only(audit) -> None:
    """Si transports along x, where the fcc cell's 1.37 A extent means the
    10 A box never truncates (bubble_positivity.md Sec. 6.11)."""
    assert "h6-cutoff" in audit.classify(
        _rec(audit, interaction_cutoff=10.0), 2)["reasons"]
    assert "h6-cutoff" not in audit.classify(
        _rec(audit, system="si_film", interaction_cutoff=10.0), 2)["reasons"]


def test_extent_gate_uses_the_solvers_own_one_percent_threshold(audit) -> None:
    assert audit.KK_TOLERANCE_PERCENT == 1.0
    assert "extent-truncated" not in audit.classify(
        _rec(audit, kk_percent="0.8"), 2)["reasons"]
    assert "extent-truncated=2.6%" in audit.classify(
        _rec(audit, kk_percent="2.6"), 2)["reasons"]


def test_a_referenced_directory_is_never_archived(audit) -> None:
    """A committed script reading it outranks every physics verdict."""
    rec = audit.classify(
        _rec(audit, g_band_eff=1, interaction_cutoff=10.0,
             read_by_code="phonon/scripts/figures/x.py"), 1)
    assert rec["verdict"] == "keep-referenced"
    assert "gband=1" in rec["reasons"]       # still recorded, just not moved


def test_the_universal_gate_cannot_become_an_archive_class(audit) -> None:
    """Every run in the corpus fails no-cm-subtraction, so classing on it
    would put the whole corpus in one bucket."""
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "_archive_runs", ROOT / "phonon/scripts/archive_runs.py")
    arch = ilu.module_from_spec(spec)
    spec.loader.exec_module(arch)
    assert arch.classify("no-cm-subtraction") == "other"
    assert arch.classify("no-cm-subtraction;gband=1") == "gband"
    assert arch.classify("h6-cutoff=10;gband=1") == "h6-cutoff"


# --- 3. the geometry ------------------------------------------------------

def test_transport_length_follows_the_configured_axis(audit) -> None:
    """MoS2 transports along z (12.294 A per cell), Si film along x."""
    lattice = [[3.16, 0.0, 0.0], [-1.58, 2.7366, 0.0], [0.0, 0.0, 24.588]]
    assert audit._transport_length(lattice, "z") == pytest.approx(24.588)
    assert audit._transport_length(lattice, "x") == pytest.approx(3.16)
    assert audit._transport_length(None, "z") is None
