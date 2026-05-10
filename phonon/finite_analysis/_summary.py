"""Common shape for the per-analysis summary dicts that the drivers return.

Every ``run_<analysis>`` driver yields a plain dict with at least a
``units`` map keyed by *bare* metric name (no unit suffix in the key
itself). The CLI assembles these into the top-level ``summary.json``.

Using a TypedDict instead of a runtime dataclass keeps per-analysis
returns lightweight (still plain ``dict``s) while documenting the contract.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AnalysisSummary(TypedDict, total=False):
    """Minimal contract: analyses may add any other keys they like.

    ``units`` maps numeric-key names to a unit string (e.g.
    ``{"fc2_max": "eV/Å²"}``). Unit suffixes do not appear in the keys
    themselves — they live here.
    """

    units: dict[str, str]


def make_summary(units: dict[str, str], **fields: Any) -> dict:
    """Convenience constructor: returns ``{"units": units, **fields}``."""
    return {"units": dict(units), **fields}
