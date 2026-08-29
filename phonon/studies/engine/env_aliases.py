"""Normalise readable campaign environment names to driver names.

The production-study driver predates the Daint campaign and uses several
compact environment names.  Campaign specifications use more descriptive
spellings.  Keeping the translation in one tested place prevents a submitted
job from silently falling back to its TOML value.
"""

from __future__ import annotations

from collections.abc import MutableMapping


ALIASES = {
    "QX_BUBBLE_BALANCE_CHECK": "QX_BBCHECK",
    "QX_ETA_OBC": "QX_ETAOBC",
    "QX_MIXING": "QX_MIX",
    "QX_MIX_METHOD": "QX_MIXMETHOD",
    "QX_POLE_ENABLED": "QX_POLE",
    "QX_RETARDED_METHOD": "QX_RETARDED",
    "QX_SSE_COM_SUBTRACT": "QX_SSE_CMSUB",
    "QX_SSE_LOWFREQ_MASK": "QX_SSE_LOWMASK",
    "QX_SSE_RAMP_ITERS": "QX_RAMP",
    "QX_SSE_VERTEX_SCALE": "QX_VSCALE",
    "QX_TAU_CHUNK_BYTES": "QX_TAUCHUNK",
}


def normalise_env(environ: MutableMapping[str, str]) -> None:
    """Populate compact driver variables and reject conflicting aliases."""
    for alias, canonical in ALIASES.items():
        if alias not in environ:
            continue
        value = environ[alias]
        if canonical in environ and environ[canonical] != value:
            raise ValueError(
                f"conflicting environment overrides: {alias}={value!r} and "
                f"{canonical}={environ[canonical]!r}")
        environ[canonical] = value
