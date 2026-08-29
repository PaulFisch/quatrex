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


def validate_restartable_env(environ: MutableMapping[str, str]) -> None:
    """Fail before a campaign run that promises, but cannot save, a restart."""
    if environ.get("QX_REQUIRE_RESTARTABLE") != "1":
        return
    required = ("QX_SAVE_SIGMA", "QX_SAVE_SIGMA_BEST")
    missing = [name for name in required if not environ.get(name)]
    if environ.get("QX_SIGMA_BEST_LIVE") != "1":
        missing.append("QX_SIGMA_BEST_LIVE=1")
    if missing:
        raise ValueError(
            "restartable campaign run is missing " + ", ".join(missing))


def best_checkpoint_stride(environ: MutableMapping[str, str]) -> int:
    """Return the positive live-best write stride (one preserves legacy)."""
    raw = environ.get("QX_SIGMA_BEST_LIVE_STRIDE", "1")
    try:
        stride = int(raw)
    except ValueError as exc:
        raise ValueError(
            "QX_SIGMA_BEST_LIVE_STRIDE must be a positive integer") from exc
    if stride < 1:
        raise ValueError(
            "QX_SIGMA_BEST_LIVE_STRIDE must be a positive integer")
    return stride


def sigma_restart_terms(
        environ: MutableMapping[str, str]) -> list[tuple[str, float]]:
    """Return the one- or two-state affine restart predictor.

    ``QX_SIGMA_SCALE`` remains the coefficient of the legacy primary state.
    Supplying ``QX_SIGMA_INIT_SECOND`` adds a second distributed snapshot with
    coefficient ``QX_SIGMA_SCALE_SECOND``.  This is useful for a secant
    continuation predictor, for example ``2 Sigma(s_b) - Sigma(s_a)``.  It
    affects only the initial iterate and is deliberately kept out of the
    production configuration surface.
    """
    primary = environ.get("QX_SIGMA_INIT")
    secondary = environ.get("QX_SIGMA_INIT_SECOND")
    if not primary:
        if secondary or "QX_SIGMA_SCALE_SECOND" in environ:
            raise ValueError(
                "QX_SIGMA_INIT_SECOND requires QX_SIGMA_INIT")
        return []
    terms = [(primary, float(environ.get("QX_SIGMA_SCALE", "1.0")))]
    if secondary:
        terms.append((secondary, float(
            environ.get("QX_SIGMA_SCALE_SECOND", "1.0"))))
    elif "QX_SIGMA_SCALE_SECOND" in environ:
        raise ValueError(
            "QX_SIGMA_SCALE_SECOND requires QX_SIGMA_INIT_SECOND")
    return terms
