"""Sanity-check pipeline configurations against literature-recommended values.

The check table covers the DFT settings (`encut`, `ecutwfc`, kpoints), the
finite-displacement / hiphive settings (`n_structures`, `cutoffs`,
`displacement_distance`), and the supercell/cutoff geometry. Each check
emits a ``CheckResult`` carrying severity, the actual value, and a
recommendation message.

Three exposures (see plan Phase 5):
  1. ``loader.load_system`` calls :func:`validate_config` after parsing
     the YAML and emits ``warnings.warn`` for severity ≥ "warn".
  2. ``python -m finite_analysis validate <config>`` prints a coloured
     pass/warn/fail table.
  3. ``cli.run`` appends the JSON-serialised result list to
     ``summary.json`` under ``parameter_validation``.

The check table is data-driven — adding a check is one entry. Each
checker is a small callable returning ``CheckResult`` (or ``None`` if
the relevant config block is absent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

# --------------------------------------------------------------------------- #
# Reference data (verified against VASP 6.4 and PBE PAW potentials)            #
# --------------------------------------------------------------------------- #

POTCAR_ENMAX_EV: dict[str, float] = {
    # Element : ENMAX (eV) from PAW_PBE POTCARs in VASP 6.4 distribution.
    "H":  250.0,
    "C":  400.0,
    "N":  400.0,
    "O":  400.0,
    "Si": 245.0,
    "Ge": 174.0,
    "Sn": 103.0,
}

NN_DISTANCE_A: dict[str, float] = {
    # Bulk nearest-neighbour distance in Å for diamond-cubic / graphite phases.
    "Si": 2.35,
    "Ge": 2.45,
    "C":  1.42,   # graphite/CNT in-plane
    "Sn": 2.81,
}

QE_RECOMMENDED_ECUTWFC_RY: dict[str, float] = {
    "Si": 60.0,
    "Ge": 80.0,
    "C":  80.0,
    "H":  60.0,
    "Sn": 70.0,
}


Severity = Literal["info", "warn", "error"]


# --------------------------------------------------------------------------- #
# Result types                                                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Check:
    key: str
    message: str
    recommendation: str


@dataclass
class CheckResult:
    check: Check
    passed: bool
    actual: Any
    severity: Severity
    notes: str = ""


# --------------------------------------------------------------------------- #
# Individual checkers                                                         #
# --------------------------------------------------------------------------- #


def _check_vasp_encut(cfg) -> CheckResult | None:
    vasp = getattr(cfg, "vasp", None)
    if vasp is None:
        return None
    pot_map = getattr(vasp, "potcar_map", None)
    encut = getattr(vasp, "encut", None)
    if pot_map is None or encut is None:
        return None
    species = list(pot_map.keys())
    enmax = max(
        (POTCAR_ENMAX_EV.get(sp, 0.0) for sp in species), default=0.0
    )
    target = 1.3 * enmax
    chk = Check(
        key="vasp.encut",
        message=f"ENCUT should be ≥ 1.3 × max(POTCAR ENMAX) = {target:.0f} eV",
        recommendation=(
            f"Set vasp.encut ≥ {target:.0f} eV (max POTCAR ENMAX from "
            f"{species}: {enmax:.0f} eV)."
        ),
    )
    if enmax == 0.0:
        return CheckResult(
            chk, passed=True, actual=encut, severity="info",
            notes="No POTCAR ENMAX known for these species; skipping.",
        )
    if encut >= target:
        return CheckResult(chk, passed=True, actual=encut, severity="info")
    sev: Severity = "error" if encut < enmax else "warn"
    return CheckResult(chk, passed=False, actual=encut, severity=sev)


def _check_qe_ecutwfc(cfg) -> CheckResult | None:
    qe = getattr(cfg, "qe", None)
    if qe is None:
        return None
    ecut = getattr(qe, "ecutwfc", None)
    pp = getattr(qe, "pseudopotentials", None) or {}
    if ecut is None:
        return None
    species = list(pp.keys())
    target = max(
        (QE_RECOMMENDED_ECUTWFC_RY.get(sp, 0.0) for sp in species), default=60.0
    )
    chk = Check(
        key="qe.ecutwfc",
        message=f"ecutwfc should be ≥ {target:.0f} Ry for species {species}",
        recommendation=f"Set qe.ecutwfc ≥ {target:.0f} Ry.",
    )
    if ecut >= target:
        return CheckResult(chk, passed=True, actual=ecut, severity="info")
    sev: Severity = "error" if ecut < 0.7 * target else "warn"
    return CheckResult(chk, passed=False, actual=ecut, severity=sev)


def _periodic_axes(cfg) -> list[int]:
    """Heuristic: an axis is treated as 'vacuum' (non-periodic) if its
    cell length exceeds 12 Å. Returns the axes that look periodic."""
    structure = getattr(cfg, "structure", None)
    if structure is None:
        return [0, 1, 2]
    lat = getattr(structure, "lattice", None)
    if lat is None:
        return [0, 1, 2]
    import numpy as np
    lengths = np.linalg.norm(np.asarray(lat), axis=1)
    return [i for i, L in enumerate(lengths) if L < 12.0]


def _check_kpoints_scf(cfg) -> CheckResult | None:
    src = getattr(cfg, "vasp", None) or getattr(cfg, "qe", None)
    if src is None:
        return None
    kgrid = getattr(src, "kpoints_scf", None)
    if kgrid is None:
        return None
    periodic = _periodic_axes(cfg)
    if not periodic:
        chk = Check(
            key="kpoints_scf",
            message="No periodic axes detected; kpoints_scf should be (1,1,1)",
            recommendation="Set kpoints_scf = [1, 1, 1] for fully finite cells.",
        )
        ok = all(int(k) == 1 for k in kgrid)
        return CheckResult(chk, passed=ok, actual=list(kgrid),
                           severity="info" if ok else "warn")
    chk = Check(
        key="kpoints_scf",
        message="kpoints_scf along each periodic axis should be ≥ 4",
        recommendation=(
            f"Set kpoints_scf[{periodic}] ≥ 4 along periodic axes "
            f"(current: {list(kgrid)})."
        ),
    )
    bad = [(ax, int(kgrid[ax])) for ax in periodic if int(kgrid[ax]) < 4]
    if not bad:
        return CheckResult(chk, passed=True, actual=list(kgrid), severity="info")
    sev: Severity = "error" if any(k < 2 for _, k in bad) else "warn"
    return CheckResult(chk, passed=False, actual=list(kgrid), severity=sev,
                       notes=f"Under-converged axes: {bad}")


def _hiphive_is_active(cfg) -> bool:
    """True iff the YAML actually configured a hiphive run (vs a default)."""
    hh = getattr(cfg, "hiphive", None)
    if hh is None:
        return False
    relax = getattr(cfg, "relax", None)
    fc_method = getattr(relax, "fc_method", None) if relax else None
    return fc_method == "hiphive"


def _check_hiphive_n_structures(cfg) -> CheckResult | None:
    if not _hiphive_is_active(cfg):
        return None
    hh = getattr(cfg, "hiphive", None)
    n_struct = getattr(hh, "n_structures", None)
    if n_struct is None:
        return None
    structure = getattr(cfg, "structure", None)
    n_atoms = len(structure.symbols) if structure else 0
    sc = list(getattr(hh, "supercell", [1, 1, 1]))
    n_super = n_atoms * (sc[0] * sc[1] * sc[2])
    target = max(6, 4 * (n_super + 7) // 8)  # 4 × ceil(n_super/8)
    chk = Check(
        key="hiphive.n_structures",
        message=f"n_structures should be ≥ {target} for {n_super}-atom supercell",
        recommendation=(
            f"Set hiphive.n_structures ≥ {target} (4 × ceil(n_super/8) = "
            f"{target}; supercell has {n_super} atoms)."
        ),
    )
    if n_struct >= target:
        return CheckResult(chk, passed=True, actual=n_struct, severity="info")
    sev: Severity = "warn" if n_struct >= target // 2 else "error"
    return CheckResult(chk, passed=False, actual=n_struct, severity=sev)


def _check_hiphive_cutoffs(cfg) -> CheckResult | None:
    if not _hiphive_is_active(cfg):
        return None
    hh = getattr(cfg, "hiphive", None)
    cutoffs = getattr(hh, "cutoffs", None)
    if cutoffs is None or len(cutoffs) < 2:
        return None
    fc2_cut, fc3_cut = float(cutoffs[0]), float(cutoffs[1])
    chk = Check(
        key="hiphive.cutoffs",
        message="FC2 cutoff should be ≥ 5 Å, FC3 cutoff ≥ 4 Å",
        recommendation="Set hiphive.cutoffs ≥ [5.0, 4.0] (Å).",
    )
    ok = fc2_cut >= 5.0 and fc3_cut >= 4.0
    sev: Severity = "info" if ok else "warn"
    return CheckResult(
        chk, passed=ok, actual=[fc2_cut, fc3_cut], severity=sev,
    )


def _check_displacement_distance(cfg) -> CheckResult | None:
    td = getattr(cfg, "thirdorder", None)
    if td is None:
        return None
    d = getattr(td, "displacement_distance", None)
    if d is None:
        return None
    chk = Check(
        key="thirdorder.displacement_distance",
        message="displacement_distance should be in [0.01, 0.05] Å",
        recommendation=(
            "Use ≈ 0.03 Å (phono3py default). Smaller for very stiff bonds; "
            "larger introduces anharmonic contamination of the harmonic FC."
        ),
    )
    ok = 0.01 <= d <= 0.05
    sev: Severity = "info" if ok else "warn"
    return CheckResult(chk, passed=ok, actual=d, severity=sev)


def _check_thirdorder_cutoff(cfg) -> CheckResult | None:
    td = getattr(cfg, "thirdorder", None)
    if td is None:
        return None
    cut = getattr(td, "cutoff_pair_distance", None)
    if cut is None:
        return None  # phono3py treats None as "all pairs in supercell"
    structure = getattr(cfg, "structure", None)
    species = list(getattr(structure, "symbols", []))
    nn = max(
        (NN_DISTANCE_A.get(sp, 0.0) for sp in species), default=0.0
    )
    target = 1.5 * nn if nn > 0 else 4.0
    chk = Check(
        key="thirdorder.cutoff_pair_distance",
        message=f"cutoff_pair_distance should be ≥ 1.5 × NN = {target:.2f} Å",
        recommendation=(
            f"Set thirdorder.cutoff_pair_distance ≥ {target:.2f} Å "
            f"(NN distance for {species}: {nn:.2f} Å)."
        ),
    )
    if cut >= target:
        return CheckResult(chk, passed=True, actual=cut, severity="info")
    sev: Severity = "error" if cut < nn else "warn"
    return CheckResult(chk, passed=False, actual=cut, severity=sev)


def _check_supercell_size(cfg) -> CheckResult | None:
    src = getattr(cfg, "thirdorder", None) or getattr(cfg, "hiphive", None)
    if src is None:
        return None
    sc = getattr(src, "supercell", None)
    if sc is None:
        return None
    structure = getattr(cfg, "structure", None)
    if structure is None:
        return None
    import numpy as np
    lat = np.asarray(structure.lattice)
    lengths = np.linalg.norm(lat, axis=1)
    extents = np.array(sc) * lengths
    cutoff = 5.0  # FC2 default
    periodic = _periodic_axes(cfg)
    chk = Check(
        key="supercell_extents",
        message=f"Supercell extent along periodic axes should be ≥ FC2 cutoff = {cutoff} Å",
        recommendation=(
            f"Increase supercell along periodic axes {periodic} so "
            f"extent_axis ≥ {cutoff} Å. Current extents: "
            f"{[float(extents[a]) for a in periodic]}."
        ),
    )
    bad = [a for a in periodic if extents[a] < cutoff]
    if not bad:
        return CheckResult(chk, passed=True, actual=list(extents), severity="info")
    return CheckResult(
        chk, passed=False, actual=list(extents), severity="warn",
        notes=f"Under-sized along axes {bad}.",
    )


CHECKERS: list[Callable[[Any], CheckResult | None]] = [
    _check_vasp_encut,
    _check_qe_ecutwfc,
    _check_kpoints_scf,
    _check_hiphive_n_structures,
    _check_hiphive_cutoffs,
    _check_displacement_distance,
    _check_thirdorder_cutoff,
    _check_supercell_size,
]


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def validate_config(cfg) -> list[CheckResult]:
    """Run every checker against ``cfg``, dropping ``None`` results."""
    out: list[CheckResult] = []
    for fn in CHECKERS:
        try:
            res = fn(cfg)
        except Exception as exc:  # noqa: BLE001
            chk = Check(
                key=fn.__name__, message="checker raised", recommendation="",
            )
            out.append(CheckResult(
                chk, passed=False, actual=None, severity="warn",
                notes=f"Checker {fn.__name__} raised: {exc!r}",
            ))
            continue
        if res is not None:
            out.append(res)
    return out


def to_dict(results: Iterable[CheckResult]) -> list[dict]:
    return [
        {
            "key": r.check.key,
            "passed": r.passed,
            "severity": r.severity,
            "actual": r.actual,
            "message": r.check.message,
            "recommendation": r.check.recommendation,
            "notes": r.notes,
        }
        for r in results
    ]


def render_table(results: Iterable[CheckResult], *, color: bool = True) -> str:
    results = list(results)
    if not results:
        return "(no checks applied)"

    if color:
        sev_glyph = {"info": "\033[32m✓\033[0m", "warn": "\033[33m⚠\033[0m",
                     "error": "\033[31m✗\033[0m"}
    else:
        sev_glyph = {"info": "OK ", "warn": "WARN", "error": "FAIL"}

    rows = [
        f"{sev_glyph[r.severity]}  {r.check.key:32s}  "
        f"actual={r.actual!r}"
        for r in results
    ]
    summary = "\n".join(rows)
    counts = {s: sum(1 for r in results if r.severity == s)
              for s in ("info", "warn", "error")}
    summary += (
        f"\n\nSummary: {counts['info']} OK, {counts['warn']} warn, "
        f"{counts['error']} fail."
    )
    return summary


def max_severity(results: Iterable[CheckResult]) -> Severity:
    severities = {r.severity for r in results}
    if "error" in severities:
        return "error"
    if "warn" in severities:
        return "warn"
    return "info"


def severity_to_exit_code(severity: Severity) -> int:
    return {"info": 0, "warn": 1, "error": 2}[severity]
