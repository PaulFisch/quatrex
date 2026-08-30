#!/usr/bin/env python3
"""Build the evidence ledger for historical Si-film transport runs.

The ledger is intentionally independent of ``audit_runs.py``.  It records one
row per unique result artifact, or per unique Slurm job when no result survives,
and de-duplicates byte-identical mirrors by SHA-256 and job ID.  Effective
``QX_*`` values printed by the engine take precedence over the TOML because the
configuration file alone is not a record of what ran.

Usage::

    python phonon/scripts/si_run_census.py
    python phonon/scripts/si_run_census.py --check
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phonon/scripts/data/si_film_run_census.csv"
RESULT_GLOBS = ("run*.npz", "probe*.npz", "conv*.npz")

FIELDS = [
    "run_id", "directory", "aliases", "artifact", "artifact_sha256",
    "source_commit", "date", "host", "job_id", "restart_lineage",
    "primitive_cells", "group_layout", "microblock_dof", "primitive_g_band",
    "solver_g_band", "fc3_span", "generated_sigma_band", "frequency_grid",
    "frequency_min_thz", "frequency_max_thz", "frequency_spacing_thz",
    "frequency_points", "aux_dw_thz", "aux_fmax_thz", "q_mesh",
    "vertex_representation", "factor_rank", "vertex_scale", "factor_fit_source",
    "decomposed_kernel", "left_temperature_k", "right_temperature_k",
    "max_iterations", "min_iterations", "mixing_method", "mixing_factor",
    "sigma_tolerance", "heat_tolerance", "retarded_method", "eta_thz",
    "eta_obc_thz2", "eta_ramp_iterations", "eta_final_thz",
    "eta_obc_ramp_iterations", "eta_obc_final_thz2", "ir_floor_cells",
    "ir_floor_final_cells", "ir_floor_ramp_iterations",
    "low_frequency_mask_thz", "g_band_taper", "sse_ramp_iterations",
    "greater_from_lesser", "cutoff_angstrom", "cutoff_taper",
    "pole_sector", "cm_subtraction", "scp_tadpole", "scp_loop",
    "obc_algorithm", "nevp_solver", "obc_scattering_contacts",
    "block_comm_size", "q_comm_size", "nranks", "scba_iterations",
    "residual", "current", "conductance", "lead_balance", "internal_spread",
    "bubble_balance", "runtime_s", "kernel_time_s", "peak_memory_gb",
    "converged", "diverged", "ballistic", "classification", "reasons",
]

ENV_MAP = {
    "QX_GBAND": ("phonon", "sse_g_band"),
    "QX_MICRO_DOF": ("phonon", "sse_microblock_dof"),
    "QX_MICRO_GBAND": ("phonon", "sse_microblock_g_band"),
    "QX_VERTEX_RANK": ("phonon", "sse_vertex_rank"),
    "QX_VSCALE": ("phonon", "sse_vertex_scale"),
    "QX_MAXIT": ("scba", "max_iterations"),
    "QX_MINIT": ("scba", "min_iterations"),
    "QX_MIXMETHOD": ("scba", "mixing_method"),
    "QX_MIX": ("scba", "mixing_factor"),
    "QX_NE": ("electron", "energy_window_num"),
    "QX_WMAX": ("electron", "energy_window_max"),
    "QX_AUXDW": ("phonon", "sse_aux_grid_dw_thz"),
    "QX_AUXFMAX": ("phonon", "sse_aux_grid_fmax_thz"),
    "QX_SIGMATOL": ("phonon", "sigma_convergence_tol"),
    "QX_HEATTOL": ("phonon", "heat_flow_conservation_tol"),
    "QX_RETARDED": ("phonon", "retarded_method"),
    "QX_DECOMPOSED_KERNEL": ("phonon", "decomposed_kernel"),
    "QX_ETA": ("phonon", "eta"),
    "QX_ETAOBC": ("phonon", "eta_obc"),
    "QX_ETA_RAMP_ITERS": ("phonon", "eta_ramp_iterations"),
    "QX_ETA_FINAL": ("phonon", "eta_final"),
    "QX_ETA_IR_FLOOR": ("phonon", "eta_ir_floor_cells"),
    "QX_ETA_IR_FLOOR_FINAL": ("phonon", "eta_ir_floor_final_cells"),
    "QX_ETA_IR_FLOOR_RAMP": ("phonon", "eta_ir_floor_ramp_iterations"),
    "QX_SSE_LOWMASK": ("phonon", "sse_low_freq_mask_thz"),
    "QX_SSE_CMSUB": ("phonon", "sse_cm_subtraction"),
    "QX_GBAND_TAPER": ("phonon", "sse_g_band_taper"),
    "QX_RAMP": ("phonon", "sse_ramp_iterations"),
    "QX_G_FROM_L": ("phonon", "sse_greater_from_lesser"),
    "QX_SCP_TADPOLE": ("phonon", "scp_tadpole"),
    "QX_TLEFT": ("phonon", "left_temperature"),
    "QX_TRIGHT": ("phonon", "right_temperature"),
}


def _toml(path: Path) -> dict:
    try:
        import tomllib
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, ValueError):
        return {}


def _dig(tree: dict, *keys, default=None):
    node = tree
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _scalar(value, default=""):
    try:
        arr = np.asarray(value)
        return arr.item() if arr.size == 1 else default
    except Exception:
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: Path, limit: int = 8_000_000) -> str:
    try:
        size = path.stat().st_size
        with path.open(errors="ignore") as handle:
            if size <= limit:
                return handle.read()
            head = handle.read(limit // 2)
            handle.seek(size - limit // 2)
            return head + "\n" + handle.read()
    except OSError:
        return ""


def _logs(directory: Path) -> tuple[str, list[Path]]:
    paths = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and "_quatrex_times" not in p.name
        and (p.name.startswith("slurm-") or p.suffix in {".log", ".out"})
    )
    return "\n".join(_read_text(p) for p in paths), paths


def _env(text: str, directory: Path) -> dict[str, str]:
    out = {}
    for key, value in re.findall(r"(QX_[A-Z0-9_]+)=(\S+)", text):
        # A failed ``exec QX_FOO=...`` command is printed by bash as
        # ``QX_FOO=value: not found``.  The colon is diagnostic punctuation,
        # not part of the intended value.  Keeping it made one failed job
        # abort the complete historical census during numeric conversion.
        out[key] = value.rstrip("'\";,:")
    for script in directory.glob("*.sh"):
        for key, value in re.findall(
            r"(?:export\s+)?(QX_[A-Z0-9_]+)=([^\s;]+)", _read_text(script)
        ):
            out.setdefault(key, value.strip("'\""))
    return out


def _cfg_value(cfg: dict, env: dict, section: str, field: str, default=""):
    fallback = _dig(cfg, section, field, default=default)
    for key, route in ENV_MAP.items():
        if route == (section, field) and key in env:
            raw = env[key]
            if raw.lower() in {"true", "false"}:
                return raw.lower() == "true"
            try:
                return float(raw) if any(c in raw for c in ".eE") else int(raw)
            except ValueError:
                # Preserve genuine string-valued overrides, but a malformed
                # numeric value in a failed job is evidence to ignore rather
                # than a reason to lose every other census row.
                if isinstance(fallback, (bool, int, float, np.number)):
                    return fallback
                return raw
    return fallback


def _find_config(directory: Path, text: str) -> Path | None:
    hits = re.findall(r"RUN config=(\S+)", text)
    candidates = []
    for raw in reversed(hits):
        if "/quatrex/" in raw:
            candidates.append(ROOT / raw.split("/quatrex/", 1)[1])
    candidates += sorted(directory.glob("*.toml"))
    candidates += sorted(directory.rglob("quatrex_config.toml"))
    return next((p for p in candidates if p.exists()), None)


def _host(directory: Path, text: str) -> str:
    rel = directory.relative_to(ROOT).as_posix()
    if "cluster/alps/" in rel or "/capstor/" in text:
        return "alps/daint"
    if "/usr/scratch/" in text or "cluster/tortin" in rel:
        return "tortin"
    return "local mirror"


def _fc3_span(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        import h5py
        with h5py.File(path, "r") as handle:
            keys = np.asarray(handle["meta/keys"], dtype=int)
        return int(np.max(np.abs(keys[:, 1:] - keys[:, :1]))) if keys.size else 0
    except Exception:
        return None


def _factor_meta(path: Path | None) -> tuple[int | str, str]:
    if path is None or not path.exists():
        return "", ""
    try:
        with np.load(path, allow_pickle=True) as data:
            rank = int(np.asarray(data["lambdas"]).size)
            meta = dict(data["meta"].item()) if "meta" in data.files else {}
        source = meta.get("source") or meta.get("method") or meta.get("cache_key") or ""
        return rank, str(source)
    except Exception:
        return "", ""


def _artifact_values(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return {}
    out = {}
    try:
        for key in (
            "source_commit", "eta", "retarded", "nblocks", "n_iter", "lead_current",
            "internal_spread", "final_bubble_balance", "t_left", "t_right",
            "converged", "diverged", "ballistic", "sse_g_band",
            "sse_microblock_dof", "sse_microblock_g_band",
            "sse_generated_sigma_band", "sse_vertex_span", "sse_vertex_rank",
            "sse_vertex_scale",
            "vertex_representation", "frequency_grid", "sse_aux_grid_dw_thz",
            "sse_aux_grid_fmax_thz", "sigma_convergence_tol",
            "decomposed_kernel", "heat_flow_conservation_tol",
            "scba_max_iterations", "scba_min_iterations",
            "scba_mixing_method", "scba_mixing_factor",
            "left_temperature", "right_temperature", "eta_ramp_iterations",
            "eta_final", "eta_obc", "eta_obc_ramp_iterations", "eta_obc_final",
            "eta_ir_floor_cells", "eta_ir_floor_final_cells",
            "eta_ir_floor_ramp_iterations", "sse_low_freq_mask_thz",
            "sse_cm_subtraction", "sse_g_band_taper", "sse_ramp_iterations",
            "sse_greater_from_lesser", "scp_tadpole", "scp_loop",
            "pole_sector_enabled", "interaction_cutoff",
            "interaction_cutoff_taper", "obc_algorithm", "nevp_solver",
            "obc_scattering_contacts",
            "block_comm_size", "q_comm_size", "nranks",
        ):
            if key in data.files:
                out[key] = _scalar(data[key])
        if "energies" in data.files:
            energies = np.asarray(data["energies"], float)
            out["energies"] = energies
        if "block_sizes" in data.files:
            out["block_sizes"] = np.asarray(data["block_sizes"], int)
        if "q_mesh" in data.files:
            out["q_mesh"] = np.asarray(data["q_mesh"], int)
        if "last_heat" in data.files:
            heat = np.asarray(data["last_heat"], float).ravel()
            if heat.size >= 2:
                scale = max(float(np.mean(np.abs(heat[[0, -1]]))), 1e-300)
                out["lead_balance"] = abs(float(heat[0] - heat[-1])) / scale
        if "final_bubble_balance" in data.files:
            bal = np.asarray(data["final_bubble_balance"]).real.ravel()
            if bal.size >= 2:
                out["bubble_balance"] = abs(float(bal[0] - bal[1])) / max(
                    abs(float(bal[0])), abs(float(bal[1])), 1e-300)
    finally:
        data.close()
    return out


def classify(record: dict) -> tuple[str, str]:
    """Assign the primary historical status and a semicolon reason list."""
    reasons = []
    if record.get("diverged"):
        return "divergent", "artifact-or-log divergence"
    if record.get("ballistic") or not record.get("scba_iterations"):
        return "analysis-only", "ballistic or no surviving SCBA iteration"
    eta = float(record.get("eta_thz") or 0.0)
    if eta != 0.0:
        reasons.append(f"eta={eta:g}")
    raw_vertex_scale = record.get("vertex_scale")
    vertex_scale = (1.0 if raw_vertex_scale in (None, "")
                    else float(raw_vertex_scale))
    if vertex_scale != 1.0:
        reasons.append(f"nonphysical vertex scale {vertex_scale:g}")
    top = float(record.get("frequency_max_thz") or 0.0)
    if top < 30.6:
        reasons.append(f"frequency top {top:g} THz < 30.6 THz")
    micro = int(record.get("microblock_dof") or 0)
    groups = str(record.get("group_layout") or "")
    if not micro and groups not in {"single dense block", ""}:
        reasons.append("legacy self-energy output support")
    if str(record.get("retarded_method") or "") != "fft":
        reasons.append("non-causal half-retarded reconstruction")
    if reasons:
        if any(r.startswith("frequency top") for r in reasons):
            return "frequency-truncated", "; ".join(reasons)
        if "legacy self-energy output support" in reasons:
            return "spatially-pinned", "; ".join(reasons)
        return "superseded", "; ".join(reasons)
    if record.get("converged"):
        lead = record.get("lead_balance")
        bubble = record.get("bubble_balance")
        if lead in (None, "") or bubble in (None, ""):
            return "superseded", "missing invariant lead or bubble balance"
        heat_tol = float(record.get("heat_tolerance") or 1e-3)
        if float(lead) > heat_tol:
            return "superseded", (
                f"lead imbalance {float(lead):.3g} > {heat_tol:.3g}")
        if float(bubble) > 1e-10:
            return "superseded", (
                f"bubble imbalance {float(bubble):.3g} > 1e-10")
        return "trustworthy", (
            "support-complete settings and passed invariant stored gates; "
            "raw internal spread is diagnostic")
    return "superseded", "not certified converged"


def _candidate_dirs(roots: list[Path]) -> list[Path]:
    out = set()
    for root in roots:
        if not root.exists():
            continue
        for cfg in root.rglob("quatrex_config.toml"):
            low = cfg.parent.as_posix().lower()
            text = _read_text(cfg)
            tree = _toml(cfg)
            grid = _dig(tree, "device", "kpoint_grid", default=[1, 1, 1])
            tdir = "xyz".index(_dig(
                tree, "device", "transport_direction", default="x"))
            transverse = any(int(k) > 1 for i, k in enumerate(grid) if i != tdir)
            named = ("sifilm" in low or bool(re.search(
                r"(?:^|/)si(?:chk|res|4x|film)", low)))
            if (named or ("\nSi = 3" in text and transverse)) and "sinw" not in low:
                out.add(cfg.parent)
        for pattern in RESULT_GLOBS:
            for artifact in root.rglob(pattern):
                low = artifact.parent.as_posix().lower()
                if "sifilm" in low or re.search(
                    r"(?:^|/)si(?:chk|res|4x|film|-l\d)", low
                ):
                    out.add(artifact.parent)
        for script in root.rglob("job.sh"):
            low = script.parent.as_posix().lower()
            if re.search(r"(?:^|/)si(?:film|fix|4x|-l\d|mic)", low):
                out.add(script.parent)
    return sorted(out)


def _records(roots: list[Path]) -> list[dict]:
    raw = []
    for directory in _candidate_dirs(roots):
        text, logs = _logs(directory)
        env = _env(text, directory)
        cfg_path = _find_config(directory, text)
        cfg = _toml(cfg_path) if cfg_path else {}
        artifacts = sorted({
            p for pattern in RESULT_GLOBS for p in directory.glob(pattern)
            if not p.name.startswith(("qfold", "decomposed", "sigma_"))
        }) or [None]
        job_ids = sorted(set(re.findall(r"slurm-(\d+)", " ".join(
            p.name for p in logs))))
        commit = ""
        m = re.search(r"(?:commit|HEAD)[ =:]([0-9a-f]{7,40})", text, re.I)
        if m:
            commit = m.group(1)
        for artifact in artifacts:
            av = _artifact_values(artifact)
            rec = {field: "" for field in FIELDS}
            rec["directory"] = directory.relative_to(ROOT).as_posix()
            rec["artifact"] = artifact.name if artifact else ""
            rec["artifact_sha256"] = _sha256(artifact) if artifact else ""
            rec["source_commit"] = av.get("source_commit") or commit
            newest = max(
                [p.stat().st_mtime for p in logs]
                + ([artifact.stat().st_mtime] if artifact else [])
                + ([cfg_path.stat().st_mtime] if cfg_path else [0.0]))
            rec["date"] = dt.date.fromtimestamp(newest).isoformat() if newest else ""
            rec["host"] = _host(directory, text)
            rec["job_id"] = ";".join(job_ids)
            rec["restart_lineage"] = env.get("QX_SIGMA_INIT", "")

            nblocks = int(av.get("nblocks") or _dig(
                cfg, "device", "num_transport_cells", default=0) or 0)
            micro_dof = int(av.get("sse_microblock_dof") or _cfg_value(
                cfg, env, "phonon", "sse_microblock_dof", 0) or 0)
            block_sizes = av.get("block_sizes")
            if block_sizes is not None and micro_dof:
                layout = [int(x // micro_dof) for x in block_sizes]
            else:
                nat = None
                xyz = (cfg_path.parent if cfg_path else directory) / "structure.xyz"
                try:
                    nat = int(xyz.read_text().splitlines()[0])
                except Exception:
                    pass
                cells_per = max(1, (nat or 2) // 2)
                layout = [cells_per] * nblocks if nblocks else []
            rec["group_layout"] = (
                "single dense block" if len(layout) == 1 else
                ",".join(map(str, layout)))
            rec["primitive_cells"] = sum(layout)
            rec["microblock_dof"] = micro_dof
            rec["primitive_g_band"] = int(
                av.get("sse_microblock_g_band") or _cfg_value(
                    cfg, env, "phonon", "sse_microblock_g_band", 0) or
                _cfg_value(cfg, env, "phonon", "sse_g_band", 1) or 1)
            rec["solver_g_band"] = 1 if micro_dof else rec["primitive_g_band"]

            fc3_raw = _dig(cfg, "phonon", "fc3_path")
            fc3_path = Path(fc3_raw) if fc3_raw else None
            if fc3_path and not fc3_path.exists() and cfg_path:
                fc3_path = cfg_path.parent / fc3_path.name
            pspan = av.get("sse_vertex_span")
            if pspan in (None, ""):
                pspan = (_fc3_span(fc3_path) if fc3_path else None) or 1
            rec["fc3_span"] = int(pspan)
            rec["generated_sigma_band"] = int(
                av.get("sse_generated_sigma_band") or
                min(max(0, rec["primitive_cells"] - 1),
                    2 * rec["fc3_span"] + rec["primitive_g_band"]))

            energies = av.get("energies")
            if energies is None:
                e0 = float(_dig(cfg, "electron", "energy_window_min", default=0) or 0)
                e1 = float(_cfg_value(
                    cfg, env, "electron", "energy_window_max", 0) or 0)
                ne = int(_cfg_value(
                    cfg, env, "electron", "energy_window_num", 0) or 0)
                energies = np.linspace(e0, e1, ne) if ne else np.empty(0)
            rec["frequency_grid"] = av.get("frequency_grid") or _dig(
                cfg, "phonon", "frequency_grid", default="window")
            rec["frequency_min_thz"] = float(energies[0]) if energies.size else ""
            rec["frequency_max_thz"] = float(energies[-1]) if energies.size else ""
            rec["frequency_points"] = int(energies.size)
            if energies.size > 1:
                rec["frequency_spacing_thz"] = float(np.median(np.diff(energies)))
            rec["aux_dw_thz"] = av.get("sse_aux_grid_dw_thz") or _cfg_value(
                cfg, env, "phonon", "sse_aux_grid_dw_thz", 0)
            rec["aux_fmax_thz"] = av.get("sse_aux_grid_fmax_thz") or _cfg_value(
                cfg, env, "phonon", "sse_aux_grid_fmax_thz", 0)
            qmesh = av.get("q_mesh")
            if qmesh is None:
                qmesh = _dig(cfg, "device", "kpoint_grid", default=[])
            rec["q_mesh"] = "x".join(map(str, qmesh))

            factor_raw = _dig(cfg, "phonon", "decomposed_vertices_path")
            qfold_raw = _dig(cfg, "phonon", "qfold_path")
            rec["vertex_representation"] = av.get("vertex_representation") or (
                "decomposed" if factor_raw else ("qfold" if qfold_raw else "gamma"))
            factor_path = Path(factor_raw) if factor_raw else None
            if factor_path and not factor_path.exists() and cfg_path:
                factor_path = cfg_path.parent / factor_path.name
            rank, source = _factor_meta(factor_path)
            rec["factor_rank"] = av.get("sse_vertex_rank") or _cfg_value(
                cfg, env, "phonon", "sse_vertex_rank", rank) or rank
            rec["vertex_scale"] = av.get("sse_vertex_scale") or _cfg_value(
                cfg, env, "phonon", "sse_vertex_scale", 1.0)
            rec["factor_fit_source"] = source
            rec["decomposed_kernel"] = av.get("decomposed_kernel") or _cfg_value(
                cfg, env, "phonon", "decomposed_kernel", "")
            rec["left_temperature_k"] = (
                av.get("left_temperature") or av.get("t_left") or
                _cfg_value(cfg, env, "phonon", "left_temperature", ""))
            rec["right_temperature_k"] = (
                av.get("right_temperature") or av.get("t_right") or
                _cfg_value(cfg, env, "phonon", "right_temperature", ""))
            rec["max_iterations"] = av.get("scba_max_iterations") or _cfg_value(
                cfg, env, "scba", "max_iterations", "")
            rec["min_iterations"] = av.get("scba_min_iterations") or _cfg_value(
                cfg, env, "scba", "min_iterations", "")
            rec["mixing_method"] = av.get("scba_mixing_method") or _cfg_value(
                cfg, env, "scba", "mixing_method", "")
            rec["mixing_factor"] = av.get("scba_mixing_factor") or _cfg_value(
                cfg, env, "scba", "mixing_factor", "")
            rec["sigma_tolerance"] = av.get("sigma_convergence_tol") or _cfg_value(
                cfg, env, "phonon", "sigma_convergence_tol", "")
            rec["heat_tolerance"] = (
                av.get("heat_flow_conservation_tol") or _cfg_value(
                    cfg, env, "phonon", "heat_flow_conservation_tol", ""))
            rec["retarded_method"] = av.get("retarded") or _cfg_value(
                cfg, env, "phonon", "retarded_method", "")
            rec["eta_thz"] = av.get("eta") if "eta" in av else _cfg_value(
                cfg, env, "phonon", "eta", "")
            for out_key, cfg_key, default in (
                ("eta_obc_thz2", "eta_obc", 0),
                ("eta_ramp_iterations", "eta_ramp_iterations", 0),
                ("eta_final_thz", "eta_final", 0),
                ("eta_obc_ramp_iterations", "eta_obc_ramp_iterations", 0),
                ("eta_obc_final_thz2", "eta_obc_final", 0),
                ("ir_floor_cells", "eta_ir_floor_cells", 0),
                ("ir_floor_final_cells", "eta_ir_floor_final_cells", 0),
                ("ir_floor_ramp_iterations", "eta_ir_floor_ramp_iterations", 0),
                ("low_frequency_mask_thz", "sse_low_freq_mask_thz", 0),
                ("g_band_taper", "sse_g_band_taper", "none"),
                ("sse_ramp_iterations", "sse_ramp_iterations", 0),
                ("greater_from_lesser", "sse_greater_from_lesser", False),
                ("cutoff_angstrom", "interaction_cutoff", 10),
                ("cutoff_taper", "interaction_cutoff_taper", "none"),
                ("cm_subtraction", "sse_cm_subtraction", False),
                ("scp_tadpole", "scp_tadpole", False),
                ("scp_loop", "scp_loop", False),
            ):
                rec[out_key] = av.get(cfg_key, _cfg_value(
                    cfg, env, "phonon", cfg_key, default))
            rec["pole_sector"] = av.get("pole_sector_enabled", (
                bool(int(env["QX_POLE"])) if "QX_POLE" in env else bool(
                    _dig(cfg, "phonon", "pole_sector", "enabled", default=False))))
            rec["obc_algorithm"] = av.get("obc_algorithm") or env.get("QX_OBC_ALG") or _dig(
                cfg, "phonon", "obc", "algorithm", default="")
            rec["nevp_solver"] = av.get("nevp_solver") or env.get("QX_NEVP") or _dig(
                cfg, "phonon", "obc", "nevp_solver", default="")
            rec["obc_scattering_contacts"] = av.get(
                "obc_scattering_contacts", bool(int(env.get(
                    "QX_SCATCONTACTS", int(_dig(
                        cfg, "phonon", "obc_scattering_contacts", default=False))))))
            rec["block_comm_size"] = av.get("block_comm_size") or env.get(
                "QX_BCS") or _dig(cfg, "compute", "comm", "block_comm_size", default=1)
            rec["q_comm_size"] = av.get("q_comm_size") or env.get(
                "QX_QCS") or _dig(cfg, "compute", "comm", "q_comm_size", default=1)
            rec["nranks"] = av.get("nranks", "")
            rec["scba_iterations"] = int(av.get("n_iter") or max(
                [int(x) for x in re.findall(r"^Iteration (\d+)", text, re.M)] or [0]))
            residuals = re.findall(r"rel Sigma\^R residual ([0-9.eE+-]+)", text)
            rec["residual"] = float(residuals[-1]) if residuals else ""
            rec["current"] = av.get("lead_current", "")
            dtemp = (float(rec["left_temperature_k"] or 0)
                     - float(rec["right_temperature_k"] or 0))
            rec["conductance"] = (
                float(rec["current"]) / dtemp if rec["current"] != "" and dtemp else "")
            rec["lead_balance"] = av.get("lead_balance", "")
            rec["internal_spread"] = av.get("internal_spread", "")
            rec["bubble_balance"] = av.get("bubble_balance", "")
            wall = re.findall(r"(?:wall(?:time)?|elapsed)[ =:]([0-9.]+)s", text, re.I)
            rec["runtime_s"] = float(wall[-1]) if wall else ""
            kernel = re.findall(r"PhPh SSE: 3 ring contraction all\s*:\s*([0-9.]+)s", text)
            rec["kernel_time_s"] = float(kernel[-1]) if kernel else ""
            rss = re.findall(r"(?:MaxRSS|peak memory)[ =:]\s*([0-9.]+)\s*(GB|MB)", text, re.I)
            if rss:
                value, unit = rss[-1]
                rec["peak_memory_gb"] = float(value) / (1024 if unit.upper() == "MB" else 1)
            rec["converged"] = bool(av.get("converged") or "SCBA converged" in text)
            rec["diverged"] = bool(av.get("diverged") or re.search(
                r"diverg|non-finite|guard abort", text, re.I))
            rec["ballistic"] = bool(av.get("ballistic") or env.get("QX_BALLISTIC") == "1")
            rec["classification"], rec["reasons"] = classify(rec)
            rec["run_id"] = (
                rec["artifact_sha256"][:16] if rec["artifact_sha256"] else
                ("job-" + job_ids[-1] if job_ids else rec["directory"].replace("/", "-")))
            raw.append(rec)

    # Artifact hashes are definitive.  Job IDs de-duplicate metadata mirrors
    # only when no result artifact survives.
    unique = []
    seen_hash, seen_job = {}, {}
    for rec in sorted(raw, key=lambda r: (r["date"], r["directory"], r["artifact"])):
        key = rec["artifact_sha256"]
        prior = seen_hash.get(key) if key else None
        if prior is None and not key and rec["job_id"]:
            prior = seen_job.get(rec["job_id"])
        if prior is not None:
            aliases = [x for x in prior["aliases"].split(";") if x]
            aliases.append(rec["directory"] + ("/" + rec["artifact"] if rec["artifact"] else ""))
            prior["aliases"] = ";".join(sorted(set(aliases)))
            continue
        unique.append(rec)
        if key:
            seen_hash[key] = rec
        elif rec["job_id"]:
            seen_job[rec["job_id"]] = rec
    return unique


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    roots = [Path(p) for p in args.root] or [ROOT / "cluster"]
    rows = _records(roots)
    if args.check:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "census.csv"
            write(rows, candidate)
            old = args.output.read_bytes() if args.output.exists() else b""
            if candidate.read_bytes() != old:
                raise SystemExit("Si census is stale; regenerate without --check")
        print(f"Si census current: {len(rows)} unique runs")
        return 0
    write(rows, args.output)
    counts = {}
    for row in rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    print(f"wrote {args.output.relative_to(ROOT)}: {len(rows)} unique runs")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
