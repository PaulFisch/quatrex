#!/usr/bin/env python3
"""Audit every run directory against the correctness gates the method now has.

The config file is not the record of what ran. ``phonon/studies/engine/run.py``
takes environment overrides that never touch the TOML and prints two lines per
run for exactly this reason -- ``RUN config=...`` (effective eta, retarded,
nblk, ne, fgrid) and ``RUN env ...`` (every ``QX_*``). ``sse_g_band`` is not
recoverable from the stored TOMLs. This script reads the logs first, the config
second, and the code default for the run's date last.

The gates, and where each was established:

  a  >= 2 transport cells per BTD block   bubble_positivity.md Sec. 8,
                                          spatial_representation.md Sec. 0.3
  b  sse_g_band = 3                       report_rerun_backlog.md Sec. 6
  c  grid resolved, extent ~2 omega_max   grid_audit.md
  d  eta = 0                              report_removed.md
  e  interaction_cutoff >= 22 A (MoS2)    bubble_positivity.md Sec. 6.8-6.10

Gate (b) is clamped to ``n_blocks - 1`` at three sites in the solver, so it
competes with gate (a) for the same cells: g_band = 3 needs >= 4 blocks and
gate (a) needs >= 2 cells per block, hence >= 8 transport cells.

Usage:
  audit_runs.py [--root cluster]        # -> phonon/scripts/data/
  audit_runs.py --check                # re-derive and diff against the CSV
  audit_runs.py --summary              # verdict counts and reclaimable bytes
"""

import argparse
import csv
import datetime
import io
import math
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# The manifest is a committed distillate of an uncommitted tree, so it lives
# where the other committed distillates do (see PLOTS.md).
DATA = REPO / "phonon" / "scripts" / "data"

# Commit 82761380, 2026-08-01: "sse_g_band default 1 -> 3". Runs before this
# date that do not set QX_GBAND ran at 1; runs after, at 3.
GBAND_DEFAULT_FLIP = datetime.date(2026, 8, 1)

# config.py:1325 -- still the shipped default, and the H6 rung.
INTERACTION_CUTOFF_DEFAULT = 10.0
# bubble_positivity.md Sec. 6.9: 21 A is 98.6 % fill and still diverges,
# 22 A is dense and converges.
MOS2_CUTOFF_MIN = 22.0

# grid_audit.md: the code's own _check_kk_grid_support warns above 1 %.
KK_TOLERANCE_PERCENT = 1.0

# Regenerable from the source force constants by build_inputs.py, or
# transient warm-start state. fc3_blocks.hdf5 is deliberately NOT here: it is
# the vertex, several audit scripts open it through a variable path, and the
# whole corpus of them is 5.5 GB against 62 GB for the two below.
BULK_PREFIXES = ("qfold_vertices", "sigma_best", "sig_", "decomposed_vertices")
BULK_NAMES = ()

TRANSPORT_AXIS = {"x": 0, "y": 1, "z": 2}

FIELDS = [
    "dir", "system", "species", "mtime", "n_blocks", "atoms_per_block",
    "cells_per_block", "g_band_eff", "g_band_src", "ne", "wmax", "fgrid",
    "aux_dw", "eta", "eta_ir_floor", "interaction_cutoff", "lowmask", "cmsub",
    "kk_percent", "arms", "ballistic", "outcome", "iterations", "last_residual", "size_gb",
    "bulk_gb", "read_by_code", "mentioned_in_docs", "verdict", "reasons",
]

# Only a reference from executable code makes a directory a live data
# dependency. A name in a markdown note is provenance, not a dependency.
CODE_SUFFIXES = (".py", ".sh")

# The audit tooling names cluster paths in its own docstrings; a run must not
# become its own dependency.
SELF = {"audit_runs.py", "archive_runs.py", "tortin.py", "daint.py"}

# Directories under a root that are not runs: the attic, and the metadata
# mirror of the Alps corpus (audited separately with --root cluster/alps).
SUBTREES = {"attic", "alps"}


def _is_run_dir(root, name):
    return (
        not name.startswith(".")
        and name != "__pycache__"
        and name not in SUBTREES
        and os.path.isdir(os.path.join(root, name))
    )


# ---------------------------------------------------------------- utilities

def _load_toml(path):
    try:
        import tomllib
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except Exception:
        return {}


def _dig(cfg, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _walk(root):
    """(path, stat) for every regular file below root, broken links skipped."""
    for base, _, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            try:
                yield path, name, os.stat(path)
            except OSError:
                continue


def _is_bulk(name):
    return name.startswith(BULK_PREFIXES) or name in BULK_NAMES


def _localise(remote_path, root):
    """Map a QX_CONFIG path onto whatever local copy exists.

    Runs point QX_CONFIG at another directory more often than not (lsM4c ->
    lsM4/, l24f-g3 -> l24/), the path recorded is the tortin or Alps one,
    and some drivers record it repo-relative rather than absolute.
    """
    if not remote_path:
        return None
    cands = []
    if "/cluster/" in remote_path:
        tail = remote_path.split("/cluster/", 1)[1]
        cands.append(Path(root) / tail)
        # the Alps metadata mirror holds the geometry for beds never pulled
        cands.append(REPO / "cluster" / "alps" / tail)
    if "/quatrex/" in remote_path:
        cands.append(REPO / remote_path.split("/quatrex/", 1)[1])
    if not remote_path.startswith("/"):
        cands.append(REPO / remote_path)          # repo-relative
    for c in cands:
        if c.exists():
            return c
    return None


# ------------------------------------------------------------ log scraping

RUN_CONFIG_RE = re.compile(
    r"^RUN config=(?P<cfg>\S+).*?eta=(?P<eta>\S+) retarded=(?P<ret>\S+) "
    r"nblk=(?P<nblk>\S+) ne=(?P<ne>\S+)(?: fgrid=(?P<fgrid>\S+))?", re.M)
RUN_ENV_RE = re.compile(r"^RUN env (?P<env>.*)$", re.M)
KK_RE = re.compile(r"still carries ([0-9.]+)% of its peak weight")
RESID_RE = re.compile(r"rel Sigma\^R residual ([0-9.eE+-]+)")
CONVERGED_RE = re.compile(r"SCBA converged after (\d+) iterations")
ITER_RE = re.compile(r"^Iteration (\d+)", re.M)


GBAND_IN_NAME = re.compile(r"(?:^|[-_])g(?:band)?([123])(?![0-9])")


def _gband_from_name(name):
    """The tortin campaign encoded the band in the run name and nowhere else.

    Those runs pre-date the ``RUN env`` line (added 2026-08-01) and their
    configs never carried ``sse_g_band``, so ``cnt-L4-gband2`` / ``l16f-g3``
    is the only surviving record of what they ran at.
    """
    m = GBAND_IN_NAME.search(os.path.basename(name))
    return int(m.group(1)) if m else None


def _logs(path):
    out = []
    # campaign drivers write the arm's log beside the arm, as <arm>.log
    sibling = str(path) + ".log"
    if os.path.isfile(sibling):
        try:
            out.append((os.path.getmtime(sibling), sibling))
        except OSError:
            pass
    for base, _, files in os.walk(path):
        for name in files:
            if "_quatrex_times" in name:
                continue
            if name.endswith((".log", ".out", ".txt")):
                p = os.path.join(base, name)
                try:
                    out.append((os.path.getmtime(p), p))
                except OSError:
                    continue
    out.sort()
    return [p for _, p in out]


def _read(path, limit=8_000_000):
    """Read a log, keeping the head and the tail of very large ones."""
    try:
        size = os.path.getsize(path)
        with open(path, errors="ignore") as fh:
            if size <= limit:
                return fh.read()
            head = fh.read(limit // 2)
            fh.seek(size - limit // 2)
            return head + "\n" + fh.read()
    except OSError:
        return ""


def scrape_logs(path):
    """Effective parameters and outcome, from the newest log that has them."""
    # A driver script can put several arms in one log (cnt-cmnull runs a
    # bare arm and a QX_SSE_CMSUB=1 arm into the same file), so the env is
    # the UNION over every RUN env line and `arms` counts them. Reading only
    # the first line reports the control's settings as the run's.
    info = {"logs": 0, "run_cfg": None, "env": {}, "kk": 0.0, "arms": 0,
            "outcome": [], "iterations": None, "last_residual": None,
            "converged_at": None, "ballistic": False}
    logs = _logs(path)
    info["logs"] = len(logs)
    for p in reversed(logs):                       # newest first
        text = _read(p)
        if not text:
            continue
        if info["run_cfg"] is None:
            m = RUN_CONFIG_RE.search(text)
            if m:
                info["run_cfg"] = m.groupdict()
        for m in RUN_ENV_RE.finditer(text):
            for k, v in re.findall(r"(QX_[A-Z_0-9]+)=(\S+)", m.group("env")):
                info["env"].setdefault(k, v)
        info["arms"] += len(RUN_CONFIG_RE.findall(text))
        kk = [float(x) for x in KK_RE.findall(text)]
        if kk:
            info["kk"] = max(info["kk"], max(kk))
        m = CONVERGED_RE.search(text)
        if m and info["converged_at"] is None:
            info["converged_at"] = int(m.group(1))
            info["outcome"].append("converged")
        if re.search(r"DUE TO TIME LIMIT|CANCELLED AT", text):
            info["outcome"].append("timeout")
        if re.search(r"oom-kill|OutOfMemoryError|out of memory", text, re.I):
            info["outcome"].append("oom")
        if "Traceback (most recent call last)" in text:
            info["outcome"].append("crash")
        res = RESID_RE.findall(text)
        if res and info["last_residual"] is None:
            info["last_residual"] = float(res[-1])
        its = ITER_RE.findall(text)
        if its and info["iterations"] is None:
            info["iterations"] = int(its[-1])
    return info


def _job_env(path):
    env = {}
    for base, _, files in os.walk(path):
        for name in files:
            if not name.endswith(".sh"):
                continue
            try:
                text = open(os.path.join(base, name), errors="ignore").read()
            except OSError:
                continue
            for k, v in re.findall(r"export (QX_[A-Z_0-9]+)=(\S+)", text):
                env.setdefault(k, v)
    return env


# ------------------------------------------------------------- geometry

def _npz_ballistic(path):
    """(any_ballistic, any_anharmonic) from the `ballistic` flag run.py stores.

    More reliable than the log: a ballistic baseline usually prints
    "SCBA converged after 0 iterations", but units_parity/cnt33_L4 prints
    "after 2" with a frozen Sigma and is still ballistic.
    """
    try:
        import numpy as np
    except ImportError:
        return False, False
    ball = anh = False
    for base, _, files in os.walk(path):
        for name in sorted(files):
            if not name.endswith(".npz") or _is_bulk(name):
                continue
            try:
                with np.load(os.path.join(base, name),
                             allow_pickle=True) as d:
                    if "ballistic" not in d.files:
                        continue
                    if bool(d["ballistic"]):
                        ball = True
                    else:
                        anh = True
            except Exception:
                continue
    return ball, anh


def _structure(path):
    """(n_atoms, lattice 3x3) from the first structure.xyz found."""
    for base, _, files in os.walk(path):
        if "structure.xyz" not in files:
            continue
        try:
            with open(os.path.join(base, "structure.xyz"),
                      errors="ignore") as fh:
                nat = int(fh.readline().strip())
                comment = fh.readline()
        except (OSError, ValueError):
            continue
        m = re.search(r'Lattice="([^"]+)"', comment)
        if not m:
            return nat, None
        vals = [float(x) for x in m.group(1).split()]
        if len(vals) != 9:
            return nat, None
        return nat, [vals[0:3], vals[3:6], vals[6:9]]
    return None, None


def _transport_length(lattice, direction):
    if lattice is None:
        return None
    vec = lattice[TRANSPORT_AXIS.get(direction, 2)]
    return math.sqrt(sum(c * c for c in vec))


NAME_HINTS = (
    ("mos2_film", ("mos2", "mos", "cvm", "lsm", "tapt")),
    # l16f/l24f/l32f are the long-chain CNT(3,3) ladder, not a model chain
    # (_extract_gband_ladder.py names them; their blocks are 36 DOF = one
    # 12-atom primitive cell).
    ("cnt33", ("cnt-l", "cnt33", "cntcal", "cnt_cal", "l4", "l10", "l16f",
               "l24f", "l32f", "newton", "jp-l4", "mix-l4", "pgate",
               "firstborn", "gband-test", "baseline_merge", "l4gpu",
               "gband", "longg3", "cons-l4", "causality-probe")),
    ("cnt80", ("cnt80",)),
    ("si_film", ("sifilm", "sichk", "sires", "si4x", "sir5", "sicensus")),
    ("sinw", ("d5", "d11", "d17", "sinw", "fcq_d5")),
    ("srtio3", ("srtio3",)),
)


def _system_from_name(name):
    low = name.lower()
    for system, prefixes in NAME_HINTS:
        if low.startswith(prefixes):
            return system
    return "unknown"


def _system(species, kgrid, nat):
    sp = set(species or ())
    if sp == {"Mo", "S"}:
        return "mos2_film"
    if sp == {"C"}:
        return "cnt80" if (nat or 0) > 40 else "cnt33"
    if sp >= {"Sr", "Ti", "O"}:
        return "srtio3"
    if sp == {"Si"}:
        if kgrid and kgrid[0] == 1 and (kgrid[1] or 1) > 1:
            return "si_film"
        if (nat or 0) >= 15:
            return "sinw"
        return "si_chain"
    return "unknown"


# -------------------------------------------------------------- audit core

def audit_dir(path, root, refs):
    name = os.path.relpath(path, root)
    rec = {f: "" for f in FIELDS}
    rec["dir"] = name

    size = bulk = 0
    newest = 0.0
    outputs = geometry = False
    for _, fname, st in _walk(path):
        size += st.st_size
        newest = max(newest, st.st_mtime)
        if _is_bulk(fname):
            bulk += st.st_size
        if fname.endswith(".npz") and not _is_bulk(fname):
            outputs = True
        if fname.startswith("summary."):
            outputs = True
        if fname in ("dynamical_matrix.mat", "qfold_vertices.npz",
                     "fc3_blocks.hdf5"):
            geometry = True
    rec["has_outputs"], rec["has_geometry"] = outputs, geometry
    rec["size_gb"] = round(size / 2 ** 30, 4)
    rec["bulk_gb"] = round(bulk / 2 ** 30, 4)
    mtime = datetime.date.fromtimestamp(newest) if newest else None
    rec["mtime"] = mtime.isoformat() if mtime else ""

    log = scrape_logs(path)
    log_date = None
    _l = _logs(path)
    if _l:
        try:
            log_date = datetime.date.fromtimestamp(os.path.getmtime(_l[-1]))
        except OSError:
            log_date = mtime
    env = dict(_job_env(path))
    env.update(log["env"])                          # RUN env wins over job.sh

    # config: the one QX_CONFIG names, else the local one
    cfg_path = None
    if log["run_cfg"]:
        cfg_path = _localise(log["run_cfg"]["cfg"], root)
    if cfg_path is None:
        local = sorted(Path(path).rglob("quatrex_config.toml"))
        cfg_path = local[0] if local else None
    cfg = _load_toml(cfg_path) if cfg_path else {}

    species = tuple(sorted(_dig(cfg, "device", "num_orbitals_per_atom",
                                default={}) or {}))
    kgrid = _dig(cfg, "device", "kpoint_grid")
    direction = _dig(cfg, "device", "transport_direction", default="z")
    nat, lattice = _structure(path)
    if nat is None and cfg_path is not None:
        nat, lattice = _structure(cfg_path.parent)
        if nat is None:
            # tortin beds keep the geometry in a sibling <name>_inputs dir
            for sib in sorted(cfg_path.parent.parent.glob("*_inputs")):
                nat, lattice = _structure(sib)
                if nat is not None:
                    break
    rec["species"] = "+".join(species)
    rec["atoms_per_block"] = nat if nat is not None else ""
    rec["system"] = _system(species, kgrid, nat)
    if rec["system"] == "unknown":
        rec["system"] = _system_from_name(name)

    n_blocks = None
    if log["run_cfg"] and log["run_cfg"]["nblk"].isdigit():
        n_blocks = int(log["run_cfg"]["nblk"])
    if n_blocks is None:
        n_blocks = _dig(cfg, "device", "num_transport_cells")
    rec["n_blocks"] = n_blocks if n_blocks is not None else ""

    rec["_transport_length"] = _transport_length(lattice, direction)

    # (b) g_band: env, else the code default for the date, then the clamp
    if "QX_GBAND" in env:
        g_cfg, src = int(env["QX_GBAND"]), "env"
    elif _dig(cfg, "phonon", "sse_g_band") is not None:
        g_cfg, src = int(_dig(cfg, "phonon", "sse_g_band")), "config"
    elif _gband_from_name(name) is not None:
        g_cfg, src = _gband_from_name(name), "name"
    elif log_date is not None:
        g_cfg = 3 if log_date >= GBAND_DEFAULT_FLIP else 1
        src = "date"
    else:
        g_cfg, src = None, "unknown"
    if g_cfg is not None and n_blocks:
        rec["g_band_eff"] = min(g_cfg, n_blocks - 1)
        rec["g_band_src"] = src + ("+clamp" if g_cfg > n_blocks - 1 else "")
    elif g_cfg is not None:
        rec["g_band_eff"], rec["g_band_src"] = g_cfg, src
    else:
        rec["g_band_src"] = "unknown"

    # (c) grid
    ne = env.get("QX_NE") or (log["run_cfg"] or {}).get("ne") \
        or _dig(cfg, "electron", "energy_window_num")
    rec["ne"] = ne if ne not in (None, "") else ""
    rec["wmax"] = env.get("QX_WMAX") or _dig(cfg, "electron",
                                             "energy_window_max") or ""
    rec["fgrid"] = (log["run_cfg"] or {}).get("fgrid") \
        or _dig(cfg, "phonon", "frequency_grid") or ""
    rec["aux_dw"] = _dig(cfg, "phonon", "sse_aux_grid_dw_thz", default="")
    rec["kk_percent"] = log["kk"] or ""
    rec["arms"] = log["arms"] or ""

    # Historical DC-channel settings remain useful provenance.
    rec["lowmask"] = env.get("QX_SSE_LOWMASK") or \
        _dig(cfg, "phonon", "sse_low_freq_mask_thz", default="")
    rec["cmsub"] = env.get("QX_SSE_CMSUB") or \
        ("1" if _dig(cfg, "phonon", "sse_cm_subtraction") else "")

    # (d) eta
    eta = env.get("QX_ETA") or (log["run_cfg"] or {}).get("eta") \
        or _dig(cfg, "phonon", "eta")
    rec["eta"] = eta if eta not in (None, "") else ""
    rec["eta_ir_floor"] = env.get("QX_ETA_IR_FLOOR") or \
        _dig(cfg, "phonon", "eta_ir_floor_cells", default="")

    # (e) interaction cutoff
    if cfg:
        ic = _dig(cfg, "phonon", "interaction_cutoff")
        rec["interaction_cutoff"] = (ic if ic is not None
                                     else INTERACTION_CUTOFF_DEFAULT)

    # A ballistic baseline reports "SCBA converged after 0 iterations": the
    # loop never ran, so it is not evidence about the anharmonic fixed point.
    npz_ball, npz_anh = _npz_ballistic(path)
    ballistic = (log["ballistic"] or env.get("QX_BALLISTIC") == "1"
                 or (npz_ball and not npz_anh))
    rec["ballistic"] = "yes" if ballistic else ""
    ran = bool(log["run_cfg"]) or log["iterations"] is not None
    rec["has_solver_log"] = ran
    if ballistic and (log["converged_at"] == 0 or (npz_ball and not npz_anh)):
        log["outcome"] = [o for o in log["outcome"] if o != "converged"]
        log["outcome"].append("ballistic")
    rec["outcome"] = "+".join(sorted(set(log["outcome"]))) or (
        "no-log" if not log["logs"] else
        ("incomplete" if ran else "no-solver-log"))
    rec["iterations"] = log["converged_at"] or log["iterations"] or ""
    rec["last_residual"] = log["last_residual"] if log["last_residual"] \
        is not None else ""
    hits = set(refs.get(name, ())) | set(refs.get(name.split("/")[0], ()))
    rec["read_by_code"] = ";".join(
        sorted(h for h in hits if h.endswith(CODE_SUFFIXES)))
    rec["mentioned_in_docs"] = ";".join(
        sorted(h for h in hits if not h.endswith(CODE_SUFFIXES)))
    return rec


def _as_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def classify(rec, cells_per_block):
    """Verdict + the list of gates the run fails."""
    rec["cells_per_block"] = cells_per_block if cells_per_block else ""
    reasons = []

    # A directory a committed script names is a live dependency; over-
    # inclusive on purpose, since a false positive only means it is not moved.
    verdict = "keep-referenced" if rec["read_by_code"] else None

    if not rec["has_solver_log"]:
        reasons.append("no-solver-log")
    else:
        eta = _as_float(rec["eta"])
        if eta is not None and eta > 1e-9:
            reasons.append(f"finite-eta={eta:g}")
        if _as_float(rec["eta_ir_floor"], 0.0) > 0:
            reasons.append("ir-floor")
        gb = rec["g_band_eff"]
        if isinstance(gb, int) and gb < 3:
            reasons.append(f"gband={gb}")
        elif rec["g_band_src"] == "unknown":
            reasons.append("gband=unknown")
        if not cells_per_block:
            reasons.append("cells_per_block=unknown")
        elif cells_per_block < 2:
            reasons.append(f"cells_per_block={cells_per_block}")
        kk = _as_float(rec["kk_percent"], 0.0)
        if kk > KK_TOLERANCE_PERCENT:
            reasons.append(f"extent-truncated={kk:g}%")
        ic = _as_float(rec["interaction_cutoff"])
        if rec["system"] == "mos2_film":
            if ic is None:
                reasons.append("h6-cutoff=unknown")
            elif ic < MOS2_CUTOFF_MIN:
                reasons.append(f"h6-cutoff={ic:g}")
        for flag in ("crash", "oom", "timeout"):
            if flag in rec["outcome"].split("+"):
                reasons.append(flag)

    if verdict is None:
        if not rec["has_solver_log"]:
            if rec["has_outputs"]:
                verdict = "no-log-outputs"
            elif rec["has_geometry"]:
                verdict = "no-log-inputs"
            else:
                verdict = "analysis-only"
        elif not reasons:
            verdict = "keep-current"
        else:
            verdict = "superseded"
    rec["verdict"] = verdict
    rec["reasons"] = ";".join(reasons)
    return rec


# ------------------------------------------------- committed-code references

def _committed_sources():
    """Every tracked (and new, untracked) .py/.sh/.md/.tex, as (rel, text).

    Untracked files count: a note written this session that cites a run is
    exactly the reason not to archive that run, and it is not committed yet.
    """
    out = ""
    for args in (["ls-files", "-z"],
                 ["ls-files", "-z", "--others", "--exclude-standard"]):
        out += subprocess.run(["git", "-C", str(REPO)] + args,
                              capture_output=True, text=True,
                              check=False).stdout
    sources = []
    for rel in out.split("\0"):
        if not rel.endswith((".py", ".sh", ".md", ".tex", ".toml")):
            continue
        if os.path.basename(rel) in SELF:
            continue
        try:
            sources.append((rel, (REPO / rel).read_text(errors="ignore")))
        except OSError:
            continue
    return sources


def _name_probes(name):
    """Literals that would appear in code if this directory were read.

    Generators spell the path three ways, and only the first carries the
    tree prefix:  "cluster/mos2f3",  CL / "cnt-L3-gband2/run.log",  and
    CL / f"newton-pc-{arm}/run.log". The third leaves only the stem, so
    every prefix ending on a separator is probed against an f-string brace.
    """
    probes = [name]
    for i, ch in enumerate(name):
        if ch in "-_" and i >= 5:
            probes.append(name[:i + 1] + "{")
    return probes


# A token in the position a path component or a quoted literal occupies.
# Requiring one of those is what stops short generic names from matching
# prose: bare `prod` is inside "production", `out` inside "output",
# `reference` inside "a reference implementation". The trailing "{" form
# catches CL / f"newton-pc-{arm}/run.log", whose source carries only the stem.
_TOKEN_RE = re.compile(r"([A-Za-z0-9_.\-]+\{?)")


def _candidate_tokens(text):
    """Every token in the text that sits where a directory name would."""
    found = set()
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(1)
        prev = text[m.start() - 1] if m.start() else ""
        nxt = text[m.end()] if m.end() < len(text) else ""
        # backtick: markdown notes cite a run as `cnt-cmnull2`
        if prev in "\"'/`" or nxt == "/":
            found.add(tok)
    return found


def build_reference_map(root):
    """dir name -> {referencing committed file}.

    Over-inclusive by design: a false positive only means a directory is not
    archived, while a false negative deletes something a figure needs. One
    pass per source file, not per (name, file) pair -- the corpus is ~600
    names against ~1500 committed sources.
    """
    names = [d for d in os.listdir(root) if _is_run_dir(root, d)]
    probe_to_name = {}
    for name in names:
        for pr in _name_probes(name):
            probe_to_name.setdefault(pr, set()).add(name)
    refs = {}
    for rel, text in _committed_sources():
        for tok in _candidate_tokens(text) & probe_to_name.keys():
            for name in probe_to_name[tok]:
                refs.setdefault(name, set()).add(rel)
    return refs


# ------------------------------------------------------------------- driver

def _tag(root):
    """cluster -> cluster; phonon/studies/out -> studies_out (they collide)."""
    parts = Path(root).parts
    return parts[-1] if parts[-1] != "out" else f"{parts[-2]}_out"


def _run_dirs(root, nested):
    """Top-level directories, or their children when a campaign nests arms.

    phonon/studies/out holds campaign directories (cnt33_gband_length) whose
    arms (L8_g3, L10_g3, ...) are the runs; auditing the parent hides which
    arm converged.
    """
    names = sorted(d for d in os.listdir(root) if _is_run_dir(root, d))
    if not nested:
        return names
    out = []
    for d in names:
        path = os.path.join(root, d)
        kids = sorted(k for k in os.listdir(path) if _is_run_dir(path, k))
        if kids:
            out.extend(os.path.join(d, k) for k in kids)
        else:
            out.append(d)
    return out


def audit_root(root, nested=False):
    root = str(root)
    refs = build_reference_map(root)
    names = _run_dirs(root, nested)
    recs = [audit_dir(os.path.join(root, n), root, refs) for n in names]

    # cells per block: the transport-direction lattice length against the
    # shortest one seen for the same system. A 12.294 A MoS2 block is one
    # cell; 24.588 A is two.
    # cells per block = this bed's transport-cell length against the shortest
    # one seen for the same system. A 12.294 A MoS2 block is one cell, 24.588
    # is two. There is no second route: gl_diag_imag in run.npz is the
    # RANK-LOCAL slice, so block DOF read from it is wrong under MPI.
    prim = {}
    for r in recs:
        L = r["_transport_length"]
        if L:
            prim[r["system"]] = min(prim.get(r["system"], L), L)
    for r in recs:
        L = r.pop("_transport_length", None)
        base = prim.get(r["system"])
        classify(r, round(L / base) if (L and base) else None)
    return recs


def write_csv(recs, out):
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)


def summarise(recs):
    from collections import Counter
    def _num(v):
        return v if isinstance(v, (int, float)) else 0.0

    total = sum(_num(r["size_gb"]) for r in recs)
    bulk = sum(_num(r["bulk_gb"]) for r in recs)
    known = any(r["bulk_gb"] != "" for r in recs)
    print(f"{len(recs)} directories, {total:.1f} GB"
          + (f", {bulk:.1f} GB regenerable bulk" if known
             else " (metadata mirror: bulk not measurable)"))
    print()
    counts = Counter(r["verdict"] for r in recs)
    print(f"{'verdict':16} {'n':>4} {'GB':>8} {'bulk GB':>9}")
    for v, n in counts.most_common():
        sub = [r for r in recs if r["verdict"] == v]
        print(f"{v:16} {n:>4} {sum(_num(r['size_gb']) for r in sub):>8.2f} "
              f"{sum(_num(r['bulk_gb']) for r in sub):>9.2f}")
    print()
    gates = Counter()
    for r in recs:
        for reason in filter(None, r["reasons"].split(";")):
            gates[reason.split("=")[0]] += 1
    print("gate failures:")
    for g, n in gates.most_common():
        print(f"  {g:24} {n}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO / "cluster"),
                    help="directory of run directories (default: cluster/)")
    ap.add_argument("--out", default=None,
                    help="CSV to write (default: "
                         "phonon/scripts/data/run_manifest_<root>.csv)")
    ap.add_argument("--check", action="store_true",
                    help="re-derive and diff against the existing CSV")
    ap.add_argument("--sizes", default=None,
                    help="`du -sk` output for the REAL tree, when --root is "
                         "a metadata-only mirror (cluster/alps)")
    ap.add_argument("--nested", action="store_true",
                    help="audit campaign sub-directories as separate runs")
    ap.add_argument("--summary", action="store_true",
                    help="print verdict counts instead of writing the CSV")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve()
    out = Path(a.out) if a.out else DATA / f"run_manifest_{_tag(root)}.csv"
    recs = audit_root(root, nested=a.nested)
    if a.sizes:
        # a metadata mirror knows what ran but not what it costs
        real = {}
        for line in Path(a.sizes).read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                real[parts[1].strip().rstrip("/")] = int(parts[0])
        missing = 0
        for r in recs:
            kb = real.get(r["dir"])
            if kb is None:
                missing += 1
            else:
                r["size_gb"] = round(kb * 1024 / 2 ** 30, 4)
            r["bulk_gb"] = ""          # not measurable from a mirror
        if missing:
            print(f"note: {missing} rows had no size in {a.sizes}",
                  file=sys.stderr)

    if a.summary:
        summarise(recs)
        return 0
    if a.check:
        buf = io.StringIO(newline="")
        w = csv.DictWriter(buf, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
        # newline="" so the comparison sees the CRLF the csv module writes
        old = out.read_text(newline="") if out.exists() else ""
        if old == buf.getvalue():
            print(f"{out}: up to date ({len(recs)} rows)")
            return 0
        print(f"{out}: STALE -- re-run without --check", file=sys.stderr)
        return 1

    write_csv(recs, out)
    print(f"wrote {out} ({len(recs)} rows)")
    summarise(recs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
