#!/usr/bin/env python3
"""Prune regenerable bulk and move superseded run directories to the attic.

Driven by the CSV that ``audit_runs.py`` writes. Two independent phases:

  prune    delete the files that ``build_inputs.py`` can rebuild from the
           source force constants (qfold_vertices.npz, fc3_blocks.hdf5) and
           the transient warm-start state (sigma_best*.npz). Runs on EVERY
           directory, superseded or not, because a directory that still feeds
           a figure needs its run.npz and its logs, not its vertices.

  archive  move the directories the audit calls superseded, no-log-inputs or
           no-solver-log into cluster/attic/<class>/, where <class> is the
           first gate they fail. Directories a committed script reads are
           never moved.

Nothing that records what was run is deleted: logs, configs, job scripts,
summary.json and run*.npz all stay. Two protections beyond the verdict:

  * a bulk file a committed script names by its full path is kept (that is
    cluster/prod/geom/sifilm_L3_nk9/qfold_vertices.npz, read by
    decomposed_sse_conservation.py, and cluster/mos2f3scp/*, read by
    _asr_project_film_job.sh);
  * git-tracked files are never touched, which is what makes this safe to
    point at phonon/studies/out.

Usage:
  archive_runs.py [--root cluster] [--manifest <csv>]   # dry run, the default
  archive_runs.py --apply
  archive_runs.py --apply --prune-only | --archive-only
"""

import argparse
import csv
import datetime
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

BULK_PREFIXES = ("qfold_vertices", "sigma_best", "sig_", "decomposed_vertices")
BULK_NAMES = ()  # fc3_blocks.hdf5 is kept -- see audit_runs.py

MOVE_VERDICTS = {"superseded", "no-log-inputs"}

# A directory a current note cites is not archived: the attic convention
# (phonon/docs/attic/README.md) is that nothing outside the attic cites what
# is in it, so moving a cited run would break the note instead of tidying it.
CITING_DOCS = "phonon/docs/"

# A bed staged in the last few days is not a dead run. cluster/sifilm8x2 was
# built on 2026-08-27 and has no log because it has not been launched yet.
MIN_AGE_DAYS = 7

# First match wins; this is the attic subdirectory a run lands in. Ordered by
# how fundamental the defect is, not by how many runs it catches.
CLASSES = (
    ("finite-eta", ("finite-eta", "ir-floor")),
    ("h6-cutoff", ("h6-cutoff",)),
    ("gband", ("gband",)),
    ("blocking", ("cells_per_block",)),
    ("extent", ("extent-truncated",)),
    ("failed", ("crash", "oom", "timeout")),
    ("inputs-only", ("no-solver-log",)),
)
# Every run in the corpus fails this one, so it can never be the class.
UNIVERSAL = {"no-cm-subtraction"}


def is_bulk(name):
    return name.startswith(BULK_PREFIXES) or name in BULK_NAMES


def tracked_files(root):
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z",
                          str(Path(root).relative_to(REPO))],
                         capture_output=True, text=True, check=False).stdout
    return {REPO / p for p in out.split("\0") if p}


def protected_paths(root):
    """Bulk files a committed script names by path -- keep these."""
    tail = "/".join(Path(os.path.relpath(root, REPO)).parts[-2:])
    # {a,b} brace lists are how the shell drivers spell a pair of inputs
    pattern = rf"{re.escape(tail)}/[A-Za-z0-9_./{{}},\-]+"
    out = subprocess.run(
        ["grep", "-rhoE", pattern, "--include=*.py", "--include=*.sh",
         "--include=*.md", str(REPO / "phonon"), str(REPO / "src"),
         str(REPO / "tests")],
        capture_output=True, text=True, check=False).stdout
    keep = set()
    for hit in out.splitlines():
        for cand in _expand_braces(hit):
            if is_bulk(os.path.basename(cand)):
                keep.add(cand)
    return keep


def _expand_braces(path):
    """cluster/x/{a,b} -> [cluster/x/a, cluster/x/b]; otherwise [path]."""
    m = re.search(r"\{([^{}]*)\}", path)
    if not m:
        return [path]
    out = []
    for part in m.group(1).split(","):
        out.extend(_expand_braces(
            path[:m.start()] + part.strip() + path[m.end():]))
    return out


def classify(reasons):
    live = [r for r in reasons.split(";") if r and r.split("=")[0]
            not in UNIVERSAL]
    for label, keys in CLASSES:
        for r in live:
            if r.split("=")[0] in keys:
                return label
    return "other"


def plan(root, manifest):
    rows = list(csv.DictReader(open(manifest)))
    today = datetime.date.today()
    tracked = tracked_files(root)
    protected = protected_paths(root)
    prunes, moves = [], []
    for r in rows:
        path = Path(root) / r["dir"]
        if not path.is_dir():
            continue
        for base, _, files in os.walk(path):
            for name in files:
                if not is_bulk(name):
                    continue
                f = Path(base) / name
                if f in tracked:
                    continue
                rel = os.path.relpath(f, REPO)
                if any(rel.endswith(p) or p.endswith(rel)
                       for p in protected):
                    continue
                try:
                    prunes.append((f, f.stat().st_size))
                except OSError:
                    continue
        if r["verdict"] not in MOVE_VERDICTS or r["read_by_code"]:
            continue
        if any(h.startswith(CITING_DOCS)
               for h in r["mentioned_in_docs"].split(";") if h):
            continue
        if r["mtime"] and (today - datetime.date.fromisoformat(
                r["mtime"])).days < MIN_AGE_DAYS:
            continue
        moves.append((path, classify(r["reasons"]), r["reasons"]))
    return prunes, moves


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO / "cluster"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete and move (default is a dry run)")
    ap.add_argument("--prune-only", action="store_true")
    ap.add_argument("--archive-only", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve()
    manifest = Path(a.manifest) if a.manifest else \
        REPO / "phonon/scripts/data" / (
            f"run_manifest_{root.name}.csv" if root.name != "out"
            else f"run_manifest_{root.parent.name}_out.csv")
    if not manifest.exists():
        sys.exit(f"no manifest at {manifest}; run audit_runs.py first")

    prunes, moves = plan(root, manifest)
    total = sum(s for _, s in prunes)
    verb = "deleting" if a.apply else "would delete"
    print(f"{verb} {len(prunes)} regenerable files, "
          f"{total / 2 ** 30:.1f} GB")
    from collections import Counter
    by_class = Counter(c for _, c, _ in moves)
    verb = "moving" if a.apply else "would move"
    print(f"{verb} {len(moves)} directories to {root.name}/attic/:")
    for c, n in by_class.most_common():
        print(f"    {c:14} {n:>3}")

    if not a.apply:
        print("\ndry run; pass --apply to act")
        return 0

    attic = root / "attic"
    if not a.archive_only:
        for f, _ in prunes:
            try:
                f.unlink()
            except OSError as exc:
                print(f"  skip {f}: {exc}", file=sys.stderr)
    if not a.prune_only:
        index = []
        rows = {r["dir"]: r for r in csv.DictReader(open(manifest))}
        for path, cls, reasons in moves:
            dest = attic / cls / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                print(f"  skip {path.name}: {dest} exists", file=sys.stderr)
                continue
            shutil.move(str(path), str(dest))
            row = dict(rows[path.name])
            row["attic_class"] = cls
            index.append(row)
        if index:
            out = attic / "INDEX.csv"
            fields = list(index[0].keys())
            existing = []
            if out.exists():
                existing = list(csv.DictReader(open(out)))
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields,
                                   extrasaction="ignore")
                w.writeheader()
                w.writerows(existing + index)
            print(f"wrote {out} ({len(existing) + len(index)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
