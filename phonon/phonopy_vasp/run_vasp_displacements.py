#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JobResult:
    run_dir: Path
    return_code: int
    skipped: bool


def find_run_dirs(workdir: Path, targets: list[str]) -> list[Path]:
    run_dirs: list[Path] = []
    for target in targets:
        root = workdir / target / "vasp"
        if not root.exists():
            continue
        run_dirs.extend(sorted(path for path in root.glob("disp-*") if path.is_dir()))
    return run_dirs


def is_complete_vasprun_xml(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    tail = path.read_bytes()[-8192:].decode("utf-8", errors="ignore")
    return "</modeling>" in tail


def summarize_vasprun_outputs(run_dirs: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    complete: list[Path] = []
    missing: list[Path] = []
    incomplete: list[Path] = []

    for run_dir in run_dirs:
        vasprun = run_dir / "vasprun.xml"
        if not vasprun.exists() or vasprun.stat().st_size == 0:
            missing.append(run_dir)
        elif is_complete_vasprun_xml(vasprun):
            complete.append(run_dir)
        else:
            incomplete.append(run_dir)

    return complete, missing, incomplete


def run_one(
    run_dir: Path,
    cmd: list[str],
    skip_existing: bool,
    dry_run: bool,
) -> JobResult:
    vasprun = run_dir / "vasprun.xml"
    if skip_existing and is_complete_vasprun_xml(vasprun):
        print(f"[skip] {run_dir} (complete vasprun.xml exists)")
        return JobResult(run_dir=run_dir, return_code=0, skipped=True)

    if skip_existing and vasprun.exists() and vasprun.stat().st_size > 0 and not is_complete_vasprun_xml(vasprun):
        print(f"[rerun] {run_dir} (existing vasprun.xml is incomplete)")

    print(f"[run]  {run_dir}")
    if dry_run:
        return JobResult(run_dir=run_dir, return_code=0, skipped=False)

    out_file = run_dir / "vasp.out"
    err_file = run_dir / "vasp.err"
    with out_file.open("w") as out, err_file.open("w") as err:
        completed = subprocess.run(cmd, cwd=str(run_dir), stdout=out, stderr=err)

    if completed.returncode != 0:
        print(f"[fail] {run_dir} (exit={completed.returncode})")
    else:
        print(f"[ok]   {run_dir}")

    return JobResult(
        run_dir=run_dir,
        return_code=completed.returncode,
        skipped=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VASP for all generated displacement folders."
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("phonon-data/si"),
        help="Root workflow directory created by generate_si_fc.py",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["harmonic", "anharmonic"],
        default=["harmonic", "anharmonic"],
        help="Which displacement groups to run",
    )
    parser.add_argument(
        "--vasp-cmd",
        default=None,
        help="Quoted VASP command, e.g. 'mpirun -np 16 vasp_std'",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of displacement jobs to run in parallel",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip folders that already contain vasprun.xml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without launching VASP",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check whether vasprun.xml outputs are complete; do not run VASP.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = args.workdir.resolve()
    run_dirs = find_run_dirs(workdir, args.targets)

    if not run_dirs:
        print("No displacement folders found.", file=sys.stderr)
        return 1

    if args.max_workers < 1:
        print("--max-workers must be >= 1", file=sys.stderr)
        return 1

    print(f"Found {len(run_dirs)} displacement folders")
    print(f"Targets: {', '.join(args.targets)}")

    if args.check_only:
        complete, missing, incomplete = summarize_vasprun_outputs(run_dirs)
        for run_dir in complete:
            print(f"[ok]         {run_dir}")
        for run_dir in missing:
            print(f"[missing]    {run_dir}")
        for run_dir in incomplete:
            print(f"[incomplete] {run_dir}")

        print("\nCheck Summary")
        print(f"  Total:      {len(run_dirs)}")
        print(f"  Complete:   {len(complete)}")
        print(f"  Missing:    {len(missing)}")
        print(f"  Incomplete: {len(incomplete)}")

        if missing or incomplete:
            print("\nSome displacement outputs are not ready for collect.")
            return 2
        return 0

    if not args.vasp_cmd:
        print("--vasp-cmd is required unless --check-only is used", file=sys.stderr)
        return 1

    cmd = shlex.split(args.vasp_cmd)
    if not cmd:
        print("Invalid --vasp-cmd", file=sys.stderr)
        return 1

    print(f"VASP cmd: {' '.join(cmd)}")
    print(f"Parallel workers: {args.max_workers}")

    skip_existing = not args.no_skip_existing
    results: list[JobResult] = []

    if args.max_workers == 1:
        for run_dir in run_dirs:
            results.append(run_one(run_dir, cmd, skip_existing, args.dry_run))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = [
                pool.submit(run_one, run_dir, cmd, skip_existing, args.dry_run)
                for run_dir in run_dirs
            ]
            for fut in as_completed(futures):
                results.append(fut.result())

    failed = [item for item in results if item.return_code != 0]
    skipped = sum(1 for item in results if item.skipped)
    completed = len(results) - skipped - len(failed)

    print("\nSummary")
    print(f"  Total:   {len(results)}")
    print(f"  Done:    {completed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {len(failed)}")

    if failed:
        print("Failed folders:")
        for item in sorted(failed, key=lambda value: str(value.run_dir)):
            print(f"  - {item.run_dir}")
        return 2

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
