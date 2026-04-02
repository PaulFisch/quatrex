"""Run all FC3 displacement calculations and reap results.

Usage:
    python run_fc3.py run     # Run all QE jobs
    python run_fc3.py reap    # Collect forces and produce FORCE_CONSTANTS_3RD
    python run_fc3.py all     # Run + reap
"""

import subprocess
import sys
from pathlib import Path

work_dir = Path(__file__).resolve().parent
n_disp = 68
na, nb, nc = 2, 2, 2
cutoff = "-3"

# ---------------------------------------------------------------------------
# Run QE displacement calculations
# ---------------------------------------------------------------------------
def run_displacements():
    results_dir = work_dir / "results"
    results_dir.mkdir(exist_ok=True)

    for i in range(1, n_disp + 1):
        inp_file = work_dir / f"DISP.supercell_template.in.{i:02d}"
        out_file = work_dir / f"DISP.supercell_template.out.{i:02d}"

        if out_file.exists():
            with open(out_file) as f:
                if "JOB DONE" in f.read():
                    print(f"  Skipping disp {i:02d}/{n_disp} (done)")
                    continue

        print(f"  Running disp {i:02d}/{n_disp}...")
        result = subprocess.run(
            ["pw.x", "-in", str(inp_file)],
            capture_output=True, text=True,
            cwd=str(work_dir),
            timeout=3600,
        )
        with open(out_file, "w") as f:
            f.write(result.stdout)

        if "JOB DONE" not in result.stdout:
            print(f"  ERROR: disp {i:02d} did not converge!")
            if result.stderr:
                print(f"  stderr: {result.stderr[-300:]}")
            sys.exit(1)

        # Print wall time
        for line in result.stdout.split("\n"):
            if "WALL" in line:
                print(f"  Done: {line.strip()}")
                break

    print(f"\nAll {n_disp} displacements completed.")


# ---------------------------------------------------------------------------
# Reap: collect forces and produce FORCE_CONSTANTS_3RD
# ---------------------------------------------------------------------------
def reap():
    file_list = []
    for i in range(1, n_disp + 1):
        out_file = work_dir / f"DISP.supercell_template.out.{i:02d}"
        if not out_file.exists():
            print(f"ERROR: Missing output file: {out_file}")
            sys.exit(1)
        file_list.append(str(out_file))

    stdin_text = "\n".join(file_list) + "\n"

    print(f"Reaping {n_disp} displacement outputs...")
    result = subprocess.run(
        ["thirdorder_espresso.py", "unitcell.in", "reap",
         str(na), str(nb), str(nc), cutoff],
        input=stdin_text, text=True,
        capture_output=True,
        cwd=str(work_dir),
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Reap failed: {result.stderr}")
        sys.exit(1)

    fc3_file = work_dir / "FORCE_CONSTANTS_3RD"
    if fc3_file.exists():
        print(f"\nFORCE_CONSTANTS_3RD written ({fc3_file.stat().st_size} bytes)")
    else:
        print("ERROR: FORCE_CONSTANTS_3RD not created!")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "reap", "all"):
        print(__doc__)
        sys.exit(1)

    action = sys.argv[1]
    if action in ("run", "all"):
        run_displacements()
    if action in ("reap", "all"):
        reap()
