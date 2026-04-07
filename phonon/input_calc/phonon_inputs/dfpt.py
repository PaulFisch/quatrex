"""DFPT force constants via QE ph.x + D3Q.

Alternative to the finite displacement (phono3py + symfc) approach.
Uses Density Functional Perturbation Theory for FC2 (ph.x + q2r.x)
and third-order DFPT for FC3 (d3q.x + d3_qq2rr.x).

Workflow:
    1. scf:    pw.x self-consistent calculation on unit cell
    2. ph:     ph.x DFPT phonon calculation on q-grid, saving drho_star
    3. q2r:    q2r.x Fourier transform dynamical matrices -> FC2
    4. d3q:    d3q.x third-order DFPT on the full q-grid
    5. qq2rr:  d3_qq2rr.x Fourier transform -> real-space FC3
    6. save:   Convert to phono3py format, write fc3.hdf5
"""

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms

from .config import DFPTConfig, QEConfig

# ---------------------------------------------------------------------------
# Unit conversions (QE internal: Ry, bohr)
# ---------------------------------------------------------------------------
BOHR_TO_ANG = 0.52917721067
RY_TO_EV = 13.605693009
RY_BOHR2_TO_EV_ANG2 = RY_TO_EV / BOHR_TO_ANG ** 2
RY_BOHR3_TO_EV_ANG3 = RY_TO_EV / BOHR_TO_ANG ** 3

DFPT_PREFIX = "dfpt_fc"
DRHO_EXT = "drho"
DRHO_DIRNAME = "FILDRHO"
D3_TMP_DIRNAME = "d3_tmp"
D3_OUT_DIRNAME = "d3_save"
D3_OUT_PREFIX = "anh"
D3_MODE = "full"


# ---------------------------------------------------------------------------
# Input file writers
# ---------------------------------------------------------------------------


def _write_ph_input(
        path: Path,
        dfpt_config: DFPTConfig,
        prefix: str = DFPT_PREFIX,
) -> None:
    """Write ph.x input for DFPT phonon calculation on a q-grid."""
    nq1, nq2, nq3 = dfpt_config.q_mesh
    with open(path, "w") as f:
        f.write("Phonon calculation on grid\n")
        f.write("&INPUTPH\n")
        f.write(f"   prefix           = '{prefix}'\n")
        f.write("   outdir           = './results'\n")
        f.write("   fildyn           = 'matdyn'\n")
        f.write("   ldisp            = .true.\n")
        f.write(f"   nq1              = {nq1}\n")
        f.write(f"   nq2              = {nq2}\n")
        f.write(f"   nq3              = {nq3}\n")
        f.write(f"   tr2_ph           = {dfpt_config.tr2_ph}\n")
        f.write("   epsil            = .false.\n")
        f.write("   lqdir            = .true.\n")
        f.write("   drho_star%open   = .true.\n")
        f.write(f"   drho_star%ext    = '{DRHO_EXT}'\n")
        f.write(f"   drho_star%dir    = './{DRHO_DIRNAME}'\n")
        f.write("/\n")


def _write_q2r_input(path: Path) -> None:
    """Write q2r.x input for FC2 Fourier transform."""
    with open(path, "w") as f:
        f.write("&INPUT\n")
        f.write("   fildyn  = 'matdyn'\n")
        f.write("   zasr    = 'crystal'\n")
        f.write("   flfrc   = 'fc2.dat'\n")
        f.write("/\n")


def _write_d3q_input(
        path: Path,
        dfpt_config: DFPTConfig,
        prefix: str = DFPT_PREFIX,
) -> None:
    """Write d3q.x input for a full-grid FC3 calculation."""
    nq1, nq2, nq3 = dfpt_config.q_mesh
    with open(path, "w") as f:
        f.write("&inputd3q\n")
        f.write(f"   mode        = '{D3_MODE}'\n")
        f.write(f"   prefix      = '{prefix}'\n")
        f.write("   outdir      = './results'\n")
        f.write(f"   d3dir       = './{D3_TMP_DIRNAME}'\n")
        f.write(f"   fildrho     = '{DRHO_EXT}'\n")
        f.write(f"   fildrho_dir = './{DRHO_DIRNAME}'\n")
        f.write(f"   fild3dyn    = './{D3_OUT_DIRNAME}/{D3_OUT_PREFIX}'\n")
        f.write("   restart     = .false.\n")
        f.write("   safe_io     = .false.\n")
        f.write("   print_star  = .true.\n")
        f.write("/\n")
        f.write(f"  {nq1}  {nq2}  {nq3}\n")


def _write_d3_asr_input(path: Path, dfpt_config: DFPTConfig) -> None:
    """Write d3_asr.x input for acoustic sum rule enforcement on FC3."""
    nq1, nq2, nq3 = dfpt_config.q_mesh
    with open(path, "w") as f:
        f.write("&input\n")
        f.write(f"   nq1    = {nq1}\n")
        f.write(f"   nq2    = {nq2}\n")
        f.write(f"   nq3    = {nq3}\n")
        f.write("   fild3  = 'mat3R'\n")
        f.write(f"   asr    = '{dfpt_config.asr}'\n")
        f.write("/\n")


def _write_d3_sparse_input(path: Path, dfpt_config: DFPTConfig) -> None:
    """Write d3_sparse.x input for FC3 sparsification."""
    nq1, nq2, nq3 = dfpt_config.q_mesh
    with open(path, "w") as f:
        f.write("&input\n")
        f.write(f"   nq1        = {nq1}\n")
        f.write(f"   nq2        = {nq2}\n")
        f.write(f"   nq3        = {nq3}\n")
        f.write("   fild3      = 'mat3R'\n")
        f.write(f"   sparse_thr = {dfpt_config.sparse_thr}\n")
        f.write("/\n")


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------


def _run_step(
        work_dir: Path,
        command: str,
        input_file: str,
        output_file: str,
        timeout: int,
        label: str = "",
        skip_existing: bool = True,
        required_files: tuple[str, ...] = (),
        input_mode: str = "flag",
) -> None:
    """Run a QE/D3Q calculation step."""
    inp = work_dir / input_file
    out = work_dir / output_file

    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    if skip_existing and out.exists() and required_files:
        if all((work_dir / f).exists() for f in required_files):
            print(f"  [{label}] Skipping (required files already exist)")
            return

    if input_mode == "flag":
        cmd = f"{command} -in {input_file}"
    elif input_mode == "stdin":
        cmd = f"{command} < {input_file}"
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    print(f"  [{label}] Running {cmd} ...")

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(work_dir),
        timeout=timeout,
    )

    with open(out, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    if result.returncode != 0:
        print(f"  ERROR: {input_file} exited with return code {result.returncode}")
        if result.stderr:
            print(f"  stderr (last 1000 chars): {result.stderr[-1000:]}")
        raise RuntimeError(
            f"Step '{label}' failed with return code {result.returncode}: {input_file}"
        )

    missing = [fname for fname in required_files if not (work_dir / fname).exists()]
    if missing:
        raise RuntimeError(
            f"Step '{label}' finished but missing expected files: {', '.join(missing)}"
        )

    for line in result.stdout.split("\n"):
        if "WALL" in line:
            print(f"    {line.strip()}")
            break


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def _parse_q2r_fc2(
    flfrc_path: Path,
    nat: int,
    q_mesh: list[int],
) -> tuple[np.ndarray, dict]:
    """Parse q2r.x force constant file (flfrc format)."""
    n1, n2, n3 = q_mesh
    n_images = n1 * n2 * n3
    n_super = n_images * nat

    with open(flfrc_path) as f:
        lines = f.readlines()

    idx = 0

    # Line 1: ntyp nat ibrav celldm(1:6)
    parts = lines[idx].split()
    ntyp = int(parts[0])
    nat_file = int(parts[1])
    if nat_file != nat:
        raise ValueError(f"nat mismatch: config={nat}, file={nat_file}")
    ibrav = int(parts[2])
    idx += 1

    masses = {}

    def _is_three_floats(tokens: list[str]) -> bool:
        if len(tokens) < 3:
            return False
        try:
            float(tokens[0])
            float(tokens[1])
            float(tokens[2])
            return True
        except ValueError:
            return False

    # Some q2r/flfrc files contain type/mass lines here, others go directly
    # to CELL vectors when ibrav == 0.
    if not (ibrav == 0 and _is_three_floats(lines[idx].split())):
        for it in range(1, ntyp + 1):
            parts = lines[idx].split()
            if len(parts) < 3:
                raise ValueError(f"Malformed type line in {flfrc_path}: {lines[idx].rstrip()}")

            # Expected format: ityp symbol mass
            try:
                type_idx = int(parts[0])
                symbol = parts[1].strip("'\"")
                mass = float(parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse type/mass line in {flfrc_path}: {lines[idx].rstrip()}"
                ) from exc

            masses[type_idx] = (symbol, mass)
            idx += 1

    # Cell vectors if ibrav == 0
    cell_bohr = None
    if ibrav == 0:
        cell_bohr = np.zeros((3, 3))
        for i in range(3):
            parts = lines[idx].split()
            if len(parts) < 3:
                raise ValueError(f"Malformed cell line in {flfrc_path}: {lines[idx].rstrip()}")
            cell_bohr[i] = [float(x) for x in parts[:3]]
            idx += 1

    # Atom lines. In the files you are producing, these are typically:
    #   x y z ityp
    atom_types = []
    atom_positions = []
    for _ in range(nat):
        parts = lines[idx].split()
        if len(parts) < 4:
            raise ValueError(f"Malformed atom line in {flfrc_path}: {lines[idx].rstrip()}")
        x, y, z = map(float, parts[:3])
        ityp = int(parts[3])
        atom_positions.append((x, y, z))
        atom_types.append(ityp)
        idx += 1

    # has_zstar
    has_zstar = "T" in lines[idx].upper()
    idx += 1

    if has_zstar:
        # epsilon (3x3)
        for _ in range(3):
            idx += 1
        # Z* for each atom (3x3 each)
        for _ in range(nat):
            idx += 1
            for _ in range(3):
                idx += 1

    # nr1 nr2 nr3
    parts = lines[idx].split()
    nr1, nr2, nr3 = int(parts[0]), int(parts[1]), int(parts[2])
    idx += 1

    fc2_r = {}

    for na in range(1, nat + 1):
        for nb in range(1, nat + 1):
            idx += 1
            for _alpha in range(1, 4):
                for _beta in range(1, 4):
                    parts = lines[idx].split()
                    i_dir, j_dir = int(parts[0]), int(parts[1])
                    idx += 1
                    for _ in range(nr1 * nr2 * nr3):
                        parts = lines[idx].split()
                        m1, m2, m3 = int(parts[0]), int(parts[1]), int(parts[2])
                        fc_val = float(parts[3])
                        idx += 1
                        key = (m1, m2, m3, na, nb)
                        if key not in fc2_r:
                            fc2_r[key] = np.zeros((3, 3))
                        fc2_r[key][i_dir - 1, j_dir - 1] = fc_val

    fc2 = np.zeros((n_super, n_super, 3, 3))
    for (m1, m2, m3, na, nb), tensor in fc2_r.items():
        l1 = (m1 - 1) % n1
        l2 = (m2 - 1) % n2
        l3 = (m3 - 1) % n3

        for t1 in range(n1):
            for t2 in range(n2):
                for t3 in range(n3):
                    t_idx = t1 * n2 * n3 + t2 * n3 + t3
                    i_t = t_idx * nat + (na - 1)

                    jl1 = (l1 + t1) % n1
                    jl2 = (l2 + t2) % n2
                    jl3 = (l3 + t3) % n3
                    jlp = jl1 * n2 * n3 + jl2 * n3 + jl3
                    j_t = jlp * nat + (nb - 1)

                    fc2[i_t, j_t] = tensor * RY_BOHR2_TO_EV_ANG2

    info = {
        "ntyp": ntyp,
        "masses": masses,
        "nr": (nr1, nr2, nr3),
        "atom_types": atom_types,
        "atom_positions": atom_positions,
        "cell_bohr": cell_bohr,
    }
    return fc2, info


def _parse_mat3r_fc3(
        mat3r_path: Path,
        nat: int,
        q_mesh: list[int],
) -> np.ndarray:
    """Parse d3_qq2rr.x mat3R output into dense FC3 array."""
    n1, n2, n3 = q_mesh
    n_images = n1 * n2 * n3
    n_super = n_images * nat

    fc3 = np.zeros((nat, n_super, n_super, 3, 3, 3))

    with open(mat3r_path) as f:
        header = f.readline().split()
        nq1_file, nq2_file, nq3_file = int(header[0]), int(header[1]), int(header[2])
        if [nq1_file, nq2_file, nq3_file] != q_mesh:
            raise ValueError(
                f"q_mesh mismatch: config={q_mesh}, "
                f"file=[{nq1_file},{nq2_file},{nq3_file}]"
            )

        nat_file = int(f.readline().split()[0])
        if nat_file != nat:
            raise ValueError(f"nat mismatch: config={nat}, file={nat_file}")

        n_entries_per_block = nat ** 3 * 27

        n_blocks = 0
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            r1 = np.array([int(parts[0]), int(parts[1]), int(parts[2])])
            r2 = np.array([int(parts[3]), int(parts[4]), int(parts[5])])

            for _ in range(n_entries_per_block):
                parts = f.readline().split()
                if not parts:
                    raise ValueError("Unexpected end of mat3R block while reading FC3 data.")

                k1 = int(parts[0]) - 1
                k2 = int(parts[1]) - 1
                k3 = int(parts[2]) - 1
                a1 = int(parts[3]) - 1
                a2 = int(parts[4]) - 1
                a3 = int(parts[5]) - 1
                re_val = float(parts[6])

                if abs(re_val) < 1e-30:
                    continue

                l1_j = r1[0] % n1
                l2_j = r1[1] % n2
                l3_j = r1[2] % n3
                lp_j = l1_j * n2 * n3 + l2_j * n3 + l3_j
                j = lp_j * nat + k2

                l1_k = r2[0] % n1
                l2_k = r2[1] % n2
                l3_k = r2[2] % n3
                lp_k = l1_k * n2 * n3 + l2_k * n3 + l3_k
                k = lp_k * nat + k3

                fc3[k1, j, k, a1, a2, a3] = re_val * RY_BOHR3_TO_EV_ANG3

            n_blocks += 1

    print(f"  Parsed {n_blocks} (R1, R2) blocks from mat3R")
    print(f"  FC3 shape: {fc3.shape}, max: {np.max(np.abs(fc3)):.4e} eV/A^3")
    return fc3


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------


def sow(
        cell: PhonopyAtoms,
        work_dir: Path,
        qe_config: QEConfig,
        dfpt_config: DFPTConfig,
) -> int:
    """Generate all DFPT input files."""
    from .qe_interface import write_qe_scf_input

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "results").mkdir(exist_ok=True)
    (work_dir / D3_OUT_DIRNAME).mkdir(exist_ok=True)
    (work_dir / D3_TMP_DIRNAME).mkdir(exist_ok=True)
    (work_dir / DRHO_DIRNAME).mkdir(exist_ok=True)

    pseudo_dir_abs = Path(qe_config.pseudo_dir).resolve()
    try:
        pseudo_dir_rel = pseudo_dir_abs.relative_to(work_dir.resolve())
    except ValueError:
        pseudo_dir_rel = pseudo_dir_abs

    qe_local = replace(
        qe_config,
        pseudo_dir=str(pseudo_dir_rel),
        kpoints_scf=dfpt_config.kpoints,
    )

    write_qe_scf_input(
        work_dir / "scf.in",
        cell,
        qe_local,
        prefix=DFPT_PREFIX,
        relative_paths=True,
    )
    print(f"  Wrote scf.in (k-mesh: {dfpt_config.kpoints})")

    _write_ph_input(work_dir / "ph.in", dfpt_config)
    print(f"  Wrote ph.in (q-mesh: {dfpt_config.q_mesh}, drho_star enabled)")

    _write_q2r_input(work_dir / "q2r.in")
    print("  Wrote q2r.in")

    _write_d3q_input(work_dir / "d3q.in", dfpt_config)
    print(f"  Wrote d3q.in (mode={D3_MODE}, q-mesh: {dfpt_config.q_mesh})")

    _write_d3_asr_input(work_dir / "d3_asr.in", dfpt_config)
    print(f"  Wrote d3_asr.in (asr={dfpt_config.asr})")

    if dfpt_config.sparse_thr is not None:
        _write_d3_sparse_input(work_dir / "d3_sparse.in", dfpt_config)
        print(f"  Wrote d3_sparse.in (thr={dfpt_config.sparse_thr})")

    return 1


def run_scf(work_dir: Path, qe_config: QEConfig, timeout: int = 3600) -> None:
    _run_step(
        work_dir,
        qe_config.pw_command,
        "scf.in",
        "scf.out",
        timeout,
        label="SCF",
    )


def run_ph(work_dir: Path, dfpt_config: DFPTConfig) -> None:
    _run_step(
        work_dir,
        dfpt_config.ph_command,
        "ph.in",
        "ph.out",
        dfpt_config.ph_timeout,
        label="ph.x",
        required_files=("FILDRHO",),
    )


def run_q2r(work_dir: Path, dfpt_config: DFPTConfig) -> None:
    _run_step(
        work_dir,
        dfpt_config.q2r_command,
        "q2r.in",
        "q2r.out",
        300,
        label="q2r",
        required_files=("fc2.dat",),
        input_mode="stdin",
    )


def run_d3q(work_dir: Path, dfpt_config: DFPTConfig) -> None:
    work_dir = Path(work_dir)

    # Ensure temp/output dirs exist even if user deleted them between sow and run
    (work_dir / D3_TMP_DIRNAME).mkdir(parents=True, exist_ok=True)
    (work_dir / D3_OUT_DIRNAME).mkdir(parents=True, exist_ok=True)

    _run_step(
        work_dir,
        dfpt_config.d3q_command,
        "d3q.in",
        "d3q.out",
        dfpt_config.d3q_timeout,
        label="d3q",
    )

    d3_files = sorted((work_dir / D3_OUT_DIRNAME).glob(f"{D3_OUT_PREFIX}*"))
    if not d3_files:
        raise RuntimeError("d3q completed with return code 0, but no D3 XML outputs were found.")


def run_qq2rr(work_dir: Path, dfpt_config: DFPTConfig) -> None:
    work_dir = Path(work_dir).resolve()
    nq1, nq2, nq3 = dfpt_config.q_mesh
    out = work_dir / "qq2rr.out"
    mat3r = work_dir / "mat3R"

    d3_files = sorted((work_dir / D3_OUT_DIRNAME).glob(f"{D3_OUT_PREFIX}*"))
    if not d3_files:
        raise FileNotFoundError(f"No {D3_OUT_PREFIX}* files found in {D3_OUT_DIRNAME}")

    stdin_list = "\n".join(str(p.relative_to(work_dir)) for p in d3_files) + "\n"
    cmd = f"{dfpt_config.d3_qq2rr_command} {nq1} {nq2} {nq3} -o mat3R"

    print(f"  [qq2rr] Running with {len(d3_files)} D3 files ...")
    result = subprocess.run(
        cmd,
        shell=True,
        input=stdin_list,
        capture_output=True,
        text=True,
        cwd=str(work_dir),
        timeout=600,
    )

    with open(out, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    if result.returncode != 0:
        print(result.stdout[-2000:])
        if result.stderr:
            print(f"  stderr (last 1000 chars): {result.stderr[-1000:]}")
        raise RuntimeError(f"Step 'qq2rr' failed with return code {result.returncode}")

    if not mat3r.exists():
        raise RuntimeError("qq2rr finished but mat3R was not created.")


def run_d3_asr(work_dir: Path, dfpt_config: DFPTConfig) -> None:
    work_dir = Path(work_dir).resolve()
    out = work_dir / "d3_asr.out"
    mat3r_in = work_dir / "mat3R"
    mat3r_out = work_dir / "mat3R.asr"

    if not mat3r_in.exists():
        raise FileNotFoundError(f"FC3 file not found: {mat3r_in}")

    cmd = (
        f"{dfpt_config.d3_asr_command} "
        f"-i {mat3r_in.name} "
        f"-o {mat3r_out.name}"
    )

    print(f"  [d3_asr] Running {cmd} ...")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(work_dir),
        timeout=600,
    )

    with open(out, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    if result.returncode != 0:
        if result.stderr:
            print(f"  stderr (last 1000 chars): {result.stderr[-1000:]}")
        raise RuntimeError(f"Step 'd3_asr' failed with return code {result.returncode}")

    if not mat3r_out.exists():
        raise RuntimeError("d3_asr finished but mat3R.asr was not created.")

    mat3r_out.replace(mat3r_in)


def run_d3_sparse(work_dir: Path, dfpt_config: DFPTConfig) -> None:
    _run_step(
        work_dir,
        dfpt_config.d3_sparse_command,
        "d3_sparse.in",
        "d3_sparse.out",
        600,
        label="d3_sparse",
        input_mode="stdin",
    )


def run_all(
        work_dir: Path,
        qe_config: QEConfig,
        dfpt_config: DFPTConfig,
) -> None:
    """Execute the full DFPT workflow."""
    work_dir = Path(work_dir)
    use_sparse = dfpt_config.sparse_thr is not None
    n_steps = 7 if use_sparse else 6

    print(f"Step 1/{n_steps}: SCF")
    run_scf(work_dir, qe_config)

    print(f"Step 2/{n_steps}: ph.x DFPT")
    run_ph(work_dir, dfpt_config)

    print(f"Step 3/{n_steps}: q2r.x (FC2)")
    run_q2r(work_dir, dfpt_config)

    print(f"Step 4/{n_steps}: d3q.x (FC3 full grid)")
    run_d3q(work_dir, dfpt_config)

    print(f"Step 5/{n_steps}: d3_qq2rr.x (FC3 real-space)")
    run_qq2rr(work_dir, dfpt_config)

    print(f"Step 6/{n_steps}: d3_asr.x (acoustic sum rule)")
    run_d3_asr(work_dir, dfpt_config)

    if use_sparse:
        print(f"Step 7/{n_steps}: d3_sparse.x (sparsification)")
        run_d3_sparse(work_dir, dfpt_config)


def reap(
        work_dir: Path,
        cell: PhonopyAtoms,
        q_mesh: list[int],
) -> Path:
    """Parse DFPT outputs and produce fc3.hdf5."""
    import h5py

    work_dir = Path(work_dir)
    nat = len(cell.symbols)

    fc2_file = work_dir / "fc2.dat"
    if not fc2_file.exists():
        raise FileNotFoundError(f"FC2 file not found: {fc2_file}")
    print("Parsing FC2 from q2r.x output...")
    fc2, _fc2_info = _parse_q2r_fc2(fc2_file, nat, q_mesh)
    print(f"  FC2 shape: {fc2.shape}, max: {np.max(np.abs(fc2)):.4e} eV/A^2")

    candidate_files = [
        work_dir / "mat3R.asr.sparse",
        work_dir / "mat3R.asr",
        work_dir / "mat3R",
    ]
    mat3r_file = next((p for p in candidate_files if p.exists()), None)
    if mat3r_file is None:
        raise FileNotFoundError("No FC3 file found: expected mat3R, mat3R.asr, or mat3R.asr.sparse")
    print("Parsing FC3 from d3_qq2rr.x output...")
    fc3 = _parse_mat3r_fc3(mat3r_file, nat, q_mesh)

    fc3_path = work_dir / "fc3.hdf5"
    with h5py.File(fc3_path, "w") as f:
        f.create_dataset("fc3", data=fc3, compression="gzip")
        f.create_dataset("fc2", data=fc2, compression="gzip")
        f.attrs["method"] = "dfpt"
        f.attrs["q_mesh"] = q_mesh
        f.attrs["nat"] = nat
    print(f"  Saved: {fc3_path} ({fc3_path.stat().st_size / 1e6:.1f} MB)")

    return fc3_path


def generate_fc_dfpt(
        cell: PhonopyAtoms,
        work_dir: Path,
        qe_config: QEConfig,
        dfpt_config: DFPTConfig,
) -> Path:
    """Full DFPT pipeline: sow + run + reap."""
    sow(cell, work_dir, qe_config, dfpt_config)
    run_all(work_dir, qe_config, dfpt_config)
    return reap(work_dir, cell, dfpt_config.q_mesh)
