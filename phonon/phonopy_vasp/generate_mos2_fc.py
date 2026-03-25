#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


MOS2_POSCAR_TEMPLATE = """MoS2-monolayer
1.0
{a1: .10f} {a2: .10f} {a3: .10f}
{b1: .10f} {b2: .10f} {b3: .10f}
{c1: .10f} {c2: .10f} {c3: .10f}
Mo S
1 2
Direct
0.0000000000 0.0000000000 0.5000000000
0.3333333333 0.6666666667 {z_top: .10f}
0.6666666667 0.3333333333 {z_bottom: .10f}
"""


DEFAULT_INCAR = """SYSTEM = MoS2 phonon displacement
PREC = Accurate
ENCUT = 520
EDIFF = 1E-8
IBRION = -1
NSW = 0
ISIF = 2
ISMEAR = 0
SIGMA = 0.05
LREAL = Auto
ADDGRID = .TRUE.
LWAVE = .FALSE.
LCHARG = .FALSE.
NCORE = 4
"""


DEFAULT_RELAX_INCAR = """SYSTEM = MoS2 relaxation
PREC = Accurate
ENCUT = 520
EDIFF = 1E-8
EDIFFG = -1E-3
IBRION = 2
NSW = 150
ISIF = 3
ISMEAR = 0
SIGMA = 0.05
LREAL = Auto
ADDGRID = .TRUE.
LWAVE = .FALSE.
LCHARG = .FALSE.
NCORE = 4
"""


DEFAULT_KPOINTS = """Automatic mesh
0
Gamma
3 3 1
0 0 0
"""


DEFAULT_RELAX_KPOINTS = """Automatic mesh
0
Gamma
12 12 1
0 0 0
"""


def run_cmd(args: list[str], cwd: Path, stdin_text: str | None = None) -> None:
    print(f"[run] ({cwd}) {' '.join(args)}")
    subprocess.run(
        args,
        cwd=str(cwd),
        input=stdin_text,
        text=True,
        check=True,
    )


def ensure_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Command not found in PATH: {name}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_mos2_poscar(path: Path, lattice_a: float, vacuum_c: float, sulfur_height: float) -> None:
    if sulfur_height <= 0:
        raise ValueError("sulfur_height must be positive (in Angstrom)")
    if vacuum_c <= 2.0 * sulfur_height:
        raise ValueError("vacuum_c is too small for the requested sulfur_height")

    z_shift = sulfur_height / vacuum_c
    z_top = 0.5 + z_shift
    z_bottom = 0.5 - z_shift

    poscar_text = MOS2_POSCAR_TEMPLATE.format(
        a1=lattice_a,
        a2=0.0,
        a3=0.0,
        b1=-0.5 * lattice_a,
        b2=(3.0**0.5) * 0.5 * lattice_a,
        b3=0.0,
        c1=0.0,
        c2=0.0,
        c3=vacuum_c,
        z_top=z_top,
        z_bottom=z_bottom,
    )
    path.write_text(poscar_text)


def prepare_reference_poscar(
    target_path: Path,
    lattice_a: float,
    vacuum_c: float,
    sulfur_height: float,
    input_poscar: Path | None,
) -> None:
    if input_poscar is not None:
        shutil.copy2(input_poscar, target_path)
        return
    write_mos2_poscar(target_path, lattice_a, vacuum_c, sulfur_height)


def copy_vasp_templates(template_dir: Path | None, target_dir: Path) -> None:
    if template_dir is None:
        return
    for name in ("INCAR", "KPOINTS", "POTCAR"):
        src = template_dir / name
        if src.exists():
            shutil.copy2(src, target_dir / name)


def ensure_default_vasp_inputs(target_dir: Path) -> None:
    incar = target_dir / "INCAR"
    if not incar.exists():
        incar.write_text(DEFAULT_INCAR)

    kpoints = target_dir / "KPOINTS"
    if not kpoints.exists():
        kpoints.write_text(DEFAULT_KPOINTS)


def ensure_default_relax_inputs(target_dir: Path) -> None:
    incar = target_dir / "INCAR"
    if not incar.exists():
        incar.write_text(DEFAULT_RELAX_INCAR)

    kpoints = target_dir / "KPOINTS"
    if not kpoints.exists():
        kpoints.write_text(DEFAULT_RELAX_KPOINTS)


def ensure_default_potcar(
    target_dir: Path,
    potpaw_root: Path,
    mo_potential: str,
    s_potential: str,
) -> None:
    potcar = target_dir / "POTCAR"
    if potcar.exists():
        return

    mo_src = potpaw_root / mo_potential / "POTCAR"
    s_src = potpaw_root / s_potential / "POTCAR"
    missing = [str(path) for path in (mo_src, s_src) if not path.exists()]
    if missing:
        raise RuntimeError(
            "Missing POTCAR source(s): " + ", ".join(missing) + ". "
            "Set --potpaw-root/--mo-potential/--s-potential appropriately."
        )

    with potcar.open("wb") as out:
        out.write(mo_src.read_bytes())
        out.write(s_src.read_bytes())


def write_manifest(manifest_path: Path, rel_paths: list[str]) -> None:
    manifest_path.write_text("\n".join(rel_paths) + "\n")


def read_manifest(manifest_path: Path) -> list[str]:
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest: {manifest_path}")
    lines = [line.strip() for line in manifest_path.read_text().splitlines()]
    return [line for line in lines if line]


def is_complete_vasprun_xml(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    tail = path.read_bytes()[-8192:].decode("utf-8", errors="ignore")
    return "</modeling>" in tail


def prepare_harmonic(
    root: Path,
    phonopy_cmd: str,
    dim: str,
    lattice_a: float,
    vacuum_c: float,
    sulfur_height: float,
    input_poscar: Path | None,
    potpaw_root: Path,
    mo_potential: str,
    s_potential: str,
    template_dir: Path | None,
) -> int:
    harmonic = root / "harmonic"
    ensure_dir(harmonic)
    prepare_reference_poscar(
        target_path=harmonic / "POSCAR",
        lattice_a=lattice_a,
        vacuum_c=vacuum_c,
        sulfur_height=sulfur_height,
        input_poscar=input_poscar,
    )

    run_cmd([phonopy_cmd, "-d", f"--dim={dim}", "-c", "POSCAR"], cwd=harmonic)

    displacement_files = sorted(harmonic.glob("POSCAR-*"))
    if not displacement_files:
        raise RuntimeError("No harmonic displacements generated by phonopy.")

    vasp_root = harmonic / "vasp"
    if vasp_root.exists():
        shutil.rmtree(vasp_root)
    ensure_dir(vasp_root)

    manifest: list[str] = []
    for idx, poscar_file in enumerate(displacement_files, start=1):
        run_dir = vasp_root / f"disp-{idx:03d}"
        ensure_dir(run_dir)
        shutil.copy2(poscar_file, run_dir / "POSCAR")
        copy_vasp_templates(template_dir, run_dir)
        ensure_default_vasp_inputs(run_dir)
        ensure_default_potcar(run_dir, potpaw_root, mo_potential, s_potential)
        manifest.append(str((run_dir / "vasprun.xml").relative_to(harmonic)))

    write_manifest(harmonic / "harmonic_manifest.txt", manifest)
    return len(displacement_files)


def prepare_anharmonic(
    root: Path,
    thirdorder_cmd: str,
    supercell: tuple[int, int, int],
    cutoff: str,
    lattice_a: float,
    vacuum_c: float,
    sulfur_height: float,
    input_poscar: Path | None,
    potpaw_root: Path,
    mo_potential: str,
    s_potential: str,
    template_dir: Path | None,
) -> int:
    anharmonic = root / "anharmonic"
    ensure_dir(anharmonic)
    prepare_reference_poscar(
        target_path=anharmonic / "POSCAR",
        lattice_a=lattice_a,
        vacuum_c=vacuum_c,
        sulfur_height=sulfur_height,
        input_poscar=input_poscar,
    )

    na, nb, nc = supercell
    run_cmd([thirdorder_cmd, "sow", str(na), str(nb), str(nc), cutoff], cwd=anharmonic)

    displacement_files = sorted(anharmonic.glob("3RD.POSCAR.*"))
    if not displacement_files:
        raise RuntimeError("No anharmonic displacements generated by thirdorder.")

    vasp_root = anharmonic / "vasp"
    if vasp_root.exists():
        shutil.rmtree(vasp_root)
    ensure_dir(vasp_root)

    manifest: list[str] = []
    for idx, poscar_file in enumerate(displacement_files, start=1):
        run_dir = vasp_root / f"disp-{idx:05d}"
        ensure_dir(run_dir)
        shutil.copy2(poscar_file, run_dir / "POSCAR")
        copy_vasp_templates(template_dir, run_dir)
        ensure_default_vasp_inputs(run_dir)
        ensure_default_potcar(run_dir, potpaw_root, mo_potential, s_potential)
        manifest.append(str((run_dir / "vasprun.xml").relative_to(anharmonic)))

    write_manifest(anharmonic / "anharmonic_manifest.txt", manifest)
    return len(displacement_files)


def collect_harmonic(root: Path, phonopy_cmd: str, dim: str) -> None:
    harmonic = root / "harmonic"
    manifest = read_manifest(harmonic / "harmonic_manifest.txt")

    for rel in manifest:
        path = harmonic / rel
        if not path.exists():
            raise RuntimeError(f"Missing VASP output: {path}")
        if not is_complete_vasprun_xml(path):
            raise RuntimeError(
                f"Incomplete VASP output: {path}. "
                "vasprun.xml appears truncated (missing </modeling>). "
                "Rerun this displacement job before collect."
            )

    run_cmd([phonopy_cmd, "-f", *manifest], cwd=harmonic)

    try:
        run_cmd([phonopy_cmd, "--fc", "FORCE_SETS"], cwd=harmonic)
    except subprocess.CalledProcessError:
        run_cmd([phonopy_cmd, f"--dim={dim}", "-c", "POSCAR", "--writefc"], cwd=harmonic)


def collect_anharmonic(
    root: Path,
    thirdorder_cmd: str,
    supercell: tuple[int, int, int],
    cutoff: str,
) -> None:
    anharmonic = root / "anharmonic"
    manifest = read_manifest(anharmonic / "anharmonic_manifest.txt")

    for rel in manifest:
        path = anharmonic / rel
        if not path.exists():
            raise RuntimeError(f"Missing VASP output: {path}")
        if not is_complete_vasprun_xml(path):
            raise RuntimeError(
                f"Incomplete VASP output: {path}. "
                "vasprun.xml appears truncated (missing </modeling>). "
                "Rerun this displacement job before collect."
            )

    stdin_text = "\n".join(manifest) + "\n"
    na, nb, nc = supercell
    run_cmd(
        [thirdorder_cmd, "reap", str(na), str(nb), str(nc), cutoff],
        cwd=anharmonic,
        stdin_text=stdin_text,
    )


def parse_supercell(supercell: str) -> tuple[int, int, int]:
    fields = supercell.split()
    if len(fields) != 3:
        raise ValueError("supercell must have three integers, e.g. '4 4 1'")
    values = tuple(int(i) for i in fields)
    if min(values) < 1:
        raise ValueError("supercell integers must be positive")
    return values


def prepare_relaxation(
    root: Path,
    lattice_a: float,
    vacuum_c: float,
    sulfur_height: float,
    input_poscar: Path | None,
    potpaw_root: Path,
    mo_potential: str,
    s_potential: str,
    template_dir: Path | None,
) -> Path:
    relax = root / "relax"
    ensure_dir(relax)

    prepare_reference_poscar(
        target_path=relax / "POSCAR",
        lattice_a=lattice_a,
        vacuum_c=vacuum_c,
        sulfur_height=sulfur_height,
        input_poscar=input_poscar,
    )
    copy_vasp_templates(template_dir, relax)
    ensure_default_relax_inputs(relax)
    ensure_default_potcar(relax, potpaw_root, mo_potential, s_potential)
    return relax


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run structural relaxation and generate harmonic/anharmonic force-constant data for monolayer MoS2 "
            "using phonopy+VASP and thirdorder+VASP."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common_options(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--workdir",
            type=Path,
            default=Path("phonon-data/mos2"),
            help="Working directory for generated files.",
        )
        p.add_argument(
            "--lattice-a",
            type=float,
            default=3.18,
            help="MoS2 in-plane lattice constant a in Angstrom.",
        )
        p.add_argument(
            "--vacuum-c",
            type=float,
            default=20.0,
            help="Out-of-plane lattice parameter c (vacuum) in Angstrom.",
        )
        p.add_argument(
            "--sulfur-height",
            type=float,
            default=1.58,
            help="S height from Mo plane in Angstrom.",
        )
        p.add_argument(
            "--input-poscar",
            type=Path,
            default=None,
            help=(
                "Optional relaxed POSCAR/CONTCAR to use as the reference structure. "
                "If set, overrides --lattice-a/--vacuum-c/--sulfur-height."
            ),
        )
        p.add_argument(
            "--supercell",
            default="4 4 1",
            help="Supercell for thirdorder sow/reap, format: 'na nb nc'.",
        )
        p.add_argument(
            "--phonopy-dim",
            default="4 4 1",
            help="Supercell dimension string for phonopy --dim.",
        )
        p.add_argument(
            "--cutoff",
            default="-3",
            help="thirdorder cutoff argument (e.g. -3 for 3rd-neighbor shell, or distance in nm).",
        )
        p.add_argument(
            "--phonopy-cmd",
            default="phonopy",
            help="Executable name/path for phonopy.",
        )
        p.add_argument(
            "--thirdorder-cmd",
            default="thirdorder-vasp",
            help="Executable name/path for thirdorder VASP frontend.",
        )
        p.add_argument(
            "--potpaw-root",
            type=Path,
            default=Path("~/vasp/potpaw_PBE").expanduser(),
            help="Root folder of VASP pseudopotentials.",
        )
        p.add_argument(
            "--mo-potential",
            default="Mo",
            help="Subfolder name under --potpaw-root for Mo POTCAR (e.g. Mo, Mo_pv).",
        )
        p.add_argument(
            "--s-potential",
            default="S",
            help="Subfolder name under --potpaw-root for S POTCAR.",
        )

    p_prepare = subparsers.add_parser("prepare", help="Generate displaced structures for VASP runs.")
    add_common_options(p_prepare)
    p_prepare.add_argument(
        "--template-dir",
        type=Path,
        default=None,
        help="Optional directory containing INCAR, KPOINTS, POTCAR to copy into each displacement folder.",
    )

    p_collect = subparsers.add_parser("collect", help="Collect VASP outputs and build FC data files.")
    add_common_options(p_collect)

    p_relax = subparsers.add_parser("relax", help="Prepare geometry relaxation input files.")
    p_relax.add_argument(
        "--workdir",
        type=Path,
        default=Path("phonon-data/mos2"),
        help="Working directory for generated files.",
    )
    p_relax.add_argument(
        "--lattice-a",
        type=float,
        default=3.18,
        help="MoS2 in-plane lattice constant a in Angstrom.",
    )
    p_relax.add_argument(
        "--vacuum-c",
        type=float,
        default=20.0,
        help="Out-of-plane lattice parameter c (vacuum) in Angstrom.",
    )
    p_relax.add_argument(
        "--sulfur-height",
        type=float,
        default=1.58,
        help="S height from Mo plane in Angstrom.",
    )
    p_relax.add_argument(
        "--input-poscar",
        type=Path,
        default=None,
        help=(
            "Optional initial POSCAR/CONTCAR for relaxation. "
            "If set, overrides --lattice-a/--vacuum-c/--sulfur-height."
        ),
    )
    p_relax.add_argument(
        "--potpaw-root",
        type=Path,
        default=Path("~/vasp/potpaw").expanduser(),
        help="Root folder of VASP pseudopotentials.",
    )
    p_relax.add_argument(
        "--mo-potential",
        default="Mo",
        help="Subfolder name under --potpaw-root for Mo POTCAR (e.g. Mo, Mo_pv).",
    )
    p_relax.add_argument(
        "--s-potential",
        default="S",
        help="Subfolder name under --potpaw-root for S POTCAR.",
    )
    p_relax.add_argument(
        "--template-dir",
        type=Path,
        default=None,
        help="Optional directory containing INCAR, KPOINTS, POTCAR to copy into relax folder.",
    )

    p_prep_relax = subparsers.add_parser(
        "prepare-from-relax",
        help="Prepare harmonic/anharmonic displacements from relaxed CONTCAR (requires relax/ to exist).",
    )
    p_prep_relax.add_argument(
        "--workdir",
        type=Path,
        default=Path("phonon-data/mos2"),
        help="Working directory for generated files.",
    )
    p_prep_relax.add_argument(
        "--supercell",
        default="4 4 1",
        help="Supercell for thirdorder sow/reap, format: 'na nb nc'.",
    )
    p_prep_relax.add_argument(
        "--phonopy-dim",
        default="4 4 1",
        help="Supercell dimension string for phonopy --dim.",
    )
    p_prep_relax.add_argument(
        "--cutoff",
        default="-3",
        help="thirdorder cutoff argument (e.g. -3 for 3rd-neighbor shell, or distance in nm).",
    )
    p_prep_relax.add_argument(
        "--phonopy-cmd",
        default="phonopy",
        help="Executable name/path for phonopy.",
    )
    p_prep_relax.add_argument(
        "--thirdorder-cmd",
        default="thirdorder-vasp",
        help="Executable name/path for thirdorder VASP frontend.",
    )
    p_prep_relax.add_argument(
        "--potpaw-root",
        type=Path,
        default=Path("~/vasp/potpaw").expanduser(),
        help="Root folder of VASP pseudopotentials.",
    )
    p_prep_relax.add_argument(
        "--mo-potential",
        default="Mo",
        help="Subfolder name under --potpaw-root for Mo POTCAR (e.g. Mo, Mo_pv).",
    )
    p_prep_relax.add_argument(
        "--s-potential",
        default="S",
        help="Subfolder name under --potpaw-root for S POTCAR.",
    )
    p_prep_relax.add_argument(
        "--template-dir",
        type=Path,
        default=None,
        help="Optional directory containing INCAR, KPOINTS, POTCAR to copy into each displacement folder.",
    )

    args = parser.parse_args()

    workdir = args.workdir.resolve()
    ensure_dir(workdir)

    if args.mode in {"prepare", "collect", "prepare-from-relax"}:
        ensure_command(args.phonopy_cmd)
        ensure_command(args.thirdorder_cmd)
        supercell = parse_supercell(args.supercell)

    if args.mode == "relax":
        template_dir = args.template_dir.resolve() if args.template_dir else None
        if template_dir and not template_dir.exists():
            raise RuntimeError(f"Template directory does not exist: {template_dir}")

        input_poscar = args.input_poscar.resolve() if args.input_poscar else None
        if input_poscar and not input_poscar.exists():
            raise RuntimeError(f"Input POSCAR/CONTCAR does not exist: {input_poscar}")

        potpaw_root = args.potpaw_root.expanduser().resolve()
        if not potpaw_root.exists():
            raise RuntimeError(f"Pseudopotential root does not exist: {potpaw_root}")

        relax_dir = prepare_relaxation(
            root=workdir,
            lattice_a=args.lattice_a,
            vacuum_c=args.vacuum_c,
            sulfur_height=args.sulfur_height,
            input_poscar=input_poscar,
            potpaw_root=potpaw_root,
            mo_potential=args.mo_potential,
            s_potential=args.s_potential,
            template_dir=template_dir,
        )
        print(f"Prepared relaxation inputs in: {relax_dir}")
        print("Run VASP relaxation there. Then run mode 'prepare' to generate displacement structures.")
        print("If relax/CONTCAR exists, mode 'prepare' will use it automatically.")
        return 0

    if args.mode == "prepare-from-relax":
        relaxed_contcar = workdir / "relax" / "CONTCAR"
        if not relaxed_contcar.exists():
            raise RuntimeError(
                f"Missing relaxed CONTCAR: {relaxed_contcar}\n"
                "Run mode 'relax' first, then VASP relaxation, before using prepare-from-relax."
            )

        template_dir = args.template_dir.resolve() if args.template_dir else None
        if template_dir and not template_dir.exists():
            raise RuntimeError(f"Template directory does not exist: {template_dir}")

        potpaw_root = args.potpaw_root.expanduser().resolve()
        if not potpaw_root.exists():
            raise RuntimeError(f"Pseudopotential root does not exist: {potpaw_root}")

        harmonic_count = prepare_harmonic(
            root=workdir,
            phonopy_cmd=args.phonopy_cmd,
            dim=args.phonopy_dim,
            lattice_a=0.0,
            vacuum_c=0.0,
            sulfur_height=0.0,
            input_poscar=relaxed_contcar,
            potpaw_root=potpaw_root,
            mo_potential=args.mo_potential,
            s_potential=args.s_potential,
            template_dir=template_dir,
        )

        anharmonic_count = prepare_anharmonic(
            root=workdir,
            thirdorder_cmd=args.thirdorder_cmd,
            supercell=supercell,
            cutoff=args.cutoff,
            lattice_a=0.0,
            vacuum_c=0.0,
            sulfur_height=0.0,
            input_poscar=relaxed_contcar,
            potpaw_root=potpaw_root,
            mo_potential=args.mo_potential,
            s_potential=args.s_potential,
            template_dir=template_dir,
        )
        print(f"Prepared harmonic displacements from relaxed CONTCAR: {harmonic_count}")
        print(f"Prepared anharmonic displacements from relaxed CONTCAR: {anharmonic_count}")
        print("Run VASP in each generated displacement directory, then run this script with mode 'collect'.")
        return 0

    if args.mode == "prepare":
        template_dir = args.template_dir.resolve() if args.template_dir else None
        if template_dir and not template_dir.exists():
            raise RuntimeError(f"Template directory does not exist: {template_dir}")

        input_poscar = args.input_poscar.resolve() if args.input_poscar else None
        if input_poscar and not input_poscar.exists():
            raise RuntimeError(f"Input POSCAR/CONTCAR does not exist: {input_poscar}")
        if input_poscar is None:
            auto_relaxed = workdir / "relax" / "CONTCAR"
            if auto_relaxed.exists():
                input_poscar = auto_relaxed.resolve()
                print(f"Using relaxed structure from: {input_poscar}")

        potpaw_root = args.potpaw_root.expanduser().resolve()
        if not potpaw_root.exists():
            raise RuntimeError(f"Pseudopotential root does not exist: {potpaw_root}")

        harmonic_count = prepare_harmonic(
            root=workdir,
            phonopy_cmd=args.phonopy_cmd,
            dim=args.phonopy_dim,
            lattice_a=args.lattice_a,
            vacuum_c=args.vacuum_c,
            sulfur_height=args.sulfur_height,
            input_poscar=input_poscar,
            potpaw_root=potpaw_root,
            mo_potential=args.mo_potential,
            s_potential=args.s_potential,
            template_dir=template_dir,
        )

        anharmonic_count = prepare_anharmonic(
            root=workdir,
            thirdorder_cmd=args.thirdorder_cmd,
            supercell=supercell,
            cutoff=args.cutoff,
            lattice_a=args.lattice_a,
            vacuum_c=args.vacuum_c,
            sulfur_height=args.sulfur_height,
            input_poscar=input_poscar,
            potpaw_root=potpaw_root,
            mo_potential=args.mo_potential,
            s_potential=args.s_potential,
            template_dir=template_dir,
        )
        print(f"Prepared harmonic displacements: {harmonic_count}")
        print(f"Prepared anharmonic displacements: {anharmonic_count}")
        print("Run VASP in each generated displacement directory, then run this script with mode 'collect'.")
        return 0

    collect_harmonic(root=workdir, phonopy_cmd=args.phonopy_cmd, dim=args.phonopy_dim)
    collect_anharmonic(
        root=workdir,
        thirdorder_cmd=args.thirdorder_cmd,
        supercell=supercell,
        cutoff=args.cutoff,
    )
    print("Generated harmonic data in harmonic/FORCE_SETS (and FORCE_CONSTANTS if supported).")
    print("Generated anharmonic data in anharmonic/FORCE_CONSTANTS_3RD.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
