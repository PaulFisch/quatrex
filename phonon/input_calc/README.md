# Phonon Transport Input Generation

Computes harmonic (FC2) and anharmonic (FC3) force constants for
NEGF phonon transport using phono3py + symfc + Quantum ESPRESSO.

## Setup

### 1. Create conda environment

```bash
conda create -n quatrex-dev python=3.13
conda activate quatrex-dev
```

### 2. Install dependencies

```bash
# Core scientific stack
pip install numpy scipy matplotlib pyyaml h5py

# Phonon libraries
pip install phonopy phono3py symfc

# Quantum ESPRESSO (must be installed separately)
# Ensure pw.x is available in PATH or specify via config
```

### 3. Pseudopotentials

PBE pseudopotentials are in `pseudo/`. The config references them via
`qe.pseudo_dir`. If running QE from a different machine, copy or symlink
the `pseudo/` directory so the relative path resolves.

## Workflow

### Full pipeline (from scratch)

```bash
# Everything in one command: vc-relax -> FC2 -> FC3
python -m phonon_inputs pipeline --config config_prim.yaml

# Or skip relaxation if lattice constant is known:
python -m phonon_inputs pipeline --config config_prim.yaml --skip-relax

# Or skip FC3 (harmonic only):
python -m phonon_inputs pipeline --config config_prim.yaml --skip-relax --skip-fc3
```

### Step-by-step

The pipeline uses the 2-atom FCC primitive cell for Si. Config file:
`config_prim.yaml`.

**Step 1: Generate displaced supercells**

```bash
python -m phonon_inputs fc3-sow --config config_prim.yaml
```

Creates `fc3_prim/` with 114 QE input files (2x2x2 supercell, 16 atoms each).

**Step 2: Run QE calculations**

```bash
python -m phonon_inputs fc3-run --config config_prim.yaml
```

Runs `pw.x` for each displacement. Skips completed jobs on restart.
For cluster submission, run each `fc3_prim/disp-NNNNN.in` independently:

```bash
mpirun -np 4 pw.x -in disp-00001.in > disp-00001.out
```

**Step 3: Extract force constants**

```bash
python -m phonon_inputs fc3-reap --config config_prim.yaml
```

Reads QE forces and produces `fc3_prim/fc3.hdf5` containing both FC2 and
FC3 via symfc.

**Step 4: Run anharmonic transport**

```bash
# Ballistic + SCBA at 300K (primitive cell, default)
python anharmonic/run_anharmonic.py --nk 4 --nfreq 101 --max-iter 20

# Multi-slab thickness sweep
python anharmonic/run_anharmonic.py --n-slabs 1 3 5 10 --resume

# Legacy conventional cell (8-atom, requires old/ data)
python anharmonic/run_anharmonic.py --cell conventional
```

### Example: Si from scratch

```bash
# Generate QE inputs for vc-relax + FC2 + FC3:
python examples/setup_si_primitive.py --sow-only

# Or run everything (if QE is available):
python examples/setup_si_primitive.py
```

### Validation scripts

```bash
python anharmonic/test_anharmonic.py               # Basic SCBA test
python anharmonic/test_anharmonic_comparison.py    # Literature comparison + T-sweep
python anharmonic/test_anharmonic_multislab.py     # Thickness sweep with checkpoints
python anharmonic/test_anharmonic_neighbors.py     # FC3 cutoff convergence study
```

## Config reference

See `config_prim.yaml` for the full configuration. Key sections:

- `structure`: Unit cell definition (inline or from file)
- `supercell`: Supercell matrix and displacement distance
- `qe`: Quantum ESPRESSO parameters (ecutwfc, k-points, pseudo_dir)
- `relax`: Structural relaxation (vc-relax or relax)
- `thirdorder`: FC3 parameters (supercell, pair cutoff, symfc)
- `block_extraction`: q-mesh and transport direction for NEGF blocks
- `quatrex_output`: Output format for the quatrex GPU solver

## CLI commands

```
python -m phonon_inputs pipeline       # Full pipeline: relax -> FC2 -> FC3
python -m phonon_inputs generate       # FC2 pipeline: displacements -> QE -> blocks
python -m phonon_inputs fc3-sow        # Generate phono3py displacements
python -m phonon_inputs fc3-run        # Run QE for FC3 displacements
python -m phonon_inputs fc3-reap       # Produce FC3 via symfc
python -m phonon_inputs fc3-all        # Full FC3: sow + run + reap
python -m phonon_inputs extract-blocks # Extract NEGF blocks from existing FC
python -m phonon_inputs validate       # Check band structure + transmission
```

## File structure

```
input_calc/
  config_prim.yaml          # Main config (2-atom FCC primitive cell)
  pseudo/                   # QE pseudopotentials
  phonon_inputs/            # Python package
    pipeline.py             # End-to-end: relax -> FC2 -> FC3
    anharmonic.py           # SCBA implementation (THz^2 units)
    cli.py                  # Command-line interface
    config.py               # Config dataclasses
    constants.py            # Physical constants and unit conversions
    convention.py           # Block extraction (Convention A/B)
    force_constants.py      # FC2/FC3 loading
    qe_interface.py         # QE input/output + relaxation
    quatrex_writer.py       # Write quatrex input files
    structure.py            # Structure loading
    thirdorder.py           # phono3py + symfc FC3 pipeline
    validation.py           # Transmission and conductance checks
  anharmonic/               # SCBA transport scripts
    run_anharmonic.py       # Main driver (ballistic + SCBA)
    test_anharmonic*.py     # Validation and comparison scripts
  examples/                 # Standalone examples
    setup_si_primitive.py   # Si from scratch: relax -> FC2 -> FC3
    si_ge_interface.py      # Si/Ge interface model
  fc3_prim/                 # phono3py work directory (generated)
  relax/                    # QE relaxation (generated)
  old/                      # Previous calculations (backup)
```
