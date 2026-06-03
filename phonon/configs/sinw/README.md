# SiNW(100) hiphive workflows

Twelve configs: three diameters × two sampling strategies × two supercell lengths.

**Default ([1,1,2] supercell along z, L/2 ≈ 5.43 Å):**

| Diameter | Atoms (primitive / `[1,1,2]`) | mc-rattle | phonon-rattle |
|----------|------------------------------:|-----------|----------------|
| ~5 Å     | 21 / 42                       | `sinw100_d5a_vasp.yaml`  | `sinw100_d5a_vasp_phonon.yaml`  |
| ~9 Å     | 37 / 74                       | `sinw100_d9a_vasp.yaml`  | `sinw100_d9a_vasp_phonon.yaml`  |
| ~12 Å    | 51 / 102                      | `sinw100_d12a_vasp.yaml` | `sinw100_d12a_vasp_phonon.yaml` |

**Extended ([1,1,4] supercell along z, L/2 ≈ 10.86 Å):**

| Diameter | Atoms (primitive / `[1,1,4]`) | mc-rattle | phonon-rattle |
|----------|------------------------------:|-----------|----------------|
| ~5 Å     | 21 / 84                       | `sinw100_d5a_vasp_sc4.yaml`  | `sinw100_d5a_vasp_sc4_phonon.yaml`  |
| ~9 Å     | 37 / 148                      | `sinw100_d9a_vasp_sc4.yaml`  | `sinw100_d9a_vasp_sc4_phonon.yaml`  |
| ~12 Å    | 51 / 204                      | `sinw100_d12a_vasp_sc4.yaml` | `sinw100_d12a_vasp_sc4_phonon.yaml` |

## When to use sc2 vs. sc4

The transport supercell length L sets the largest q-wavelength the FC2 fit
can resolve. Hiphive requires `cutoffs[0] < L/2`; cutoffs **within 20 %
of L/2** routinely produce imaginary modes at the Z-axis zone boundary
because those modes' FCs are determined by very little data.

* **sc2 (default)** — `cutoffs[0]=5.0` Å vs. L/2 = 5.43 Å.
  Sits inside the warning band (`parameter_validation` raises a `warn`).
  Works for sanity-check fits and ballistic transport, but **routinely
  picks up imaginary modes at Z(0.5)** in the more open-geometry wires
  (d9a, d12a). Use only as a smoke test or when you must save DFT cost.
* **sc4 (recommended for production)** — `cutoffs[0]=5.0` Å vs.
  L/2 = 10.86 Å. Comfortable 2× margin; eliminates the zone-boundary
  imaginary modes seen at sc2. DFT cost per structure grows ~4×–8× (it's
  super-linear in N), but `n_structures` drops from 32 → 16, so the
  total DFT time is roughly comparable.

For ALL production work prefer the **sc4** configs. Keep the sc2
configs around for fast iteration / debugging.

Every config encodes the same physics-correct recipe (see
`phonon/phonon_inputs/HIPHIVE_FITTING_NOTES.md` for the *why*):

| Knob                  | Setting                          | Why for SiNW                                                                |
|-----------------------|----------------------------------|-----------------------------------------------------------------------------|
| `fit_method`          | `ardr`                           | hiphive's default for under-/marginally-determined FC3; parameter-free      |
| `rotational_sum_rule` | `post_fit`                       | **Critical for 1-D systems**: closes the spurious ZA-branch gap at Γ        |
| `n_structures`        | 32                               | Master pool; gives 2–4× overdetermination of the FC3 fit at cutoffs `[5,4]` |
| `rattle_std`          | 0.03 Å                           | Si–Si bond strain ~1.3% per step; harmonic regime                            |
| `rattle_n_iter`       | 2                                | Cumulative atomic displacement ~ `rattle_std × n_iter` ≈ 0.06 Å (linear, not √N). Earlier defaults of 10 gave ~0.3 Å peaks — solidly anharmonic |
| `rattle_d_min`        | 1.4 Å                            | Above the H–H equilibrium distance (1.42 Å) — avoids unphysical collisions  |
| `cutoffs[0]` (FC2)    | `2 × envelope + 0.5` Å           | Spans the wire cross-section + 1 NN shell. For d5a → 7.4 Å, d9a → 11.3 Å, d12a → 14.5 Å (capped at 12 Å) |
| `cutoffs[1]` (FC3)    | `4.0` Å                          | FC3 spans 1 NN triplet                                                       |
| `n_structures`        | `4 × ⌈n_super / 8⌉`              | Validator-recommended lower bound. d5a sc4 → 45, d9a sc4 → 76, d12a sc4 → 104 |
| `convergence.sizes`   | `[12, 18, 24, 32]`               | Refit at every size; saturation curve identifies the minimum sufficient n_struct |

These knobs are the outcome of the investigation in
`scratch/imag_audit/INVESTIGATION_NOTES.md`. The previous configs used
`cutoffs[0]=5.0`, `rattle_n_iter=20`, `n_structures=16` — a combination
that produced 45 imag modes for d5a sc4 and 286 imag modes for d9a sc4
(the under-determined fit on the larger wire collapsed the H-derived
band to ω² < 0 in the Γ-point dynamical matrix). The values above are
each individually necessary to bring the dispersion back to clean
acoustic-near-zero behaviour at Γ.

**`fc3_asr_legs.leg_j_rel`** is *not* a fit-quality indicator on
SiNW. Hiphive's `ClusterSpace` enforces axis-i ASR for FC3 but not
axis-j/k; bulk Si fits naturally give `leg_j_rel = 0.000`, but a
low-symmetry quasi-1-D SiNW fit will report `leg_j_rel ≈ 0.8`
regardless of `fit_method` / `n_structures` / cutoffs. Ignore that
metric for SiNW; check `dispersion.n_imaginary` instead. If a
downstream consumer of FC3 needs axis-j/k ASR strictly enforced,
do the projection explicitly via
`phonon_inputs.fc3_compression.asr_project_factor`.

---

## Workflow A — mc-rattle (standard)

Most predictable, fastest convergence. Run for every new diameter first.

```bash
cd phonon

# 1) Relax the cell (one-off, ~1–4 h on 4 nodes).
python -m phonon_inputs.cli pipeline --config configs/sinw/sinw100_d9a_vasp.yaml --skip-relax  # if relax already done
# OR
python -m phonon_inputs.cli generate --config configs/sinw/sinw100_d9a_vasp.yaml

# 2) Emit 32 mc-rattled supercells with VASP inputs.
python -m phonon_inputs.cli fc3-hiphive-sow --config configs/sinw/sinw100_d9a_vasp.yaml

# 3) Run DFT for each (cluster job; ~6–12 h total at 4 nodes for d9a).
python -m phonon_inputs.cli fc3-hiphive-run --config configs/sinw/sinw100_d9a_vasp.yaml
# (or submit your usual array sbatch over fc3_hiphive_*/disp-*/)

# 4a) Convergence sweep BEFORE the final reap.
#     Refits at n_struct = {12, 18, 24, 32} with both ardr and rfe.
#     Outputs: convergence_summary.json, convergence_vs_n_structures.pdf
python -m phonon_inputs.cli fc3-hiphive-convergence \
    --config configs/sinw/sinw100_d9a_vasp.yaml
#  Inspect convergence_vs_n_structures.pdf — pick the smallest size where
#  rmse_test plateaus AND dispersion_max_thz stops moving. If the curve is
#  still climbing at size=32, bump n_structures / pool_size and re-sow.

# 4b) Final reap at the chosen size (using the config's n_structures).
python -m phonon_inputs.cli fc3-hiphive-reap --config configs/sinw/sinw100_d9a_vasp.yaml
#  Produces fc3.hdf5, hiphive_fit.json. Watch for:
#    rotational_residual_before / after  — projection should drop ≥3 orders.
#    rmse_test / rmse_train              — should be close (no over-fit).

# 5) Validate the FC3 with the finite_analysis pipeline.
python -m finite_analysis run \
    --config configs/sinw/sinw100_d9a_vasp.yaml \
    --analyses all \
    --out-dir reaps/hiphive_sinw100_d9a_vasp
#  Critical checks in physical/physical.json:
#    dispersion.n_imaginary == 0
#    dispersion.max_acoustic_at_gamma_thz < 0.1
#  (fc3_asr_legs.leg_j_rel ≈ 0.8 on SiNW is a hiphive cluster-space
#   limitation — not a fit defect; see HIPHIVE_FITTING_NOTES.md §2.)
```

## Workflow B — phonon-rattle (finite-T refinement of an mc-rattle FC2)

**Phonon-rattle is a refinement on top of a converged mc-rattle FC2**,
not a from-scratch sampler. The literature (Eriksson, Fransson, Erhart,
*Adv. Theory Simul.* **2**, 1800184; Carrete *et al.* 2017) uses
phonon-rattle only after a converged harmonic model is in hand,
because phonon-rattle's per-mode displacement amplitude is
∝ 1/√|ω| — a handful of small-|ω| imaginary modes in the seed FC2
collapses to "Duplicates in permutation" or NaN POSCARs.

**Two ways to provide the seed FC2:**

1. **Reuse the mc-rattle reap (recommended).** After Workflow A has
   produced `<work_dir>/fc3.hdf5`, point
   `hiphive.phonon_rattle_seed_fc3` at it. The phonon-rattle config
   skips the bootstrap stage entirely:
   ```yaml
   hiphive:
     rattle_method: phonon
     phonon_rattle_seed_fc3: ../fc3_hiphive_sinw100_d5a_sc4_vasp/fc3.hdf5
     # (path relative to the config file's directory)
     ...
   ```

2. **Bootstrap** (legacy path, often fails on SiNW). With no seed
   path set, `fc3-hiphive-sow` emits a small mc-rattle bootstrap pool
   into `<work_dir>/bootstrap/`. `fc3-hiphive-bootstrap-reap` fits it
   with ARDR + rotational projection. The fit is checked for PSD
   compliance; if the seed has more than
   `phonon_rattle_max_seed_imag` imaginary modes (default 6) the
   bootstrap-reap refuses to write `fc2_seed.npy` and tells you to
   switch to path 1.

```bash
cd phonon

# 1) Stage-1 sow: writes a small mc-rattle BOOTSTRAP pool into
#    ./fc3_hiphive_sinw100_d9a_vasp_phonon/bootstrap/.
python -m phonon_inputs.cli fc3-hiphive-sow --config configs/sinw/sinw100_d9a_vasp_phonon.yaml

# 2) DFT for the bootstrap pool only.
python -m phonon_inputs.cli fc3-hiphive-run --config configs/sinw/sinw100_d9a_vasp_phonon.yaml
#  Or: run VASP on every fc3_hiphive_sinw100_d9a_vasp_phonon/bootstrap/disp-*/

# 3) Fit FC2 on the bootstrap forces and emit fc2_seed.npy.
python -m phonon_inputs.cli fc3-hiphive-bootstrap-reap \
    --config configs/sinw/sinw100_d9a_vasp_phonon.yaml

# 4) Stage-2 sow: detects fc2_seed.npy and emits the main 32-structure
#    PHONON-RATTLED pool at phonon_rattle_temperature_k.
python -m phonon_inputs.cli fc3-hiphive-sow --config configs/sinw/sinw100_d9a_vasp_phonon.yaml

# 5) DFT for the main pool.
python -m phonon_inputs.cli fc3-hiphive-run --config configs/sinw/sinw100_d9a_vasp_phonon.yaml

# 6) Convergence sweep (same harness as Workflow A).
python -m phonon_inputs.cli fc3-hiphive-convergence \
    --config configs/sinw/sinw100_d9a_vasp_phonon.yaml

# 7) Final reap + finite_analysis (same as Workflow A steps 4b/5).
python -m phonon_inputs.cli fc3-hiphive-reap --config configs/sinw/sinw100_d9a_vasp_phonon.yaml
python -m finite_analysis run --config configs/sinw/sinw100_d9a_vasp_phonon.yaml --analyses all \
    --out-dir reaps/hiphive_sinw100_d9a_vasp_phonon
```

To re-seed the phonon-rattle pool from scratch (e.g. after fixing the
FC2 fit) delete `fc2_seed.npy` and `bootstrap/`; the next sow falls back
to Stage 1.

---

## What to look at if imaginary modes show up

1. **`hiphive_fit.json["rotational_residual_after"]`.**
   Should be ≥ 3 orders below `before`. If they're equal you likely
   forgot `rotational_sum_rule: post_fit` (the d5a/d9a/d12a configs in
   this directory have it on by default; older copies may not).

2. **`physical/physical.json["fc3_asr_legs"]["leg_j_rel"]`.**
   *Not* a SiNW fit-quality indicator. Hiphive's ClusterSpace enforces
   axis-i FC3 ASR but not axis-j/k; SiNW fits consistently give
   `leg_j_rel ≈ 0.8` and bulk Si fits give `≈ 0.0`. Ignore this metric
   for SiNW unless your downstream consumer (e.g. the SSE bubble) needs
   strict axis-j/k ASR — in which case project explicitly via
   `phonon_inputs.fc3_compression.asr_project_factor`.

3. **`physical/physical.json["dispersion"]["n_imaginary"]`.**
   Counts q-modes with ω² < 0. Should be 0 on the supercell q-mesh.
   If non-zero **only at q = Γ**, the structure isn't fully relaxed —
   check `relax_sinw100_*/CONTCAR` against the original POSCAR.
   If imaginary at non-Γ q-points, the FC2 fit is incomplete —
   either bump `n_structures` or extend `cutoffs[0]`.

4. **`physical/physical.json["fc2_psd"]["min_eigval"]`.**
   Negative dynamical-matrix eigenvalues at the supercell level. A
   value of −1e−15 is rounding; below −1e−6 is a real instability.

5. **Convergence-plot saturation.**
   `convergence_vs_n_structures.pdf` — `rmse_test` should plateau and
   `dispersion_max_thz` should stop drifting by the chosen `n_structures`.
   If still moving, increase `pool_size` and re-sow.

## Tunable knobs (per system)

* `n_structures`, `pool_size` — for the d12a wire, 32 is borderline; if
  `rmse_test` is still falling at size=32 set `pool_size: 48` (or higher)
  and re-sow. The convergence sweep takes seconds per fit so over-sampling
  is cheap; under-sampling is what causes the imaginary-mode trouble.
* `cutoffs[0]` (FC2) — extend to 6.0 Å for the d12a wire if the LA-branch
  slope at Γ is off from elastic theory.
* `phonon_rattle_temperature_k` — set to the experimental T of interest
  for the downstream calculation (300 K is the conventional default).
  Reaps at multiple temperatures share the same bootstrap seed FC2; just
  duplicate the `*_phonon.yaml` config with a different `work_dir` and
  `phonon_rattle_temperature_k`.
* `phonon_rattle_qm` — `true` (default) uses Bose–Einstein occupation;
  set to `false` for purely classical sampling (e.g. comparison with MD).
* `phonon_rattle_imag_freq_factor` — **must be > 0**. Hiphive maps any
  imaginary mode `ω² < 0` to `imag_freq_factor × |ω²|` before sampling.
  `1.0` (the hiphive default) flips imaginary modes to a positive
  same-magnitude frequency. **`0.0` is unsafe**: it zeroes the
  frequency, then hiphive divides by it (`1/ω` in the amplitude) and
  the resulting displacements are NaN — VASP fails with garbage
  POSCARs. The codebase rejects `0.0` at sow time. If your seed FC2
  has many imaginary modes (`bootstrap_reap` warns), increase
  `phonon_rattle_bootstrap_n` or switch to `rattle_method: mc`.
