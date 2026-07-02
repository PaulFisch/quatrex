# Tensor-decomposed coupled-q SSE — implementation + wave-0 findings (2026-07-02)

Implements the factored three-phonon SSE of `document/src/theory/50_sse_decomposed.tex`
in the production solver, plus the offline factor pipeline and the measurement
campaign prerequisites. Status: **kernel exact and merged; bulk Si highly
compressible (GO); the skinny-Gram kernel is NOT the film's speed lever — the
production win is the `reconstruct` mode (factored storage, dense compute).**

## What exists now

- **Fitters** (`phonon/phonon_inputs/fc3_compression.py`, TD-A): `enforce_asr=True`
  defaults everywhere, `fit_production()` (INDSCAL, 16 seeded restarts, hard
  ASR gate at 1e-10·norm), `export_production_factors()`, deterministic seeds,
  18 unit tests (`tests/phonon_inputs/test_fc3_compression.py`).
- **Offline pipeline**: `phonon/phonon_inputs/fc3_factor_device.py` (fit cached per
  (ansatz, R, fc3-hash); exact per-leg device gather mirroring `se_q.py` phases)
  → `src/quatrex/phonon/vertex_factors.py` (npz schema, λ-sorted truncation).
  Builder: `build_inputs.py --decompose-ranks R1,R2 [--decompose-ansatz INDSCAL]
  [--decompose-only]`, with a phase-convention self-check of factor-reconstructed
  Φ̃ blocks against the dense qfold entries (gate: ≤10× fit rel_err).
- **Solver** (`sse_phonon_phonon.py`): `decomposed_vertices_path` +
  `sse_vertex_rank` (config-only R-sweep) + `decomposed_kernel = "reconstruct" |
  "gram"`; mutual exclusion with `qfold_path`; ballistic λ-zeroing in `run.py`;
  `write_config.py --decomposed-vertices/--vertex-rank`.
- **Kernels**: `bubble_factored.py` (skinny Grams + entrywise q-convolution +
  sandwich; INDSCAL shared-leg cache, CP role-keyed; fold terms merged 6→4);
  `_reconstructed_qvertices` (rank-local dense dict from factors, offset table
  shared across I).

## Correctness (all green)

- The two STALE dense oracle tests were repaired — they predated three verified
  physics changes: the left-vertex conjugation (d308a8b5), the exact
  ji-transpose + q-negation bosonic fold (447725cd), and the σ^≷ sign
  convention (f7613b5e); production also DC-zeroes the ω=0 bin of Σ^≷ (and of
  the hard-masked Hilbert Σ^R). The coupled-q fixture now runs nq=3 so the
  fold's q-negation is actually exercised.
- `test_compute_coupled_q_factored_matches_dense`: factored (both kernels ×
  INDSCAL/CP) == dense fed the identical vertex, rtol 1e-9 (measured ~5e-15),
  non-TRS G.
- `tests/quatrex/phonon/test_vertex_factors.py`: npz round-trip, truncation,
  format gate, config mutual-exclusion.
- End-to-end sifilm smoke (nk=3, ns=3, R=16, rel_err 9%): SCBA runs, heat
  currents dense-vs-factored agree to ~0.1–0.3% (the compression error),
  bubble energy balance identical to dense (~4e-4, a fixture property).
- 3-rank q-distributed (q_comm=3) factored run matches serial.

## Bulk-Si compressibility (TD-B): GO

The 5³ hiPhive film tensor (`si_big_hiphive`, (6, 2250, 2250) mass-weighted,
solver units) has an mSVD spectrum reaching **10% Frobenius at rank 10, 2% at
rank 20, 1% at rank 21** (`phonon/scripts/out/bulk_si_compressibility/`) — far
inside the R*≤64 gate, and ~20× more compressible than the d11a device tensor
(0.49 at rank 16). INDSCAL tracks mSVD within ~2× rank on the small tensor
(2×2×2: mSVD 5.5%@16, INDSCAL 9.1%@16, 4.0%@32). Full INDSCAL/CP sweep of the
big tensor in `rank_sweep.csv` (background job at the time of writing).

## Micro-benchmark verdict (`phonon/studies/_bench_factored_sse.py`)

Real ns3_nk9 per-rank shapes (nd=6, nq=81, q_own=3, n_tau=60), single thread,
parity-checked (≤5e-15):

| R  | dense  | gram kernel | speedup |
|----|--------|-------------|---------|
| 8  | 25.8 s | 3.4 s       | 7.5×    |
| 16 | 25.7 s | 14.9 s      | 1.7×    |
| 32 | 25.8 s | 140 s       | 0.2×    |

The Gram q-convolution is memory-bound at O(R²) per (quad, q, τ) while the
dense ring's 6×6-block GEMMs are cheap: flops break even near R² ≈ 3·b⁴
(R ≈ 60 at b=6) but the entrywise einsum runs ~10× below GEMM throughput, so
the crossover in TIME is R ≈ 16–20. **The plan's ≥10×-at-R=64 gate fails on
film shapes — by design of the film, not a bug.** The gram kernel wins only at
small R or LARGE blocks (wires: b=48+ → 3·b⁴/R² ≈ 4000 at R=64).

## Production consequence: `decomposed_kernel = "reconstruct"` (default)

The factored representation's real production value on films is **memory +
build time**, not flops: reconstruct the rank-local slice of the dense vertex
dict from the factors once at first compute (per-(d1,d2) offset table shared
across I) and run the untouched dense path at full speed. ns8_nk9: ~45 MB/rank
vs the 1.2 GB replicated qfold dict — lifts the ns8 rank cap (54→108+) and
removes the O(nk²)-pair dense fold from the builder (`--decompose-only`).
The R-sweep (G vs R) stays a config-only knob (`sse_vertex_rank`) at dense
speed.

## Wave-1 plan (unchanged otherwise)

1. Dense ns3_nk9 baseline refresh (mandatory — Jun-11 numbers pre-date the
   sign fix) once the node clears (cnt33 cutoff sweep running).
2. Factored R-sweep ns3_nk9 with `decomposed_kernel="reconstruct"`,
   R ∈ {8, 16, 32, 64} truncated from one R=128 INDSCAL file; pick R* by
   |ΔG| < 1% vs dense.
3. ns5/ns8 (wave 2) use `reconstruct` + `--decompose-only` for the memory win.
