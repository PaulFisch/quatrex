# GPU campaign on Alps/daint GH200 (2026-07-31)

Scaling, larger structures, peak-gap attribution, and mixed precision
for the phonon NEGF-SCBA engine on CuPy. All runs through
`phonon/scripts/daint.py` (lp16, debug partition; 2-node runs
explicitly approved); configs eta = 0 throughout. Correctness context:
CPU/GPU/GH200 engine parity ≤ 8e-13 (commit 570dfbda), rank-count
invariance at 2e-15.

## 1. Where the time goes (CNT L4, b=36, ne=41, 1 GH200)

0.79 s/iteration:

| component | s | note |
|---|---|---|
| SSE ring contraction | 0.41 | 11.4 TF/s FP64 = ~100% of its shape roofline |
| OBC (Sancho-Rubio) | 0.33 | CPU-orchestration-bound: batched-LU GPU kernels are only ~4 ms; the Python iteration syncs on a host norm check every step |
| FFT/fold/transposes | 0.02 | |
| RGF selected solve | 0.017 | |
| assemble/mix/conv | 0.007 | |
| residual (python) | ~0.00 | |

The OBC is **iteration-invariant** here (fixed leads, eta constant, no
scattering contacts) yet recomputed every iteration; at ne=241 it is
1.8 s of a 3.9 s iteration, and 12.9 s/iter for d11a. **An opt-in OBC
cache is the single biggest engine win** (up to ~2x end-to-end at
production grids), ahead of any GEMM work. (The first-iteration OBC is
3-8x its steady cost — warm-start effects; a cache also removes that.)

## 2. Why not 67 TF/s? (peak-gap verdict chain)

67 TF/s (FP64 tensor-core theoretical)
→ **43-54 TF/s** sustained big-square ZGEMM/DGEMM (cuBLAS + power/clock
  reality; measured n=1024..8192)
→ **~13.5 TF/s** b=36 batched ring-shape ceiling (measured; saturated
  in w by 481, flat to 8192)
→ **11.4-12.5 TF/s** achieved in-engine (~94% GPU-busy in stage 3).

Evidence: nsys shows all ring GEMMs on FP64 **tensor-core kernels**
(`sm90_xmma_gemm_cf64…tensor16x8x16`, CUTLASS `tensorop_z884gemm`,
93% of GPU time) with 32/64-wide tiles against b=36 operands — the
loss is **tile quantization** (~40-45% padding waste per GEMM, squared
across two small dims), not launch overhead (kernel count flat:
GF/s identical 11.32/11.32/11.34/11.35 across L4/L8/L16/L32 while
quads grow 27x), not bandwidth (AI = 0.375 b flop/byte ⇒ b=36 needs
only ~0.85 TB/s of 4 TB/s), not complex arithmetic (batched Z is
FASTER per flop than batched D: 12.5 vs 8.4 TF/s at b=36), and not a
missing TC path. Larger blocks recover efficiency: b=63 27-28.5 TF/s
(42% of peak), b=135 22-27 TF/s. The 4M real-split emulation is 2x
SLOWER — dead end. **Conclusion: the ring runs at the library ceiling
for its block sizes; the remaining gap is a cuBLAS small-tile fact.**

## 3. Scaling (CNT L4, ne=241; steady s/iter; lead current invariant
to 1e-11 across every configuration)

| config | s/iter | E(p) | notes |
|---|---|---|---|
| 1 GPU | 3.87 | — | OBC 1.79, ring 1.92 (14.6 TF/s) |
| 2 GPU stack | 2.11 | 0.92 | |
| 4 GPU stack (1 node) | 1.10 | 0.88 | ring 12.1 TF/s (smaller tau batch) |
| 4 GPU stack, NCCL | 1.08 | 0.90 | == device_mpi; transposes ~2 ms either way |
| 8 GPU stack (2 nodes) | 0.62 | 0.78 | cross-node transposes 4-5 ms |
| 2 GPU block (ne=41) | 0.46 vs 0.79 | 0.85 | halo+fold on device; OBC halves (one lead per rank) |
| 4 GPU q-axis (film L3 nk5) | ran | — | first GPU run of the qfold path |

Communication is negligible at these sizes (per-rank alltoall slice
~12 MB, sub-ms on NVLink, few ms on Slingshot; aws-ofi-nccl plugin
present and loaded). The efficiency decay is almost entirely the known
stage-3 GF/s drop with shrinking per-rank tau batch (w=481→60 is an
intrinsic ~23% ring loss at p=8) plus the constant per-rank OBC —
roofline-corrected E*(8) ≈ 0.95. **Production recommendation: 4 GPUs/
node over stack (or stack x q for films) is near-free; block axis only
for memory.**

## 4. Larger structures (1 GH200, campaign grid ne=41; d11a ne=81)

| case | s/iter | ring TF/s | mempool | vs EPYC recorded |
|---|---|---|---|---|
| CNT L4 | 0.79 | 11.3 | 1.0 GB | |
| CNT L8 | 1.47 | 11.3 | 2.8 GB | seconds-class/iter on 64c |
| CNT L16 | 2.81 | 11.3 | 6.2 GB | |
| CNT L32 | 5.49 | 11.3 | 13.2 GB | |
| d11a L3 (b=135, ne=81) | 56.8 | 26.7 | 33.2 GB | 2.2-40 min/iter |
| d11a L3 (full ne=101) | 71.6 | 26.4 | 39.5 GB | " |

Perfect linear-in-quads scaling; even d11a's full default grid fits a
single 96 GB GPU, so the tau-chunk memory floor (E4) stays unneeded
for every staged case (it becomes relevant only at ne≳241 for b=135).
d11a inputs staged from the legacy ASR-generation export — timing
valid, physics numbers not to be quoted. One GH200 ≈ 10-40x a 64-core
EPYC node on these cases (case-dependent).

## 5. Mixed precision (`phonon.sse_ring_dtype = "complex64"`, opt-in)

Implementation: per-quad ring GEMMs as batched CGEMM (vertex caches
prescaled by an exact power of two per side — raw FC3 ~1e20 overflows
float32 in the Bose-enhanced low-omega bins; output buffers stay
complex128 — a c64 out buffer clips the largest Sigma^>(tau) to inf);
cross-quad accumulation, FFTs, fold, Dyson, OBC all complex128.
Gamma-only dense path; default complex128 bit-identical (suite green).

Speed (GH200): ring-shape CGEMM 1.24x / 1.41x / 1.35x at b=36/63/135;
in-engine CNT L4: ring 0.41→0.35 s, iteration 0.79→0.73 s.
d11a full grid (ne=101): ring 55.1→41.8 s (1.32x), iteration
71.6→58.7 s (1.22x), mempool 39.5→24.3 GB (1.6x less memory — the c64
band-link copies; useful in its own right for big cases).

Accuracy (CNT L4, eta=0, 3 SCBA iterations, vs fp64):
heat current 1.6e-6 relative, Sigma 3e-7, **bubble-balance
conservation drift < 1e-6** (vs ~1e-14 fp64), G diagonals 1-4e-6,
slab absorption 1.5e-4 (difference of large numbers).

**Verdict: usable for scans/exploration where 1e-6 heat accuracy and
1e-6-class conservation residuals are acceptable; NOT for the
conservation-grade eta=0 production numbers (the balance gate is the
project's honesty anchor at ~1e-10). The gain is modest (~10-30%
end-to-end) because the c64 advantage at small tiles is bandwidth,
not compute — both precisions ride the same tensor cores.** TF32
(CUPY_TF32=1) untested-by-default; expected accuracy ~1e-3 — not
worth it at these gains.

## 6. Ranked recommendations

1. **OBC cache across SCBA iterations** (opt-in `obc_cache`, valid
   when leads/eta/scattering-contacts are iteration-invariant; must
   respect the eta-ramp paths): up to ~2x end-to-end at production
   grids; also fixes the OBC's GPU-unfriendly sync-per-step pattern.
2. **Default GPU comm profile**: device_mpi everywhere on daint (NCCL
   equal on-node, needs nothing extra); stack-first rank layout, block
   axis only for memory; films stack x q.
3. `sse_ring_dtype = "complex64"` documented as a scan-mode option.
4. Larger-block devices (b≥63) are the natural GPU targets (2-2.4x
   the ring efficiency of b=36).
5. E4 tau-chunking: deferred until a b=135 case at ne≳241 is real.

## 7. Full L16 pair on 2x4 GH200 (ne=161, converged, eta=0)

g_band=3 (boxcar) vs g_band=1 + Bartlett taper, otherwise identical
(linear 0.1 mixing, TOML defaults). Fixed a g_band>1 GPU bug first
(the band-pattern extension fed numpy index arrays to cupyx
coo_matrix; every earlier GPU run was band-1).

| | g3 | g1 + bartlett |
|---|---|---|
| converged after | 98 it (~5 min) | 67 it (~2.3 min) |
| s/iter (8 GPUs) | 3.07 | 2.04 |
| lead current (internal units) | **38.36** | **17.71** |
| lead-to-lead balance | 9.5e-5 | 3.4e-3 |
| internal interface spread | 6.3% | 47.5% |
| mempool peak | 5.6 GB/GPU | smaller |

The band-1 taper halves every |d|=1 G link and off-diagonal Sigma
block, and at L16 that costs a factor ~2.17 in transmitted heat and a
47% interior heat-profile dip (structurally incomplete interior
Sigma) — the g3 boxcar keeps the interior flat to 6% with 36x tighter
lead balance. Both fixed points are clean (no eta, no instability);
the g1t damping even converges faster. Memory was never a constraint
(5.6 GB/GPU of 96) — no batching changes needed. NOTE: not directly
comparable to the tortin campaign L16 numbers (different
taper band/settings and 600-cap non-converged trajectories there).

### Full length ladder (all converged, ne=161, 2x4 GH200)

| L | variant | iters | lead current | lead balance | interior spread |
|---|---|---|---|---|---|
| 16 | g3 | 98 | 38.36 | 9.5e-5 | 6.3% |
| 16 | g1t | 67 | 17.71 | 3.4e-3 | 47% |
| 24 | g3 | 97 | 44.10 | 3.0e-3 | 11% |
| 24 | g1t | 71 | 16.32 | 3.7e-3 | 72% |
| 32 | g3 | 177 | 56.22 | 9.9e-3 | 18% |
| 32 | g1t | 71 | 16.04 | 3.7e-3 | 86% |

Physics reading: the **g3 boxcar lead current GROWS with length**
(38 → 44 → 56) — anomalous gain for a wire at fixed dT, consistent
with the documented non-PSD band-mask pathology (the boxcar Schur
mask injects non-causal gain in interior Sigma blocks; the effect
compounds with device length, the lead balance degrades in step
9.5e-5 → 1e-2, and convergence slows to 177 iterations). The
**g1t series saturates physically** (17.7 → 16.3 → 16.0, stable
balance ~3.5e-3) but underweights coherence ~2x and its interior
profile dips to 86% — structurally incomplete interior Sigma, honest
leads. Neither truncation is converged in the band parameter at these
lengths: the natural next rung is **g_band=2 + Bartlett** (PSD taper
at a band where the diagonal AND first-off-diagonal Sigma blocks are
exact) — the campaign machinery runs it as-is (~15 debug-min per
point). All eight runs eta=0, no instabilities.

## 8. cuTile fused-ring experiment: correct, but far below cuBLAS
(2026-07-31)

Goal: reimplement the ring as one fused cuda.tile kernel (tile sizes
matched to b instead of cuBLAS's 32/64 menu — padded-flop ceilings
~28 TF/s at b=36 vs cuBLAS's 13.5, ~64 at b=63; T/U kept on-chip;
complex128 as in-kernel 4M split since Tile IR has no complex dtypes).
Kernel + harness: `phonon/studies/_cutile_ring.py` (smoke/check/bench).

**Environment trap (cost half the investigation; now handled in the
study file):** tileiras resolves its nvvm/ptxas backend via PATH /
CUDA_HOME / CUDA_PATH / **CUDA_ROOT**. cuda.tile sanitizes the first
three in the compiler subprocess but NOT CUDA_ROOT; a system CUDA 13.1
behind CUDA_ROOT poisons every mma/lineinfo compile with the generic
"failed to compile Tile IR program" (same root cause as
NVIDIA/cutile-python#72, which was a bad CUDA_HOME). Scrubbing
CUDA_ROOT and prepending the wheel's cu13 toolkit fixes everything.

Results (tileiras 13.3.36, cuda-tile 1.5.0):
- **Correctness: perfect.** smoke + full check PASS on sm_86 (A1000)
  and sm_90 (GH200): the fused 4M ring matches the production cuBLAS
  ring at 3-8e-15 across b ∈ {36,63,135} and every tile config.
- **Performance on GH200: 0.8-1.0 TF/s effective — 0.03-0.08x
  cuBLAS** (9.8-28 TF/s on the same shapes/harness). The smallest
  tile (T8/E2) wins and throughput is flat in shape and batch — the
  signature of the one-release-old sm_90 backend not lowering f64 mma
  to DMMA (plus immature codegen for the nested fused loops). The
  arXiv CUDA-Tile evaluation saw a >4x maturity spread across
  backends for tuned AI kernels; FP64 fused contractions are clearly
  far behind that.

Pre-registered verdict: **<1.1x = negative result, park.** The
tile-quantization headroom (28 vs 13.5 TF/s at b=36) remains real in
principle, but today's compiler cannot approach it. Re-evaluate on
future tileiras releases: `_cutile_ring.py smoke && check --quick`
(laptop) then the bench job — the harness is ready and the assessment
criteria (≥1.3x at b=36, ≥0.9x at 63/135) stand. This is, to our
knowledge, the first FP64/complex cuTile-on-Hopper datapoint.

## Budget

~26 debug jobs incl. the 2-node L16 pair: **1.19 node-hours** total on
lp16 (sacct, month-to-date).

## 9. Film / coupled-q SSE (2026-08-02)

Question: is the dense-q film ring near the previously quoted 42% of
peak (that figure was b=63, Gamma-only)? Answer: no — and after
batching it sits at its own shape ceiling.

**b=18/54 batched-ring shape ceilings (GH200, cuBLAS Z, full ring):**

| b  | w=60 | w=241 | w=481 | note |
|----|------|-------|-------|------|
| 15 | 0.35 | 1.36  | 2.69  | TF/s |
| 18 | 0.71 | 2.59  | 5.48  | film block size |
| 36 | 9.8  | 12.3  | 13.7  | CNT (campaign) |
| 54 | 22.8 | 24.7  | 25.4  | film 3-slab dense |
| 63 | 27.5 | 28.3  | 28.2  | the "42%" row |

**In-engine film ring (mos2f3, nq=25, 7 pairs, 4375 qtasks, ne=121,
1 GH200):**

| kernel | ring s/it | TF/s | % of 67 peak |
|---|---|---|---|
| per-task loop (legacy) | 6.47 | 2.46 | 3.7% |
| batched (sse_dense_q_batched) | 2.33 | **6.84** | 10.2% |

The legacy loop ran one (q-pair, quad) task per launch — batch=w=241,
which the microbench shows is itself capped at 2.59 TF/s at b=18: the
old rate was the PER-LAUNCH ceiling, not Python overhead alone. The
batched kernel (C x w ~ 1e5-deep strided batches, single-gather legs,
scatter-add over q_ext) saturates the b=18 quantization ceiling.
Correctness: identical heat matrix to the baseline smoke; the nq=3
reference test passes at g_band=1/2/3 on numpy+cupy.

Iteration: 9.53 -> 5.35 s/it (1.8x); OBC (2.9 s) is again the top
non-ring cost (obc_cache recommendation unchanged). The coupled-q
GFLOP model is now computed in-engine (was 0 for all film runs).

Remaining headroom at b=18 is structural (tile quantization ~(32/18)^3):
the factored gram kernel (q-FFT collapse, N_q^2 -> N_q log N_q) is the
next lever — pending the factorisation audit gates (mass-weighted ASR
fixed; post-min-image conservation ladder in progress; note the shipped
sifilm nk9 dense reference itself predated the min-image fix).

## 10. Fused-ring kernel, rounds 1-2 (2026-08-03): the empirical ceiling

Hand-written CUDA C++ fused chain (T/U never touch HBM), 11 variants
+ 4 operation restructures + cuTENSOR/cuTile, all parity 2-5e-15:

| candidate | b=18 @ depth | vs cuBLAS (7.6-7.9) |
|---|---|---|
| cuBLAS 3-GEMM composite | 7.6-7.9 TF/s | 1.00 |
| fused v1 (o1, panel-streamed) | **8.9-9.0** | **1.17-1.19** |
| fused rt22/rt23/rt33 (register tiles) | < v1 | — |
| fused v3 (fixed-k2 mapping) | 4.4-4.8 | 0.56-0.62 |
| cuTENSOR einsum mapping | 7.1-7.2 | 0.90 |
| grouped tall-GEMM restructure | <= 5.7 | <= 0.73 |
| cuTile (sec. 8) | 0.8-1.0 | 0.03-0.08 |

ncu on v1: smem/L1 pipe 79% vs FMA 28%, 162 regs -> 17% occupancy —
the on-chip pipes bind, not HBM (29 GB/s) or arithmetic. Literature
(MAGMA/DBCSR class, verified): published batched-FP64 best at b~16-32
= 90% of the nB/8 complex roofline = 12-31% of raw peak; the fused-
chain roofline (~18 TF/s = 53% of the 34 TF/s vector peak) has no
published FP64 demonstration and is NOT reachable here — every
organisation that cut one stage's traffic paid it back elsewhere.
VERDICT: empirical max ~9 TF/s (26% of peak) at b=18; +15-19% over
cuBLAS = below the 1.5x integration bar -> parked (sources
phonon/studies/_fused_ring.* / _ring_variants_bench.py). Tensor
cores irrelevant at b=18 (bandwidth-bound); b>=36 belongs to cuBLAS.
Realised film levers instead: bosonic fold (1.5x flops, merged) +
OBC cache mode + stack scaling.
