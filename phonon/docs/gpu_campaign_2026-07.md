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

## Budget

~26 debug jobs incl. the 2-node L16 pair: **1.19 node-hours** total on
lp16 (sacct, month-to-date).
