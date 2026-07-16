# jprobe_snaps — Σ snapshots and probe results

`*.npz` files are large (45–330 MB) and gitignored; they are mirrored
here on the laptop and originate on the cluster
(`/usr/scratch/mont-fort11/pfischill/quatrex/phonon/studies/out/anderson_test/jprobe_snaps/`).
Each snapshot stores the solver's flat-nnz `sigma_lesser/greater/retarded`
(decode through `SCBA(cfg).data.<sigma>.blocks`, matching `sse_g_band`).

| file | what |
|---|---|
| L2_fp.npz / L2_fp_run.npz | L2 converged fixed point (masked kernel) |
| L2_andstall.npz | L2 Anderson stall point |
| L4_stall.npz (+ _run) | L4 iteration-350 sawtooth stall (masked kernel) |
| L4_trough.npz / L4_peak.npz | +9/+17 iterations within one sawtooth cycle |
| firstborn_L{2,3,4}.npz | first Born from ballistic G (masked kernel) |
| fb_L4_w100 / fb_L4_eta1e4 / fb_L4_ne180.npz | first-Born toggle probes |
| fb_L4_newcode.npz | legacy-path A/B gate (bit-identical to firstborn_L4) |
| fb_L4_gband2.npz | first Born with the exact kernel (PSD on all slabs) |
| L10_g2_it85.npz | L10 exact-kernel state just before the pinned phase |
| decoded_sigmaR_blocks*.npz | slab-diagonal Σ^R blocks (library-decoded) |
| jp_L2_fp / jp_L2_stall / jp_L4_stall | power-iteration results (result.json) |

Probe eigenvalues: L2 fp −4.33/−3.94; L2 stall −3.51/−3.30;
L4 stall −5.07/−4.75/−4.15/−4.15 (all IR bins 1–5).
