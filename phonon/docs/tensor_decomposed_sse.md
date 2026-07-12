# Tensor-decomposed coupled-q SSE

Implements the factored three-phonon SSE of `document/src/theory/50_sse_decomposed.tex`
in the production solver, plus the offline factor pipeline.

**Status: the factored kernel is the production kernel** (`decomposed_kernel = "gram"`,
now the default). It is exact -- it reproduces the dense ring fed the identical
vertex to ~5e-15 -- and on the real `sifilm` ns3_nk9 shapes it is **~10^2--10^3x
faster than the dense ring**.

> This supersedes the 2026-07-02 wave-0 verdict ("the skinny-Gram kernel is NOT the
> film's speed lever; the production win is the `reconstruct` mode"). That verdict
> was correct *about the kernel as it then stood* and wrong about the method: the
> kernel was missing two exact algebraic collapses, and without them a low-rank
> method is genuinely slower than the dense contraction it replaces.

## The two collapses

Both are exact identities, not approximations. Both were absent from the kernel
*and* from the theory chapter, and the chapter's cost model
(`eq:cost_phph_dist_cp`) hid the second by multiplying every term by `N_q^2`.

**1. The quad sum factorises.** Per output pair `(I,J)` the ring sums over quads
`(K1,K2,K1p,K2p)`. The a-line Gram depends only on `(K1,K1p)` and the b-line only
on `(K2,K2p)`; the factored vertex has full offset support and the G band
constrains the two lines separately, so the quad set is exactly the Cartesian
product of the two link sets. The Hadamard is bilinear, hence

    sum_quads Pa[a] o Pb[b] = ( sum_a Pa[a] ) o ( sum_b Pb[b] )

-- sum the Grams first and take **one** Hadamard per pair instead of one per quad
(181 quads on ns3).

**2. The q'-sum is a circular convolution.** `Pa` depends only on `q'` and `Pb`
only on `q_ext - q'`, so `sum_{q'} Pa[q'] o Pb[q_ext - q']` is a circular
convolution on the transverse torus (`build_q_diff_map` is literally
`(i-j) mod n`) and runs as an FFT: `O(N_q^2) -> O(N_q log N_q)`. Reuses
`qttools.fft.fft_circular_convolve`.

This one is available **only** in the factored form. The dense vertex
`Phi~(q', q_ext-q')` does not separate in the momenta, so the dense ring is stuck
at `O(N_q^2)` at any rank. The decomposition is what buys the FFT.

Cost goes from `O(n_quads * N_q^2 * b^4)` to `O(N_q * (R b^2 + R^2 b))`, the Gram
becoming the floor -- which is the correct asymptotic bound, since projecting `g`
onto a rank-R basis cannot cost less than `R * nnz(g)`.

## Measured (sifilm ns3_nk9: b=6, N_q=81, n_tau=60, single thread, full q)

All three paths fed the same factors; `new` is parity-checked against `dense`.

| R | dense | legacy gram | **new** | new/dense | new/legacy | parity |
|----|---------|---------|---------|--------|-------|--------|
| 8  | 852.9 s | 85.5 s  | 0.88 s  | **967x** | 91x | 4.5e-15 |
| 16 | 853.2 s | --      | 2.67 s  | **319x** | --  | 5.7e-15 |
| **32** | 854.1 s | --  | 9.36 s  | **91x**  | --  | 5.8e-15 |
| 64 | 853.3 s | --      | 38.7 s  | **22x**  | --  | 5.6e-15 |
| 128| 852.5 s | --      | 200.3 s | 4.3x     | --  | 5.4e-15 |

Bulk Si FC3 reaches 1% Frobenius at R=21, so **R~32 is the operating point**, where
the new kernel is **91x** the dense ring and the legacy kernel was **0.2x** (i.e.
slower than dense) -- a ~450x swing at the rank the fit actually needs. The
`>=10x at R=64` campaign gate, which the legacy kernel failed, is passed at 22x.

The dense time is rank-independent, as it must be: the vertex is reconstructed at
rank R offline, but the ring it feeds is a b x b x b contraction either way. That
is the whole point -- the dense path cannot spend a low rank.

The legacy kernel's failure is now explained rather than accepted: it paid `R^2`
once per `(quad, q_ext, q')`, i.e. `181 * 81 * 81` times per pair, through an
`einsum` gather running ~30x below GEMM throughput. Both collapses attack exactly
that term.

## Gamma-only devices (nanowires, CNTs) -- previously unreachable

The factored kernel now also serves `nq == 1`, where the convolution is the
identity and the Gram collapse `b^4 -> R b^2 + R^2 b` stands alone. This is the
regime of every transversely-finite system (cnt33/cnt80, the Si nanowires,
SrTiO3), which the factored path could not reach **at all** before -- and where
the block size is largest, so the `b^4` saving is biggest.

Measured at the d5a Si-nanowire block size (`b=63`, nslabs=3, n_tau=60, 1 thread):

| R | dense | new | speedup | parity |
|----|---------|--------|-----------|---------|
| 16 | 780.4 s | 0.22 s | **3553x** | 8.8e-16 |
| 32 | 774.7 s | 0.42 s | **1842x** | 1.3e-15 |

The wire is where the decomposition pays most, and it was the one place the
factored kernel was never wired in.

## What exists

- **Fitters** (`phonon/phonon_inputs/fc3_compression.py`): INDSCAL (production;
  algebraic init + ALS + L-BFGS, 16 restarts, hard ASR gate) and a tensorly CP
  fallback. `enforce_asr=True` by default.
- **Offline pipeline**: `phonon/phonon_inputs/fc3_factor_device.py` (fit cached per
  (ansatz, R, fc3-hash); per-leg device gather mirroring `se_q.py` phases) ->
  `src/quatrex/phonon/vertex_factors.py`. Builder:
  `build_inputs.py --decompose-ranks R1,R2 [--decompose-only]`, with a
  phase-convention self-check against the dense qfold entries.
- **Solver**: `decomposed_vertices_path` + `sse_vertex_rank` (config-only R-sweep,
  lambda-sorted truncation from one high-rank file) + `decomposed_kernel`.
- **Kernel**: `src/quatrex/phonon/bubble_factored.py`.
- **Bench**: `phonon/studies/_bench_factored_sse.py` (3-way, `--verify`).

## Correctness

- `test_compute_coupled_q_factored_matches_dense`: factored == dense fed the
  identical vertex, INDSCAL x CP x {gram, reconstruct}, non-TRS G, rtol 1e-9
  (measured ~5e-15).
- `test_compute_gamma_factored_matches_dense`: the same at the zone centre.
- `test_q_convolution_matches_explicit_q_diff_map_sum`: the FFT == the explicit
  `q_diff_map` double sum.
- The quad-product structure is a property of the enumeration, verified directly
  against `_phi_pair_index` for several geometries and offset ranges.

## Found while running this: the transverse-mesh validator rejected every film

The Gamma-centred-mesh check in `SigmaPhononPhonon.__init__` compared
`mesh % 1.0` against `want % 1.0`. The mesh lives on a torus, so a momentum that
lands a hair *below* zero wraps to `~0.99999999996` and reads as maximally
distant from `0.0`. That happens whenever the stored `kpoint_shift` is rounded
rather than bit-exact -- and `build_inputs.py` writes `kshift.npy` rounded
(`0.4444444444`, not `0.4444444444444444`). So the validator rejected the very
geometries the builder produces:

    ValueError: Coupled-q vertices assume the Gamma-centered mesh q = k/n ...
                kpoint_shift=(0.0, 0.4444444444, 0.4444444444) does not produce it.

Fixed by comparing the signed circular distance
`(mesh - want + 0.5) % 1.0 - 0.5`. A genuinely wrong shift (e.g. 0.3) is still
rejected. This was only visible by actually launching a film run; no test covered
it.
