# Positivity of the three-phonon bubble: where it comes from, and what breaks it

2026-08-08. Theory note. The numerical companion is
`phonon/studies/_bubble_positivity.py` (artifacts in
`phonon/studies/out/positivity/`); the measurement that prompted it is
in `mos2_conservation_audit.md` § "Why h_L = -h_R".

## The question

Every failing MoS2 film state carries negative occupation,
`(-i G^<)_ii < 0`, with the worst entry as large in magnitude as the
largest positive one; every control (ballistic MoS2, converged Si film,
converged CNT L4) has exactly zero violations. Since
`G^< = G^R Sigma^<_tot (G^R)^dagger` is a congruence and congruence
preserves positive semi-definiteness, this *proves* that
`-i Sigma^<_tot` has a negative eigenvalue. The question is where a
PSD-preserving chain loses PSD.

## 0. Conventions

The solver stores occupation-positive Green's functions,
`X := -i G^< >= 0` and `Y := -i G^> >= 0`, the same convention as the
lead injection `sigma^{<,>} = i n^{(+1)} Gamma`. The bubble prefactor is
`p_c = 0.5j * hbar * dw / (2 pi)` (`units.py:55-61`), i.e. `i` times a
positive real. With `G^< = iX`,

    Sigma_raw = i p (iX)(iY)-type products = -i p K,   K := ring(X, Y),

so `-i Sigma_raw = -p K <= 0` (textbook sign), and the code negates at
`sse_phonon_phonon.py:1681-1682` to store `-i Sigma = p K >= 0`. The
whole positivity question is therefore the single statement

    K >= 0   whenever   X, Y >= 0.

## 1. The ring is a congruence

The kernel (`src/quatrex/phonon/bubble.py:198`, and the same expression
block-resolved at `sse_phonon_phonon.py:403`) computes

    S[a,J] = Phi_L[a,c,e] A[c,b] B[e,d] Phi_R[J,d,b].

Introduce the pair indices `P = (c,e)` and `Q = (b,d)` and set
`M[a,P] = Phi_L[a,c,e]`, `N[J,Q] = Phi_R[J,d,b]`. Because

    (A (x) B)[(c,e),(b,d)] = A[c,b] B[e,d],

the contraction is exactly `S = M (A (x) B) N^T`. This is a congruence
`M (A (x) B) M^dagger` if and only if

    Phi_R[J,d,b] = conj(Phi_L[J,b,d]).                        (1)

Note the index order in the code: the second vertex is contracted with
its leg pair **transposed** relative to the first, `Phi_R[J,d,b]` and
not `Phi_R[J,b,d]`. That transposition is not cosmetic; with the other
ordering the form would be `M (A (x) B) S M^dagger` with `S` the
leg-swap permutation, which is not a congruence and carries no
positivity statement at all. The code's index placement is precisely
the one the theorem needs.

**Theorem 1.** If (1) holds and `A, B >= 0` (Hermitian PSD), then
`S >= 0`.

*Proof.* `A, B >= 0` implies `A (x) B >= 0` (the Kronecker product of
PSD matrices is PSD: its eigenvalues are the pairwise products of the
factors' eigenvalues). Under (1), `S = M (A (x) B) M^dagger` is a
congruence of a PSD matrix, hence PSD: for any vector `v`,
`v^dagger S v = (M^dagger v)^dagger (A (x) B) (M^dagger v) >= 0`. []

Condition (1) unpacks differently in the two production paths:

- **Gamma-only (nq = 1)**, `phi_perms(pl, pr)` with both factors the
  same real-space dict (`sse_phonon_phonon.py:1283-1285`): (1) requires
  the vertex to be **real** and symmetric under exchange of its two
  contracted legs, `Phi[J,b,d] = Phi[J,d,b]` (including their block
  indices, `Phi[(J,K,K')] = Phi[(J,K',K)]^T`).
- **Coupled-q (nq > 1)**, the film's path: the left factor is
  explicitly conjugated (`sse_phonon_phonon.py:1846-1848`,
  `phi_perms(conj(pl), pr)`) with `pl` from `Phi(q', q2)` and `pr` from
  `Phi(q2, q')`. Then (1) becomes

      Phi(q2,q1)[J,d,b] = Phi(q1,q2)[J,b,d],                  (1')

  i.e. exchanging the two contracted legs *together with their
  momenta*. Reality is **not** required here — the conjugation on the
  left factor is exactly what turns `M (.) N^T` back into
  `M (.) M^dagger`. This is a genuine property of the q-fold: the two
  legs pick up their phases symmetrically (`phonon/solver/se_q.py:41-46`,
  `Mq = (Mb * p1[None,:,None]) * p2[None,None,:]`), so (1') holds for
  any real-space vertex with the leg-exchange symmetry.

Condition (1)/(1') is the S3 subgroup property of the physical fc3. It
is inherited from hiphive's `symmetrize=True`, but it is **never
re-checked after** device truncation, minimum-image folding,
`vertex_cutoff`, or `support_pairs` masking
(`phonon/phonon_inputs/separable.py:176-184` uses the raw fit as-is).
The audit script referenced at `phonon/solver/se_finite.py:372`
(`phonon/scripts/verify/audit_qfold_trs.py`) does not exist in the
tree. This is hypothesis H1 below.

## 2. The rest of the bubble preserves PSD

The remaining steps are all non-negative combinations of PSD matrices,
which is a cone operation and therefore closed:

- **Quadrature over omega'.** The convolution is a zero-padded *linear*
  correlation over `n_fft = 2 n_conv - 1` with the output sliced to
  `[:n_conv]` (`sse_phonon_phonon.py:902, 1654-1655`), so every weight
  is exactly `+1` or `0`.
- **The bosonic fold.** The absorption legs are built by DFT index
  reversal plus `q -> -q` plus the ji-transpose
  (`:1080-1111`, `:2301`, `_build_fold_plan:2337-2346`), realising
  `G^<_ij(q,-w) = G^>_ji(-q,w)`. The tau reversal and the q negation
  are permutations; the ji-transpose maps a Hermitian PSD `X` to
  `X^T = conj(X)`, which has the same eigenvalues. All three preserve
  PSD, and the three-term sum is a sum of PSD terms.
- **Masking.** The `omega = 0` and low-frequency masks
  (`:1027-1048`, `:1661-1672`) multiply whole frequency bins by zero on
  both legs and output. Zero is a legitimate non-negative weight.
- **Linear mixing.** `(1-a) Sigma_prev + a Sigma_new` with `a in (0,1]`
  is a convex combination (`scba.py:713-725`).

## 3. Positivity is an invariant of the exact SCBA

**Theorem 2.** If the contact self-energy is PSD and the bubble is
evaluated on PSD legs, then positivity is preserved by the whole
iteration.

*Proof.* `Sigma^<_contact = i n_alpha Gamma_alpha` with `n_alpha >= 0`
(`solver.py:160-172`, the `omega = 0` Bose singularity clipped to zero)
and `Gamma_alpha >= 0`; the bubble is PSD by Theorem 1; the sum of PSD
matrices is PSD; `G^< = G^R Sigma^<_tot (G^R)^dagger` is a congruence,
so `-i G^< >= 0`; and the next iteration's legs are then PSD, closing
the induction. Convex mixing between two PSD iterates stays PSD. []

The consequence is what makes this question decidable: **a measured
negative occupation is a structural signature, not physics.** The
iteration cannot reach a gain state through any operation covered
above. Something outside the list is doing it.

Note also that Theorem 2 is independent of energy conservation. The
Phi-derivable identity `P_in = P_out` holds on the film to 1e-9
throughout, measured pre-mixing at every grid tested including 15001
points. **Conservation and positivity are separate gates and only the
first was ever checked.**

## 4. The band mask: why there is no clever taper

The self-energy is written only on the block-tridiagonal. The bound is
hard-wired at `sse_phonon_phonon.py:410`,

    for J in range(max(0, I - 1), min(self.n_blocks, I + 2)):

and is **independent of `sse_g_band`**, which widens only the inner G
leg loops at `:411-418`. It is also not merely a choice: the RGF reads
only `blocks[j,j]` and `blocks[i,i+1]`, so a wider Sigma would be
structurally unusable by the transport solver. Sigma is therefore
Hadamard-masked onto the tridiagonal,

    Sigma -> Sigma o M,    M[I,J] = w_{|I-J|} for |I-J| <= g, else 0.

By the Schur product theorem, `A o M >= 0` for every PSD `A` if and
only if `M >= 0`. So the question is which band masks are PSD.

**Theorem 3 (maximum-weight PSD band mask).** Let `M` be the symmetric
Toeplitz mask above with unit diagonal `w_0 = 1` (the local scattering
channel must not be reweighted).

(a) `M >= 0` for every block count `n` iff the symbol
    `f(theta) = 1 + 2 sum_{d=1..g} w_d cos(d theta)` is non-negative.

(b) At `g = 1` the eigenvalues are `1 + 2 w_1 cos(k pi/(n+1))`,
    `k = 1..n`, so `M >= 0` iff `w_1 <= 1/(2 cos(pi/(n+1)))`, which
    tends to **`w_1 <= 1/2`** as `n -> infinity`. For the boxcar
    `w_1 = 1` this gives the block-count law

        min eig = 1 - 2 cos(pi/(n+1)),

    i.e. **0** at `n = 2`, `-0.414` at `n = 3`, `-0.618` at `n = 4`,
    `-0.879` at `n = 8`, `-1` asymptotically. The damage grows with the
    number of blocks, so merging cells into fewer, bigger blocks helps
    even before it retains any extra weight.

(c) The Bartlett/Fejer choice `w_d = 1 - d/(g+1)` gives
    `f(theta) = |sum_{k=0..g} exp(i k theta)|^2 / (g+1) >= 0`, with
    `w_1 = g/(g+1) -> 1` as `g -> infinity`.

*Proof.* (a) is Herglotz: a finitely-supported symmetric sequence is
positive definite iff it is the Fourier coefficient sequence of a
non-negative density. (b) is the closed-form spectrum of a tridiagonal
Toeplitz matrix. (c) is the Fejer kernel identity. []

Three consequences, in order of importance.

1. **The boxcar is indefinite.** `w_1 = 1` at `n = 3` gives
   `min eig = 1 - 2 cos(pi/4) = 1 - sqrt(2) = -0.414`; asymptotically
   `-1`. This is the film's configuration (`sse_g_band_taper` unset,
   default `"none"`). The repo's own test already asserts it:
   `tests/quatrex/phonon/test_sse_phonon_phonon.py:678
   test_taper_restores_causality_psd` requires
   `lam_box < -1e-4 * scale`.

2. **Bartlett band-1 is not one option among many — it is the
   maximum-weight PSD tridiagonal mask.** By (b), any PSD tridiagonal
   projection has `w_1 <= 1/2` (asymptotically), and Bartlett band-1
   attains it. So *every* PSD-preserving tridiagonal projection
   underweights nearest-neighbour coherence by at least a factor two.
   There is no cleverer taper to look for. This explains the measured
   CNT bracket quantitatively: boxcar 13.19 / 15.16 / 19.33 at
   L = 16/24/32 versus tapered 6.09 / 5.61 / 5.51, a factor 2.2-3.5,
   with the boxcar current *growing* with length (the gain signature)
   and the tapered one saturating (`gpu_campaign_2026-07.md:136-186`,
   `phonon/scripts/data/gband_ladder.npz`). Neither is a converged
   transport result; they bracket it.

3. **The taper is only self-consistent at `g_band = 1`.** The weights
   are built for `g_band` (`:388-391`, `w_d = 1 - d/(g_band+1)`) but
   the output band stays at 1, so the effective output mask has symbol
   `1 + 2 (g/(g+1)) cos theta`, which by (b) is PSD only for
   `g <= 1`. At `g_band = 2` or `3` the taper does **not** restore
   output PSD, and nothing tests that — the only PSD test in the tree
   exercises `g_band = 1`.

### 4.1 How much damage does the mask actually do?

Indefiniteness of `M` is necessary for a PSD failure but not
sufficient: if `A` already has no weight outside the band, `A o M = A`
and nothing happens. The sharp statement is a perturbation bound.

**Proposition 4.** For `A >= 0`,
`lambda_min(A o M) >= -||A - A o M||_2`, the norm of the discarded
part.

*Proof.* `A o M = A - (A - A o M)`, and Weyl's inequality gives
`lambda_min(A o M) >= lambda_min(A) - ||A - A o M||_2 >= -||A - A o M||_2`. []

So the injected negativity is bounded by the weight the truncation
throws away. This is the quantitative bridge to the numerics: measuring
`||Sigma_{I,I+2}|| / ||Sigma_{I,I+1}||` bounds the damage, and it is
the discriminant between systems, since the mask itself is equally
indefinite everywhere.

### 4.2 The only escape is a wider block

Theorem 3 says the band cannot be tapered without losing a factor two
of coherence, and the BTD solver says the band cannot be widened beyond
`|I-J| <= 1`. The remaining lever is what a block *is*. Putting two
transport cells in one slab moves half the nearest-neighbour cell links
from the off-diagonal, where they must be damped, into the **diagonal
block, where the mask is exactly 1 and untouched**. The guaranteed
retained physical range goes from 1 cell to 2 cells.

The prediction is graded, not binary. With blocks `[c0 c1][c2 c3][c4 c5]`
the link `c0-c1` becomes intra-block while `c1-c2` remains inter-block,
so doubling **halves the incidence** of the truncation rather than
removing it. Measured in §6.5: a factor 1.85 on the Si L8 ladder, and
exact restoration once the blocking leaves only two blocks.

**This is a config-and-inputs change, not a solver change.** The block
size is derived from the input files alone, as
`supercell_size = (transport extent of dynamical_matrix.mat) // 2`
(`src/quatrex/device/inputs.py:993-998`, computed only along the
transport axis). A `.mat` carrying `+-1` transport neighbours gives one
cell per block; `+-2` gives **two cells per block automatically**, and
`+-3` gives three. `_get_transport_block` (`inputs.py:882-928`)
assembles them correctly -- verified against the definition:

    b00 = [[D0, D1], [D-1, D0]]      b01 = [[D2, 0], [D1, D2]]

Both exact. Note `b01` carries `D(+2)`, so the FC2 range extends in the
same move. The only reason every shipped device has one cell per block
is the builder: `phonon/studies/engine/build_inputs.py:108` calls
`get_btd_blocks_folded`, which folds all `n >= 2` FC2 coefficients into
`H00`/`H01` before writing, so `+-2` never reaches the file. The fc3
and q-fold then have to carry matching `block_sizes`; that is a
re-block, not a refit, since `fc3_to_phi_blocks(phi_dense, block_sizes)`
(`fc3_loader.py:45`) already accepts an arbitrary partition.

Caveat for the `+-2` route: `_trim_zeros_cells` (`inputs.py:770-800`)
shrinks the array to its non-zero bounding box, so the `D(+-2)` blocks
must be genuinely non-zero -- zero padding is deleted and
`supercell_size` collapses back to 1, which is the
`ValueError: inconsistent shapes` already on disk in
`cluster/mos2f3/slurm-4315507.out`.

## 5. The dual grid is exonerated

The bubble runs on its own auxiliary frequency grid
(`sse_aux_grid_dw_thz`, `sse_aux_grid_fmax_thz`). This is not the
mechanism, and there is no positivity reason to change it.

- **Prolongation P** (primary -> aux, applied to the G legs only,
  `_interp_axis0`, `:2785-2791`): `(1 - w) data[lo] + w data[hi]` with
  `w` clipped to `[0,1]` (`:2744`) and gated by a boolean valid mask
  (`:2745`). A convex combination of PSD matrices is PSD.
- **Restriction R** (aux -> primary, applied to the Sigma outputs
  only): `"adjoint"` (default, `:2764-2776`) has entries
  `(1-p) * valid * col_w / row_w` with `col_w = dw * omega_aux >= 0`,
  `p in [0,1]`, and `row_w = width * omega_prim >= 0`, with rows of
  zero weight set to zero. Every entry is non-negative. `"sample"`
  (`:2797-2801`) is again a convex combination.
- Masking is applied consistently on both grids: `out_mask` on the
  primary grid before interpolation and after restriction, `conv_mask`
  on the aux grid in between (`:1038-1045`, `:1661-1672`).

A non-negative combination of PSD matrices is PSD, so both directions
preserve positivity bin by bin. **Answer to "does it have to do with
different grids within the bubble": no.** Independent confirmation from
the existing runs: `mos2f3` (u121, `sse_aux_grid_dw_thz = 0.0`, i.e.
aux grid *off*) shows gain fraction 0.071 with worst/max `-1.00`, and
`mos2f3nu` (aux grid *on*) shows 0.045 with worst/max `-1.00`. Both
fail, so the dual grid is neither necessary nor sufficient for the
failure.

One residual dual-grid artefact, worth recording but not a PSD break:
aux bins strictly between `prim[0] = 0` and `prim[1]` receive a linear
ramp up from the zeroed DC sample, so the near-DC leg weight depends on
the primary grid's DC spacing. Non-negative, hence PSD-safe.

## 6. Hypothesis table

| # | Assumption of the theorem | Code site | Failure mode | Live in the film? | Test |
|---|---|---|---|---|---|
| H1 | vertex leg-exchange symmetry (1)/(1') | `separable.py:176-184` (raw fit); no audit exists | structural, first order in the defect | **unknown — never measured** | N1 |
| H2 | Sigma not Hadamard-masked | `sse_phonon_phonon.py:410`, RGF reads tridiagonal only | structural, `min eig` down to `1-sqrt(2)` | **yes** | N2, N3 |
| H3 | contact `Gamma_alpha >= 0` | `solver.py:238-257`; spectral NEVP at `eta_obc = 0`; no PSD check anywhere | numerical; marginal modes with `Im k = 0` can be dropped and the surface GF rebuilt from a rank-deficient pseudo-inverse | **yes**, and the complex-symmetric repair at `solver.py:246` is skipped because `kpoint_grid=[5,5,1]` makes `len(global_stack_shape)==3` | N5 |
| H4 | exact arithmetic in the RGF | `rgf.py:284-295, 364-384` | roundoff, but the PSD pieces are summed with indefinite `X^dagger - X` terms whose norm can dominate at `eta = 0` | **yes** | iteration trace |
| H5 | convex mixing | `scba.py:759-782` (Anderson/RRE, signed coefficients) | structural | no for `mixing_method="linear"`; **yes** for the Anderson runs | config audit |
| H6 | Sigma written on its full support | storage sparsity pattern (`utils.py:51-57` box mask) | structural, a second Hadamard mask | plausible; the film's 9.27 A cell against a 10 A cutoff leaves the `(0,1)` block partly uncovered | N3 |

Not suspects, verified: the Kronecker/congruence structure itself
(§1), the quadrature weights, the bosonic fold, the frequency masks,
the dual grid (§5), `n_alpha >= 0`, and the `sse_g_band` leg mask for
this device specifically — at `n_blocks = 3` the clamp gives
`g_band = 2` and the leg mask is the full ones-matrix, i.e. rank-1 PSD
and inert. (`sse_g_band` is a real PSD hazard on longer devices; it is
just not the film's.)

## 6.5 Measured (2026-08-08)

`_bubble_positivity.py vertex` and `... blocking`; artifacts
`out/positivity/vertex_symmetry.json`, `blocking.json`. The blocking run
is the Gamma slice at eta = 0, with ballistic legs from the production
spectral NEVP OBC and the 3-term bosonic fold; the three fold terms are
reported separately because each is a ring of two PSD legs and so must
be PSD on its own -- that is what validates the reimplementation.

**H1 is clean.** Every shipped vertex satisfies the congruence premise
to roundoff: reality exactly 0, leg exchange (1) at 1.3e-16 (MoS2),
7.8e-17 (Si), 0.0 (CNT), and the q-folded pairing (1') -- never checked
before, the audit `se_finite.py:372` points at does not exist -- at
1.3e-16 (MoS2, all 625 q-pairs) and 2.2e-16 (Si, 6561 pairs).

**Theorem 1 holds on real production inputs.** With PSD legs (measured
2.9e-16 / 3.5e-16 for MoS2 and Si, 2.3e-13 for the CNT), the unmasked
Sigma^< is PSD: worst relative negative eigenvalue 9.8e-14 (MoS2 nu
vertex), 8.4e-15 (MoS2 scp), and strictly positive definite for Si
(-1.4e-9) and the CNT (-2.7e-8). Each fold term separately 1e-12 to
1e-17.

**H2 is real, large and universal.** Masking that same Sigma onto the
block-tridiagonal:

| device | blocks | boxcar worst neg | Bartlett band-1 | discarded |
|---|---|---|---|---|
| MoS2 L3 (nu vertex) | 3 | **3.3e-02** | 3.3e-14 | 41.4 % |
| MoS2 L3 (scp vertex) | 3 | **4.0e-03** | 3.2e-15 | 20.0 % |
| Si film L8 | 8 | **2.0e-04** | -4.1e-07 | 63.8 % |
| CNT33 L4 | 4 | **1.6e-01** | -6.7e-07 | 43.5 % |

The boxcar destroys PSD everywhere and the Bartlett band-1 taper
restores it everywhere, exactly as Theorem 3(b,c) requires.

**The blocking ladder behaves as predicted.** Si L8 is the only local
device with enough slabs for a graded ladder:

| cells per block | blocks | worst neg |
|---|---|---|
| 1 | 8 | 1.98e-04 |
| 2 | 4 | **1.07e-04** (factor 1.85) |
| 4 | 2 | -1.4e-09 (PSD restored) |

and the CNT L4 at 2 cells per block (2 blocks) likewise returns exactly
its unmasked value. Note *why* the 2-cell rung improves: the discarded
Frobenius weight is unchanged to six digits (63.82 % both times, it is
dominated by the far corner d = 7), so the gain comes from the mask
being **less indefinite** at fewer blocks -- the `1 - 2 cos(pi/(n+1))`
law of Theorem 3(b), which predicts 0.879/0.618 = 1.42 against the
measured 1.85. Blocking buys positivity through mask conditioning
first and retained weight second.

**The discriminant is the block-distance profile.** `||Sigma_d||/||Sigma_0||`,
omega-integrated (the peak-weight bin agrees to within a few per cent):

| device | d=1 | d=2 | d=3 | far corner |
|---|---|---|---|---|
| MoS2 L3 (nu) | 0.679 | **0.550** | -- | -- |
| MoS2 L3 (scp) | 0.149 | 0.207 | -- | -- |
| CNT L4 | 0.680 | 0.432 | 0.394 | -- |
| Si film L8 | 0.002 | 0.001 | 0.001 | 0.829 (d=7) |

Sample the profile at the bin carrying the most weight, never at a fixed
"mid band" frequency: at eta = 0 the spectral function is a comb of
sharp poles, so a fixed omega can land between them where G is ~1e-20
and every ratio is denormal noise. That mistake was made and caught
here -- an earlier draft of this table quoted numbers taken at
omega = 8 THz, which for the MoS2 film is a null between poles.

For MoS2 with the production (intra-slab-only) vertex the **dropped
Sigma_02 block is larger than the retained Sigma_01** -- the profile has
no block-distance decay at all, because with a block-diagonal vertex
Sigma_IJ = M_I (G_IJ (x) G_IJ) M_J^dagger follows |G_IJ|, and a 3-slab
(~19 A) MoS2 device is fully coherent end to end. Si is the opposite:
the band captures everything except a delocalised contact-to-contact
corner.

**H3 is exonerated as the discriminant.** The contact `Gamma` is indeed
not PSD -- nothing checks it -- but the worst relative negative
eigenvalue is 2.5e-11 for MoS2, against 6.6e-05 for Si and 1.8e-03 for
the CNT. The failing system has the *cleanest* contact, so this is
anti-correlated with the disease.

### 6.6 The 6-cell MoS2 pair: blocking helps only where Sigma decays

Built with `phonon/studies/engine/reblock_device.py` (exact re-partition
plus replication along transport -- no DFT, no refit, no q-fold rerun,
since the MoS2 per-slab vertex blocks are bit-identical across slabs for
all 625 q-pairs). Two devices, the SAME 108-dof physical film:

- `cluster/mos2f6x1` -- 6 blocks x 1 cell (18)
- `cluster/mos2f6x2` -- 3 blocks x 2 cells (36), written with the 2-cell
  block as the unit cell so the production loader reads it back with no
  change to `src/quatrex`.

Build gates: the dense 6-cell FC2 operator and the dense vertex are
unchanged by the re-blocking, and the offline re-blocking of `6x1` to
three blocks reproduces the `6x2` device file **bit-identically**
(1.484e-01 worst neg, 48.48 % discarded, both ways). Unmasked Sigma is
PSD for both (7.1e-14, 6.7e-14), legs PSD to 5e-16.

On a common normalisation (the unmasked Sigma's global max |eigenvalue|,
69.3), the masked worst negative eigenvalue is

| blocking | lambda_min | relative | discarded |
|---|---|---|---|
| 6 blocks x 1 cell | **-2.73** | 0.039 | 71.3 % |
| 3 blocks x 2 cells | **-7.75** | 0.112 | 48.5 % |
| 2 blocks x 3 cells | -4.9e-12 | 0.000 | 0 % |

**Doubling the slab makes MoS2 worse by 2.8x**, the opposite of the Si
result, and it is not a normalisation artefact (the absolute eigenvalue
moves the same way). Note the 2-cell blocking discards *less* weight
(48.5 % vs 71.3 %) and has a better-conditioned mask
(-0.414 vs -0.802) and is still worse -- so neither Proposition 4's
bound nor the block-count law predicts the sign here. Both are upper
bounds, not estimates.

The block-distance profile explains it (omega-integrated):

| device | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|
| MoS2 6 cells | 0.759 | 0.778 | 0.635 | 0.561 | 0.551 |
| Si film L8 | 0.002 | 0.001 | 0.001 | 0.001 | 0.001 |

Si's Sigma decays by three orders of magnitude in one block, so widening
the retained band captures essentially all of it and the mask tends to
the identity. MoS2's does not decay **at all** over six cells (~74 A) --
d2 exceeds d1. For such a delocalised Sigma, a wider block retains more
coherent off-band weight but still cuts it off, and the discontinuity at
the band edge grows rather than shrinks.

### Why MoS2's Sigma is delocalised (it is not the force constants)

The natural worry is that the fc3/fc2 range was cut too short. It was
not, and two measurements separate the candidates.

**Not the force constants.** The FC2 interlayer coupling is
`|D(+-1)|/|D(0)| = 0.0075` -- 0.75 %, the vdW gap -- and the fitted fc3
is intra-slab only. Both are short-ranged, which is the physically
correct description of a van der Waals stack, not an under-converged
fit.

**Not the infrared channel either.** The near-DC legs (`omega < 1.5` THz)
carry **96 %** of `||G^<||` (the `1/omega^2` uniform-translation
channel), yet building Sigma from them alone gives `||Sigma|| = 0.049`
against `205.9` from the rest -- **0.02 %**. The vertex annihilates that
channel: `||Phi.G|| / (||Phi|| ||G||)` is suppressed to `2.4e-4` at the
dominant bin. This independently re-confirms the ASR vertex-cancellation
established by `_ir_killtest.py` (`ir_residue_derivation.md`), and it
rules the IR channel out as the source of the delocalisation.

**It is the propagator.** In the window that actually builds Sigma
(`omega >= 1.5` THz), the ballistic `-i G^<` block profile is

    d = 0..5 :  1.000  0.920  1.069  0.866  0.624  0.570

i.e. no decay across six cells. At eta = 0 with ballistic leads there is
no damping anywhere in the problem, so a propagating mode is coherent
end to end by construction and `G_IJ` cannot fall off with `|I-J|`. With
the intra-slab MoS2 vertex `Sigma_IJ = M_I (G_IJ (x) G_IJ) M_J^dagger`,
so Sigma inherits exactly that delocalisation.

The consequence is structural rather than parametric: **no block size
localises a coherent propagator.** Blocking can only help a self-energy
that already decays, which is why Si improves and MoS2 does not, and why
the only blocking that removes the defect for MoS2 is the one that
leaves no truncation at all.

**Rule, on two systems:** blocking buys positivity only when Sigma
actually decays with block distance. It is a convergence knob for local
self-energies, not a repair for delocalised ones. For MoS2 the only
blocking that restores PSD is the degenerate one where the device is at
most two blocks, i.e. no truncation at all.

Caveat to keep: these are **ballistic legs**. At eta = 0 the
non-interacting G carries no damping, so nothing forces G_IJ to decay;
a converged interacting state would provide its own damping and steepen
the profile. That said, the failing runs diverge from near the ballistic
state, so this is the profile that governs the onset -- and Si, whose
legs are equally undamped, decays steeply anyway, so the difference
between the two systems is material rather than an artefact of using
ballistic legs.

### 6.7 The decisive run: zero truncation does NOT stop the divergence

The 2-block device is the configuration in which the `|I-J| <= 1`
output band is complete, so **no Hadamard mask is applied at all** and
positivity is preserved exactly by Theorems 1 and 2. If H2 were the
driver, that run has to behave differently. It does not.

Same 108-dof 6-cell film, same vertex, same eta = 0 config, 121-point
grid, daint jobs 4383302 / 4383303:

| run | blocks | Sigma truncation | balance | gain frac | worst/max | outcome |
|---|---|---|---|---|---|---|
| `mos2f6x1` | 6 | maximal (71 % discarded) | 2.000 | 0.41 % | -1.00 | diverged, it 28 |
| `mos2f6x3` | 2 | **none** | 1.051 | **13.2 %** | -0.80 | diverged, it 28 |

Both abort at the *same* iteration, and the untruncated run carries
**more** gain, not less. The `Sigma^R` residual sequence is identical to
five significant figures for the first eight iterations
(1.0000, 2.7687, 2.2241, 1.7580, 1.0562, 0.74565, 0.99637, 0.99856)
across the two blockings; only the per-slab internal spread differs. So
the trajectory is set by something the blocking does not touch.

**Verdict on H2: real, provable, measurable -- and not the cause.** The
band mask is a genuine PSD defect (Theorem 3, and 3.3e-2 to 1.6e-1 of
injected negative eigenvalue in §6.5), but removing it entirely leaves
the MoS2 divergence untouched. This closes the structural line of
enquiry: H1 clean, H3 anti-correlated, H2 falsified as a driver, the
dual grid exonerated. What remains is H4 -- amplification of an O(1e-16)
seed by the eta = 0 near-singular acoustic resolvent -- which is
consistent with the CNT L4 carrying the largest injected negativity of
any system measured here and still converging.

Grid caveat: 121 points is far below the resolution the grid audit says
is needed for converged transport numbers. It is adequate for *this*
comparison, because both legs use the same grid and the divergence is
known to persist at 4001 and 15001 points for the standard blocking. A
15001-point, 8-iteration confirmation of the untruncated leg is running
(job 4383310).

**What this does and does not settle.** H2 is confirmed as a real,
first-order PSD defect present in every production run. It is not by
itself sufficient: the CNT L4 carries the largest injected negativity
(1.6e-01) and still converges cleanly. So the truncation supplies the
seed and the SCBA decides whether it is amplified -- consistent with the
CNT ladder, where the boxcar current grows with length (13.19 / 15.16 /
19.33 at L = 16/24/32) while the same kernel at L4 is stable.

## 7. What the theory says to measure

Superseded by §6.5: H1 is clean, H3 is anti-correlated, and H2 is
confirmed at 1e-4 to 1.6e-1. What remains open, in order:

1. **H4, the amplification.** H2 supplies a seed of the right sign in
   every system, but only some systems run away with it. The open
   question is no longer "where does the negativity come from" but
   "what decides whether the SCBA amplifies it" -- the eta = 0
   near-singular acoustic resolvent (`rgf.py:284-295, 364-384` sums PSD
   pieces against indefinite `X^dagger - X` terms) is the candidate,
   and the test is an iteration-resolved positivity trace, not another
   discretisation change.
2. **The MoS2 blocking ladder.** The Si ladder confirms the mechanism,
   but MoS2's profile has *no* block-distance decay, so one doubling
   may not be enough; it needs a ladder (2, 3 cells per block), which
   in turn needs a 6-cell device. See §4.2.
3. **A production positivity gate.** Nothing in `src/quatrex` checks
   PSD of `Sigma^{<,>}`, `Gamma_alpha`, or `G^{<,>}` at any point. The
   metric used here (worst negative eigenvalue relative to the global
   max, skipping bins below 1e-6 of it) is cheap enough to run per
   iteration behind a flag.
