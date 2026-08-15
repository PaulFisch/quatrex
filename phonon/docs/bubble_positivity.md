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

> **SUPERSEDED by Secs. 6.8-6.10 (2026-08-09).** This paragraph is the
> state of the enquiry on 2026-08-08 and was never revised. H4 is NOT
> what remained: **H6**, the `interaction_cutoff` box mask on the storage
> pattern, is confirmed causal by the single-variable A/B of Sec. 6.9,
> and MoS2 converges at eta = 0 once the cutoff exceeds the device
> (`mos2L2conv` 29 iterations, `mos2L4conv` 30). Read on before quoting
> this. Flagged 2026-08-15 after it was propagated into a session
> summary as the current verdict.

Grid caveat: 121 points is far below the resolution the grid audit says
is needed for converged transport numbers. It is adequate for *this*
comparison, because both legs use the same grid and the divergence is
known to persist at 4001 and 15001 points for the standard blocking. A
15001-point, 8-iteration confirmation of the untruncated leg is running
(job 4383310).

### 6.8 The assumption that actually breaks: H6, the storage pattern

If the 2-block run has no Sigma truncation, why is it still not PSD?
Because the block band is not the only mask. G and Sigma are stored in
buffers allocated on the **interaction sparsity pattern** -- a box
cutoff along transport, `compute_sparsity_pattern(grid,
max_interaction_cutoff, strategy="box")` (`scba.py:107-113`), unioned
with the `g_band` blocks (`:126-150`) and the FC2 pattern
(`solver.py:104`). Every entry outside it is silently dropped, on the
G legs *and* on the Sigma output. That is an **orbital-level** 0/1
Hadamard mask, and it is completely independent of the blocking.

The cutoff defaults to `interaction_cutoff = 10.0` Angstrom
(`config.py:1261`). The MoS2 transport cell is 12.294 A, so the 6-cell
device is **74 A** long: the mask discards most of the operator.

Measured on the full production pattern (Gamma slice, ballistic legs):

| device | fill | mask lambda_min / lambda_max | legs after mask | Sigma after mask |
|---|---|---|---|---|
| 6 blocks x 1 cell | 80.3 % | -13.75 / 88.7 | 6.7e-02 | **1.000** |
| 2 blocks x 3 cells | 41.4 % | -11.16 / 47.0 | 2.1e-01 | **1.000** |

against `6e-14` for the same Sigma unmasked. `worst_rel = 1.000` means
the most negative eigenvalue equals the largest in magnitude -- exactly
the `worst/max = -1.00` the `mos2f6x1` run reports, so the offline model
reproduces the observed gain signature.

Three things follow.

1. **The mask hits the legs first.** `-i G^<` is already indefinite
   (6.7e-2 / 2.1e-1) *before* the bubble runs, so Theorem 1's premise
   fails at the input, not at the output. No amount of care inside the
   bubble can recover it.
2. **It is blocking-independent**, which is precisely why removing the
   block truncation (§6.7) changed nothing and why the two blockings
   produced identical residual sequences.
3. **The mask encodes an assumption of locality**, and MoS2 at eta = 0
   violates it. §6.6 measured `-i G^<` as flat across six cells; a 10 A
   box cutoff on a 74 A coherent propagator throws away the majority of
   a non-negligible operator. For a genuinely local self-energy (Si,
   whose Sigma falls three orders in one block) the same mask is nearly
   the identity and harmless -- which is the discriminant between the
   failing and the clean systems.

So H2 and H6 are the same disease at two scales: Hadamard-masking an
object that is not local. H2 is the block-level instance and is not
sufficient on its own; H6 is the orbital-level instance and is
saturating.

**The prescription this implies** is a cutoff at least as long as the
device, so the pattern is dense and no locality is assumed. That is
cheap for short devices (L3 is 54 dof) and is the next test: same
config, same eta = 0, `interaction_cutoff` raised past the device
length. If the gain disappears, H6 is confirmed causal; if it does not,
the remaining candidate is H4 (the eta = 0 resolvent).

### 6.9 CONFIRMED: the cutoff is causal (single-variable A/B)

Same device (2 cells, 2 blocks, 36 dof -- so the block band is complete
and H2 is absent), same 15001-point grid, same eta = 0, same linear
mixing, same everything. **The only variable is
`phonon.interaction_cutoff`.** Daint jobs 4383378 / 4383393.

| cutoff | pattern | residual trace | lead balance | gain | worst/max |
|---|---|---|---|---|---|
| **30 A** (> the 24.6 A device) | dense, no mask | 1.000 -> 0.889 -> 0.777 -> 0.674 -> 0.579 -> 0.490 -> 0.406 -> **0.337**, monotone | **6.1e-05** | **0.00000** | **0.000e+00** |
| **10 A** (production default) | box-masked | 1.000 -> **3.7e+07**, ABORTED at it 4 | 1.3e-01 | 0.02169 | **-1.000** |

Removing the locality cutoff turns a run that explodes by seven orders
of magnitude in one iteration into a clean monotone descent with
**exactly zero negative occupation**. That is Theorems 1 and 2 verified
end to end on a production run: with no Hadamard mask anywhere in the
chain, positivity is preserved exactly, as the congruence argument says
it must be.

**Verdict for the campaign.** The MoS2 film instability is not a
physical loop-gain, not a soft mode, not the infrared channel, not the
dual grid, and not the block band. It is the **locality assumption
built into the interaction sparsity pattern** -- a 10 A box cutoff
applied to a device 25-200 A long whose propagator does not decay
(§6.6). The mask is indefinite (§6.8), it hits the G legs before the
bubble even runs, and it is what injects the gain.

Caveats, stated plainly:

- One device, one grid. The 2-cell device is the shortest that admits a
  complete block band, and it has no interior slab, so it settles the
  positivity question and nothing about the heat profile.
- This shows the instability is removed, not that the resulting
  transport number is converged.
- Cost is the real obstacle to using this in production: an adequate
  cutoff means a dense pattern, and `nnz` then grows as `L^2` rather
  than `L r_c`.

  **Correction (2026-08-09).** The claim that "the binding constraint is
  the tau buffers" was wrong, and so was the reading of the traceback
  that put `perm_cache` at 97 GB. The accounting is now a validated
  model, `phonon/studies/_memory_model.py`, gated on all three measured
  OOMs. On the 6-cell film the SSE phase is **100.1 GB against 97.4
  observed**, spread over the tau buffers (20.9), the band-link dicts
  (13.9), the `_stack` duplicate (13.9), the per-pair outputs (13.9) and
  `perm_cache` (11.7). No single term dominates. The ~39000 allocations
  in the `perm_cache` traceback are `phi_perms` CALLS, not live entries
  (4 quads x 25^2 q-pairs x 2 perms = 2500 entries = 12.6 GB).

  Three of those terms were avoidable and are now fixed behind opt-in
  switches (commit `bf4fdd31`): `sse_release_leg_blocks` (-13.9 GB),
  `sse_perm_cache_share="auto"` (-8.8 GB, the offset key collapses the
  cache to ONE distinct key on every shipped device), and a
  `--max-batch` default of 512 instead of 100000, which is what actually
  OOMed `mos2f4dense` through the ~21 RGF backward temporaries.

**Consequence for every earlier MoS2 result.** All of them ran at the
10 A default on devices of 37 A (L3) and longer, so the divergence that
dominated this campaign is very likely an artefact of that default
rather than physics. The kappa_z ladder needs re-running against a
cutoff ladder before any of it is trusted.

Immediate follow-ups: a cutoff ladder (10, 15, 20, 25, 30 A) on one
fixed device to locate the turnover; then a length ladder at
cutoff >= device length, to find where memory binds.

### 6.10 The cutoff ladder: the criterion is mask PSD-ness, to 1.4 %

2-cell / 2-block device (block band complete, so the box pattern is the
only mask), nf = 15001, eta = 0, 8 iterations, only
`interaction_cutoff` varying. The device's orbital z-span is
**21.569 A**, so the pattern is dense -- and the mask PSD -- for any
cutoff above that.

| cutoff | fill | mask lambda_min | mask PSD | outcome |
|---|---|---|---|---|
| 10 A | 65.3 % | -4.99 | no | diverged 3.7e+07 |
| 12 A | 70.8 % | -4.23 | no | diverged 1.7e+08 |
| 15 A | 84.7 % | -4.41 | no | diverged 1.1e+08 |
| 20 A | 95.8 % | -3.45 | no | diverged 5.5e+06 |
| **21 A** | **98.6 %** | **-2.53** | **no** | **diverged 4.5e+06** |
| **22 A** | **100 %** | **0.00** | **yes** | **converged, zero gain** |
| 30 A | 100 % | 0.00 | yes | converged, zero gain |

The turnover is exactly at the dense threshold, and the 21/22 pair is
the sharp version: **1.4 % of missing entries is the difference between
a clean monotone descent and divergence by six orders of magnitude.**
That rules out the reading in which the cutoff was merely too short to
capture real coupling -- 98.6 % of the weight is present at 21 A. What
matters is not how much is kept but whether what is kept is a PSD mask.

Two further checks. 22 A and 30 A agree **bit-identically**
(lead_current 2487.39 both), as they must once both patterns are dense:
so the result is cutoff-independent above the threshold, i.e. the cutoff
is converged. And the aux bubble grid preserves stability and zero gain
but is not accurate -- at `sse_aux_grid_dw_thz = 0.02` the current moves
to 2741.1, **+10.2 %**. The aux grid is a reachability lever, not a free
one.

Memory: the SSE tau buffers scale with the *aux* grid
(`n_fft = 2*ne_conv - 1`), the primary G/Sigma buffers with the primary
`nf`. Both a 6-cell and a 4-cell dense run at nf = 15001 OOMed at
~100 GB/GPU on two nodes.

**Superseded (2026-08-09).** The conclusion drawn here -- that going
longer needs a non-uniform grid "rather than more nodes, since the
2-node cap binds first" -- was wrong on both halves. The node cap was a
policy limit, not a physical one (4 nodes ran the 4-cell device at
15001), and the grid lever is far larger than assumed: on this same
device the physical current changes by **0.31 % from nf = 15001 to
4001 and -0.02 % to 2001** (jobs 4384261/4384264), because the reported
`lead_current` is the legacy unweighted sum and must be multiplied by
`dw` before any cross-grid comparison. What the grid buys is memory
(`B_G ∝ ne`), and 7.5x of it is available for ~0.3 %. The aux grid is
the expensive axis instead: `aux_dw = 0.01` costs **+2.84 %** and 0.02
costs **+10.20 %**, i.e. roughly second order in `dw`.

### 6.10b The PSD taper on device: positivity fixed, divergence not

`phonon.interaction_cutoff_taper = "triangular"` weights the retained
entries by `max(0, 1 - |z_i - z_j|/R)`, which is positive definite, so
the mask is PSD at every radius (section 4 and `cutoff_mask.json`). Run
on the 2-cell film at R = 12 A against the boxcar at the same radius,
same 15001-point grid, same 8 iterations, eta = 0:

| run | mask | worst occupation | bins < 0 | S_median | outcome |
|---|---|---|---|---|---|
| mos2f2c22 | dense (22 A) | +9.9e-15 | 0.0 % | 0.99 | converges |
| mos2f2c12 | boxcar 12 A | **-1.00** | **84.5 %** | 0.00 | diverged 1.7e+08 |
| tapT12b | **triangular 12 A** | **+2.0e-08** | **0.0 %** | 0.40 | diverged 3.7e+07 |

**The taper does exactly what it was designed to do.** Negative
occupation goes from -1.00 on 84.5 % of the live bins to zero, on the
same device at the same radius: positivity is restored end to end, which
is the strongest confirmation yet that the mechanism of section 6.9 is
correctly identified and that a PD mask is its cure.

**And the run still diverges.** So for this device at this radius,
mask PSD-ness is necessary but not sufficient. The reason is visible in
the sum rule: S_median falls from 0.99 (dense) to 0.40, i.e. the R = 12
triangle discards about 60 % of the spectral weight of a device whose
span is 21.57 A. What is left is a PSD but severely truncated model, and
*that* model diverges by amplitude with positivity intact -- the same
signature as the Si film (`grid_audit.md`), not the mask signature.

**The R = 30 control settles it, negatively.** At R = 30 A the support
already covers the 21.57 A device, so the BOXCAR at that radius is the
dense reference and converges (2487.39). Adding the triangular weight to
that same complete support -- w = 0.28 at the device span -- makes it
**diverge at 1.76e+06**. The taper therefore does not merely fail to
rescue a truncated model: applied to a model that was converging, it
breaks it.

That is decisive against the taper as a scaling route here. Reweighting
the long-range part of G and Sigma sharpens the resolvent, and at eta = 0
the amplitude instability that the Si film shows (`grid_audit.md`) takes
over. The 6-cell run at R = 25 A (span 70.7 A) diverged at 9.8e+03, same
picture.

The honest reading: the taper is a correct and verified cure for the
POSITIVITY defect -- negative occupation goes to exactly zero -- and it
is not a cure for the run. It buys stability against the mask mechanism
and loses it to the resolvent mechanism. The earlier hope that a cheap
PSD mask buys long devices for free is withdrawn. What it is still good
for is diagnosis: it cleanly separates the two failure modes, which is
how the 2-cell device at R = 12 was shown to be suffering from both.

Any future use needs the amplitude side handled first (damping to ~0.02
or a damped-warm-started Anderson, which is what makes the refined Si
grids descend), and then an R-ladder to find whether a PSD radius exists
that is mild enough to be accurate.

### 6.11 Why only MoS2: the mask is PSD for every clean system

The same box mask, evaluated at the production default 10 A on each
system's own device geometry, predicts the behaviour of every run in
this campaign with no exceptions and no fitting:

| system | tiling vector | mask axis | extent on that axis | fill @ 10 A | mask PSD | observed |
|---|---|---|---|---|---|---|
| MoS2 film L3 | a3 = (0,0,12.294) | z | 33.86 A | 46.9 % | **no** (-6.27) | diverges |
| MoS2 film 6 cells | a3 | z | 70.74 A | 25.2 % | **no** (-7.06) | diverges |
| CNT33 L4 | a3 = (0,0,2.459) | z | 8.61 A | 100 % | yes | converges |
| Si film L3 nk5 | a1 = (0,2.734,2.734) | x | 1.37 A | 100 % | yes | converges |
| Si film L8 nk9 | a1 | x | 1.37 A | 100 % | yes | converges |

MoS2 is the only system whose device extends past the cutoff along the
axis the mask measures, and it is the only one that fails. The CNT is
short along transport (2.46 A per cell, 8.6 A over four cells), so 10 A
covers it whole.

Si is dense for a subtler reason worth recording, because it is a latent
trap rather than a property of the material. `transport_ind` is used
**twice with different meanings**: as a lattice-vector index when the
device is tiled
(`create_coordinate_grid`, `inputs.py:99-106`, `coords + i *
lattice_vectors[transport_ind]`) and as a **cartesian axis** when the
mask is built (`compute_sparsity_pattern(..., strategy="box")`, which
compares `positions[:, axis]`). Those agree only for an axis-aligned
lattice. Si's film is the FCC primitive with
`a1 = (0, 2.734, 2.734)` -- zero x-component -- so tiling along a1 never
extends the device along x, the Si device has an x-extent of 1.37 A at
*any* length, and the box mask is unconditionally dense. Si is therefore
immune to this bug by accident of its cell orientation, not because its
self-energy is local.

That also corrects the earlier reading in §6.6/§6.8, which attributed
Si's cleanliness to the locality of its Sigma. The locality is real (the
block-distance profile falls three orders in one block) but it is not
what protects Si here: Si simply never has a mask to be damaged by.

**Consequence:** the box strategy is only meaningful when the transport
lattice vector is parallel to the named cartesian axis. For any device
where it is not, the cutoff silently does something other than what it
says. That is worth a validator.

### 6.11a Neither mask is active on Si (2026-08-10)

Two masks could in principle damage a Si run, and at the production
settings neither does. This closes the "is the Si divergence a
truncation artefact" question without a run.

The **band mask** is complete. `sse_g_band` is clamped to `n_blocks - 1`
(`sse_phonon_phonon.py:372`), so on the short Si devices the default
`sse_g_band = 3` never truncates: 4 blocks give band 3, 3 blocks band 2,
2 blocks band 1, and in each case the band reaches every block pair. The
mask is the all-ones matrix, whose `lambda_min` is 0. A blocking ladder
over these devices therefore cannot vary mask positivity at all; the
band-mask row of §4 applies only where `sse_g_band < n_blocks - 1`.
cvSiA ran with no `QX_GBAND` override, so it had complete coverage.

**But blocking is not thereby irrelevant, and the ladder measured it.**
si4x1 and si4x2 are the same 4-primitive-cell device (the reblock
verifies the dense FC2 and fc3 operators are unchanged, and the
iteration-0 currents are bit-identical: 216.8884, 86.1743, 29.377), and
they differ only in BTD partition -- 4 blocks of 6 dof against 2 blocks
of 12 dof. Their SCBA paths do not resemble each other:

| iteration | 1 | 3 | 5 | 7 | 9 | 11 | per-it factor |
|---|---|---|---|---|---|---|---|
| si4x1, 4x6 | 29.79 | 26.95 | 24.38 | 22.05 | 20.97 | -- | 0.951 |
| si4x2, 2x12 | 0.978 | 0.764 | 0.628 | 0.510 | 0.407 | 0.317 | 0.900 |

Both have a complete, PSD mask and identical ballistic transport, so
neither mask positivity nor the physics accounts for the factor ~30 in
iterate amplitude. Whatever drives the Si instability is carried by the
block partition itself, not by the truncation the partition implies.
Candidate channels, none yet tested: the count of independent
off-diagonal Sigma blocks the SSE assembles (16 pairs against 4), the
RGF off-diagonal post-pass at band 3 against band 1, and the OBC NEVP
conditioning of a 6-dof against a 12-dof lead cell. The practical
reading is that coarser blocking is the stabler way to run Si.

The **box mask** is inert for the §6.11 reason, and the margin is
thinner than that section implies. Measuring separation along the
cartesian x (what the code does) gives a 1.37 A extent at any length.
Measuring it along the tiling vector a1 itself (what the cutoff means)
gives 3.87 A per cell:

| Si cells | span on x (code) | fill @ 10 A | span on a1 (intended) | fill @ 10 A | lambda_min |
|---|---|---|---|---|---|
| 3 (cvSiA) | 1.37 A | 100 % | 9.67 A | 100 % | -6.1e-15 |
| 4 (si4x1/2) | 1.37 A | 100 % | 13.53 A | 90.6 % | **-2.77** |

At 3 cells the device is shorter than the cutoff on either reading, so
the convention bug is invisible and the mask is dense either way. At 4
cells the two readings part company: the intended cutoff would truncate
and inject a first-order PSD defect, and only the axis bug keeps the run
dense. Fixing the validator called for in §6.11 would therefore change
the behaviour of 4-cell Si, and any such fix must be landed together
with a cutoff larger than the device, not on its own.

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

## 7. Distributing the transverse-q axis (2026-08-12)

The q axis used to be replicated on every rank: `allocate_data` built
`(stack_size, *global_stack_shape[1:], nnz)` and only axis 0 was split, so
a 25- or 81-point mesh multiplied every buffer everywhere and raising
`q_comm_size` made per-rank memory WORSE (it shrank the stack section and
left `nq` whole). `DSDBSparse` now sections it, opt-in via
`q_distributed=True`, with `local_q_shape`, `local_q_offset`, `q_owner`
and `local_q_index`.

Two properties keep that contained. The q axis takes no part in the block
structure or the nnz pattern, so `dtranspose` -- an all-to-all over the
stack and nnz axes -- carries it through untouched. And every peer of a
stack all-to-all or a block halo holds the SAME q section, because the
rank layout gives the stack communicator colour `q_idx * block + block_idx`
and the block communicator colour `rest`, so both are taken at fixed
`q_idx`; buffers match with no q-aware padding. The partition is
`rank * nq // size`, the one the SSE's external-q loop already uses, so the
owned data slice and the owned loop range coincide.

`nq > 1` with `comm.block.size > 1` also works now. That was refused
because `_exchange_band_halo` sized its buffers `(local_tau, b_K, b_Kp)`
with no transverse axis; it takes them from the buffer now. The bosonic
fold needed nothing -- it works on the nnz axis with `data.shape[:-1]`
carried through.

### Why the SSE cannot yet consume a sectioned G

The bubble is a CONVOLUTION over q: at external `q_ext` it sums over
internal `q'` the pairs `(q', q_2)` with `q' + q_2 = q_ext`
(`_contract_dense_q`, `iq2 = qdm[iq_ext, iqp]`). So owning a slice of `q'`
is NOT enough to compute even one owned `q_ext` -- every term needs a
SECOND internal momentum, generally on another rank. This is why the
current design deliberately replicates: "the q-folded internal q' Green's
functions are kept whole/local on every rank ... no internal-q gather".

`compute()` therefore refuses a sectioned G rather than running on it. It
would not crash: `nq` is read from `global_stack_shape` at one site and
from the local `data.shape` at another, so the q-difference and `q -> -q`
arithmetic would silently index the wrong momenta and the error would
surface as physics.

### The rotation that lifts it

Systolic, not a gather -- a gather would restore the replication it exists
to remove:

* each rank keeps its own internal slice `A_r` fixed as one leg;
* a second buffer `B` starts as `A_r` and is passed once around `comm.q`,
  `P_q` steps;
* at step `s` rank `r` holds `(A_r, B_{(r+s) \bmod P_q})` and contracts
  every pair `q' in A_r`, `q_2 in B`, accumulating at `q_ext = q' + q_2`;
* across all `r` and `s` each ordered slice pair occurs exactly once, so
  the q sum is complete;
* the accumulated `Sigma` is then reduce-scattered over `comm.q` so each
  rank keeps its own `q_ext` slice.

What this does and does not buy, and the distinction matters for sizing:
the LEGS drop to `nq/P_q` plus one rotating slice, and they are the
dominant term (~16 buffers in the model of `phonon/studies/_memory_model.py`).
The `Sigma` tau accumulators do NOT -- the `q_ext = q' + q_2` produced by
one slice pair are spread over the whole mesh, so they stay full-`nq`
until the reduce-scatter. The saving is therefore large but not `P_q`-fold,
and a model that reports it as `P_q`-fold will under-size a launch.

Implementation notes for that work: the qtasks cache is keyed
`(q_lo, q_hi, nq)` and would additionally have to carry which slice pair is
held; the dense q-folded vertex `qv` is indexed by global `(q1, q2)` and
needs the same slicing; and the `sse_greater_from_lesser` cross-term
accumulator and the `q_ext -> -q_ext` gather both cross ranks once `q_ext`
is sectioned.

### The block-count floor the first device run exposed (2026-08-13)

`nq > 1` with `block_comm_size = 2` had never run on a device before the
halo fix, and the first attempt (daint 4419787, 2-cell MoS2 film, `nblk =
2`) died at iteration 0 on every odd rank with

```
IndexError: Negative block indices are not supported.
```

four frames below the precondition it violated. Both contacts build their
periodic superblocks from a diagonal block and its immediate
off-diagonals, so `_compute_obc` reads `blocks[n, n]`, `blocks[m, n]` and
`blocks[n, m]` with `n = num_local_blocks - 1` and `m = n - 1`. A rank
holding one block has `m = -1`.

The two contacts are not symmetric here. Rank 0 survives on a single
block: `local_block_sizes` is sliced as `block_sizes[offset:]` rather than
`block_sizes[offset:next_offset]`, so it runs to the end of the device,
and the rank holds the nnz for the trailing rows -- `blocks[1, 0]` was
checked against a dense reference at `-np 2` and returns the right values.
The last rank has no such slack, because the block it needs lies behind
its own offset. So the floor is on the LAST section only, and
`validate_obc_block_sections` in `src/quatrex/phonon/solver.py` refuses
exactly that case.

Combined with the band-halo bound already in the SSE, a block-parallel
configuration must satisfy both

```
sections[-1] >= 2                       contact OBC
min(sections) >= g_band + 1             band halo (only checked for g_band > 1)
```

with `sections = get_section_sizes(n_blocks, block_comm_size)`. For
`block_comm_size = 2` that is `n_blocks >= 4` outright, and `n_blocks >= 6`
to keep `g_band = 2`. The 2-cell films that most of the MoS2 campaign
used cannot be split at all; `mos2f6x1` (6 cells, `sections = [3, 3]`) is
the shortest bed on disk that can.

The band-halo check does not subsume the OBC one: it is guarded by
`g_band > 1`, so a run at `g_band = 1` passes it with one block per rank
and then dies in the OBC. That is precisely what 4419787 did.

### The parity pair: the block split changes no physics (2026-08-13)

`mos2f6x1` (6 cells, `nq = 25`, `ne = 1001`, `g_band = 2`, `eta = 0`,
3 SCBA iterations, 8 ranks on 2 GH200 nodes), run twice at identical
settings and differing only in `block_comm_size`:

| | `bcs = 1` (4434361) | `bcs = 2` (4434371) |
|---|---|---|
| `lead_current` | 74.67378382 | 74.67378382 |
| `last_heat[0]` | 136.64106226 | 136.64106226 |
| `last_heat[-1]` | 12.70650537 | 12.70650537 |
| it-0 current conservation, abs | 16569.77588 | 16569.77734 |
| it-1 / it-2 rel Sigma^R residual | 9.5299e3 / 7.2545e3 | 9.5299e3 / 7.2545e3 |
| lead balance, it 1 / it 2 | 9.6842e-01 / 1.6597e0 | 9.6842e-01 / 1.6597e0 |
| GPU mempool peak | 14.27 GB | 16.80 GB |

The lead currents agree to every printed digit across the whole
three-iteration trajectory, not just at iteration 0. The one iteration-0
number that moves, the absolute current conservation, differs by 8.8e-8
relative -- reduction order, on a sum of order 1.7e4.

The internal spread is NOT comparable between the two columns (4.83 vs
1.66) and its disagreement is not a parity failure: the distributed RGF
leaves the internal interfaces `NaN` by construction, so the NaN-aware
`nanmax - nanmin` sees only the two leads and the spread collapses onto
the lead balance. `last_heat` shows this directly -- `[136.64, 368.49,
319.87, 274.43, 343.36, 7.64, 12.71]` at `bcs = 1` against `[136.64, nan,
nan, nan, nan, nan, 12.71]` at `bcs = 2`. Only the lead balance is a
cross-configuration gate.

Both runs diverge (residual ~1e4 by iteration 1, sign inversion at
iteration 2). That is the known MoS2 behaviour at these settings and this
pair says nothing about it: 3 iterations at `ne = 1001` is a plumbing
test, and no conductance claim follows from it.

Memory: the model in `phonon/studies/_memory_model.py` predicted
19.0-24.2 GB at `p_block = 1` and 24.7-30.2 GB at `p_block = 2`, against
14.27 and 16.80 GB measured. It over-predicts by 25-45 % here, well
outside the ~10 % it achieves on the fixtures, and in the safe direction.
The measured cost of the block split itself is +18 % per rank, which is
the halo and the duplicated boundary blocks.

### Si repeats the parity, and exposes what the memory model gets wrong

`sifilm5b` (5 cells, `nq = 81`, `ne = 1001`, `g_band = 1`, `eta = 0`,
3 iterations, 8 ranks on 2 nodes). Five blocks section as `[3, 2]`, so
this run sits exactly on the new floor -- the last rank owns the minimum
two blocks the contact OBC needs.

| | `bcs = 1` (4434636) | `bcs = 2` (4434643) |
|---|---|---|
| `lead_current` | 12629.3405080856 | 12629.3405078001 |
| it-0 / it-1 / it-2 conservation, abs | 1393.27591 / 90.28466 / 410.10090 | 1393.27590 / 90.28467 / 410.10090 |
| it-1 / it-2 rel Sigma^R residual | 7.3553e1 / 4.2493e1 | 7.3553e1 / 4.2493e1 |
| lead balance, it 1 / it 2 | 3.3939e-03 / 1.7774e-02 | 3.3939e-03 / 1.7774e-02 |
| GPU mempool peak | 8.68 GB | 12.96 GB |

The lead currents agree to 2.3e-11 relative. Si also behaves where MoS2
does not: the residual falls (73.6 -> 42.5) and the lead balance stays at
1e-2, against MoS2's 1e4 residual and sign inversion at the same
iteration count.

The memory model is wrong on Si by a factor of three, and the term
breakdown says why. It predicts 26.9-28.5 GB at `p_block = 1`, of which
`perm_cache` alone is 20.57 GB (76.5 %); the run peaked at 8.68 GB. The
remaining modelled terms sum to 6.33 GB, so the true `perm_cache` was
about 2.4 GB -- an 8.8-fold collapse, squarely inside the 4-16x that
`sse_perm_cache_share` is documented to buy. The formula
`32 b^3 (nq^2/P_q) Q` is the ABSOLUTE-key cache; every run in this
campaign sets `QX_PERMSHARE=auto`, and the model has no term for it.

That is a large error in the safe direction, and it is worth leaving
alone rather than fitting: one measured collapse factor is one data
point, and the factor depends on how the vertex keys coincide, which is
structure-specific. The rule to use when sizing is that a prediction
dominated by `perm_cache` is an upper bound and nothing more. On MoS2,
where `perm_cache` is 3.7 % of the total, the model is within 25 %
(14.27 against 19.0-24.2, and 16.80 against 24.7-30.2).

Measured cost of the block split itself: +18 % per rank on MoS2, +49 % on
Si. It is not free, and on a device that already fits it buys nothing but
the ability to go longer.
