# The spatially analytic Green-function tail

Working record of the programme in
`~/Downloads/spatially_analytic_G_bubble_experiment_plan.md` (2026-08-27): can
the long-range spatial part of `G` be carried by complex-band modes, and does
that recover a transport-relevant part of the cubic self-energy more cheaply
than a wider explicit band or reblocking.

**Status: gates not yet decided.** What is below is what the machinery has
established, including three corrections to the proposal's own statements and
one result that constrains which beds can be used at all. Read
`spatial_truncation_derivation.md` first; it derives the support law and records
the prior verdict this programme is designed against.

## 1. What the proposal got right, and three things it did not

### The modal-pair vertex projection is correct

The proposal's Eq. (47) survives contact with the kernel's actual index
convention. With `phi_left(a,c,e)`, `phi_right(J,d,b)`, `G_a` on `(c,b)` and
`G_b` on `(e,d)` (`phonon/solver/bubble.py:61-84`), and with
`alpha = K1-I`, `beta = K2-I`, `gamma = K2'-J`, `delta = K1'-J`, the two leg
separations are `R+delta-alpha` and `R+gamma-beta`, which is Eq. (2). The DOF
contraction factorises because `c, e` touch only the left vertex and the
`omega` leg while `b, d` touch only the right vertex and the `Omega-omega` leg.

One hazard for the implementation and not for the algebra: the proposal's
`a,b,c,d` are CELL indices and `bubble.py`'s are DOF indices. They collide.

### `zeta^R` does not leave the frequency integral

`xi_p = xi_p(omega)` and `eta_q = eta_q(Omega-omega)`, so Eq. (47) keeps
`[xi_p(omega) eta_q(Omega-omega)]^R` under the integral and the recurrence
`t_{R+1} = zeta t_R` (Eq. 49) acts on the INTEGRAND, not on `Sigma_R`. It is
still separable, so the FFT survives, but the cost is then

    explicit  ~ 49 * 3 * n_fft * n_dof^3          per output block
    analytic  ~ 49 * r^2 * n_fft log n_fft
    ratio     ~ r^2 log n_fft / (3 n_dof^3),   r = M * n_dof

with `M = 2p+b` the range of the dressed `Sigma^R`. At full rank that is 0.6x on
Si (`n_dof = 6`) and 0.1x on CNT (36). So the analytic contraction verifies the
algebra; it does not by itself buy a saving, and any saving has to be argued
against the production RGF's cost of producing far `G` blocks rather than
against the dense reference. `zeta^R` only factors out once summed over `R`,
which is the auxiliary realisation of the proposal's Sec. 28.

### The PSD sign is the opposite of the proposal's

The proposal writes `-i Sigma^{<,>} = L L^dagger`. This tree's convention is the
other one: `grids.boson_contact_self_energies_from_gamma` sets
`Sigma^< = -i n Gamma`, so `i Sigma^<` is the positive object. Measured on a
frozen chain, the negative spectral weight is

| object | of `+i M` | of `-i M` |
|---|---|---|
| `G^<` | 5.6e-04 | 9.994e-01 |
| `G^>` | 5.6e-04 | 9.994e-01 |
| `Sigma^<_tot` | 1.2e-01 | 8.8e-01 |

Getting it backwards does not fail loudly; it clips almost the whole spectrum
and returns a factor of the wrong object. The 1.2e-01 in the correct sign is
not a numerical artefact -- it is the anharmonic source's own non-positivity,
which `bubble_positivity.md` has been tracking for other reasons.

### A block-Hankel rank is not an exponent count

For `g_n = sum_p A_p xi_p^n` the Hankel matrix factors as `H = L R` with `L`
carrying `xi_p^i A_p`, so

    rank H = sum_p rank(A_p) <= r * b.

A scalar sequence's Hankel rank IS its exponent count; a block sequence's is
that count times the residue rank. Verified by planting both: four exponentials
with full-rank 2x2 residues give Hankel rank 8 and four distinct exponents at
multiplicity 2; the same four with rank-one residues give Hankel rank 4. The
whole programme turns on an exponent count, so the two numbers are reported
separately.

## 2. The shell decomposition replaces the band sweep

The proposal's E1 sweeps `g_cutoff` and differences the results. That cannot
answer the question it is asked.

- `supp Sigma^(b) = 2p + b`, so beyond `R > 2p+3` the band-3 reference is
  identically zero and the "long-propagation share" is 1 for free. The sweep is
  informative only on `2 <= R <= 2p+3`, which is the window
  `spatial_truncation_derivation.md` already measured as vertex-dominated.
- The partial sums are cumulative. `Sigma` is bilinear in `G`, so raising the
  band changes blocks that already existed, through interference; a difference
  of two bands is not the contribution of a shell.
- `Sigma^(b=0)` is not "the vertex-near term". `_filter_g_blocks` keeps the
  block-DIAGONAL `G_KK`, and a diagonal block of a dense inverse already
  carries the whole device's long-range physics.

Splitting the legs by distance shell instead is exact:

    Sigma_R = sum_{m,m'} Sigma_R^{(m,m')} ,   m = |K - K'| .

`compute_phph_self_energy` already knows `(K1,K1',K2,K2')` per task, so this is
an extension of the accumulation key (`se_finite.py`, `shell_bins`/`shells_out`,
default off). Measured on the real kernel: the shells sum to the undecomposed
ring at 1.9e-16, and the shelled total agrees with the default path at 2.8e-16,
the roundoff of regrouping a 36-term sum.

It also yields the finer geometry statement the derivation implies and nothing
had checked: an output at separation `R` takes weight only from leg shells in
`[R-2p, R+2p]`. Zero violations at any output distance on the analytic bed.

On a converged 8-cell chain the decomposition separates cleanly -- `R = 0..2` is
carried by shells 0-2 and `R >= 5` almost entirely by shells `4-5` and `6+` --
which is the vertex-near / propagation-tail split the proposal asks for, and
which the band sweep cannot produce.

## 3. The sizing law: 12 cells is too short

A pure-tail output block needs the leg reach `R >= R0 + 2p` AND both endpoints
clear of the edges. There are two edge effects, not one. `build_device_fc3_blocks`
emits only `0 <= K,K' < n_slabs`, so the `(alpha,beta)` sum loses terms within
`p` of an end and the projected vertex becomes `I`-dependent there; and the OBC
matches the UNDRESSED lead while the interior is dressed, so `G^R` near a
contact carries the growing branch too. With that margin `m_edge`,

    admissible I:  p + m_edge <= I  and  I + R <= N - 1 - p - m_edge
    => a pure-tail block exists only if  N >= R + 2(p + m_edge) + 1 .

At `R0 = 4, p = 1, m_edge = 2` that is `R >= 6` and `N >= 13`. A 12-cell device
has an EMPTY pure-tail region. Every tail statistic prints its admissible-`I`
count and fails loudly when it is zero.

## 4. The reference solver at eta = 0 diverges on both real beds

This constrains which frozen states can be used and was not anticipated.

`phonon/solver`'s dense SCBA, at `eta = 0` exactly, with the spectral (NEVP) OBC
and NO spatial cutoff at all (`sigma_cutoff = g_cutoff = None`):

| bed | blocking | grid | outcome |
|---|---|---|---|
| gapped 1-DOF chain, L8-L10 | -- | fmax 9, dw 0.05-0.075 | converges, resid 1e-8, conservation 3e-4 |
| Si film (transverse q=0), L16 x 6 dof | 1 cell/block | fmax 32, dw 0.133 | linear: stalls at resid 0.80; Anderson: diverges, guard aborts at iteration 24 |
| CNT (3,3) Gamma, L13 x 36 dof | 1 cell/block | fmax 98, dw 0.456 | diverges, guard aborts at iteration 7 |

Both real beds show `Gamma sign viol` on nearly every frequency sample from the
first iterations -- `Sigma^R` with the wrong sign of `Im`, i.e. gain. Neither
run has any mask to blame: this is the untruncated kernel.

That is consistent with what the tree already records rather than new. Si has no
fine-grid limit (`si-no-fine-grid-limit`), and `bubble_positivity.md` Sec. 6.11a
measured the same Si device converging at 2 cells per block and not at 1, with a
factor ~30 in iterate amplitude and no mask difference between them --
"whatever drives the Si instability is carried by the block partition itself".
Reproduced here: the reblocked device (4 blocks x 12 dof) reaches resid 0.91 in
60 iterations where the same 8 primitive cells at 1 cell/block reach 0.99, and
it does so 68x faster.

**Consequence for the programme.** The frozen state has to be the one that
physically exists, which is the state production reaches -- `sigma_cutoff = 1`,
`g_cutoff = 3`, and >= 2 cells per block on Si. That is not a compromise of the
experiment: "frozen" means the arms differ only in how `Sigma` is REPRESENTED
when evaluated on a fixed state, and the state is an input to that, not an
output of it. It does mean every number carries the settings its state was
converged at.

## 5. First result: arms A and B are bit-identical

On a converged 8-cell chain, the four-arm factorial

| arm | `sigma_cutoff` | `g_cutoff` | J_L |
|---|---|---|---|
| A | 1 | 3 | 4.184534e-10 |
| B | 1 | None | 4.184534e-10 |
| C | None | 3 | 4.274608e-10 |
| D | None | None | 4.229376e-10 |

A and B agree to every printed digit. Widening `G` while the output stays pinned
at `|I-J| <= 1` changes nothing, which is
`spatial_truncation_derivation.md`'s "the pin does not care how far `G` reaches"
reproduced on a different bed, through different code, and at the level of a
current rather than a block norm. `b = 3` is exact for the retained band, so
this is the expected answer and it is a check on the machinery.

`C -> D` -- given the pin has already been removed, does widening `G` move a
current -- came out at 1.07e-02 against a pre-registered floor of 9.10e-03 on
that bed. Not quoted as a result: an 8-cell chain has no pure-tail region
(Sec. 3) and it is a 1-DOF bed with a random vertex.

## 6. A trap: a grid sample on a lead band edge is singular at eta = 0

The group velocity vanishes at a band edge, so at `eta = 0` the surface Green's
function has no imaginary part to regularise it and the Dyson solve raises
"singular matrix". Measure zero in principle and immediate in practice, because
`fmax` and the band edges are both round numbers: the gapped chain's edge at
1.0 THz is hit exactly by `fmax = 9, nfreq_pos = 180`. The bed builder nudges
the sample count until no sample lands on an edge, ignoring edges below 1e-3 THz
(the acoustic zero at Gamma, which sits at `omega = 0`, is excluded from every
physical integral by `pos_mask`, and is regularised by `dc_handling`).

## 7. Prototypes, and what they do not settle

Two measurements taken while designing the programme, on a 40-cell scalar chain
with semi-infinite contacts and uniform dressing. They are prototypes, not
results: one degree of freedom, one frequency, and a source that is not an SCBA
self-energy.

**Contacts only.** `G^R` has Hankel rank 2 in `R` at fixed source -- not 1,
because an interior source sees both contacts -- and `G^<` also rank 2, with
exponents `{lambda, 1/lambda}` to four digits. The second branch is the retarded
root's GROWING partner, carrying the wave from the far source. So the proposal's
single-index decaying form (its Eq. 15) is not what a two-terminal device
produces.

**Uniform interior source.** `G^<` rises to rank 4, with moduli
`{|lambda|, |lambda|, 1/|lambda|, 1/|lambda|}`. A four-term fit
`(A+Br) lambda^r + (C+Dr) lambda^{-r}` does NOT reproduce the sequence, so the
structure is richer than a doubled pair and this prototype does not settle it.
That is what the source-resolved experiment is for: three arms (left contact,
right contact, anharmonic source), whose sum must reproduce the frozen `G^<` by
linearity -- measured at 3.6e-16 -- and three directional pencil estimates per
arm, where the source-resolved algebra predicts the advanced conjugates along
`J`, the retarded roots along `I`, and their products along the diagonal.

What is already clear from the algebra: the residue carries
`(lambda_m lambda_n^*)^I`, so `Sigma` is SEMISEPARABLE, not Toeplitz, and the
proposal's Eq. (53) state-space ansatz `Sigma_R = C A^{R-1} B` is the wrong
class -- it is the special case `mu_a nu_a = 1`. The semiseparable class has the
same `O(N r)` matvec through prefix/suffix recurrences, so nothing is lost by
fitting the right one.

## 8. Where the code is

| what | where |
|---|---|
| pencil -> modes: batched, arbitrary degree `2M+1`, NEVP residual as a mask | `src/quatrex/phonon/spatial_modes.py` |
| coefficient fit, amplitude pruning, geometric sums | `src/quatrex/phonon/spatial_fit.py` |
| block-Hankel rank, block-ESPRIT, semiseparable fit | `src/quatrex/phonon/spatial_hankel.py` |
| exact shell decomposition of the bubble | `phonon/solver/se_finite.py` (`shell_bins`) |
| analytic beds | `phonon/solver/toy_models.py` |
| frozen device bed + study scaffolding | `phonon/studies/_spatial_bed.py` |
| tail attribution + the four-arm factorial | `phonon/studies/_spatial_tail_tails.py` |
| source-resolved Keldysh rank | `phonon/studies/_spatial_tail_rank.py` |
| invariants | `tests/quatrex/phonon/test_spatial_tail.py` |

The pencil generalisation is the one change that alters an existing answer.
`bloch_modes` solved a quadratic; once the output pin is removed `Sigma^R` has
range `M = 2p+b > 1` and the recurrence is `sum_{n=-M}^{M} a_n lambda^n = 0`.
`qttools.nevp.NEVP` was already defined for the general case and `Full`
linearises any length -- checked on a degree-4 3x3 pencil, all 12 roots at
8.8e-14. The consequence propagates: the root count is `2Mb`, the retained
branch `Mb`, so `r = M n_dof` and not `n_dof` in every cost estimate.

## 9. Open

The gates. E1's `C -> D` on a real bed, and the source-resolved Keldysh rank on
the same states, both waiting on frozen beds converged at production settings.
