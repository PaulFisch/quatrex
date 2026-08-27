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
output of it.

### Reblocked, at the production settings: still no frozen real bed

Following the tree's own recipe -- coarser blocking plus `g_band = 3` -- on
devices reblocked exactly (`reblock_device.py` verifies the dense FC2 and FC3
operators are unchanged), with `retarded = "half"`, `sigma_cutoff = 1`,
`g_cutoff = 3`, `eta = 0`:

| bed | blocking | blocks x dof | outcome |
|---|---|---|---|
| Si film | 2 cells/block | 16 x 12 | diverges, resid 95 by iteration 16 |
| Si film | 3 cells/block | 12 x 18 | oscillates at resid ~1.1, `dJ/J` 2 %, **conservation 0.05-0.09** |
| Si film | 1 cell/block | 16 x 6 | stalls at resid 0.80 (`fft`) / diverges (Anderson) |
| CNT (3,3) | 1 cell/block | 13 x 36 | diverges at iteration 7 |
| CNT (3,3) | 2 cells/block | 8 x 72 | running |

Coarser blocking does help, and measurably: the 3-cell blocking brings the lead
balance from 1.000 -- both leads emitting, the divergence signature -- to
0.05-0.09, while the 1-cell and 2-cell blockings do not. That is the tree's
`bubble_positivity.md` Sec. 6.11a result reproduced from the other side. But the
`Sigma` residual still does not fall, so none of these is a frozen state.

Three things the dense reference does not have that production does, each a
candidate and each a change to a shared reference solver rather than to this
programme: per-frequency mixing (`low_freq_mixing_factor = 0.02` in the stored
config, and `scba_loop` has no per-frequency mixing at all); the frequency grid
(production runs `energy_window_max = 15` on Si, i.e. `fmax ~ omega_max`, where
the aliasing gate here forces `2 omega_max`); and the IR machinery
(`sse_low_freq_mask_thz`, `eta_ir_floor`), which an acoustic device at `eta = 0`
plausibly needs and which the gapped chain demonstrably does not.

So the quantitative results below are the analytic chain's. Whether they carry
to a device is open, and closing it is a question about the reference solver's
convergence, not about the spatial representation.

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

## 9. First gate readings, on a converged 20-cell chain

`chain_L20`: gapped 1-DOF chain, 20 cells, cubic 6e16, 281 frequencies,
`eta = 0`, converged to `resid = 9.7e-09` with `conservation = 1.4e-03` in 66
iterations. It satisfies the sizing law (`R <= 13`). The floor pre-registered
from its own conservation error is `4.06e-03`.

### E1, the four-arm factorial

| arm | `sigma_cutoff` | `g_cutoff` | `eps(J_L)` | first-order `dJ_L/J_L` |
|---|---|---|---|---|
| A | 1 | 3 | 6.311e-03 | -1.51e-02 |
| B | 1 | None | 6.311e-03 | -1.51e-02 |
| C | None | 3 | 6.382e-03 | -1.66e-03 |

**A and B agree to every printed digit**, on this bed as on the 8-cell one.
Widening `G` while the output stays pinned at `|I-J| <= 1` changes no current at
all -- `spatial_truncation_derivation.md`'s "the pin does not care how far `G`
reaches", reproduced at the level of a current rather than a block norm, on a
different bed, through different code.

`C -> D` -- the question nothing had asked -- came out at **6.38e-03 against a
floor of 4.06e-03**, i.e. above it by a factor 1.6. Given the pin already
removed, widening `G` does move the lead current, by about 0.6 %. Marginal, and
on a bed that cannot yet be trusted for it: `G^R`'s block profile is flat out to
`R = 9` because the median modal range is 2.05 cells but the range at the
frequency carrying most of `G^<` is 636, so the bed's own tail is not resolved.
The number is reported as a gate reading, not as a physical result.

`eps_Toeplitz` is 3-9 % on `Sigma` and 1-11 % on `G` over the interior, falling
with distance. So a separation-only representation is a few-percent
approximation here rather than an exact one, which is the direction the
semiseparable structure predicts.

The shell decomposition separates cleanly at every distance: `R = 0..2` carried
by leg shells 0-2, `R = 5` by `(6+, 4-5)` at 23 %, `R >= 6` almost entirely by
`(6+, 6+)`. That is the vertex-near / propagation-tail split the proposal asks
for, and it is what the band sweep cannot produce.

### E3/E4, the Keldysh rank -- **the second gate passes**

Eleven frequencies spread over the band, interior anchor, span 14 blocks. The
source arms reproduce the frozen `G^<` to `1.2e-15`, so the decomposition is
exact and a rank can be attributed.

Median numerical rank over frequency, at four tolerances (Hankel cap 7):

| object | 1e-2 | 1e-3 | 1e-4 | 1e-6 |
|---|---|---|---|---|
| `G^R` | 2 | 3 | 4 | 7 (cap) |
| `G^<` | 4 | 5 | 7 (cap) | 7 (cap) |
| `G^>` | 4 | 5 | 7 (cap) | 7 (cap) |
| `Y = G^R L` | 5 | 7 (cap) | 7 (cap) | 7 (cap) |

Three readings.

**The Keldysh rank is bounded, not device-scaling.** Four to five exponentials
at a practical tolerance on a 20-cell device. The proposal's Sec. 39.2 stop
condition -- "Keldysh rank scales like the full device" -- is not met.

**It costs about twice the retarded rank**, 4 against 2 at 1e-2 and 5 against 3
at 1e-3. That is the pre-registered prediction from the source-resolved
derivation: two contact families, each contributing the retarded set and its
reciprocal partner.

**The positivity factor `Y` is not lower rank than what it factorises** -- 5
against 4 at 1e-2, and saturating earlier. So the factorised formulation of the
proposal's Sec. 9 buys positivity by construction and does not buy rank, which
was the other thing it was hoped for.

A rank at the cap is a lower bound and is reported as one: a Hankel matrix
cannot express more than its own size, and a span of 14 blocks gives a cap of 7.
Resolving the 1e-4 column needs a longer device.

## 10. The analytic contraction is exact where it is supposed to be

Eq. (47), verified against the ring it would replace, on a bed whose legs are
exactly modal (the gapped chain's bulk `G(n) = G(0) lambda^n` is rank one in the
Bloch factor, so any disagreement is the algebra and not the representation):

| `R` | 0 | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|---|
| rel. err | 6.7e-01 | 7.2e-02 | 3.7e-16 | 6.7e-16 | 3.2e-16 | 1.5e-16 |

Roundoff from `R = 2p` outwards, and deliberately wrong below it. The negative
control matters: a formula that happened to work everywhere would mean the
validity window `R >= R0 + 2p` was not what was being tested. Below it an
internal leg is asked for a NEGATIVE separation, where `lambda^{-n}` is the
growing partner and the modal form is not the Green function.

Three things in Eq. (47) are easy to get wrong and none of them fails loudly.
The proposal's `a,b,c,d` are CELL indices while `bubble.py`'s are DOF indices,
and they collide. The left vertex is conjugated (`se_finite.py`) and the modal
factors are not conjugated with it -- a no-op at Gamma with a real FC3 and
wrong in `se_q`. And `zeta^R` stays inside the frequency integral. That last one
decides the implementation: `analytic_tail` drives the ORDINARY kernel with
rank-one modal legs rather than reimplementing the contraction, so the frequency
convolution is the same FFT and a disagreement is unambiguous.

Pairs are screened by FC3-weighted tail amplitude (Eq. 50) and not by
`|lambda|`, because a mode near the unit circle that the vertex barely projects
onto is still irrelevant. `|zeta| >= 1` has no geometric sum and returns `inf`
unless given a device length, rather than a plausible negative number.

**This verifies the algebra. It does not demonstrate a saving** -- see the cost
argument in Sec. 1, which says the analytic route is 0.6x on Si and 0.1x on CNT
at full rank, and worse if the Keldysh legs need the doubled exponent set that
Sec. 9 measured.

## 11. The observable comparison, and criterion 5

Seven arms on the same frozen 20-cell chain, one dense Dyson/Keldysh re-solve
each, all relative to the untruncated reference D.

| arm | what | `eps(J_L)` | `|Sigma|` discarded |
|---|---|---|---|
| A | production pin, band 3 | 6.31e-03 | -- |
| B | pin, wide `G` | 6.31e-03 | -- |
| C | no pin, band 3 | 6.38e-03 | -- |
| **E** | **modal legs beyond band 3, direct fit** | **2.43e-01** | -- |
| **F** | **congruence: modal `G^R`, `G^<` rebuilt** | **3.27e-02** | -- |
| **R2** | **reblock, 2 cells/block** | **2.74e-03** | 58.8 % |
| R3 | reblock, 3 cells/block | 1.18e-02 | 47.4 % |
| R4 | reblock, 4 cells/block | 1.12e-02 | 37.8 % |

Two readings, and they point the same way.

**Reblocking at two cells per block is more accurate than the production pin**,
2.74e-03 against 6.31e-03, while discarding 58.8 % of `|Sigma|` by weight. The
discarded weight is not the error: a tridiagonal restriction at a coarser
blocking throws away more of the matrix and keeps more of the current, because
what it keeps is what the current is made of.

**The modal decompression is 38x worse than the pin it would replace.** Supplying
every leg block beyond band 3 from an exponential fit instead of from storage
moves the lead current by 24 %.

So the proposal's fifth go/no-go criterion -- "a reduced modal-pair
representation is cheaper than reblocking or direct wider-band recursion at the
same accuracy" -- is not met on this bed. It is not at the same accuracy. And
this is the criterion the tree already expected to fail:
`spatial_truncation_derivation.md` measured the discarded weight moving five
points with the range of `G` and thirty with the blocking, and concluded "the
modal route addresses the smaller term". That was an argument from block
weights; this is the same conclusion at the level of a current, with the modal
machinery actually built.

### The congruence route, measured

Arm E is the DIRECT fit of `G^{<,>}` -- route A of the proposal's Sec. 8, the
one it itself calls least safe. Route B continues `G^R` modally instead and
rebuilds `G~^{<,>} = G~^R Sigma_tot^{<,>} G~^A` with `G~^A` literally the
conjugate transpose, so positivity is a congruence rather than a hope. Adding it
as arm F:

| arm | `eps(J_L)` | negative spectral weight of `i G^<` |
|---|---|---|
| E, direct fit of `G^<` | 2.43e-01 | 0.090 |
| **F, congruence via modal `G^R`** | **3.27e-02** | **0.071** |
| (exact) | -- | 0.067 |
| R2, reblock 2 cells/block | 2.74e-03 | -- |

**The congruence route is 7.4x better than the direct fit and lands on the exact
positivity**, which is both of the proposal's Sec. 9 claims confirmed. Its error
is also flat in the fit tolerance -- 0.69, 0.71, 0.72 on `G^<` at
`eps = 1e-2, 1e-3, 1e-4` -- where the direct fit runs away by eight orders,
0.24, 1.75, 7.7e+07. A congruence of a bounded source cannot blow up.

**It does not overturn the verdict.** Arm F is still 5x worse than the
production pin and 12x worse than reblocking at two cells per block.

And the reason is measurable rather than a matter of tuning: **26-30 % of the
fitted residue weight sits in exponents with `|xi| > 1`.** That branch is
physical -- in a two-terminal device it is the wave from the FAR contact, which
the scalar prototype of Sec. 7 already showed -- but an outward continuation
from an interior anchor cannot carry it, because extrapolating a growing
exponent away from the anchor diverges (keeping it puts the far blocks out by a
factor 40 rather than by a few percent). Dropping it is required, and it costs a
quarter of the amplitude. No rank and no tolerance recovers that; it is a
property of continuing a one-sided sequence in a two-terminal device, and the
repair is the proposal's own "explicit boundary + modal interior" architecture
rather than a better fit.

Two limits remain. The bed is a 1-DOF chain with a random vertex, so the
percentages are the bed's. And one fit per frequency is reused at every cell
pair, which is the translation-invariance assumption `eps_Toeplitz` puts at
8-11 %.

### The ordering survives a real device, on a state that does not

The same eight arms on the reblocked Si film -- 12 blocks of 18 DOF, 36
primitive cells, the production FC3, `retarded = "half"`, `g_cutoff = 3` -- whose
frozen state DIVERGED (`resid = 1.0`, lead balance 1.0, `J_L = -3.0e-08` against
`J_R = +3.6e-08`, i.e. both leads emitting). Every absolute number there is
meaningless and none is quoted. What is legible is the ordering, because the
arms differ only in how `Sigma` is represented on one fixed `(G, Sigma)` pair,
which is a well-posed question whether or not that pair is a fixed point:

| bed | reblock | congruence | no pin | pin | direct fit |
|---|---|---|---|---|---|
| chain (converged) | 2.7e-03 | 3.3e-02 | 6.4e-03 | 6.3e-03 | 2.4e-01 |
| Si film (diverged) | 1.4e-01 | 6.4e-01 | 1.0e+00 | 2.0e+00 | 1.0e+01 |

Reblocking is the most accurate representation on both and the direct modal fit
is the worst on both, by an order of magnitude at each end. That is
corroboration and not evidence: a diverged state can order representations
correctly by accident. It is recorded because the alternative -- quoting nothing
from the only real device that ran -- would hide a consistency that does exist.

On that bed the modal continuation is in any case unusable on its own terms:
half the frequencies refuse the fit outright (241 accepted, 240 refused) and the
median far-block error among those accepted is 1.0, i.e. 100 %.

## 12. Open

- E1's `C -> D` on a bed whose own tail is resolved (`xi` of a few cells across
  the band, not 600 at the band bottom), and on a real device.
- The exponent identity: whether the recovered exponents are the advanced
  conjugates along `J` and the retarded roots along `I`. Not yet answered,
  because the operator's bands are ill-defined on this bed -- `Sigma^R` carries
  only 45 % of its weight within `|I-J| <= 1` and 90 % only by `|I-J| <= 8`, so
  there is no low-order pencil to compare against.
- The decompressor and the analytic contraction, both gated on the E6 cost
  argument in Sec. 1.
- Whether any real device admits a frozen state at all under the reference
  kernel; see Sec. 4. The CNT bed does not converge under `retarded="fft"` nor
  under production's `"half"` with the production cutoffs, and the Si film does
  not converge at one cell per block under any arm tried.
- Whether the "explicit boundary + modal interior" architecture recovers the
  30 % of weight the one-sided continuation has to drop. That is the only
  remaining construction that could change Sec. 11, and it is a different
  experiment rather than a better fit.
