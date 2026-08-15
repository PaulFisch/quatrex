# Observations, 2026-08-15

Two threads ran on this day: the pole/QNM sector was taken to a verdict, and
the spatial half of the method proposal was investigated and closed. This is
the consolidated record; the detail lives in the topical documents named
throughout.

It also carries an unusually long correction list. That is recorded rather than
tidied away, because the failures share one shape and the shape is instructive.

---

## 1. The pole sector has a verdict

### 1.1 The frozen Si census (jobs 4479538 / 4489601)

The measurement the method was waiting on, and the one
`pole_sector_observations.md` Sec. 1.8 said would settle it. `sichk_base`,
`ne = 141`, `wmax = 35` with `fft`, eta = 0, 81 q, 18 candidates each. The
pole-off arm converged to `9.2597e-04`, reproducing the `siladder` base arm, so
the bed is the same one.

`extraction_only` hands the ring an empty pole set, which makes every stage
bit-identical to the pole-free baseline. That is now verified ON DEVICE and not
only in unit tests: the census and baseline stages agree to every printed digit
at the same iteration, residual `9.6952e-01` and lead balance `1.2363e-05`.

| | cold (iteration 1) | warm (converged) |
|---|---|---|
| median gamma [THz] | 0.0361 | 0.161 |
| median `E_leg^max` | 1.22 | 0.0314 |
| q whose MEDIAN line is unrepresentable | 46/81 | **0/81** |
| q with ANY unrepresentable line | 81/81 | **66/81** |
| isolated candidates | 638 | 374 |
| median gamma/separation | 0.751 | 1.52 |
| continuation failures (upper half plane) | 68 | 36 |

**The bulk broadens away and a tail does not.** Self-consistency widens the
lines 4.5x and the grid then carries the median one to 3 %, but two thirds of
the q still hold at least one line it cannot represent. Those survivors are
also less isolated, not more.

Against the method proposal's decision tree that is its third branch -- narrow
but strongly mixed, wanting a cluster representation -- and not the simple-pole
one the sector builds. Full detail: `pole_sector_observations.md` Sec. 13.

### 1.2 The shipped gate would have given the opposite answer

Read `accepted` alone and the population survived: 584 -> 504, under-resolved
95.5 % -> 95.1 %. That is the crude `q_omega = gamma/(2h)` rule against `q_in`,
and it disagrees with the exact line-weight test about a physics conclusion
rather than about a threshold. `leg_weight_tol` had defaulted to 0 since the
exact gate was written.

Changed to 0.05, chosen from the census rather than by taste: over its 81 q,
a tolerance of 0.01 promotes the median mode in 76 of them, 0.05 in 32, 0.10 in
18. The census found the bulk carried to 3 % and a tail of one to four modes
per q that is not, so 0.05 promotes the tail and leaves the median mode alone.
`leg_weight_tol = 0` still restores the old rule.

Consequence worth knowing: `screen()` treats the two gates as ALTERNATIVES, so
under the shipped default `samples_per_halfwidth`, `q_in` and `q_out` are never
consulted. Marked in place.

### 1.3 Defects found and fixed

**The census was written by every rank.** Job 4479538 came back unparseable --
rows interleaved mid-word -- because every rank assembles the same operator and
solves the same poles, and `PoleSector._report_census` had no rank guard. The
tell was in the log: `_census_over_q` DOES guard the `q (...)` header, so the
stage carried 14 headers against 179 bodies. Fixed; the `base` arm was
unaffected and its Sigma was reused, so the repeat cost 0.25 nh.

**`window()` searched a different range on every rank.** It took its upper
bound from `self.freqs[-1]`, the rank-local slice, while
`_pole_frequency_context` states that every rank solves the same poles. Now
taken from `global_freqs`; serial runs unaffected.

**`edge_factor` had never fired.** `PoleSector` accepts `band_edges`, `screen`
and `trust_radius` both use it, and neither solver call site ever passed one --
so the proposal's "do not force band-edge continua into isolated poles" was
unenforced for the whole campaign. `lead_band_edges` now derives them from the
lead dispersion, behind `band_edges = "none" | "lead"`, default off because
switching it on changes which poles are promoted.

Two further defects surfaced from WRITING that test rather than running it: the
new function was imported in the wrong scope, and `_census_over_q`'s blanket
`except` turned the resulting `AttributeError` into "census failed" on all 15 q
-- a survey reporting that it visited nothing. It now re-raises `AttributeError`,
`NameError` and `ImportError`, which are never "this q is hard".

### 1.4 Also added

Named linewidths (`gamma_hwhm`, `Gamma_fwhm`, `is_passive`) pinned against
physical requirements rather than prose; the finite-support line-weight gate of
the audit's Eq. (14); anharmonic `dz/dlambda` verified both by finite difference
and by the physical check that on a contact-free bed it returns the entire half
width; five new census columns; and the two "mandatory" calibration beds. The
overlap sweep calibrates the hard-coded 0.5 simple/cluster threshold: the
coherent cluster tracks the exact resolvent to 8e-03 at `chi = 0.085` and still
8e-03 at `chi = 15.6`, so overlap does not break the representation, while
scalar occupations degrade 2.7e-02 -> 29 % and saturate.

---

## 2. The spatial half

### 2.1 What was built

`src/quatrex/phonon/spatial_modes.py` -- device complex bands from the pencil
the lead OBC already solves, with the SCBA substitution
`H_00 -> H_00 + Sigma^R_00`. Undressed, an in-band mode sits on the unit circle
and its range `xi = -1/ln|lambda|` is infinite; dressed, the reciprocal pair
splits and it acquires a finite one. On a chain at `Sigma = -i Gamma` that is
554 cells at `Gamma = 0.05`, 55.4 at 0.5, 5.6 at 5.0.

`xi = v_g/gamma` to 1e-4 in weak damping, which is the bridge between the two
halves: the pole census already measures `gamma` per mode, so a required
spatial range follows from a measurement already taken.

Measured on the stored dynamical matrices (`spatial_band_range.md`):

| bed | branch-max abs(v_g) med [cells*THz] | range at the bed's gamma [cells] |
|---|---|---|
| Si, 81 q, 486 branches | 0.967 | 3.05 to 28.8, median 6.05 |
| CNT (3,3), 36 branches | 5.16 | 2.6 median, 10.7 at the narrowest mode |

And the reconstruction of the proposal's Eq. (158): fitting
`G(n) = V diag(lambda^n) C` from `n = 1, 2` alone predicts every block out to
`n = 12` at roundoff on real CNT (36 DOF) and Si (6 DOF) cells.

### 2.2 The rank and the fit anchor are one choice

Truncating the mode set looked free and is not. Modes with `|lambda| = 1e-3`
contribute `1e-9` by three cells, yet dropping them costs `1e-4` -- the fit was
anchored where they are still in the data, so their weight lands on the
survivors. On CNT at rank 22 of 36, the error at `n = 5` is 1.2e-02 anchored at
`n = 1,2` and 2.1e-07 anchored at `n = 5,6`.

It cuts both ways: a fit anchored far out cannot determine a fast mode's
coefficient at all, so SHORT-range blocks degrade even at full rank. The anchor
selects the window of distances the representation is valid on, and the rank
follows from it. The pole sector reached the same conclusion for its own local
model with `_fit_anchor`.

### 2.3 Reweighting the mask cannot replace a modal sector

Two bounds, both structural.

At the output the band is pinned at `|I-J| <= 1`, so the mask is the
tridiagonal Toeplitz `[w_1, 1, w_1]` with symbol `1 + 2 w_1 cos(theta)`,
non-negative only for `w_1 <= 1/2`. A weighting faithful to a range `xi` has
`w_1 = exp(-1/xi)`, so PSD-ness demands `xi <= 1/ln 2 = 1.44 cells` -- against
every range measured. This derives the existing empirical result rather than
restating it: Bartlett has `w_1 = b/(b+1) <= 1/2` only at `b = 1`.

On the legs, a truncated geometric weight is PSD only once the band exceeds the
range, by a factor that itself grows with the range (1.2x at `lambda = 0.3`,
3.0x at 0.91) -- the regime in which no truncation was needed.

---

## 3. The three spatial truncations

Derived in `spatial_truncation_derivation.md`. With vertex reach `p` and leg
band `b`,

    supp(Sigma) = { |I - J| <= 2p + b },

confirmed exactly: leg band 0,1,2,3 gives reach 2,3,4,5. So `Sigma` is NOT
tridiagonal and its reach grows with the leg band.

| # | truncation | status |
|---|---|---|
| 1 | `Phi` to the nearest-neighbour shell | **exact** -- the FC3 cutoff (4 Ang on MoS2) is shorter than the transport cell (12.294 Ang), so nothing beyond the shell exists |
| 2 | legs to `\|K-K'\| <= b` (`sse_g_band`) | **exact** for the retained output once `b >= 3`, which is the default and the cap |
| 3 | output pinned to `\|I-J\| <= 1` | **the live one** |

Truncation 3 discards about 30 % of the `Sigma` weight on a device long enough
to have settled (10.5 % at seven cells, 29.3 % at ten, 30.3 % at fourteen --
the short bed is not a conservative proxy). Its dependence on the
Green-function range is secondary: five points between ranges of 2 and 20 cells
against thirty from the device length.

The lever is the blocking, since `2p + b` is in CELLS. On twelve cells:

| cells per block | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| discarded | 32.1 % | 5.4 % | 2.5 % | 0.30 % |

`reblock_device.py` already exists, and its docstring already states the
mechanism.

**But the pin is an accuracy defect, not a stability one.** That was tested
before this session: `bubble_positivity.md` Sec. 6.7 ran the same MoS2 film at
six blocks (maximal truncation) and at two (no mask at all), and both diverged
at the same iteration with the untruncated run carrying MORE gain. Re-blocking
is worth doing for accuracy; it does not fix a divergence.

---

## 4. Corrections

Five claims of mine were withdrawn within the day, plus two stale records
fixed. They are listed because the pattern is the useful part.

1. **"The boxcar leg band cuts the modes that carry heat"** -- 32 % at `b = 1`
   and 7 % at 3. Measured over ALL `Sigma` blocks; the 7 % lived entirely in
   `|I-J| >= 2` blocks production never outputs.
2. **"`g_band = 3` is exact, so no spatial truncation is live"** -- exact for
   the legs, but the output pin is a separate hard-coded truncation and I
   stopped before reaching it.
3. **"The output pin is what the long-range modal machinery repairs"** -- it is
   range-insensitive; the near tail is fed by the DIAGONAL of `G` through the
   vertex's reach.
4. **"The FC3 guard is dead instrumentation"** -- the builder does not truncate
   (`vertex_cutoff` defaults to `None`); the guard reads zero because zero is
   correct.
5. **"The pin explains why si4x1 diverges and si4x2 converges"** -- already
   tested and set aside in `bubble_positivity.md`, whose Sec. 8 attributes the
   Si pair to the partition itself and not to the truncation it implies.

Also: the seven-cell figure for the pin (10.8 %) was finite-size limited and is
superseded by ~30 %; and a stored note read "MoS2 bulk FCP has exactly zero fc3
across the vdW gap", which describes the REJECTED ARDR fit -- production uses
least squares, which keeps those couplings.

**The shape.** Every one was a number read without its mechanism: an error
percentage without asking which blocks it lived in, a zero without asking what
produced it, a convergence difference without checking whether it had already
been explained. The algebra that settles the first three is four lines; the
fourth needed one line of a function signature; the fifth needed reading a
document already in the tree. None needed a new measurement.

The practical rule this leaves: before quoting a fraction, state which set it
is a fraction OF, and before proposing a run, grep the docs for the experiment.

---

## 5. Where things stand

**Closed.** The pole sector has its verdict and its calibration; the spatial
leg is built as far as diagnostics and will not proceed, because the ring does
not ask the question it answers.

**CORRECTED 2026-08-15, after this document was first written.** The paragraph
here originally read "the MoS2 film has no eta = 0 fixed point in any tested
corner ... what remains is H4". **Both halves are wrong**, and my own memory
file had the right answer while I wrote the negation of it.

MoS2 HAS converged eta = 0 fixed points:

| run | job | device | `interaction_cutoff` | outcome |
|---|---|---|---|---|
| `mos2L2conv` | 4384190 | 2 cells, nf 4001 | 30 A | **converged, 29 iterations**, residual 9.34e-04, lead balance 1.52e-04 |
| `mos2L4conv` | 4384165 | 4 cells, nf 4001 | 48 A | **converged, 30 iterations**, residual 9.04e-04, lead balance 2.63e-04 |
| `mos2L6n4scba` | 4384160 | 6 cells | 75 A | OOM at 97.8 GB/GPU, 0 iterations -- memory-blocked, NOT divergent |

The cause is **H6**, the `interaction_cutoff` box mask on the storage pattern --
a second Hadamard mask at the orbital level, distinct from the block-band H2.
`bubble_positivity.md` Sec. 6.8-6.10 establishes it: a single-variable A/B
(jobs 4383378/4383393) where only the cutoff moves gives monotone convergence
with gain fraction exactly 0.00000 at 30 A and 3.7e+07 at 10 A, and the cutoff
ladder shows the criterion is mask PSD-ness rather than retained weight -- the
21 A rung has **98.6 % fill and still diverges**, 22 A is dense and converges.

Why I got it wrong: Sec. 6.7 of that document ends "what remains is H4", and
Secs. 6.8-6.10 -- which supersede it -- were written a day later without
revising that sentence. I read 6.7, stopped, and propagated its conclusion. The
thesis chapter had already been corrected (`75_mos2.tex` opens its
corresponding subsection with "That reading is wrong").

So H4 is not the standing hypothesis. What it could still own is one datum,
Sec. 6.10b: at a COMPLETE 30 A support, merely reweighting it with the
triangular taper turns a converging run into a divergence at 1.76e6 while
restoring positivity perfectly. That is circumstantial and is the only
H4-flavoured evidence in the tree.

**Half-closed, later the same day.** The absolute prefactor question splits in
two, and the vertex half is now settled exactly. `vertex_normalisation.md`:
the code's mass-weighted FC3 is phono3py's times exactly `CONVERSION_FC3_THZ`,
worst deviation 1.1e-13 over every triplet and band on the checked-in Si
primitive reap, and its Fourier fold is phono3py's to 0.0e+00 at commensurate q
once the documented A -> B gauge is applied. F28 had put native within ~15 % with
`div4` and `x4` excluded by an order of magnitude; the element comparison says
the vertex is not close to phono3py's, it IS phono3py's.

What that leaves open is everything downstream of the vertex -- the bubble
prefactor, the Keldysh assembly, the grid -- which only an end-to-end number
tests. Bulk-Si kappa from the code's own SCBA in the diffusive limit against
phono3py RTA (~110 W/m/K on our own FC3) has still never been run, and the
Si-film over-scatter against Guo (45 % against 10 %) still has no explanation.
It can no longer be blamed on the vertex.

**New, and unmeasured.** H and the vertex do not use the same periodic image.
H comes from phonopy's shortest vectors, tie-averaged; the vertex fold uses one
wrapped cell index, no basis offset, no average. Same gauge, different image, and
they part company at q not commensurate with the FC supercell -- 25-46 % of
`||D_B||` on the 2x2x2 Si bed. Whether it matters on the production film
(5x5x5, `nk = 9`) is an FC-weighted question that has not been asked; the pair
counts alone (22.8 %) are an upper bound, not an error.

**Cheap and unmeasured.** The output pin's ~30 % is a 1-DOF chain with a random
vertex; the same split on the real `Phi` blocks would turn it into a production
number, offline.

daint stands at 274.77/300 nh. The two census jobs were the only compute spent.
