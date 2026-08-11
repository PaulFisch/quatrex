# Pole-subtracted modal SCBA: what is implemented, and what it measures

Companion to `pole_subtracted_modal_scba.md`, which is the design note. This
one records the formulas **as built** and every number measured against them.
Where the two disagree, the design note is the older document and this one
says so explicitly.

Scope: the frequency half of the method (Parts 0-5 of the design note). The
spatial/modal half (Part II) is not started.

---

## 1. Analytic continuation of the scattering self-energy

### 1.1 The identity the method rests on

Production builds the retarded scattering self-energy on the real axis as

$$\Sigma_s^R(\omega) = \tfrac12\Delta(\omega) + \tfrac{i}{2}\mathcal{H}[\Delta](\omega),
\qquad \Delta = \Sigma^> - \Sigma^<$$

with `hilbert_transform` (`core/fft_utils.py`) using exact cell-integrated log
weights. The same cell-wise-constant model of `Δ` continued to **complex** `z`
with the complex logarithm is

$$F(z) = \frac{i}{2\pi}\sum_k \Delta_k\Big[\mathrm{Log}\big(z-\omega_k+\tfrac h2\big)
      - \mathrm{Log}\big(z-\omega_k-\tfrac h2\big)\Big]
   + \frac{i}{2\pi}\sum_k \bar\Delta_k\Big[\mathrm{Log}\big(z+\omega_k+\tfrac h2\big)
      - \mathrm{Log}\big(z+\omega_k-\tfrac h2\big)\Big]$$

with the bosonic partner `\bar\Delta_k = \Delta_k(-q)^*` and the `k = 0` term
dropped from the mirror sum. The pole cell contributes
`Log(i0 + h/2) - Log(i0 - h/2) = -i\pi`, i.e. `+\Delta_n/2`: the `½Δ` term is
**not** a separate addition, it falls out of the same formula.

> **Measured.** `max|F(ω+i0) − (½Δ + ½i·H[Δ])| / max|Σ^R| = 8.1e-14` against a
> faithful re-implementation of the production kernel. The jump across the cut
> is exactly `Δ` to 8 digits.

Order of the branches is load-bearing: `Log(z-ω_k+h/2) - Log(z-ω_k-h/2)`
reversed gives exactly `−1×` the answer.

### 1.2 The second sheet

$$\Sigma_s^{R,\mathrm{II}}(z) = F(z) + \Delta_{\rm an}(z)$$

`Δ_an` is a degree-`p` least-squares polynomial in `t = (ω − Re z)/h` over the
nearest `window` cells on each side, evaluated at `z`. Error
`O((|Im z|/h)^{p+1})`, controlled precisely because promoted poles satisfy
`γ ≪ h`. Negative-`Re z` probes are served by the mirrored branch.

### 1.3 Both terms are linear in Δ — which is what makes it distributable

The fit is `coeff = pinv(V) · src` with the Vandermonde `V` depending only on
the grid, so

$$\Delta_{\rm an}(z) = \big(\text{powers} \cdot \mathrm{pinv}(V)\big)\,\Delta$$

is a weight matrix exactly like `F`'s. `local_fit_weights` returns
`(w_pos, w_mir)` of shape `(P, K)` to be used with `contract_delta`.

> **Measured.** Weight form vs the direct implementation: `6.7e-16`, `1.8e-15`,
> `4.7e-15` for derivative orders 0, 1, 2.

Neither branch reindexes the frequency axis — the mirror is elementwise (a
conjugation plus `q → -q` on transverse axes) and the negative-frequency
reflection lives in the `z + ω_k` argument. So each rank contracts the columns
it owns and one sum-reduction completes the continuation.

> **Measured.** 4 MPI ranks vs serial: `1e-12`. Serial uses an identity
> reducer, so that path is bit-identical, not merely equivalent.

**Deviation from the design note.** The note prescribed contracting in the
`"nnz"` distribution state and `dtranspose`-ing to `"stack"`. That is
unnecessary and more expensive: it moves the whole buffer, where the
reduction moves only `(P, nnz)`, and it would require distributed root
finding. Every rank ends with the same operator and solves the same poles.

---

## 2. The nonlinear eigenvalue problem

### 2.1 The operator

$$M(z) = z^2 I - D - \Sigma_c^R(z) - \Sigma_s^R(z)$$

in THz², block-tridiagonal in every configuration because the Σ output band is
pinned at `|I−J| ≤ 1`. `η ≡ 0` throughout.

Contacts are held flat at the grid point nearest `Re z`. Passing the contact
array unreduced assembles `M` at every frequency at once — see §9.2.

### 2.2 Bordered Newton with the border eliminated

With `x₁ = M⁻¹(Mr)` and `x₂ = M⁻¹(M′r)`:

$$\delta z = \frac{c^\dagger r - 1 - c^\dagger x_1}{c^\dagger x_2},
\qquad \delta r = -x_1 - \delta z\, x_2$$

Two solves and one factorisation per step; no `(N+1)` system. `r` is
deliberately **not** renormalised inside the loop — the gauge `c†r = 1` is what
closes the bordered system, and rescaling breaks convergence.

> **Measured.** 3 iterations to `|M(z_α)| = 1.2e-14` on a 1-DOF bed.
> Quasiparticle seeding (`γ ≈ −Im(v†Δv)/(4Ω)`) converges 9/9 poles where a
> fixed γ guess converged 1/9.

### 2.3 Residues

Left vector `l = M(z)^{-H}c`, normalised so `l†M′(z)r = 1`, giving

$$R_\alpha = r_\alpha l_\alpha^\dagger, \qquad
G^R(\omega) \simeq \sum_\alpha \frac{R_\alpha}{\omega - z_\alpha}$$

> **Measured.** `l†M′r = 1.000000e+00`. The pole model against a direct
> inversion: ratio to the true `‖G^R‖` is 1.034, 1.012, 1.005 at
> `Re z + 0.05, 0.02, 0.01`. The residue is the true one; `G_PP`'s scale is
> right.

---

## 3. The Keldysh cluster

### 3.1 Two representations, and they are not interchangeable

**Congruence form** (`pole_keldysh_sparse`), manifestly PSD when `S` is:

$$G_{PP}(\omega)_{ij} = \sum_{\alpha\beta} u_{i\alpha}\,
   \frac{S_{\alpha\beta}(\omega)}{(\omega-z_\alpha)(\omega-z_\beta^*)}\,u^*_{j\beta}$$

**Partial-fraction form** (`pole_keldysh_pf_sparse`), which is what the
analytic sectors can represent:

$$G_{PP}(\omega)_{ij} = \sum_{\alpha\beta} u_{i\alpha}
   \left[\frac{c^a_{\alpha\beta}}{\omega-z_\alpha}
       + \frac{c^b_{\alpha\beta}}{\omega-z_\beta^*}\right] u^*_{j\beta}$$

$$c^a_{\alpha\beta} = \frac{S_{\alpha\beta}(z_\alpha)}{z_\alpha - z_\beta^*},
\qquad
c^b_{\alpha\beta} = -\frac{S_{\alpha\beta}(z_\beta^*)}{z_\alpha - z_\beta^*}$$

> **Measured.** The two agree to `7.0e-16` for a source constant in frequency
> and differ by `7.0e-3` for one with a 2 %/THz slope.

`G_{\rm reg} \equiv G - G_{PP}` is exact for **any** `G_PP`, so the sector sum
holds only if the leg SUBTRACTED from the ring and the leg the sectors PUT
BACK are literally the same object. Production now builds both in the
partial-fraction form.

> **Measured.** `G_PP(pf)` against the representation `leg_partial_fractions`
> produces: `0.0` — the same object, not a close one.

### 3.2 The source

`S(ω) = V†Σ_tot(ω)V`, projected on the stored pattern, plus both contact
corners projected separately (they live on dense corners, not the pattern).

Per-pole evaluation, not one frozen value per cluster: the residue at
`z_α` carries `S(z_α)`. Negative-frequency partners are served by the bosonic
mirror `S(−ω) = S(ω)^*` rather than by extrapolating a model across the band.

> **Measured.** On a cluster with poles at 8 and 11 THz and a frequency-varying
> source, the frozen value (read at the first pole) is **27 % off** on the
> second.

---

## 4. The bubble kernels

### 4.1 Elementary pole convolution

$$J(p,q;\omega) = \int\!\frac{d\omega'}{2\pi}
  \frac{1}{(\omega'-p)(\omega-\omega'-q)} =
  \begin{cases}
   -i/(\omega-p-q) & \operatorname{Im}p<0,\ \operatorname{Im}q<0\\
   +i/(\omega-p-q) & \operatorname{Im}p>0,\ \operatorname{Im}q>0\\
   0 & \text{mixed}
  \end{cases}$$

Only two of every four pole pairings survive.

### 4.2 Leg splitting

$$\frac{S_{\alpha\beta}}{(\omega-z_\alpha)(\omega-z_\beta^*)}
 = \frac{c}{\omega-z_\alpha} - \frac{c}{\omega-z_\beta^*},
 \qquad c = \frac{S_{\alpha\beta}}{z_\alpha-z_\beta^*}$$

With a polynomial source `S(ω) = Σ_m S^{(m)} t^m`, `t = (ω−Ω_c)/h`, the same
two poles survive with residues `S^{(m)} t_a^m / \mathrm{gap}` and
`−S^{(m)} t_b^m / \mathrm{gap}`, plus a polynomial quotient of degree `m−2`
which is **empty for m ≤ 1**.

> **Measured.** Order 0 reduces to the frozen expression exactly (`0.0`);
> order 1 reproduces the rational function to `2.2e-16`. Order ≥ 2 **refuses**
> rather than dropping the quotient — the dropped piece is the smooth part of
> the source, so the answer would look reasonable and be wrong.

### 4.3 The pole-pole sector

$$\Sigma_{SS}(\omega)_{\mu\mu'} = \frac{i\hbar}{2}
 \sum_{\alpha\beta\gamma\delta}\bar\Phi_{\mu,\alpha\beta}\,
 \bar\Phi^*_{\mu',\gamma\delta}\; C_{\alpha\delta\beta\gamma}(\omega)$$

evaluated only on the stored pattern, so `|I−J| ≤ 1` is automatic. The
retarded partner is free: `Δ_SS` is a sum of simple poles, so

$$\Sigma^R_{SS}(\omega) = \sum_{j:\ \operatorname{Im}p_j<0}\frac{c_j}{\omega-p_j}$$

— keep the lower-half-plane poles. Manifestly causal, and it must **not** pass
through the numerical Hilbert transform.

Because `core/scba.py` already adds `½(Σ^< − Σ^>)` globally to the stored
retarded buffer, the injected quantity is the Kramers-Kronig half alone:

$$\Sigma^R_{\rm inject} = \Sigma^R_{SS} - \tfrac12\big(\Sigma^>_{SS}-\Sigma^<_{SS}\big)$$

### 4.4 The mixed sectors

Substituting `u = ω − ω'` turns the convolution into a resolvent of the
background:

$$M(\omega) = \int\!\frac{d\omega'}{2\pi}\frac{c}{\omega'-p}R(\omega-\omega')
 = c\int\!\frac{du}{2\pi}\frac{R(u)}{z-u},\qquad z=\omega-p$$

For a cell-wise-constant `R` this is a matmul with

$$W_{ik} = \frac{1}{2\pi}\Big[\mathrm{Log}\big(z_i-\omega_k+\tfrac h2\big)
                      - \mathrm{Log}\big(z_i-\omega_k-\tfrac h2\big)\Big]$$

so the narrow denominator is integrated, never sampled.

**The integral runs over the whole axis** while the solver holds `G` only for
`ω ≥ 0`. The negative half follows from `R(−ω) = R(ω)^*` but must be supplied
explicitly — the cell kernel integrates exactly the cells it is given.

> **Measured.** One-sided integral wrong by **28 %**; mirrored, `2.5e-4`.

**Blocked contraction.** The pattern-level form is `O(nnz_out × nnz_in)` and
refuses above 4096 entries, which no real device is under. The contraction is a
matrix triple product

$$\Sigma[I,J] = \sum_{a,b} B_L[I,a]\,M[a,b]\,B_R[J,b]^{\!\top}$$

with every factor block-banded (`M` on `G`'s block-tridiagonal pattern,
`B_{L,R}` inheriting `|I−a| ≤ 1` from the cubic vertex), so each output block
costs a handful of `b×b` GEMMs and the dense `(n_dof, n_dof)` vertex — 1.5 GB
per pole at production size — is never formed.

> **Measured.** Blocked vs pattern-level: `1e-12`. Production kernel vs an
> explicit ring at `γ/h = 40`: `3.6e-3`, magnitudes within 0.2 %.

`Σ_SR` and `Σ_RS` are built from independent `leg`/`conjugate` pairs.

> **Measured.** On a leg-exchange-symmetric vertex — what hiphive's
> `symmetrize=True` produces, hence every real device — they are **exactly
> equal** (`1.3e-15`). They differ only on the random, unsymmetrised Φ that
> unit-test beds use.

---

## 5. Why the decomposition is positive by construction

With `A = G_PP`, `B = G_reg`, and the mixed sectors scaled by `λ`:

$$Q(\lambda) = B(A,A) + B(B,B) + \lambda\big[B(A,B)+B(B,A)\big]
 = \lambda\,B(A{+}B,A{+}B) + (1-\lambda)\big[B(A,A)+B(B,B)\big]$$

`B(A+B,A+B) = B(G,G)` is PSD and `B(A,A) + B(B,B)` is a sum of PSD objects, so
`Q(λ)` is a **convex combination of two PSD objects — PSD for every λ ∈ [0,1]**.

> **Measured.** `+1.05e-06, +1.66e-06, +2.23e-06, +2.77e-06, +3.28e-06` at
> `λ = 0, 0.25, 0.5, 0.75, 1`, i.e. PSD and improving with λ. Under grid
> refinement the hybrid stays `+3.3e-06` from `dw/γ = 6.7` to `0.42`.

So neither the decomposition nor a sector-quadrature mismatch can produce a
positivity violation. Any violation observed on a device is a bug elsewhere.

---

## 6. The prize

Discrete convolution of two `γ = 0.004` Lorentzians against the exact identity
`L_{Ω₁,γ₁} * L_{Ω₂,γ₂} = L_{Ω₁+Ω₂,γ₁+γ₂}`:

| n_freq | dw/γ | peak rel. err | weight rel. err |
|---|---|---|---|
| 201 | 100 | 1.0 (misses the line) | 1.0 |
| 401 | 50 | **30.8** | **253** |
| 1601 | 12.5 | 6.96 | 15.5 |
| 6401 | 3.1 | 1.03 | 0.71 |
| 25601 | 0.8 | 5.8e-3 | 1.2e-3 |
| 102401 | 0.2 | 3.4e-12 | 1.3e-4 |

An unresolved narrow leg does not merely lose weight, it *gains* up to 250×
depending on bin alignment.

Mixed-sector route comparison against an exact residue reference:

| γ/h | grid | cells | moments |
|---|---|---|---|
| 20 | 5.8e-4 | 1.9e-3 | 1.8e-3 |
| 1 | 3.6e-3 | 1.3e-3 | 1.2e-3 |
| 0.2 | **0.74** | 2.7e-3 | 1.2e-3 |
| 0.04 | **6.5** | 4.6e-3 | 1.2e-3 |
| 0.008 | **36** | 5.1e-3 | 1.2e-3 |

The design note's recommended `grid` route is **not viable** — it collapses
exactly where the sector is needed. `cells` is production; `moments` is the
measured refinement, flat in γ. `rational` raises `NotImplementedError`.

This table is also why `Σ_SR + Σ_RS = compute_linearized(dG = G_PP)`, which the
design note prescribes, was **not** taken: `compute_linearized` is a grid
convolution, so on promoted poles (`γ ≪ h` by construction) it would be
conserving and wrong.

---

## 7. Placement in the SSE, and why it differs by sector

| contribution | added | reason |
|---|---|---|
| leg subtraction `G − G_PP` | before the DC mask and the aux bridge | interpolating a narrow Lorentzian through the coarse bridge is the error being removed |
| `Σ_SS^{<,>}` | **after** `_restrict_from_aux` | never crosses the energy-weighted adjoint transfer, so it cannot leak energy through it |
| `Σ_SS^R` | separately, closed form | routing it through the discrete Hilbert transform would restore the grid dependence the sector removes |
| `Σ_SR + Σ_RS` | to the raw conv-grid output, before `Δ` is formed | one narrow factor against a smooth background; the numerical transform is the right tool |

The mixed background leg is masked **exactly as the ring masks its own legs**
(`|ω| < max(1e-6, sse_low_freq_mask_thz)`). The ω = 0 bin carries the
near-singular acoustic peak (`|G^>(0)| ≫ neighbours`) and the ring excludes it;
feeding the mixed convolution an unmasked leg makes the two sectors integrate
different data.

---

## 8. Gates

| gate | formula | status |
|---|---|---|
| zero-pole identity | empty window ⇒ byte-identical | **PASS**, 13 observables, `rtol = atol = 0`, GPU |
| sector sum | `B(G,G) − [SS+SR+RS+RR]` | **8.7e-16**, weights SS 0.96 / SR 0.077 / RS 0.077 / RR 0.11 |
| Keldysh identity | `ε_KI = ‖Σ^R − Σ^A − (Σ^< − Σ^>)‖_F / ‖Σ^< − Σ^>‖_F` | implemented, roundoff on a correct assembly |
| positivity | `−iΣ^{<,>} ⪰ 0` and `−iG^{<,>} ⪰ 0` | implemented behind `psd_check` |
| energy balance | `\|P_in − P_out\|/\|P_in\|`, `QX_BBCHECK=1` only | live |

Convention traps, both encountered:

* **Both** Keldysh components carry `sign = −1`. This solver uses
  `σ^{<,>} = +i n(+1)Γ`, so `−iΣ^<` and `−iΣ^>` are both PSD. The textbook
  `+iG^> ⪰ 0` does not apply.
* `ε_KI` does **not** decompose into Hermitian and anti-Hermitian projections
  — `Σ^R − Σ^A` is anti-Hermitian by construction, so both projections are
  identically zero. The useful auxiliary numbers are built from the inputs:
  `eps_delta_skew` tests `Σ^< − Σ^>` and `eps_kk_hermitian` tests the recovered
  KK part. A double-counted retarded half is a MAGNITUDE error and is invisible
  to the symmetry checks.
* `P_in = P_out` is a scalar trace identity. A spatially wrong Σ satisfies it.
  On this campaign the residual and the heat profile disagreed three times.
  **Read the heat profile first.**

---

## 9. Bugs found, with the mechanism

### 9.1 Hysteresis never engaged (the root cause of non-convergence)

The promoted set was stored as `{id(sol) for sol in accepted}` and tested with
`id(sol) in self._promoted`. `bordered_newton` builds a fresh `PoleSolution`
every SCBA iteration, so those CPython object ids can never match — and CPython
recycles ids of freed objects, so a match could occur arbitrarily.
`was_promoted` was therefore always `False` and every pole was screened at the
strict `q_in = 1.0` instead of the lenient `q_out = 2.0` it earns once in the
sector.

The `q_in`/`q_out` gap **is** the hysteresis, and its config docstring states
the consequence: *"a mode that changes sector every iteration makes the
fixed-point map discontinuous"*.

> **Measured**, 80 SCBA iterations: membership went 3 poles (it 1-10) → **0**
> (11-13) → 1 (14-23) → **0** (24-25). Each transition switches the entire
> pole-sector contribution on or off. The balance residual oscillates
> `5e-3 → 3.6e-2 → 1e-4` without settling, while the pole-free baseline sits at
> `1e-7` throughout. **Nothing diverges** — it is a limit cycle.

Identity across iterations is now carried by **position**: a pole moves by at
most the predictor shift between iterates, so anything within a cluster width
of a previously promoted pole is the same mode.

### 9.2 `M(z)` assembled at every frequency at once

`set_operator_context` was handed `obc_blocks.retarded[0]`, which is
`(n_freq, b, b)`. `M(z)` is one matrix, so this assembled it at 201 frequencies
simultaneously. Contacts are now sampled at the grid point nearest `Re z` —
the flat-contact approximation the docstring already claimed.

### 9.3 `btd_matvec` output allocation

Allocated with `zeros_like(x)`, so blocks carrying a probe axis the vector
lacks made the in-place `+=` raise instead of broadcast. Not a GPU bug — it
reproduces on numpy. Every unit test fed unstacked blocks, so only the
production assembly, which carries the axis, ever exercised it.

### 9.4 Mixed background leg unmasked

Found by **bisection**, after four mechanism hypotheses had each been proposed
and refuted by measurement. Scaling the injected `Σ_SR + Σ_RS` by λ:

| λ | `Σ^>` worst, before | after | factor |
|---|---|---|---|
| 0.25 | −3.805e-02 | −7.88e-04 | 48 |
| 0.5 | −7.678e-02 | −2.44e-03 | 31 |
| 1.0 | −1.506e-01 | −9.24e-03 | 16 |

Strictly **linear** in λ before the fix — no cancellation involved, i.e. a term
that should not be there. Afterwards the scaling is **quadratic** (predicted
5.8e-4 and 2.3e-3 from the λ=1 value, measured 7.9e-4 and 2.4e-3), which a sign
or factor error cannot produce.

### 9.5 The positivity gate itself had a flipped sign

It reported `sigma_greater worst = −1.000e+00` — uniformly negative — on the
**pole-free baseline**. See §8. Every `greater` reading before the fix was
meaningless. Lesson: validate a diagnostic against a known-good run before
trusting it to localise a defect.

### Hypotheses proposed and refuted, for the record

| hypothesis | refuted by |
|---|---|
| residue mis-scaling | `l†M′r = 1.000000`; pole model ratio 1.005 near the pole |
| mixed quadratures break PSD | hybrid stays PSD at 6 % cross-pair error |
| non-congruent `G_PP` not PSD | PSD to roundoff in all three forms |
| indefinite `G_reg` | hybrid stays PSD even at `G_reg` worst `= −1.0` |

Each cost a cluster round trip. The bisection and the convexity algebra
together constrained the answer more than all four.

---

## 10. Device measurements

Bed `pgate`: the Γ-only 4-cell CNT from `cluster/l4gpu` (12 atoms/cell, 144
DOF), `retarded_method = "fft"`, one GH200.

**Converged comparison**, 80 iterations, mixing 0.02, all pre-hysteresis-fix:

| leg | resid | heat profile | non-uniformity | lead_current |
|---|---|---|---|---|
| base | 7.11e-08 | 53.80 51.84 51.90 52.07 53.83 | 0.038 | 53.81 |
| `rr_ss` | 2.87e-04 | 53.35 51.96 51.77 52.89 54.64 | **0.054** | 53.99 |
| `rr_ss_sr` | 1.00e-04 | −27.44 49.00 39.06 33.62 135.57 | 3.56 | 81.50 |

`rr_ss` is correct at convergence. Under Anderson (depth 8, 80 iterations) the
same legs give base 0.027, `rr_ss` 0.040, and `rr_ss_sr` no longer negative but
still tilted — consistent with a limit cycle traversed differently.

**Neither mixer converged, and they disagree**: `lead_current` 53.81 (linear)
vs 48.56 (Anderson), both "NOT CONVERGED" after 80 iterations. Comparisons on
this bed must be made at matched **residual**, never at matched iteration
count — the baseline's own current drifted 16 % between iteration 4 and 80.

---

## 10b. Hysteresis fix, measured

Same bed, 40 iterations, mixing 0.05, `rr_ss_sr`, WITH the position-keyed
hysteresis:

Membership over 39 iterations:

    3x 2 poles -> 7x 1 -> 2x 0 -> 7x 1 -> 3x 0 -> 3x 2 -> 2x 4 -> 2x 3 -> 10x 1

It still churns early but **settles to a stable 1-pole set for the last 10
iterations**. Before the fix it never settled -- it was still flipping 1 -> 0
at iteration 25 of 80.

Balance residual over exactly those stable iterations:

    2.113e-03 -> 2.078e-03 -> 2.047e-03 -> 2.018e-03

Smooth and **monotonically decreasing**, against `5e-3 -> 3.6e-2 -> 1e-4`
before. So the chain is confirmed: stable membership gives a continuous
fixed-point map, and the iteration then converges -- slowly (~1.5 % per
iteration at mixing 0.05), but monotonically, which it never did before.

### The residual positivity violation has moved to the mask boundary

| stage | `Sigma^>` worst | where |
|---|---|---|
| original | -1.506e-01 | w[128], mid-band |
| after the leg mask fix | -9.24e-03 | w[75] |
| after the hysteresis fix | -4.71e-02 | **w[1], adjacent to the masked DC bin** |

Three different defects in sequence, and what survives now sits at the hard
mask's edge rather than in the band. That is a direct argument for treating
the `omega -> 0` region analytically instead of masking it: the present
treatment displaces the artefact to the boundary rather than removing it.

## 11. Open

* Verify the hysteresis fix stabilises membership and lets `rr_ss_sr` converge.
* The residual second-order (λ²) positivity violation, `−9.2e-3` at λ = 1.
* **Can the DC mask be dropped entirely?** It is a grid convention, not
  physics: the code states the device `G^<` has no `1/ω` pole. Two obstacles.
  `cell_resolvent_weights` models `R` as cell-wise **constant**, so integrating
  the DC cell reproduces the artefact rather than curing it — doing it properly
  means modelling the `ω → 0` form, and it is `G^>`, not `G^<`, that is
  near-singular (`G^> = G^< + (G^R − G^A)`). And sector consistency is binding:
  the ring masks, so dropping it must happen on both sides together.
* Six config fields still unconsumed: `audit_frequencies`, `ss_kernel`,
  `keldysh_split`, `contour_quad_points`, `source_model`, `source_fit_tol`.
* Phase 6: outgoing sheet, aux grid, coupled-q (`nq > 1`), `comm.block > 1`.
* Part II (spatial/modal) entirely.
