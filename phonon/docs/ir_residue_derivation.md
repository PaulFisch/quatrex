# The infrared channel of the phonon-phonon bubble: continuum derivation and the surgical subtraction

2026-08-05. Companion measurements: `phonon/studies/_ir_exponents.py`,
`_ir_killtest.py`, `_ir_residue_check.py` (artifacts under
`phonon/studies/out/ir_residue/`). All numbers below are measured on the
MoS2 L3 film at eta = eta_obc = 0 unless stated; the dense bed
reproduces the recorded engine G to ratio 1.0000
(`run_ballistic.npz`, spectral NEVP OBC).

Everything is derived from the continuum open-system equations; no
numerical regulator, cutoff, or broadening appears anywhere.

## 0. Conventions

Mass-weighted displacements `w = sqrt(m) u`; dynamical matrix `D`
(THz^2); frequencies are linear THz (`nu`, written `w` below).
Production sign conventions throughout: system matrix `w^2 I - D`,
contact `Sigma^< = +i n_alpha Gamma_alpha` (occupation-positive,
`-i G^{<,>} >= 0`), `Gamma_alpha = i(Sigma^R_alpha - Sigma^A_alpha)`,
`A = -i(G^> - G^<)`. Bose pole coefficient `c_alpha = k_B T_alpha / h`
(`bose_pole_coeff`; 6.25 THz at 300 K). Bosonic fold
`G^<_ij(-w) = G^>_ji(w)`. The SSE bubble (eq:sigma_guo):

    Sigma^{<,>}(w) = (i hbar / 2) Phi [ int dw'/2pi
                       G^{<,>}(w') (x) G^{<,>}(w - w') ] Phi .

## 1. The infrared limit of the legs (open harmonic system, eta = 0)

### 1.1 Lead self-energy at small w

The leads are semi-infinite harmonic crystals; their retarded surface
self-energy on the contact block expands as

    Sigma^R_alpha(w) = Sigma_alpha(0) - i w V_alpha + O(w^2),
    Gamma_alpha(w)   = 2 w V_alpha + O(w^3),      V_alpha >= 0.

`Sigma_alpha(0)` is Hermitian (the static termination correction);
`Gamma` is odd and opens linearly because the acoustic mode density
does. The sign of the `-i w V` term is fixed by `Gamma >= 0`.
Measured (film, translation-basis eigenvalues, w->0 extrapolated):
`V_{L,T} = V_{R,T} = diag(0.11436, 0.11438, 0.11732)` THz.

### 1.2 Static screening theorem: K t = 0

Let `K = D_dev + Sigma_L(0) + Sigma_R(0)` be the static lead-screened
device stiffness, and `t_beta` the mass-weighted uniform translations
(`(t_beta)_{j,beta'} = sqrt(m_j) delta_{beta beta'}`). Then

    K t_beta = 0  exactly.

Proof: a uniform translation of device plus leads is a zero mode of
the infinite crystal (acoustic sum rule of the fc2). Eliminating the
leads at their *static* response — which is what `Sigma_alpha(0)`
is — preserves that zero mode: under a uniform device shift the leads
follow rigidly and exert no restoring force. Measured:
`|K t| / |K| = 3.1e-6`, saturated by the fc2's own ASR quality
(5.4e-6), i.e. exact to the input's precision.

### 1.3 Laurent structure of the legs

Write `P` for the projector onto `T = span{t_beta}` (rank 3) and
block-invert the Dyson equation at small w. On `T` the static term
vanishes (Sec. 1.2), leaving `[w^2 + i w V_T]^{-1}` with
`V_T = V_{L,T} + V_{R,T}`; the Schur coupling through the orthogonal
(gapped) space enters at higher order. Hence

    G^R(w)|_T = -i V_T^{-1} / w + O(1) ,

and with `Sigma^< = i sum_alpha n_alpha(w) 2 w V_alpha`,
`n_alpha(w) = c_alpha/w - 1/2 + O(w)`:

    -i G^<(w)|_T = C2 / w^2 - V_T^{-1} / w + O(1)
    -i G^>(w)|_T = C2 / w^2 + V_T^{-1} / w + O(1)

    C2 = V_T^{-1} ( sum_alpha 2 c_alpha V_{alpha,T} ) V_T^{-1} .

The even double pole `C2/w^2` comes from the Bose `c/w` times the
acoustic `1/w` of the resolvent; the odd simple pole is
*parameter-free*: its residue is `-+ V_T^{-1}`, independent of
temperature (it descends from the `-1/2` of the Bose expansion, and
equals the Laurent part of the spectral function,
`A|_T = 2 V_T^{-1} / w`). Both signs and the fold
`G^<(-w) = G^>(w)^T` are satisfied identically by this pair.

Verified on the film (translation basis): predicted C2 eigenvalues
(53.283, 54.651, 54.662) vs measured `w^2(-iG^<)|_T`
(53.277, 54.600, 54.683), matrix relative error 5.9e-4; retarded
residue `w G^R|_T = -i V_T^{-1}` to 2.9e-3. The measured exponents
(slopes -1.999/-1.982/-1.951 for `|G^<|`, -0.986/-0.960 for `|G^R|`)
and the eigenvector identification of the channel with the
translations (overlap 1.000000) are in
`out/ir_residue/exponents_mos2f3nu.json`, `killtest_film.json`.

### 1.4 Physical identity

The channel is the Brownian motion of the device centre of mass,
driven by the thermal leads: a free (zero-stiffness) collective
coordinate with damping `V_T` and thermal drive `sum 2 c_alpha
V_alpha`; `C2/w^2` is its classical equipartition spectrum. At equal
temperatures `C2 = 2 c V_T^{-1}`. This weight is *real physics of the
open harmonic system* — it belongs in the Dyson G, the current, and
the observables. What it must not do is scatter (Sec. 2). It lives
exclusively at transverse q = Gamma (gapped q-points carry ~1e-17 of
it) and, being a free-particle spectrum, it also makes the equal-time
`<ww>` of the open device formally divergent — the previously recorded
ill-conditioning of the eta = 0 `<uu>` quadrature on IR-resolved grids
(`scp_uu_min_thz` note in `sse_phonon_phonon.py`) is this same
channel.

## 2. The vertex theorem and its breakdown under device truncation

### 2.1 Extended theory: the channel cannot scatter

Translation invariance of the full cubic potential,
`U_3(u + c) = U_3(u)`, gives the cubic acoustic sum rule

    sum_{j in crystal} sqrt(m_j) Phi^mw_{i, j beta, k} = 0
    for every (i, k, beta),

i.e. the mass-weighted vertex annihilates the *global* translation on
every leg. The w -> 0, q -> 0 channel of Sec. 1 is spatially exactly
that global translation, so its contribution to `Sigma^B` vanishes
identically: uniform motion changes no interatomic distance. At small
finite wavevector the matrix element vanishes as `O(q)` (Herring), so
the channel's true contribution is `O(w^2/c_s^2)`-suppressed — finite
and small, not divergent.

### 2.2 The model truncation creates the artifact

The production model keeps `Phi` only for triplets fully inside the
device and runs the bubble legs over device DOF only. The restricted
sum rule fails by exactly the dropped device-lead mixed triplets:

    sum_{j in dev} sqrt(m_j) Phi^mw_{ijk}
      = - sum_{j in leads} sqrt(m_j) Phi^mw_{ijk}  !=  0

for (i,k) within cutoff of the boundary. The model's cubic potential
is therefore *not* translation invariant, and the model scatters
centre-of-mass motion with an O(1) leak where the true theory has
O(q). Measured leak: `|Phi . t| / |Phi| = 8.5e-4` per leg
(edge-class dominated — which is why the interior ASR repair, job
4327969, tracked the unrepaired orbit: it fixed the 1.7e-5 interior
part of a leak that is structurally 5e-3-class at the edges).

### 2.3 Consequences (all previously measured, now attributed)

The bubble integrand acquires an uncancelled `leak^2 * C2 / w'^2`
channel — non-integrable. On a grid the first bins contribute
`~ 1/dw^2` each, so the discrete sum grows as the grid refines:
the u2001 grid (dw = 0.008) carries ~200x the nu-grid's first-bin
weight and exploded at iteration 4 where the nu grid orbits.
Ring-contraction measurement: full legs scale as `w'^-2` (one leg
near DC) and `w'^-4` (both legs); with the rank-3 translation channel
projected out, both are flat (slopes 0.00 over three decades). The
channel/regular crossover sits at 0.18 THz — the 1.5 THz mask that
rescued the fine grid removed the artifact *and* a factor ~8 of
physical spectrum above it. The CNT and Si films carry the same
truncation leak but their device modes sit above the
channel-dominated region (the established spectral-overlap
discriminator), which is why they converge regardless.

## 3. The surgical subtraction

### 3.1 Definition: subtract the exact centre-of-mass channel

The subtracted field is not the Laurent series but the *exact*
Green's function of the CM coordinate itself — the lead-damped free
particle on the translation subspace (all objects 3x3 in the `t_hat`
basis, embedded by `t_hat . t_hat^T`):

    S^R(w)     = [ w^2 + i w V_T ]^{-1}
    S^{<}(w)   = S^R(w) [ i sum_alpha 2 w n_alpha(w) V_{alpha,T} ] S^A(w)
    S^{>}(w)   = S^R(w) [ i sum_alpha 2 w (n_alpha(w)+1) V_{alpha,T} ] S^A(w)

with `V_alpha` from the run's own lead model
(`Gamma_alpha(w)/2w -> V_alpha` at build time; no fit, no free
parameter, no scale). Because `S` is a genuine open-subsystem
Green's-function pair it satisfies the bosonic fold and, at equal
temperatures, the KMS condition *exactly at every frequency* — not
just asymptotically — while its Laurent parts coincide with Sec. 1.3
(`C2/w^2 -+ V_T^{-1}/w`). The strict-Laurent variant fails detailed
balance at the truncated order (measured 4.5e-2 on the chain gate);
the CM form restores it to 1.3e-14 and shrinks the ledger defect
`Delta` from 2e-4 to 2e-7 (Sec. 4.2, `_ir_conserve_gate.py`).

The subtracted bubble uses legs

    Gbar^{<,>} = G^{<,>} - S^{<,>}      (SSE legs only)

on the q = Gamma pair, with *zero add-back*. The Dyson equation, the
current, and all observables keep the full `G` (mask precedent). The
physical statement is a two-fluid split: `G = S + Gbar`, CM Brownian
motion plus internal motion; the extended theory's vertex couples
only to the internal part, and the model's leaked CM coupling is the
artifact being removed.

### 3.2 Why zero add-back is the physical value

Standard singularity subtraction would add back the analytic
(principal-value / residue-calculus) integral of `S` against the
other leg. Here the physics fixes that term instead: the true theory
evaluates the channel's vertex contraction to its `O(w^2)` Herring
value, which the device-truncated vertex cannot represent — the
leaked O(1) contraction is spurious in its entirety. Setting the
channel's anharmonic contribution to zero is exact at `w' -> 0` and
errs at finite `w'` only by the physically small gradient-vertex
term the model never contained. This is the residue-theorem
connection: the Laurent coefficients are computed exactly
(Sec. 1.3), and the residue integrals that quadrature would need are
replaced by their symmetry-dictated value.

The subtraction is scale-free: `S` is subtracted at every frequency,
its tail decaying as `1/w^2`; the channel's *deviation* from its
Laurent form — the physical finite-frequency acoustic content of the
translation corner — stays in `Gbar` untouched. Nothing outside the
rank-3 corner is modified at any frequency.

### 3.3 What remains near DC

After subtraction the legs are bounded at w = 0 (measured: the
projected contraction is flat over three decades), the integrand is
Riemann-integrable, and ordinary quadrature converges *faster* on
finer grids — restoring the resolution-consistency a discretisation
must have. No PV quadrature is needed on the <,> legs at all; the
only principal value in the pipeline remains the existing
Kramers-Kronig construction of `Sigma^R`, which now acts on subtracted
(bounded) inputs.

## 4. Conservation

### 4.1 The bare discrete balance, proven

The production gate is `B = sum_w w Tr[Sigma^<(w) G^>(w) -
Sigma^>(w) G^<(w)] = 0` (transpose pairing). Inserting the bubble and
using the fold on the closing G, each term becomes a closed triangle
of three lesser functions at frequencies `(w1, w2, w3)` constrained
by `w1 + w2 + w3 = 0`, contracted with the two vertices:

    T(w1, w2, w3) = Phi_{mu n1 n2} G^<_{n1 n4}(w1) G^<_{n2 n3}(w2)
                    Phi_{mu' n3 n4} G^<_{mu mu'}(w3)-type closure.

With the S3-symmetric vertex and trace cyclicity, `T` is symmetric
under permutations of its three frequency slots, so

    B  proportional to  sum_{w1+w2+w3=0} w3 T
       = (1/3) sum (w1 + w2 + w3) T = 0 .

The ingredients are exactly: (i) S3 symmetry of Phi, (ii) the bosonic
fold of every leg, (iii) permutation symmetry of the constrained
frequency set (which the zero-padded FFT realises). This is why the
bare bubble conserves to 1e-14..1e-16 in every recorded run.

### 4.2 The subtracted balance

`S` is constructed fold-symmetric (`S^<(-w) = S^>(w)^T`, Sec. 1.3),
so the identical proof applies verbatim with every leg `Gbar`:

    sum_w w Tr[Sigma_sub^< Gbar^> - Sigma_sub^> Gbar^<] = 0
    exactly, on any grid.

This is the conserving gate for the scheme. The production ledger
paired with the full Dyson `G` differs from it by

    Delta = sum_w w Tr[Sigma_sub^< S^> - Sigma_sub^> S^<] ,

the anharmonic energy the model exchanges with the centre-of-mass
channel. In the difference the even `C2` part meets
`Sigma_sub^< - Sigma_sub^>` (which vanishes ~ w at DC), so `Delta` is
finite and small — it is precisely the CM-drag term the true theory
suppresses to `O(q^2)`, reported as a number, not silently absorbed.
This also explains the historical landmine: the earlier "data-driven
ressub" probe modified legs without a fold-symmetric `S` and without
the consistent pairing, and degraded the balance from 7e-18 to
1.2e-6; and it is the same legs-AND-outputs symmetry that makes the
lowmask conserving — the mask is the crude limit of this scheme, with
the rank-3 Laurent corner replaced by everything below 1.5 THz.

Detailed balance at equilibrium: the CM-channel `S` satisfies KMS
exactly (Sec. 3.1), hence so does `Gbar`, hence
`Sigma_sub^> = e^{beta h w} Sigma_sub^<` exactly. Measured on the
chain gate: 1.29e-14 (vs 4.5e-2 for the strict-Laurent variant — the
measurement that fixed the scheme's final form).

Chain-gate numbers (`_ir_conserve_gate.py`, 6-DOF diatomic device,
spectral OBC, T_L/T_R = 305/295, random S3 vertex WITHOUT sum rule):
bare balance 0.8-1.4e-16; subtracted Gbar-paired balance 1-4e-17 on
every rung; ledger defect Delta = 2.3e-7 -> 1.6e-7 (falling with dw);
disease/cure at w_out = 1 THz: bare 4.50e4 / 8.72e4 / 1.71e5 under
dw = 0.1/0.05/0.025 (the exact 1/dw law), subtracted
1.20e3 / 1.27e3 / 1.30e3 (first-order convergence to ~1.33e3).

## 5. Discrete rule and implementation seam (P3, summary only)

On the one-sided production grid: evaluate `V_alpha` once from the
contact model, form `S^{<,>}(w_k)` on the conv grid, subtract from
the q = Gamma leg arrays before the tau FFT (both legs — the
`w' = w` locus is then covered by symmetry), leave outputs and the
Hilbert path unchanged. Rank-3 outer products, `O(n_w b^2)` cost.
Aux-grid mode: subtract on the primary grid before interpolation
(same reasoning as the existing pre-interpolation DC mask). Opt-in
flag, legacy-identical default. Finite-nq caveat: on finer q-meshes,
near-Gamma points acquire large-but-finite (gap-capped) channel
weight that the Gamma-only subtraction does not touch; it is measure-
suppressed by the 2D q-sum, and the QCONV ladder will quantify it.

## 6. Pre-registered gates and falsification (P2)

- V1 exactness: scalar/rank-3 models of the *derived* class
  (C2/w^2 even + V^{-1}/w odd); subtracted quadrature error must fall
  uniformly with refinement; the bare rule must reproduce the 1/dw
  divergence law.
- V2 conservation: Gbar-paired balance <= 1e-14 on the two-temperature
  fold-symmetric model; `Delta` reported.
- V3 equilibrium: detailed balance and `<ww>` against the mode sum
  with the CM channel handled analytically.
- V4 toy disease: acoustic-only toy ladder — Sigma error vs nfreq
  must turn from growing to falling; Jacobian spectral radius vs bare.
- V5 devices: CNT33 L4 single-shot null test (a-priori bound: CNT has
  the same channel; the subtraction's effect there must be within the
  derived size and leave the converged state a fixed point);
  film single-shot + rho(J) at the deepest saved state.
- Falsification: if the exactly-subtracted film still orbits at
  eta = 0, the residual marginal interlayer gain is physical, and the
  negative result stands on a now-clean discretisation: that is the
  answer, not a failure of the method.

## 7. Numbers (all measured this session, laptop)

| quantity | predicted | measured |
|---|---|---|
| `K t` (rel) | 0 | 3.1e-6 (= fc2 ASR quality) |
| `V_{L,T}` eigs (THz) | — | 0.11436, 0.11438, 0.11732 (= V_R) |
| C2 eigs (THz) | 53.283, 54.651, 54.662 | 53.277, 54.600, 54.683 |
| C2 matrix rel err | — | 5.9e-4 |
| `w G^R|_T` vs `-i V_T^{-1}` | — | 2.9e-3 |
| leg slopes (G^<, G^R) | -2, -1 | -1.999, -0.986 |
| channel eigvec vs t | 1 | 1.000000 |
| vertex leak / leg | 0 in ext. theory | 8.5e-4 |
| ring slopes full / projected | -2, -4 / regular | -2.0, -4.0 / 0.00, 0.00 |
| channel/regular crossover | — | 0.18 THz |

Film-scale demonstration (`_ir_subtraction_demo.py`: real 54-DOF
ballistic legs, real 15-block vertex, W = 12 THz, dw ladder
0.1/0.05/0.025/0.0125, values normalised to the coarsest rung):

| w_out | bare | CM-subtracted |
|---|---|---|
| 0.5 THz | 1.0 / 2.20 / 4.65 / 9.59 (1/dw law) | 1.0 / 1.06 / 1.10 / 1.14 |
| 1.0 THz | 1.0 / 1.87 / 3.68 / 7.35 | 1.0 / 1.00 / 1.02 / 1.07 |
| 3.0 THz | 1.0 / 0.57 / 0.38 / 14.8 (cancellation noise) | 1.0 / 1.01 / 1.02 / 1.06 |

The 3 THz bare row is float noise: the e37-scale channel terms cancel
past machine precision (e20 residue, erratic ladder) — on fine grids
the bare bubble is not merely divergent at small w_out but
catastrophically ill-conditioned at every output. The subtracted
bubble is well-conditioned and grid-stable everywhere. The CM form
also removes the Laurent variant's mid-band over-subtraction
(Sigma_sub(3 THz) smaller by 170x).

Reproduce:
`QTX_ARRAY_MODULE=numpy python phonon/studies/_ir_exponents.py`,
`_ir_killtest.py`, `_ir_residue_check.py`, `_ir_conserve_gate.py`,
`_ir_subtraction_demo.py`.
