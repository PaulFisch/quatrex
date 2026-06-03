# Hiphive Fitting & Sum-Rule Notes

Captures the design decisions behind `HiphiveConfig.fit_method`,
`HiphiveConfig.rotational_sum_rule`, and the new `convergence` field.
Intended audience: anyone refitting an existing reap or sweeping
parameters.

Verified against **hiphive 1.5** and **trainstation 1.1**.

---

## 1. Fit methods

`hiphive.utilities.prepare_structures` + `hiphive.StructureContainer`
produce the regression matrix `(A, b)` where `A` is `(n_eqn × n_param)`
(linearised forces vs. cluster-expansion parameters) and `b` is the
flattened force residual. The hyperparameter-free linear solve is then
delegated to `trainstation.Optimizer(fit_data, fit_method=...)`. The 11
methods available in this codebase are:

| `fit_method`                     | Sparse | Required `fit_kwargs`            | Best at | Notes |
|----------------------------------|--------|----------------------------------|---------|-------|
| `least-squares`                  | no     | —                                | overdet. + low noise | unstable in underdet., no train/test split |
| `least-squares-with-reg-matrix`  | no     | `regularization`                 | overdet. + Tikhonov |  |
| `ridge`                          | no     | `alpha` (default 1e-6)           | overdet. with weak ill-conditioning |  |
| `lasso`                          | yes    | `alpha`                          | underdet., feature-selection target | non-convex if `alpha` too small |
| `adaptive-lasso`                 | yes    | `alpha`, `gamma`                 | underdet., better consistency than plain lasso |  |
| `elasticnet`                     | yes    | `alpha`, `l1_ratio`              | mix of ridge + lasso |  |
| `omp`                            | yes    | `n_nonzero_coefs` *or* `tol`     | strict sparsity target |  |
| `rfe`                            | varies | `n_features_to_select`           | underdet. with known parameter budget | greedy backwards-elim |
| `rfe` (via `step`+CV)         | varies | `n_features_to_select`, `cv`     | unknown target sparsity | CV picks rank |
| `ardr` *(automatic relevance det.)* | yes | `threshold_lambda` (default 1e4) | underdet., wants automatic sparsity | Bayesian, no CV needed |
| `bayesian-ridge`                 | no     | `alpha_1`, `alpha_2`, `lambda_1` | overdet., uncertainty estimates | doesn't sparsify |
| `split-bregman`                  | yes    | `mu`, `lmbda`                    | very sparse + smooth | slow, niche |

### Recommendations from the hiphive paper (Eriksson, Fransson, Erhart, *Adv. Theory Simul.* **2**, 1800184 (2019)) and our usage

| Regime                                            | Recommendation         | Why |
|---------------------------------------------------|------------------------|-----|
| `n_eqn ≥ 4 · n_param` (overdetermined)            | `least-squares`        | Best Cramér–Rao bound when ground truth is dense. |
| `n_eqn ≈ n_param` (boundary)                      | `ridge` (α≈1e-6)       | Near-singular `A^T A` regularised at machine precision. |
| `n_eqn < n_param` (underdetermined, typical FC3)  | **`ardr`** (default) or **`rfe`** | Both choose sparsity automatically. `ardr` is faster and parameter-free; `rfe` gives a smoother train/test curve. |
| Sparsity target known a priori                    | `omp`                  | Exact `n_nonzero_coefs` enforcement; deterministic. |

Hiphive's own documentation [1] explicitly recommends `ardr` as the
default for FC3 fits where `n_param > n_eqn`. Our SiNW d5a reaps fall
into this regime: with 24 rattled structures × 33 atoms × 3 force
components = 2 376 equations versus ~5 000 cluster parameters at
`cutoffs=[5.0, 4.0]`.

### Convergence harness defaults

The Phase-5 `--convergence-check` mode sweeps `n_structures ∈ {6, 12,
18, 24}` and refits with each of `cfg.convergence.fit_methods` (default
`("rfe", "ardr")`). Output: `convergence_vs_n_structures.pdf`, with
RMSE train/test (left axis, log-scale) and dispersion-maximum-frequency
(right axis) per `(n_structures, fit_method)` cell.

[1] https://hiphive.materialsmodeling.org/dev/advanced_topics/sparse_regression.html

---

## 2. Sum rules

### 2.1 Acoustic sum rule (ASR)

ASR encodes translational invariance: shifting *all* atoms by the same
vector cannot produce a force. Equivalently,

```
Σ_j  Φ²_{ij,αβ} = 0,        Σ_{j,k} Φ³_{ijk,αβγ} = 0.
```

Hiphive enforces ASR in two complementary places:

1. **At cluster-space construction.**
   `ClusterSpace(prototype, cutoffs)` always includes the translational
   sum-rule constraints in the orbit-symmetry projector. The resulting
   cluster space is ASR-by-construction in the *fit basis*. This is
   already on for every fit we run.

2. **As a post-fit projection on the force constants.**
   `ForceConstants.enforce_acoustic_sum_rules()` projects the final FC2
   and FC3 tensors onto the ASR null-space. Numerically distinct from
   (1) because the orbit basis can have a tiny ASR residual under finite
   precision and the rotational sum rule (below) shifts parameters off
   the strict ASR manifold.

The current `hiphive_fc3.reap` already calls (2); we keep this.

### 2.2 Rotational sum rules

Rotational invariance encodes that rigidly rotating all atoms cannot
produce a force. Two flavours, both implemented in
`hiphive.core.rotational_constraints.enforce_rotational_sum_rules`:

* **Huang** sum rules — relate Φ² alone (Born-Huang elasticity tensor
  symmetry). Necessary for the elastic constants to obey the Cauchy
  identity and for the ZA flexural branch in nanowires / sheets to
  start at ω=0 with the correct ∝ q² dispersion.
* **Born-Huang** sum rules — couple Φ² and Φ³, ensuring that the energy
  expansion is invariant under infinitesimal rotations through 3rd
  order.

**Why this matters for nanowires.** A SiNW (or sheet) has a free
transverse direction; without rotational invariance the ZA branch
develops a spurious low-frequency gap at Γ (often ~0.5–2 THz) and the
in-plane sound velocities pick up errors at the ~10 % level. References:

* Mounet, Marzari, *Phys. Rev. B* **71**, 205214 (2005) — Cauchy
  identity violation traced to missing rotational invariance in DFT FC2
  fits.
* Carrete *et al.*, *Mater. Res. Lett.* **5**, 1 (2017) — quantifies
  ZA-branch artefacts in BTE thermal conductivity when rotational rules
  are dropped on graphene-like systems.
* Eriksson, Fransson, Erhart (2019) — hiphive paper, § "Sum rules";
  explicitly recommends enabling rotational rules for low-dimensional
  systems.

### 2.3 The three options exposed by `HiphiveConfig.rotational_sum_rule`

| Value         | What we do                                                                | Cost | Recommended for |
|---------------|---------------------------------------------------------------------------|------|-----------------|
| `"off"`       | Run the fit; only ASR (1) + post-fit ASR projection (2) are applied.      | nil  | 3-D bulk crystals where rotational violation is small (check `rotational_residual` in `hiphive_fit.json`). |
| `"post_fit"`  | Run the fit, then call `enforce_rotational_sum_rules(cs, parameters, sum_rules=['Huang', 'Born-Huang'], alpha=1e-6)` to project the parameter vector. Re-runs `ForceConstants.enforce_acoustic_sum_rules` afterwards (rotational projection shifts parameters off the ASR manifold). | <1 s | **Default for SiNW and 1-D / 2-D systems.** Closes the ZA gap at Γ. **In practice, with our typical FC3 cutoff of 4 Å, the Born-Huang projection touches only FC2 parameters — see "Empirical behaviour" below.** |
| `"constrained"` | Build the *constrained* cluster space: append the rotational-constraint rows from `get_rotational_constraint_matrix(cs)` to `(A, b)` with a large weight before calling `Optimizer`. The fit is exactly rotationally invariant; the train RMSE is slightly higher. | ~1.5× fit | Tight elasticity targets (e.g., matching experimental sound velocities) where post-fit projection is too lossy. |

The Phase-5 implementation defaults `rotational_sum_rule` to `"off"`
(matches the current behaviour); SiNW configs flip it to `"post_fit"`.

The post-fit pass logs the **rotational residual** before/after into
`hiphive_fit.json` (`rotational_residual_before`,
`rotational_residual_after`) so the user can see how much was projected
away. A "before/after" ratio above 0.1 indicates the underlying fit is
fighting rotational invariance — usually a sign that the FC3 cutoff is
too short.

#### Empirical behaviour of post-fit projection

Measured for the d5a SC4 fit (cutoffs `[6.09, 4.0]`, ARDR, see
`scratch/imag_audit/pass4_d5a_rsr_delta.json`):

* `||M @ params||` goes from **48.8 → 0.003** ✓ (projection works).
* Relative change in cluster parameters: **1.2e-3**.
* **`||FC2_after − FC2_before||_F / ||FC2||_F ≈ 3 %`** — FC2 moves a few percent.
* **`||FC3_after − FC3_before||_F / ||FC3||_F = 0.000`** — FC3 is unchanged.

The Born-Huang constraint couples Φ² and Φ³ in principle, but with our
typical FC3 cutoff of 4 Å the cubic orbits don't span the parameter
directions that the rotational matrix `M` selects, so the projection
is effectively FC2-only in our cluster spaces. Earlier copies of this
note claimed the projection cleans both FC2 and FC3 — that is wrong
in our regime.

A side-effect on SiNW fits: the projection enforces rotational
invariance *perfectly* but can shift FC2 onto a slightly less-PSD
manifold, **adding a few small-magnitude (≤ 2 THz) Γ-point imaginary
modes** as a trade-off. With a wide-enough FC2 cutoff (≥ wire diameter)
the projection still converges to a clean spectrum; with a tight
cutoff it can introduce more imaginary modes than it removes. The
d5a SC4 cutoff sweep shows this in detail
(`scratch/imag_audit/INVESTIGATION_NOTES.md`, Pass 1).

---

## 3. Verification protocol

For each candidate (`fit_method` × `rotational_sum_rule`):

1. **Hold-out RMSE.** `trainstation.Optimizer` exposes `rmse_test`
   directly (set `train_size < 1.0`); we use 0.8 / 0.2 split by default.
   Plot vs. `n_structures`; convergence means the curve flattens.
2. **Imaginary modes.** Run phonopy on the fitted FC2 on a fine
   (32×32×32 for bulk, 1×1×32 for nanowire) q-mesh; record `max(ω²<0)`.
   For an ASR+rotational fit on a stable structure this should be
   < 0.1 THz; persistent imaginary modes at the band edge are the
   smoking gun of incomplete sampling.
3. **LA-branch slope at Γ.** Fit `ω = v_s · q` on the lowest-3 modes of
   the band-structure path for `|q| < 0.1 (2π/a)`; compare to elastic
   constants (or experimental sound velocity). > 10 % deviation
   typically requires more rattled structures *or* a `phonon`-rattle
   pool at the right temperature (Phase 6).
4. **Wall time.** Per-fit-method time + memory; recorded in
   `convergence_summary.json`.

The convergence harness emits all four metrics in
`convergence_summary.json` and the consolidated plot
`convergence_vs_n_structures.pdf`.

---

## 4. Implementation notes (for future readers)

* **API stability.** Hiphive flags
  `enforce_rotational_sum_rules` as "interface may change in future
  releases". The wrapper in `hiphive_fc3._apply_rotational_sum_rules`
  pins the call signature and re-raises a clear error if the import
  fails on a hiphive upgrade.
* **Mutually exclusive options.** `rotational_sum_rule="constrained"`
  is incompatible with sparse fitters (`lasso`, `omp`, `ardr`) because
  they don't accept augmented-Lagrangian rows. The implementation
  raises a `ValueError` at config-load time when the combination is
  requested.
* **Reapability.** All sum-rule projections happen in `reap`, never in
  `sow`; you can re-run `reap` with a different
  `rotational_sum_rule` against the same DFT forces.
