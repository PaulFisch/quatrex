# Energy-conserving 3-phonon vertex: findings & new operating rules (2026-06-11/12)

TL;DR — the production bubble now conserves the scattering energy current to
**machine precision (1e-15)** with the *natural* device vertex, no fix-up hooks.
Getting there uncovered that the FC3 "ASR projection" we ran everywhere was
(a) unnecessary, (b) leg-asymmetric (breaking the conservation law), and
(c) suppressing 3-phonon scattering ~4-5x vs phono3py. **All anharmonic
production numbers for cnt33/cnt80/sinw produced before 2026-06-12 underestimate
scattering by roughly that factor and are superseded.**

---

## 1. The conservation law and why it matters

For the Phi-derivable (Luttinger-Ward / Baym-Kadanoff) 3-phonon bubble in the
SCBA, the scattering in/out energy flows must cancel exactly
(Lu & Wang, PRB 76, 165418 (2007), App. B):

    P_in  = sum_w hbar*w Tr[Sigma^<(w) G^>(w)]
          = sum_w hbar*w Tr[Sigma^>(w) G^<(w)] = P_out

This is implemented as `_phonon_bubble_energy_balance` in `core/scba.py`:
transpose-paired trace (Sigma_ij G_ji), same-iterand pairing (raw Sigma=F[G^n]
against G^n, checked BEFORE mixing and BEFORE the skew-Hermitian projection via
`QX_BALANCE_PRE_SYM=1`), folded to the one-sided zero-based grid through the
bosonic reflection G^<_ij(-w) = G^>_ji(w).

A standalone replica of the production discrete pipeline
(`phonon/scripts/verify/balance_roundoff_test.py`) proves the identity holds to
machine eps in float64 *and* float128 — **iff the vertex carries full S3
(triplet-permutation) symmetry**. No partial symmetry suffices: random Phi
violates at O(0.25); symmetrizing any single leg 2-cycle still leaves O(0.2-0.6);
only the full S3 average conserves (1e-16). Literature agreement:

- Lu & Wang PRB 76, 165418: SCBA conserves the energy current; the proof
  manipulates the cubic coefficients via their permutation symmetry.
- Wang, Wang & Lu, EPJ B 62, 381 (2008): the cubic vertex "is symmetric with
  respect to simultaneous permutations of the triplet".
- Luisier PRB 86, 245407 (2012): the device vertex is the *third derivative of
  the device potential* — plain truncation, symmetric by construction.

A vertex that is not totally symmetric is not the third derivative of ANY
potential -> the theory is not Phi-derivable -> conservation fails and the
scattering physics is inconsistent. **Exact conservation is therefore not
cosmetic: it certifies the vertex is a genuine derivative tensor.**

## 2. What broke conservation in production (three stacked causes)

Measured decomposition on cnt33 L2/T300 (121 pts, linear broadening):

| configuration                                  | bubble balance |
|------------------------------------------------|----------------|
| shipped vertex + zero-mode projector           | 6.0e-6         |
| exact-S3 vertex + projector                    | 2.7e-6         |
| exact-S3 vertex, projector off                 | 1.5e-15        |
| **plain/raw vertex, projector off, no hooks**  | **2.5e-16..1.6e-15** |

1. **Vertex S3 violation** — see §3. (The earlier `QX_SYMMETRIZE_FC3=1` hook
   never fixed it because it was itself buggy: the two 3-cycle key permutations
   need the INVERSE axis transpose — key-perm (1,2,0) pairs with numpy
   transpose (2,0,1) — and the average must be /6 with missing orbit members
   as zero. Fixed as mode `=2`; now obsolete since the natural vertex is S3.)
2. **Zero-mode projector** — applied two-sided to the Sigma *output*
   (Q F[G] Q) while the balance/heat trace pairs with the full G. A toy
   projector removing 1 of 6 directions violates the identity at 0.46.
   Conserving replacement: the **SSE low-frequency cutoff** (omega-diagonal
   masking of SSE inputs+outputs is inner/outer consistent — verified exactly
   harmless in the replica, including the omega=0-bin-only default).
3. **Skew-Hermitian Sigma symmetrization** (`_symmetrize_sigma`) — modifies
   Sigma after the SSE; its own contribution is the pre-sym (1e-15) vs
   post-sym (~3e-6) gap. The raw SSE output is the better object on a
   conserving vertex; re-examine whether this projection is needed at all.

Excluded by experiment along the way: frequency-window truncation (110 THz
window reproduces the 55 THz residual to 4 digits — consistent triad
truncation cancels), floating-point accumulation (replica scales with eps),
the fold layout (exact), the omega=0 inner-only masking (the two removed
relabeling partners cancel pairwise).

## 3. Root cause of the S3 violation: the ASR projection (NOT the folding)

Systematic reconstruction from the saved hiphive FCP
(`phonon/scripts/verify/plain_truncation_vertex.py`):

- **The minimum-image device folding is exact and S3-preserving.** Folded
  (1,1,3) cnt33 blocks at `enforce_asr=False`: S3 violation 0.0. A plain
  rebuild from a (1,1,5) supercell — where the 2.5 A (= 1.02 cell) fc3 cutoff
  cannot wrap — matches the folded blocks to 7.7e-16. No supercell aliasing.
  (The 2026-06-11 "image folding pins the external leg" hypothesis was wrong.)
- **`enforce_asr_fc3_matrices` (phonon_inputs/separable.py) causes the entire
  violation**, with two defects:
  (a) *leg-asymmetric*: projects (I - PP^T) on the two supercell legs but
      never the compact external leg -> destroys S3 (reproduces exactly the
      shipped violation, worst 1.276 / mean 0.60);
  (b) *over-strong*: PP^T spans the full per-atom-class q=0 subspace
      (3*nat = 36 dims for cnt33) instead of the **3 global translation
      vectors** the acoustic sum rule actually constrains. It changed the
      cnt33 tensor by **136% (rel L1)** while the raw fit's ASR residual was
      5.7e-14.
- **Both production FC3 fits are already ASR-exact raw**: cnt33 (hiphive,
  (1,1,3), cutoffs 3.5/2.5, ardr) residual 5.7e-14; d5a (hiphive, (1,1,4),
  cutoffs 7.88/4.0 + fc4, rfe) residual 2.7e-14. hiphive enforces
  translational invariance in the fit. **Nothing was generated with "old
  code"; no FC3 regeneration is needed.**
- The F9-era "d5a leg-ASR residual 0.796" was measured on the
  *device-truncated* tensor (leg sums over kept atoms only) — expected for any
  truncated vertex (the missing neighbors carry the balance), true of
  Luisier's plain truncation too, and harmless: a sub-tensor of a symmetric
  tensor restricted to the same index set on all three legs is still a
  symmetric derivative tensor -> Phi-derivable -> conserving.

## 4. Physics validation: linewidths vs phono3py

`phonon/scripts/verify/cnt33_linewidth_vertex_check.py` — single-shot NEGF
bubble (= Fermi golden rule) vs phono3py on the same FC2/FC3 and q-mesh:

- **raw / plain-truncation vertex: order-unity agreement** (R ~ 0.3-1.5 over
  all bands with real phase space, both at mesh 8/eta 0.2 and mesh 16/eta 0.1;
  the flat high-omega bands where gamma_p3p < 1e-3 THz show the expected
  eta-floor artifact, not vertex error).
- **ASR-projected vertex: uniform ~4-5x suppression** (median R ~ 0.18-0.25).

Hence "low conservation -> better physics" is literal here: restoring the
conserving vertex also restores the phono3py-consistent scattering strength.
Corollary: with this FC3 the true 300 K CNT(3,3) linewidths are O(1 THz) —
**scattering is strong**, and the corrected SCBA is much harder to converge
than the old (weakened) runs: Anderson(5)/0.1 limit-cycles, linear 0.05
oscillates. Mixing-strategy study + SCBA-level linewidth comparison running
(2026-06-12); vertex ramp (`sse_ramp_iterations`) and small linear mixing are
the current candidates.

## 5. New operating rules

1. **Do not ASR-project the FC3** (`build_inputs.py` now defaults OFF; legacy
   behavior behind an explicit `--asr` flag). For hiphive fits the raw tensor
   is already exact. If a future FC3 fit genuinely violates the sum rule,
   implement a *minimal, leg-symmetric* projection: remove only the 3 global
   translation vectors, on all three legs (S3-preserving by construction) —
   never the current projector.
2. **Plain truncation of the device vertex is correct and conserving.**
   The minimum-image folded construction is identical to it whenever the FC3
   range fits the fit supercell (verified for cnt33). Non-zero partial leg
   sums of the truncated vertex are expected open-system physics, not a defect
   to project away.
3. **Zero-mode projector stays OFF** for conserving runs; use
   `sse_low_freq_cutoff_thz` for low-omega stability instead (conserving).
4. **Gate anharmonic runs on BOTH lead balance (~1e-3) and the bubble energy
   balance** — with the conserving vertex the bubble balance sits at 1e-15;
   anything above ~1e-12 now indicates a code/config regression, not "noise".
5. **Linear broadening** `(omega^2 + 2i*eta*omega)` is the default (the old
   squared form's -eta^2 band-edge shift suppressed the low-omega plateau);
   eta -> 0 recovers exact channel quantization (cnt33 T_eff -> 3.97 ~ 4
   acoustic channels at eta=0.01; eta=0.45 under-reports ballistic G by ~35%
   at 300 K — quote matched-eta ratios or eta-extrapolate absolute G).
6. **All pre-2026-06-12 anharmonic production results (cnt33/cnt80/sinw) are
   superseded** (vertex suppressed 4-5x). Ballistic numbers are unaffected.
   The Si-film/coupled-q path never applied the projection (build_sifilm does
   not pass enforce_asr) — film anharmonic numbers are NOT affected by this
   particular bug (they were superseded earlier by the coupled-q conjugation
   fix, 2026-06-10).
7. d5a strategy: raw FC3 is fine; stability at small eta comes from the SSE
   cutoff (masks the soft twist modes out of the bubble, conservingly), a
   well-resolved grid (eta/d-omega ~ 1), and slow mixing/ramp — not from
   vertex mutilation.

## 6. Verification artifacts

- `verify/balance_roundoff_test.py` — discrete-identity replica (S3
  requirement, fold layout, DC masking, precision scaling).
- `verify/plain_truncation_vertex.py` — FCP-based plain rebuild, S3 checks,
  folded-vs-plain equivalence, ASR-projection forensics.
- `verify/cnt33_linewidth_vertex_check.py` — golden-rule linewidths vs
  phono3py, raw vs ASR-projected vertex (logs:
  `out/prod/audit_conv/cnt33_linewidth_check{,_fine}.log`).
- `verify/cnt33_scba_linewidth_plot.py` — SCBA-level linewidth comparison.
- Runs in `out/prod/audit_conv/`: `baltest_presym` (pre/post-sym split),
  `baltest_wide` (window), `baltest_noproj` (projector A/B),
  `baltest_s3fix` (exact-S3 hook), `baltest_conserving` (S3 + no projector,
  1.5e-15), `baltest_plain` (natural vertex, 2.5e-16, no hooks),
  `conv_*` / `lbcnt_plain_*` (mixing study, in progress).
- `geom/cnt33_L2/fc3_blocks_plain.hdf5` — the conserving cnt33 L2 vertex.
- Audit history: `phonon/docs/solver_audit.md` (balance sections, including
  the superseded intermediate hypotheses, kept for the record).
