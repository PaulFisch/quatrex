# Anharmonic phonon-phonon NEGF — observations & decisions for the production quatrex solver

Consolidated findings from the dense-reference investigation (May–Jun 2026), written so the
production `src/quatrex/phonon` phph solver can be designed/fixed with the physics already pinned
down. Cross-refs to the running log are `F##` in `phonon/CLAUDE.md`. **Nothing here changes
production code** — it records what the dense reference taught us.

---

## 1. The approximation hierarchy (the single most important result)

We measured what each numerical truncation of the 3-phonon self-energy costs, against the
**full** (unapproximated) self-energy, using the dense reference
(`se_finite.compute_phph_self_energy_finite_multi_slab` for Γ; `se_q…_multi_slab` for q-resolved;
study scripts `cutoff_sse`/`si_film_approx_study.py`). Ranking, **most-to-least damaging**:

| truncation | knob | effect vs full | verdict |
|---|---|---|---|
| **inter-slab (off-diagonal) Green's function $G_{KK'}$** | `g_cutoff` | **CATASTROPHIC**: 5× (d5a), 30× (d11a) error in $\Sigma^<$; **negative/unphysical** conductance in the q-resolved film | **must keep** |
| off-diagonal self-energy blocks $\Sigma_{IJ}$, $I\neq J$ | `sigma_cutoff` | off-diag carries 47% of the Frobenius weight but only **−1.8%** of conductance | safe to drop (diagonal-$\Sigma$ ok) |
| FC3 vertex slab range | `vertex_cutoff` / `fc3_nn_cutoff` | 1st-NN-slab ≡ full to **<0.1%** (and 1st-NN-atom ≡ full FD FC3 to 0.3%) | safe to truncate to 1st NN |

**Design consequence for production:** the dominant, non-negotiable ingredient is the **off-diagonal
(inter-slab) G** inside the bubble. The vertex range and the $\Sigma$ output-block range can be
aggressively truncated; the G range cannot. This is the opposite of what is cheapest to communicate,
which is why it was gotten wrong in production (see §2).

---

## 2. Status of the production quatrex phph (`sse_phonon_phonon.py`) — KNOWN DEFECT

The production `SigmaPhononPhonon` computes the **diagonal-G approximation by default and
exclusively**:
- `_gather_diagonal_blocks` only all-gathers $G_{KK}$ over `comm.stack`; *"BT-off-diagonal G blocks
  never enter the contraction."*
- the contraction (`_compute_in_stack`) feeds `G_inner_a=gl_diag[K1]`, `G_inner_b=gl_diag[K2]` — only
  on-site slab blocks; the pair index carries $(K_1,K_2)$, never $(K_1',K_2')$.
- so $\Sigma_{IJ}=\sum_{K_1K_2}\Phi_{I,K_1K_2}[G_{K_1K_1}\!\ast\!G_{K_2K_2}]\Phi_{J,K_2K_1}$.

By §1 this is the **catastrophic** truncation. It was almost certainly chosen to avoid communicating
off-diagonal G across `comm.stack`.

**Decision for the rewrite:** the default must be the **full banded** bubble — include $G_{KK'}$ for
$|K-K'|\le b$ with a band/cutoff $b$ that can span **several transport cells** (a transport cell may
hold several primitive slabs; the cutoff mirrors the banded electronic-structure RGF, not a hard
nearest-primitive-neighbour). Diagonal-G must be an **opt-in** flag, never the default. The working
blueprint is `se_finite.compute_phph_self_energy_finite_multi_slab`, which already builds the full
$(K_1,K_2,K_1',K_2')$ pair index and uses $G_{K_1K_1'}$, $G_{K_2K_2'}$ correctly; the cost is
gathering the off-diagonal G blocks within the band (extra communication along the relevant axis).

**Existing distribution (production):** energy (`comm.stack`) is **flat** — the bubble is an energy
convolution, replicated per stack rank (F22); spatial blocks (`comm.block`, the $(I,J)$ output loop)
are **near-ideal** (5.36× @ 6). A banded-G implementation should distribute the off-diagonal-G
gather along block/q, not energy.

---

## 3. The absolute self-energy prefactor is OPEN — do not trust absolute anharmonic numbers yet

- The implemented kernel is the standard sunset $\Sigma=(i\hbar/2)\Phi GG\Phi$
  (`theory.tex` eq:sigma_guo), which reduces analytically to the correct Fermi-golden-rule linewidth
  (eq:fgr). The default prefactor is **native (unscaled)**; `constants.PHPH_SYMMETRY_FACTOR=1.0`.
- **Open discrepancy:** the Si thin film over-scatters vs Guo-Bescond-Zhang (~45% reduction at 1.5 nm
  vs their ~10%). A factor ~2–4 would reconcile it (Guo report a "factor-of-4" over-count of the
  repeated pairings in the Luisier convention), but **this is NOT verified** and was **retracted**
  (F24-CORRECTION): an earlier ÷4 default was reverted.
- The phono3py golden-rule cross-check is **EXHAUSTED and inconclusive** (F28): tested on-shell,
  integrated, and mode-resolved over 2³/4³ meshes, the ratio
  $R=\int(-\mathrm{Im}\Sigma_{\rm NEGF})/\int 2\omega_s\gamma_{\rm p3p}$ never converges to a clean
  mesh-independent constant — the NEGF Lorentzian-η and phono3py's tetrahedron sample the sparse
  3-phonon joint-DOS differently. **But it does answer the ÷4 question:** the well-sampled (high-ω
  optical) modes give R/(2π)² of **order 1** (2.6 at 2³ → 0.4–0.95 at 4³), **never 4**; if native
  were 4× too large the ÷4 hypothesis would put them at ≈4. **→ no support for ÷4; native is the
  defensible default.** The (2π)² is the fixed units convention (code's linear-THz Σ vs phono3py's
  THz half-linewidth γ), not a physical factor. **The Γ-optical mode is decisive, and the
  area-integrated ratio is η-INVARIANT:** the on-shell peak scales ∝η (γ_NEGF=0.13/0.26/0.44 THz at
  η=0.02/0.04/0.08 → the spurious "factor 4"); the **area integral** R/(2π)² is η-locked at
  **1.05/1.07/1.06** (±4 THz window), spanning 0.8–1.1 across windows/grids. **→ native matches phono3py
  to ~10–15%; a factor-of-4 is excluded by an order of magnitude (÷4→0.26, ×4→4.2). The ÷4 is dead;
  native is correct, verified to ≲15%.** Fig `prefactor_verification.pdf`, results.tex `sec:res_prefactor`.
- **Definitive test still TODO (do before any production anharmonic claim):** bulk-Si κ from the
  code's **own** SCBA Green's functions (thick-film limit) vs phono3py RTA (~110 W/mK, F7/F13) on the
  identical FC3 — this uses the code's internal G convention and isolates the true absolute prefactor.
- **Implication:** ratios/trends (G_anh/G_ball vs length, the §1 approximation comparisons,
  distributed correctness) are prefactor-independent and trustworthy; **absolute** conductances are
  not, until the κ benchmark is done.

---

## 4. Distributed q (transverse momentum) — what exists, what doesn't

- **q-communicator: in production.** `QuatrexCommunicator.configure` has a third split
  (`q_comm_size` → `comm.q`), ranks factor as q×stack×block, reducing to the 2-axis layout at q-size 1
  (`src/qttools/comm/comm.py`, `src/quatrex/core/config.py`). Backward compatible; 27 comm tests pass.
- **Distributed-q self-energy: NOT in the production phph solver.** It is realised on the **dense
  reference** kernel `phonon/solver/se_q.py` driven over `comm.q`:
  - `phph_q_comm_validate.py`: external-q split + `comm.q.all_gather_v` of internal-q G; **bit-identical
    to the serial oracle** at P=1/2/4.
  - `phph_q_dist_scaling.py`: **6.58× @ 8 ranks (82%)**, Σ bit-identical across ranks (distributes over
    raw `COMM_WORLD`).
  The production `SigmaPhononPhonon` has no q-axis; the q-port is pending.
- **Physics:** the periodic self-energy couples q by crystal-momentum conservation
  ($\Sigma(\qp)=\frac1{N_q}\sum_{\qp'}\Phi GG\Phi$); distributing **external** q divides the work,
  the **internal** q' G must be exchanged each iteration (the q-axis is the *scalable* axis; the
  energy axis is not).
- **GPU memory (the binding constraint on device):** the dense vertex $\tilde\Phi(\qp,\qp')$ is
  $\cO(N_q^2\ndof^3)$ (16 GB at 8×8 mesh, n_dof=63). **Streaming** it from $T(\qp)$+$M$ per batch
  (`stream_phi`) drops the peak to $\cO(N_q\ndof^2)$, bit-identical, composes with `comm.q` at no
  extra communication. Production should adopt streaming for the q-vertex.

---

## 5. The dense reference solver (the validated blueprint, `phonon/solver/`)

Capabilities that the production solver should match (these are what we actually ran the science on):
- `se_finite.compute_phph_self_energy_finite_multi_slab` — **full off-diagonal** Γ-only multi-slab Σ
  (the correct $G_{KK'}$ treatment); `sigma_cutoff`/`g_cutoff`/`vertex_cutoff` knobs.
- `se_q.compute_phph_self_energy_q_dense[_multi_slab]` — q-resolved coupled-q Σ (single + off-diagonal),
  `symmetry_factor`, `stream_phi`; reduces to `se_finite` at 1×1 (rel err 0) and to the Γ bubble (2.6e-16).
- `dense.transmission_finite` / `transmission_q` — Γ-only and q-resolved SCBA drivers;
  `mass_profile` (per-slab masses → Si/Ge heterostructure), `legacy_prefactor`.
- `leads.build_device_hamiltonian_massprofile` — per-slab-mass device for heterostructures.
- **Cost caveat (F19):** the dense bubble is **cubic** in #cells and does NOT scale to n_slabs≥2;
  production must use the banded RGF + distribution. RGF≡dense to ~1e-13; energy-parallel 5.3×@8.

---

## 6. Computed results so far (all dense reference; read with the §3 prefactor caveat)

- **Si thin-film cross-plane κ vs Guo (F23):** ballistic conductance matches Guo's scale to ~1%
  (q-converged at nk≥8, ~920 MW/m²K; κ_ball linear in L). Anharmonic over-scatters (native): see §3.
- **Si/Ge heterostructure (F27):** Ge mass-mismatch barrier raises **ballistic** R by +344–369%
  (interface reflection); phph enhancement is +32%→+51% (2.3→5.4 nm) in the pure Si film but
  ~neutral in the barrier-dominated heterostructure (barrier filters to low-ω anharmonically-inactive
  phonons). Magnitude inherits the §3 caveat.
- **SiNW d5a vs Luisier 2012 (at ÷4 prefactor, caveat):** G_anh/G_ball = 0.769 (L=1), 0.772 (L=2) —
  ~23% resistive reduction, ~constant with length; vs Luisier's ~41% (1/1.7) for 3 nm wires. Our ~1 nm
  wire is boundary-limited (F21: <10 nm MFP carries 1.3% of bulk κ) → smaller, mostly size-driven
  reduction; native prefactor would increase it. Qualitatively consistent (resistive, anharmonic<ballistic).
- **Bulk Si κ (FD FC3, phono3py RTA):** 110.7 (11³) → 115.2 (19³) W/mK vs exp ~150 (2×2×2 FC3 underestimate).
  This is the prefactor-benchmark reference (§3).
- **Mode physics (F21):** lifetimes τ∝ω⁻² (median 4.2 ps); 50% of κ from MFP>134 nm; 93% acoustic;
  Umklapp 0.44@300K; mean |γ|=1.04 (matches exp → FC3 anharmonicity strength validated).
- **FC method (F20):** DFPT vs FD agree to 1.6% on Γ-optical.
- **Distributed (F12/F19/F22/F23):** RGF linear vs dense cubic (57×@32); energy-parallel 5.3×@8;
  production phph block 5.36×@6 / energy flat; q-axis 6.58×@8.

---

## 7. Open / in-progress

- **Prefactor κ-benchmark (§3)** — the decisive TODO before trusting absolute anharmonic numbers.
- **Ge thin film (new material):** first DFT was **k-under-converged** (PBE Ge near-metallic; 2×2×2
  supercell k → Γ-optical 6.2 vs ~9 THz, κ=5.5 vs ~60). A **denser-k rerun (4×4×4, ecutwfc 60)** is
  in progress; after it finishes, the 8 phono3py None/duplicate supercells must be generated+run
  before reap (the reap indexes the full displacement set; sow skips None — fixed in `thirdorder.py`
  by generating them explicitly, see `fc3_ge/disp-000{27..30,83..86}`).
- **d5a Luisier L=3** still running; **production q-self-energy port** pending.

---

## 8. Operational lessons (carry into production runs)
- phono3py FC3 sow skips symmetry-redundant ("None") supercells but the reap needs the full
  displacement set → generate+run the None ones explicitly before reaping.
- PBE Ge is near-metallic → FC needs a dense supercell k-mesh (2×2×2 too coarse; use ≥4×4×4) + smearing.
- QE `pw.x` deadlocks at `mpirun -np>1` on this node → run serial; `conv_thr=1e-10` stalls (use 1e-8).
- Kill stale python orphans before launching (they starve the bubble memory governor); single-thread
  BLAS + `QUATREX_PHPH_THREADS`; the dense bubble is single-core-bound for n_slabs=1 — scale via
  n_slabs≥2 / RGF / distribution.
