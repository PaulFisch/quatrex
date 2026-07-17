"""Assemble the campaign artifact page (self-contained HTML, base64 figures).

Output: phonon/studies/out/anderson_test/campaign_report/campaign_report.html
"""
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REP = ROOT / "phonon/studies/out/anderson_test/campaign_report"
FIG = REP / "fig"


def img(name, caption=""):
    b64 = base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (f'<figure><div class="frame">'
            f'<img src="data:image/png;base64,{b64}" alt="{name}" '
            f'loading="lazy"></div>{cap}</figure>')


def tile(v, unit, k):
    u = f" <small>{unit}</small>" if unit else ""
    return (f'<div class="tile"><div class="v">{v}{u}</div>'
            f'<div class="k">{k}</div></div>')


HEAD = """<title>Phonon SCBA campaign — observables, convergence, the kernel-band mechanism</title>
<style>
:root{
  --bg:#fbfaf7;--ink:#20242a;--ink2:#5a616b;--ink3:#8a9099;
  --accent:#155e75;--rule:#e3e0d8;--card:#ffffff;--warn:#9a3412;
}
@media (prefers-color-scheme: dark){:root{
  --bg:#14171c;--ink:#e6e4de;--ink2:#a8adb5;--ink3:#767c85;
  --accent:#67b7d1;--rule:#2b3038;--card:#1b1f26;--warn:#f59e6b;}}
:root[data-theme="dark"]{
  --bg:#14171c;--ink:#e6e4de;--ink2:#a8adb5;--ink3:#767c85;
  --accent:#67b7d1;--rule:#2b3038;--card:#1b1f26;--warn:#f59e6b;}
:root[data-theme="light"]{
  --bg:#fbfaf7;--ink:#20242a;--ink2:#5a616b;--ink3:#8a9099;
  --accent:#155e75;--rule:#e3e0d8;--card:#ffffff;--warn:#9a3412;}
body{background:var(--bg);color:var(--ink);
  font:16px/1.6 Georgia,'Times New Roman',serif;margin:0;}
main{max-width:65rem;margin:0 auto;padding:2.5rem 1.4rem 5rem;}
header .eyebrow{font:600 12px/1.4 system-ui,sans-serif;
  letter-spacing:.12em;text-transform:uppercase;color:var(--accent);}
h1{font-size:1.9rem;line-height:1.25;margin:.4rem 0 .6rem;
  text-wrap:balance;}
h2{font-size:1.25rem;margin:2.6rem 0 .7rem;padding-top:1.2rem;
  border-top:1px solid var(--rule);text-wrap:balance;}
p{margin:.7rem 0;} .meta{color:var(--ink2);font-size:.92rem;}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(13rem,1fr));
  gap:.7rem;margin:1.3rem 0;}
.tile{background:var(--card);border:1px solid var(--rule);
  border-radius:6px;padding:.75rem .9rem;}
.tile .v{font:600 1.25rem/1.3 system-ui,sans-serif;
  font-variant-numeric:tabular-nums;}
.tile .k{color:var(--ink2);font:12.5px/1.45 system-ui,sans-serif;
  margin-top:.15rem;}
figure{margin:1.4rem 0;}
.frame{background:#fff;border:1px solid var(--rule);border-radius:6px;
  padding:.5rem;overflow-x:auto;}
.frame img{max-width:100%;display:block;margin:0 auto;}
figcaption{color:var(--ink2);font-size:.88rem;margin-top:.45rem;}
.tblwrap{overflow-x:auto;margin:1rem 0;}
table{border-collapse:collapse;font-size:.92rem;min-width:34rem;
  font-family:system-ui,sans-serif;}
th,td{border:1px solid var(--rule);padding:.45rem .65rem;
  text-align:left;vertical-align:top;}
th{background:var(--card);}
td.num{font-variant-numeric:tabular-nums;text-align:right;}
code{font:.88em/1.4 ui-monospace,monospace;background:var(--card);
  border:1px solid var(--rule);border-radius:4px;padding:0 .25em;}
</style>
"""

body = f"""
<main>
<header>
  <div class="eyebrow">Quatrex · phonon-phonon SCBA · full campaign report (2026-07-10 → 07-17)</div>
  <h1>Phonon SCBA campaign: observables, convergence, and the kernel-band mechanism</h1>
  <p class="meta">Systems: CNT(3,3) at L = 2–10 cells, d5a SiNW. Post-refactor
  verification → observables pipeline → mixing-scheme study → measured Jacobians →
  synthetic grid study → the bubble kernel's inner-band causality defect, its fix
  (<code>sse_g_band = 2</code>), the exact-kernel length series → the
  exact-Jacobian Newton solver (bilinear JVP, recycled deflation) and the
  η = 0 suite (§6).</p>
  <div class="tiles">
    {tile("0.569 → 0.362", "", "r = J/J<sub>ball</sub>, CNT L2 → L7 on the exact kernel (all converged fixed points)")}
    {tile("−1.6·10³", "", "first-Born causality violation of the masked kernel on interior slabs (edge slabs exactly causal)")}
    {tile("311", "its", "L4 to tolerance on the exact kernel, plain linear α=0.2 (masked kernel: 1200+ and never)")}
    {tile("L7 | L8", "", "sharp stability boundary of the bubble-only model (soft-mode collapse beyond)")}
    {tile("5·10⁻⁵", "rel", "J<sub>L</sub> agreement across the 11 converged L2 schemes")}
    {tile("−4.33 / −5.07", "", "measured dominant Jacobian eigenvalue (L2 fixed point / L4 stall; IR-localized, damped-stable)")}
  </div>
</header>

<section>
  <h2>1 · The convergence story, resolved</h2>
  <p>Damped linear mixing converged at L2 (222 iterations) and L3 (347),
  entered a sawtooth limit cycle at L4 and diverged at L10 — and the mixer
  campaign showed no accelerator fixed it. Direct power-iteration
  measurements found real, negative, infrared-localised Jacobian spectra,
  damped-stable at every probed point (L2 fixed point −4.33/−3.94; L2
  Anderson stall −3.51/−3.30; L4 stall −5.07…−4.15), ruling out local
  linear instability. The resolution (§5): the bubble kernel was masked to
  the RGF block-tridiagonal band, injecting non-causal gain on interior
  slabs; the sawtooth was its gain-relaxation cycle. With the exact band,
  the length series converges through L7 under plain linear mixing.</p>
  {img("s1a_cnt_ladder", "Masked kernel: residual histories by device length under damped linear mixing.")}
  {img("s1e_jacobian", "Measured Jacobian eigenvalues (power iteration) vs the damping bound α = 2/(1+|λ|): every probed point is damped-stable.")}
  {img("s1b_l2_schemes", "Two-cell CNT: mixing-scheme comparison and the RRE parameter sweep (masked kernel).")}
  {img("s1c_d5a_schemes", "d5a SiNW (α = 0.1, broadening floor): scheme comparison — safeguarded Anderson wins there.")}
  {img("s1d_forensics", "Anderson forensics: least-squares conditioning varies over three decades; coefficient norms 2–20.")}
</section>

<section>
  <h2>2 · Observables (theory §Observables), all full runs</h2>
  {img("s2a_dos", "Density of states, SCBA vs ballistic, with sum rules.")}
  {img("s2b_transmission", "Transmission: Caroli vs effective.")}
  {img("s2c_heat_spectra", "Spectral heat current with the infrared plateau check.")}
  {img("s2d_teff_occ", "Effective temperature and occupation profiles.")}
  {img("s2e_msd", "Mean-square displacements.")}
  {img("s2f_ledger", "Conservation ledger: raw vs telescoped interface currents.")}
</section>

<section>
  <h2>3 · Cross-run comparisons</h2>
  {img("s3a_length", "Length series (masked kernel, superseded by §5 for L ≥ 3).")}
  {img("s3b_agreement", "Fixed-point agreement across converged schemes (relative error).")}
  {img("s3c_systems", "System comparison: CNT vs d5a.")}
</section>

<section>
  <h2>4 · Synthetic grid study (E1–E7)</h2>
  <p>A two-site chain with physical vertex scales isolates the grid
  mechanisms: iteration convergence is not grid convergence (O(Δω/Γ)
  linewidth errors against the self-consistent width); cold starts at
  η ≪ Δω are poisoned by the unresolved δ-seed (false convergence by
  residual swamping); sub-bin alignment moves the physics (25% swing);
  sharp–sharp flat-band channels are silently deleted below resolution and
  bistable above (contact-broadening continuation reaches the scattering
  branch; an η bootstrap cannot — no occupation); measured Arnoldi spectra
  give the general damped-stability criterion Re λ &lt; 1 (the sharp-pair
  scattering branch sits at λ = +1.98, unreachable by any damping).</p>
  {img("toy_f1_linewidth", "E1: converged linewidth vs grid; relative error collapses on Δω/Γ.")}
  {img("toy_f2_alignment", "E2: sub-bin pole position moves the converged linewidth and rate.")}
  {img("toy_f3_branches", "E5–E6: grid- and history-selected fixed points of the flat-band pair.")}
  {img("toy_f4_spectra", "E7: measured Jacobian spectra; Re λ = 1 separates damping-reachable from unreachable.")}
</section>

<section>
  <h2>5 · The kernel-band mechanism and the exact-kernel campaign</h2>
  <p>The root cause of §1: the bubble contraction masked the kernel
  G⊗G to the RGF block-tridiagonal band. A masked positive-semidefinite
  form is not positive-semidefinite (Schur product theorem; the
  tridiagonal-ones mask is PSD only up to two slabs), so every interior
  slab of a device with three or more cells acquired non-causal
  <em>gain</em> components of Σ — already in the first Born term (min
  eigenvalue −1.6·10³, edge slabs exactly causal, invariant under
  window/η/grid parity), breathing ×4 with the L4 sawtooth. The fix
  (<code>sse_g_band = 2</code>): one extra selected off-diagonal of
  G<sup>≶</sup> from the recursive solver completes the kernel for
  nearest-neighbour vertices — diagonal Σ blocks become exact, validated
  bit-identical on the legacy path, PSD on every slab, at ×1.53 kernel
  cost (production iteration 24/42/48 s at L5/6/7).</p>
  {img("w3_gband_mechanism", "First Born: interior slabs non-causal under the masked kernel; band 2 restores exact positivity; the interior correction is O(1).")}
  {img("w2_l4_legacy_vs_g2", "CNT L4, linear mixing: the 500-iteration sawtooth is the masked kernel's gain-relaxation cycle; the exact kernel converges in 311 iterations.")}
  {img("w1_g2_length_series", "Exact-kernel length series: L2–L7 all converge under plain linear mixing; r(L) decreases smoothly; L8+ is lattice-unstable.")}
  {img("w4_ne_scan", "The masked kernel's grid-density lottery at L4 — closed by the exact kernel.")}
  <p><b>The remaining failure is the model, not the numerics.</b> At eight
  and ten cells the exact-kernel iteration stays causal on every slab but
  the accumulated Re Σ<sup>R</sup> softens the infrared modes through
  ω² = 0 (minimum eigenvalue of h₀₀ + Re Σ<sup>R</sup> reaches
  −92 THz² interior, −282 edge): a dynamically unstable effective
  lattice. Every iteration-level stabiliser fails (broadening floor,
  vertex ramp, dressed contacts, static tadpole); the missing counterterm
  is the quartic (FC4) loop, which the bubble-only production model
  omits. The stability boundary is sharp: L7 converges with no anomalous
  slowdown, L8 diverges at iteration 63.</p>
  {img("w5_l10_forensics", "L10 forensics: the IR bins seed the cascade; all stabilisers fail; renormalised modes pushed through ω² = 0.")}
  {img("w7_g2_mixers", "Exact kernel, L4: reduced-rank extrapolation still stagnates where plain linear converges.")}
  {img("w6_contact_model", "Contact models (exact kernel, L4): GW-ordering scattering-dressed contacts are stable and shift J by 25–30% against ideal reservoirs.")}
  <div class="tblwrap"><table>
    <tr><th>L</th><th>outcome (exact kernel, linear α=0.2)</th>
        <th>iterations</th><th>J (W)</th><th>r = J/J<sub>ball</sub></th></tr>
    <tr><td>2</td><td>converged</td><td class="num">222</td>
        <td class="num">44.2</td><td class="num">0.569</td></tr>
    <tr><td>3</td><td>converged</td><td class="num">209</td>
        <td class="num">38.9</td><td class="num">0.500</td></tr>
    <tr><td>4</td><td>converged</td><td class="num">311</td>
        <td class="num">35.2</td><td class="num">0.453</td></tr>
    <tr><td>5</td><td>converged</td><td class="num">304</td>
        <td class="num">32.4</td><td class="num">0.417</td></tr>
    <tr><td>6</td><td>converged</td><td class="num">241</td>
        <td class="num">30.7</td><td class="num">0.396</td></tr>
    <tr><td>7</td><td>converged</td><td class="num">314</td>
        <td class="num">28.1</td><td class="num">0.362</td></tr>
    <tr><td>8</td><td>diverged (soft-mode collapse)</td><td class="num">63</td>
        <td class="num">—</td><td class="num">—</td></tr>
    <tr><td>10</td><td>diverged (soft-mode collapse)</td><td class="num">119</td>
        <td class="num">—</td><td class="num">—</td></tr>
  </table></div>
</section>

<section>
  <h2>6 · The exact-Jacobian Newton solver and the η = 0 suite</h2>
  <p>The SCBA map's nonlinearity terminates: the bubble is exactly
  quadratic in G, so its Fréchet derivative is the mixed-leg
  (cut-line, 2PI-kernel) contraction B(δG,G)+B(G,δG), evaluated by
  <code>compute_linearized</code> through the unmodified production
  pipeline — no differencing parameter, no subtraction of large terms.
  Composed with the frozen-G dense Dyson identities this gives an exact
  Jacobian–vector product (<code>mixing_method = "newton"</code>),
  validated in layers: bilinear ≡ polarisation 8.7·10⁻¹⁶ (5.7·10⁻¹⁶ on
  10⁻⁸-small directions where polarisation degrades to 4.4·10⁻⁸);
  Dyson vs the recursive solver on its skew-hermitian invariant
  subspace 2.3·10⁻¹⁰; composed JVP vs finite differences of the full
  production iteration 1.6·10⁻⁹; the two routes mutually 1.3·10⁻¹⁴;
  frozen-G reconstruction self-check 4·10⁻¹⁰ at every Newton step.</p>
  <p><b>Deflation preconditioner, measured in both regimes.</b>
  Harmonic-Ritz pairs recycled from each step's Arnoldi relation (exact
  images, zero extra kernel cost, Ritz-residual filtered, accumulated
  across steps, and gated on the step size — a stale basis measurably
  poisons the inner solve during large steps). In the small-step
  endgame the bare inner GMRES saturates its 30-vector cap without
  meeting the forcing tolerance while the deflated solve meets it at
  3–10 vectors; over matched 35-iteration budgets: 295 vs 642 exact
  JVPs to the same residual. The fresh-basis variant (the literal
  low-rank Schur surrogate) halves the inner dimension but pays exactly
  that in setup — net neutral.</p>
  {img("pc_bench", "Deflation benchmark (CNT L4 from the archived stall state): the extended bare solver (black) pins at the 30-vector cap; guarded recycling (green) runs the endgame at m = 3–10. Right: residual against cumulative exact JVPs — 295 vs 642.")}
  <p><b>Two-phase deep convergence (L4, cold start).</b> The mid-field
  (rel. residual 10⁻¹–10⁻³) is strongly curved — trust-capped Newton
  advances slower than plain damped iteration there, whatever the inner
  solve costs — so the switch belongs below it: ~306 damped sweeps to
  10⁻³, then <b>three</b> Newton steps: 1.2·10⁻³ → 1.3·10⁻⁴ →
  6.7·10⁻⁷ → 1.9·10⁻¹⁰ (inner m = 3/17/30). Same fixed point as the
  linear baseline (J = 35.21, 0.03%), seven orders deeper, at
  essentially the cost linear paid to reach 10⁻³. On L8 (no stable
  fixed point) the solver diagnoses instead of diverging: six steps
  with the inner iteration retaining 62–79% at the full Krylov budget
  — the near-singular operator — while the trust region keeps
  iterates bounded.</p>
  <p><b>The η = 0 suite (L3).</b> With no artificial broadening
  anywhere (device η = 0 exactly, bare spectral open boundaries),
  plain damped iteration converges in the same 210 iterations to the
  same fixed point as η = 10⁻¹² — heat currents agree to 2·10⁻¹³: the
  production numbers are genuine η = 0 results. Σ-dressed leads
  (GW ordering) reach a 2·10⁻⁵ residual but carry a 1.2% lead
  imbalance and 13% telescoped-current spread — the dressed map does
  not conserve at the bare-lead level. Masking three-phonon scattering
  below 2 THz (the grid itself must stay zero-anchored for the
  convolution arithmetic) converges tightly at r = 0.809 vs the full
  kernel's 0.500: <b>over 60% of the anharmonic resistance at 300 K
  flows through channels involving a sub-2-THz phonon</b> — the
  infrared bins are the dominant resistive phase space, not a
  numerical nuisance.</p>
  <div class="tblwrap"><table>
    <tr><th>run</th><th>setting</th><th>outcome</th></tr>
    <tr><td>cnt-L3-eta0</td><td>η = 0 exact, bare spectral OBC, linear</td>
        <td>converged, ≡ η=10⁻¹² fixed point to 1.6·10⁻¹³ (r = 0.500)</td></tr>
    <tr><td>cnt-L3-eta0-scat</td><td>+ Σ-dressed leads</td>
        <td>residual 2·10⁻⁵; 1.2% lead imbalance, 13% internal spread</td></tr>
    <tr><td>cnt-L3-eta0-mask</td><td>+ scattering masked &lt; 2 THz</td>
        <td>converged, tightly conserving, r = 0.809</td></tr>
    <tr><td>newton-L4-v2</td><td>two-phase, switch at rel 10⁻³</td>
        <td>309 its → residual 1.9·10⁻¹⁰, J ≡ linear (0.03%)</td></tr>
    <tr><td>newton-L8</td><td>Newton on the unstable model</td>
        <td>bounded; inner solve retains 62–79% at the cap (diagnosis)</td></tr>
    <tr><td>newton-d5a-chain</td><td>T-continuation 150→200 K, deflated Newton</td>
        <td>in flight (rung 150; marginal spectrum, m = 40 inner solves)</td></tr>
  </table></div>
</section>

<section>
  <h2>Verdicts</h2>
  <div class="tblwrap"><table>
    <tr><th>question</th><th>answer</th></tr>
    <tr><td>Why did no mixing α work at L4/L10?</td>
        <td>The masked bubble kernel injected non-causal gain on interior
        slabs (measured in the first Born term; Schur product argument).
        With the exact band (<code>sse_g_band = 2</code>) L4 converges in
        311 iterations under plain linear mixing and the grid-density
        lottery closes. L8+ fails through genuine soft-mode lattice
        instability of the bubble-only model (missing FC4 loop
        counterterm) — a model limit, not a scheme limit.</td></tr>
    <tr><td>Convergent scheme per system?</td>
        <td>CNT on the exact kernel: plain linear α=0.2 through L7; RRE
        stagnates at L4 (both build dampings). d5a (masked-kernel era):
        safeguarded Anderson in 49 iterations. Rerun of d5a on the exact
        kernel pending.</td></tr>
    <tr><td>Do different schemes give the same physics?</td>
        <td>Across the 11 converged (masked-kernel) L2 schemes, J_L agrees
        to 5·10⁻⁵ relative; L2 is bit-identical under the exact kernel (no
        interior slab). For L ≥ 3 the masked-kernel physics carries an O(1)
        interior self-energy error and is superseded by §5.</td></tr>
    <tr><td>Contact model</td>
        <td>Scattering-dressed (GW-ordering) contacts are stable on the
        exact kernel and shift J by 25–30% at L4 — a leading-order model
        choice for short devices; at η = 0 (L3) they carry a 1.2% lead
        imbalance and a 13% telescoped-current spread the bare-lead map
        does not have.</td></tr>
    <tr><td>Deep convergence?</td>
        <td>Two-phase: damped iteration through the curved mid-field to
        rel 10⁻³, then exact-Jacobian Newton — three steps to 1.9·10⁻¹⁰
        at L4, same fixed point, same total cost as linear-to-10⁻³.
        Recycled harmonic-Ritz deflation cuts the endgame inner solve
        3–10× per step (2.2× total), gated against staleness.</td></tr>
    <tr><td>Is η = 10⁻¹² a real η = 0?</td>
        <td>Yes — exact-η=0 reproduces it to 1.6·10⁻¹³ (identical
        210-iteration trajectory, L3, bare spectral OBC).</td></tr>
    <tr><td>Can the infrared bins be excluded?</td>
        <td>No — masking scattering below 2 THz raises r from 0.500 to
        0.809: the sub-2-THz channels carry ~62% of the anharmonic
        resistance. (A grid not anchored at 0 is refused by the kernel:
        the FFT convolution and bosonic fold require zero anchoring.)</td></tr>
  </table></div>
  <p class="meta">Provenance: engine snapshots (run*.npz), per-run logs and
  Σ snapshots mirrored under <code>phonon/studies/out/anderson_test/</code>,
  <code>phonon/studies/out/newton_pc_bench/</code>
  and <code>cluster/</code> on the laptop; figures from
  <code>phonon/scripts/figures/gband_campaign_figs.py</code>,
  <code>pc_bench_figs.py</code> and
  <code>_campaign_figures.py</code>; synthetic study
  <code>phonon/studies/_toy_grid_*.py</code>; JVP validation
  <code>phonon/studies/_jvp_validate.py</code>, <code>_newton_ab.py</code>,
  <code>phonon/scripts/verify/newton_unit.py</code>. Document: theory
  <code>40_scba</code> (exact linearisation sub:exact_jvp +
  eq:jvp_woodbury), results <code>60_eta0</code> (ssec:res_gband,
  ssec:res_newton), <code>30_cnt</code> (exact-kernel series incl. L3),
  <code>90_scaling</code> (band cost). The d5a temperature-continuation
  chain is in flight.</p>
</section>
</main>
"""

out = REP / "campaign_report.html"
out.write_text(HEAD + body)
print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)")
