# What came out of the report, and why

Companion to the 2026-08-16 restructure of `document/`. Nothing here is a
judgement that the work was wasted; it is a record of what is no longer
presented as a result, so that nobody re-derives it and so that anything
worth reviving can be found.

Everything below was in `src/results/` before the restructure and is not in
it after. Recoverable from git at `document/` commit `31fa30c^`.

## Superseded by a later measurement

| what | where it was | superseded by |
|---|---|---|
| The MoS2 "unstable spiral" and its soft-mode-feedback reading (116 lines) plus the 128 lines refuting it | `75_mos2.tex` §spiral | The cutoff/positivity finding. Only the correction survives, as results §3.1 |
| The claim that no converged eta=0 film SCBA exists | `75_mos2.tex` §ladder | `mos2L2conv` (29 it) and `mos2L4conv` (30 it), then the whole cvM ladder |
| `tab:res_asr_compare`, the projected-vs-raw FC3 table whose own caption began "Retracted historical record" | `20_nanowires.tex` | Removal commit 268b3174: the projection is leg-asymmetric, breaks vertex S3, suppresses linewidths 4-5x vs phono3py. The lesson survives as results §1.3 |
| The greyed retracted finite-broadening series in the Si film figure | `40_film_hetero.tex`, `si_film_vs_guo.pdf` | The eta=0 coupled-q fixed points. The figure no longer draws them; the broadening evidence is the new appendix |
| The IR taper presented as the ingredient that turns a residual floor into a fixed point, and `G0 = 17.4` as the reference value | `60_eta0.tex` | `spectral_deformation_audit.md` (2026-07-06): the taper is unphysical, raises G by 33% against the grid-converged bare 12.86, and breaks the interior ledger by ~5%. Results §4.2 now reports it as the negative result |
| The masked-kernel three- and four-cell CNT transport values (0.501, 0.480) | `30_cnt.tex`, `64_gband.tex` | Exact-kernel values 0.500 and 0.453 |
| All pre-2026-06-12 anharmonic cnt33/cnt80/sinw/srtio3 numbers | several | The ASR-projection removal (4-5x linewidth suppression), the SSE sign fix, and the Sigma dw misscaling of `solver_audit.md` #9 |
| The `÷4` self-energy prefactor | archive F23/F24 | F24-CORRECTION retracts it; `vertex_normalisation.md` then settles the prefactor exactly (1.1e-13 vs phono3py) |

## Removed as broadened data

Every finite-broadening transport number is out of the main chapters. The
evidence that broadening biases the answer is now `src/appendices/broadening.tex`
and is the argument for using none:

- the cnt33 broadening sweep 0.30-0.90 THz (ratio 0.79 -> 0.92) and its
  extrapolation to 0.68-0.74 against the directly converged 0.574;
- the SiNW ballistic length decay, which is coherence attenuation set by the
  broadening, not a mean free path;
- the Si film 45-62% three-phonon reduction against the reference's 16.4%,
  which resolved to 16.0-21.4% once the broadening was removed.

`50_static_scp.tex`'s loop/tadpole magnitudes were also measured on the dense
solver at a finite broadening. The ordering survives as results §5.2 with a
`\todo` on it, because the per-system conclusion does not depend on the
broadening but the absolute values do.

## Removed as provenance rather than result

- `92_gpu.tex`'s kernel-optimisation diary: eleven kernel organisations, two
  operation-level rewrites, cuTENSOR, grouped tall-GEMM and cuTile, each held
  to a pre-registered bar and parked. One sentence in results §7.2 now.
- `85_decomposed_sse.tex` L23-41, the min-image aliasing archaeology.
- `62_solver_campaign.tex` as a sweep log. Its transferable content -- that the
  mixer ranking inverts between systems and that the resonance-gain model
  underpredicts the measured Jacobian by 2-3.5x and inverts the ranking it was
  meant to explain -- is results §4.5.
- `95_summary.tex`, whose table carried rows marked "in flight" and "campaign
  not completed", and whose MoS2 row (R = 6.8, 68%) contradicted its own
  section (7.2, 72%). Folded into the conclusion, corrected.

## Not presented, and named as such

The `mos2f3*` campaign: 39 jobs, close to 190 committed node-hours, every one
at `interaction_cutoff = 10 A` on a device longer than the cutoff, every one
divergent. Results §7.4 states this explicitly rather than omitting it, because
the ratio of that number to the 13.5 node-hours of the cutoff ladder that
diagnosed it is the useful part.
