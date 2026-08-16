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

The `mos2f3*` campaign: 43 ledger rows (39 distinct configs on disk), every one
at `interaction_cutoff = 10 A` on a device longer than the cutoff, every one
divergent. Results §7.4 states this explicitly rather than omitting it. It no
longer quotes node-hours: Paul asked for the lesson without the accounting, and
the figures that were there had been wrong in two ways -- "roughly 304 committed
at submission" was a post-reset running total (the committed sum over all 222
ledger rows is 545.83 nh, across three CSCS reconciliations), and "39 jobs" was
a config count copied from `mos2_kappa_z_ladder.md`.

Two `mos2f3*` sub-runs survive as data rather than as results: `mos2f3nu` feeds
the grid-comb figure and `mos2f3b2` the harmonic ladder.

## Appendix removed entirely (2026-08-16, second pass)

`src/appendices/production_coupled_q.tex` (260 lines, plus
`prod_qfilm_qconv` and `prod_qfilm_conservation`). Three independent
reasons, any one of which would have been enough:

* its central open issue -- "the production film is numerically stable only
  for eta >= 0.4 THz; at eta <= 0.2 the Dyson solve produces NaN" -- is
  **contradicted** by the converged unbroadened coupled-q film now in
  results section 2.2 (70 iterations at three cells, 52 at eight). Keeping
  it would have put a false statement in the document;
* its CNT length ladder ran at eta = 0.45 THz, did not settle to a fixed
  point ("the final iterate diverges and is discarded"), and its snapshots
  were purged in commit 843c3069, so its ratios are unreproducible;
* its own text says the contribution is "software and verification, not new
  physics", and that verification is now results section 7.1 (dense parity
  across 144 cases, 8e-13 CPU/GPU agreement, rank-invariance at 2e-15).

Nothing outside the file referenced any of its ten labels.

## Broadening evidence consolidated

The section "The conductance ratio requires eta -> 0 extrapolation" moved
out of the conservation appendix into `broadening.tex`, with its table and
figure. It is the measured core of that appendix's argument: at matched
broadening the lead balance sits flat at 1e-5 while the conductance ratio
swings 13 points, so the conservation diagnostic does not see the bias.

After this pass the broadening parameter appears in exactly four files:
`broadening.tex` (its subject), `conservation.tex` (the mechanism -- what a
damping term does to the conservation identity, retitled so the table of
contents no longer advertises it), and `phonon_solver.tex` /
`finite_analysis.tex` (a documented package API parameter, each now stating
it is zero in every calculation reported). `quatrex_doc.tex`'s worked config
example had `phonon.eta = 0.45` and now has `0`.

## Run families that were unrecorded, and their disposition (2026-08-16)

An audit of the ledger against `document/src/results/` found 39 jobs whose
outcome appeared in neither the report nor this file. Their disposition:

### Now in the report

| family | what it established | where |
|---|---|---|
| `tap*` (7 jobs, `tapT12b` on disk) | The triangular `interaction_cutoff_taper` is PSD at every radius, drives the worst occupation from -1.00 on 84.5 % of live bins to +2e-08 on none, **and the run still diverges**. The R = 30 control is decisive: applied to a support that already covers the device, i.e. to a converging model, the taper breaks it. Definiteness is necessary and not sufficient | results §3.1 |
| `gl_*` (12 jobs) | The primary grid is ~7x finer than the observable needs; the auxiliary bubble grid is the expensive axis and is not converged (+10.20/+7.74/+4.36 % at `aux_dw` = 0.02/0.01/0.005); `sse_aux_restrict` moves the answer several per cent. Every ladder run in the report has the aux grid off | results §3.4 |
| `srcoarse` / `speccoarse` (2 jobs) | `si4x2` runs clean under the spectral OBC at `ne = 141`, so the fine-grid divergence belongs to the grid and not the 2x2 blocking; Sancho-Rubio returns NaN at both spacings and so cannot be the independent arm | results §4.3 |
| `mos2sood*`, `mos2L*n4{ball,scba}` (8 jobs) | Bed selection for the pole sector. Both reasons the layered material cannot test it -- the q-resolved beds run `retarded_method = "half"` where the sector needs `"fft"`, and their grids are fine enough that nothing is unresolved by construction | results §5.4 |

### Probes with no separable result

Recorded here rather than in the report, because each answered an engineering
question inside another campaign and none is a finding on its own.

- `mosreach*`, `mosr6b*` (4 jobs) -- `QX_GBAND` spatial-reach probes.
- `mos2f4dense` (1 job) -- 4-cell dense-mask film; its row is in the
  `mos2_kappa_z_ladder.md` mask audit.
- `siDENSE`, `sireblk{,2,3}`, `si8kprobe` (5 jobs) -- dense/reblock/8k probes.
- `pconv2`, `pnf` (2 jobs, 4.00 nh) -- **cancelled while pending and never
  ran**; they are charged in the ledger because it charges submitted walltime.

### Report claims whose raw data is no longer on disk

Not a removal, but the provenance should be on the record:

- results §7.3 (the memory scaling) rests entirely on five probes --
  `lsM4probe`, `lsM6probe`, `lsM6probe16`, `lsM4q2`, `lsM6q2` -- whose run dirs
  are gone. The numbers survive only in `sse_memory_scaling.md`.
- `cvM6` (24 nh) is absent, but `cvM6b`, the run the ladder actually uses, is
  present.
- `sires4001`, `siladder`, `obcprobe`/`obc4k`/`obc8k2n`/`obcsr`/`obcfull`,
  `cluster/bench-decomp`, `cluster/bench-legacy`, `cluster/l4bench`,
  `cluster/filmq` are all referenced by a number in the report and absent from
  disk; each figure generator carries the numbers as literals with the source
  named in its module docstring.
