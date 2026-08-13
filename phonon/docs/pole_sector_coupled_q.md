# Coupled-q pole sector: design

2026-08-13. Companion to `pole_sector_state_and_next_steps.md`. What it takes
to run the pole sector on a q-resolved device, and why the production route is
much cheaper than the current refusal implies.

---

## 0. The refusal, and the half of it that is wrong

`pole_sector.set_operator_context` raises for `nq != 1`:

    pole_sector: coupled-q (N q-points) is not wired yet;
    the pole set is per-q and the vertex fold has to follow it.

The first clause is right: the pole set is per-q. The second is right only for
the routes that add analytic sectors back beside the ring, and the production
route is not one of them.

`leg = "congruence"` adds **no** sector. Its whole action is to modify the leg
handed to the ring, through `set_pole_channel`, and the ring consumes that as a
plain elementwise subtract on the full stack in the `"nnz"` state
(`sse_phonon_phonon.py`):

    gl_in = gl_in - p_l
    gg_in = gg_in - p_g

`p_l` carries whatever shape `g_lesser.data` has, q axis included. The q
convolution happens afterwards, inside the ring that already performs it. So on
the production route the pole sector is **per-q and uncoupled**: solve

    M_q(z) = z^2 I - D(q) - Sigma^R_c,q(z) - Sigma^R_s,q(z)

for each q, build that q's leg correction, write it into that q's slice, and the
existing fold does the rest.

The vertex fold is needed only where sectors are restored analytically, because
there

    Sigma_q(w) = sum_{q'} B[ G_{q'}, G_{q-q'} ]

pairs pole sets belonging to **different** q. That is `congruence_analytic` and
the local finite-cell route, and they stay refused.

## 0.1 What is already in place

* `_pole_blocks(matrix, index_slice=...)` already takes a stack slice, so `D(q)`
  costs nothing new.
* `solver._census_over_q` already walks the q axis with exactly the slicing
  Stage 1 needs — per-q `delta`, per-q `D(q)`, per-q contact blocks — and is
  covered by `test_census_walks_every_q_and_survives_one_failing`.
* The vertex is q-independent as the code stands: `phi_blocks = vertices[(0,0)]`.

Stage 1 is therefore that loop, plus the leg.

---

## 1. Stage 1 — per-q solve, per-q leg

### 1.1 One `PoleSector` per q

The tracker, the promoted set, the epoch counter and the predictor history are
all per-q. A mode at `q_1` has no relation to one at `q_2`, so a shared tracker
would match them across q, and the subspace-angle test would then either fuse
two unrelated modes or churn membership every iteration — which is exactly what
the hysteresis exists to prevent.

Hold `list[PoleSector]`, built lazily, indexed row-major over `nk`. Memory per
instance is small (the state holds pattern-sized arrays only for promoted
clusters); the cost is the solve, not the object.

### 1.2 `_update_pole_sector` loops q

Reuse the slicing `_census_over_q` proves:

    delta[:, iq, :]
    _pole_blocks(dynamical_matrix, index_slice=idx)
    obc_blocks.retarded[0][idx],  obc_blocks.retarded[-1][idx]

`set_operator_context` keeps its `nq != 1` guard untouched and never sees a q
axis — it receives a slice. That way the guard still protects the allocating
path for the routes that genuinely cannot do q.

### 1.3 `_build_pole_keldysh` per q

It currently flattens with `reshape(n_freq, -1)`. It becomes a loop writing each
q's correction into `state.g_pp_*[:, iq, :]`. Everything inside is already
per-q and needs no change in kind: the projected source `V_q^H Sigma_tot,q V_q`,
the two `apply_sparse` contractions, `G^R_q(w_k)`, and the cell average.

The one thing to get right is that the **contact corners are per-q too**
(`obc_blocks.lesser[0]` etc. carry the q axis). Omitting them is not a small
error — they are what drives the device, and without them `G^R Sigma G^A` is not
`G^<`.

### 1.4 Cost, and the knob for it

Stage 1 is `nq x` the single-q solve. The q-resolved beds in the tree are MoS2
`kpoint_grid = [5,5,1]` (25 q) and Si `[1,9,9]` (81 q). Two knobs:

* `pole_sector.q_stride` (default 1) — sample the q axis for a survey.
* `pole_sector.q_max` — hard cap on how many q are solved.

Skipping is safe by construction: an unpromoted q keeps its untouched leg, and
the route is bit-identical to pole-off there. Skipped q must be **reported**,
not silently dropped, or a survey reads as a full run.

### 1.5 Reporting

Per-q promotion yield and census, plus one aggregate line. A q that promotes
nothing has to be visible; the failure mode to avoid is a q axis where 24 of 25
are empty and the log only shows the one that is not.

---

## 2. Stage 2 — the fold, only if Stage 1 earns it

Opening `congruence_analytic` and the local route to q means pairing
`(q', q - q')` pole sets inside the sector kernels: a redesign of
`pf_self_energy`, `pf_mixed_self_energy`, and `pole_local.correct_spectrum`'s
term list, with the q fold and the bosonic mirror interacting.

Do not start it before Stage 1 has measured whether any q carries narrow modes.
On CNT the answer at every q available was no (`cnt_observations.md` Sec. 6),
and building the fold for a device with nothing to extract repeats the mistake
this whole investigation has been correcting.

---

## 3. Verification

| check | what it catches |
|---|---|
| Gate 0 on a q bed: empty pole window, `rtol = atol = 0` vs pole-off | the q loop perturbing a run it should not touch |
| `nq = 1` regression, byte-identical to today on CNT | the loop changing the single-q path |
| synthetic 2-q test, deliberately different pole sets per q | one q's poles leaking into another's leg |
| `P_in` vs `P_out` and lead balance on a q bed, sector on | promoting at some q and not others breaking conservation |
| full suite at 440 phonon / 560 total, 5 known failures | everything else |

The 2-q test is the important one and has no analogue today. The census test
(`test_census_walks_every_q_and_survives_one_failing`) already asserts each q
receives its own distinct slice; the leg test must assert the same for what
comes back out.

---

## 4. What this does not do

It does not make the pole sector useful on a q bed. It makes it *runnable*
there, so the census can say whether it would be. Those are different claims and
the second one is not yet supported by any measurement on any bed.

Note also that no run in the recorded history has used `q_comm_size > 1` — 94 of
94 are `qcs=1`, so the q axis has only ever been carried in the stack. Stage 1
inherits that: it loops the stack's q axis on one rank. A q-distributed pole
solve is a separate piece of work and is not planned here.
