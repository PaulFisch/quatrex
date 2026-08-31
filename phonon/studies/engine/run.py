"""Production phonon-transport SCBA driver + snapshot.

  QX_CONFIG (required, toml path), QX_NPZ (snapshot out, default <dir>/run.npz),
Run: ``mpirun -np N python run.py`` (config via QX_CONFIG).
"""
import os
import traceback
from pathlib import Path

import numpy as np

from env_aliases import (
    best_checkpoint_stride,
    normalise_env,
    sigma_restart_terms,
    validate_restartable_env,
)

from quatrex.core.config import parse_config, setup_context
from qttools import xp
from qttools.comm import comm as ranks
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host

normalise_env(os.environ)
validate_restartable_env(os.environ)
_BEST_LIVE_STRIDE = best_checkpoint_stride(os.environ)

CFG = os.environ["QX_CONFIG"]
cfg = parse_config(CFG)

# --- env overrides (TOML is the base) -----------------------------------
if os.environ.get("QX_MIX"):      cfg.scba.mixing_factor = float(os.environ["QX_MIX"])
if os.environ.get("QX_MIXMETHOD"):cfg.scba.mixing_method = os.environ["QX_MIXMETHOD"]
if os.environ.get("QX_ADEPTH"):   cfg.scba.anderson_depth = int(os.environ["QX_ADEPTH"])
if os.environ.get("QX_ADPERIOD"): cfg.scba.anderson_period = int(os.environ["QX_ADPERIOD"])
if os.environ.get("QX_ADWARMUP"): cfg.scba.anderson_warmup_iters = int(os.environ["QX_ADWARMUP"])
if os.environ.get("QX_ADRESTART"):cfg.scba.anderson_restart = int(os.environ["QX_ADRESTART"])
if os.environ.get("QX_ADRIDGE"):  cfg.scba.anderson_ridge = float(os.environ["QX_ADRIDGE"])
if os.environ.get("QX_ADSTEPCAP"):cfg.scba.anderson_step_cap = float(os.environ["QX_ADSTEPCAP"])
if os.environ.get("QX_ADREVERT"): cfg.scba.anderson_revert_factor = float(os.environ["QX_ADREVERT"])
if os.environ.get("QX_ADSTAG"):
    cfg.scba.anderson_stagnation_restart = int(os.environ["QX_ADSTAG"])
if os.environ.get("QX_MIXDIAG"):
    cfg.scba.mixer_diagnostics = bool(int(os.environ["QX_MIXDIAG"]))
if os.environ.get("QX_MAXIT"):    cfg.scba.max_iterations = int(os.environ["QX_MAXIT"])
if os.environ.get("QX_MINIT"):    cfg.scba.min_iterations = int(os.environ["QX_MINIT"])
if os.environ.get("QX_NE"):       cfg.electron.energy_window_num = int(os.environ["QX_NE"])
# The 3-phonon bubble has support to TWICE the band top, so a grid that
# stops at the band top truncates the Kramers-Kronig integral for Re Sigma^R
# and the solver warns. Extending the window is the direct test of that.
if os.environ.get("QX_WMAX"):     cfg.electron.energy_window_max = float(os.environ["QX_WMAX"])
if os.environ.get("QX_RETARDED"): cfg.phonon.retarded_method = os.environ["QX_RETARDED"]
if os.environ.get("QX_FC3"):      cfg.phonon.fc3_path = os.environ["QX_FC3"]
if os.environ.get("QX_DECOMPOSED"):
    # Study-only switch from a legacy dense q-fold artifact to the equivalent
    # production tensor factors.  The two inputs are mutually exclusive in
    # SigmaPhononPhonon, so clear the inherited dense path explicitly.
    cfg.phonon.decomposed_vertices_path = os.environ["QX_DECOMPOSED"]
    cfg.phonon.qfold_path = None
if os.environ.get("QX_ALGORITHM"):      cfg.phonon.solver.algorithm = os.environ["QX_ALGORITHM"]
if os.environ.get("QX_SIGMATOL"): cfg.phonon.sigma_convergence_tol = float(os.environ["QX_SIGMATOL"])
if os.environ.get("QX_HEATTOL"):  cfg.phonon.heat_flow_conservation_tol = float(os.environ["QX_HEATTOL"])
if os.environ.get("QX_OBC_MEMO"): cfg.phonon.obc.memoizer.mode = os.environ["QX_OBC_MEMO"]
if os.environ.get("QX_OBC_ALG"): cfg.phonon.obc.algorithm = os.environ["QX_OBC_ALG"]
if os.environ.get("QX_NEVP"): cfg.phonon.obc.nevp_solver = os.environ["QX_NEVP"]
if os.environ.get("QX_GBAND"):    cfg.phonon.sse_g_band = int(os.environ["QX_GBAND"])
if os.environ.get("QX_MICRO_DOF"):
    cfg.phonon.sse_microblock_dof = int(os.environ["QX_MICRO_DOF"])
if os.environ.get("QX_MICRO_GBAND"):
    cfg.phonon.sse_microblock_g_band = int(os.environ["QX_MICRO_GBAND"])
if os.environ.get("QX_VERTEX_RANK"):
    cfg.phonon.sse_vertex_rank = int(os.environ["QX_VERTEX_RANK"])
if os.environ.get("QX_DECOMPOSED_KERNEL"):
    cfg.phonon.decomposed_kernel = os.environ["QX_DECOMPOSED_KERNEL"]
if os.environ.get("QX_SCATCONTACTS"): cfg.phonon.obc_scattering_contacts = bool(int(os.environ["QX_SCATCONTACTS"]))
if os.environ.get("QX_BBCHECK"):  cfg.phonon.bubble_balance_check = bool(int(os.environ["QX_BBCHECK"]))
if os.environ.get("QX_BCS"):      cfg.compute.comm.block_comm_size = int(os.environ["QX_BCS"])
if os.environ.get("QX_QCS"):      cfg.compute.comm.q_comm_size = int(os.environ["QX_QCS"])
if os.environ.get("QX_COMM_BACKEND"):
    # One backend for every per-op comm selector. Pydantic validates the
    # value; comm.configure rejects backend/array-module mismatches.
    _cb = os.environ["QX_COMM_BACKEND"]
    for _f in type(cfg.compute.comm).model_fields:
        if _f.endswith(("all_to_all", "all_gather", "all_reduce",
                        "bcast", "send_recv")):
            setattr(cfg.compute.comm, _f, _cb)
if os.environ.get("QX_TLEFT"):    cfg.phonon.left_temperature = float(os.environ["QX_TLEFT"])
if os.environ.get("QX_TRIGHT"):   cfg.phonon.right_temperature = float(os.environ["QX_TRIGHT"])
# Experimental fixed-functional root-finder controls.  These are study-driver
# overrides, analogous to QX_ADEPTH, and make trust/ridge sweeps attributable
# in the printed QX_* provenance instead of requiring edited input TOMLs.
if os.environ.get("QX_RRE_CYCLE"):
    cfg.scba.experimental_mixer.rre_cycle = int(os.environ["QX_RRE_CYCLE"])
if os.environ.get("QX_RRE_RIDGE"):
    cfg.scba.experimental_mixer.rre_ridge = float(os.environ["QX_RRE_RIDGE"])
if os.environ.get("QX_BROYDEN_WARMUP"):
    cfg.scba.experimental_mixer.broyden_warmup_iters = int(
        os.environ["QX_BROYDEN_WARMUP"])
if os.environ.get("QX_BROYDEN_RIDGE"):
    cfg.scba.experimental_mixer.broyden_ridge = float(
        os.environ["QX_BROYDEN_RIDGE"])
if os.environ.get("QX_BROYDEN_TRUST"):
    cfg.scba.experimental_mixer.broyden_trust = float(
        os.environ["QX_BROYDEN_TRUST"])
if os.environ.get("QX_RPM_SUBSPACE"):
    cfg.scba.experimental_mixer.rpm_max_subspace = int(
        os.environ["QX_RPM_SUBSPACE"])
if os.environ.get("QX_JFNK_WARMUP"):
    cfg.scba.experimental_mixer.jfnk_warmup_iters = int(
        os.environ["QX_JFNK_WARMUP"])
if os.environ.get("QX_JFNK_KRYLOV"):
    cfg.scba.experimental_mixer.jfnk_max_krylov = int(
        os.environ["QX_JFNK_KRYLOV"])
if os.environ.get("QX_JFNK_INNER"):
    cfg.scba.experimental_mixer.jfnk_inner_tol = float(
        os.environ["QX_JFNK_INNER"])
if os.environ.get("QX_JFNK_FORCING"):
    cfg.scba.experimental_mixer.jfnk_forcing = os.environ["QX_JFNK_FORCING"]
if os.environ.get("QX_JFNK_MAX_NEWTON"):
    cfg.scba.experimental_mixer.jfnk_max_newton = int(
        os.environ["QX_JFNK_MAX_NEWTON"])
if os.environ.get("QX_JFNK_EPS"):
    cfg.scba.experimental_mixer.jfnk_eps = float(os.environ["QX_JFNK_EPS"])
if os.environ.get("QX_JFNK_TRUST"):
    cfg.scba.experimental_mixer.jfnk_trust = float(
        os.environ["QX_JFNK_TRUST"])
if os.environ.get("QX_JFNK_TRUSTMAX"):
    cfg.scba.experimental_mixer.jfnk_trust_max = float(
        os.environ["QX_JFNK_TRUSTMAX"])
if os.environ.get("QX_JFNK_DAMP"):
    cfg.scba.experimental_mixer.jfnk_newton_damp = float(
        os.environ["QX_JFNK_DAMP"])
if os.environ.get("QX_JFNK_PTC"):
    cfg.scba.experimental_mixer.jfnk_ptc = float(
        os.environ["QX_JFNK_PTC"])
# Exact-Jacobian Newton-Krylov (mixing_method = "newton") knobs.
if os.environ.get("QX_NEWTON_WARMUP"):  cfg.scba.experimental_mixer.newton_warmup_iters = int(os.environ["QX_NEWTON_WARMUP"])
if os.environ.get("QX_NEWTON_SWITCH"):  cfg.scba.experimental_mixer.newton_switch_tol = float(os.environ["QX_NEWTON_SWITCH"])
if os.environ.get("QX_NEWTON_KRYLOV"):  cfg.scba.experimental_mixer.newton_max_krylov = int(os.environ["QX_NEWTON_KRYLOV"])
if os.environ.get("QX_NEWTON_INNER"):   cfg.scba.experimental_mixer.newton_inner_tol = float(os.environ["QX_NEWTON_INNER"])
if os.environ.get("QX_NEWTON_FORCING"): cfg.scba.experimental_mixer.newton_forcing = os.environ["QX_NEWTON_FORCING"]
if os.environ.get("QX_NEWTON_MAX"):     cfg.scba.experimental_mixer.newton_max_newton = int(os.environ["QX_NEWTON_MAX"])
if os.environ.get("QX_NEWTON_TRUST"):   cfg.scba.experimental_mixer.newton_trust = float(os.environ["QX_NEWTON_TRUST"])
if os.environ.get("QX_NEWTON_TRUSTMAX"):cfg.scba.experimental_mixer.newton_trust_max = float(os.environ["QX_NEWTON_TRUSTMAX"])
if os.environ.get("QX_NEWTON_DAMP"):    cfg.scba.experimental_mixer.newton_damp = float(os.environ["QX_NEWTON_DAMP"])
if os.environ.get("QX_NEWTON_BACKTRACK"): cfg.scba.experimental_mixer.newton_backtrack = int(os.environ["QX_NEWTON_BACKTRACK"])
if os.environ.get("QX_NEWTON_RECONTOL"): cfg.scba.experimental_mixer.newton_recon_check_tol = float(os.environ["QX_NEWTON_RECONTOL"])
if os.environ.get("QX_JVP_FORM"):    cfg.scba.experimental_mixer.newton_jvp_form = os.environ["QX_JVP_FORM"]
if os.environ.get("QX_NEWTON_PRECOND"): cfg.scba.experimental_mixer.newton_precond = os.environ["QX_NEWTON_PRECOND"]
if os.environ.get("QX_NEWTON_PRECOND_RANK"): cfg.scba.experimental_mixer.newton_precond_rank = int(os.environ["QX_NEWTON_PRECOND_RANK"])
# Pole-subtracted SCBA sector (phonon/docs/pole_scba_implemented.md).
# The config validators refuse the combinations that would be silently wrong
# (retarded="half", an IR broadening floor, a pole window overlapping either the
# low-frequency mask or the CM channel), so these overrides cannot smuggle one in.
if os.environ.get("QX_POLE"):     cfg.phonon.pole_sector.enabled = bool(int(os.environ["QX_POLE"]))
if os.environ.get("QX_POLE_NP"):  cfg.phonon.pole_sector.max_poles = int(os.environ["QX_POLE_NP"])
if os.environ.get("QX_POLE_SECTORS"): cfg.phonon.pole_sector.sectors = os.environ["QX_POLE_SECTORS"]
if os.environ.get("QX_POLE_WMIN"): cfg.phonon.pole_sector.omega_min_thz = float(os.environ["QX_POLE_WMIN"])
if os.environ.get("QX_POLE_WMAX"): cfg.phonon.pole_sector.omega_max_thz = float(os.environ["QX_POLE_WMAX"])
if os.environ.get("QX_POLE_SHEET"): cfg.phonon.pole_sector.sheet = os.environ["QX_POLE_SHEET"]
if os.environ.get("QX_POLE_PGAMMA"): cfg.phonon.pole_sector.samples_per_halfwidth = float(os.environ["QX_POLE_PGAMMA"])
if os.environ.get("QX_POLE_QSTRIDE"): cfg.phonon.pole_sector.q_stride = int(os.environ["QX_POLE_QSTRIDE"])
if os.environ.get("QX_POLE_QMAX"): cfg.phonon.pole_sector.q_max = int(os.environ["QX_POLE_QMAX"])
if os.environ.get("QX_POLE_QBATCH"): cfg.phonon.pole_sector.q_batch = int(os.environ["QX_POLE_QBATCH"])
if os.environ.get("QX_POLE_LEGWTOL"): cfg.phonon.pole_sector.leg_weight_tol = float(os.environ["QX_POLE_LEGWTOL"])
if os.environ.get("QX_POLE_LEGWTOLOUT"): cfg.phonon.pole_sector.leg_weight_tol_out = float(os.environ["QX_POLE_LEGWTOLOUT"])
if os.environ.get("QX_POLE_BUBCORR"): cfg.phonon.pole_sector.bubble_correction = os.environ["QX_POLE_BUBCORR"]
if os.environ.get("QX_POLE_SIGMIN"): cfg.phonon.pole_sector.covariance_sigma_min = float(os.environ["QX_POLE_SIGMIN"])
if os.environ.get("QX_POLE_EXTRACT"): cfg.phonon.pole_sector.extraction_only = bool(int(os.environ["QX_POLE_EXTRACT"]))
if os.environ.get("QX_POLE_PSD"): cfg.phonon.pole_sector.psd_check = bool(int(os.environ["QX_POLE_PSD"]))
if os.environ.get("QX_POLE_MIXSCALE"): cfg.phonon.pole_sector.mixed_scale = float(os.environ["QX_POLE_MIXSCALE"])
if os.environ.get("QX_POLE_LEG"): cfg.phonon.pole_sector.leg = os.environ["QX_POLE_LEG"]
if os.environ.get("QX_POLE_CELLAVG"): cfg.phonon.pole_sector.cell_average = bool(int(os.environ["QX_POLE_CELLAVG"]))
# Newton BUDGET, not tolerance: eps_nep refusals are "did not reach 1e-10 in
# max_iter steps", and with trust_radius_cells = 0.25 a run of 8 steps can
# travel at most 2 cells from its seed. Raising either admits no worse pole --
# newton_tol is untouched -- it only lets the solve finish.
if os.environ.get("QX_POLE_NEWTIT"): cfg.phonon.pole_sector.newton_max_iterations = int(os.environ["QX_POLE_NEWTIT"])
if os.environ.get("QX_POLE_TRUST"): cfg.phonon.pole_sector.trust_radius_cells = float(os.environ["QX_POLE_TRUST"])
# Which quantity decides a pole was FOUND. "locate" gates on the frequency
# error eps_z; "residual" is the legacy scaled-matrix-residual gate, kept so
# old runs reproduce. newton_tol is untouched by either.
if os.environ.get("QX_POLE_ACCEPT"): cfg.phonon.pole_sector.accept = os.environ["QX_POLE_ACCEPT"]
if os.environ.get("QX_POLE_LOCTOL"): cfg.phonon.pole_sector.locate_tol = float(os.environ["QX_POLE_LOCTOL"])
if os.environ.get("QX_POLE_LOCTOLOUT"): cfg.phonon.pole_sector.locate_tol_out = float(os.environ["QX_POLE_LOCTOLOUT"])
if os.environ.get("QX_POLE_FREEZE"): cfg.phonon.pole_sector.freeze_membership = os.environ["QX_POLE_FREEZE"] not in ("0", "false", "False")
if os.environ.get("QX_POLE_EPOCH"): cfg.phonon.pole_sector.epoch_iterations = int(os.environ["QX_POLE_EPOCH"])
# Physical trust radius as a fraction of min(nearest seed, nearest band edge).
# Set tiny to reproduce the old grid-tied radius, which floors it at
# trust_radius_cells * h.
if os.environ.get("QX_POLE_TRUSTF"): cfg.phonon.pole_sector.trust_factor = float(os.environ["QX_POLE_TRUSTF"])
if os.environ.get("QX_POLE"):
    # Re-validate: the pole gates are cross-field, so an override that creates
    # an inconsistent combination must fail here rather than at iteration 40.
    cfg.phonon = type(cfg.phonon).model_validate(cfg.phonon.model_dump())
if os.environ.get("QX_RING_DTYPE"): cfg.phonon.sse_ring_dtype = os.environ["QX_RING_DTYPE"]
if os.environ.get("QX_G_FROM_L"):
    cfg.phonon.sse_greater_from_lesser = bool(int(os.environ["QX_G_FROM_L"]))
if os.environ.get("QX_FOLDVERIFY"):
    cfg.phonon.sse_fold_verify_iterations = int(os.environ["QX_FOLDVERIFY"])
if os.environ.get("QX_QBATCH"):   cfg.phonon.sse_dense_q_batched = bool(int(os.environ["QX_QBATCH"]))
if os.environ.get("QX_FREQGRID"):
    _fg = os.environ["QX_FREQGRID"].strip().lower()
    if _fg not in ("window", "file"):
        raise SystemExit(f"QX_FREQGRID must be 'window' or 'file', "
                         f"got {os.environ['QX_FREQGRID']!r}")
    cfg.phonon.frequency_grid = _fg
if cfg.phonon.frequency_grid == "file" and os.environ.get("QX_NE"):
    print("WARNING: QX_NE has no effect with frequency_grid='file' "
          "(the grid comes from phonon_energies.npy).", flush=True)

# Memory knobs. Neither had an override before, and both are the ones that
# actually decide whether a run fits: QX_MAXBATCH bounds the ~21 RGF backward
# temporaries of (batch, nq, b, b), QX_TAUCHUNK bounds the coupled-q ring
# intermediates. Both are exact -- they change only how the work is split.
if os.environ.get("QX_MAXBATCH"):
    cfg.phonon.solver.max_batch_size = int(os.environ["QX_MAXBATCH"])
if os.environ.get("QX_TAUCHUNK"):
    cfg.phonon.sse_tau_chunk_bytes = int(os.environ["QX_TAUCHUNK"])
if os.environ.get("QX_RELEASE_LEGS"):
    cfg.phonon.sse_release_leg_blocks = bool(int(os.environ["QX_RELEASE_LEGS"]))
if os.environ.get("QX_PERMSHARE"):
    cfg.phonon.sse_perm_cache_share = os.environ["QX_PERMSHARE"]

# Honor the (possibly-overridden) comm grid + threading + profiler.
setup_context(cfg)

from quatrex.core.scba import SCBA  # noqa: E402  (after setup_context)

scba = SCBA(cfg)
ph = scba.subsystems["phonon"]

# Frozen auxiliary-state census: solve/project the q-resolved pole clusters in
# the production phonon solver, but do not pass the old additive pole channel
# into the FFT ring.  The latter is precisely the representation under review
# and, for a tensor-factorized q ring, its full-device nnz layout does not even
# match the pair-local leg layout.  The extracted state is serialized after the
# run and contracted by _si_auxiliary_scba_review.py in one enriched basis.
if os.environ.get("QX_POLE_STATE_ONLY") == "1":
    n_disabled = 0
    for inter in getattr(scba, "interactions", []):
        if hasattr(inter, "_inject_pole_sector"):
            inter._inject_pole_sector = lambda _scba: None
            n_disabled += 1
    if ranks.rank == 0:
        print(f"QX_POLE_STATE_ONLY: disabled {n_disabled} additive pole-ring "
              "injector(s); solver extraction remains enabled", flush=True)

# Ballistic reference: zero the 3-phonon vertex (Sigma_phph == 0) while
# keeping the SCBA machinery (leads/OBC/MW current) identical -> the clean,
# same-normalization baseline for G_anh/G_ball.
if os.environ.get("QX_BALLISTIC") == "1":
    # The 3-phonon vertex lives on the SSE (inter.sigma_phonon_phonon), and
    # the SSE precomputes ``_phi_pair_index`` holding REFERENCES to the
    # phi_blocks arrays. Must zero IN PLACE (``arr[...] = 0``), not reassign,
    # so the precomputed index sees zeros -> Sigma_phph == 0 exactly.
    n_zeroed = 0
    for inter in getattr(scba, "interactions", []):
        sse = getattr(inter, "sigma_phonon_phonon", None)
        if sse is None:
            continue
        pb = getattr(sse, "phi_blocks", None)
        if pb is not None:
            for k in list(pb):
                pb[k][...] = 0.0
                n_zeroed += 1
        qv = getattr(sse, "_qvertices", None)
        if qv is not None:
            for blocks in qv.values():
                for k in list(blocks):
                    blocks[k][...] = 0.0
        # Tensor-decomposed vertex: zeroing the CP weights kills every
        # factored contraction (the sandwich carries lambda once per side)
        # and the phi_blocks reconstructed from the factors are zeroed
        # above/below like any dense vertex.
        vf = getattr(sse, "_vfactors", None)
        if vf is not None:
            vf.lambdas[...] = 0.0
        # belt-and-suspenders: zero the precomputed pair-index phi arrays too.
        ppi = getattr(sse, "_phi_pair_index", None)
        if ppi is not None:
            for quads in ppi.values():
                for q in quads:
                    q[4][...] = 0.0
                    q[5][...] = 0.0
        sse._tau_cache = None
    # With the vertex zeroed, Sigma_phph == 0 identically -- but the dense-q
    # contraction machinery would still allocate its full tau/fold buffer
    # stack (OOM at ~100 GB on the mos2 nk7 mesh, job 4344975). Drop the
    # phonon-phonon interaction from the registry so the SSE never runs;
    # the in-place zeroing above stays as belt-and-suspenders for anything
    # else holding vertex references.
    scba.interactions = [
        inter for inter in getattr(scba, "interactions", [])
        if getattr(inter, "sigma_phonon_phonon", None) is None
    ]
    if ranks.rank == 0:
        print(f"BALLISTIC: zeroed {n_zeroed} phi_blocks in place; "
              "phonon-phonon interaction removed from the registry",
              flush=True)

w = np.abs(np.asarray(get_host(ph.local_frequencies)))
# The cell measure belongs to the frequency integral on every grid.  On a
# uniform grid it is constant, so omitting it leaves balance ratios unchanged
# while making the reported current scale as 1/delta_omega under refinement.
w = w * np.asarray(get_host(ph.local_frequency_weights))


def _heat(mw):
    """Local hbar-omega-weighted heat current per interface (and per q)."""
    mw = np.asarray(get_host(mw))
    ww = w.reshape((-1,) + (1,) * (mw.ndim - 1))
    return np.real(np.sum(ww * mw, axis=0))


_it = {"n": 0}
_iter_sigma_max = []  # per-iteration, per-omega max |Sigma^<|
_iter_heat = []  # per-SCBA-iteration heat (rank-0-local frequency slice)
_orig = SCBA._has_converged

# --- eta=0 spectral diagnostic (QX_DIAG_SPECTRAL=1): per-iteration, FULL-omega
# G^R DOS fed into the bubble, |Sigma^R(w)|, |Sigma^<(w)|, and the raw vs
# windowed G^< magnitude actually convolved (the latter two computed inside the
# SSE and reduced to the global per-omega max -- read off the SSE object here).
_DIAG = os.environ.get("QX_DIAG_SPECTRAL") == "1"
_iter_gin_dos = []   # (n_iter, ne)  -Im Tr G^R  (G fed INTO the bubble)
_iter_sigR_w = []    # (n_iter, ne)  max|Sigma^R(w)|
_iter_sigL_w = []    # (n_iter, ne)  max|Sigma^<(w)|
_iter_graw_w = []    # (n_iter, ne)  raw max|G^<(w)| before the window
_iter_gwin_w = []    # (n_iter, ne)  windowed max|G^<(w)| into the FFT
_sse_diag = None
for _inter in getattr(scba, "interactions", []):
    _s = getattr(_inter, "sigma_phonon_phonon", None)
    if _s is not None:
        _sse_diag = _s
        break
_eg_full = np.abs(np.asarray(get_host(scba.energies)).real)   # (ne,) THz grid


def _gather_full(local_w):
    """Rank-local per-omega vector (ne_local,) -> full (ne,) by disjoint
    placement + a stack SUM all-reduce (each omega bin owned by one rank).
    Mirrors the current_spectrum gather below."""
    local_w = np.asarray(local_w, float).ravel()
    lf = np.abs(np.asarray(get_host(ph.local_frequencies)))
    full = np.zeros(_eg_full.size, dtype=np.float64)
    i0 = int(np.argmin(np.abs(_eg_full - float(lf.flat[0]))))
    full[i0:i0 + local_w.size] = local_w
    if ranks.stack.size > 1:
        recv = np.empty_like(full)
        ranks.stack.all_reduce(np.ascontiguousarray(full), recv, op="sum")
        full = recv
    return full


_best = {"res": np.inf, "sig": None}
_mean = {"n": 0, "sum": None}
_MEAN_SKIP = int(os.environ.get("QX_SIGMA_MEAN_SKIP", "4"))


def _logged(self):
    mw = getattr(ph, "meir_wingreen_current", None)
    if mw is not None and ranks.rank == 0:
        h = _heat(mw)
        _iter_heat.append(np.asarray(h))
        print(f"[it {_it['n']:2d}] energy J(local)={np.round(h, 4)}", flush=True)
    # Orbit-mean Sigma (QX_SAVE_SIGMA_MEAN): for a spiral orbit around
    # the fixed point (underdamped complex eigenpair, the mos2 film
    # limit cycle) the period-mean cancels the rotating component --
    # the mean IS the fixed-point estimate; restart from it.
    if os.environ.get("QX_SAVE_SIGMA_MEAN"):
        _it["mean_seen"] = _it.get("mean_seen", 0) + 1
        if _it["mean_seen"] > _MEAN_SKIP:
            _cur = tuple(
                np.asarray(get_host(b.data)).copy()
                for b in (self.data.sigma_lesser, self.data.sigma_greater,
                          self.data.sigma_retarded_hermitian))
            if _mean["sum"] is None:
                _mean["sum"] = list(_cur)
            else:
                for _a, _b in zip(_mean["sum"], _cur):
                    _a += _b
            _mean["n"] += 1
    if ranks.rank == 0:
        # per-omega magnitude of the raw SSE output: localizes WHERE a
        # diverging update grows (the omega bin), at negligible cost.
        sd = np.asarray(get_host(self.data.sigma_lesser.data))
        _iter_sigma_max.append(np.abs(sd).reshape(sd.shape[0], -1).max(axis=1))
    if _DIAG:
        # collective gathers (ALL ranks). G^R DOS = -Im Tr G^R; per-omega
        # |Sigma| = max over the (full, bcs=1) nnz block.
        gloc = -np.asarray(
            get_host(self.data.g_retarded.diagonal())).imag.sum(axis=1)
        gin = _gather_full(gloc)
        sr = np.asarray(get_host(self.data.sigma_retarded_hermitian.data))
        sl = np.asarray(get_host(self.data.sigma_lesser.data))
        sR = _gather_full(np.abs(sr).reshape(sr.shape[0], -1).max(axis=1))
        sL = _gather_full(np.abs(sl).reshape(sl.shape[0], -1).max(axis=1))
        if ranks.rank == 0:
            _iter_gin_dos.append(gin)
            _iter_sigR_w.append(sR)
            _iter_sigL_w.append(sL)
            graw = getattr(_sse_diag, "_diag_graw_w", None)
            gwin = getattr(_sse_diag, "_diag_gwin_w", None)
            if graw is not None and gwin is not None:
                _iter_graw_w.append(np.asarray(graw))
                _iter_gwin_w.append(np.asarray(gwin))
    _it["n"] += 1
    _done = _orig(self)

    # Best-residual Sigma snapshot (QX_SAVE_SIGMA_BEST): save the ITERATE
    # whose map residual was just measured.  At this point ``data.sigma_*``
    # is the raw map value F[Sigma_n], while ``sigma_*_prev`` is Sigma_n.
    # The former implementation ran before ``_orig`` and therefore paired
    # residual n-1 with F[Sigma_n].  A nominal minimum-residual snapshot could
    # consequently restart at a much larger residual.  Evaluate first and
    # preserve ``prev`` so the archived residual and state are the same point.
    _mixer = getattr(self, "_anderson_mixer", None)
    _is_probe = bool(_mixer is not None and _mixer.probing)
    if os.environ.get("QX_SAVE_SIGMA_BEST") and not _is_probe:
        _res = float(getattr(self, "_last_rel_sigma", np.inf))
        if np.isfinite(_res) and _res < _best["res"]:
            _best["res"] = _res
            _best["sig"] = tuple(
                np.asarray(get_host(b.data)).copy()
                for b in (self.data.sigma_lesser_prev,
                          self.data.sigma_greater_prev,
                          self.data.sigma_retarded_hermitian_prev))
            # QX_SIGMA_BEST_LIVE=1: also write the snapshot NOW (same
            # format as the end-of-run save, which stays the default
            # when unset and loses the state on a walltime kill).
            # Long films have multi-gigabyte distributed Sigma states. Save
            # the first best state, every requested stride thereafter, and a
            # converged state. The in-memory best is still updated at every
            # iteration and is written unconditionally at normal shutdown.
            _live_due = (_it["n"] == 1 or
                         _it["n"] % _BEST_LIVE_STRIDE == 0 or _done)
            if (os.environ.get("QX_SIGMA_BEST_LIVE") == "1" and _live_due):
                _sl_b, _sg_b, _sr_b = _best["sig"]
                np.savez(_sigma_file(os.environ["QX_SAVE_SIGMA_BEST"]),
                         sigma_lesser=_sl_b, sigma_greater=_sg_b,
                         sigma_retarded=_sr_b)
                if ranks.rank == 0:
                    print(f"SAVED SIGMA(best-live, res={_best['res']:.4e}) "
                          f"{os.environ['QX_SAVE_SIGMA_BEST']}", flush=True)
    return _done


SCBA._has_converged = _logged

if ranks.rank == 0:
    print(f"RUN config={CFG} phonon={cfg.scba.phonon} eta=0 "
          f"retarded={cfg.phonon.retarded_method} nblk={ph.block_sizes.shape[0]} "
          f"ne={int(scba.energies.shape[0])} "
          f"fgrid={cfg.phonon.frequency_grid} "
          f"micro={cfg.phonon.sse_microblock_dof}/"
          f"{cfg.phonon.sse_microblock_g_band} "
          f"bcs={cfg.compute.comm.block_comm_size} qcs={cfg.compute.comm.q_comm_size} "
          f"nranks={ranks.size}", flush=True)
    # Provenance: every QX_* override in the log, always. The 2026-08-01
    # film probe series became unattributable because the per-run envs
    # were recorded nowhere (job.sh overwritten by later launches).
    _qx = {k: v for k, v in sorted(os.environ.items())
           if k.startswith("QX_")}
    print("RUN env " + " ".join(f"{k}={v}" for k, v in _qx.items()),
          flush=True)

# Warm start: load Sigma^{<,>,R} from a previous cell's QX_SAVE_SIGMA file
# (vertex-scale continuation / temperature annealing / wall-time chaining).
# Optional QX_SIGMA_SCALE multiplies the loaded Sigma -- e.g.
# (lambda_new/lambda_old)^2 when stepping the vertex scale (Sigma ~ lambda^2).
# Multi-rank: each rank saves/loads ITS deterministic slice of the
# distributed buffers (file suffix .rank<r>.npz); the layout is reproducible
# iff the restart uses the SAME rank grid, which is asserted via the shapes.
def _sigma_file(base: str) -> str:
    if ranks.size == 1:
        return base
    stem = base[:-4] if base.endswith(".npz") else base
    return f"{stem}.rank{ranks.rank}.npz"


_restart_terms = sigma_restart_terms(os.environ)
if _restart_terms:
    _nparts = int(os.environ.get("QX_SIGMA_INIT_PARTS", "0") or 0)
    if _nparts and len(_restart_terms) != 1:
        raise SystemExit(
            "QX_SIGMA_INIT_PARTS cannot be combined with an affine two-state "
            "restart; use matching distributed snapshots")
    if _nparts:
        if ranks.size != 1:
            raise SystemExit(
                "QX_SIGMA_INIT_PARTS reconstructs a single-rank frozen state; "
                "launch with --ranks 1")
        _base = _restart_terms[0][0]
        _stem = _base[:-4] if _base.endswith(".npz") else _base
        _pieces = [np.load(f"{_stem}.rank{ir}.npz") for ir in range(_nparts)]
        _states = [{
            key: np.concatenate([part[key] for part in _pieces], axis=0)
            for key in ("sigma_lesser", "sigma_greater", "sigma_retarded")
        }]
    else:
        _states = [np.load(_sigma_file(path))
                   for path, _coefficient in _restart_terms]
    for _key, _buf in (("sigma_lesser", scba.data.sigma_lesser),
                       ("sigma_greater", scba.data.sigma_greater),
                       ("sigma_retarded", scba.data.sigma_retarded_hermitian)):
        _buf.data[:] = 0.0
        for (_path, _coefficient), _state in zip(_restart_terms, _states):
            _loaded = _state[_key]
            if os.environ.get("QX_SIGMA_INIT_PRIMITIVE_DOF"):
                from quatrex.phonon.experimental.pole.btd_linalg import remap_full_block_snapshot

                _loaded = remap_full_block_snapshot(
                    _loaded,
                    int(os.environ["QX_SIGMA_INIT_PRIMITIVE_DOF"]),
                    get_host(_buf.rows), get_host(_buf.cols))
            if _loaded.shape != _buf.data.shape:
                raise SystemExit(
                    f"QX_SIGMA_INIT slice mismatch for {_key}: snapshot "
                    f"{_loaded.shape} vs local {_buf.data.shape} -- restart "
                    "with the same rank grid as the saving run.")
            _buf.data[:] += _coefficient * xp.asarray(_loaded)
    _labels = []
    for _path, _coefficient in _restart_terms:
        _label = (f"{_path} ({_nparts} gathered parts)" if _nparts else
                  _sigma_file(_path))
        _labels.append(f"{_coefficient:+g} * {_label}")
    print("WARM START from affine " + " ".join(_labels), flush=True)

err = None
try:
    scba.run()
except Exception:
    err = traceback.format_exc()
    print(f"[rank {ranks.rank}] RUN RAISED:\n{err}", flush=True)


def _save_pole_states(path: str) -> None:
    """Save the small q-resolved rational state, never the dense pole legs.

    This is the bridge used by the auxiliary-SCBA rank gate.  The ordinary
    engine snapshot stores only sampled G/Sigma arrays; that loses the pole
    locations, coherent subspaces and projected Keldysh sources needed to ask
    whether the exact cluster--cluster output can be carried with fewer states
    than a fine frequency grid.  ``QX_SAVE_POLE_STATES`` is study-only and has
    no effect on the solver or fixed point.
    """
    qstates = list(getattr(ph, "pole_q_states", []) or [])
    if not qstates and getattr(ph, "pole_state", None) is not None:
        qstates = [((), ph.pole_state)]
    if ranks.stack.size != 1:
        raise NotImplementedError(
            "QX_SAVE_POLE_STATES requires an undistributed frequency axis. "
            "The promoted cluster count/rank can differ between stack ranks, "
            "so gathering sources by list position is not defined; use "
            "QX_SIGMA_INIT_PARTS with a one-rank frozen run.")
    rows = []
    for qidx, state in qstates:
        if state is None:
            continue
        legs = list(getattr(state, "legs", []) or [])
        sl = getattr(state, "source_lesser", [])
        sg = getattr(state, "source_greater", [])
        fit_values = list(getattr(state, "source_fit", []) or [])
        for m, cl in enumerate(legs):
            if m >= len(sl) or m >= len(sg):
                continue
            rows.append((tuple(int(i) for i in qidx), cl,
                         np.asarray(get_host(sl[m])),
                         np.asarray(get_host(sg[m])),
                         float(fit_values[m]) if m < len(fit_values)
                         else np.nan))
    full_frequencies = np.asarray(get_host(ph.local_frequencies))
    full_widths = np.asarray(get_host(ph.local_frequency_weights))
    if ranks.rank != 0:
        return
    if not rows:
        print("QX_SAVE_POLE_STATES: no allocated pole/source state to save",
              flush=True)
        return
    qdim = max((len(q) for q, *_ in rows), default=0)
    q_index = np.array([q + (0,) * (qdim - len(q)) for q, *_ in rows],
                       dtype=np.int64)
    pole_offsets = [0]
    source_offsets = [0]
    poles, us, vs, src_l, src_g, labels, fits = [], [], [], [], [], [], []
    for _q, cl, sl, sg, fit in rows:
        z = np.asarray(get_host(cl.z), dtype=complex).reshape(-1)
        poles.append(z)
        us.append(np.asarray(get_host(cl.u), dtype=complex))
        vs.append(np.asarray(get_host(cl.v), dtype=complex))
        sl = np.asarray(sl, dtype=complex)
        sg = np.asarray(sg, dtype=complex)
        expected = (len(full_frequencies), z.size, z.size)
        if sl.shape != expected or sg.shape != expected:
            raise ValueError(
                f"pole source has shapes {sl.shape}/{sg.shape}, expected "
                f"{expected}; refusing an ambiguous auxiliary-state export")
        src_l.append(sl.reshape(-1))
        src_g.append(sg.reshape(-1))
        pole_offsets.append(pole_offsets[-1] + z.size)
        source_offsets.append(source_offsets[-1] + sl.size)
        labels.append(str(getattr(cl, "label", "")))
        fits.append(fit)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        q_index=q_index,
        q_shape=np.array([int(k) for k in cfg.device.kpoint_grid if k > 1],
                         dtype=np.int64),
        pole_offsets=np.asarray(pole_offsets, dtype=np.int64),
        source_offsets=np.asarray(source_offsets, dtype=np.int64),
        poles=np.concatenate(poles),
        coupling_u=np.concatenate(us, axis=1),
        coupling_v=np.concatenate(vs, axis=1),
        source_lesser=np.concatenate(src_l),
        source_greater=np.concatenate(src_g),
        source_fit=np.asarray(fits, dtype=float),
        labels=np.asarray(labels),
        block_sizes=np.asarray(get_host(ph.block_sizes), dtype=np.int64),
        local_frequencies=np.asarray(full_frequencies, dtype=float),
        local_frequency_weights=np.asarray(full_widths, dtype=float),
    )
    print(f"SAVED POLE STATES {target} ({len(rows)} clusters, "
          f"{pole_offsets[-1]} poles)", flush=True)


if os.environ.get("QX_SAVE_POLE_STATES") and err is None:
    _save_pole_states(os.environ["QX_SAVE_POLE_STATES"])

# --- per-omega current spectrum (for transmission plots), gathered over the
# stack (energy) partition. Runs on ALL ranks (collective all_reduce). ---
spec_full = None
_mw = getattr(ph, "meir_wingreen_current", None)
if _mw is not None:
    _mw = np.asarray(get_host(_mw))
    _eg = np.abs(np.asarray(get_host(scba.energies)).real)
    _lf = np.abs(np.asarray(get_host(ph.local_frequencies)))
    _full = np.zeros((_eg.size,) + _mw.shape[1:], dtype=np.float64)
    _i0 = int(np.argmin(np.abs(_eg - float(_lf.flat[0]))))
    _full[_i0:_i0 + _mw.shape[0]] = np.real(_mw)
    if ranks.stack.size > 1:
        _recv = np.empty_like(_full)
        ranks.stack.all_reduce(np.ascontiguousarray(_full), _recv, op="sum")
        _full = _recv
    spec_full = _full

# Optional harmonic identity audit.  This is intentionally evaluated after
# the production solve and only for a one-block grouped device, where G^R is
# the complete finite-device matrix and the two contact self-energies retained
# by PhononSolver are unambiguous.  It never feeds anything back into SCBA.
caroli_full = caroli_current_full = caroli_error = None
if os.environ.get("QX_DIAG_CAROLI") == "1":
    try:
        contacts = getattr(ph, "_single_block_contacts", None)
        if contacts is None or int(ph.block_sizes.shape[0]) != 1:
            raise ValueError(
                "QX_DIAG_CAROLI requires one grouped Dyson block with two "
                "separately retained contacts"
            )
        from quatrex.phonon.experimental.ballistic_audit import (
            caroli_number_current,
            caroli_transmission,
            spectrum_error,
        )
        _gr = scba.data.g_retarded.blocks[0, 0]
        _tc = caroli_transmission(_gr, contacts[0][0], contacts[1][0])
        _jc = caroli_number_current(
            _tc, ph.left_occupancies, ph.right_occupancies)

        def _gather_caroli(_local):
            _local = np.asarray(get_host(_local), dtype=np.float64)
            _eg = np.abs(np.asarray(get_host(scba.energies)).real)
            _lf = np.abs(np.asarray(get_host(ph.local_frequencies)))
            _full = np.zeros((_eg.size,) + _local.shape[1:], dtype=np.float64)
            _i0 = int(np.argmin(np.abs(_eg - float(_lf.flat[0]))))
            _full[_i0:_i0 + _local.shape[0]] = _local
            if ranks.stack.size > 1:
                _recv = np.empty_like(_full)
                ranks.stack.all_reduce(
                    np.ascontiguousarray(_full), _recv, op="sum")
                _full = _recv
            return _full

        caroli_full = _gather_caroli(_tc)
        caroli_current_full = _gather_caroli(_jc)
        if ranks.rank == 0 and spec_full is not None:
            caroli_error = spectrum_error(
                caroli_current_full, np.asarray(spec_full)[..., 0])
            print(
                "CAROLI AUDIT "
                f"rel_l2={caroli_error['relative_l2']:.3e} "
                f"active_max={caroli_error['active_max_relative']:.3e}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 -- optional diagnostic
        if ranks.rank == 0:
            print(f"Caroli audit failed: {exc!r}", flush=True)

# --- snapshot (rank 0; last_heat is the stack/q-reduced fixed-point current) ---
if cfg.outputs.save_profiling_results:
    Profiler().dump_stats()

if xp.__name__ == "cupy":
    from mpi4py import MPI as _MPI
    _mp = np.array([xp.get_default_memory_pool().total_bytes() / 1e9])
    _MPI.COMM_WORLD.Allreduce(_MPI.IN_PLACE, _mp, op=_MPI.MAX)
    if ranks.rank == 0:
        print(f"GPU mempool peak (max over ranks): {_mp[0]:.2f} GB",
              flush=True)

# Per-slab scattering absorption + same-instant global balance: COLLECTIVE
# (stack all-reduce inside) -- must run on ALL ranks, before the rank-0
# snapshot gate (rank-0-only invocation deadlocks any stack>1 run).
_slab_pa, _final_bal = None, None
try:
    _slab_pa = scba._phonon_slab_absorption()
    if _slab_pa is not None:
        _final_bal = scba._phonon_bubble_energy_balance()
except Exception as exc:  # noqa: BLE001 -- diagnostic, never fatal
    if ranks.rank == 0:
        print(f"slab_absorption failed: {exc!r}", flush=True)

# Final-iterate G diagonals (per-DOF LDOS proxy -Im G^R and occupation
# numerator Im G^<) and the per-omega bubble-balance spectra. Collective
# gathers; cheap ((ne, N_D) reals). QX_SAVE_DIAG_G=0 disables.
_gr_diag = _gl_diag = _bb_spec = None
try:
    if os.environ.get("QX_SAVE_DIAG_G", "1") == "1" and cfg.scba.phonon:
        _mask = scba.data.g_lesser._stack_padding_mask
        _gr_diag = ranks.stack.all_gather_v(
            np.asarray(get_host(-scba.data.g_retarded.diagonal().imag)),
            axis=0, mask=_mask)
        _gl_diag = ranks.stack.all_gather_v(
            np.asarray(get_host(scba.data.g_lesser.diagonal().imag)),
            axis=0, mask=_mask)
    _spectra = getattr(scba, "_bubble_balance_spectra", None)
    if _spectra is not None:
        _bb_spec = np.stack([
            ranks.stack.all_gather_v(
                np.ascontiguousarray(get_host(sp)), axis=0)
            for sp in _spectra])
except Exception as exc:  # noqa: BLE001 -- diagnostic, never fatal
    if ranks.rank == 0:
        print(f"G-diagonal gather failed: {exc!r}", flush=True)

if ranks.rank == 0:
    npz = os.environ.get("QX_NPZ") or str(Path(cfg.output_dir).parent / "run.npz")
    Path(npz).parent.mkdir(parents=True, exist_ok=True)
    # NOTE: _heat() sums this rank's LOCAL frequency slice only. On a
    # single-rank run that is the full spectrum; multi-rank runs mark the
    # key so a partial slice is never mistaken for the physical current
    # (the stack-reduced value is out["last_heat"]/out["lead_current"]).
    final_heat = (_heat(ph.meir_wingreen_current)
                  if getattr(ph, "meir_wingreen_current", None) is not None else None)
    _heat_partial = ranks.stack.size > 1
    from quatrex.grid.energies import frequency_cell_widths
    out = dict(
        energies=np.asarray(get_host(scba.energies)).real,
        source_commit=str(os.environ.get("QX_SOURCE_COMMIT", "")),
        # Current integrals use the frequency-cell measure on every grid.
        # Keep the grid marker so historical raw-sum artifacts can still be
        # distinguished by the run census.
        uniform_frequency_grid=bool(
            getattr(ph, "uniform_frequency_grid", True)),
        frequency_cell_widths=np.asarray(get_host(
            frequency_cell_widths(xp.asarray(scba.energies).real))),
        eta=0.0, retarded=str(cfg.phonon.retarded_method),
        nblocks=int(ph.block_sizes.shape[0]), phonon=bool(cfg.scba.phonon),
        block_sizes=np.asarray(get_host(ph.block_sizes), dtype=np.int64),
        sse_g_band=int(cfg.phonon.sse_g_band),
        sse_microblock_dof=int(cfg.phonon.sse_microblock_dof),
        sse_microblock_g_band=int(cfg.phonon.sse_microblock_g_band),
        sse_generated_sigma_band=int(
            getattr(_sse_diag, "_sigma_micro_span", 1)),
        sse_vertex_span=int(getattr(_sse_diag, "_vertex_span", 1)),
        vertex_representation=(
            "decomposed" if cfg.phonon.decomposed_vertices_path is not None
            else ("qfold" if cfg.phonon.qfold_path is not None else "gamma")),
        sse_vertex_rank=int(
            getattr(getattr(_sse_diag, "_vfactors", None), "rank", 0)),
        decomposed_kernel=str(cfg.phonon.decomposed_kernel),
        q_mesh=np.asarray(cfg.device.kpoint_grid, dtype=np.int64),
        frequency_grid=str(cfg.phonon.frequency_grid),
        frequency_max_thz=float(np.asarray(get_host(scba.energies)).real[-1]),
        eta_obc=0.0,
        sse_greater_from_lesser=bool(cfg.phonon.sse_greater_from_lesser),
        pole_sector_enabled=bool(cfg.phonon.pole_sector.enabled),
        interaction_cutoff=float(cfg.phonon.interaction_cutoff),
        sigma_convergence_tol=float(cfg.phonon.sigma_convergence_tol),
        heat_flow_conservation_tol=float(
            cfg.phonon.heat_flow_conservation_tol),
        scba_max_iterations=int(cfg.scba.max_iterations),
        scba_min_iterations=int(cfg.scba.min_iterations),
        scba_mixing_method=str(cfg.scba.mixing_method),
        scba_mixing_factor=float(cfg.scba.mixing_factor),
        left_temperature=float(cfg.phonon.left_temperature),
        right_temperature=float(cfg.phonon.right_temperature),
        obc_algorithm=str(cfg.phonon.obc.algorithm),
        nevp_solver=str(cfg.phonon.obc.nevp_solver),
        obc_scattering_contacts=bool(cfg.phonon.obc_scattering_contacts),
        vertex_input_path=str(
            cfg.phonon.decomposed_vertices_path
            or cfg.phonon.qfold_path
            or cfg.phonon.fc3_path),
        ballistic=bool(os.environ.get("QX_BALLISTIC") == "1"),
        n_iter=_it["n"],
        block_comm_size=int(cfg.compute.comm.block_comm_size),
        q_comm_size=int(cfg.compute.comm.q_comm_size), nranks=int(ranks.size),
    )
    if final_heat is not None:
        out["final_heat"] = final_heat
        # Lead-to-lead conductance: interfaces 0 & -1, summed over any
        # transverse-q axes, robust to NaN internal interfaces (the Inv
        # solver fills only the leads). This is the physical film quantity
        # (the internal interfaces differ structurally). NB at stack>1 this
        # is the rank-0-local frequency slice; use np=1 for the clean number
        # or read the (q-summed) last_heat when the RGF path supplies it.
    # Fixed-point (last-iterate) heat -- the canonical conductance source
    # (all-reduced over stack + q-summed by the SCBA), read together with
    # `converged`. converged = the SCBA reached self-consistency before
    # max_iter; a False here means the last_heat is a non-converged iterate.
    out["diverged"] = bool(getattr(scba, "_diverged", False))
    # The SCBA sets _converged in the genuine convergence-return path;
    # a crashed run (err set) must not be reported as converged.
    out["converged"] = (bool(getattr(scba, "_converged", False))
                        and not out["diverged"] and err is None)
    # The reported current is the LAST iterate (the actual fixed point when
    # converged == True; otherwise transparently the last iterate). No
    # "best-conserved iterate" is kept -- over a non-converged trajectory it
    # is not a fixed point and headlining it misrepresents a non-result.
    lh = getattr(scba, "_last_heat_current", None)
    if lh is not None:
        lh = np.asarray(get_host(lh))
        out["last_heat"] = lh
        out["lead_current"] = 0.5 * (abs(float(np.real(lh[0]))) + abs(float(np.real(lh[-1]))))
    # Partition-dependent diagnostic: max-min spread over ALL RGF interfaces.
    # Besides finite-eta absorption it can contain physical redistribution
    # through the anharmonic interaction channel. For non-local FC2/Sigma a
    # cut is meaningful only when every crossing term is included. This is
    # NOT the convergence or global conservation gate.
    sp = getattr(scba, "_last_heat_spread", None)
    if sp is not None:
        out["internal_spread"] = float(sp)
    bb = getattr(scba, "_bubble_balance_history", None)
    if bb:
        # (P_in, P_out, resid) per iteration -- the Phi-derivable energy
        # balance of the bubble; resid ~roundoff = conserving SSE.
        out["iter_bubble_balance"] = np.asarray(bb, dtype=float)
    # Per-slab scattering energy absorption at the final iterate (computed
    # COLLECTIVELY above the rank-0 gate; written here): the block-resolved
    # bubble balance connecting adjacent interface heat currents by energy
    # continuity (J_k = J_{k-1} + P_abs(k) + eta term).
    if _slab_pa is not None:
        out["slab_absorption"] = np.asarray(get_host(_slab_pa))
    if _final_bal is not None:
        # Same-instant global balance (same Sigma/G pairing as the slab
        # binning): sum(slab_absorption) == P_out - P_in to roundoff.
        out["final_bubble_balance"] = np.asarray(
            [_final_bal[0], _final_bal[1]], dtype=complex)
    if _iter_sigma_max:
        out["iter_sigma_max"] = np.asarray(_iter_sigma_max)
    _mx = getattr(scba, "_anderson_mixer", None)
    if _mx is not None and getattr(_mx, "diagnostics", None):
        # Per-step mixer forensics (scba.mixer_diagnostics=true): residual
        # norm, LS conditioning, |gamma|, and safeguard flags.
        _dg = _mx.diagnostics
        out["iter_mixer_kind"] = np.array([d["kind"] for d in _dg])
        for k in ("fnorm", "cond", "gnorm", "m",
                  "capped", "reverted", "restarted"):
            out[f"iter_mixer_{k}"] = np.array(
                [d.get(k, np.nan) for d in _dg], dtype=float)
    if _DIAG and _iter_gin_dos:
        # eta=0 spectral diagnostic, full-omega per iteration (see _logged).
        out["iter_gin_dos"] = np.asarray(_iter_gin_dos)
        out["iter_sigR_w"] = np.asarray(_iter_sigR_w)
        out["iter_sigL_w"] = np.asarray(_iter_sigL_w)
        if _iter_graw_w:
            # Save the frequency axis alongside the spectral diagnostics.
            out["iter_graw_w"] = np.asarray(_iter_graw_w)
            out["iter_gwin_w"] = np.asarray(_iter_gwin_w)
            _wf = getattr(_sse_diag, "_diag_win_freqs", None)
            if _wf is not None:
                out["gwin_freqs"] = np.asarray(_wf)
    if _iter_heat:
        # SCBA convergence trace: heat per interface per iteration. At
        # stack>1 this is the rank-0-local frequency slice (relative
        # convergence is still meaningful); at stack==1 it is the full heat.
        out["iter_heat"] = np.asarray(_iter_heat)
    if _gr_diag is not None:
        # (ne, *nk, N_D): per-DOF -Im G^R (LDOS proxy) and Im G^< of the
        # final iterate -> post-hoc n_i(omega), T_eff and LDOS
        # (phonon/postproc/local_observables.py).
        out["gr_diag_imag"] = np.asarray(_gr_diag)
        out["gl_diag_imag"] = np.asarray(_gl_diag)
    if _bb_spec is not None:
        # (2, ne): per-omega (P_in, P_out) spectra of the bubble balance;
        # with current_spectrum this resolves the energy sum rule D(omega).
        out["bubble_balance_spectrum"] = np.asarray(_bb_spec)
    if spec_full is not None:
        # (ne, *nk, n_interfaces) real Meir-Wingreen number-current spectrum;
        # together with energies + lead temperatures this gives T_eff(w) and
        # the spectral heat current (see prod/plot_transmission.py).
        out["current_spectrum"] = spec_full
        out["t_left"] = float(cfg.phonon.left_temperature)
        out["t_right"] = float(cfg.phonon.right_temperature)
    if caroli_full is not None:
        out["caroli_transmission"] = np.asarray(caroli_full)
        out["caroli_current_spectrum"] = np.asarray(caroli_current_full)
        if caroli_error is not None:
            for _key, _value in caroli_error.items():
                out[f"caroli_mw_{_key}"] = float(_value)
    if err:
        out["error"] = err
    np.savez(npz, **out)
    # Headline the CONVERGED fixed point and its status -- not a
    # "best-conserved" transient. A non-converged run says so, and shows
    # its last iterate + iteration count so it is read as a non-result.
    _status = ("converged" if out.get("converged")
               else ("DIVERGED" if out.get("diverged") else "NOT CONVERGED"))
    _fh_tag = "final_heat(rank0-slice)=" if _heat_partial else "final_heat="
    print(f"SAVED {npz}  [{_status} after {out.get('n_iter')} it]  "
          f"{_fh_tag}"
          f"{None if final_heat is None else np.round(final_heat, 3)}  "
          f"lead_current={out.get('lead_current')}", flush=True)

# Sigma snapshot for wall-time chaining / warm starts: EVERY rank writes its
# own slice (multi-rank layouts restart via QX_SIGMA_INIT with the same grid).
if os.environ.get("QX_SAVE_SIGMA") and err is None:
    np.savez(_sigma_file(os.environ["QX_SAVE_SIGMA"]),
             sigma_lesser=np.asarray(get_host(scba.data.sigma_lesser.data)),
             sigma_greater=np.asarray(get_host(scba.data.sigma_greater.data)),
             sigma_retarded=np.asarray(
                 get_host(scba.data.sigma_retarded_hermitian.data)))
    if ranks.rank == 0:
        print(f"SAVED SIGMA {os.environ['QX_SAVE_SIGMA']}"
              f"{' (per-rank slices)' if ranks.size > 1 else ''}", flush=True)
# Orbit-mean snapshot (see _logged; the spiral-orbit fixed-point estimate).
if os.environ.get("QX_SAVE_SIGMA_MEAN") and err is None and _mean["n"] > 0:
    _sl, _sg, _sr = (a / _mean["n"] for a in _mean["sum"])
    np.savez(_sigma_file(os.environ["QX_SAVE_SIGMA_MEAN"]),
             sigma_lesser=_sl, sigma_greater=_sg, sigma_retarded=_sr)
    if ranks.rank == 0:
        print(f"SAVED SIGMA(mean over {_mean['n']} it) "
              f"{os.environ['QX_SAVE_SIGMA_MEAN']}", flush=True)
# Minimum-residual snapshot (tracked per iteration in _logged).
if os.environ.get("QX_SAVE_SIGMA_BEST") and err is None and _best["sig"] is not None:
    _sl, _sg, _sr = _best["sig"]
    np.savez(_sigma_file(os.environ["QX_SAVE_SIGMA_BEST"]),
             sigma_lesser=_sl, sigma_greater=_sg, sigma_retarded=_sr)
    if ranks.rank == 0:
        print(f"SAVED SIGMA(best, res={_best['res']:.4e}) "
              f"{os.environ['QX_SAVE_SIGMA_BEST']}", flush=True)
