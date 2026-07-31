"""Production phonon-transport SCBA driver + snapshot.

Honors the TOML ``[compute.comm]`` rank grid via :func:`setup_context` (a
distributed run uses ``block_comm_size`` x ``q_comm_size``) and dumps the
per-phase profiler JSON when the config enables it.

Env overrides (optional, on top of the TOML):
  QX_CONFIG (required, toml path), QX_NPZ (snapshot out, default <dir>/run.npz),
  QX_BALLISTIC=1 (zero the 3-phonon vertex -> the G_ball baseline),
  QX_ETA QX_MIX QX_MAXIT QX_NE QX_RETARDED QX_FC3 QX_ETAOBC
  QX_BCS QX_QCS (comm sizes -- override the TOML for a one-config rank sweep),
  QX_COMM_BACKEND=host_mpi|device_mpi|nccl (all per-op comm selectors at once),
  QX_FREQGRID=file (non-uniform grid from phonon_energies.npy),
  QX_AUXDW/QX_AUXFMAX (auxiliary uniform bubble grid, THz).

Backend: the array module (NumPy vs CuPy) is selected by QTX_ARRAY_MODULE
(qttools default "cupy" with silent NumPy fallback); set it explicitly for
reproducible CPU-vs-GPU parity runs.

Run: ``mpirun -np N python run.py`` (config via QX_CONFIG).
"""
import os
import traceback
from pathlib import Path

import numpy as np

from quatrex.core.config import parse_config, setup_context
from qttools import xp
from qttools.comm import comm as ranks
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host

CFG = os.environ["QX_CONFIG"]
cfg = parse_config(CFG)

# --- env overrides (TOML is the base) -----------------------------------
if os.environ.get("QX_ETA"):      cfg.phonon.eta = float(os.environ["QX_ETA"])
if os.environ.get("QX_MIX"):      cfg.scba.mixing_factor = float(os.environ["QX_MIX"])
if os.environ.get("QX_MIXMETHOD"):cfg.scba.mixing_method = os.environ["QX_MIXMETHOD"]
if os.environ.get("QX_ADEPTH"):   cfg.scba.anderson_depth = int(os.environ["QX_ADEPTH"])
if os.environ.get("QX_ADPERIOD"): cfg.scba.anderson_period = int(os.environ["QX_ADPERIOD"])
if os.environ.get("QX_MAXIT"):    cfg.scba.max_iterations = int(os.environ["QX_MAXIT"])
if os.environ.get("QX_MINIT"):    cfg.scba.min_iterations = int(os.environ["QX_MINIT"])
if os.environ.get("QX_NE"):       cfg.electron.energy_window_num = int(os.environ["QX_NE"])
if os.environ.get("QX_RETARDED"): cfg.phonon.retarded_method = os.environ["QX_RETARDED"]
if os.environ.get("QX_FC3"):      cfg.phonon.fc3_path = os.environ["QX_FC3"]
if os.environ.get("QX_ETAOBC"):   cfg.phonon.eta_obc = float(os.environ["QX_ETAOBC"])
if os.environ.get("QX_ETA_RAMP_ITERS"): cfg.phonon.eta_ramp_iterations = int(os.environ["QX_ETA_RAMP_ITERS"])
if os.environ.get("QX_ETA_FINAL"):      cfg.phonon.eta_final = float(os.environ["QX_ETA_FINAL"])
if os.environ.get("QX_ALGORITHM"):      cfg.phonon.solver.algorithm = os.environ["QX_ALGORITHM"]
if os.environ.get("QX_ETA_IR_FLOOR"):   cfg.phonon.eta_ir_floor_cells = float(os.environ["QX_ETA_IR_FLOOR"])
if os.environ.get("QX_SIGMATOL"): cfg.phonon.sigma_convergence_tol = float(os.environ["QX_SIGMATOL"])
if os.environ.get("QX_VSCALE"):   cfg.phonon.sse_vertex_scale = float(os.environ["QX_VSCALE"])
if os.environ.get("QX_GBAND"):    cfg.phonon.sse_g_band = int(os.environ["QX_GBAND"])
if os.environ.get("QX_GBAND_TAPER"): cfg.phonon.sse_g_band_taper = os.environ["QX_GBAND_TAPER"]
if os.environ.get("QX_RAMP"):     cfg.phonon.sse_ramp_iterations = int(os.environ["QX_RAMP"])
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
# Exact-Jacobian Newton-Krylov (mixing_method = "newton") knobs.
if os.environ.get("QX_NEWTON_WARMUP"):  cfg.scba.experimental_mixer.newton_warmup_iters = int(os.environ["QX_NEWTON_WARMUP"])
if os.environ.get("QX_NEWTON_SWITCH"):  cfg.scba.experimental_mixer.newton_switch_tol = float(os.environ["QX_NEWTON_SWITCH"])
if os.environ.get("QX_NEWTON_KRYLOV"):  cfg.scba.experimental_mixer.newton_max_krylov = int(os.environ["QX_NEWTON_KRYLOV"])
if os.environ.get("QX_NEWTON_TRUST"):   cfg.scba.experimental_mixer.newton_trust = float(os.environ["QX_NEWTON_TRUST"])
if os.environ.get("QX_NEWTON_TRUSTMAX"):cfg.scba.experimental_mixer.newton_trust_max = float(os.environ["QX_NEWTON_TRUSTMAX"])
if os.environ.get("QX_JVP_FORM"):    cfg.scba.experimental_mixer.newton_jvp_form = os.environ["QX_JVP_FORM"]
if os.environ.get("QX_NEWTON_PRECOND"): cfg.scba.experimental_mixer.newton_precond = os.environ["QX_NEWTON_PRECOND"]
if os.environ.get("QX_NEWTON_PRECOND_RANK"): cfg.scba.experimental_mixer.newton_precond_rank = int(os.environ["QX_NEWTON_PRECOND_RANK"])
if os.environ.get("QX_SSE_LOWMASK"): cfg.phonon.sse_low_freq_mask_thz = float(os.environ["QX_SSE_LOWMASK"])
# Non-uniform frequency grid: primary grid from phonon_energies.npy
# (QX_FREQGRID=file) + auxiliary uniform bubble grid (spacing/extent).
if os.environ.get("QX_FREQGRID"):
    _fg = os.environ["QX_FREQGRID"].strip().lower()
    if _fg not in ("window", "file"):
        raise SystemExit(f"QX_FREQGRID must be 'window' or 'file', "
                         f"got {os.environ['QX_FREQGRID']!r}")
    cfg.phonon.frequency_grid = _fg
if cfg.phonon.frequency_grid == "file" and os.environ.get("QX_NE"):
    print("WARNING: QX_NE has no effect with frequency_grid='file' "
          "(the grid comes from phonon_energies.npy).", flush=True)
if os.environ.get("QX_AUXDW"):    cfg.phonon.sse_aux_grid_dw_thz = float(os.environ["QX_AUXDW"])
if os.environ.get("QX_AUXFMAX"):  cfg.phonon.sse_aux_grid_fmax_thz = float(os.environ["QX_AUXFMAX"])

# Honor the (possibly-overridden) comm grid + threading + profiler.
setup_context(cfg)

from quatrex.core.scba import SCBA  # noqa: E402  (after setup_context)

scba = SCBA(cfg)
ph = scba.subsystems["phonon"]

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
        # SCP tadpole holds its own dense FC3 copy (made at __init__, before
        # this zeroing) -- disable it so the ballistic baseline is vertex-free.
        if getattr(sse, "_scp_tadpole", False):
            sse._scp_tadpole = False
            if getattr(sse, "_sigma_static", None) is not None:
                sse._sigma_static[...] = 0.0
    if ranks.rank == 0:
        print(f"BALLISTIC: zeroed {n_zeroed} phi_blocks in place", flush=True)

w = np.abs(np.asarray(get_host(ph.local_frequencies)))
if not getattr(ph, "uniform_frequency_grid", True):
    # Non-uniform grid: fold the per-bin quadrature cell widths into the
    # heat integral (uniform grids keep the legacy unweighted sum).
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


def _logged(self):
    mw = getattr(ph, "meir_wingreen_current", None)
    if mw is not None and ranks.rank == 0:
        h = _heat(mw)
        _iter_heat.append(np.asarray(h))
        print(f"[it {_it['n']:2d}] energy J(local)={np.round(h, 4)}", flush=True)
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
    return _orig(self)


SCBA._has_converged = _logged

if ranks.rank == 0:
    print(f"RUN config={CFG} phonon={cfg.scba.phonon} eta={cfg.phonon.eta} "
          f"retarded={cfg.phonon.retarded_method} nblk={ph.block_sizes.shape[0]} "
          f"ne={int(scba.energies.shape[0])} "
          f"fgrid={cfg.phonon.frequency_grid} "
          f"bcs={cfg.compute.comm.block_comm_size} qcs={cfg.compute.comm.q_comm_size} "
          f"nranks={ranks.size}", flush=True)

# Warm start: load Sigma^{<,>,R} from a previous cell's QX_SAVE_SIGMA file
# (vertex-scale continuation / temperature annealing). Optional QX_SIGMA_SCALE
# multiplies the loaded Sigma -- e.g. (lambda_new/lambda_old)^2 when stepping
# the vertex scale (Sigma ~ lambda^2). Single-rank layouts only.
if os.environ.get("QX_SIGMA_INIT"):
    if ranks.size > 1:
        raise SystemExit("QX_SIGMA_INIT requires a single-rank run "
                         "(the flat nnz layout must match the snapshot).")
    _sd = np.load(os.environ["QX_SIGMA_INIT"])
    _scale = float(os.environ.get("QX_SIGMA_SCALE", "1.0"))
    scba.data.sigma_lesser.data[:] = _scale * xp.asarray(_sd["sigma_lesser"])
    scba.data.sigma_greater.data[:] = _scale * xp.asarray(_sd["sigma_greater"])
    scba.data.sigma_retarded_hermitian.data[:] = _scale * xp.asarray(
        _sd["sigma_retarded"])
    print(f"WARM START from {os.environ['QX_SIGMA_INIT']} "
          f"(scale {_scale})", flush=True)

err = None
try:
    scba.run()
except Exception:
    err = traceback.format_exc()
    print(f"[rank {ranks.rank}] RUN RAISED:\n{err}", flush=True)

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
    final_heat = (_heat(ph.meir_wingreen_current)
                  if getattr(ph, "meir_wingreen_current", None) is not None else None)
    from quatrex.grid.energies import frequency_cell_widths
    out = dict(
        energies=np.asarray(get_host(scba.energies)).real,
        # Heat-key convention marker: on a UNIFORM grid the heat keys are
        # the legacy unweighted sums (integral / dw); on a non-uniform
        # grid the cell widths are folded in and the keys ARE integrals.
        uniform_frequency_grid=bool(
            getattr(ph, "uniform_frequency_grid", True)),
        frequency_cell_widths=np.asarray(get_host(
            frequency_cell_widths(xp.asarray(scba.energies).real))),
        eta=float(cfg.phonon.eta), retarded=str(cfg.phonon.retarded_method),
        nblocks=int(ph.block_sizes.shape[0]), phonon=bool(cfg.scba.phonon),
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
    # eta-absorption diagnostic: max-min spread over ALL interfaces (contains
    # the physical internal dip from finite eta; NOT the convergence gate)
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
            # graw lives on the primary grid (== energies); gwin on the
            # bubble's conv grid (the aux grid when sse_aux_grid_dw_thz
            # is on, else identical) -- save that axis alongside.
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
    if err:
        out["error"] = err
    np.savez(npz, **out)
    if os.environ.get("QX_SAVE_SIGMA") and ranks.size == 1 and not err:
        np.savez(os.environ["QX_SAVE_SIGMA"],
                 sigma_lesser=np.asarray(get_host(scba.data.sigma_lesser.data)),
                 sigma_greater=np.asarray(
                     get_host(scba.data.sigma_greater.data)),
                 sigma_retarded=np.asarray(
                     get_host(scba.data.sigma_retarded_hermitian.data)))
        print(f"SAVED SIGMA {os.environ['QX_SAVE_SIGMA']}", flush=True)
    # Headline the CONVERGED fixed point and its status -- not a
    # "best-conserved" transient. A non-converged run says so, and shows
    # its last iterate + iteration count so it is read as a non-result.
    _status = ("converged" if out.get("converged")
               else ("DIVERGED" if out.get("diverged") else "NOT CONVERGED"))
    print(f"SAVED {npz}  [{_status} after {out.get('n_iter')} it]  "
          f"final_heat="
          f"{None if final_heat is None else np.round(final_heat, 3)}  "
          f"lead_current={out.get('lead_current')}", flush=True)
