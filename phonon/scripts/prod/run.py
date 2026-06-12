"""PRODUCTION phonon-transport SCBA driver + snapshot (committed).

Committed port of the /tmp runner. Unlike that script it honors the TOML
``[compute.comm]`` rank grid via :func:`setup_context` (so a distributed
scaling run actually uses ``block_comm_size`` x ``q_comm_size``), and it dumps
the per-phase profiler JSON when the config enables it.

Env overrides (optional, on top of the TOML):
  QX_CONFIG (required, toml path), QX_NPZ (snapshot out, default <dir>/run.npz),
  QX_BALLISTIC=1 (zero the 3-phonon vertex -> the G_ball baseline),
  QX_ETA QX_MIX QX_MAXIT QX_NE QX_RETARDED QX_FC3 QX_ETAOBC QX_ZMP QX_ZMFLOOR
  QX_BCS QX_QCS (comm sizes -- override the TOML for a one-config rank sweep).

Run: ``mpirun -np N python run.py`` (config via QX_CONFIG).
"""
import os
import traceback
from pathlib import Path

import numpy as np

from quatrex.core.config import parse_config, setup_context
from qttools.comm import comm as ranks
from qttools.profiling import Profiler

CFG = os.environ["QX_CONFIG"]
cfg = parse_config(CFG)

# --- env overrides (TOML is the base) -----------------------------------
if os.environ.get("QX_ETA"):      cfg.phonon.eta = float(os.environ["QX_ETA"])
if os.environ.get("QX_MIX"):      cfg.scba.mixing_factor = float(os.environ["QX_MIX"])
if os.environ.get("QX_MIXMETHOD"):cfg.scba.mixing_method = os.environ["QX_MIXMETHOD"]
if os.environ.get("QX_ADEPTH"):   cfg.scba.anderson_depth = int(os.environ["QX_ADEPTH"])
if os.environ.get("QX_MAXIT"):    cfg.scba.max_iterations = int(os.environ["QX_MAXIT"])
if os.environ.get("QX_MINIT"):    cfg.scba.min_iterations = int(os.environ["QX_MINIT"])
if os.environ.get("QX_NE"):       cfg.electron.energy_window_num = int(os.environ["QX_NE"])
if os.environ.get("QX_RETARDED"): cfg.phonon.retarded_method = os.environ["QX_RETARDED"]
if os.environ.get("QX_FC3"):      cfg.phonon.fc3_path = os.environ["QX_FC3"]
if os.environ.get("QX_ETAOBC"):   cfg.phonon.eta_obc = float(os.environ["QX_ETAOBC"])
if os.environ.get("QX_SIGMATOL"): cfg.phonon.sigma_convergence_tol = float(os.environ["QX_SIGMATOL"])
if os.environ.get("QX_ZMP"):      cfg.phonon.zero_mode_projection = os.environ["QX_ZMP"] == "1"
if os.environ.get("QX_ZMFLOOR"):  cfg.phonon.zero_mode_floor_thz = float(os.environ["QX_ZMFLOOR"])
if os.environ.get("QX_BCS"):      cfg.compute.comm.block_comm_size = int(os.environ["QX_BCS"])
if os.environ.get("QX_QCS"):      cfg.compute.comm.q_comm_size = int(os.environ["QX_QCS"])

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
        # belt-and-suspenders: zero the precomputed pair-index phi arrays too.
        ppi = getattr(sse, "_phi_pair_index", None)
        if ppi is not None:
            for quads in ppi.values():
                for q in quads:
                    q[4][...] = 0.0
                    q[5][...] = 0.0
        sse._tau_cache = None
    if ranks.rank == 0:
        print(f"BALLISTIC: zeroed {n_zeroed} phi_blocks in place", flush=True)

# QX_SYMMETRIZE_FC3=1: S3-symmetrize the device vertex in place (diagnostic
# experiment: the Phi-derivable energy balance is exact only for a totally
# symmetric vertex; the Gamma-folded device construction pins the external
# leg and breaks it structurally).
# NB =1 is BUGGY, kept only to reproduce the 2026-06-10/11 sym_full runs:
# (a) the two 3-cycle key permutations need the INVERSE axis transpose
# ((1,2,0) pairs with transpose (2,0,1) and vice versa); (b) acc/cnt over
# present keys is not the S3 projection. Use =2 for the exact projection.
if os.environ.get("QX_SYMMETRIZE_FC3") == "1":
    for inter in getattr(scba, "interactions", []):
        sse = getattr(inter, "sigma_phonon_phonon", None)
        if sse is None or getattr(sse, "phi_blocks", None) is None:
            continue
        pb = sse.phi_blocks
        orig = {k: v.copy() for k, v in pb.items()}
        for (I, K1, K2) in list(pb):
            acc = orig[(I, K1, K2)].copy()
            cnt = 1
            for key, tr in (((K1, I, K2), (1, 0, 2)), ((I, K2, K1), (0, 2, 1)),
                            ((K2, K1, I), (2, 1, 0)), ((K1, K2, I), (1, 2, 0)),
                            ((K2, I, K1), (2, 0, 1))):
                if key in orig:
                    acc += orig[key].transpose(tr)
                    cnt += 1
            pb[(I, K1, K2)][...] = acc / cnt
        if ranks.rank == 0:
            print(f"SYMMETRIZED FC3: {len(pb)} blocks (S3 average)", flush=True)

# QX_SYMMETRIZE_FC3=2: EXACT S3 projection -- full-group average with the
# correct axis transport (3-cycles use the inverse permutation as the numpy
# transpose) and missing orbit members counted as zero (/6). Verified to
# give worst-case S3 violation ~1e-16 on the cnt33_L2 device vertex.
# Absent orbit keys cannot be created in place (the SSE pair index is
# already built) -- they are only reported.
if os.environ.get("QX_SYMMETRIZE_FC3") == "2":
    _G = (((0, 1, 2), (0, 1, 2)), ((1, 0, 2), (1, 0, 2)),
          ((0, 2, 1), (0, 2, 1)), ((2, 1, 0), (2, 1, 0)),
          ((1, 2, 0), (2, 0, 1)), ((2, 0, 1), (1, 2, 0)))
    for inter in getattr(scba, "interactions", []):
        sse = getattr(inter, "sigma_phonon_phonon", None)
        if sse is None or getattr(sse, "phi_blocks", None) is None:
            continue
        pb = sse.phi_blocks
        orig = {k: v.copy() for k, v in pb.items()}
        n_missing = 0
        for key in list(pb):
            acc = None
            for kp, tr in _G:
                ik = tuple(key[i] for i in kp)
                if ik in orig:
                    t = orig[ik].transpose(tr)
                    acc = t.copy() if acc is None else acc + t
                else:
                    n_missing += 1
            pb[key][...] = acc / 6.0
        if ranks.rank == 0:
            print(f"SYMMETRIZED FC3 (exact /6): {len(pb)} blocks, "
                  f"{n_missing} missing orbit members counted as zero",
                  flush=True)

w = np.abs(np.asarray(ph.local_frequencies))


def _heat(mw):
    """Local hbar-omega-weighted heat current per interface (and per q)."""
    mw = np.asarray(mw)
    ww = w.reshape((-1,) + (1,) * (mw.ndim - 1))
    return np.real(np.sum(ww * mw, axis=0))


_it = {"n": 0}
_iter_heat = []  # per-SCBA-iteration heat (rank-0-local frequency slice)
_orig = SCBA._has_converged


def _logged(self):
    mw = getattr(ph, "meir_wingreen_current", None)
    if mw is not None and ranks.rank == 0:
        h = _heat(mw)
        _iter_heat.append(np.asarray(h))
        print(f"[it {_it['n']:2d}] energy J(local)={np.round(h, 4)}", flush=True)
    _it["n"] += 1
    return _orig(self)


SCBA._has_converged = _logged

if ranks.rank == 0:
    print(f"RUN config={CFG} phonon={cfg.scba.phonon} eta={cfg.phonon.eta} "
          f"retarded={cfg.phonon.retarded_method} nblk={ph.block_sizes.shape[0]} "
          f"ne={cfg.electron.energy_window_num} "
          f"bcs={cfg.compute.comm.block_comm_size} qcs={cfg.compute.comm.q_comm_size} "
          f"nranks={ranks.size}", flush=True)

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
    _mw = np.asarray(_mw)
    _eg = np.abs(np.asarray(scba.energies).real)
    _lf = np.abs(np.asarray(ph.local_frequencies))
    _full = np.zeros((_eg.size,) + _mw.shape[1:], dtype=np.float64)
    _i0 = int(np.argmin(np.abs(_eg - float(_lf.flat[0]))))
    _full[_i0:_i0 + _mw.shape[0]] = np.real(_mw)
    if ranks.stack.size > 1:
        _recv = np.empty_like(_full)
        ranks.stack.all_reduce(np.ascontiguousarray(_full), _recv, op="sum")
        _full = _recv
    spec_full = _full

# --- snapshot (rank 0; best_heat is the canonical stack/q-reduced current) ---
if cfg.outputs.save_profiling_results:
    Profiler().dump_stats()

if ranks.rank == 0:
    npz = os.environ.get("QX_NPZ") or str(Path(cfg.output_dir).parent / "run.npz")
    Path(npz).parent.mkdir(parents=True, exist_ok=True)
    final_heat = (_heat(ph.meir_wingreen_current)
                  if getattr(ph, "meir_wingreen_current", None) is not None else None)
    out = dict(
        energies=np.asarray(scba.energies).real,
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
        # or read the (q-summed) best_heat when the RGF path supplies it.
    # Converged-iterate (fixed-point) heat -- the canonical conductance source
    # (all-reduced over stack + q-summed by the SCBA). Use this, NOT the
    # best-conserved transient. converged = the SCBA stopped before max_iter.
    out["converged"] = bool(_it["n"] < cfg.scba.max_iterations)
    lh = getattr(scba, "_last_heat_current", None)
    if lh is not None:
        lh = np.asarray(lh)
        out["last_heat"] = lh
        out["lead_current"] = 0.5 * (abs(float(np.real(lh[0]))) + abs(float(np.real(lh[-1]))))
    bh = getattr(scba, "_best_heat_current", None)
    if bh is not None:
        out["best_heat"] = np.asarray(bh)
        out["best_cons"] = float(getattr(scba, "_best_heat_conservation", float("nan")))
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
    if _iter_heat:
        # SCBA convergence trace: heat per interface per iteration. At
        # stack>1 this is the rank-0-local frequency slice (relative
        # convergence is still meaningful); at stack==1 it is the full heat.
        out["iter_heat"] = np.asarray(_iter_heat)
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
    print(f"SAVED {npz}  final_heat="
          f"{None if final_heat is None else np.round(final_heat, 3)}  "
          f"best={None if bh is None else np.round(np.asarray(bh), 3)}  "
          f"best_cons={out.get('best_cons')}", flush=True)
