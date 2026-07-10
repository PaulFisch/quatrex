"""Production transport study: T-sweeps, length ladders, spectral current,
summaries and the document figures.

``run`` drives the PRODUCTION phonon-transport study matrices on the node,
SEQUENTIALLY -- one mpirun at a time (node hygiene: ``pipeline.launch_cell``
asserts the node is idle before every launch). Each cell = a ballistic
baseline (QX_BALLISTIC=1) + the anharmonic SCBA, snapshotting both NPZs and
recording a manifest that :mod:`phonon.studies.summarize` consumes. Resumable:
a cell whose NPZ already exists is skipped.

Geometry inputs (dynamical_matrix.mat, fc3_blocks.hdf5, structure.xyz) are
built ONCE per (system, length) into a cached work dir; only
quatrex_config.toml is rewritten per cell (T/eta/mixing). Snapshots +
manifest.json go to ``phonon/scripts/out/prod/<study>/`` (the historical run
dirs -- existing data is read from and resumed there).

``plot`` renders either
- ``--what transmission``: per-cell effective transmission
  T(w) = I(w)/dn(w) and spectral heat current from the NPZ
  ``current_spectrum`` (per-omega Meir-Wingreen, q-summed), or
- ``--what figures``: the document transport figures from each study's
  ``summary.csv`` into ``document/fig/transport_sweeps/`` (existing names, so
  the reruns drop in without touching the .tex includes).

Usage (from the repo root)::

    python -m phonon.studies transport run --study cnt33 [--dry-run]
    python -m phonon.studies transport run --study sifilm --cells ns3_nk9
    python -m phonon.studies transport plot --what transmission \
        --run-dir phonon/scripts/out/prod/sinw_d5a [--tags T30 L2]
    python -m phonon.studies transport plot --what figures [--studies cnt33 ...]

The matrices encode the CLAUDE.md recipe (linear mixing for the soft d5a,
Anderson depth 5 for the well-conditioned CNT/d11a, L>=2 for the prod OBC,
SiNW eta 0.11, film q-parallel with block=1).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from phonon.studies import pipeline, summarize

ROOT = pipeline.ROOT
OUT_ROOT = ROOT / "phonon/scripts/out/prod"   # historical run dirs (data lives here)
WORK_ROOT = pipeline.GEOM                     # shared geometry inputs
DOC_FIG = ROOT / "document/fig/transport_sweeps"

STUDIES = ("cnt33", "cnt80", "sinw_d5a", "sinw_d11a", "srtio3", "sifilm")

HBAR_EV = 6.582119569e-16
KB_EV = 8.617333262e-5
THZ_TO_RAD = 2.0 * np.pi * 1e12

GEOM_FILES = ("dynamical_matrix.mat", "fc3_blocks.hdf5", "structure.xyz",
              "phonon_energies.npy", "qfold_vertices.npz", "kshift.npy",
              "decomposed_vertices.npz")


# --------------------------- study matrices ---------------------------
def matrix(study):
    """Return (cells, manifest_cells). A cell drives one ball+anh run pair; the
    manifest row carries the metadata phonon.studies.summarize needs."""
    cells, man = [], []

    def add(system, length, tag, sweep, t_mean, n_slabs, cfg, nranks,
            nk=None, tdir="z", fmax=None, nfreq=None, emin=0.0, dt=10.0,
            do_anh=True, ring_threads=1):
        cells.append(dict(system=system, length=length, nk=nk, tag=tag,
                          cfg=cfg, nranks=nranks, do_anh=do_anh,
                          ring_threads=ring_threads))
        man.append(dict(tag=tag, system=system, sweep=sweep, t_mean=t_mean,
                        n_slabs=n_slabs, tdir=tdir, fmax=fmax, nfreq=nfreq,
                        emin=emin, dt=dt, nk=list(nk)))

    # Parallelism: the 3-phonon bubble (~99% of a step) parallelises over its
    # omega batch via the ring_contract thread pool (QX_RING_THREADS) -- ~15-27x,
    # bit-exact, no MPI comm -- so single-node CNT/SiNW/SrTiO3 run 1 rank x N
    # pool threads. (MPI stack/block parallelism still exists and is what
    # distributes ENERGY/BLOCK memory across nodes & GPUs at scale; it composes
    # with the pool -- each rank pools its local tau slice. block_comm_size>1 is
    # fixed and works when num_blocks >= 2*bcs.)
    # Mixing: the well-conditioned CNT/d11a converge with Anderson (the linear
    # SCBA oscillates at these eta -- F30 used Anderson); throttle depth to 5 to
    # bound the history memory (the nk9 film NaN was a large-nq Anderson case, so
    # the film stays linear). d5a (ultra-soft twist) uses gentle linear.
    AND = dict(mixing_method="anderson", anderson_depth=5, mix=0.2)
    if study == "cnt33":
        # 181 pts (d_omega 0.306) + eta 0.31 (eta/d_omega 1.0): bubble cost is
        # LINEAR in nfreq, so the finer grid is cheap and the smaller eta
        # roughly halves the internal eta-absorption dip (3.2% -> ~2%).
        fmax, nfreq = 55.0, 181
        # temperature sweep at L=2 (lowest prod length)
        for T in (30, 50, 100, 150, 200, 300):
            add("cnt33", 2, f"T{T}", "temperature", T, 2,
                dict(ncells=2, temperature=T, eta=0.31, **AND, nfreq=nfreq,
                     fmax=fmax, max_iter=100, bcs=1, qcs=1),
                1, fmax=fmax, nfreq=nfreq, ring_threads=16)
        # length ladder at 300 K (prod reaches beyond the dense L=3)
        for L in (2, 3, 4):
            add("cnt33", L, f"L{L}", "length", 300, L,
                dict(ncells=L, temperature=300, eta=0.31, **AND, nfreq=nfreq,
                     fmax=fmax, max_iter=100, bcs=1, qcs=1),
                1, fmax=fmax, nfreq=nfreq, ring_threads=16)

    elif study == "cnt80":
        # 121 pts (d_omega 0.417, eta 0.45 -> ratio 1.08, well resolved) +
        # 60-iter cap: the 96-DOF (8,0) bubble is ~50x cnt33 per ring call, so
        # grid x iters is the difference between ~5 h and ~1.5 h per cell.
        fmax, nfreq = 50.0, 121
        for L in (2, 3):
            add("cnt80", L, f"L{L}", "length", 300, L,
                dict(ncells=L, temperature=300, eta=0.45, **AND, nfreq=nfreq,
                     fmax=fmax, max_iter=60, bcs=1, qcs=1),
                1, fmax=fmax, nfreq=nfreq, ring_threads=16)

    elif study in ("sinw_d11a", "sinw_d5a"):
        # 41 pts to 18 THz = EXACTLY the dense-validated F10 grid (d_omega 0.45,
        # eta_w 0.11, converged ratio 0.942) -- and the bubble cost is linear in
        # nfreq while quartic in the 135-DOF d11a block size, so the grid is the
        # cheap knob.
        # d5a: the F10 coarse grid (41 pts to 18 THz, eta 0.11) is
        # LOAD-BEARING -- resolving the near-DC soft-mode bins (161 pts at
        # eta 0.11) destabilises the SCBA at ANY coupling (even lambda=0.3),
        # while the coarse grid (bins start at 0.45 THz) converges at full
        # coupling (E-matrix probes; E5 = T100 lambda=1 monotone). Fine grids
        # need eta >= 2*d_omega instead. d11a (no soft mode): resolved grid.
        # Sigma residual decays ~3%/iter -> relax sigma tol; the heat
        # observable plateaus far earlier and the lead-balance gate guards it.
        if study == "sinw_d5a":
            # 2026-06-12 (sign-fix era): the old "coarse grid is load-bearing"
            # lore is superseded -- with the corrected SSE sign the RESOLVED
            # grid (201 pts, eta/d_omega 1.2) converges cleanly at T <= 200
            # (T100: Sigma residual 7e-5 in ~40 Anderson its) and roughly
            # halves the lead-balance eta-commutator floor vs 101 pts. The
            # SSE cutoff masks the soft twist channel (conserving). 300 K is
            # marginal-QP (gamma/omega ~ 0.5 at the softest mode): cells
            # abort fast via QX_ABORT_RESIDUAL and are recorded as diverged.
            fmax, nfreq, eta = 18.0, 201, 0.11
            mixarg = dict(mixing_method="anderson", anderson_depth=5,
                          mix=0.05, sse_cutoff=0.5)
        else:
            fmax, nfreq, eta = 18.0, 161, 0.225
            mixarg = dict(**AND, sigma_tol=3e-2)
        for L in (2, 3):
            add(study, L, f"L{L}", "length", 300, L,
                dict(ncells=L, temperature=300, eta=eta, **mixarg, nfreq=nfreq,
                     fmax=fmax, max_iter=100, bcs=1, qcs=1),
                1, fmax=fmax, nfreq=nfreq, ring_threads=16)
        # temperature sweep at L=2 (d5a soft: low T converges; high T may not)
        Ts = (30, 50, 100, 150, 200, 300) if study == "sinw_d5a" else (200, 300, 400)
        for T in Ts:
            add(study, 2, f"T{T}", "temperature", T, 2,
                dict(ncells=2, temperature=T, eta=eta, **mixarg, nfreq=nfreq,
                     fmax=fmax, max_iter=100, bcs=1, qcs=1),
                1, fmax=fmax, nfreq=nfreq, ring_threads=16,
                do_anh=(study == "sinw_d5a"))

    elif study == "sifilm":
        fmax, nfreq = 15.0, 121
        # thickness sweep at nk=9 (the prod qfold builder needs ODD nk for the
        # Gamma-centered IDFT; nk>=8 is q-converged per F23). q-parallel, block=1.
        for ns in (3, 5, 8):
            add("sifilm", ns, f"ns{ns}_nk9", "thickness", 300, ns,
                dict(nslabs=ns, nk=9, tdir="x", temperature=300, eta=0.4,
                     nfreq=nfreq, fmax=fmax, max_iter=40, bcs=1, qcs=27),
                {3: 108, 5: 108, 8: 54}[ns],  # ns8 vertices 1.2 GB/rank -> cap
                nk=9, tdir="x", fmax=fmax, nfreq=nfreq)
        # q-convergence at ns=3 (odd meshes)
        for nk in (5, 7, 9):
            add("sifilm", 3, f"ns3_nk{nk}", "qconv", 300, 3,
                dict(nslabs=3, nk=nk, tdir="x", temperature=300, eta=0.4,
                     nfreq=nfreq, fmax=fmax, max_iter=40, bcs=1, qcs=nk * nk),
                {5: 100, 7: 98, 9: 108}[nk],
                nk=nk, tdir="x", fmax=fmax, nfreq=nfreq)

    elif study == "srtio3":
        # Gamma-only finite SrTiO3 slab -- strongly anharmonic best-effort.
        # Larger eta (broad) + Anderson; the transverse instabilities fold out so
        # the Gamma-only dispersion is stable (verified).
        fmax, nfreq = 26.0, 121
        for L in (2, 3):
            add("srtio3", L, f"L{L}", "length", 300, L,
                dict(ncells=L, temperature=300, eta=0.3, **AND, nfreq=nfreq,
                     fmax=fmax, bcs=1, qcs=1),
                1, fmax=fmax, nfreq=nfreq, ring_threads=16)
        for T in (200, 300, 600):
            add("srtio3", 2, f"T{T}", "temperature", T, 2,
                dict(ncells=2, temperature=T, eta=0.3, **AND, nfreq=nfreq,
                     fmax=fmax, bcs=1, qcs=1),
                1, fmax=fmax, nfreq=nfreq, ring_threads=16)

    else:
        raise SystemExit(f"unknown study {study}")
    return cells, man


# --------------------------- run driver ---------------------------
def geometry_dir(system, length, nk=None):
    """Build inputs once per (system, length[, nk]); return the work dir."""
    key = f"{system}_L{length}" + (f"_nk{nk}" if nk else "")
    work = WORK_ROOT / key
    stamp = work / ".built"
    if stamp.exists():
        return work
    work.mkdir(parents=True, exist_ok=True)
    if system == "sifilm":
        pipeline.build_geometry("sifilm", work, nslabs=length, nk=nk, nproc=8)
    else:  # cnt33/cnt80/sinw_*/srtio3: finite -L cells along the wire/slab
        pipeline.build_geometry(system, work, ncells=length)
    stamp.write_text("ok")
    return work


def cell_workdir(geom, tag, out_dir):
    """Per-cell work dir (symlinks the shared geometry) so cells do not clobber
    each other's quatrex_config.toml."""
    cw = out_dir / "work" / tag
    cw.mkdir(parents=True, exist_ok=True)
    for fn in GEOM_FILES:
        src = Path(geom) / fn
        dst = cw / fn
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
    return cw


def run(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m phonon.studies transport run", description=__doc__)
    p.add_argument("--study", required=True, choices=STUDIES)
    p.add_argument("--rank-budget", type=int, default=120,
                   help="max CORES for a cell (nranks x ring_threads); the "
                        "Gamma-only ring pool widens up to this, capped at 64 "
                        "(one socket -- the per-cell efficient ceiling)")
    p.add_argument("--cells", nargs="*", default=None,
                   help="run only these tags (default: the full matrix)")
    p.add_argument("--no-concurrent", action="store_true",
                   help="run Gamma-only (nranks==1) cells one at a time instead "
                        "of one-per-NUMA-socket concurrently (the default fills "
                        "the whole 2-socket node; films keep their own MPI layout)")
    p.add_argument("--eta0", action="store_true",
                   help="redo the study at eta=0 (1e-12): zero the broadening and "
                        "let the anharmonic Sigma^R self-broaden. Adds periodic-"
                        "Pulay (anderson_period=4) to break the eta=0 marginal-mode "
                        "plateau and raises the d5a soft-mode SSE cutoff to 0.7 THz. "
                        "Writes to <study>_eta0/ (the finite-eta production is kept).")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    cells, man = matrix(a.study)
    if a.eta0:
        # eta -> 0: the physical limit (the conservation floor D(omega) ~ eta
        # vanishes and the eta-absorption dip goes to 0); validated on cnt33
        # (converges, ~15% higher/true transmission, machine-precision baseline
        # conservation). Periodic-Pulay breaks the marginal-mode limit cycle that
        # plain Anderson hits at eta=0 (d5a 1.6e-3 -> 4.5e-5); d5a additionally
        # needs the soft twist band (< ~0.7 THz) excluded from the SSE.
        for c in cells:
            cfg = c["cfg"]
            cfg["eta"] = 1e-12
            cfg["band_limit"] = True   # generic auto band-support cutoff (both edges)
            cfg.pop("sse_cutoff", None)  # superseded by band_limit (no hand-set freq)
            # CAUSAL retarded self-energy at eta=0 (2026-06-16). The "half" rule
            # (Sigma^R = anti-Hermitian part only) DROPS the Kramers-Kronig real
            # part; paired with the Phi-derivable bubble Sigma^<> it violates
            # per-interface heat conservation and DESTROYS the SCBA fixed point on
            # longer cells -- cnt33 L3 diverges (linear/Broyden to 1e9) or
            # limit-cycles (Anderson, lead balance -> 0.9) with a negative,
            # unphysical heat current [177,-29,-528,-76]. The fft KK real part
            # restores conservation (lead balance 0.9 -> 1e-5) and a genuine fixed
            # point (cnt33 L3: residual 9.9e-4, heat [39.3,36.3,36.1,39.3]).
            cfg["retarded"] = "fft"
            # The causal map is contractive under PLAIN LINEAR mixing; its
            # marginal eta=0 mode makes Anderson plateau (~0.1) and Broyden diverge
            # (verified), so force linear (the CLAUDE.md default) and lift the cap
            # -- linear needs ~300 its to reach 1e-3 (cells stop early at the gate).
            cfg["mixing_method"] = "linear"
            cfg.pop("anderson_depth", None)
            cfg.pop("anderson_period", None)
            cfg["max_iter"] = max(int(cfg.get("max_iter", 100)), 450)
            # Longer cells have a denser near-marginal spectrum: beta=0.2
            # oscillates/floors on L>=4 (L4 spikes to ~8 then floors at ~8e-3
            # Sigma resid; verified beta 0.10/0.15/0.20 all floor ~8e-3). Use a
            # smaller step for L>=4 -- it removes the overshoot spikes and gives
            # the best best-iterate, though the marginal mode still prevents the
            # tight 1e-3 Sigma gate (heat current conserves; lead J ~37.4 is
            # beta-robust). Genuine pure-eta=0 limit for the longest cell.
            if int(cfg.get("ncells", 1)) >= 4:
                cfg["mix"] = 0.1
    # Biggest devices first (cost ~ device size x q-points): finish the
    # expensive cells before the cheap tail.
    cells.sort(key=lambda c: c["length"] * (c["nk"] ** 2 if c["nk"] else 1),
               reverse=True)
    # Ring-pool width for the Gamma-only cells (nranks==1): cells run one at a
    # time (node hygiene), so each takes the whole budget. The per-ring bubble
    # is cache-bound per chunk and scales ~56x to 64 threads (measured cnt33 L2,
    # w=241: 814 GF/s @32 -> 1232 GF/s @64); the SSE floors the actual split at
    # n_tau//4 so large-w cells use more and small-w cells don't over-fragment.
    # The film's q-MPI cells keep their own rank layout.
    nthr = max(16, min(64, a.rank_budget))
    for c in cells:
        if c["nranks"] == 1:
            c["ring_threads"] = nthr

    out_dir = OUT_ROOT / (a.study + ("_eta0" if a.eta0 else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    # carry per-cell work dir into the manifest (for structure.xyz / A_c)
    for c, m in zip(cells, man):
        key = f"{c['system']}_L{c['length']}" + (f"_nk{c['nk']}" if c['nk'] else "")
        m["work"] = str(WORK_ROOT / key)
    (out_dir / "manifest.json").write_text(
        json.dumps({"study": a.study, "cells": man}, indent=2))

    if a.cells:
        cells = [c for c in cells if c["tag"] in a.cells]

    for c in cells:
        c["cores"] = int(c["nranks"]) * int(c["ring_threads"])
    mode = "one-per-NUMA-socket concurrent" if not a.no_concurrent else "sequential"
    print(f"=== study {a.study}: {len(cells)} cells -> {out_dir} "
          f"(Gamma-only: {mode}; budget {a.rank_budget} cores) ===", flush=True)
    for c in cells:
        print(f"  {c['tag']:14s} nranks={c['nranks']} x ring{c['ring_threads']} "
              f"= {c['cores']:3d} cores  {c['cfg']}", flush=True)
    if a.dry_run:
        return 0

    # build all geometries up front (cheap, and keeps the run loop launch-only)
    for c in cells:
        c["geom"] = str(geometry_dir(c["system"], c["length"], c["nk"]))

    def _matches(npz_path, cfg_dict):
        """True iff an existing snapshot was computed at the SAME (eta, grid,
        retarded_method) as this cell. Guards against the resume logic reusing a
        STALE ballistic (wrong eta/nfreq) as the reference for a new anharmonic
        run -- which silently corrupts every G_anh/G_ball ratio (2026-06-13) --
        AND against reusing a stale retarded_method snapshot: at eta=0 "half"
        (anti-Hermitian Sigma^R) and "fft" (causal Kramers-Kronig Sigma^R) give a
        DIFFERENT fixed point (half diverges on L3/L4; fft converges), so a
        half-rule npz must NOT satisfy an fft request (2026-06-17)."""
        if not npz_path.exists():
            return False
        try:
            d = np.load(npz_path, allow_pickle=True)
            eta_ok = abs(float(d["eta"]) - float(cfg_dict["eta"])) < 1e-9
            ne_ok = len(d["energies"]) == int(cfg_dict["nfreq"])
            ret_have = str(d["retarded"]) if "retarded" in d else "half"
            ret_ok = ret_have == str(cfg_dict.get("retarded", "half"))
            return eta_ok and ne_ok and ret_ok
        except Exception:
            return False

    jobs = []   # one solver invocation each (ballistic or anharmonic)
    for c in cells:
        tag = c["tag"]
        ball_npz = out_dir / f"{tag}_ball.npz"
        anh_npz = out_dir / f"{tag}_anh.npz"
        ball_ok = _matches(ball_npz, c["cfg"])
        anh_ok = _matches(anh_npz, c["cfg"])
        if ball_ok and (anh_ok or not c["do_anh"]):
            print(f"[skip] {tag} (matched npz present)", flush=True)
            continue
        # purge stale (eta/grid-mismatched) snapshots so they regenerate matched
        if ball_npz.exists() and not ball_ok:
            ball_npz.unlink()
            print(f"[stale] {tag}_ball.npz (eta/grid mismatch) -> regenerate",
                  flush=True)
        if anh_npz.exists() and not anh_ok:
            anh_npz.unlink()
        cw = cell_workdir(c["geom"], tag, out_dir)
        cfg = pipeline.write_config(c["system"], cw, **c["cfg"])
        if not ball_npz.exists():
            jobs.append(dict(config=cfg, npz=ball_npz,
                             log=out_dir / f"{tag}_ball.log", ballistic=True,
                             env={"QX_MAXIT": 3, "QX_MINIT": 1},
                             nranks=c["nranks"], ring_threads=c["ring_threads"],
                             tag=f"{tag}/ball"))
        if c["do_anh"] and not anh_npz.exists():
            jobs.append(dict(config=cfg, npz=anh_npz,
                             log=out_dir / f"{tag}_anh.log",
                             nranks=c["nranks"], ring_threads=c["ring_threads"],
                             tag=f"{tag}/anh"))

    # Gamma-only single-rank cells fill the 2-socket node ONE PER NUMA SOCKET
    # (concurrent, NUMA-pinned -- a single cell is capped at ~one socket, so this
    # is the real full-node lever); films (nranks>1) keep their own MPI rank
    # layout and run one mpirun at a time.
    gamma = [j for j in jobs if int(j["nranks"]) == 1]
    films = [j for j in jobs if int(j["nranks"]) > 1]

    def _run_serial(j):
        rc = pipeline.launch_cell(
            j["config"], j["npz"], j["log"], nranks=j["nranks"],
            ring_threads=j["ring_threads"], ballistic=j.get("ballistic", False),
            env=j.get("env"))
        print(f"[{j['tag']}] rc={rc} npz={Path(j['npz']).exists()}", flush=True)

    if gamma:
        if a.no_concurrent or len(gamma) == 1:
            for j in gamma:
                _run_serial(j)
        else:
            print(f"=== {len(gamma)} Gamma-only jobs -> socket-pinned concurrent "
                  f"(fills both NUMA sockets) ===", flush=True)
            pipeline.launch_cells_concurrent(gamma)
    for j in films:
        _run_serial(j)

    print(f"=== study {a.study} COMPLETE; summarizing -> {out_dir} ===",
          flush=True)
    summarize.summarize(out_dir, out_dir, False)
    return 0


# --------------------------- spectral transmission ---------------------------
# Uses the per-omega Meir-Wingreen current spectrum saved by engine/run.py
# (``current_spectrum``, shape (ne, *nk, n_interfaces)). With the lead
# self-energies Sigma^< = i Gamma n, Sigma^> = i Gamma (n+1), the raw
# per-energy lead trace is EXACTLY
#
#     I(w) = T(w) * (n_L(w) - n_R(w)),
#
# so the (effective) transmission is T(w) = I(w) / dn(w) -- dimensionless,
# exact for the ballistic run, and the standard "effective transmission"
# for the anharmonic run (Buettiker-probe-like interpretation). Transverse
# q-axes are summed (total transmission of all transverse channels).

def bose(freq_thz, temp):
    x = HBAR_EV * THZ_TO_RAD * np.asarray(freq_thz) / (KB_EV * temp)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        nb = 1.0 / np.expm1(x)
    return np.where(np.isfinite(nb), nb, 0.0)


def lead_spectrum(npz):
    """(freqs, T_eff(w), j(w)) from one snapshot; None if no spectrum."""
    if "current_spectrum" not in npz.files:
        return None
    spec = np.asarray(npz["current_spectrum"])      # (ne, *nk, n_int)
    freqs = np.abs(np.asarray(npz["energies"]).real)
    # sum any transverse-q axes -> (ne, n_int); average the two leads
    while spec.ndim > 2:
        spec = np.nansum(spec, axis=1)
    # Signed hot-lead spectrum (eq. T_eff = J_L(w)/(n_L - n_R)); abs()
    # would silently rectify negative (backscattered) bins.
    lead = np.sign(np.nansum(spec[:, 0].real)) * spec[:, 0].real
    if freqs[0] < 1e-6:
        lead = lead.copy()
        lead[0] = 0.0  # omega=0 bin carries no heat; mask the DC artifact
    dn = bose(freqs, float(npz["t_left"])) - bose(freqs, float(npz["t_right"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_eff = np.where(dn > 0, lead / dn, 0.0)
    return freqs, t_eff, freqs * lead


def plot_transmission(run_dir, tags=None):
    """Effective transmission + spectral/cumulative heat current per cell;
    ballistic vs anharmonic overlay. Figures -> phonon/studies/out/fig."""
    from phonon.studies import style
    import matplotlib.pyplot as plt

    rd = Path(run_dir)
    tags = tags or sorted(
        p.name[:-len("_ball.npz")] for p in rd.glob("*_ball.npz"))

    for tag in tags:
        panels = {}
        for kind in ("ball", "anh"):
            f = rd / f"{tag}_{kind}.npz"
            if not f.is_file():
                continue
            data = lead_spectrum(np.load(f, allow_pickle=True))
            if data is not None:
                panels[kind] = data
        if not panels:
            print(f"{tag}: no current_spectrum in NPZs (rerun with the "
                  "updated engine/run.py to record spectra)")
            continue

        fig, (ax1, ax2, ax3) = style.figure(ncols=3, width=4.2, height=3.4)
        line_kw = {"ball": dict(color="C0", label="ballistic"),
                   "anh": dict(color="C3", label="anharmonic")}
        for kind, (freqs, t_eff, jw) in panels.items():
            ax1.plot(freqs, t_eff, **line_kw[kind])
            ax2.plot(freqs, jw, **line_kw[kind])
            cum = np.cumsum(jw)
            ax3.plot(freqs, cum / max(cum[-1], 1e-300), **line_kw[kind])
        ax1.set_xlabel("frequency (THz)")
        ax1.set_ylabel(r"$T_{\rm eff}(\omega)$ (channels)")
        ax1.legend()
        ax2.set_xlabel("frequency (THz)")
        ax2.set_ylabel(r"spectral heat current $\omega I(\omega)$")
        ax3.set_xlabel("frequency (THz)")
        ax3.set_ylabel(r"cumulative $G(<\omega)/G$")
        fig.suptitle(f"{rd.name} {tag}", fontsize=10)
        style.save(fig, f"{rd.name}_{tag}_transmission")
        plt.close(fig)


# --------------------------- document figures ---------------------------
# Consumes the per-study ``summary.csv`` (legacy schema: sweep, t_mean,
# n_slabs, G_ball_W_per_m2_K, G_anh_W_per_m2_K, ratio) and renders the figures
# referenced by ``document/src/results.tex`` into
# ``document/fig/transport_sweeps/`` under their existing names, so reruns
# drop in without touching the .tex includes. Ratios are the exact
# (normalization-free) quantity; absolute G uses the dense-matching analytic
# constant from the summary.

def load_summary(study):
    import csv
    p = OUT_ROOT / study / "summary.csv"
    if not p.exists():
        return []
    with open(p) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("t_mean", "n_slabs", "G_ball_W_per_m2_K", "G_anh_W_per_m2_K", "ratio"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else None
    return rows


def temp_rows(rows):
    return sorted([r for r in rows if r["sweep"] == "temperature" and r["ratio"]],
                  key=lambda r: r["t_mean"])


def len_rows(rows):
    return sorted([r for r in rows if r["sweep"] == "length" and r["ratio"]],
                  key=lambda r: r["n_slabs"])


def fig_cnt(rows):
    from phonon.studies import style
    import matplotlib.pyplot as plt

    T = temp_rows(rows)
    if T:
        Ts = [r["t_mean"] for r in T]
        # NB: the document figures cnt33_temperature_{g,ratio} are owned by
        # phonon/scripts/figures/cnt33_finite_eta_bias.py (archived dense
        # reference + eta=0 overlay); this run-summary plot uses its own names
        # so a stale/unconverged production sweep can never overwrite them.
        fig, ax = style.figure(width=5.0, height=3.8)
        ax.plot(Ts, [r["G_ball_W_per_m2_K"] for r in T], "o-", color="C0", label=r"$G_{\rm ball}$")
        ax.plot(Ts, [r["G_anh_W_per_m2_K"] for r in T], "s-", color="C1", label=r"$G_{\rm anh}$")
        ax.set_xlabel("temperature (K)"); ax.set_ylabel(r"$G$ (W m$^{-2}$ K$^{-1}$)")
        ax.legend(); ax.set_title("(3,3) CNT production, L=2", fontsize=9)
        style.save(fig, "prod_cnt_temperature_g", directory=DOC_FIG); plt.close(fig)
        fig, ax = style.figure(width=5.0, height=3.8)
        ax.plot(Ts, [r["ratio"] for r in T], "^-", color="C3")
        ax.set_xlabel("temperature (K)"); ax.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$")
        ax.set_title("(3,3) CNT production ratio vs $T$", fontsize=9)
        style.save(fig, "prod_cnt_temperature_ratio", directory=DOC_FIG); plt.close(fig)
    L = len_rows(rows)
    if L:
        Ls = [r["n_slabs"] for r in L]
        fig, ax = style.figure(width=5.0, height=3.8)
        ax.plot(Ls, [r["ratio"] for r in L], "o-", color="C0")
        ax.set_xlabel("device length $L$ (transport cells)")
        ax.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$")
        ax.set_xticks(Ls); ax.set_title("(3,3) CNT length ladder (production RGF)", fontsize=9)
        style.save(fig, "prod_cnt_ladder", directory=DOC_FIG); plt.close(fig)
    print(f"  cnt33 figs: temp={len(T)} len={len(L)}")


def fig_sinw(d5, d11):
    """Overlay d5a + d11a: prod_sinw_vs_T_d5_d11, prod_sinw_vs_length_d5_d11."""
    from phonon.studies import style
    import matplotlib.pyplot as plt

    styles = {"d5a": dict(color="C0", marker="o"), "d11a": dict(color="C1", marker="s")}
    data = {"d5a": d5, "d11a": d11}
    # temperature
    fig, (axg, axr) = style.figure(ncols=2, width=4.5, height=3.8)
    any_t = False
    for name, rows in data.items():
        T = temp_rows(rows)
        if not T:
            continue
        any_t = True
        Ts = [r["t_mean"] for r in T]
        axg.plot(Ts, [r["G_ball_W_per_m2_K"] for r in T], ls="--", **styles[name])
        axg.plot(Ts, [r["G_anh_W_per_m2_K"] for r in T], ls="-",
                 label=name, **styles[name])
        axr.plot(Ts, [r["ratio"] for r in T], ls="-", label=name, **styles[name])
    if any_t:
        axg.set_xlabel("T (K)"); axg.set_ylabel(r"$G$ (W m$^{-2}$K$^{-1}$)")
        axg.set_title("SiNW (dashed ballistic, solid anharmonic)", fontsize=9); axg.legend()
        axr.set_xlabel("T (K)"); axr.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$"); axr.legend()
        style.save(fig, "prod_sinw_vs_T_d5_d11", directory=DOC_FIG)
    plt.close(fig)
    # length
    fig, ax = style.figure(width=5.2, height=3.8)
    any_l = False
    for name, rows in data.items():
        L = len_rows(rows)
        if not L:
            continue
        any_l = True
        Ls = [r["n_slabs"] for r in L]
        ax.plot(Ls, [r["ratio"] for r in L], ls="-", label=name, **styles[name])
    if any_l:
        ax.set_xlabel("device length $L$ (cells)"); ax.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$")
        ax.legend(); ax.set_title("SiNW length ladder (production)", fontsize=9)
        style.save(fig, "prod_sinw_vs_length_d5_d11", directory=DOC_FIG)
    plt.close(fig)
    print(f"  sinw figs: d5a={len(d5)} d11a={len(d11)} rows")


def fig_film(rows):
    """prod_si_film_vs_guo: G vs thickness with the Guo reference points."""
    from phonon.studies import style
    import matplotlib.pyplot as plt

    th = sorted([r for r in rows if r["sweep"] == "thickness" and r["ratio"]],
                key=lambda r: r["n_slabs"])
    if not th:
        print("  film: no thickness rows yet")
        return
    # 1 unit cell = 5.4018 A; thickness(nm) = n_slabs * 0.54018 (1 slab = 1 uc here)
    nm = [r["n_slabs"] * 0.54018 for r in th]
    fig, ax = style.figure(width=5.6, height=4.0)
    # Production values are W/m^2/K; the Guo literals are MW/m^2/K.
    ax.plot(nm, [r["G_ball_W_per_m2_K"] * 1e-6 for r in th], "o-", color="C0", label=r"$G_{\rm ball}$ (prod)")
    ax.plot(nm, [r["G_anh_W_per_m2_K"] * 1e-6 for r in th], "s-", color="C1", label=r"$G_{\rm anh}$ (prod)")
    # Guo et al. PRB 102 195412 (2020) anharmonic reference points
    ax.plot([3 * 0.54018, 5 * 0.54018], [939.72, 890.97], "D", color="k",
            ms=7, label="Guo 2020 $G_{\\rm anh}$")
    ax.set_xlabel("film thickness (nm)"); ax.set_ylabel(r"$G$ (MW m$^{-2}$K$^{-1}$)")
    ax.legend(); ax.set_title("Si film cross-plane conductance vs Guo 2020", fontsize=9)
    style.save(fig, "prod_si_film_vs_guo", directory=DOC_FIG); plt.close(fig)
    print(f"  film fig: {len(th)} thickness rows")


def plot_figures(studies):
    data = {s: load_summary(s) for s in studies}
    print("rows per study:", {s: len(r) for s, r in data.items()})
    if data.get("cnt33"):
        fig_cnt(data["cnt33"])
    if data.get("sinw_d5a") or data.get("sinw_d11a"):
        fig_sinw(data.get("sinw_d5a", []), data.get("sinw_d11a", []))
    if data.get("sifilm"):
        fig_film(data["sifilm"])
    print(f"figures -> {DOC_FIG}")


def plot(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m phonon.studies transport plot", description=__doc__)
    p.add_argument("--what", choices=("transmission", "figures"),
                   default="figures")
    p.add_argument("--run-dir", default=None,
                   help="transmission: study run dir "
                        "(e.g. phonon/scripts/out/prod/sinw_d5a)")
    p.add_argument("--tags", nargs="*", default=None,
                   help="transmission: default every tag with a *_ball.npz")
    p.add_argument("--studies", nargs="*", default=list(STUDIES),
                   help="figures: studies whose summary.csv to render")
    a = p.parse_args(argv)
    if a.what == "transmission":
        if not a.run_dir:
            p.error("--what transmission requires --run-dir")
        plot_transmission(a.run_dir, a.tags)
    else:
        plot_figures(a.studies)
    return 0
