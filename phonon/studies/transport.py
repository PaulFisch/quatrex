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
              "phonon_energies.npy", "qfold_vertices.npz", "kshift.npy")


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
                        emin=emin, dt=dt))

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
    AND = dict(mixing_method="anderson", anderson_depth=5, mix=0.1)
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
            fmax, nfreq, eta = 18.0, 41, 0.11
            mixarg = dict(mix=0.05, sigma_tol=3e-2)
        else:
            fmax, nfreq, eta = 18.0, 161, 0.225
            mixarg = dict(**AND, sigma_tol=3e-2)
        for L in (2, 3):
            add(study, L, f"L{L}", "length", 300, L,
                dict(ncells=L, temperature=300, eta=eta, **mixarg, nfreq=nfreq,
                     fmax=fmax, max_iter=100, bcs=1, qcs=1),
                1, fmax=fmax, nfreq=nfreq, ring_threads=16)
        # temperature sweep at L=2 (d5a soft: low T converges; high T may not)
        Ts = (30, 50, 100, 150) if study == "sinw_d5a" else (200, 300, 400)
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
                        "Gamma-only ring pool widens up to this, capped at 32")
    p.add_argument("--cells", nargs="*", default=None,
                   help="run only these tags (default: the full matrix)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    cells, man = matrix(a.study)
    # Biggest devices first (cost ~ device size x q-points): finish the
    # expensive cells before the cheap tail.
    cells.sort(key=lambda c: c["length"] * (c["nk"] ** 2 if c["nk"] else 1),
               reverse=True)
    # Ring-pool width for the Gamma-only cells (nranks==1): cells run one at a
    # time (node hygiene), so each may take the whole budget -- capped at 32,
    # beyond which the bandwidth-bound bubble loses per-thread efficiency. The
    # film's q-MPI cells keep their own rank layout.
    nthr = max(16, min(32, a.rank_budget))
    for c in cells:
        if c["nranks"] == 1:
            c["ring_threads"] = nthr

    out_dir = OUT_ROOT / a.study
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
    print(f"=== study {a.study}: {len(cells)} cells -> {out_dir} "
          f"(sequential, budget {a.rank_budget} cores) ===", flush=True)
    for c in cells:
        print(f"  {c['tag']:14s} nranks={c['nranks']} x ring{c['ring_threads']} "
              f"= {c['cores']:3d} cores  {c['cfg']}", flush=True)
    if a.dry_run:
        return 0

    # build all geometries up front (cheap, and keeps the run loop launch-only)
    for c in cells:
        c["geom"] = str(geometry_dir(c["system"], c["length"], c["nk"]))

    for c in cells:
        tag = c["tag"]
        ball_npz = out_dir / f"{tag}_ball.npz"
        anh_npz = out_dir / f"{tag}_anh.npz"
        if ball_npz.exists() and (anh_npz.exists() or not c["do_anh"]):
            print(f"[skip] {tag} (npz present)", flush=True)
            continue
        cw = cell_workdir(c["geom"], tag, out_dir)
        cfg = pipeline.write_config(c["system"], cw, **c["cfg"])
        print(f"[start] {tag} nranks={c['nranks']} "
              f"ring_threads={c['ring_threads']}", flush=True)
        if not ball_npz.exists():
            rc = pipeline.launch_cell(
                cfg, ball_npz, out_dir / f"{tag}_ball.log",
                nranks=c["nranks"], ring_threads=c["ring_threads"],
                ballistic=True, env={"QX_MAXIT": 3, "QX_MINIT": 1})
            print(f"[ball] {tag} rc={rc} npz={ball_npz.exists()}", flush=True)
        if c["do_anh"] and not anh_npz.exists():
            rc = pipeline.launch_cell(
                cfg, anh_npz, out_dir / f"{tag}_anh.log",
                nranks=c["nranks"], ring_threads=c["ring_threads"])
            print(f"[anh ] {tag} rc={rc} npz={anh_npz.exists()}", flush=True)

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
    lead = 0.5 * (np.abs(spec[:, 0]) + np.abs(spec[:, -1]))
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
        # cnt33_temperature_g : G_ball + G_anh vs T
        fig, ax = style.figure(width=5.0, height=3.8)
        ax.plot(Ts, [r["G_ball_W_per_m2_K"] for r in T], "o-", color="C0", label=r"$G_{\rm ball}$")
        ax.plot(Ts, [r["G_anh_W_per_m2_K"] for r in T], "s-", color="C1", label=r"$G_{\rm anh}$")
        ax.set_xlabel("temperature (K)"); ax.set_ylabel(r"$G$ (W m$^{-2}$ K$^{-1}$)")
        ax.legend(); ax.set_title("(3,3) CNT production, L=2", fontsize=9)
        style.save(fig, "cnt33_temperature_g", directory=DOC_FIG); plt.close(fig)
        # cnt33_temperature_ratio
        fig, ax = style.figure(width=5.0, height=3.8)
        ax.plot(Ts, [r["ratio"] for r in T], "^-", color="C3")
        ax.set_xlabel("temperature (K)"); ax.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$")
        ax.set_title("(3,3) CNT: channel freezing as $T\\to0$", fontsize=9)
        style.save(fig, "cnt33_temperature_ratio", directory=DOC_FIG); plt.close(fig)
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
    """Overlay d5a + d11a: ballistic_vs_T_d5_d11, ballistic_vs_length_d5_d11."""
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
        style.save(fig, "ballistic_vs_T_d5_d11", directory=DOC_FIG)
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
        style.save(fig, "ballistic_vs_length_d5_d11", directory=DOC_FIG)
    plt.close(fig)
    print(f"  sinw figs: d5a={len(d5)} d11a={len(d11)} rows")


def fig_film(rows):
    """si_film_vs_guo: G vs thickness with the Guo reference points."""
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
    ax.plot(nm, [r["G_ball_W_per_m2_K"] for r in th], "o-", color="C0", label=r"$G_{\rm ball}$ (prod)")
    ax.plot(nm, [r["G_anh_W_per_m2_K"] for r in th], "s-", color="C1", label=r"$G_{\rm anh}$ (prod)")
    # Guo et al. PRB 102 195412 (2020) anharmonic reference points
    ax.plot([3 * 0.54018, 5 * 0.54018], [939.72, 890.97], "D", color="k",
            ms=7, label="Guo 2020 $G_{\\rm anh}$")
    ax.set_xlabel("film thickness (nm)"); ax.set_ylabel(r"$G$ (MW m$^{-2}$K$^{-1}$ scale)")
    ax.legend(); ax.set_title("Si film cross-plane conductance vs Guo 2020", fontsize=9)
    style.save(fig, "si_film_vs_guo", directory=DOC_FIG); plt.close(fig)
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
