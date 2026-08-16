"""Strong-coupling SCBA convergence investigation.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from phonon.studies import pipeline, style

OUT = pipeline.OUT / "convergence"


def _cell(tag: str, work: Path, *, env: dict, ring_threads: int,
          warm_from: Path | None = None, sigma_scale: float = 1.0,
          check_idle: bool = True) -> dict:
    """Run one convergence cell; returns its trace summary."""
    cell_env = dict(env)
    sigma_out = OUT / f"{tag}_sigma.npz"
    cell_env["QX_SAVE_SIGMA"] = sigma_out
    if warm_from is not None:
        cell_env["QX_SIGMA_INIT"] = warm_from
        cell_env["QX_SIGMA_SCALE"] = sigma_scale
    rc = pipeline.launch_cell(
        work / "quatrex_config.toml", OUT / f"{tag}.npz", OUT / f"{tag}.log",
        nranks=1, ring_threads=ring_threads, env=cell_env,
        check_idle=check_idle,
    )
    return _cell_summary(tag, rc)


def _cell_summary(tag: str, rc: int) -> dict:
    """Parse a finished cell's log into the trace summary (shared by the
    sequential and socket-pinned-concurrent ``mix`` paths)."""
    sigma_out = OUT / f"{tag}_sigma.npz"
    trace = pipeline.parse_scba_trace(OUT / f"{tag}.log")
    res = trace["residual"]
    summary = {
        "tag": tag, "rc": rc,
        "iterations": int(res.size),
        "final_residual": float(res[-1]) if res.size else float("nan"),
        "min_residual": float(res.min()) if res.size else float("nan"),
        "converged": bool(res.size and res[-1] < 1e-3),
        "sigma_snapshot": str(sigma_out) if sigma_out.exists() else None,
    }
    print(f"[convergence] {tag}: it={summary['iterations']} "
          f"final={summary['final_residual']:.3e} "
          f"converged={summary['converged']}", flush=True)
    return summary


def _mix_job(tag: str, work: Path, ring_threads: int) -> dict:
    """A launch_cells_concurrent job dict for one independent ``mix`` cell."""
    return dict(config=work / "quatrex_config.toml", npz=OUT / f"{tag}.npz",
                log=OUT / f"{tag}.log", ring_threads=ring_threads, nranks=1,
                env={"QX_SAVE_SIGMA": str(OUT / f"{tag}_sigma.npz")}, tag=tag)


def _prepare_work(args, tag: str, **config_kwargs) -> Path:
    """Per-cell work dir hard-linked to the shared geometry + fresh config."""
    import os
    geom = pipeline.GEOM / args.geometry
    work = OUT / "work" / tag
    work.mkdir(parents=True, exist_ok=True)
    for f in geom.iterdir():
        dst = work / f.name
        if not dst.exists() and f.is_file():
            os.link(f, dst)
    pipeline.write_config(args.system, work, **config_kwargs)
    return work


def run(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", required=True,
                   choices=["mix", "lambda", "anneal"])
    p.add_argument("--system", default="cnt33")
    p.add_argument("--geometry", default="cnt33_L2",
                   help="geometry dir under the shared geom/ tree")
    p.add_argument("--ncells", type=int, default=2)
    p.add_argument("--eta", type=float, default=0.45)
    p.add_argument("--nfreq", type=int, default=121)
    p.add_argument("--fmax", type=float, default=55.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--dt", type=float, default=10.0,
                   help="lead temperature drop (K)")
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument("--mix", type=float, default=0.1)
    p.add_argument("--mix-grid", type=float, nargs="*",
                   default=[0.02, 0.05, 0.1, 0.4])
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--temps", type=float, nargs="+",
                   default=[100.0, 200.0, 300.0])
    p.add_argument("--anderson", action="store_true",
                   help="mix strategy: also run Anderson(m) at each factor")
    p.add_argument("--anderson-grid", nargs="+", default=[],
                   metavar="DEPTH:BETA",
                   help="mix strategy: explicit Anderson cells, e.g. "
                        "'5:0.05 8:0.1' (independent of --mix-grid)")
    p.add_argument("--ring-threads", type=int, default=64)  # cnt33 w=241 sweet spot (56x); SSE floors at n_tau//4
    p.add_argument("--allow-concurrent", action="store_true",
                   help="skip the node-idle check (the cell budget is on you)")
    p.add_argument("--no-concurrent", action="store_true",
                   help="'mix' strategy: run cells one at a time instead of "
                        "one-per-NUMA-socket (the default fills both sockets; "
                        "lambda/anneal are warm-start chains and stay sequential)")
    p.add_argument("--tag-prefix", default="",
                   help="prefix for cell tags/logs (namespace per system, "
                        "e.g. 'd5a_')")
    p.add_argument("--sse-cutoff", type=float, default=0.0,
                   help="SSE low-frequency cutoff in THz (soft-mode systems)")
    p.add_argument("--retarded", default="half", choices=["half", "fft"],
                   help="Sigma^R rule: 'half' (anti-Hermitian only, drops the "
                        "real part) or 'fft' (causal Kramers-Kronig real part "
                        "via the Hilbert transform)")
    p.add_argument("--band-limit", action="store_true",
                   help="band-limit the SSE above the harmonic band top (eta->0)")
    p.add_argument("--broyden", action="store_true",
                   help="mix strategy: also run type-I Broyden at each factor")
    p.add_argument("--anderson-period", type=int, default=1,
                   help="anderson periodic-Pulay stride (prod L3 used 4)")
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    base_cfg = dict(ncells=args.ncells, eta=args.eta, nfreq=args.nfreq,
                    fmax=args.fmax, temperature=args.temperature,
                    dt=args.dt, max_iter=args.max_iter, retarded=args.retarded)
    if args.sse_cutoff > 0.0:
        base_cfg["sse_cutoff"] = args.sse_cutoff
    if args.band_limit:
        base_cfg["band_limit"] = True
    results = []

    if args.strategy == "mix":
        # The mix cells are INDEPENDENT -> fill both NUMA sockets concurrently
        # (one cell/socket). Prepare them all, then launch.
        jobs = []
        for mix in args.mix_grid:
            tag = f"{args.tag_prefix}mix_lin{mix:g}"
            jobs.append(_mix_job(tag, _prepare_work(args, tag, **base_cfg, mix=mix),
                                 args.ring_threads))
            if args.anderson:
                tag = f"{args.tag_prefix}mix_and{mix:g}"
                jobs.append(_mix_job(tag, _prepare_work(
                    args, tag, **base_cfg, mix=mix, mixing_method="anderson",
                    anderson_period=args.anderson_period),
                    args.ring_threads))
            if args.broyden:
                tag = f"{args.tag_prefix}mix_broy{mix:g}"
                jobs.append(_mix_job(tag, _prepare_work(
                    args, tag, **base_cfg, mix=mix, mixing_method="broyden"),
                    args.ring_threads))
        for spec in args.anderson_grid:
            depth, beta = spec.split(":")
            tag = f"{args.tag_prefix}mix_and_d{int(depth)}_b{float(beta):g}"
            jobs.append(_mix_job(tag, _prepare_work(
                args, tag, **base_cfg, mix=float(beta),
                mixing_method="anderson", anderson_depth=int(depth)),
                args.ring_threads))

        if not args.no_concurrent and not args.allow_concurrent and len(jobs) > 1:
            rcs = pipeline.launch_cells_concurrent(jobs)
            results.extend(_cell_summary(j["tag"], rcs.get(str(j["npz"]), -1))
                           for j in jobs)
        else:  # sequential (single cell, --no-concurrent, or --allow-concurrent)
            for j in jobs:
                rc = pipeline.launch_cell(
                    j["config"], j["npz"], j["log"], nranks=1,
                    ring_threads=j["ring_threads"], env=j["env"],
                    check_idle=not args.allow_concurrent)
                results.append(_cell_summary(j["tag"], rc))

    elif args.strategy == "lambda":
        if args.anderson:
            base_cfg["mixing_method"] = "anderson"
        warm, lam_prev = None, None
        for lam in args.lambdas:
            tag = f"{args.tag_prefix}lam{lam:g}_T{args.temperature:g}"
            work = _prepare_work(args, tag, **base_cfg, mix=args.mix,
                                 vertex_scale=lam)
            scale = 1.0 if lam_prev is None else (lam / lam_prev) ** 2
            summary = _cell(tag, work, env={}, warm_from=warm,
                            sigma_scale=scale,
                            ring_threads=args.ring_threads)
            results.append(summary)
            if summary["sigma_snapshot"]:
                warm, lam_prev = Path(summary["sigma_snapshot"]), lam
            else:
                print(f"[convergence] {tag} left no Sigma snapshot; "
                      "continuation chain broken", flush=True)
                break

    elif args.strategy == "anneal":
        warm = None
        for temp in args.temps:
            tag = f"{args.tag_prefix}anneal_T{temp:g}"
            cfg = dict(base_cfg, temperature=temp, mix=args.mix)
            work = _prepare_work(args, tag, **cfg)
            summary = _cell(tag, work, env={}, warm_from=warm,
                            sigma_scale=1.0,
                            ring_threads=args.ring_threads)
            results.append(summary)
            if summary["sigma_snapshot"]:
                warm = Path(summary["sigma_snapshot"])

    (OUT / f"{args.tag_prefix}{args.strategy}_summary.json").write_text(
        json.dumps(results, indent=2))
    n_conv = sum(r["converged"] for r in results)
    print(f"[convergence] {args.strategy}: {n_conv}/{len(results)} "
          f"cells converged; summary -> {OUT}/{args.strategy}_summary.json",
          flush=True)
    return 0


def plot(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="convergence_traces")
    p.add_argument("--glob", default="*.log",
                   help="log-file pattern under the convergence out dir "
                        "(e.g. 'mix_*.log' for the mixing sweep)")
    args = p.parse_args(argv)

    logs = sorted(OUT.glob(args.glob))
    if not logs:
        print(f"no convergence logs under {OUT}")
        return 1
    fig, axes = style.figure(ncols=3, width=4.0)
    for log in logs:
        trace = pipeline.parse_scba_trace(log)
        if not trace["residual"].size:
            continue
        it = np.arange(1, trace["residual"].size + 1)
        axes[0].semilogy(it, trace["residual"], label=log.stem)
        axes[1].semilogy(it, np.maximum(trace["lead_balance"], 1e-16))
        if trace["bubble_balance"].size:
            axes[2].semilogy(np.arange(1, trace["bubble_balance"].size + 1),
                             np.maximum(trace["bubble_balance"], 1e-18))
    axes[0].set_xlabel("SCBA iteration")
    axes[0].set_ylabel(r"rel $\Sigma^R$ residual")
    axes[0].axhline(1e-3, color="k", lw=0.7, ls="--")
    axes[0].legend(fontsize=6)
    axes[1].set_xlabel("SCBA iteration")
    axes[1].set_ylabel(r"lead balance $|J_L-J_R|/|J|$")
    axes[2].set_xlabel("SCBA iteration")
    axes[2].set_ylabel("bubble energy balance")
    style.save(fig, args.name)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run(sys.argv[1:]))
