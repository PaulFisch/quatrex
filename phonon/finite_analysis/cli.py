"""argparse front-end for the finite-structure validation pipeline.

Usage::

    python -m phonon_inputs.finite_analysis \
        --config phonon/configs/sinw/sinw100_d5a_vasp.yaml \
        --analyses fc_quality,sparsity,decomposition,physical \
        --out-dir finite_analysis_out/sinw100

The full analysis set is::

    --analyses all              # every analysis below
    --analyses fc_quality       # FC2/FC3 magnitude binning + dispersion
    --analyses sparsity         # 2D heatmap, 1D decay, 3D scatter, nnz table
    --analyses decomposition    # six-method rank sweep + reconstruction nnz
    --analyses physical         # ASR, perm, Hermiticity, dispersion sanity
    --analyses sse_sparsity     # synthetic + (optional) quatrex Σ blocks
    --analyses cutoffs          # cutoff-hierarchy parametric sweep

Each analysis writes into a per-system subdirectory; a top-level
``summary.json`` rolls up the headline numbers across all analyses.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

from .loader import load_system


ALL_ANALYSES = (
    "fc_quality",
    "sparsity",
    "decomposition",
    "physical",
    "sse_sparsity",
    "cutoffs",
    "sigma_audit",
    "transport_quality",
)


def _parse_analyses(arg: str) -> list[str]:
    if arg.strip().lower() == "all":
        return list(ALL_ANALYSES)
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    bad = [p for p in parts if p not in ALL_ANALYSES]
    if bad:
        raise SystemExit(
            f"unknown analyses: {bad}; choose from {ALL_ANALYSES} or 'all'"
        )
    return parts


def _parse_ranks(arg: str) -> list[int]:
    return [int(x) for x in arg.split(",") if x.strip()]


def _validate_cli(argv: list[str]) -> int:
    """Subcommand: ``validate <config> [--no-color]`` — print parameter
    sanity table; exit code 0/1/2 by max severity."""
    p = argparse.ArgumentParser(
        prog="phonon_inputs.finite_analysis validate",
        description="Sanity-check a YAML config against literature DFT/FC settings.",
    )
    p.add_argument("config", type=Path, help="phonon_inputs YAML config file")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colour in the output table")
    args = p.parse_args(argv)

    import yaml as _yaml

    from phonon_inputs.config import config_from_dict

    from .parameter_validation import (
        validate_config, render_table, max_severity, severity_to_exit_code,
    )

    cfg = config_from_dict(_yaml.safe_load(args.config.read_text()))
    results = validate_config(cfg)
    print(render_table(results, color=not args.no_color))
    return severity_to_exit_code(max_severity(results))


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "validate":
        return _validate_cli(argv[1:])
    if argv and argv[0] == "run":
        argv = argv[1:]  # explicit "run" subcommand stripped before argparse

    p = argparse.ArgumentParser(
        prog="phonon_inputs.finite_analysis",
        description=__doc__.split("Usage::")[0],
    )
    p.add_argument("--config", type=Path, required=True,
                   help="phonon_inputs YAML config file")
    p.add_argument("--fc3-path", type=Path, default=None,
                   help="Override the resolved fc3.hdf5 path")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Per-system output directory")
    p.add_argument("--analyses", type=str, default="fc_quality,sparsity,physical",
                   help="Comma-separated subset of "
                        f"{ALL_ANALYSES} or 'all' (default: a fast subset)")
    p.add_argument("--name", type=str, default=None,
                   help="System name (default: config filename stem)")
    p.add_argument("--n-slabs-hint", type=int, default=None,
                   help="Force this many transport slabs")
    p.add_argument("--transport-axis", type=int, default=2,
                   help="Cartesian axis along which to slab the device")
    p.add_argument("--rank-sweep", type=_parse_ranks, default=[2, 4, 8, 16],
                   help="Comma-separated decomposition ranks (default: 2,4,8,16)")
    p.add_argument("--skip-pcp", action="store_true",
                   help="Skip the PCP fitter in the decomposition sweep")
    p.add_argument("--n-freq-pos", type=int, default=64,
                   help="Positive-frequency grid points for the synthetic GF")
    p.add_argument("--eta-thz", type=float, default=None,
                   help="Lorentzian half-width for the synthetic GF (default: 2 dω)")
    p.add_argument("--temperature", type=float, default=300.0,
                   help="Bose temperature in K (default: 300)")
    p.add_argument(
        "--with-quatrex-crosscheck", action="store_true",
        help=(
            "Also run the dense-supercell SCBA cross-check inside "
            "sse_sparsity. Off by default: it allocates a "
            "(n_fft × n_dof³) complex tensor on the FULL supercell — "
            "~8 GiB for d5a, ~200 GiB for d9a, ~600 GiB for d12a — "
            "and runs single-threaded, so on any system bigger than "
            "the analytic Si chain it stalls the analysis for minutes "
            "or hours. Only enable on small (< 50-atom) supercells "
            "where you want to verify the block-decomposed bubble "
            "against the dense reference."
        ),
    )
    # Legacy alias: --skip-quatrex-run used to flip the default the
    # other way (the cross-check ran unless skipped, which was the
    # wrong default for SiNW-scale systems). Accept but ignore.
    p.add_argument(
        "--skip-quatrex-run", action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--gamma-lead-thz", type=float, default=None,
                   help="Synthetic lead broadening Γ for cutoffs transport "
                        "(default: from finite_analysis.constants)")
    p.add_argument("--T-L", type=float, default=None,
                   help="Left-lead temperature K for Landauer Q (default: 305)")
    p.add_argument("--T-R", type=float, default=None,
                   help="Right-lead temperature K for Landauer Q (default: 295)")
    p.add_argument(
        "--tq-q-mesh", type=str, default="1,1",
        help="transport_quality: transverse q-mesh as 'Nx,Ny' "
             "(default '1,1', appropriate for 1-D nanowires).",
    )
    p.add_argument(
        "--tq-temperature", type=float, default=300.0,
        help="transport_quality: device temperature in K (default 300).",
    )
    p.add_argument(
        "--tq-force-recompute", action="store_true",
        help="transport_quality: ignore any cached cells and recompute. "
             "Without this flag the analysis reuses "
             "<out-dir>/transport_quality/cache/{method}_r{rank}.npz so "
             "rerunning with new ranks only touches the missing cells.",
    )
    p.add_argument("--sigma-max-dist", type=int, default=None,
                   help="sigma_audit: largest |I-J| computed (default: n_blocks-1)")
    p.add_argument("--sse-sigma-distance", type=int, default=None,
                   help="sse_sparsity: largest |I-J| in the heatmap "
                        "(default: n_blocks-1, the full block matrix; "
                        "pass 1 to recover the tridiagonal-only behaviour)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    analyses = _parse_analyses(args.analyses)

    # Silence the qttools / cupy / phonopy chatter unless the user asked.
    if not args.verbose:
        warnings.simplefilter("ignore")

    bundle = load_system(
        args.config,
        name=args.name,
        transport_axis=args.transport_axis,
        n_slabs_hint=args.n_slabs_hint,
        fc3_path_override=args.fc3_path,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {
        "system": {
            "name": bundle.name,
            "n_super": bundle.n_super,
            "n_dof": bundle.n_dof,
            "block_sizes": bundle.block_sizes.tolist(),
            "transport_axis": bundle.transport_axis,
            **bundle.meta,
        }
    }

    # Per-system parameter-validation block (also emitted as warnings by the
    # loader, but this preserves the audit trail in summary.json for reports).
    import yaml as _yaml
    from phonon_inputs.config import config_from_dict as _config_from_dict

    from .parameter_validation import (
        validate_config as _validate_config,
        to_dict as _validation_to_dict,
    )

    summary["parameter_validation"] = _validation_to_dict(
        _validate_config(_config_from_dict(_yaml.safe_load(args.config.read_text())))
    )

    if "fc_quality" in analyses:
        from .fc_quality import run_fc_quality
        summary["fc_quality"] = run_fc_quality(bundle, out / "fc_quality")
    if "sparsity" in analyses:
        from .sparsity import run_sparsity
        summary["sparsity"] = run_sparsity(bundle, out / "sparsity")
    if "decomposition" in analyses:
        from .decomposition import run_decomposition
        summary["decomposition"] = run_decomposition(
            bundle, out / "decomposition",
            scalar_ranks=args.rank_sweep, skip_pcp=args.skip_pcp,
            verbose=args.verbose,
        )
    if "physical" in analyses:
        from .physical_tests import run_physical_tests
        summary["physical"] = run_physical_tests(bundle, out / "physical")
    if "sse_sparsity" in analyses:
        from .sse_sparsity_driver import run_sse_sparsity
        run_quatrex = bool(args.with_quatrex_crosscheck)
        if args.skip_quatrex_run:
            warnings.warn(
                "--skip-quatrex-run is now the default behaviour; flag "
                "ignored. Use --with-quatrex-crosscheck to opt in.",
                DeprecationWarning, stacklevel=2,
            )
        try:
            summary["sse_sparsity"] = run_sse_sparsity(
                bundle, out / "sse_sparsity",
                n_freq_pos=args.n_freq_pos, eta_thz=args.eta_thz,
                temperature_k=args.temperature,
                run_quatrex=run_quatrex,
                sigma_block_distance=args.sse_sigma_distance,
            )
        except ImportError as exc:
            # Soft fallback: rerun without the quatrex-side cross-check.
            warnings.warn(
                f"sse_sparsity --quatrex unavailable ({exc}); "
                f"falling back to synthetic-GF only.",
                stacklevel=2,
            )
            summary["sse_sparsity"] = run_sse_sparsity(
                bundle, out / "sse_sparsity",
                n_freq_pos=args.n_freq_pos, eta_thz=args.eta_thz,
                temperature_k=args.temperature,
                run_quatrex=False,
                sigma_block_distance=args.sse_sigma_distance,
            )
    if "cutoffs" in analyses:
        from .sse_sparsity_driver import run_cutoffs
        summary["cutoffs"] = run_cutoffs(
            bundle, out / "cutoffs",
            n_freq_pos=args.n_freq_pos, eta_thz=args.eta_thz,
            temperature_k=args.temperature,
            gamma_lead_thz=args.gamma_lead_thz,
            T_L=args.T_L, T_R=args.T_R,
        )
    if "sigma_audit" in analyses:
        from .sse_sparsity_driver import run_sigma_block_audit
        summary["sigma_audit"] = run_sigma_block_audit(
            bundle, out / "sigma_audit",
            n_freq_pos=args.n_freq_pos, eta_thz=args.eta_thz,
            temperature_k=args.temperature,
            max_block_distance=args.sigma_max_dist,
        )
    if "transport_quality" in analyses:
        from .transport_quality import run_transport_quality
        try:
            qx, qy = (int(s) for s in args.tq_q_mesh.split(","))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"--tq-q-mesh must be 'Nx,Ny' (got {args.tq_q_mesh!r}): {exc}"
            )
        summary["transport_quality"] = run_transport_quality(
            bundle, out / "transport_quality",
            scalar_ranks=args.rank_sweep,
            q_mesh_transverse=(qx, qy),
            temperature=args.tq_temperature,
            force_recompute=args.tq_force_recompute,
            verbose=args.verbose,
        )

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nfinite_analysis: wrote {out}/")
    for k in summary:
        if k == "system":
            continue
        print(f"  - {k}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
