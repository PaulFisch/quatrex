"""Generate a SiGe alloy nanowire by random Si→Ge substitution in a SiNW.

Loads an existing SiNW(100) config and replaces ``--frac-ge`` of the Si
core atoms with Ge (seeded for reproducibility). The H passivation is
unchanged. The lattice constant stays at the Si value — this is intended
as a mass-mismatch test, not a fully relaxed alloy.

Usage:
    python examples/setup_sige_nw.py --base configs/sinw/sinw100_vasp.yaml --frac-ge 0.5
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import add_common_args, write_config  # noqa: E402


def make_config(base_path: Path, frac_ge: float, seed: int = 42) -> dict:
    cfg = yaml.safe_load(Path(base_path).read_text())
    syms = list(cfg["structure"]["symbols"])
    si_idx = [i for i, s in enumerate(syms) if s == "Si"]
    rng = np.random.default_rng(seed)
    n_swap = int(round(frac_ge * len(si_idx)))
    swap = rng.choice(si_idx, size=n_swap, replace=False)
    for i in swap:
        syms[i] = "Ge"

    cfg = copy.deepcopy(cfg)
    cfg["structure"]["symbols"] = syms
    cfg["vasp"]["potcar_map"]["Ge"] = "Ge"
    tag = f"sigenw100_fge{int(frac_ge * 100):02d}"
    cfg["relax"]["work_dir"] = f"./relax_{tag}_vasp"
    if "hiphive" in cfg:
        cfg["hiphive"]["work_dir"] = f"./fc3_hiphive_{tag}_vasp"
    if "thirdorder" in cfg:
        cfg["thirdorder"]["work_dir"] = f"./fc3_{tag}_vasp"
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    here = Path(__file__).resolve()
    p.add_argument(
        "--base", type=Path,
        default=here.parent.parent / "configs" / "sinw" / "sinw100_vasp.yaml",
        help="Base SiNW config to alloy",
    )
    p.add_argument("--frac-ge", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    add_common_args(
        p, default_out=here.parent.parent / "configs" / "sigenw"
        / "sigenw100_fgeXX_vasp.yaml",
    )
    args = p.parse_args()

    cfg = make_config(args.base, args.frac_ge, args.seed)
    out = (
        args.out if args.out.name != "sigenw100_fgeXX_vasp.yaml"
        else args.out.with_name(
            f"sigenw100_fge{int(args.frac_ge * 100):02d}_vasp.yaml"
        )
    )
    write_config(cfg, out)


if __name__ == "__main__":
    main()
