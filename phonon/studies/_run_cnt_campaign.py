#!/usr/bin/env python
"""The clean CNT (3,3) campaign on daint (Alps).

Every arm is a 16-cell CNT (3,3) device that changes exactly ONE variable
against ``c16-kk``, the corrected bed.  The point of the campaign is that the
CNT corpus fails four gates at once and has never had a single-variable A/B on
any of them:

  extent    the grid stops at 55 THz; the band top is 46.335 THz, so the
            3-phonon bubble is supported to 92.67 THz and the grid covers
            59.4 % of it.  ``--aux-fmax 97`` extends the KK support without
            spending Dyson solves, at the primary spacing so the extent fix
            does not smuggle in a refinement.
  KK half   the only converged g_band=3 family runs ``retarded_method=half``,
            so Sigma^R has no real part at all.
  box mask  ``interaction_cutoff`` is the shipped 10.0 A default on every CNT
            run ever.  It is dense only at L4 (8.61 A span).  At L16 the fill
            is 46.1 %, which is the MoS2 rung that diverged (46.9 %); 40 A
            gives 100 % fill on a 38.12 A device.
  blocking  0 of 146 CNT directories run more than one cell per block.

Beds are built under ``$REPO/cluster/<arm>`` on Alps: the geometry is
symlinked from ``cluster/l16`` (or, for the reblocked arm, produced by
``reblock_device.py``), and each arm carries its OWN config so the run is
self-describing rather than a stack of QX_* overrides.  Arms need their own
directory rather than a shared bed because ``scba.py:1375`` writes per-
iteration dumps into ``output_dir``.

Usage
-----
    python phonon/studies/_run_cnt_campaign.py --stage A1            # dry run
    python phonon/studies/_run_cnt_campaign.py --stage A1 --prep --go
    python phonon/studies/_run_cnt_campaign.py --stage A1 --launch --go
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "/capstor/scratch/cscs/pfischil/quatrex"
GEOM = "cluster/l16"          # 16-cell CNT(3,3) geometry already on Alps
# The reblock source is the cutoff-corrected bed, so c16x2 INHERITS
# interaction_cutoff and the aux grid instead of needing a second patch:
# reblock_device.py copies the source config and rewrites only
# num_transport_cells and the /cluster/<name> path.
BASE = "cluster/c16-cut40"

# The login node has no `python` on PATH and the venv's interpreter is a
# symlink into the uenv image, so every remote python call has to go through
# `uenv run` -- the same route daint.py setup uses (daint.py:100-113).
UENV = "uenv run prgenv-gnu/26.3:v1 --view=default --"
PY = f"{UENV} /capstor/scratch/cscs/pfischil/quatrex-venv/bin/python"

FMAX = 55.0                   # primary window; band top is 46.335 THz
AUX_FMAX = 97.0               # >= 2*omega_max = 92.67, with 5 % margin
CUT_WIDE = 40.0               # 100 % mask fill on a 38.12 A device
NCELLS = 16
MAXIT = 400

# Geometry files symlinked into every arm's bed.  phonon_energies.npy is NOT
# among them: write_config.py rewrites it to match --nfreq.
GEOM_FILES = ("dynamical_matrix.mat", "fc3_blocks.hdf5", "structure.xyz")


def aux_dw(nfreq: int) -> float:
    """Auxiliary spacing == primary spacing, so the aux grid EXTENDS without
    refining.  grid_audit.md measures aux density as the unconverged and
    expensive axis (+4.4 to +10.2 %), so it is held fixed here."""
    return FMAX / (nfreq - 1)


# name, stage, nfreq, retarded, aux on, cutoff, gband, partition, walltime, note
ARMS = [
    ("c16-half",  "A1", 161, "half", False, None,     3, "debug",  "00:30:00",
     "reproduces l16f-g3 at 1 node: 97 it, lead_current 38.361"),
    ("c16-fft",   "A1", 161, "fft",  False, None,     3, "normal", "01:30:00",
     "KK on, but on a 55 THz grid -- 2.6 % truncated"),
    ("c16-kk",    "A1", 161, "fft",  True,  None,     3, "normal", "01:30:00",
     "the corrected bed; carries QX_SAVE_SIGMA for the linewidths"),
    ("c16-cut40", "A1", 161, "fft",  True,  CUT_WIDE, 3, "normal", "01:30:00",
     "box mask off -- tests the anomalous gain"),
    ("c16-ball",  "A1", 161, "fft",  True,  None,     3, "debug",  "00:20:00",
     "Landauer reference at the arm-2 grid"),
    ("c16-kk-lfm", "A1b", 161, "fft", True, None,     3, "normal", "01:30:00",
     "low_freq_mixing_thz=2.0/factor=0.02 -- does the IR marginal mode drive "
     "the fft limit cycle?  TOML-only, no QX_ override exists"),
    ("c16-g1",    "A2", 161, "fft",  True,  None,     1, "debug",  "00:30:00",
     "band ladder, boxcar b_G=1"),
    ("c16-g2",    "A2", 161, "fft",  True,  None,     2, "normal", "01:00:00",
     "band ladder, b_G=2"),
    ("c16-ne241", "A3", 241, "fft",  True,  None,     3, "normal", "02:00:00",
     "grid ladder with the extent defect removed"),
    ("c16-ne361", "A3", 361, "fft",  True,  None,     3, "normal", "02:00:00",
     "grid ladder, the rung that diverged at 55 THz"),
    ("c16x2",     "A4", 161, "fft",  True,  CUT_WIDE, 3, "normal", "02:00:00",
     "8 blocks x 2 cells -- gate (a) and (b) together, a first"),
]

# Stage M: the mixer sweep on the corrected bed.  40_selfconsistency.tex:233-236
# measured this on a TWO-cell tube and found plain damped linear best, with
# Anderson two orders above it; the corrected 16-cell bed needs more than 400
# linear iterations, so whether that ordering survives the length is open.
# "newton" is reachable only through QX_MIXMETHOD -- write_config.py's
# --mixing-method does not list it, though config.py:515 allows it.
MIXERS = ["anderson", "rre", "broyden", "jfnk", "rpm", "newton"]
MIXER_MAXIT = 600
for _m in MIXERS:
    ARMS.append((f"c16-mx-{_m}", "M", 161, "fft", True, None, 3,
                 "normal", "01:30:00", f"mixer sweep: {_m} on the corrected bed"))

FIELDS = ("name stage nfreq retarded aux cutoff gband partition time note").split()


def arms(stage: str | None):
    for row in ARMS:
        a = dict(zip(FIELDS, row))
        if stage in (None, "all", a["stage"]):
            yield a


def write_config_cmd(a) -> str:
    """The write_config.py invocation for one arm, with the REMOTE work dir."""
    n = a["nfreq"]
    flags = [
        f"{PY} {REPO}/phonon/studies/engine/write_config.py",
        "--system cnt33", f"--work {REPO}/cluster/{a['name']}",
        f"-L {NCELLS}", "--eta 0", f"--nfreq {n}", f"--fmax {FMAX}",
        f"--retarded {a['retarded']}", "--mix 0.1",]
    mx = a["name"].removeprefix("c16-mx-") if a["name"].startswith("c16-mx-") else None
    if mx and mx != "newton":
        flags.append(f"--mixing-method {mx}")
    flags += [
        f"--max-iter {MAXIT}", "--sigma-tol 1e-3",
    ]
    if a["aux"]:
        flags += [f"--aux-dw {aux_dw(n):.10g}", f"--aux-fmax {AUX_FMAX}"]
    return " ".join(flags)


def patch_cutoff_cmd(name: str, cutoff: float) -> str:
    """write_config.py has no --interaction-cutoff flag and never emits the
    key, so every CNT run in the corpus silently inherits config.py:1442's
    10.0 A.  Insert it directly under [phonon]; idempotent, so re-running prep
    does not stack duplicate keys."""
    toml = f"{REPO}/cluster/{name}/quatrex_config.toml"
    return (f"grep -q '^interaction_cutoff = ' {toml}"
            f" || sed -i '/^\\[phonon\\]$/a interaction_cutoff = {cutoff}' {toml};"
            f" grep -q '^interaction_cutoff = {cutoff}$' {toml}")


def prep_script(stage: str | None) -> str:
    """One shell script, run once over ssh.  Idempotent: re-running it
    rewrites the configs and leaves the geometry symlinks alone."""
    out = ["set -euo pipefail", "export UENV_WARN_MIGRATE=1", f"cd {REPO}",
           f'test -d {GEOM} || {{ echo "missing {GEOM}"; exit 1; }}']
    for a in arms(stage):
        n = a["name"]
        if n == "c16x2":
            continue                       # built by reblock, see reblock_cmd
        out += [
            f"", f"# --- {n}: {a['note']}",
            f"mkdir -p {REPO}/cluster/{n}",
        ]
        for f in GEOM_FILES:
            out.append(f"ln -sfn {REPO}/{GEOM}/{f} {REPO}/cluster/{n}/{f}")
        out.append(write_config_cmd(a))
        if a["cutoff"]:
            out.append(patch_cutoff_cmd(n, a["cutoff"]))
    return "\n".join(out) + "\n"


def reblock_cmd() -> str:
    """--tdir takes a LETTER: reblock_device.py:203 does "xyz".index(tdir),
    so the --tdir 0 in cluster/alps/do_reblock.sh would raise.  Needs the
    stage-A1 prep to have built BASE first."""
    cut = [a for a in ARMS if a[0] == "c16x2"][0][5]
    return (f"set -euo pipefail\nexport UENV_WARN_MIGRATE=1\ncd {REPO}\n"
            f'test -f {BASE}/quatrex_config.toml || '
            f'{{ echo "run --stage A1 --prep first"; exit 1; }}\n'
            f"{PY} phonon/studies/engine/reblock_device.py"
            f" --src {BASE} --cells {NCELLS} --per-block 2 --tdir z"
            f" --out cluster/c16x2\n"
            f"# the cutoff is inherited from {BASE}; verify rather than patch\n"
            + patch_cutoff_cmd("c16x2", cut) + "\n"
            f"grep -c '^interaction_cutoff = ' cluster/c16x2/quatrex_config.toml")


def _bcs(a, ranks: int = 4) -> int:
    """Largest admissible block_comm_size for this arm.

    Block-first is faster here, measured 2026-08-28 at a fixed four GPUs on
    the sixteen-block bed: bcs 1/2/4 give 4.083 / 3.635 / 3.439 s per
    iteration, ring 3.641 / 3.447 / 2.820, with identical residual
    trajectories.  That inverts gpu_campaign_2026-07.md Sec. 6 item 2
    ("stack-first rank layout, block axis only for memory") on this bed.

    The cap is the runtime's: sse_phonon_phonon.py:402-418 refuses unless every
    block rank owns at least ``g_band + 1`` blocks.  So sixteen blocks at band
    3 admit 4, and the REBLOCKED bed -- eight blocks of two cells -- admits
    only 2.  bcs must also divide the rank count.
    """
    n_blocks = NCELLS // 2 if a["name"] == "c16x2" else NCELLS
    cap = max(1, n_blocks // (a["gband"] + 1))
    return max(b for b in (1, 2, 4) if b <= min(cap, ranks) and ranks % b == 0)


def launch_cmd(a) -> list[str]:
    # QX_SAVE_SIGMA_BEST + _LIVE writes the min-residual Sigma as it improves,
    # so an arm killed at the walltime can be restarted with QX_SIGMA_INIT
    # instead of thrown away.  The A1 arms were launched without it.
    # Performance, measured on this bed 2026-08-28 (c16-obc against c16-half):
    #   QX_OBC_MEMO=cache   PhononSolver 0.639 -> 0.331 s (1.93x).  The OBC
    #     input is iteration-invariant here (eta=0, fixed leads, no scattering
    #     contacts), which is the mode's stated precondition (config.py:713-729);
    #     the residual trajectory is unchanged to 5 significant figures.
    #   QX_MAXBATCH=100000  what the archive l16 beds used; write_config's
    #     default is 512.  No measurable effect on the ring.
    #   End to end 4.393 -> 4.072 s/iter, i.e. 7.3 %.  The ring contraction is
    #     90 % of the iteration and neither knob touches it; gpu_campaign
    #     _2026-07.md Sec. 8/10 reports the fused-ring attempts below cuBLAS,
    #     so there is no further win there.  complex64 is ruled out for eta=0
    #     conservation-grade runs by that document's Sec. 5.
    env = ["QX_POLE_PSD=1", "QX_BBCHECK=1", f"QX_GBAND={a['gband']}",
           "QX_COMM_BACKEND=device_mpi", "QX_OBC_MEMO=cache",
           "QX_MAXBATCH=100000",
           f"QX_BCS={_bcs(a)}",
           f"QX_SAVE_SIGMA_BEST={REPO}/cluster/{a['name']}/sigma_best",
           "QX_SIGMA_BEST_LIVE=1"]
    if a["name"] == "c16-ball":
        env.append("QX_BALLISTIC=1")
    if a["name"].startswith("c16-mx-"):
        env += [f"QX_MIXMETHOD={a['name'].removeprefix('c16-mx-')}",
                f"QX_MAXIT={MIXER_MAXIT}"]
    if a["name"] == "c16-kk":
        # linewidths are NOT in run.npz: they need a Sigma snapshot plus a
        # Dyson re-solve (_resonance_gain_study.py).
        env.append(f"QX_SAVE_SIGMA={REPO}/cluster/c16-kk/sigma_final")
    cmd = [sys.executable, str(ROOT / "phonon/scripts/daint.py"), "launch",
           "--name", a["name"],
           "--config", f"cluster/{a['name']}/quatrex_config.toml",
           "--nodes", "1", "--ranks", "4",
           "--partition", a["partition"], "--time", a["time"]]
    for e in env:
        cmd += ["--env", e]
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="A1",
                   choices=["A1", "A1b", "A2", "A3", "A4", "M", "all"])
    p.add_argument("--prep", action="store_true", help="build the beds on Alps")
    p.add_argument("--reblock", action="store_true", help="build cluster/c16x2")
    p.add_argument("--launch", action="store_true", help="submit the arms")
    p.add_argument("--go", action="store_true",
                   help="actually do it (default is a dry run)")
    a = p.parse_args()

    sel = list(arms(a.stage))
    print(f"# stage {a.stage}: {len(sel)} arm(s)\n")
    for x in sel:
        aux = f"aux {aux_dw(x['nfreq']):.5g}->{AUX_FMAX:g}" if x["aux"] else "aux off"
        cut = f"cut {x['cutoff']:g}" if x["cutoff"] else "cut 10 (default)"
        print(f"  {x['name']:<10} ne={x['nfreq']:<4} {x['retarded']:<4} {aux:<20}"
              f" {cut:<18} b_G={x['gband']}  {x['partition']}/{x['time']}")
        print(f"  {'':<10} {x['note']}")

    if a.prep:
        script = prep_script(a.stage)
        print("\n# --- prep, over ssh daint ---\n" + script)
        if a.go:
            subprocess.run(["ssh", "daint", "bash -s"], input=script,
                           text=True, check=True)
            print("[prep] done")

    if a.reblock:
        print("\n# --- reblock ---\n" + reblock_cmd())
        if a.go:
            subprocess.run(["ssh", "daint", "bash -s"], input=reblock_cmd(),
                           text=True, check=True)
            print("[reblock] done")

    if a.launch:
        print("\n# --- launch ---")
        for x in sel:
            cmd = launch_cmd(x)
            print("  " + " ".join(shlex.quote(c) for c in cmd))
            if a.go:
                subprocess.run(cmd, check=True)

    if not a.go and (a.prep or a.reblock or a.launch):
        print("\n[dry run] nothing executed; add --go")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
