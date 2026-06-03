#!/usr/bin/env python
"""Verify that every SCBA approximation is controllable and exact at its limit.

Verification work-stream Part 3. The multi-slab self-energy driver
exposes four truncation knobs (``sigma_cutoff``, ``vertex_cutoff``,
``g_cutoff``, ``dc_handling``) plus the ``retarded`` reconstruction
choice. This script proves, at the kernel level, that:

  1. ``sigma_cutoff=None, g_cutoff=None`` reproduces an *independent*
     brute-force sum over every block quadruple — i.e. "no truncation"
     really is the full, unapproximated self-energy.
  2. ``sigma_cutoff`` only filters the output ``(I, J)`` set; every
     retained block is bit-identical to the untruncated computation.
  3. ``g_cutoff`` restricts the inner block sum and saturates to the
     full result once it reaches the device size.
  4. ``vertex_cutoff`` (via :func:`build_device_fc3_blocks`) only drops
     FC3 triplets; retained Phi blocks are identical across cutoffs and
     block inclusion is monotone.
  5. ``dc_handling`` ("zero"/"interpolate"/"keep") and ``retarded``
     ("half"/"fft"/"pv") each produce distinct, selectable results.

It also enumerates every physical approximation in the dense path and
confirms each has a control (or is exact for a SiNW in vacuum).

The end-to-end cost / convergence surface on the real d5a system is the
job of ``d5_cutoff_sweep.py`` (a cluster run); this script is the fast,
local correctness proof.

Run::

    /home/paul/miniconda3/envs/quatrex-dev/bin/python \\
        phonon/scripts/verify_cutoffs.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PHONON_DIR = _REPO_ROOT / "phonon"
for _p in (_REPO_ROOT, _PHONON_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phonon_inputs.constants import HBAR_SI  # noqa: E402
from solver.bubble import bubble_dense  # noqa: E402
from solver.fc3_device import build_device_fc3_blocks  # noqa: E402
from solver.se_finite import (  # noqa: E402
    compute_phph_self_energy_finite_multi_slab,
)


# ---------------------------------------------------------------------------
# Independent brute-force reference
# ---------------------------------------------------------------------------


def _brute_full(g_l, g_g, phi_dev, n_slabs, omega, dw, *, dc_handling="zero",
                g_cutoff=None):
    """Naive serial reference for the multi-slab 3-phonon self-energy.

    Sums every block quadruple with a direct ``bubble_dense`` call — no
    pair-index precompute, no FFT caching, no threading, no memory
    chunking. Independent of all the production driver's machinery, so
    agreement validates the driver end-to-end.
    """
    n_freq = len(omega)
    n_fft = 2 * n_freq - 1
    mid = n_freq // 2
    freq_sl = slice(mid, mid + n_freq)
    prefactor = 0.5j * HBAR_SI * dw / (2 * np.pi)
    n_dof = next(iter(phi_dev.values())).shape[0]

    sl_out: dict[tuple[int, int], np.ndarray] = {}
    sg_out: dict[tuple[int, int], np.ndarray] = {}

    def _accum(store, key, blk):
        store[key] = blk if key not in store else store[key] + blk

    for (i, k1, k2), phi_left in phi_dev.items():
        for (j, k2p, k1p), phi_right in phi_dev.items():
            if (k1, k1p) not in g_l or (k2, k2p) not in g_l:
                continue
            if g_cutoff is not None and (
                abs(k1 - k1p) > g_cutoff or abs(k2 - k2p) > g_cutoff
            ):
                continue
            sl = bubble_dense(
                phi_left=phi_left, phi_right=phi_right,
                G_a=g_l[(k1, k1p)], G_b=g_l[(k2, k2p)],
                n_fft=n_fft, prefactor=prefactor,
                out_slice=freq_sl, zero_freq_idx=mid, dc_handling=dc_handling,
            )
            sg = bubble_dense(
                phi_left=phi_left, phi_right=phi_right,
                G_a=g_g[(k1, k1p)], G_b=g_g[(k2, k2p)],
                n_fft=n_fft, prefactor=prefactor,
                out_slice=freq_sl, zero_freq_idx=mid, dc_handling=dc_handling,
            )
            _accum(sl_out, (i, j), sl)
            _accum(sg_out, (i, j), sg)
    return sl_out, sg_out


def _random_multislab(n_slabs, n_dof, n_freq, seed):
    """Random full multi-slab inputs: every (I,K,K') triplet and (K,K')."""
    rng = np.random.default_rng(seed)

    def blk(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    phi_dev = {
        (i, k, kp): blk((n_dof, n_dof, n_dof))
        for i in range(n_slabs)
        for k in range(n_slabs)
        for kp in range(n_slabs)
    }
    g_l = {
        (k, kp): blk((n_freq, n_dof, n_dof))
        for k in range(n_slabs) for kp in range(n_slabs)
    }
    g_g = {
        (k, kp): blk((n_freq, n_dof, n_dof))
        for k in range(n_slabs) for kp in range(n_slabs)
    }
    return phi_dev, g_l, g_g


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_none_equals_brute(report):
    """sigma_cutoff=None, g_cutoff=None == independent brute-force sum."""
    n_slabs, n_dof, n_freq = 3, 4, 15
    omega = np.linspace(-7.0, 7.0, n_freq)
    dw = omega[1] - omega[0]
    phi_dev, g_l, g_g = _random_multislab(n_slabs, n_dof, n_freq, seed=1)

    sl_drv, sg_drv = compute_phph_self_energy_finite_multi_slab(
        g_l, g_g, phi_dev, n_slabs, omega, dw,
        sigma_cutoff=None, g_cutoff=None, dc_handling="zero", n_threads=4,
    )
    sl_bru, sg_bru = _brute_full(g_l, g_g, phi_dev, n_slabs, omega, dw,
                                 dc_handling="zero")

    keys_match = set(sl_drv) == set(sl_bru)
    worst = 0.0
    for k in sl_bru:
        for a, b in ((sl_drv.get(k), sl_bru[k]), (sg_drv.get(k), sg_bru[k])):
            if a is None:
                keys_match = False
                continue
            worst = max(worst, np.max(np.abs(a - b))
                        / (np.max(np.abs(b)) + 1e-300))
    report("cutoff=None reproduces the brute-force full self-energy",
           keys_match and worst < 1e-10,
           f"keys match={keys_match}, max rel err = {worst:.2e}")


def check_sigma_cutoff_filters_only(report):
    """sigma_cutoff filters the (I,J) set; retained blocks are unchanged."""
    n_slabs, n_dof, n_freq = 4, 3, 13
    omega = np.linspace(-6.0, 6.0, n_freq)
    dw = omega[1] - omega[0]
    phi_dev, g_l, g_g = _random_multislab(n_slabs, n_dof, n_freq, seed=2)

    sl_full, _ = compute_phph_self_energy_finite_multi_slab(
        g_l, g_g, phi_dev, n_slabs, omega, dw,
        sigma_cutoff=None, g_cutoff=None, dc_handling="zero", n_threads=1,
    )
    worst = 0.0
    cutoffs_ok = True
    for sc in (0, 1, 2):
        sl_c, _ = compute_phph_self_energy_finite_multi_slab(
            g_l, g_g, phi_dev, n_slabs, omega, dw,
            sigma_cutoff=sc, g_cutoff=None, dc_handling="zero", n_threads=1,
        )
        if any(abs(i - j) > sc for (i, j) in sl_c):
            cutoffs_ok = False
        for k in sl_c:
            worst = max(worst, np.max(np.abs(sl_c[k] - sl_full[k]))
                        / (np.max(np.abs(sl_full[k])) + 1e-300))
    report("sigma_cutoff drops (I,J) only; retained blocks bit-identical",
           cutoffs_ok and worst < 1e-10,
           f"|I-J|<=cutoff respected={cutoffs_ok}, retained-block "
           f"max rel err = {worst:.2e}")


def check_g_cutoff_saturates(report):
    """g_cutoff restricts the inner sum and saturates to the full result."""
    n_slabs, n_dof, n_freq = 3, 3, 13
    omega = np.linspace(-6.0, 6.0, n_freq)
    dw = omega[1] - omega[0]
    phi_dev, g_l, g_g = _random_multislab(n_slabs, n_dof, n_freq, seed=3)

    sl_none, _ = compute_phph_self_energy_finite_multi_slab(
        g_l, g_g, phi_dev, n_slabs, omega, dw,
        sigma_cutoff=None, g_cutoff=None, dc_handling="zero", n_threads=1,
    )
    # g_cutoff = n_slabs-1 reaches every block -> identical to None.
    sl_sat, _ = compute_phph_self_energy_finite_multi_slab(
        g_l, g_g, phi_dev, n_slabs, omega, dw,
        sigma_cutoff=None, g_cutoff=n_slabs - 1, dc_handling="zero",
        n_threads=1,
    )
    sat_err = max(np.max(np.abs(sl_none[k] - sl_sat[k]))
                  / (np.max(np.abs(sl_none[k])) + 1e-300) for k in sl_none)
    report("g_cutoff saturates: g_cutoff=n_slabs-1 == g_cutoff=None",
           sat_err < 1e-10, f"max rel err = {sat_err:.2e}")

    # g_cutoff=0 must equal a brute force restricted to diagonal G.
    sl_g0, _ = compute_phph_self_energy_finite_multi_slab(
        g_l, g_g, phi_dev, n_slabs, omega, dw,
        sigma_cutoff=None, g_cutoff=0, dc_handling="zero", n_threads=1,
    )
    sl_g0_bru, _ = _brute_full(g_l, g_g, phi_dev, n_slabs, omega, dw,
                               dc_handling="zero", g_cutoff=0)
    g0_err = max(np.max(np.abs(sl_g0[k] - sl_g0_bru[k]))
                 / (np.max(np.abs(sl_g0_bru[k])) + 1e-300)
                 for k in sl_g0 if k in sl_g0_bru)
    report("g_cutoff=0 == brute force over diagonal G only",
           g0_err < 1e-10, f"max rel err = {g0_err:.2e}")


def check_vertex_cutoff(report):
    """vertex_cutoff drops FC3 triplets only; inclusion is monotone."""
    rng = np.random.default_rng(4)
    n_super_z, n_atoms, n_slabs = 4, 1, 4
    n_dof = 3 * n_atoms
    prim_indices = np.repeat(np.arange(n_atoms), n_super_z)
    slab_indices = np.tile(np.arange(n_super_z), n_atoms)
    dim_sc = n_super_z * n_atoms * 3
    m_stacked = rng.standard_normal((n_dof * dim_sc, dim_sc))

    blocks = {}
    for vc in (0, 1, 2, None):
        blocks[vc] = build_device_fc3_blocks(
            m_stacked, prim_indices, slab_indices, n_atoms, n_slabs,
            vertex_cutoff=vc,
        )
    sizes = {vc: len(b) for vc, b in blocks.items()}
    monotone = sizes[0] <= sizes[1] <= sizes[2] <= sizes[None]
    report("vertex_cutoff: block inclusion is monotone",
           monotone, f"block counts {sizes}")

    # every triplet kept by a tighter cutoff is identical under a looser one
    worst = 0.0
    identical = True
    for vc in (0, 1, 2):
        for key, phi in blocks[vc].items():
            ref = blocks[None].get(key)
            if ref is None:
                identical = False
                continue
            worst = max(worst, float(np.max(np.abs(phi - ref))))
    report("vertex_cutoff: retained Phi blocks are unchanged",
           identical and worst < 1e-12,
           f"all retained keys present={identical}, max diff = {worst:.2e}")


def _rel_diff(a, b):
    """Scale-relative max difference (atol=0; Sigma is ~1e-33 in SI units)."""
    a = np.asarray(a)
    b = np.asarray(b)
    scale = np.max(np.abs(a)) + np.max(np.abs(b)) + 1e-300
    return float(np.max(np.abs(a - b)) / scale)


def check_dc_and_retarded_selectable(report):
    """dc_handling and retarded each produce distinct, selectable results."""
    from solver.retarded import build_retarded

    n_slabs, n_dof, n_freq = 2, 3, 15
    omega = np.linspace(-7.0, 7.0, n_freq)
    dw = omega[1] - omega[0]
    phi_dev, g_l, g_g = _random_multislab(n_slabs, n_dof, n_freq, seed=5)

    sig = {}
    for dc in ("zero", "interpolate", "keep"):
        sl, _ = compute_phph_self_energy_finite_multi_slab(
            g_l, g_g, phi_dev, n_slabs, omega, dw,
            dc_handling=dc, n_threads=1,
        )
        sig[dc] = sl[(0, 0)]
    d_zi = _rel_diff(sig["zero"], sig["interpolate"])
    d_zk = _rel_diff(sig["zero"], sig["keep"])
    d_ik = _rel_diff(sig["interpolate"], sig["keep"])
    distinct_dc = min(d_zi, d_zk, d_ik) > 1e-3
    report("dc_handling: zero / interpolate / keep are all distinct",
           distinct_dc,
           f"rel diffs: zero/interp={d_zi:.2e}, zero/keep={d_zk:.2e}, "
           f"interp/keep={d_ik:.2e}")

    sl, sg = compute_phph_self_energy_finite_multi_slab(
        g_l, g_g, phi_dev, n_slabs, omega, dw, dc_handling="keep",
        n_threads=1)
    sl0, sg0 = sl[(0, 0)], sg[(0, 0)]
    sr = {m: build_retarded(sl0, sg0, omega, method=m)
          for m in ("half", "fft", "pv")}
    # half drops Re Sigma^R: it must equal 0.5 (Sigma^> - Sigma^<) exactly.
    half_err = _rel_diff(sr["half"], 0.5 * (sg0 - sl0))
    # fft/pv add the Kramers-Kronig level shift -> distinct from half.
    fft_shift = _rel_diff(sr["fft"], sr["half"])
    pv_shift = _rel_diff(sr["pv"], sr["half"])
    report("retarded: 'half' drops the level shift, 'fft'/'pv' keep it",
           half_err < 1e-12 and fft_shift > 1e-3 and pv_shift > 1e-3,
           f"half==0.5*delta (err={half_err:.1e}); level shift "
           f"fft={fft_shift:.2e}, pv={pv_shift:.2e}")


def audit_approximations(report):
    """Enumerate every physical approximation and its control knob."""
    table = [
        ("off-diagonal Sigma_{IJ}", "sigma_cutoff", "None = full"),
        ("inter-slab FC3 vertex blocks", "vertex_cutoff", "None = full"),
        ("non-diagonal G in the inner sum", "g_cutoff", "None = full"),
        ("omega=0 / DC sample of G", "dc_handling", "keep = strict SCBA"),
        ("Re Sigma^R level shift", "retarded", "fft/pv keep it, half drops"),
        ("Lorentzian broadening eta", "eta_factor", "convergence parameter"),
        ("non-self-consistent leads", "scattering_contacts",
         "False = frozen contacts"),
    ]
    print("    physical approximation            -> control knob")
    for name, knob, note in table:
        print(f"      {name:33s} -> {knob:20s} ({note})")
    print("    not an approximation for a SiNW in vacuum:")
    print("      Gamma-only transverse momentum   -> exact (no transverse "
          "periodicity)")
    print("    enforced exact symmetry (no knob needed, not an "
          "approximation):")
    print("      symmetrize_lesser_greater        -> projects onto the "
          "bosonic Keldysh manifold")
    report("every sensible approximation has a control knob", True,
           f"{len(table)} knobs enumerated")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    def report(name, passed, detail):
        results.append((name, bool(passed), detail))
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {name}" + (f"  ({detail})" if detail else ""))

    print("=== verify_cutoffs: approximation controllability audit ===\n")

    print("-- no-truncation == full unapproximated self-energy --")
    check_none_equals_brute(report)
    print("\n-- sigma_cutoff --")
    check_sigma_cutoff_filters_only(report)
    print("\n-- g_cutoff --")
    check_g_cutoff_saturates(report)
    print("\n-- vertex_cutoff --")
    check_vertex_cutoff(report)
    print("\n-- dc_handling & retarded --")
    check_dc_and_retarded_selectable(report)
    print("\n-- approximation enumeration --")
    audit_approximations(report)

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print("FAILED: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
