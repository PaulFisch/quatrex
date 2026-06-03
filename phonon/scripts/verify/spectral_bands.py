#!/usr/bin/env python
"""Phonon spectral function A(q, omega) + decomposed band renormalisation.

Ties Phase 2 (loop/tadpole static self-energy from the SCBA) to Phase 3
(postproc.spectral): run the device SCBA with the static self-energy on, read
``Sigma_static`` from the result, build D(q) along a q-path, and emit the
decomposed bands (bare -> +loop -> +tadpole) + the A(q, omega) heat-map.

Real use (bulk Si / a device):
    phonon = <load your Phonopy object with force constants>
    res = transmission_finite(phonon, fc3_hdf5, loop=True, tadpole=True,
                              fc4_hdf5=fc4, temperature=300.0, ...)
    sigma_static = res["sigma_static"]                 # (n_dof, n_dof) at Gamma
    D_q = dynamical_matrix_qpath(phonon, q_path)        # (nq, n_dof, n_dof)
    bundle = band_renormalization_bundle(D_q, omega_grid, eta_w,
                                         sigma_loop=sigma_static)
    save_spectral("spectral.npz", **bundle, tick_positions=..., tick_labels=...)

Self-contained demo (no phonopy / DFT needed) -- exercises the whole chain on a
toy diatomic chain and writes spectral.npz + plot_spectral.py (+ a PDF if
matplotlib is present):
    python phonon/scripts/verify/spectral_bands.py --demo --out /tmp/spec
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

from postproc.io import save_spectral, write_reference_plot_script
from postproc.spectral import band_renormalization_bundle


def _demo(out_dir: Path):
    """Full pipeline on a toy diatomic chain with a quartic loop."""
    from solver.grids import build_frequency_grid
    from solver.leads import build_device_hamiltonian, compute_obc_batch
    from solver.dense import scba_loop
    from solver.static_se import build_static_self_energy_hook
    from solver.toy_models import diatomic_chain
    from phonon_inputs.constants import THZ_TO_RAD

    toy = diatomic_chain()
    n_dof = toy.n_dof
    freqs, dw, eta_w, z2, pos_mask, mid = build_frequency_grid(
        (0.01, 25.0, 60), eta_factor=1.0)
    nfreq = len(freqs)
    h00 = toy.h00.astype(complex)
    h01 = toy.h01.astype(complex)
    h_d = build_device_hamiltonian(h00, h01, 1)
    obc = compute_obc_batch(z2, h00, h01, freqs, 310.0, 290.0, n_slabs=1)

    # quartic loop self-energy (on-site, positive -> stiffening)
    fc4 = np.zeros((n_dof,) * 4)
    idx = np.arange(n_dof)
    fc4[idx, idx, idx, idx] = 40.0
    hook = build_static_self_energy_hook(
        dw_thz=dw, n_dof=n_dof, n_slabs=1, fc4_dev_mw=fc4, use_loop=True)

    def zero_bubble(gl, gg):
        z = np.zeros((1, nfreq, n_dof, n_dof), dtype=complex)
        return z, z.copy()

    res = scba_loop(
        z2_arr=z2, freqs_thz=freqs, dw_thz=dw, omega_rad=freqs * THZ_TO_RAD,
        pos_mask=pos_mask, n_slabs=1, n_dof=n_dof, N_D=n_dof,
        H_D_list=[h_d], obc_list=[obc], btd_blocks_list=[(h00, h01)],
        n_kpts=1, se_kernel=zero_bubble, T_L=310.0, T_R=290.0,
        max_scba_iter=60,
        scba_tol=1e-3, conservation_tol=1e-2, mixing=0.5, anderson_mixing=False,
        anderson_depth=5, scattering_contacts=False, retarded="fft",
        verbose=False, masses_primitive=toy.masses,
        static_se_hook=hook, stage_loop_first=True)
    sigma_static = res["Sigma_static"][0].real
    print(f"  loop Sigma_static max = {np.max(np.abs(sigma_static)):.3e} THz^2")

    # band path of the periodic chain: q = 0 .. pi (Gamma -> zone boundary)
    qs = np.linspace(0.0, np.pi, 101)
    D_q = np.stack([h00 + h01 * np.exp(1j * q) + h01.conj().T * np.exp(-1j * q)
                    for q in qs])
    grid = freqs[pos_mask]
    bundle = band_renormalization_bundle(
        D_q, grid, eta_w_thz=3 * dw, q_distance=qs / np.pi,
        sigma_loop=sigma_static)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz = save_spectral(out_dir / "spectral.npz", **bundle,
                        tick_positions=[0.0, 1.0], tick_labels=["G", "X"])
    plot = write_reference_plot_script(out_dir / "plot_spectral.py")
    print(f"  wrote {npz}\n  wrote {plot}")
    try:
        import subprocess
        subprocess.run([sys.executable, plot, npz,
                        str(out_dir / "spectral.pdf")], check=True)
    except Exception as exc:                       # matplotlib may be absent
        print(f"  (skipped PDF render: {exc})")
    db = bundle["bands"]
    print(f"  bare max {db['bare'].max():.3f} THz -> loop max "
          f"{db['loop'].max():.3f} THz (stiffened)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true",
                    help="run the self-contained toy demo")
    ap.add_argument("--out", default="/tmp/claude/spectral",
                    help="output directory")
    args = ap.parse_args()
    if args.demo:
        _demo(Path(args.out))
    else:
        ap.error("only --demo is self-contained; see the module docstring for "
                 "the real (phonopy) workflow.")


if __name__ == "__main__":
    main()
