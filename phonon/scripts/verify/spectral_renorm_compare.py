#!/usr/bin/env python
"""Renormalised vs bare phonon spectral function for CNT(3,3) and d5a SiNW.

For each structure, build the zone-centre device spectral function
    A(Gamma, omega) = -1/pi Im Tr G^R(omega),
    G^R = [(omega+i eta)^2 I - D - Sigma_static - Sigma_B(omega)]^{-1}
and overlay the toggles the user asked for:
  (a) with / without Kramers-Kronig : bubble retarded built with retarded="fft"
      (full causal: broadening + KK real shift) vs "half" (broadening only).
  (a) with / without tadpole        : static Sigma_T added (self-consistent run).
  (b) vs the bare D diagonalisation : "bare" curve = no self-energy, plus the
      bare eigenfrequencies sqrt(eig D) as vertical lines.
The bubble Sigma_B is a device (Gamma-folded) object, so A is plotted vs omega.
A second panel shows the q_z-resolved bands bare vs +Sigma_static (q-independent
static SE added to D(q_z) and re-diagonalised), the band analogue of (b).

Each Sigma is taken from a CONVERGED transmission_finite run (self_energy_retarded
/ sigma_static), so no fragile one-shot static solve (F34).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import warnings; warnings.filterwarnings("ignore")
from phonon.finite_analysis.loader import load_system
from solver.dense import transmission_finite
from postproc.spectral import (
    dynamical_matrix_qpath, dynamical_matrix_at_q,
    spectral_function_qw, frequencies_from_dynamical,
)

STRUCTS = [
    ("CNT(3,3)", "phonon/configs/cnt/cnt33_vasp.yaml", 18.0),
    ("SiNW d5a", "phonon/configs/sinw/sinw100_d5a_vasp_sc4_fc4.yaml", 16.0),
]
OUT = _REPO / "document/fig/transport_sweeps"


def converged_se(ph, fc3, fmax, *, retarded, tadpole, **extra):
    """Run one SCBA; return (freqs, Sigma_B(omega), Sigma_static, D_bare)."""
    common = dict(
        fc3_hdf5=fc3, transport_direction="z", n_slabs=1, retarded=retarded,
        freq_range_thz=(0.05, fmax, 81), eta_factor=0.5, temperature=300.0,
        delta_T=10.0, max_scba_iter=60, scba_tol=1e-3, conservation_tol=5e-2,
        enforce_asr=True, verbose=False,
    )
    r = transmission_finite(ph, tadpole=tadpole, **{**common, **extra})
    freqs = np.asarray(r["freqs_thz"])
    sb = np.asarray(r["self_energy_retarded"])          # (nw, N, N)
    ss = r.get("sigma_static")
    ss = None if ss is None else np.asarray(ss)
    phi = r.get("phi_eff")
    D = (np.asarray(phi) - ss) if (phi is not None and ss is not None) else None
    conv = r.get("scba_converged"); res = r.get("scba_residual")
    print(f"    [{retarded}, tadpole={tadpole}] conv={conv} resid={res:.2e} "
          f"||Sig_static||={0.0 if ss is None else np.linalg.norm(ss):.3f}")
    return freqs, sb, ss, D


def main():
    fig, axes = plt.subplots(2, len(STRUCTS), figsize=(12, 9))
    for col, (name, cfg, fmax) in enumerate(STRUCTS):
        print(f"== {name} ==")
        bundle = load_system(str(_REPO / cfg), validate=False, transport_axis=2)
        ph = bundle.phonon
        fc3 = str(Path(bundle.meta["fc3_path"]).expanduser().resolve())

        # three converged runs: bubble half, bubble fft, bubble+tadpole fft
        f_h, sb_h, _, D_h = converged_se(ph, fc3, fmax, retarded="half", tadpole=False)
        f_f, sb_f, _, D_f = converged_se(ph, fc3, fmax, retarded="fft", tadpole=False)
        try:
            f_t, sb_t, ss_t, D_t = converged_se(
                ph, fc3, fmax, retarded="fft", tadpole=True,
                stage_loop_first=True, static_mixing=0.1)
        except Exception as exc:
            print(f"    tadpole run failed ({type(exc).__name__}); "
                  f"skipping +tadpole curve")
            f_t = sb_t = ss_t = D_t = None

        # bare D: first FINITE candidate (the d5a tadpole run returns NaN
        # sigma_static -> its phi_eff-derived D is NaN; fall back to D(Gamma)).
        D = None
        for cand in (D_t, D_h, D_f):
            if cand is not None and np.all(np.isfinite(cand)):
                D = np.asarray(cand).real; break
        if D is None:
            D = dynamical_matrix_at_q(ph, [0, 0, 0]).real
        # drop any NaN tadpole pieces so the +tadpole curve is simply omitted
        if ss_t is not None and not np.all(np.isfinite(ss_t)):
            f_t = sb_t = ss_t = None
        N = D.shape[0]
        eta_w = 1.5 * (f_h[1] - f_h[0])

        def A(grid, sb=None, ss=None):
            return spectral_function_qw(
                D[None], grid, eta_w,
                sigma_static=(None if ss is None else ss.real[None]),
                sigma_b=(None if sb is None else sb[None]))[0]

        ax = axes[0, col]
        ax.semilogy(f_h, A(f_h) + 1e-6, "k-", lw=1.0, label="bare D")
        ax.semilogy(f_h, A(f_h, sb_h) + 1e-6, color="tab:blue", lw=1.0,
                    label="+bubble (half, no KK)")
        ax.semilogy(f_f, A(f_f, sb_f) + 1e-6, color="tab:red", lw=1.0,
                    label="+bubble (fft, KK)")
        if f_t is not None:
            ax.semilogy(f_t, A(f_t, sb_t, ss_t) + 1e-6, color="tab:green",
                        lw=1.0, label="+bubble+tadpole (fft)")
        for w0 in np.abs(frequencies_from_dynamical(D)):
            ax.axvline(w0, color="0.7", lw=0.3, zorder=0)
        ax.set_title(f"{name}: A(Gamma, omega)")
        ax.set_xlabel(r"$\omega$ [THz]"); ax.set_ylabel(r"$A(\Gamma,\omega)$")
        ax.set_xlim(0, fmax); ax.legend(fontsize=7, loc="upper right")

        # q_z-resolved bands: bare vs +Sigma_static
        ax2 = axes[1, col]
        nq = 81
        qz = np.linspace(0, 0.5, nq)
        Dq = dynamical_matrix_qpath(
            ph, np.column_stack([np.zeros(nq), np.zeros(nq), qz])).real
        qd = qz * 2.0
        bare_bands = frequencies_from_dynamical(Dq)
        for n in range(N):
            ax2.plot(qd, np.abs(bare_bands[:, n]), color="0.5", lw=0.5,
                     label="bare D" if n == 0 else None)
        if ss_t is not None:
            ren = frequencies_from_dynamical(Dq + ss_t.real[None])
            for n in range(N):
                ax2.plot(qd, np.abs(ren[:, n]), color="tab:green", lw=0.5,
                         ls="--", label="+tadpole (Sigma_static)" if n == 0 else None)
        ax2.set_title(f"{name}: bands bare vs +static SE")
        ax2.set_xlabel(r"$q_z\ [\pi/a]$"); ax2.set_ylabel(r"$\omega$ [THz]")
        ax2.set_ylim(0, fmax); ax2.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"spectral_renorm_compare.{ext}", dpi=140,
                    bbox_inches="tight")
    print(f"wrote {OUT / 'spectral_renorm_compare.pdf'}")


if __name__ == "__main__":
    main()
