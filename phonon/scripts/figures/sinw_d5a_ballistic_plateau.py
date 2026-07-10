"""Taper justification on the BALLISTIC SiNW d5a: the omega^2 IR occupancy
taper crushes the low-frequency transmission, while the IR-occupancy
subtraction (full physical Bose occupation) preserves the acoustic plateau.

Reads ONLY saved data (phonon/studies/out/conv1e10/, sinw d5a L2, eta=1e-12,
retarded=fft, 305/295 K, both ballistic i.e. vertex zeroed, both converged):

  sinw_d5a_L2_eta0_diag_ball.npz    lead occupancies multiplied by the
      omega^2/(omega^2 + omega_reg^2) IR taper with ir_taper_cells = 5.891
      -> omega_reg = 5.891*dw = 2.160 THz  (work/sinw_d5a_L2_eta0_diag/
      quatrex_config.toml);
  sinw_d5a_L2_irsub2_smoke_ball.npz  sse_ir_subtraction = true: FULL physical
      Bose occupation, no omega^2 taper (log: "IR occupation subtraction ON").

No bare-Bose-without-either third variant exists in conv1e10 -- but the
IR-subtracted run IS the full physical occupancy (the subtraction only acts
inside the SSE bubble, which is zeroed here), so the two runs are exactly
"physical occupancy" vs "omega^2-tapered occupancy".

Effective transmission T(omega) = I(omega)/Delta n(omega) with the PHYSICAL
Bose difference Delta n = n(omega,T_L) - n(omega,T_R) (same convention as
_eta0_diag_plots.py / transmission_physicality.py; lead-averaged |I|).
The d5a wire has N_ac = 4 acoustic channels (2 flexural + LA + twist) --
the expected omega->0 plateau, marked as the dotted line (review fix: the
old "(plateau)" y-label was vague; this is a transmission in channels).
Right panel: the ratio T_taper/T_full against the analytic taper factor --
the crushing is exactly the applied omega^2/(omega^2+omega_reg^2).

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/sinw_d5a_ballistic_plateau.py
Figure -> document/fig/transport_sweeps/sinw_d5a_ballistic_plateau.{pdf,png}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

CONV = ROOT / "phonon/studies/out/conv1e10"
NPZ_TAPER = CONV / "sinw_d5a_L2_eta0_diag_ball.npz"     # omega^2 IR taper
NPZ_FULL = CONV / "sinw_d5a_L2_irsub2_smoke_ball.npz"   # full Bose (IR-sub)
FIGDIR = ROOT / "document/fig/transport_sweeps"
HBAR = 1.054571817e-34
KB = 1.380649e-23
N_AC = 4                       # d5a acoustic channels: 2 flexural + LA + twist
TAPER_CELLS = 5.890909090909091   # ir_taper_cells of the diag run (its toml)
WMAX = 6.0                     # low-omega window (THz)


def _bose(w_thz, T):
    w = np.asarray(w_thz, float)
    x = np.where(w > 1e-9, w * 1e12 * 2 * np.pi * HBAR / (KB * T), 1.0)
    return np.where(w > 1e-9, 1.0 / np.expm1(x), 0.0)


def _load(path):
    d = np.load(path, allow_pickle=True)
    en = np.asarray(d["energies"], float)
    dn = _bose(en, float(d["t_left"])) - _bose(en, float(d["t_right"]))
    cs = np.asarray(d["current_spectrum"])
    cur = 0.5 * (np.abs(cs[:, 0]) + np.abs(cs[:, -1]))   # lead-averaged |I|
    return en, np.where(dn > 1e-9, cur / dn, np.nan)


def _g0_check(en, T_of_w, temperature):
    """Quantitative eq:g0 check: the cumulative linear-response conductance
    of the ballistic transmission approaches N_ac * g0 as T -> 0, with
    g0 = pi^2 kB^2 T / (3 h) the thermal conductance quantum."""
    import sys as _sys
    from pathlib import Path as _Path
    _ph = _Path(__file__).resolve().parents[2]
    if str(_ph) not in _sys.path:
        _sys.path.insert(0, str(_ph))
    from solver.observables import thermal_conductance

    ok = np.isfinite(T_of_w)
    G = thermal_conductance(np.where(ok, T_of_w, 0.0), en, temperature)
    # Discrete ideal reference on the SAME grid (T(omega) = N_ac): cancels
    # the coarse-grid quadrature error of the low-T thermal window, so the
    # ratio isolates the plateau fidelity. The continuum quantum g0 is
    # reported for scale.
    G_ref = thermal_conductance(np.full_like(en, N_AC), en, temperature)
    g0 = np.pi**2 * KB**2 * temperature / (3.0 * 2.0 * np.pi * HBAR)
    print(f"g0 check (T={temperature:g} K): G = {G:.4e} W/K; discrete "
          f"N_ac reference = {G_ref:.4e} W/K -> ratio {G / G_ref:.4f}; "
          f"continuum N_ac*g0 = {N_AC * g0:.4e} W/K")
    return G / G_ref


def main():
    en, T_full = _load(NPZ_FULL)
    en_t, T_tap = _load(NPZ_TAPER)
    assert np.allclose(en, en_t)
    # eq:g0: at low T only the acoustic plateau contributes, so the
    # conductance of the full-Bose curve approaches N_ac quanta.
    _g0_check(en, T_full, 10.0)
    dw = en[1] - en[0]
    w_reg = TAPER_CELLS * dw
    m = (en > 0) & (en <= WMAX + 0.5 * dw)

    print(f"grid dw={dw:.4f} THz; taper omega_reg={w_reg:.3f} THz")
    print(f"T_full first bins  ({en[1]:.2f},{en[2]:.2f} THz):"
          f" {T_full[1]:.3f}, {T_full[2]:.3f}  (N_ac={N_AC})")
    print(f"T_taper first bins ({en[1]:.2f},{en[2]:.2f} THz):"
          f" {T_tap[1]:.3f}, {T_tap[2]:.3f}")

    fig, axes = style.figure(ncols=2, width=4.4, height=3.3)

    ax = axes[0]
    ax.axhline(N_AC, color="#029e73", ls=":", lw=1.0)
    ax.annotate(r"$N_{\rm ac}=4$ (2 flexural + LA + twist)",
                (0.15, N_AC + 0.15), fontsize=7, color="#029e73")
    ax.plot(en[m], T_full[m], "o-", color="#0173b2", lw=1.4,
            label="full Bose occupancy (IR-subtracted run)")
    ax.plot(en[m], T_tap[m], "s--", color="#d55e00", lw=1.3,
            label=r"$\omega^2$-tapered occupancy"
                  rf" ($\omega_{{\rm reg}}={w_reg:.2f}$ THz)")
    ax.set_xlabel("frequency (THz)")
    ax.set_ylabel(r"ballistic transmission $T(\omega)=I/\Delta n$ (channels)")
    ax.set_xlim(0, WMAX)
    ax.set_ylim(0, 5.4)
    ax.legend(fontsize=7, loc="upper right")

    # ratio vs the analytic taper factor: the crushing IS the taper
    ax = axes[1]
    wf = np.linspace(1e-3, WMAX, 400)
    ax.plot(wf, wf**2 / (wf**2 + w_reg**2), "-", color="0.55", lw=1.2,
            label=r"analytic $\omega^2/(\omega^2+\omega_{\rm reg}^2)$")
    ok = m & np.isfinite(T_full) & np.isfinite(T_tap) & (T_full > 1e-6)
    ax.plot(en[ok], T_tap[ok] / T_full[ok], "s", color="#d55e00", ms=5,
            mfc="none", mew=1.3, label=r"$T_{\rm taper}/T_{\rm full}$ (data)")
    ax.set_xlabel("frequency (THz)")
    ax.set_ylabel("low-frequency suppression")
    ax.set_xlim(0, WMAX)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")

    style.save(fig, "sinw_d5a_ballistic_plateau", directory=FIGDIR)


if __name__ == "__main__":
    main()
