"""Synthetic grid-resolution study: flat bands, Lorentzians, SCBA stability.

Usage:  python phonon/studies/_toy_grid_study.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon_inputs.constants import THZ_TO_RAD  # noqa: E402
from solver.dense import scba_loop  # noqa: E402
from solver.grids import build_frequency_grid  # noqa: E402
from solver.leads import (  # noqa: E402
    build_device_hamiltonian, compute_obc_batch)
from solver.se_finite import (  # noqa: E402
    compute_phph_self_energy_finite_multi_slab)

T_L, T_R = 310.0, 290.0
N_SLABS = 2


def flatband_chain(omega_a=8.0, omega_flat=10.0, g=0.1, kappa=0.0,
                   eps_flat=0.0):
    """2-DOF cell: dispersive branch A (0..omega_a) + (near-)flat band B.

    eps_flat > 0 gives B a bandwidth of 4*eps_flat -> the leads broaden it
    (the contact-broadened control); eps_flat = 0 leaves B's linewidth
    purely anharmonic."""
    k = (omega_a ** 2) / 4.0
    h00 = np.array([[2 * k, kappa],
                    [kappa, omega_flat ** 2 + 2 * eps_flat]], complex)
    h01 = np.array([[-k, 0.0], [0.0, -eps_flat]], complex)
    phi = np.zeros((2, 2, 2))
    for idx in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        phi[idx] = g  # B <-> A+A decay channel (fully symmetric)
    return h00, h01, phi


def sharp_pair_chain(omega_a=8.0, omega1=5.0, omega2=10.0, g=1e18,
                     eps_flat=0.0):
    """3-DOF cell: dispersive A (leads) + two flat bands at omega1 and
    omega2 = 2*omega1 coupled ONLY to each other by phi[B2,B1,B1] = g —
    the sharp-sharp-sharp three-phonon cycle whose widths are mutually
    generated (the theory's dangerous case)."""
    k = (omega_a ** 2) / 4.0
    h00 = np.array([[2 * k, 0, 0],
                    [0, omega1 ** 2 + 2 * eps_flat, 0],
                    [0, 0, omega2 ** 2 + 2 * eps_flat]], complex)
    h01 = np.zeros((3, 3), complex)
    h01[0, 0] = -k
    h01[1, 1] = -eps_flat
    h01[2, 2] = -eps_flat
    phi = np.zeros((3, 3, 3))
    for idx in ((2, 1, 1), (1, 2, 1), (1, 1, 2)):
        phi[idx] = g
    return h00, h01, phi


def run_case(h00, h01, phi, nfreq_pos, fmax, *, eta_w=1e-6, mixing=0.2,
             max_iter=400, tol=1e-6, dc_handling="interpolate",
             return_greens=False, sigma_init=None):
    freqs, dw, _, z2, pos_mask, mid = build_frequency_grid(
        (0.01, fmax, nfreq_pos), eta_w_thz=eta_w)
    nfreq = len(freqs)
    n_dof = h00.shape[0]
    N_D = N_SLABS * n_dof
    h_d = build_device_hamiltonian(h00, h01, N_SLABS)
    obc = compute_obc_batch(z2, h00, h01, freqs, T_L, T_R, n_slabs=N_SLABS)
    phi_dev = {(i, i, i): phi.astype(complex) for i in range(N_SLABS)}

    def se_kernel(gl_q, gg_q):
        sig_l = np.zeros((1, nfreq, N_D, N_D), complex)
        sig_g = np.zeros_like(sig_l)

        def gd(dense):
            return {(k, kp): dense[:, k * n_dof:(k + 1) * n_dof,
                                   kp * n_dof:(kp + 1) * n_dof]
                    for k in range(N_SLABS) for kp in range(N_SLABS)}

        sl_b, sg_b = compute_phph_self_energy_finite_multi_slab(
            gd(gl_q[0]), gd(gg_q[0]), phi_dev, N_SLABS, freqs, dw,
            dc_handling=dc_handling, n_threads=1)
        for (i, j), b in sl_b.items():
            sig_l[0, :, i * n_dof:(i + 1) * n_dof,
                  j * n_dof:(j + 1) * n_dof] = b
        for (i, j), b in sg_b.items():
            sig_g[0, :, i * n_dof:(i + 1) * n_dof,
                  j * n_dof:(j + 1) * n_dof] = b
        return sig_l, sig_g

    t0 = time.perf_counter()
    res = scba_loop(
        z2_arr=z2, freqs_thz=freqs, dw_thz=dw,
        omega_rad=freqs * THZ_TO_RAD, pos_mask=pos_mask,
        n_slabs=N_SLABS, n_dof=n_dof, N_D=N_D,
        H_D_list=[h_d], obc_list=[obc], btd_blocks_list=[(h00, h01)],
        n_kpts=1, se_kernel=se_kernel, T_L=T_L, T_R=T_R,
        max_scba_iter=max_iter, scba_tol=tol, conservation_tol=1e-2,
        mixing=mixing, anderson_mixing=False, anderson_depth=5,
        scattering_contacts=False, retarded="fft", verbose=False,
        divergence_guard=False, return_greens=return_greens,
        masses_primitive=np.ones(n_dof), sigma_init=sigma_init)
    res["wall"] = time.perf_counter() - t0
    res["freqs"] = freqs
    res["dw"] = dw
    return res


def rate_stats(hist, mixing):
    """Late-phase residual rate -> effective |lambda| (negative branch) and
    the erratic-vs-geometric character (std of log-increments)."""
    r = np.asarray(hist, float)
    r = r[np.isfinite(r) & (r > 0)]
    if r.size < 12:
        return dict(rate=np.nan, lam=np.nan, jitter=np.nan,
                    best=float(np.nanmin(r)) if r.size else np.nan)
    tail = np.log(r[-max(8, min(60, r.size // 2)):])
    incr = np.diff(tail)
    slope = float(np.mean(incr))
    rho = float(np.exp(slope))          # per-iteration residual factor
    # real-negative-eigenvalue branch: m = 1 - a(1+|lam|) = -rho (growth)
    # or +rho (decay slower than linear-from-positive):
    lam_grow = (1.0 + rho) / mixing - 1.0
    lam_decay = (1.0 - rho) / mixing - 1.0
    lam = lam_grow if slope > 0 else lam_decay
    return dict(rate=rho, lam=float(lam), jitter=float(np.std(incr)),
                best=float(r.min()))


def emergent_gamma(res, omega, idx, n_dof):
    """Which fixed point did we land on: anharmonic half-width from the
    converged Sigma^R at the flat-band pole (slab average)."""
    sr = np.asarray(res["Sigma_R"])
    if sr.ndim == 4:
        sr = sr[0]
    ib = int(np.argmin(np.abs(np.asarray(res["freqs"]) - omega)))
    vals = [-sr[ib, k * n_dof + idx, k * n_dof + idx].imag
            for k in range(N_SLABS)]
    return float(np.mean(vals) / (2.0 * omega))


def heat_current(res):
    return float(np.real(np.asarray(res["spectral_J_L"]).sum()) * res["dw"])


def interp_sigma(sig, f_from, f_to):
    """Linear interpolation of Sigma(omega) onto another grid, per element."""
    s = np.asarray(sig)
    if s.ndim == 4:
        s = s[0]
    flat = s.reshape(len(f_from), -1)
    out = np.empty((len(f_to), flat.shape[1]), complex)
    for j in range(flat.shape[1]):
        out[:, j] = (np.interp(f_to, f_from, flat[:, j].real)
                     + 1j * np.interp(f_to, f_from, flat[:, j].imag))
    return out.reshape((1, len(f_to)) + s.shape[1:])


def first_born_gamma(h00, h01, phi, fmax, *, nfreq_pos=960,
                     omega_flat=10.0, idx=1, eta_w=1e-4):
    """A-priori linewidth (theory eq:fgr analogue): one bubble evaluation
    on the BALLISTIC G, then Gamma = -Im Sigma^R_BB(Omega)/(2 Omega)."""
    from solver.retarded import build_retarded
    from solver.leads import solve_green_batch
    freqs, dw, _, z2, pos_mask, mid = build_frequency_grid(
        (0.01, fmax, nfreq_pos), eta_w_thz=eta_w)
    nfreq = len(freqs)
    n_dof = h00.shape[0]
    N_D = N_SLABS * n_dof
    h_d = build_device_hamiltonian(h00, h01, N_SLABS)
    obc = compute_obc_batch(z2, h00, h01, freqs, T_L, T_R, n_slabs=N_SLABS)
    zero = np.zeros((nfreq, N_D, N_D), complex)
    _, gl, gg = solve_green_batch(z2, h_d, obc, zero, zero, zero)
    phi_dev = {(i, i, i): phi.astype(complex) for i in range(N_SLABS)}

    def gd(dense):
        return {(k, kp): dense[:, k * n_dof:(k + 1) * n_dof,
                               kp * n_dof:(kp + 1) * n_dof]
                for k in range(N_SLABS) for kp in range(N_SLABS)}

    sl_b, sg_b = compute_phph_self_energy_finite_multi_slab(
        gd(gl), gd(gg), phi_dev, N_SLABS, freqs, dw,
        dc_handling="interpolate", n_threads=1)
    sl = np.zeros((nfreq, N_D, N_D), complex)
    sg = np.zeros_like(sl)
    for (i, j), b in sl_b.items():
        sl[:, i * n_dof:(i + 1) * n_dof, j * n_dof:(j + 1) * n_dof] = b
    for (i, j), b in sg_b.items():
        sg[:, i * n_dof:(i + 1) * n_dof, j * n_dof:(j + 1) * n_dof] = b
    sr = build_retarded(sl, sg, freqs, method="fft")
    ib = int(np.argmin(np.abs(freqs - omega_flat)))
    im = -sr[ib, idx, idx].imag
    return float(im / (2.0 * omega_flat))


def measure_gamma(h00, h01, phi, fmax, *, nfreq_pos=960):
    """Emergent anharmonic half-width of the flat band: Lorentzian fit to
    the B-projected spectral function on a fine, converged grid."""
    res = run_case(h00, h01, phi, nfreq_pos, fmax, mixing=0.3,
                   max_iter=300, tol=1e-8, return_greens=True)
    freqs = res["freqs"]
    gr = res["G_retarded"][0]           # (nfreq, N_D, N_D)
    a_b = -gr[:, 1, 1].imag - gr[:, 3, 3].imag   # B orbitals, both slabs
    pos = freqs > 0
    f = freqs[pos]
    a = a_b[pos]
    ipk = int(np.argmax(a))
    half = a[ipk] / 2.0
    left = np.where(a[:ipk] < half)[0]
    right = np.where(a[ipk:] < half)[0]
    if left.size and right.size:
        gam = 0.5 * (f[ipk + right[0]] - f[left[-1]])
    else:
        gam = np.nan
    return dict(gamma=float(gam), omega_peak=float(f[ipk]),
                converged=bool(res["converged"]),
                resid=float(res["scba_residual"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "phonon/studies/out/toy_grid")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    omega_a, omega_flat = 8.0, 10.0
    fmax = 2.2 * 2 * omega_a            # covers the 3-phonon support
    rows = []

    print("== Gamma calibration (first Born) ==", flush=True)
    # The bubble prefactor carries hbar_SI; physical vertices are ~1e19-20
    # in these units (cf. the real FC3 blocks, max|Phi| ~ 1e20).
    g0 = 1e19
    h00, h01, phi = flatband_chain(omega_a, omega_flat, g0)
    gam0 = first_born_gamma(h00, h01, phi, fmax, omega_flat=omega_flat)
    print(f"  reference g0={g0:.0e}: Gamma_FB={gam0:.5f} THz", flush=True)
    targets = (0.02, 0.2, 2.0)
    G_LIST = tuple(float(g0 * np.sqrt(t / gam0)) for t in targets)
    gammas = {}
    for g, t in zip(G_LIST, targets):
        h00, h01, phi = flatband_chain(omega_a, omega_flat, g)
        gam = first_born_gamma(h00, h01, phi, fmax, omega_flat=omega_flat)
        gammas[g] = dict(gamma=gam, target=t)
        print(f"  g={g:.3e}: Gamma_FB={gam:.4f} THz (target {t})",
              flush=True)

    print("== E1: resolution ladder ==", flush=True)
    for g in G_LIST:
        h00, h01, phi = flatband_chain(omega_a, omega_flat, g)
        for nf in (30, 60, 120, 240, 480):
            res = run_case(h00, h01, phi, nf, fmax)
            st = rate_stats(res["convergence_history"], 0.2)
            row = dict(exp="E1", g=g, nfreq=nf, dw=res["dw"],
                       gamma=gammas[g]["gamma"],
                       dw_over_gamma=res["dw"] / gammas[g]["gamma"],
                       gamma_em=emergent_gamma(res, omega_flat, 1, 2),
                       J=heat_current(res),
                       converged=bool(res["converged"]),
                       n_it=len(res["convergence_history"]),
                       cons=float(res["conservation_err"]), **st)
            rows.append(row)
            print(f"  g={g:.2e} nf={nf}: dw/G={row['dw_over_gamma']:6.2f} "
                  f"conv={row['converged']} best={row['best']:.1e} "
                  f"rate={row['rate']:.3f} |lam|={row['lam']:.2f} "
                  f"jit={row['jitter']:.2f} G_em={row['gamma_em']:.4f}",
                  flush=True)

    print("== E2: alignment scan ==", flush=True)
    g = G_LIST[1]
    nf = 60
    dw = fmax / nf
    for frac in np.linspace(0.0, 1.0, 9):
        # move the pole across one grid cell
        of = omega_flat + (frac - 0.5) * dw
        h00, h01, phi = flatband_chain(omega_a, of, g)
        res = run_case(h00, h01, phi, nf, fmax)
        st = rate_stats(res["convergence_history"], 0.2)
        d = abs(((of / dw) % 1.0) - 0.5) * dw  # distance to nearest bin...
        gem = emergent_gamma(res, of, 1, 2)
        rows.append(dict(exp="E2", g=g, nfreq=nf, dw=res["dw"], frac=frac,
                         omega_flat=of, gamma_em=gem,
                         converged=bool(res["converged"]),
                         n_it=len(res["convergence_history"]),
                         cons=float(res["conservation_err"]), **st))
        print(f"  frac={frac:.2f} (of={of:.3f}): conv={res['converged']} "
              f"best={st['best']:.1e} rate={st['rate']:.3f} "
              f"jit={st['jitter']:.2f} G_em={gem:.4f}", flush=True)

    print("== E3: DC handling + contact broadening ==", flush=True)
    for dc in ("interpolate", "zero", "keep"):
        for eps in (0.0, 1.0):
            h00, h01, phi = flatband_chain(omega_a, omega_flat, G_LIST[1],
                                           eps_flat=eps)
            try:
                res = run_case(h00, h01, phi, 60, fmax, dc_handling=dc)
            except RuntimeError as exc:
                print(f"  dc={dc} eps={eps}: OBC failed ({exc})", flush=True)
                continue
            st = rate_stats(res["convergence_history"], 0.2)
            rows.append(dict(exp="E3", dc=dc, eps_flat=eps, g=G_LIST[1],
                             gamma_em=emergent_gamma(res, omega_flat, 1, 2),
                             nfreq=60, converged=bool(res["converged"]),
                             n_it=len(res["convergence_history"]),
                             cons=float(res["conservation_err"]), **st))
            print(f"  dc={dc:11s} eps={eps}: conv={res['converged']} "
                  f"best={st['best']:.1e} n_it={len(res['convergence_history'])}",
                  flush=True)

    print("== E4: window content (fixed dw) ==", flush=True)
    dw_ref = fmax / 240
    for mult in (2.2, 1.5, 1.1):
        fm = mult * 2 * omega_a
        nf = max(2, int(round(fm / dw_ref)))
        h00, h01, phi = flatband_chain(omega_a, omega_flat, G_LIST[1])
        res = run_case(h00, h01, phi, nf, fm)
        st = rate_stats(res["convergence_history"], 0.2)
        rows.append(dict(exp="E4", fmax_mult=mult, g=G_LIST[1], nfreq=nf,
                         converged=bool(res["converged"]),
                         n_it=len(res["convergence_history"]),
                         cons=float(res["conservation_err"]),
                         gamma_em=emergent_gamma(res, omega_flat, 1, 2),
                         J=heat_current(res), **st))
        print(f"  fmax={mult}x2wmax (nf={nf}): conv={res['converged']} "
              f"best={st['best']:.1e} cons={res['conservation_err']:.1e}",
              flush=True)

    print("== E5: sharp-sharp flat-band pair ==", flush=True)
    # calibrate the pair coupling via first Born on B2 (omega2 = 10)
    g0p = 1e19
    EPS_PAIR = 0.02  # tiny dispersion: bound states become escaping, still sharp
    h00, h01, phi = sharp_pair_chain(omega_a, 5.0, 10.0, g0p, eps_flat=EPS_PAIR)
    gam0p = first_born_gamma(h00, h01, phi, fmax, omega_flat=10.0, idx=2,
                             nfreq_pos=8000, eta_w=5e-3)
    print(f"  pair reference g0={g0p:.0e}: Gamma_FB={gam0p:.6f}", flush=True)
    if not (gam0p > 1e-8):
        raise RuntimeError("pair calibration failed: first-Born Gamma ~ 0")
    for t in (0.02, 0.2):
        gp = float(g0p * np.sqrt(t / max(gam0p, 1e-300)))
        for nf in (30, 60, 120, 240, 480):
            h00, h01, phi = sharp_pair_chain(omega_a, 5.0, 10.0, gp,
                                             eps_flat=EPS_PAIR)
            res = run_case(h00, h01, phi, nf, fmax)
            st = rate_stats(res["convergence_history"], 0.2)
            dw = res["dw"]
            ok = bool(res["converged"] and np.isfinite(st["best"]))
            gem2 = emergent_gamma(res, 10.0, 2, 3)
            gem1 = emergent_gamma(res, 5.0, 1, 3)
            rows.append(dict(exp="E5", g=gp, gamma_t=t, nfreq=nf, dw=dw,
                             dw_over_gamma=dw / t,
                             gamma_em_B2=gem2, gamma_em_B1=gem1,
                             J=heat_current(res),
                             converged=ok,
                             n_it=len(res["convergence_history"]),
                             cons=float(res["conservation_err"]), **st))
            print(f"  Gt={t} nf={nf}: dw/G={dw / t:7.2f} "
                  f"conv={res['converged']} best={st['best']:.1e} "
                  f"n_it={len(res['convergence_history'])} "
                  f"G_em(B2)={gem2:.4f} G_em(B1)={gem1:.4f}",
                  flush=True)
    # alignment scan on the UNRESOLVED sharp pair
    print("== E5b: alignment of the sharp pair ==", flush=True)
    gp = float(g0p * np.sqrt(0.02 / max(gam0p, 1e-300)))
    nf = 60
    dw = fmax / nf
    for frac in np.linspace(0.0, 1.0, 9):
        o1 = 5.0 + (frac - 0.5) * dw
        h00, h01, phi = sharp_pair_chain(omega_a, o1, 2 * o1, gp,
                                         eps_flat=EPS_PAIR)
        res = run_case(h00, h01, phi, nf, fmax)
        st = rate_stats(res["convergence_history"], 0.2)
        okb = bool(res["converged"] and np.isfinite(st["best"]))
        gem2 = emergent_gamma(res, 2 * o1, 2, 3)
        rows.append(dict(exp="E5b", g=gp, frac=frac, omega1=o1, nfreq=nf,
                         gamma_em_B2=gem2, converged=okb,
                         n_it=len(res["convergence_history"]),
                         cons=float(res["conservation_err"]), **st))
        print(f"  frac={frac:.2f}: conv={res['converged']} "
              f"best={st['best']:.1e} G_em(B2)={gem2:.4f}", flush=True)

    print("== E6: warm-start branch-sustain test ==", flush=True)
    # Can a coarse grid SUSTAIN the scattering fixed point it cannot reach
    # cold?  Seed the coarse iteration with the fine-grid converged Sigma.
    t6 = 0.2
    gp6 = float(g0p * np.sqrt(t6 / gam0p))
    h00, h01, phi = sharp_pair_chain(omega_a, 5.0, 10.0, gp6,
                                     eps_flat=EPS_PAIR)
    ref = run_case(h00, h01, phi, 480, fmax)
    gref = emergent_gamma(ref, 10.0, 2, 3)
    print(f"  fine reference nf=480: Gamma_em(B2)={gref:.4f} "
          f"(first-Born target {t6})", flush=True)
    for nf in (30, 60, 120, 240):
        cold = run_case(h00, h01, phi, nf, fmax)
        fr_c = np.asarray(cold["freqs"])
        si = (interp_sigma(ref["Sigma_l"], ref["freqs"], fr_c),
              interp_sigma(ref["Sigma_g"], ref["freqs"], fr_c))
        warm = run_case(h00, h01, phi, nf, fmax, sigma_init=si)
        for tag, res in (("cold", cold), ("warm", warm)):
            gem = emergent_gamma(res, 10.0, 2, 3)
            st = rate_stats(res["convergence_history"], 0.2)
            rows.append(dict(exp="E6", nfreq=nf, mode=tag, g=gp6,
                             gamma_t=t6, gamma_em=gem, gamma_ref=gref,
                             J=heat_current(res),
                             converged=bool(res["converged"]
                                            and np.isfinite(st["best"])),
                             n_it=len(res["convergence_history"]), **st))
            print(f"  nf={nf} {tag}: Gamma_em(B2)={gem:.4f} "
                  f"conv={res['converged']} "
                  f"n_it={len(res['convergence_history'])} "
                  f"best={st['best']:.1e}", flush=True)

    (args.out / "results.json").write_text(json.dumps(
        dict(gammas={str(k): v for k, v in gammas.items()}, rows=rows),
        indent=1,
        default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"saved {args.out}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
