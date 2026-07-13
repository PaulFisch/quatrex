"""E7: direct Jacobian spectrum of the toy SCBA map (real-embedded Arnoldi).

Measures the dominant eigenvalues of J = dF/dSigma at (or near) the fixed
point for selected (coupling, grid) cases -- the toy analogue of the
production _jacobian_probe.py, with the OBC cached so an F-evaluation costs
one bubble + one dense G-solve.  J is real-linear (KK / conjugations), so
the probe real-embeds the complex Sigma vector and runs real Arnoldi; the
Hessenberg eigenvalues then come in genuine complex-conjugate pairs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _toy_grid_study as T  # noqa: E402
from _toy_grid_study import (build_frequency_grid, build_device_hamiltonian,  # noqa: E402
                             compute_obc_batch, N_SLABS, T_L, T_R,
                             compute_phph_self_energy_finite_multi_slab)
from solver.retarded import build_retarded  # noqa: E402
from solver.leads import solve_green_batch  # noqa: E402


def make_F(h00, h01, phi, nf, fmax, *, eta_w=1e-6, dc_handling="interpolate"):
    """Cached-OBC evaluation of Sigma -> F(Sigma) (one SCBA sweep)."""
    freqs, dw, _, z2, pos_mask, mid = build_frequency_grid(
        (0.01, fmax, nf), eta_w_thz=eta_w)
    nfreq = len(freqs)
    n_dof = h00.shape[0]
    N_D = N_SLABS * n_dof
    h_d = build_device_hamiltonian(h00, h01, N_SLABS)
    obc = compute_obc_batch(z2, h00, h01, freqs, T_L, T_R, n_slabs=N_SLABS)
    phi_dev = {(i, i, i): phi.astype(complex) for i in range(N_SLABS)}

    def gd(dense):
        return {(k, kp): dense[:, k * n_dof:(k + 1) * n_dof,
                               kp * n_dof:(kp + 1) * n_dof]
                for k in range(N_SLABS) for kp in range(N_SLABS)}

    def F(sl, sg):
        sr = build_retarded(sl, sg, freqs, method="fft")
        _, gl, gg = solve_green_batch(z2, h_d, obc, sr, sl, sg)
        sl_b, sg_b = compute_phph_self_energy_finite_multi_slab(
            gd(gl), gd(gg), phi_dev, N_SLABS, freqs, dw,
            dc_handling=dc_handling, n_threads=1)
        sl_new = np.zeros((nfreq, N_D, N_D), complex)
        sg_new = np.zeros_like(sl_new)
        for (i, j), b in sl_b.items():
            sl_new[:, i * n_dof:(i + 1) * n_dof,
                   j * n_dof:(j + 1) * n_dof] = b
        for (i, j), b in sg_b.items():
            sg_new[:, i * n_dof:(i + 1) * n_dof,
                   j * n_dof:(j + 1) * n_dof] = b
        return sl_new, sg_new

    return F, freqs, dw, N_D, nfreq


def pack(sl, sg):
    z = np.concatenate([sl.ravel(), sg.ravel()])
    return np.concatenate([z.real, z.imag])


def unpack(v, nfreq, N_D):
    n = nfreq * N_D * N_D
    z = v[:2 * n] + 1j * v[2 * n:]
    return z[:n].reshape(nfreq, N_D, N_D), z[n:].reshape(nfreq, N_D, N_D)


def arnoldi_spectrum(F, sl0, sg0, nfreq, N_D, *, m=24, seed=0):
    """Top eigenvalues of J at (sl0, sg0) via real-embedded Arnoldi."""
    f0l, f0g = F(sl0, sg0)
    x0 = pack(sl0, sg0)
    scale = float(np.linalg.norm(x0))
    if scale == 0.0:
        scale = 1.0
    eps = 1e-7 * scale

    rng = np.random.default_rng(seed)
    Q = []
    H = np.zeros((m + 1, m))
    q = rng.standard_normal(x0.size)
    q /= np.linalg.norm(q)
    Q.append(q)
    for k in range(m):
        dl, dg = unpack(eps * Q[k], nfreq, N_D)
        fl, fg = F(sl0 + dl, sg0 + dg)
        w = (pack(fl, fg) - pack(f0l, f0g)) / eps
        for i in range(k + 1):
            H[i, k] = float(np.dot(Q[i], w))
            w = w - H[i, k] * Q[i]
        H[k + 1, k] = float(np.linalg.norm(w))
        if H[k + 1, k] < 1e-12 * scale:
            m = k + 1
            break
        Q.append(w / H[k + 1, k])
        print(f"    arnoldi {k + 1}/{m}", flush=True)
    ev = np.linalg.eigvals(H[:m, :m])
    return ev[np.argsort(-np.abs(ev))]


def main() -> int:
    out = Path(__file__).resolve().parent / "out/toy_grid"
    out.mkdir(parents=True, exist_ok=True)
    omega_a, omega_flat = 8.0, 10.0
    fmax = 2.2 * 2 * omega_a
    g0 = 1e19
    h00, h01, phi = T.flatband_chain(omega_a, omega_flat, g0)
    gam0 = T.first_born_gamma(h00, h01, phi, fmax, omega_flat=omega_flat)
    G_LIST = {t: float(g0 * np.sqrt(t / gam0)) for t in (0.02, 0.2, 2.0)}
    results = []

    def probe(name, h00c, h01c, phic, nf, sl0, sg0, extra=None):
        F, freqs, dw, N_D, nfreq = make_F(h00c, h01c, phic, nf, fmax)
        ev = arnoldi_spectrum(F, sl0, sg0, nfreq, N_D)
        top = [dict(re=float(e.real), im=float(e.imag), mod=float(abs(e)))
               for e in ev[:6]]
        rec = dict(case=name, nfreq=nf, dw=dw, top=top, **(extra or {}))
        results.append(rec)
        tops = "  ".join(f"{t['mod']:.3f}({t['re']:+.2f}{t['im']:+.2f}i)"
                         for t in top[:4])
        print(f"  {name} nf={nf}: |lam| {tops}", flush=True)

    # --- broad-bath model at the converged fixed point --------------------
    for label, t, nf in (("weak", 0.02, 480),
                         ("strong", 2.0, 240), ("strong", 2.0, 480),
                         ("strong", 2.0, 960)):
        g = G_LIST[t]
        h00c, h01c, phic = T.flatband_chain(omega_a, omega_flat, g)
        print(f"== {label} g={g:.2e} nf={nf}: fixed point run ==", flush=True)
        res = T.run_case(h00c, h01c, phic, nf, fmax)
        sl0 = np.asarray(res["Sigma_l"])[0]
        sg0 = np.asarray(res["Sigma_g"])[0]
        probe(label, h00c, h01c, phic, nf, sl0, sg0,
              extra=dict(gamma_t=t, converged=bool(res["converged"]),
                         gamma_em=T.emergent_gamma(res, omega_flat, 1, 2)))

    # --- sharp-sharp pair on the scattering branch (eps chain, short) ----
    print("== sharp pair: eps chain to the scattering branch ==", flush=True)
    h00p, h01p, phip = T.sharp_pair_chain(omega_a, 5.0, 10.0, g0,
                                          eps_flat=0.02)
    gam0p = T.first_born_gamma(h00p, h01p, phip, fmax, omega_flat=10.0,
                               idx=2, nfreq_pos=8000, eta_w=5e-3)
    gp = float(g0 * np.sqrt(0.2 / gam0p))
    cont = None
    for eps in (1.0, 0.1, 0.02):
        h00e, h01e, phie = T.sharp_pair_chain(omega_a, 5.0, 10.0, gp,
                                              eps_flat=eps)
        si = None if cont is None else (cont["Sigma_l"], cont["Sigma_g"])
        cont = T.run_case(h00e, h01e, phie, 480, fmax, eta_w=1e-6,
                          sigma_init=si, max_iter=200)
        print(f"   eps={eps}: best "
              f"{min(cont['convergence_history']):.1e}", flush=True)
    sl0 = np.asarray(cont["Sigma_l"])[0]
    sg0 = np.asarray(cont["Sigma_g"])[0]
    probe("sharp-pair-branch", h00e, h01e, phie, 480, sl0, sg0,
          extra=dict(gamma_t=0.2,
                     gamma_em=T.emergent_gamma(cont, 10.0, 2, 3)))
    # ballistic branch of the SAME system for contrast
    cold = T.run_case(h00e, h01e, phie, 480, fmax, eta_w=1e-6)
    probe("sharp-pair-ballistic", h00e, h01e, phie, 480,
          np.asarray(cold["Sigma_l"])[0], np.asarray(cold["Sigma_g"])[0],
          extra=dict(gamma_t=0.2,
                     gamma_em=T.emergent_gamma(cold, 10.0, 2, 3)))

    (out / "results_e7.json").write_text(json.dumps(
        dict(results=results), indent=1,
        default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"saved {out}/results_e7.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
