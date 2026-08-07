"""Is coarse-grid convergence just BLINDNESS to the resonance?

Paul's hypothesis (2026-08-07), against the naive reading of the
measured rho(dw/Gamma) non-monotonicity:

  "if we have e.g. Delta_omega/Gamma = 60, I don't intuitively see how
   we can accurately represent the resonance. Intuitively, the reason we
   converge is that we are so far away from what drives the physics that
   we don't even register it. Then, only as we get smaller, we hit some
   resonances, but don't resolve them enough to converge. Hence this
   non-monotonicity would make sense, and we would need to resolve even
   more to converge a proper resonance."

Three regimes, and the test needs a metric for "does the grid register
the resonance at all". Use the exact per-orbital spectral sum rule: for
each orbital i, completeness of the eigenvectors gives

    S_i = int_0^inf 2 omega (-1/pi) Im G^R_ii(omega) d omega = 1

exactly. Restricted to the two flat-band (B) orbitals -- whose entire
linewidth is anharmonic in this model -- S_B is a clean 0..1
"fraction of the resonance the grid registered".

Predictions, pre-registered in the output JSON before the run:
  R1 blind      S_B << 1  -> rho_damped < 1 (converges) but Gamma_em is
                            badly wrong and the state sits near ballistic
  R2 transition S_B ~ 0.2-0.95 -> rho_damped maximal, may exceed 1
  R3 resolved   S_B -> 1  -> rho_damped < 1 AGAIN and Gamma_em -> Gamma_ref
Falsifier: if rho_damped rises monotonically as S_B -> 1, the
instability is physical, not a representation artefact.

STABILITY CONVENTION (a correction to the earlier _grid_stability_law
run): arnoldi_spectrum returns the Jacobian of the RAW map F. The SCBA
iterate is damped, x <- (1-a) x + a F(x), so its eigenvalue is
m = 1 - a + a*lambda and the convergence criterion is |m| < 1, NOT
|lambda| < 1. Both are reported; rho_damped is the one that decides
convergence.

Run:  QTX_ARRAY_MODULE=numpy OMP_NUM_THREADS=4 \
        python phonon/studies/_grid_regimes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "phonon"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import phonon.studies._toy_grid_study as T  # noqa: E402
from phonon.studies._grid_sideband import _hwhm_interp  # noqa: E402
from phonon.studies._toy_grid_e7 import (  # noqa: E402
    arnoldi_spectrum, make_F)
from solver.leads import (  # noqa: E402
    build_device_hamiltonian, compute_obc_batch, solve_green_batch)
from solver.grids import build_frequency_grid  # noqa: E402

OUT = ROOT / "phonon/studies/out/grid_audit"

OMEGA_A, OMEGA_FLAT = 8.0, 10.0
FMAX = 2.2 * 2 * OMEGA_A
G_WEAK = 7.46761274223835e17          # the E1 "Gamma = 0.02" calibration
MIXING = 0.2                          # run_case default -> damped map
# The ladder must be pushed until the OBSERVABLE stops moving, not until
# S_B saturates: at nf = 1920 -> 3840 the width still changed by 27 %, so
# "resolved" was not established by the first pass. Rungs are cached in
# regimes.json, so extending only costs the new rungs.
NFS = (30, 60, 120, 240, 480, 960, 1920, 3840, 7680, 15360, 30720)
OFFSETS = (0.0, 0.2, 0.4, 0.6, 0.8)   # sub-cell alignment control
B_IDX = (1, 3)                        # flat-band orbitals, both slabs
A_IDX = (0, 2)                        # dispersive control

PREDICTIONS = {
    "R1_blind": "S_B << 1 -> rho_damped < 1, converges, Gamma_em far from "
                "reference, state near ballistic",
    "R2_transition": "S_B ~ 0.2-0.95 -> rho_damped maximal, may exceed 1",
    "R3_resolved": "S_B -> 1 -> rho_damped < 1 again AND Gamma_em -> ref",
    "falsifier": "rho_damped rising monotonically as S_B -> 1 would mean "
                 "the instability is physical, not a representation "
                 "artefact",
}


def sum_rule(freqs, gr, idx):
    """S_i = int_0^inf 2 w (-1/pi) Im G^R_ii dw  (exact value 1)."""
    pos = freqs > 0
    w = freqs[pos]
    out = []
    for i in idx:
        a = -(1.0 / np.pi) * gr[pos, i, i].imag
        out.append(float(np.trapezoid(2.0 * w * a, w)))
    return out


def ballistic_reference(nfreq_pos):
    """Ballistic (Sigma = 0) Gamma_em and J on the same grid."""
    h00, h01, phi = T.flatband_chain(OMEGA_A, OMEGA_FLAT, G_WEAK)
    freqs, dw, _, z2, _, _ = build_frequency_grid(
        (0.01, FMAX, nfreq_pos), eta_w_thz=1e-6)
    n_dof = h00.shape[0]
    N_D = T.N_SLABS * n_dof
    h_d = build_device_hamiltonian(h00, h01, T.N_SLABS)
    obc = compute_obc_batch(z2, h00, h01, freqs, T.T_L, T.T_R,
                            n_slabs=T.N_SLABS)
    zero = np.zeros((len(freqs), N_D, N_D), complex)
    gr, gl, gg = solve_green_batch(z2, h_d, obc, zero, zero, zero)
    pos = freqs > 0
    a_b = -gr[:, 1, 1].imag - gr[:, 3, 3].imag
    return dict(gamma_em=float(_hwhm_interp(freqs[pos], a_b[pos])),
                S_B=sum_rule(freqs, gr, B_IDX))


def one_rung(nf, frac, prev=None):
    """One (resolution, sub-cell offset) point.

    DEGENERATE-COLD-START GUARD. The scba_loop residual
    (phonon/solver/dense.py:900, ||f||/||sol||) is grid-dependent at the
    first iteration -- measured 5.6e-5 at nf=15360 but 8.8e-8 at
    nf=30720 -- so with a fixed tol=1e-7 the finest rung declares
    convergence at iteration 1 without ever iterating and reports the
    BALLISTIC branch (S_B = 0.006, Gamma = 6e-4). Any rung stopping in
    <= 2 iterations is re-run warm-started from the previous rung's
    fixed point, interpolated onto this grid (verified: nf=30720 then
    runs 24 iterations to S_B = 1.0000, Gamma = 0.01792).
    """
    wf = OMEGA_FLAT + frac * (FMAX / nf)
    h00, h01, phi = T.flatband_chain(OMEGA_A, wf, G_WEAK)
    res = T.run_case(h00, h01, phi, nf, FMAX, max_iter=400, tol=1e-7,
                     return_greens=True)
    warm = False
    if len(res["convergence_history"]) <= 2 and prev is not None:
        fnew = build_frequency_grid((0.01, FMAX, nf))[0]
        si = (T.interp_sigma(np.asarray(prev["Sigma_l"])[0],
                             prev["freqs"], fnew),
              T.interp_sigma(np.asarray(prev["Sigma_g"])[0],
                             prev["freqs"], fnew))
        res = T.run_case(h00, h01, phi, nf, FMAX, max_iter=400, tol=1e-7,
                         return_greens=True, sigma_init=si)
        warm = True
    freqs = res["freqs"]
    gr = res["G_retarded"][0]
    pos = freqs > 0
    a_b = -gr[:, 1, 1].imag - gr[:, 3, 3].imag
    gam_em = _hwhm_interp(freqs[pos], a_b[pos])
    s_b = sum_rule(freqs, gr, B_IDX)
    s_a = sum_rule(freqs, gr, A_IDX)
    # SECOND, independent width reader: -Im Sigma^R at the mode bin /(2w).
    # It disagrees with the spectral FWHM by ~40% at nf=3840, so both are
    # tracked until they agree -- a single reader cannot certify
    # convergence.
    gam_sig = float(T.emergent_gamma(res, wf, 1, T.N_SLABS))
    # a-priori blindness: the linewidth THIS grid's own quadrature sees
    gam_fb = T.first_born_gamma(h00, h01, phi, FMAX, nfreq_pos=nf,
                                omega_flat=wf)
    # Jacobian of the raw map at the fixed point, complex spectrum.
    # Krylov vectors are 64*nfreq doubles (~31 MB at nf=30720), so the
    # basis is trimmed on the largest rungs; m=16 vs m=24 agreed to 5e-6
    # on the dominant modulus, so m=10 is ample for rho.
    m_arn = 16 if nf < 15360 else 10
    F, _, _, N_D, nfreq = make_F(h00, h01, phi, nf, FMAX)
    ev = arnoldi_spectrum(F, np.asarray(res["Sigma_l"])[0],
                          np.asarray(res["Sigma_g"])[0], nfreq, N_D,
                          m=m_arn)
    damped = np.abs(1.0 - MIXING + MIXING * ev)
    return dict(
        nf=nf, frac=frac, dw=float(res["dw"]), omega_flat=float(wf),
        converged=bool(res["converged"]),
        n_it=int(len(res["convergence_history"])),
        gamma_em=float(gam_em), gamma_fb=float(gam_fb),
        gamma_sig=gam_sig,
        S_B=float(np.mean(s_b)), S_A=float(np.mean(s_a)),
        rho_raw=float(np.abs(ev).max()), rho_damped=float(damped.max()),
        lam_top=[float(ev[0].real), float(ev[0].imag)],
        J=float(np.real(np.sum(res["spectral_J_L"])) * res["dw"]),
        warm_started=warm,
    ), res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"predictions": PREDICTIONS, "mixing": MIXING, "fmax": FMAX,
           "g_weak": G_WEAK, "offsets": list(OFFSETS), "rows": []}
    # reuse rungs already computed (the ladder is extended, not redone)
    cache = {}
    old = OUT / "regimes.json"
    if old.exists():
        for r in json.loads(old.read_text()).get("rows", []):
            if "gamma_sig" in r:          # only rows with the new reader
                cache[(r["nf"], round(r["frac"], 6))] = r
        if cache:
            print(f"reusing {len(cache)} cached rungs from {old.name}")
    print("registration ladder: weak flat-band resonance, "
          f"mixing a={MIXING} (rho_damped = max|1-a+a*lam|)")
    print(f"  {'nf':>5} {'dw':>8} {'S_B':>16} {'S_A':>6} {'Gam_FB':>16} "
          f"{'Gam_em':>16} {'rho_raw':>13} {'rho_damp':>13} {'conv':>5}")
    prev_res = None
    for nf in NFS:
        pts = []
        for f in OFFSETS:
            hit = cache.get((nf, round(f, 6)))
            # a cached degenerate rung (n_it <= 2) must be recomputed
            if hit is not None and hit.get("n_it", 99) > 2:
                pts.append(hit)
                continue
            row, res = one_rung(nf, f, prev=prev_res)
            pts.append(row)
            if f == OFFSETS[0]:
                prev_res = res
        rep["rows"].extend(pts)

        def med(k):
            v = np.array([p[k] for p in pts], float)
            v = v[np.isfinite(v)]
            return (np.median(v), v.min(), v.max()) if v.size else (
                np.nan, np.nan, np.nan)

        sb, sb0, sb1 = med("S_B")
        sa, _, _ = med("S_A")
        gf, gf0, gf1 = med("gamma_fb")
        ge, ge0, ge1 = med("gamma_em")
        rr, rr0, rr1 = med("rho_raw")
        rd, rd0, rd1 = med("rho_damped")
        nconv = sum(p["converged"] for p in pts)
        print(f"  {nf:5d} {pts[0]['dw']:8.4f} "
              f"{sb:6.3f}[{sb0:.2f},{sb1:.2f}] {sa:6.3f} "
              f"{gf:7.5f}[{gf0:.4f},{gf1:.4f}] "
              f"{ge:7.5f}[{ge0:.4f},{ge1:.4f}] "
              f"{rr:5.2f}[{rr0:.2f},{rr1:.2f}] "
              f"{rd:5.3f}[{rd0:.3f},{rd1:.3f}] {nconv}/{len(pts)}")

    # reference = finest rung, offset-median
    fine = [r for r in rep["rows"] if r["nf"] == NFS[-1]]
    g_ref = float(np.median([r["gamma_em"] for r in fine]))
    rep["gamma_ref"] = g_ref
    ball = {nf: ballistic_reference(nf) for nf in NFS}
    rep["ballistic"] = {str(k): v for k, v in ball.items()}

    print(f"\nreference Gamma (nf={NFS[-1]}, offset-median) = {g_ref:.6f} THz")
    print("CONVERGENCE is judged by the SUCCESSIVE-RUNG change, not by the "
          "distance to the finest rung (which is 0 by construction).")
    print(f"  {'nf':>6} {'S_B':>7} {'Gam_spec':>9} {'Gam_sigma':>10} "
          f"{'d(spec)':>9} {'d(sig)':>9} {'rho_raw':>8} {'regime':>11}")
    prev_e = prev_s = None
    for nf in NFS:
        pts = [r for r in rep["rows"] if r["nf"] == nf]
        if not pts:
            continue
        sb = float(np.median([p["S_B"] for p in pts]))
        ge = float(np.median([p["gamma_em"] for p in pts]))
        gs = float(np.median([p.get("gamma_sig", np.nan) for p in pts]))
        rr = float(np.median([p["rho_raw"] for p in pts]))
        de = abs(ge - prev_e) / ge if prev_e else np.nan
        ds = abs(gs - prev_s) / gs if prev_s else np.nan
        reg = ("blind" if sb < 0.2 else
               "resolved" if sb > 0.95 else "transition")
        print(f"  {nf:6d} {sb:7.3f} {ge:9.5f} {gs:10.5f} {de:9.3f} "
              f"{ds:9.3f} {rr:8.2f} {reg:>11}")
        prev_e, prev_s = ge, gs
    print("  d(...) = |Gamma(n) - Gamma(n/2)| / Gamma(n); the ladder is "
          "converged only where BOTH readers agree AND both d -> 0")

    (OUT / "regimes.json").write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT / 'regimes.json'}")


if __name__ == "__main__":
    main()
