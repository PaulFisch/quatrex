"""Numerical verification of the resonance / loop-gain theory of the
phonon SCBA (thesis 40_scba.tex, sub:grid_resolution, eq:resolvent_gain).

Runs on tortin against the stored jprobe Sigma snapshots (CNT(3,3),
181-pt grid 0..55 THz, dw = 55/180 = 0.3056 THz, eta = 1e-12 i.e. the
eta = 0 limit; NO broadening is added anywhere by this study). One NPZ
per state under phonon/studies/out/resonance_gain/.

Four demonstrations
-------------------
1. spectra: rebuild G^R(omega) from the snapshot Sigma through the
   PRODUCTION engine solver (same spectral OBC / system matrix /
   z2 = omega^2 + 2i*eta*omega as the run that made the snapshot),
   project the spectral function on the harmonic eigenmodes s of the
   device dynamical matrix, A_s(omega) = -2 Im <s|G^R(omega)|s>
   (thesis eq:spectral_mode with A = i(G^R - G^A)), and fit the
   omega^2-Lorentzian A(omega) = 2 Z (2 Omega Gamma) /
   ((omega^2-Omega^2)^2 + (2 Omega Gamma)^2) + c to every resolvable
   peak. Independently, the half-width is read from the self-energy,
   Gamma_s = -Im <s|Sigma^R(Omega_s)|s> / (2 Omega_s), split into the
   anharmonic part (snapshot Sigma^R, BTD-banded as the Dyson solve
   consumes it) and the lead part (OBC Sigma^R corners).
2. kicks (L2_fp only): mode-diagonal perturbation
   dSigma^R = amp * |s><s| on the three grid bins nearest Omega_s,
   one engine Dyson re-solve (NO self-consistency), measure
   ||dA_s|| / ||A_s|| (l2 over the grid) against the theory ratio
   |dSigma| / (2 Omega_s Gamma_s). Two kick phases: amp*(-i)
   (width-like) and amp*(+1) (pole-shift-like; the discretisation
   mechanism of the theory acts on this one). Two amplitudes for
   linearity.
3. channels: bosonic full-axis extension of the engine G
   (G^<_ij(-w) = G^>_ji(w), the engine's own fold; DC bin zeroed like
   the production kernel), reference bubble through the dense kernel
   phonon.solver.se_finite.compute_phph_self_energy_finite_multi_slab
   with the ENGINE's vertex blocks (sse.phi_blocks) and prefactor
   (0.5j hbar dw / 2pi), then the mode-pair channel decomposition:
   with the mode-DIAGONAL approximation G^x ~ sum_s g^x_s |s><s|,
   the bubble separates into scalar channels
       Sigma^{x,(s1,s2)}_{s''}(w) = pref * K[s'',s1,s2]
                                    * (g^x_{s1} (*) g^x_{s2})(w),
   where K is the squared mode-basis vertex Phi~^2_{s'' s1 s2}
   generalised to the slab-resolved quadratic form that carries the
   kernel's g-band / sigma-band masks (the July snapshots were made
   with the band-1 masked kernel; for L2 the band is complete).
   Gates: (a) 'wiring': the kernel run on the exactly-mode-diagonal
   G_test must equal the channel sum to rounding; (b) 'fixed point':
   the kernel on the rebuilt G must reproduce the snapshot
   Sigma^{<,>} on omega >= 0 (global sign recorded -- the engine
   stores occupation-positive G/Sigma, the fold makes the bubble
   even under the global <-> sign flip); (c) 'closure': channel sum
   vs the full-G reference = the mode-off-diagonal error.
   Output: partial widths Gamma^{(s1,s2)}_{s''} =
   -Im Sigma^{R,(s1,s2)}_{s''}(Omega_s'') / (2 Omega_s''), the
   channel-fraction matrix and its row sums.
4. gain: the theory's link-gain matrix (eq:resolvent_gain)
       M[s'',s] = Omega_s'' * sum_{s'} (Gamma^{(s,s')}_{s''}
                  + Gamma^{(s',s)}_{s''}) / (Omega_s Gamma_s)
   (the two-term sum is the exact bilinear sensitivity of the channel
   sum to mode s). Branch 'phys': Gamma_s = full physical width
   (anharmonic + lead). Branch 'grid': each column multiplied by the
   discretisation enhancement max(1, dw/Gamma_s) of the theory
   (unresolved poles enter the discrete convolution with weight
   sensitivity ~ dw/Gamma_s^2). Spectral radius over all modes /
   sharp modes (Gamma < dw) / IR modes (Omega <= 2 THz), to be
   compared with the measured power-iteration eigenvalues
   (jp_* result.json): L2 fp 4.33/3.94, L2 stall 3.51/3.30,
   L4 stall 5.07/4.75/4.15.

Usage (tortin, single rank):
    QX_ROOT=/usr/scratch/mont-fort11/pfischill/quatrex \
    PYTHONPATH=$QX_ROOT/src:$QX_ROOT/phonon:$QX_ROOT \
    python _resonance_gain_study.py --state L2_fp [--stages spectra,kicks,channels,gain]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("QX_ROOT", "/usr/scratch/mont-fort11/pfischill/quatrex"))
for p in (str(ROOT / "src"), str(ROOT / "phonon"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("QTX_ARRAY_MODULE", "numpy")

AT = ROOT / "phonon/studies/out/anderson_test"
OUT = ROOT / "phonon/studies/out/resonance_gain"

STATES = {
    "L2_fp": dict(
        config=AT / "cnt33_linear/quatrex_config.toml",
        snapshot=AT / "jprobe_snaps/L2_fp.npz",
        n_slabs=2, g_cut=1, sig_cut=1,   # band-1 masked kernel (complete for L2)
        measured_lambda=[4.3326, 3.9422]),
    "L2_andstall": dict(
        config=AT / "cnt33_linear/quatrex_config.toml",
        snapshot=AT / "jprobe_snaps/L2_andstall.npz",
        n_slabs=2, g_cut=1, sig_cut=1,
        measured_lambda=[3.5073, 3.3000]),
    "L4_stall": dict(
        config=AT / "cnt33_L4_linear/quatrex_config.toml",
        snapshot=AT / "jprobe_snaps/L4_stall.npz",
        n_slabs=4, g_cut=1, sig_cut=1,   # snapshots predate sse_g_band>1 (2026-07-14)
        measured_lambda=[5.0721, 4.7534, 4.1522, 4.1434]),
}

N_THREADS = int(os.environ.get("QX_STUDY_THREADS", "8"))


# ---------------------------------------------------------------------------
# Engine plumbing
# ---------------------------------------------------------------------------


def build_engine(config_path):
    """Instantiate the production SCBA for `config_path` (single rank)."""
    from quatrex.core.config import parse_config, setup_context
    cfg = parse_config(str(config_path))
    setup_context(cfg)
    from quatrex.core.scba import SCBA
    scba = SCBA(cfg)
    ph = scba.subsystems["phonon"]
    sse = None
    for inter in getattr(scba, "interactions", []):
        sse = getattr(inter, "sigma_phonon_phonon", None)
        if sse is not None:
            break
    return cfg, scba, ph, sse


def densify(m, n_dof_total):
    """DSDBCOO (stack state, single rank) -> dense (ne, N, N)."""
    from qttools.utils.gpu_utils import get_host
    data = np.asarray(get_host(m.data))
    rows = np.asarray(get_host(m.rows)).astype(int)
    cols = np.asarray(get_host(m.cols)).astype(int)
    data = data.reshape(-1, data.shape[-1])
    out = np.zeros((data.shape[0], n_dof_total, n_dof_total), complex)
    out[:, rows, cols] = data
    return out


def load_state(scba, snapshot_path):
    """Load a QX_SAVE_SIGMA snapshot into the SCBA data buffers."""
    from qttools import xp
    sd = np.load(snapshot_path)
    for key, buf in (("sigma_lesser", scba.data.sigma_lesser),
                     ("sigma_greater", scba.data.sigma_greater),
                     ("sigma_retarded", scba.data.sigma_retarded_hermitian)):
        assert sd[key].shape == buf.data.shape, (key, sd[key].shape, buf.data.shape)
        assert buf.distribution_state == "stack"
        buf.data[:] = xp.asarray(sd[key])
    return {k: np.asarray(sd[k]) for k in
            ("sigma_lesser", "sigma_greater", "sigma_retarded")}


def solve_G(scba, ph, N):
    """One engine Dyson solve at the current Sigma buffers -> dense G's."""
    scba.data.g_retarded.allocate_data()
    for g in (scba.data.g_lesser, scba.data.g_greater, scba.data.g_retarded):
        g.data[:] = 0.0
    ph.solve(scba.data.sigma_lesser, scba.data.sigma_greater,
             scba.data.sigma_retarded_hermitian,
             out=(scba.data.g_lesser, scba.data.g_greater,
                  scba.data.g_retarded))
    return (densify(scba.data.g_retarded, N),
            densify(scba.data.g_lesser, N),
            densify(scba.data.g_greater, N))


def btd_mask(N, n_slabs, band):
    b = N // n_slabs
    m = np.zeros((N, N))
    for i in range(n_slabs):
        for j in range(n_slabs):
            if abs(i - j) <= band:
                m[i*b:(i+1)*b, j*b:(j+1)*b] = 1.0
    return m


# ---------------------------------------------------------------------------
# Stage 1: mode-projected spectra + Lorentzian fits
# ---------------------------------------------------------------------------


def lorentz_w2(w, Om, Ga, Z, c):
    return 2.0 * Z * (2.0 * Om * Ga) / ((w**2 - Om**2)**2
                                        + (2.0 * Om * Ga)**2) + c


def fit_mode(w, A, Om0, Ga0):
    """Least-squares Lorentzian fit around the peak of one A_s comb.

    Returns dict(Omega, Gamma, Z, c, ok, n_win, resid).
    """
    from scipy.optimize import curve_fit
    p = int(np.argmax(A))
    peak = A[p]
    lo, hi = p, p
    while lo > 1 and A[lo - 1] < A[lo] and A[lo - 1] > 0.02 * peak:
        lo -= 1
    while hi < len(w) - 2 and A[hi + 1] < A[hi] and A[hi + 1] > 0.02 * peak:
        hi += 1
    lo, hi = max(lo, p - 8), min(hi, p + 8)
    sl = slice(lo, hi + 1)
    n_win = hi - lo + 1
    out = dict(Omega=np.nan, Gamma=np.nan, Z=np.nan, c=np.nan,
               ok=False, n_win=n_win, resid=np.nan, peak_w=w[p])
    if n_win < 4 or peak <= 0:
        return out
    try:
        p0 = [max(w[p], 1e-2), max(Ga0, 1e-4), 1.0, 0.0]
        popt, _ = curve_fit(lorentz_w2, w[sl], A[sl], p0=p0, maxfev=20000)
        Om, Ga, Z, c = popt
        resid = float(np.linalg.norm(lorentz_w2(w[sl], *popt) - A[sl])
                      / np.linalg.norm(A[sl]))
        out.update(Omega=abs(Om), Gamma=abs(Ga), Z=Z, c=c,
                   ok=resid < 0.35, resid=resid)
    except Exception:
        pass
    return out


def stage_spectra(state, w, dw, evals2, evecs, Gr, sig_R_snap, obc_R,
                  n_slabs):
    N = evecs.shape[0]
    nm = N
    Omega = np.sqrt(np.clip(evals2, 0.0, None))
    # Mode projections.
    P = evecs                                    # (N, nm), real orthonormal
    A_modes = -2.0 * np.einsum("is,wij,js->ws", P, Gr.imag, P,
                               optimize=True).T       # (nm, ne)
    band = min(1, n_slabs - 1)
    m_btd = btd_mask(N, n_slabs, band)
    sigR_s = np.einsum("is,wij,js->ws", P, sig_R_snap * m_btd, P,
                       optimize=True).T                # (nm, ne) complex
    obc_s = np.einsum("is,wij,js->ws", P, obc_R, P, optimize=True).T

    def half_width_at(vals_im, Om_s):
        """-Im X_s(Omega_s) / (2 Omega_s), linear interp on the grid."""
        if Om_s < w[1]:
            return np.nan
        v = np.interp(Om_s, w, vals_im)
        return -v / (2.0 * Om_s)

    Gam_anh = np.array([half_width_at(sigR_s[s].imag, Omega[s])
                        for s in range(nm)])
    Gam_lead = np.array([half_width_at(obc_s[s].imag, Omega[s])
                         for s in range(nm)])
    fits = []
    for s in range(nm):
        Ga0 = np.nansum([Gam_anh[s], Gam_lead[s]])
        fits.append(fit_mode(w, A_modes[s], Omega[s],
                             Ga0 if np.isfinite(Ga0) and Ga0 > 0 else dw))
    res = dict(
        Omega_harm=Omega,
        A_modes=A_modes,
        sigR_modes=sigR_s,
        obcR_modes=obc_s,
        Gamma_anh=Gam_anh,
        Gamma_lead=Gam_lead,
        fit_Omega=np.array([f["Omega"] for f in fits]),
        fit_Gamma=np.array([f["Gamma"] for f in fits]),
        fit_Z=np.array([f["Z"] for f in fits]),
        fit_c=np.array([f["c"] for f in fits]),
        fit_ok=np.array([f["ok"] for f in fits]),
        fit_resid=np.array([f["resid"] for f in fits]),
        fit_nwin=np.array([f["n_win"] for f in fits]),
        peak_w=np.array([f["peak_w"] for f in fits]),
    )
    Gam_tot = np.where(np.isfinite(Gam_anh), Gam_anh, 0.0) \
        + np.where(np.isfinite(Gam_lead), Gam_lead, 0.0)
    res["Gamma_tot"] = Gam_tot
    n_res = int(np.sum((Gam_tot > dw) & (Omega > w[1])))
    print(f"[spectra] {state}: {nm} modes, {n_res} with Gamma_tot > dw "
          f"(dw={dw:.4f} THz); Gamma_anh range "
          f"[{np.nanmin(Gam_anh):.2e}, {np.nanmax(Gam_anh):.2e}] THz",
          flush=True)
    return res


# ---------------------------------------------------------------------------
# Stage 2: distortion-response kicks
# ---------------------------------------------------------------------------


def stage_kicks(scba, ph, N, w, dw, spec, sig_R_snap, base_A):
    from qttools import xp
    Omega = spec["Omega_harm"]
    Gam_tot = spec["Gamma_tot"]
    nm = N
    valid = (Omega > 0.2) & np.isfinite(Gam_tot) & (Gam_tot > 0)
    order = np.argsort(np.where(valid, Gam_tot, np.inf))
    sharp = [s for s in order if valid[s]][:6]
    order_desc = np.argsort(np.where(valid, -Gam_tot, -np.inf))
    broad = [s for s in order_desc if valid[s] and s not in sharp][:4]
    mids = [s for s in order[len(order) // 2 - 2: len(order) // 2 + 2]
            if valid[s] and s not in sharp and s not in broad][:4]
    modes = sharp + mids + broad
    print(f"[kicks] modes: {modes}", flush=True)

    P = np.asarray(spec["A_modes"])  # base A_s
    rows = np.asarray(scba.data.sigma_retarded_hermitian.rows).astype(int)
    cols = np.asarray(scba.data.sigma_retarded_hermitian.cols).astype(int)
    base_sr = np.array(scba.data.sigma_retarded_hermitian.data, copy=True)

    records = []
    t0 = time.time()
    # Fixed ABSOLUTE kick amplitude shared by all modes: the theory ratio
    # x = |dSigma|/(2 Omega Gamma) then spreads with the mode sharpness
    # (per-mode-scaled amplitudes collapse x onto the chosen c_rel). The
    # per-mode-scaled pair (c_rel = 1e-3, 1e-2) doubles as the linearity
    # check.
    a0 = 1e-3 * float(np.median([2.0 * Omega[s] * Gam_tot[s]
                                 for s in modes]))
    for s in modes:
        Om, Ga = Omega[s], Gam_tot[s]
        p = int(np.argmin(np.abs(w - Om)))
        bins = [b for b in (p - 1, p, p + 1) if 0 < b < len(w)]
        e = np.ascontiguousarray(np.real(np.asarray(spec_evec(scba, s))))
        proj = np.outer(e, e)
        proj_nnz = proj[rows, cols]
        for phase, tag in ((-1j, "imag"), (1.0, "real")):
            for amp in (1e-3 * 2.0 * Om * Ga, 1e-2 * 2.0 * Om * Ga,
                        a0, 3.0 * a0):
                c_rel = amp / (2.0 * Om * Ga)
                scba.data.sigma_retarded_hermitian.data[:] = base_sr
                for b in bins:
                    scba.data.sigma_retarded_hermitian.data[b] += \
                        xp.asarray(phase * amp * proj_nnz)
                Gr_k, _, _ = solve_G(scba, ph, N)
                A_k = -2.0 * np.einsum("wij,i,j->w", Gr_k.imag, e, e)
                dA = A_k - base_A[s]
                y = float(np.linalg.norm(dA) / np.linalg.norm(base_A[s]))
                # peak-local response: the theory's statement is about the
                # distortion of the resonance peak itself, not the l2 norm
                # over the entire comb of the projected spectral function.
                pk = int(np.argmax(base_A[s]))
                win = slice(max(pk - 3, 0), min(pk + 4, len(w)))
                y_peak = float(np.max(np.abs(dA[win]))
                               / np.max(base_A[s][win]))
                x_phys = amp / (2.0 * Om * Ga)
                x_grid = amp / (2.0 * Om * max(Ga, dw / 2.0))
                records.append((s, Om, Ga, tag == "real", c_rel, amp,
                                x_phys, x_grid, y, y_peak))
                print(f"[kicks] s={s:3d} Om={Om:6.2f} Ga={Ga:.2e} "
                      f"{tag}@{c_rel:.0e}: y={y:.3e} y_peak={y_peak:.3e} "
                      f"x_phys={x_phys:.3e} x_grid={x_grid:.3e} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    scba.data.sigma_retarded_hermitian.data[:] = base_sr
    rec = np.array(records, dtype=float)
    return dict(kick_table=rec,
                kick_columns=np.array(
                    "mode Omega Gamma_tot is_real c_rel amp x_phys x_grid "
                    "y y_peak".split()))


_EVECS_CACHE = {}


def spec_evec(scba, s):
    return _EVECS_CACHE["evecs"][:, s]


# ---------------------------------------------------------------------------
# Stage 3: channel decomposition
# ---------------------------------------------------------------------------


def fold_full_axis(w, Gl, Gg):
    """Engine bosonic fold to the symmetric axis: X^<_ij(-w) = X^>_ji(w)."""
    neg_l = Gg[1:][::-1].transpose(0, 2, 1)
    neg_g = Gl[1:][::-1].transpose(0, 2, 1)
    w_full = np.concatenate([-w[1:][::-1], w])
    Gl_f = np.concatenate([neg_l, Gl], axis=0)
    Gg_f = np.concatenate([neg_g, Gg], axis=0)
    return w_full, Gl_f, Gg_f


def g_blocks_from_dense(G, n_slabs, b, g_cut):
    out = {}
    for K in range(n_slabs):
        for Kp in range(n_slabs):
            if abs(K - Kp) <= g_cut:
                out[(K, Kp)] = np.ascontiguousarray(
                    G[:, K*b:(K+1)*b, Kp*b:(Kp+1)*b])
    return out


def dense_from_blocks(blocks, n_slabs, b, ne):
    N = n_slabs * b
    out = np.zeros((ne, N, N), complex)
    for (I, J), blk in blocks.items():
        out[:, I*b:(I+1)*b, J*b:(J+1)*b] = blk
    return out


def build_K_tensor(phi_blocks, evecs, n_slabs, b, g_cut, sig_cut):
    """Slab-resolved squared mode vertex K[s'', s1, s2].

    K = F^T Q F over the slab legs, F[(s'' I),(s1 K1),(s2 K2)] =
    sum_{ace} Phi_dev[a,c,e] e^{s''}_a|_I e^{s1}_c|_K1 e^{s2}_e|_K2,
    Q allowing |I-J| <= sig_cut, |K1-K1'| <= g_cut, |K2-K2'| <= g_cut.
    For a complete band this reduces to Phi~^2.
    """
    N = n_slabs * b
    nm = N
    Phi = np.zeros((N, N, N))
    for (I, K1, K2), blk in phi_blocks.items():
        Phi[I*b:(I+1)*b, K1*b:(K1+1)*b, K2*b:(K2+1)*b] += np.real(blk)
    # slab-resolved eigenvector legs (N, nm*nb)
    E = np.zeros((N, nm * n_slabs))
    for K in range(n_slabs):
        E[K*b:(K+1)*b, K::n_slabs] = evecs[K*b:(K+1)*b, :]
    M = nm * n_slabs
    T1 = np.tensordot(Phi, E, axes=([0], [0]))       # (c, e, m0)
    T2 = np.tensordot(T1, E, axes=([0], [0]))        # (e, m0, m1)
    F = np.tensordot(T2, E, axes=([0], [0]))         # (m0, m1, m2)
    del T1, T2
    F = F.reshape(nm, n_slabs, nm, n_slabs, nm, n_slabs)
    F = np.ascontiguousarray(F.transpose(0, 2, 4, 1, 3, 5))
    F = F.reshape(nm * nm * nm, n_slabs**3)
    # allowed mask over ((I,K1,K2),(J,K1',K2'))
    idx = np.arange(n_slabs)
    okS = (np.abs(idx[:, None] - idx[None, :]) <= sig_cut)
    okG = (np.abs(idx[:, None] - idx[None, :]) <= g_cut)
    Q = np.einsum("IJ,kl,mn->IkmJln", okS.astype(float), okG.astype(float),
                  okG.astype(float)).reshape(n_slabs**3, n_slabs**3)
    K = np.einsum("ij,ij->i", F @ Q, F)
    return K.reshape(nm, nm, nm)


def stage_channels(state, cfginfo, w, dw, evecs, Gl, Gg, phi_blocks,
                   snap, spec, n_slabs):
    from phonon.solver.se_finite import (
        bubble_prefactor, compute_phph_self_energy_finite_multi_slab)

    N = evecs.shape[0]
    nm = N
    b = N // n_slabs
    g_cut, sig_cut = cfginfo["g_cut"], cfginfo["sig_cut"]
    w_full, Gl_f, Gg_f = fold_full_axis(w, Gl, Gg)
    nf = len(w_full)
    mid = nf // 2
    n_fft = 2 * nf - 1
    freq_sl = slice(mid, mid + nf)
    pref = bubble_prefactor(dw)

    # --- full-G reference bubble (dense kernel, engine vertex) ----------
    t0 = time.time()
    gl_b = g_blocks_from_dense(Gl_f, n_slabs, b, g_cut)
    gg_b = g_blocks_from_dense(Gg_f, n_slabs, b, g_cut)
    sl_ref_b, sg_ref_b = compute_phph_self_energy_finite_multi_slab(
        gl_b, gg_b, phi_blocks, n_slabs, w_full, dw,
        sigma_cutoff=None, g_cutoff=g_cut, dc_handling="zero",
        n_threads=N_THREADS)
    sl_ref = dense_from_blocks(sl_ref_b, n_slabs, b, nf)
    sg_ref = dense_from_blocks(sg_ref_b, n_slabs, b, nf)
    print(f"[channels] reference bubble: {time.time()-t0:.1f}s", flush=True)

    # --- fixed-point check vs snapshot Sigma^{<,>} on omega >= 0 --------
    # The engine masks the omega = 0 OUTPUT bin of the SSE (the DC bin
    # carries the folded acoustic near-divergence); exclude it from the
    # comparison, as the engine's own map does.
    fp = {}
    for key, ref in (("lesser", sl_ref), ("greater", sg_ref)):
        snapd = snap[f"sigma_{key}"]      # dense (ne, N, N), densified upstream
        nrm = np.linalg.norm(snapd[1:])
        for sign in (+1.0, -1.0):
            fp[f"{key}_{'p' if sign > 0 else 'm'}"] = float(
                np.linalg.norm(sign * ref[mid + 1:] - snapd[1:]) / nrm)
    fp_sign = 1.0 if (fp["lesser_p"] <= fp["lesser_m"]) else -1.0
    fp_err = min(fp["lesser_p"], fp["lesser_m"])
    print(f"[channels] fixed-point check: sign={fp_sign:+.0f} "
          f"rel err lesser={fp_err:.3e} greater="
          f"{min(fp['greater_p'], fp['greater_m']):.3e}", flush=True)

    # --- mode-diagonal channel machinery --------------------------------
    P = evecs
    gl_m = np.einsum("is,wij,js->sw", P, Gl_f, P, optimize=True)
    gg_m = np.einsum("is,wij,js->sw", P, Gg_f, P, optimize=True)
    gl_m[:, mid] = 0.0
    gg_m[:, mid] = 0.0

    K = build_K_tensor(phi_blocks, P, n_slabs, b, g_cut, sig_cut)
    print(f"[channels] K tensor built ({K.nbytes/1e6:.0f} MB)", flush=True)

    def convs(gm):
        gh = np.fft.fft(np.pad(gm, ((0, 0), (0, n_fft - nf))), axis=1)
        prod = gh[:, None, :] * gh[None, :, :]
        return np.fft.ifft(prod, axis=2)[:, :, freq_sl]

    CL = convs(gl_m)
    CG = convs(gg_m)

    # --- wiring gate: kernel on exactly-mode-diagonal G_test ------------
    G_test_l = np.einsum("sw,is,js->wij", gl_m, P, P, optimize=True)
    G_test_g = np.einsum("sw,is,js->wij", gg_m, P, P, optimize=True)
    sl_t_b, sg_t_b = compute_phph_self_energy_finite_multi_slab(
        g_blocks_from_dense(G_test_l, n_slabs, b, g_cut),
        g_blocks_from_dense(G_test_g, n_slabs, b, g_cut),
        phi_blocks, n_slabs, w_full, dw,
        sigma_cutoff=sig_cut, g_cutoff=g_cut, dc_handling="zero",
        n_threads=N_THREADS)
    sl_t = dense_from_blocks(sl_t_b, n_slabs, b, nf)
    ref_t = np.einsum("is,wij,js->sw", P, sl_t, P, optimize=True)
    ch_t = pref * np.einsum("abc,bcw->aw", K, CL, optimize=True)
    gate = float(np.linalg.norm(ch_t - ref_t) / np.linalg.norm(ref_t))
    print(f"[channels] wiring gate (must be ~0): rel={gate:.3e}", flush=True)

    # --- closure: channel sum vs full-G reference -----------------------
    # Metric excludes the DC neighbourhood |omega| <= 2.5 dw (the masked
    # DC bin and the folded acoustic near-divergence around it dominate
    # the norm without entering any width at Omega_s > 0); the full-axis
    # variant is kept alongside.
    band_out = btd_mask(N, n_slabs, sig_cut)
    ref_l = np.einsum("is,wij,js->sw", P, sl_ref * band_out, P, optimize=True)
    ref_g = np.einsum("is,wij,js->sw", P, sg_ref * band_out, P, optimize=True)
    ch_l = pref * np.einsum("abc,bcw->aw", K, CL, optimize=True)
    ch_g = pref * np.einsum("abc,bcw->aw", K, CG, optimize=True)
    keep = np.abs(w_full) > 2.5 * dw
    clos_l = float(np.linalg.norm((ch_l - ref_l)[:, keep])
                   / np.linalg.norm(ref_l[:, keep]))
    clos_g = float(np.linalg.norm((ch_g - ref_g)[:, keep])
                   / np.linalg.norm(ref_g[:, keep]))
    clos_l_full = float(np.linalg.norm(ch_l - ref_l) / np.linalg.norm(ref_l))
    clos_g_full = float(np.linalg.norm(ch_g - ref_g) / np.linalg.norm(ref_g))
    print(f"[channels] closure (mode-diagonal, |w|>2.5dw): "
          f"lesser {clos_l:.3%} greater {clos_g:.3%} "
          f"(full-axis {clos_l_full:.3%}/{clos_g_full:.3%})", flush=True)

    # --- partial widths at the receiving mode frequencies ---------------
    skew = np.imag(fp_sign * pref * 0.5 * (CL - CG))      # (nm, nm, nf)
    skew_pos = skew[:, :, mid:]
    ref_skew = np.imag(fp_sign * 0.5 * (ref_l - ref_g))[:, mid:]
    Omega = spec["Omega_harm"]
    Gch = np.zeros((nm, nm, nm))
    Gref_at = np.zeros(nm)
    wpos = w
    for spp in range(nm):
        Om = Omega[spp]
        if Om < wpos[1]:
            continue
        i = min(int(np.searchsorted(wpos, Om)) - 1, len(wpos) - 2)
        t = (Om - wpos[i]) / (wpos[i + 1] - wpos[i])
        sk = (1 - t) * skew_pos[:, :, i] + t * skew_pos[:, :, i + 1]
        Gch[spp] = K[spp] * (-sk) / (2.0 * Om)
        Gref_at[spp] = -((1 - t) * ref_skew[spp, i]
                         + t * ref_skew[spp, i + 1]) / (2.0 * Om)
    row_sum = Gch.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        row_norm_ref = row_sum / Gref_at
        row_norm_snap = row_sum / spec["Gamma_anh"]
    print(f"[channels] row sums / reference width: "
          f"median {np.nanmedian(row_norm_ref):.3f}, "
          f"/ snapshot anh width: median {np.nanmedian(row_norm_snap):.3f}",
        flush=True)

    return dict(
        w_full=w_full, fp_sign=fp_sign,
        fp_err_lesser=fp_err,
        fp_err_greater=min(fp["greater_p"], fp["greater_m"]),
        wiring_gate=gate, closure_lesser=clos_l, closure_greater=clos_g,
        closure_lesser_full=clos_l_full, closure_greater_full=clos_g_full,
        g_modes_lesser=gl_m.astype(np.complex64),
        g_modes_greater=gg_m.astype(np.complex64),
        Gch=Gch, Gamma_ref_at=Gref_at,
        row_sum=row_sum, row_norm_ref=row_norm_ref,
        row_norm_snap=row_norm_snap,
        ch_skew_modes=np.ascontiguousarray(
            (pref * 0.5 * (CL - CG)).imag[:, :, mid:]).astype(np.float32),
        ref_sigR_im_modes=ref_skew.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Stage 4: link-gain matrix + spectral radius
# ---------------------------------------------------------------------------


def stage_gain(state, w, dw, spec, ch):
    Omega = spec["Omega_harm"]
    Gam_tot = spec["Gamma_tot"]
    Gch = ch["Gch"]
    nm = len(Omega)
    valid = (Omega > 0.15) & np.isfinite(Gam_tot) & (Gam_tot > 0)

    # link gain: sensitivity of every receiving mode s'' to source mode s
    S = Gch.sum(axis=2) + Gch.sum(axis=1)     # (s'', s) two-leg sensitivity
    with np.errstate(divide="ignore", invalid="ignore"):
        M_phys = (Omega[:, None] * S) / (Omega[None, :] * Gam_tot[None, :])
    M_phys[~np.isfinite(M_phys)] = 0.0
    enh = np.maximum(1.0, dw / np.where(Gam_tot > 0, Gam_tot, np.inf))
    M_grid = M_phys * enh[None, :]

    def rho(M, sel):
        idx = np.where(sel)[0]
        if len(idx) == 0:
            return 0.0, np.zeros(0, complex)
        ev = np.linalg.eigvals(M[np.ix_(idx, idx)])
        return float(np.max(np.abs(ev))), ev

    sharp = valid & (Gam_tot < dw)
    ir = valid & (Omega <= 2.0)
    out = dict(M_phys=M_phys, M_grid=M_grid, enh=enh,
               sel_valid=valid, sel_sharp=sharp, sel_ir=ir)
    for name, sel in (("all", valid), ("sharp", sharp), ("ir", ir)):
        for br, M in (("phys", M_phys), ("grid", M_grid)):
            r, ev = rho(M, sel)
            out[f"rho_{br}_{name}"] = r
            out[f"eigs_{br}_{name}"] = ev
            print(f"[gain] {state}: rho({br},{name} n={sel.sum()}) = {r:.3f}",
                  flush=True)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_state(state, stages, tag=""):
    info = STATES[state]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== {state}: config={info['config']}", flush=True)
    cfg, scba, ph, sse = build_engine(info["config"])
    N = int(np.sum(np.asarray(ph.block_sizes)))
    n_slabs = info["n_slabs"]
    assert N // n_slabs * n_slabs == N

    snap_flat = load_state(scba, info["snapshot"])
    snap = {}
    for k in ("sigma_lesser", "sigma_greater", "sigma_retarded"):
        buf = {"sigma_lesser": scba.data.sigma_lesser,
               "sigma_greater": scba.data.sigma_greater,
               "sigma_retarded": scba.data.sigma_retarded_hermitian}[k]
        snap[k] = densify(buf, N)

    w = np.abs(np.asarray(scba.energies).real).ravel()
    dw = float(w[1] - w[0])

    # dense device dynamical matrix + harmonic modes
    D = densify(ph.dynamical_matrix, N)[0]
    D = 0.5 * (D + D.conj().T).real
    evals2, evecs = np.linalg.eigh(D)
    _EVECS_CACHE["evecs"] = evecs

    Gr, Gl, Gg = solve_G(scba, ph, N)
    obc_R = np.zeros((len(w), N, N), complex)
    b = N // n_slabs
    if ph.obc_blocks.retarded[0] is not None:
        obc_R[:, :b, :b] = np.asarray(ph.obc_blocks.retarded[0])
    if ph.obc_blocks.retarded[-1] is not None:
        obc_R[:, -b:, -b:] = np.asarray(ph.obc_blocks.retarded[-1])

    results = dict(state=state, w=w, dw=dw, N=N, n_slabs=n_slabs,
                   evals2=evals2, evecs=evecs,
                   measured_lambda=np.array(info["measured_lambda"]),
                   g_cut=info["g_cut"], sig_cut=info["sig_cut"])

    spec = stage_spectra(state, w, dw, evals2, evecs, Gr,
                         snap["sigma_retarded"], obc_R, n_slabs)
    results.update({f"spec_{k}": v for k, v in spec.items()})

    if "kicks" in stages:
        kicks = stage_kicks(scba, ph, N, w, dw, spec,
                            snap["sigma_retarded"], spec["A_modes"])
        results.update(kicks)

    if "channels" in stages or "gain" in stages:
        ch = stage_channels(state, info, w, dw, evecs, Gl, Gg,
                            sse.phi_blocks, snap, spec, n_slabs)
        results.update({f"ch_{k}": v for k, v in ch.items()
                        if k not in ("w_full",)})
        if "gain" in stages:
            gain = stage_gain(state, w, dw, spec, ch)
            results.update({f"gain_{k}": v for k, v in gain.items()})

    out_npz = OUT / f"{state}{tag}.npz"
    np.savez_compressed(out_npz, **results)
    print(f"SAVED {out_npz}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True, choices=list(STATES))
    ap.add_argument("--stages", default="spectra,kicks,channels,gain")
    ap.add_argument("--tag", default="", help="output filename suffix")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    run_state(args.state, stages, tag=args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
