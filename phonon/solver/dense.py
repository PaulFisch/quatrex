"""Dense reference NEGF/SSE solver for anharmonic phonon transport.

Public entry points:

  * :func:`transmission_finite` — Γ-only one-shot or SCBA driver
    (formerly ``phonon_inputs.anharmonic.anharmonic_transmission_finite``).
  * :func:`transmission_q` — q-resolved SCBA driver
    (formerly ``phonon_inputs.anharmonic.anharmonic_transmission_q``).
  * :func:`scba_loop` — the shared SCBA fixed-point loop (callable for
    custom drivers that supply their own self-energy kernel and OBC).
  * :func:`gamma_project_M_blocks` — direct Γ-only supercell→primitive
    projection of the FC3 vertex.
  * :func:`compare_q11_to_finite` — regression check ensuring
    ``transmission_q(q_mesh=(1,1))`` reproduces :func:`transmission_finite`.

Implements the self-consistent Born approximation for 3-phonon
scattering following Guo *et al.*, Phys. Rev. B **102**, 195412 (2020).
The bubble integrand (Guo Eq. 8, diagonal block) is

    Σ^{<}(ω) = (i ℏ / 2) Σ_{c,d,e,f} Φ_{a c d}
        ∫ dω'/(2π) G^<_{cf}(ω') G^<_{de}(ω−ω') Φ_{b e f}.

Internal units: THz² for dynamical matrices and self-energies.
Self-energies always carry a q-axis (shape
``(n_slabs, n_kpts, nfreq, n_dof, n_dof)``), even when ``n_kpts == 1``.

Σ^R reconstruction
------------------
``retarded`` controls how Σ^R is built from Σ^{<,>}:
  - ``"half"`` — Σ^R = ½ (Σ^> − Σ^<) (no Kramers–Kronig).
  - ``"pv"``  — singularity-subtracted principal-value integral (O(nfreq²)).
  - ``"fft"`` — FFT Hilbert transform (O(nfreq log nfreq)).
Both entry points default to ``"half"`` so q=(1,1) reproduces finite.
"""

from __future__ import annotations

import warnings

import numpy as np

from phonon_inputs.constants import CONVERSION_THZ2, HBAR_SI, THZ_TO_RAD

from .diagnostics import (
    check_broadening_sign,
    check_full_axis_symmetry,
    symmetrize_lesser_greater,
)
from .grids import bose_full_axis, build_frequency_grid
from .leads import (
    ballistic_transmission_z2,
    build_device_hamiltonian,
    compute_obc_batch,
    solve_green_batch,
)
from .fc3_device import build_device_fc3_blocks
from .retarded import build_retarded
from .se_finite import (
    compute_phph_self_energy_finite,
    compute_phph_self_energy_finite_multi_slab,
)
from .se_q import compute_phph_self_energy_q_dense
from .zero_modes import build_translation_projector, project_self_energy


# ---------------------------------------------------------------------------
# FC3 loading
# ---------------------------------------------------------------------------


def load_fc3_raw(fc3_source):
    """Load raw FC3 array from various sources.

    Parameters
    ----------
    fc3_source : str, Path, ndarray, or dict
        - str/Path: HDF5 file with an ``"fc3"`` dataset.
        - ndarray:  raw FC3 passed directly.
        - dict:     must contain ``"fc3"`` (raw array) or ``"ph3"``
          (phono3py object whose ``.fc3`` attribute is the raw array).
    """
    if isinstance(fc3_source, np.ndarray):
        return fc3_source
    if isinstance(fc3_source, dict):
        if "fc3" in fc3_source:
            return np.asarray(fc3_source["fc3"])
        if "ph3" in fc3_source:
            return np.asarray(fc3_source["ph3"].fc3)
        raise ValueError(
            "fc3 dict has neither 'fc3' nor 'ph3' key. "
            "Pass the fc3.hdf5 path or the raw fc3 array instead. "
            f"Available keys: {list(fc3_source.keys())}"
        )
    import h5py
    with h5py.File(str(fc3_source), "r") as f:
        return np.array(f["fc3"])


# ---------------------------------------------------------------------------
# Stabilized Anderson mixing
# ---------------------------------------------------------------------------


def _anderson_mix(x_in, x_out, x_hist, f_hist, prev_fnorm, *,
                  depth, beta, reg=1e-3, step_cap=2.0):
    """Stabilized Anderson (Pulay/DIIS) mixing step.

    The plain Anderson scheme diverges on strongly anharmonic systems
    where the lowest-order self-energy is comparable to the device
    Hamiltonian (the d5a SiNW has ``|Sigma| ~ |H|``): the residual
    oscillates, stale history pollutes the least-squares, and an
    ill-conditioned normal-equation system produces an over-shooting
    step. This variant adds three standard safeguards:

      1. **restart** -- the history is cleared whenever the residual
         norm grows, so only a locally contracting stretch of iterates
         feeds the extrapolation;
      2. **regularization** -- the normal equations are damped at a
         relative ``reg`` level (1e-3, vs the 1e-8 of the bare scheme);
      3. **step cap** -- an Anderson step that overshoots the damped
         linear step by more than ``step_cap`` is rejected in favour of
         that linear step.

    On the d5a SiNW this converges the SCBA in ~47 iterations where the
    bare Anderson diverges (heat-flow conservation blows up to 100 %)
    and linear mixing needs ~65. ``x_hist`` / ``f_hist`` are mutated in
    place. Returns ``(x_mixed, fnorm)``; feed ``fnorm`` back as
    ``prev_fnorm`` on the next call.
    """
    f_k = x_out - x_in
    fnorm = float(np.linalg.norm(f_k))
    if fnorm > prev_fnorm:
        x_hist.clear()
        f_hist.clear()
    x_hist.append(x_in)
    f_hist.append(f_k)
    x_lin = x_in + beta * f_k
    m = len(f_hist)
    if m >= 2:
        n_use = min(m - 1, depth)
        dF = np.column_stack([
            f_hist[-n_use + j] - f_hist[-n_use + j - 1]
            for j in range(n_use)])
        dX = np.column_stack([
            x_hist[-n_use + j] - x_hist[-n_use + j - 1]
            for j in range(n_use)])
        FtF = dF.conj().T @ dF
        reg_abs = reg * np.trace(FtF).real / max(FtF.shape[0], 1)
        try:
            gamma = np.linalg.solve(
                FtF + reg_abs * np.eye(FtF.shape[0]), dF.conj().T @ f_k)
            x_and = (x_in + beta * f_k) - (dX + beta * dF) @ gamma
        except np.linalg.LinAlgError:
            x_and = x_lin
        lin_step = np.linalg.norm(x_lin - x_in) + 1e-300
        x_mixed = (x_lin
                   if np.linalg.norm(x_and - x_in) > step_cap * lin_step
                   else x_and)
    else:
        x_mixed = x_lin
    if len(x_hist) > depth + 2:
        x_hist.pop(0)
        f_hist.pop(0)
    return x_mixed, fnorm


# ---------------------------------------------------------------------------
# Safeguarded Anderson acceleration
# ---------------------------------------------------------------------------


def _block_weight_vector(sig_l, sig_g, n_slabs, n_dof, *, floor_rel=1e-6):
    """Per-entry weights placing every (I, J) slab-pair block on an
    equal footing in the safeguarded Anderson least-squares.

    The fixed-point vector ``[Sigma^<, Sigma^>]`` is dominated in
    2-norm by the large diagonal blocks; the small far-off-diagonal
    blocks are then invisible to both the secant least-squares and the
    restart test. Weighting each block by ``1/sqrt(||block|| + floor)``
    rescales them to comparable size. Computed once from the first
    bubble output and held fixed.

    Returns a real vector of length ``sig_l.size + sig_g.size`` (the
    weights are shared between the ``Sigma^<`` and ``Sigma^>`` halves
    of the fixed-point vector).
    """
    n_kpts, nfreq, N_D, _ = sig_l.shape
    mag = np.abs(sig_l) + np.abs(sig_g)
    w = np.ones((n_kpts, nfreq, N_D, N_D))
    block_norms = []
    for I in range(n_slabs):
        sI = slice(I * n_dof, (I + 1) * n_dof)
        for J in range(n_slabs):
            sJ = slice(J * n_dof, (J + 1) * n_dof)
            block_norms.append(float(np.linalg.norm(mag[:, :, sI, sJ])))
    scale = max(block_norms) if block_norms else 1.0
    floor = floor_rel * scale + 1e-300
    for I in range(n_slabs):
        sI = slice(I * n_dof, (I + 1) * n_dof)
        for J in range(n_slabs):
            sJ = slice(J * n_dof, (J + 1) * n_dof)
            bn = float(np.linalg.norm(mag[:, :, sI, sJ]))
            w[:, :, sI, sJ] = 1.0 / np.sqrt(bn + floor)
    w_flat = w.ravel()
    return np.concatenate([w_flat, w_flat])


class _AndersonAccelerator:
    """Safeguarded Anderson (Pulay/DIIS) acceleration.

    Fixes the failure mode of :func:`_anderson_mix`, which cleared its
    entire history on *any* residual uptick: on a noisy residual (the
    d5a multi-slab SCBA near 1e-4) that collapses Anderson to a bare
    linear step every iteration, which then diverges. The safeguards
    here follow the literature-standard recipe (Walker & Ni, SIAM JNA
    49 (2011); Toth & Kelley, SIAM JNA 53 (2015)):

      * history is *kept* across residual upticks -- Anderson is
        allowed to be non-monotone;
      * the least-squares is solved by a truncated SVD (``lstsq`` with
        ``rcond``), filtering ill-conditioned / linearly-dependent
        secant directions instead of discarding the whole history;
      * an over-long extrapolation step is replaced by the damped
        linear step *for that iteration only*;
      * the history is restarted only on genuine stagnation -- after
        ``stagnation`` consecutive iterations with no new best
        residual;
      * the residual is measured in a block-weighted norm so every
        (I, J) self-energy block is resolved.

    ``step(x_in, x_out)`` returns ``(x_mixed, fnorm)``.
    """

    def __init__(self, depth, beta, *, weights=None, cond_max=1e8,
                 step_cap=2.0, stagnation=5):
        self.depth = int(depth)
        self.beta = float(beta)
        self.weights = weights
        self.cond_max = float(cond_max)
        self.step_cap = float(step_cap)
        self.stagnation = int(stagnation)
        self.x_hist: list = []
        self.f_hist: list = []
        self.best_fnorm = float("inf")
        self.n_since_best = 0

    def fnorm(self, f):
        if self.weights is None:
            return float(np.linalg.norm(f))
        return float(np.linalg.norm(self.weights * f))

    def _truncate(self):
        while len(self.x_hist) > self.depth + 2:
            self.x_hist.pop(0)
            self.f_hist.pop(0)

    def step(self, x_in, x_out):
        f_k = x_out - x_in
        fnorm = self.fnorm(f_k)

        if fnorm < self.best_fnorm:
            self.best_fnorm = fnorm
            self.n_since_best = 0
        else:
            self.n_since_best += 1
        # Stagnation-only restart: NOT triggered by a single uptick.
        if self.n_since_best >= self.stagnation:
            self.x_hist.clear()
            self.f_hist.clear()
            self.n_since_best = 0

        self.x_hist.append(x_in)
        self.f_hist.append(f_k)

        x_lin = x_in + self.beta * f_k
        m = len(self.f_hist)
        if m < 2:
            return x_lin, fnorm

        n_use = min(m - 1, self.depth)
        dF = np.column_stack([
            self.f_hist[-n_use + j] - self.f_hist[-n_use + j - 1]
            for j in range(n_use)])
        dX = np.column_stack([
            self.x_hist[-n_use + j] - self.x_hist[-n_use + j - 1]
            for j in range(n_use)])

        if self.weights is None:
            dF_w, f_w = dF, f_k
        else:
            dF_w = dF * self.weights[:, None]
            f_w = f_k * self.weights

        try:
            gamma, *_ = np.linalg.lstsq(dF_w, f_w, rcond=1.0 / self.cond_max)
        except np.linalg.LinAlgError:
            self._truncate()
            return x_lin, fnorm

        x_and = x_lin - (dX + self.beta * dF) @ gamma

        lin_step = np.linalg.norm(x_lin - x_in) + 1e-300
        x_mixed = (x_lin
                   if np.linalg.norm(x_and - x_in) > self.step_cap * lin_step
                   else x_and)
        self._truncate()
        return x_mixed, fnorm


def _run_jfnk(x0_l, x0_g, scba_step, scba_tol, max_iter, verbose):
    """Jacobian-free Newton-Krylov solve of the SCBA fixed point.

    Wraps ``residual(x) = scba_step(x) - x`` and hands it to
    :func:`scipy.optimize.newton_krylov`. Robust even when plain
    fixed-point iteration is linearly unstable. Complex self-energies
    are packed into a real vector for the solve.

    Returns ``(sig_l, sig_g, residual)``; ``residual`` is ``None`` when
    SciPy is unavailable.
    """
    try:
        from scipy.optimize import newton_krylov
    except ImportError:
        if verbose:
            print("    JFNK requested but SciPy is unavailable; skipping")
        return x0_l, x0_g, None

    shape = x0_l.shape
    sz = x0_l.size

    def _unpack(xc):
        return xc[:sz].reshape(shape), xc[sz:].reshape(shape)

    def residual_real(xr):
        xc = np.ascontiguousarray(xr).view(np.complex128)
        sl, sg = _unpack(xc)
        sl_n, sg_n, _ = scba_step(sl, sg)
        res = np.concatenate([sl_n.ravel(), sg_n.ravel()]) - xc
        return np.ascontiguousarray(res).view(np.float64)

    x0c = np.concatenate([x0_l.ravel(), x0_g.ravel()])
    x0r = np.ascontiguousarray(x0c).view(np.float64)
    f_tol = max(scba_tol * (float(np.linalg.norm(x0c)) + 1e-300), 1e-300)
    newton_iter = min(int(max_iter), 40)

    try:
        sol_r = newton_krylov(residual_real, x0r, f_tol=f_tol,
                              maxiter=newton_iter, verbose=bool(verbose))
        sol_c = np.ascontiguousarray(sol_r).view(np.complex128)
    except Exception as exc:  # noqa: BLE001 - scipy NoConvergence
        if exc.__class__.__name__ != "NoConvergence" or not exc.args:
            raise
        sol_c = np.ascontiguousarray(
            np.asarray(exc.args[0]).ravel()).view(np.complex128)
        if verbose:
            print("    JFNK: newton_krylov did not meet f_tol; "
                  "using best iterate")

    sl, sg = _unpack(sol_c)
    fr = residual_real(np.ascontiguousarray(
        np.concatenate([sl.ravel(), sg.ravel()])).view(np.float64))
    fc = np.ascontiguousarray(fr).view(np.complex128)
    resid = float(np.linalg.norm(fc) / (np.linalg.norm(sol_c) + 1e-300))
    return sl, sg, resid


# ---------------------------------------------------------------------------
# SCBA loop
# ---------------------------------------------------------------------------


def scba_loop(
    *,
    z2_arr, freqs_thz, dw_thz, omega_rad, pos_mask,
    n_slabs, n_dof, N_D,
    H_D_list, obc_list, btd_blocks_list,
    n_kpts,
    se_kernel,
    T_L, T_R,
    max_scba_iter, scba_tol, conservation_tol,
    mixing, anderson_mixing, anderson_depth,
    scattering_contacts,
    retarded,
    verbose,
):
    """Shared SCBA loop for both q-dense and finite paths.

    Self-energies always have shape
    ``(n_slabs, n_kpts, nfreq, n_dof, n_dof)``, even when ``n_kpts == 1``.

    Parameters
    ----------
    se_kernel : callable(G_lesser_slab, G_greater_slab)
        Must return ``(Σ_l, Σ_g)``, each of shape
        ``(n_slabs, n_kpts, nfreq, n_dof, n_dof)``.
    retarded : {"pv", "fft", "half"}
        Method for rebuilding Σ^R from the mixed Σ^{<,>}.
    """
    nfreq = len(freqs_thz)

    Sigma_R = np.zeros((n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
    Sigma_l = np.zeros_like(Sigma_R)
    Sigma_g = np.zeros_like(Sigma_R)

    convergence_history = []
    J_total_prev = 0.0
    rel_change = float('inf')
    sig_change = float('inf')
    conservation_err = 1.0
    _anderson_x_hist = []
    _anderson_f_hist = []
    _anderson_prev_fnorm = float('inf')

    spectral_J_L = np.zeros(nfreq)
    spectral_J_R = np.zeros(nfreq)

    sl0 = slice(0, n_dof)
    sl_last = slice((n_slabs - 1) * n_dof, n_slabs * n_dof)

    for scba_iter in range(max_scba_iter):
        if scattering_contacts and scba_iter > 0:
            for iq, (H_00_iq, H_01_iq) in enumerate(btd_blocks_list):
                lead_L = Sigma_R[0, iq]
                lead_R = Sigma_R[-1, iq]
                try:
                    obc_try = compute_obc_batch(
                        z2_arr, H_00_iq, H_01_iq, freqs_thz, T_L, T_R,
                        n_slabs=n_slabs,
                        lead_sigma_r_L=lead_L, lead_sigma_r_R=lead_R)
                    if not np.any(np.isnan(obc_try["Sigma_L_R"])):
                        obc_list[iq] = obc_try
                    elif verbose:
                        print(f"    WARNING: scattering-contact OBC "
                              f"diverged for iq={iq}, keeping previous")
                except RuntimeError:
                    if verbose:
                        print(f"    WARNING: scattering-contact OBC "
                              f"failed for iq={iq}, keeping previous")

        G_lesser_slab = np.zeros(
            (n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
        G_greater_slab = np.zeros_like(G_lesser_slab)
        G_R_slab = np.zeros_like(G_lesser_slab)
        spectral_J_L[:] = 0.0
        spectral_J_R[:] = 0.0

        for iq in range(n_kpts):
            H_D = H_D_list[iq]
            obc = obc_list[iq]

            Sig_R_dev = np.zeros((nfreq, N_D, N_D), dtype=complex)
            Sig_l_dev = np.zeros_like(Sig_R_dev)
            Sig_g_dev = np.zeros_like(Sig_R_dev)
            for l in range(n_slabs):
                sl = slice(l * n_dof, (l + 1) * n_dof)
                Sig_R_dev[:, sl, sl] = Sigma_R[l, iq]
                Sig_l_dev[:, sl, sl] = Sigma_l[l, iq]
                Sig_g_dev[:, sl, sl] = Sigma_g[l, iq]

            G_ret, G_less, G_great = solve_green_batch(
                z2_arr, H_D, obc, Sig_R_dev, Sig_l_dev, Sig_g_dev)

            for l in range(n_slabs):
                sl = slice(l * n_dof, (l + 1) * n_dof)
                G_lesser_slab[l, iq] = G_less[:, sl, sl]
                G_greater_slab[l, iq] = G_great[:, sl, sl]
                G_R_slab[l, iq] = G_ret[:, sl, sl]

            SLg_Gl = obc["Sigma_L_greater"][:, sl0, sl0] @ G_less[:, sl0, sl0]
            SLl_Gg = obc["Sigma_L_lesser"][:, sl0, sl0] @ G_great[:, sl0, sl0]
            spectral_J_L += HBAR_SI * omega_rad * np.real(
                np.trace(SLg_Gl - SLl_Gg, axis1=-2, axis2=-1))

            SRl_Gg = obc["Sigma_R_lesser"][:, sl_last, sl_last] @ G_great[:, sl_last, sl_last]
            SRg_Gl = obc["Sigma_R_greater"][:, sl_last, sl_last] @ G_less[:, sl_last, sl_last]
            spectral_J_R += HBAR_SI * omega_rad * np.real(
                np.trace(SRl_Gg - SRg_Gl, axis1=-2, axis2=-1))

        spectral_J_L /= n_kpts
        spectral_J_R /= n_kpts

        J_L_total = np.sum(spectral_J_L[pos_mask]) * dw_thz * 1e12
        J_R_total = np.sum(spectral_J_R[pos_mask]) * dw_thz * 1e12
        J_total = 0.5 * (J_L_total + J_R_total)
        J_denom = abs(J_L_total) + abs(J_R_total)
        conservation_err = (abs(J_L_total - J_R_total) / J_denom
                            if J_denom > 0 else 0.0)

        if verbose and scba_iter <= 2:
            max_l_err = 0.0
            max_r_err = 0.0
            for l in range(n_slabs):
                for iq in range(n_kpts):
                    le, re = check_full_axis_symmetry(
                        G_R_slab[l, iq], G_lesser_slab[l, iq],
                        G_greater_slab[l, iq], freqs_thz)
                    max_l_err = max(max_l_err, le)
                    max_r_err = max(max_r_err, re)
            if max_l_err > 1e-3 or max_r_err > 1e-3:
                print(f"    Symmetry: |G^<+G^>(-)|={max_l_err:.2e}, "
                      f"|G^R-G^R(-)*|={max_r_err:.2e}")

        Sigma_l_new, Sigma_g_new = se_kernel(
            G_lesser_slab, G_greater_slab)

        symmetrize_lesser_greater(Sigma_l_new, Sigma_g_new)

        sig_r_norm = np.max(np.abs(Sigma_l_new)) + np.max(np.abs(Sigma_g_new))

        if verbose and scba_iter == 0:
            gl_max = np.max(np.abs(G_lesser_slab))
            print(f"    G diagnostic: max|G^<| = {gl_max:.4e}")
            print(f"    Self-energy: max|Σ^R| = {sig_r_norm:.4e} THz²")

        _Sigma_prev = np.concatenate([
            Sigma_l.ravel(), Sigma_g.ravel()])

        if scba_iter == 0:
            Sigma_l = Sigma_l_new.copy()
            Sigma_g = Sigma_g_new.copy()
        elif anderson_mixing:
            x_in = np.concatenate([Sigma_l.ravel(), Sigma_g.ravel()])
            x_out = np.concatenate([
                Sigma_l_new.ravel(), Sigma_g_new.ravel()])
            x_mixed, _anderson_prev_fnorm = _anderson_mix(
                x_in, x_out, _anderson_x_hist, _anderson_f_hist,
                _anderson_prev_fnorm, depth=anderson_depth, beta=mixing)
            sz = Sigma_l.size
            Sigma_l = x_mixed[:sz].reshape(Sigma_l.shape)
            Sigma_g = x_mixed[sz:].reshape(Sigma_g.shape)
        else:
            alpha = mixing
            Sigma_l = (1 - alpha) * Sigma_l + alpha * Sigma_l_new
            Sigma_g = (1 - alpha) * Sigma_g + alpha * Sigma_g_new

        Sigma_R = build_retarded(
            Sigma_l, Sigma_g, freqs_thz, method=retarded)

        total_viol = 0
        total_max = 0.0
        for l in range(n_slabs):
            for iq in range(n_kpts):
                nv, mv = check_broadening_sign(
                    Sigma_R[l, iq], freqs_thz, "SCBA", tol=1e-8)
                total_viol += nv
                total_max = max(total_max, mv)
        if total_viol > 0 and verbose:
            print(f"    WARNING: Γ sign violations: {total_viol} points, "
                  f"max = {total_max:.2e}")

        if scba_iter > 0:
            rel_change = abs(J_total - J_total_prev) / (abs(J_total_prev) + 1e-30)
            _Sigma_now = np.concatenate([
                Sigma_l.ravel(), Sigma_g.ravel()])
            sig_norm = np.linalg.norm(_Sigma_now)
            sig_change = (np.linalg.norm(_Sigma_now - _Sigma_prev)
                          / (sig_norm + 1e-30) if sig_norm > 0 else 0.0)
            convergence_history.append(rel_change)
            if verbose:
                print(f"    SCBA iter {scba_iter + 1}: "
                      f"J = {J_total:.4e} W, "
                      f"conservation = {conservation_err:.4e}, "
                      f"dJ/J = {rel_change:.4e}, "
                      f"dΣ/Σ = {sig_change:.4e}, "
                      f"max|Σ^R| = {np.max(np.abs(Sigma_R)):.2e} THz²")
            numerically_converged = (sig_change < scba_tol
                                      and rel_change < scba_tol)
            conserving = conservation_err < conservation_tol

            if numerically_converged and conserving:
                if verbose:
                    print(f"    Converged after {scba_iter + 1} iterations "
                          f"(dJ/J={rel_change:.2e}, dΣ/Σ={sig_change:.2e}, "
                          f"conservation={conservation_err:.2e})")
                break

            if numerically_converged and not conserving:
                if verbose:
                    print(f"    Numerically converged but conservation "
                          f"NOT satisfied ({conservation_err:.2e} > "
                          f"{conservation_tol:.2e}), continuing...")
        else:
            if verbose:
                print(f"    SCBA iter 1: "
                      f"J_L = {J_L_total:.4e} W, J_R = {J_R_total:.4e} W")

        J_total_prev = J_total

    else:
        if verbose:
            print(f"  WARNING: SCBA did not converge after {max_scba_iter} "
                  f"iterations (dJ/J={rel_change:.2e}, dΣ/Σ={sig_change:.2e}, "
                  f"conservation={conservation_err:.2e})")

    return {
        "spectral_J_L": spectral_J_L,
        "spectral_J_R": spectral_J_R,
        "Sigma_R": Sigma_R,
        "Sigma_l": Sigma_l,
        "Sigma_g": Sigma_g,
        "conservation_err": conservation_err,
        "convergence_history": convergence_history,
    }


# ---------------------------------------------------------------------------
# Device-storage SCBA loop for the multi-slab finite path
# ---------------------------------------------------------------------------


def scba_loop_dev(
    *,
    z2_arr, freqs_thz, dw_thz, omega_rad, pos_mask,
    n_slabs, n_dof, N_D,
    H_D_list, obc_list, btd_blocks_list,
    n_kpts,
    se_kernel,
    T_L, T_R,
    max_scba_iter, scba_tol, conservation_tol,
    mixing, anderson_mixing, anderson_depth,
    scattering_contacts,
    retarded,
    verbose,
    solver=None,
    anderson_safeguard=True,
    zero_mode_projection=True,
    gate_on_conservation=False,
    divergence_guard=True,
    divergence_factor=10.0,
    divergence_patience=4,
    masses_primitive=None,
):
    """SCBA fixed-point with device-storage self-energies.

    Mirrors :func:`scba_loop` but stores ``Sigma^{<,>,R}`` as
    ``(n_kpts, nfreq, N_D, N_D)`` device-matrix buffers so the
    multi-slab driver can populate off-diagonal blocks ``Sigma_{IJ}``
    produced by :func:`compute_phph_self_energy_finite_multi_slab`.

    The per-iteration map ``Sigma -> G -> Sigma`` is isolated in the
    nested ``_scba_step``; the outer driver applies one of several
    fixed-point accelerators to it.

    Parameters
    ----------
    se_kernel : callable(G_less_dev, G_great_dev)
        ``G_*_dev`` have shape ``(n_kpts, nfreq, N_D, N_D)``. Returns
        ``(Sigma_l, Sigma_g)`` of the same shape.
    retarded : {"pv", "fft", "half"}
        Method for rebuilding ``Sigma^R`` from the mixed
        ``Sigma^{<,>}`` pair.
    solver : {"linear", "anderson", "jfnk", "anderson+jfnk"}
        Fixed-point accelerator. ``"anderson+jfnk"`` runs safeguarded
        Anderson and, if it fails to converge, restarts a
        Jacobian-free Newton-Krylov solve from the best Anderson
        iterate.
    anderson_safeguard : bool
        ``True`` uses the safeguarded :class:`_AndersonAccelerator`;
        ``False`` keeps the legacy hard-restart :func:`_anderson_mix`.
    zero_mode_projection : bool
        Project ``Sigma^{<,>}`` onto the translation-free subspace
        every iteration (a discrete acoustic sum rule on the
        self-energy). Requires ``masses_primitive``.
    gate_on_conservation : bool
        ``True`` reproduces the legacy combined stop test (SCF residual
        AND heat-flow conservation). ``False`` (default) stops on the
        SCF residual alone and reports conservation as a diagnostic.
    divergence_guard : bool
        Abort early -- returning the best iterate -- once the residual
        exceeds ``divergence_factor`` x its running minimum for
        ``divergence_patience`` consecutive iterations.
    masses_primitive : (n_atoms,) array, optional
        Per-atom primitive-cell masses; required when
        ``zero_mode_projection`` is set.
    """
    if solver is None:
        solver = "anderson" if anderson_mixing else "linear"
    if solver not in ("linear", "anderson", "jfnk", "anderson+jfnk"):
        raise ValueError(
            f"Unknown solver={solver!r}. Use 'linear', 'anderson', "
            "'jfnk', or 'anderson+jfnk'.")

    nfreq = len(freqs_thz)
    sl0 = slice(0, n_dof)
    sl_last = slice((n_slabs - 1) * n_dof, n_slabs * n_dof)
    shape = (n_kpts, nfreq, N_D, N_D)

    Q_trans = None
    if zero_mode_projection:
        if masses_primitive is None:
            raise ValueError(
                "zero_mode_projection=True requires masses_primitive")
        n_atoms_prim = len(np.asarray(masses_primitive))
        if n_dof % n_atoms_prim != 0:
            raise ValueError(
                f"n_dof={n_dof} is not a multiple of "
                f"len(masses_primitive)={n_atoms_prim}")
        Q_trans = build_translation_projector(
            masses_primitive, n_slabs, n_cart=n_dof // n_atoms_prim)

    # Buffers reused by every _scba_step call.
    G_less_dev_q = np.zeros(shape, dtype=complex)
    G_great_dev_q = np.zeros_like(G_less_dev_q)
    G_ret_dev_q = np.zeros_like(G_less_dev_q)
    _spec_L = np.zeros(nfreq)
    _spec_R = np.zeros(nfreq)

    def _scba_step(sig_l, sig_g):
        """One SCBA pass: input ``Sigma^{<,>}`` -> bubble of the
        resulting ``G``. Pure map; ``sig_l``/``sig_g`` are not mutated.
        Returns ``(sig_l_new, sig_g_new, info)``.
        """
        Sig_R = build_retarded(sig_l, sig_g, freqs_thz, method=retarded)

        G_less_dev_q[:] = 0.0
        G_great_dev_q[:] = 0.0
        G_ret_dev_q[:] = 0.0
        _spec_L[:] = 0.0
        _spec_R[:] = 0.0

        for iq in range(n_kpts):
            G_ret, G_less, G_great = solve_green_batch(
                z2_arr, H_D_list[iq], obc_list[iq],
                Sig_R[iq], sig_l[iq], sig_g[iq])
            G_less_dev_q[iq] = G_less
            G_great_dev_q[iq] = G_great
            G_ret_dev_q[iq] = G_ret

            obc = obc_list[iq]
            SLg_Gl = obc["Sigma_L_greater"][:, sl0, sl0] @ G_less[:, sl0, sl0]
            SLl_Gg = obc["Sigma_L_lesser"][:, sl0, sl0] @ G_great[:, sl0, sl0]
            _spec_L[:] += HBAR_SI * omega_rad * np.real(
                np.trace(SLg_Gl - SLl_Gg, axis1=-2, axis2=-1))

            SRl_Gg = obc["Sigma_R_lesser"][:, sl_last, sl_last] @ G_great[:, sl_last, sl_last]
            SRg_Gl = obc["Sigma_R_greater"][:, sl_last, sl_last] @ G_less[:, sl_last, sl_last]
            _spec_R[:] += HBAR_SI * omega_rad * np.real(
                np.trace(SRl_Gg - SRg_Gl, axis1=-2, axis2=-1))

        spec_L = _spec_L / n_kpts
        spec_R = _spec_R / n_kpts
        J_L_total = np.sum(spec_L[pos_mask]) * dw_thz * 1e12
        J_R_total = np.sum(spec_R[pos_mask]) * dw_thz * 1e12
        J_total = 0.5 * (J_L_total + J_R_total)
        J_denom = abs(J_L_total) + abs(J_R_total)
        conservation_err = (abs(J_L_total - J_R_total) / J_denom
                            if J_denom > 0 else 0.0)

        sig_l_new, sig_g_new = se_kernel(G_less_dev_q, G_great_dev_q)
        symmetrize_lesser_greater(sig_l_new, sig_g_new)
        if Q_trans is not None:
            # Discrete acoustic sum rule: strip the rigid-translation
            # component so the bubble cannot renormalise the zero modes.
            sig_l_new = project_self_energy(sig_l_new, Q_trans)
            sig_g_new = project_self_energy(sig_g_new, Q_trans)

        info = {
            "Sigma_R": Sig_R,
            "spectral_J_L": spec_L,
            "spectral_J_R": spec_R,
            "J_L_total": J_L_total,
            "J_R_total": J_R_total,
            "J_total": J_total,
            "conservation_err": conservation_err,
        }
        return sig_l_new, sig_g_new, info

    def _update_scattering_obc(Sig_R):
        """Refresh the lead OBC self-energies from the current Sigma^R."""
        for iq, (H_00_iq, H_01_iq) in enumerate(btd_blocks_list):
            lead_L = Sig_R[iq, :, sl0, sl0]
            lead_R = Sig_R[iq, :, sl_last, sl_last]
            try:
                obc_try = compute_obc_batch(
                    z2_arr, H_00_iq, H_01_iq, freqs_thz, T_L, T_R,
                    n_slabs=n_slabs,
                    lead_sigma_r_L=lead_L, lead_sigma_r_R=lead_R)
                if not np.any(np.isnan(obc_try["Sigma_L_R"])):
                    obc_list[iq] = obc_try
                elif verbose:
                    print(f"    WARNING: scattering-contact OBC "
                          f"diverged for iq={iq}, keeping previous")
            except RuntimeError:
                if verbose:
                    print(f"    WARNING: scattering-contact OBC "
                          f"failed for iq={iq}, keeping previous")

    def _broadening_report(Sig_R):
        total_viol = 0
        total_max = 0.0
        for iq in range(n_kpts):
            for l in range(n_slabs):
                sl = slice(l * n_dof, (l + 1) * n_dof)
                nv, mv = check_broadening_sign(
                    Sig_R[iq, :, sl, sl], freqs_thz, "SCBA", tol=1e-8)
                total_viol += nv
                total_max = max(total_max, mv)
        return total_viol, total_max

    convergence_history = []

    # --- iteration 0: lowest-order Born (Sigma = 0 -> bare G -> bubble) -
    Sigma_l, Sigma_g, info = _scba_step(
        np.zeros(shape, dtype=complex), np.zeros(shape, dtype=complex))
    if verbose:
        gl_max = np.max(np.abs(G_less_dev_q))
        sig0 = np.max(np.abs(Sigma_l)) + np.max(np.abs(Sigma_g))
        print(f"    G diagnostic: max|G^<| = {gl_max:.4e}")
        print(f"    Self-energy: max|Sigma_new| = {sig0:.4e} THz^2")
        print(f"    SCBA iter 1: J_L = {info['J_L_total']:.4e} W, "
              f"J_R = {info['J_R_total']:.4e} W")

    use_anderson = solver in ("anderson", "anderson+jfnk")

    accel = None
    if use_anderson and anderson_safeguard:
        weights = _block_weight_vector(Sigma_l, Sigma_g, n_slabs, n_dof)
        accel = _AndersonAccelerator(anderson_depth, mixing, weights=weights)
    _ax_hist, _af_hist, _aprev = [], [], float("inf")

    best_resid = float("inf")
    best_l = Sigma_l.copy()
    best_g = Sigma_g.copy()
    converged = False

    if solver in ("linear", "anderson", "anderson+jfnk"):
        J_total_prev = info["J_total"]
        min_resid = float("inf")
        n_diverging = 0
        for scba_iter in range(1, max_scba_iter):
            if scattering_contacts:
                _update_scattering_obc(info["Sigma_R"])

            Sigma_l_out, Sigma_g_out, info = _scba_step(Sigma_l, Sigma_g)

            x_in = np.concatenate([Sigma_l.ravel(), Sigma_g.ravel()])
            x_out = np.concatenate([Sigma_l_out.ravel(), Sigma_g_out.ravel()])
            f = x_out - x_in
            resid = float(np.linalg.norm(f)
                          / (np.linalg.norm(x_in) + 1e-300))
            convergence_history.append(resid)

            if verbose and scba_iter <= 3:
                max_l_err = 0.0
                max_r_err = 0.0
                for iq in range(n_kpts):
                    for l in range(n_slabs):
                        sl = slice(l * n_dof, (l + 1) * n_dof)
                        le, re = check_full_axis_symmetry(
                            G_ret_dev_q[iq, :, sl, sl],
                            G_less_dev_q[iq, :, sl, sl],
                            G_great_dev_q[iq, :, sl, sl], freqs_thz)
                        max_l_err = max(max_l_err, le)
                        max_r_err = max(max_r_err, re)
                if max_l_err > 1e-3 or max_r_err > 1e-3:
                    print(f"    Symmetry: |G^<+G^>(-)|={max_l_err:.2e}, "
                          f"|G^R-G^R(-)*|={max_r_err:.2e}")

            if solver == "linear":
                x_mixed = x_in + mixing * f
            elif accel is not None:
                x_mixed, _ = accel.step(x_in, x_out)
            else:
                x_mixed, _aprev = _anderson_mix(
                    x_in, x_out, _ax_hist, _af_hist, _aprev,
                    depth=anderson_depth, beta=mixing)

            sz = Sigma_l.size
            Sigma_l = x_mixed[:sz].reshape(shape)
            Sigma_g = x_mixed[sz:].reshape(shape)

            J_total = info["J_total"]
            conservation_err = info["conservation_err"]
            rel_change = (abs(J_total - J_total_prev)
                          / (abs(J_total_prev) + 1e-30))
            J_total_prev = J_total

            if resid < best_resid:
                best_resid = resid
                best_l = Sigma_l.copy()
                best_g = Sigma_g.copy()

            if verbose:
                viol, vmax = _broadening_report(info["Sigma_R"])
                msg = (f"    SCBA iter {scba_iter + 1}: J = {J_total:.4e} W, "
                       f"conservation = {conservation_err:.4e}, "
                       f"resid = {resid:.4e}, dJ/J = {rel_change:.4e}, "
                       f"max|Sigma^R| = {np.max(np.abs(info['Sigma_R'])):.2e}")
                if viol > 0:
                    msg += f"  [Gamma sign viol: {viol} pts, max {vmax:.2e}]"
                print(msg)

            numerically_converged = resid < scba_tol
            conserving = conservation_err < conservation_tol
            if numerically_converged and (
                    conserving or not gate_on_conservation):
                converged = True
                if verbose:
                    print(f"    Converged after {scba_iter + 1} iterations "
                          f"(resid={resid:.2e}, dJ/J={rel_change:.2e}, "
                          f"conservation={conservation_err:.2e})")
                break
            if numerically_converged and gate_on_conservation:
                if verbose:
                    print(f"    Numerically converged but conservation NOT "
                          f"satisfied ({conservation_err:.2e} > "
                          f"{conservation_tol:.2e}), continuing...")

            min_resid = min(min_resid, resid)
            if divergence_guard and scba_iter > 2:
                if resid > divergence_factor * min_resid:
                    n_diverging += 1
                else:
                    n_diverging = 0
                if n_diverging >= divergence_patience:
                    if verbose:
                        print(f"    Divergence guard: resid {resid:.2e} "
                              f"exceeds {divergence_factor:g}x running min "
                              f"{min_resid:.2e}; aborting at iter "
                              f"{scba_iter + 1}")
                    break
        else:
            if verbose:
                print(f"  WARNING: SCBA did not converge after "
                      f"{max_scba_iter} iterations "
                      f"(best resid={best_resid:.2e})")
    else:
        # Pure JFNK: seed from the lowest-order Born iterate.
        best_l, best_g = Sigma_l.copy(), Sigma_g.copy()

    # --- JFNK (pure, or fallback when Anderson did not converge) --------
    if solver in ("jfnk", "anderson+jfnk") and not converged:
        if verbose:
            why = ("pure JFNK solve" if solver == "jfnk"
                   else f"Anderson stalled (best resid {best_resid:.2e}); "
                        f"JFNK fallback")
            print(f"    {why}")
        jl, jg, jresid = _run_jfnk(
            best_l, best_g, _scba_step, scba_tol, max_scba_iter, verbose)
        if jresid is not None:
            convergence_history.append(jresid)
            if jresid <= best_resid:
                best_l, best_g, best_resid = jl, jg, jresid
            converged = best_resid < scba_tol
            if verbose:
                print(f"    JFNK final resid = {jresid:.4e} "
                      f"({'converged' if converged else 'not converged'})")

    # --- consistent final evaluation for the returned self-energy ------
    Sigma_l, Sigma_g = best_l, best_g
    _, _, info = _scba_step(Sigma_l, Sigma_g)
    Sigma_R = info["Sigma_R"]

    return {
        "spectral_J_L": info["spectral_J_L"],
        "spectral_J_R": info["spectral_J_R"],
        "Sigma_R": Sigma_R,
        "Sigma_l": Sigma_l,
        "Sigma_g": Sigma_g,
        "conservation_err": info["conservation_err"],
        "convergence_history": convergence_history,
        "converged": converged,
        "scba_residual": best_resid,
    }


# ---------------------------------------------------------------------------
# Γ-only supercell→primitive projection of the FC3 vertex
# ---------------------------------------------------------------------------


def gamma_project_M_blocks(
    M_blocks: np.ndarray, prim_indices: np.ndarray, n_atoms: int,
) -> np.ndarray:
    """Project ``M_blocks`` onto the primitive cell at q=Γ.

    Equivalent to ``np.einsum('ci,aij,dj->acd', T0, M_blocks, T0.conj())``
    where ``T0 = build_gathering_matrix(..., q=(0, 0), ...)`` — at Γ the
    matrix is real, integer-valued (each column has exactly one nonzero
    equal to 1), so the einsum reduces to summing supercell-image
    rows/cols grouped by ``prim_indices``. We do that fold directly with
    two ``np.add.at`` reductions, avoiding the ``(n_dof, dim_sc)`` dense
    ``T0`` (~99 % zeros at Γ).
    """
    n_super = len(prim_indices)
    n_dof = 3 * n_atoms
    dim_sc = 3 * n_super
    assert M_blocks.shape == (n_dof, dim_sc, dim_sc)

    M_r = M_blocks.reshape(n_dof, n_super, 3, dim_sc)
    M_left = np.zeros((n_dof, n_atoms, 3, dim_sc), dtype=M_blocks.dtype)
    np.add.at(M_left, (slice(None), prim_indices, slice(None), slice(None)), M_r)

    M2 = M_left.reshape(n_dof, n_atoms, 3, n_super, 3)
    M_proj = np.zeros((n_dof, n_atoms, 3, n_atoms, 3), dtype=M_blocks.dtype)
    np.add.at(M_proj, (slice(None), slice(None), slice(None), prim_indices, slice(None)), M2)

    return M_proj.reshape(n_dof, n_dof, n_dof)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def transmission_finite(
    phonon,
    fc3_hdf5=None,
    freq_range_thz: tuple[float, float, int] = (0.01, 16.0, 101),
    transport_direction: str = "x",
    eta_factor: float = 1.0,
    temperature: float = 300.0,
    delta_T: float = 10.0,
    max_scba_iter: int = 80,
    scba_tol: float = 1e-3,
    conservation_tol: float = 1e-3,
    mixing: float = 0.3,
    anderson_mixing: bool = False,
    anderson_depth: int = 8,
    n_slabs: int = 1,
    verbose: bool = True,
    M_stacked_override: np.ndarray = None,
    retarded: str = "fft",
    scattering_contacts: bool = False,
    hilbert_retarded: bool = False,
    sigma_cutoff: int | None = None,
    vertex_cutoff: int | None = None,
    g_cutoff: int | None = None,
    dc_handling: str = "interpolate",
    solver: str | None = None,
    anderson_safeguard: bool = True,
    zero_mode_projection: bool = True,
    gate_on_conservation: bool = False,
    divergence_guard: bool = True,
) -> dict:
    """Reference anharmonic phonon transport (Gamma-point, finite device).

    Computes the full multi-slab 3-phonon self-energy on a finite
    ``n_slabs``-cell device. By default every approximation against
    the strict SCBA expression is disabled (see ``sigma_cutoff``,
    ``vertex_cutoff``, ``g_cutoff``, ``dc_handling`` below); the
    remaining approximation in the dense path is the non-self-consistent
    lead self-energy (controlled by ``scattering_contacts``).

    Parameters
    ----------
    retarded : {"fft", "pv", "half"}
        Method for reconstructing Sigma^R from the mixed Sigma^{<,>}.
        Default ``"fft"`` retains the level shift via the FFT-Hilbert
        transform; ``"half"`` reproduces the historical broadening-only
        approximation; ``"pv"`` does the singularity-subtracted PV
        integral (slowest).
    hilbert_retarded : bool
        Legacy flag.  If ``True``, overrides ``retarded`` to ``"fft"``.
    sigma_cutoff : int or None
        Maximum ``|I - J|`` for produced Sigma blocks. ``None``
        (default) imposes no truncation; pass ``0`` to recover the
        BTD-diagonal-only behaviour of the legacy solver.
    vertex_cutoff : int or None
        Maximum slab-distance retained in the FC3 vertex blocks
        ``Phi_{I, K, K'}``. ``None`` (default) keeps every triplet
        supported by the supercell FC3.
    g_cutoff : int or None
        Maximum ``|K - K'|`` for G blocks used in the inner bubble
        sum. ``None`` (default) keeps every available block; pass
        ``0`` to restrict to diagonal G.
    dc_handling : {"interpolate", "zero", "keep"}
        How to treat the omega = 0 sample of G before the bubble FFT
        (see :func:`phonon.solver.bubble.bubble_dense`). The legacy
        behaviour is ``"zero"``; the new default ``"interpolate"``
        replaces ``G[0]`` by the midpoint of its neighbours.
    solver : {None, "linear", "anderson", "jfnk", "anderson+jfnk"}
        Fixed-point accelerator for the SCBA loop. ``None`` (default)
        derives it from ``anderson_mixing`` for back-compatibility
        (``"anderson"`` if set, else ``"linear"``). ``"anderson+jfnk"``
        falls back to a Jacobian-free Newton-Krylov solve if Anderson
        stalls.
    anderson_safeguard : bool
        ``True`` (default) uses the safeguarded Anderson accelerator;
        ``False`` keeps the legacy hard-restart scheme.
    zero_mode_projection : bool
        ``True`` (default) projects the rigid-translation component out
        of the bubble self-energy each iteration (a discrete acoustic
        sum rule that removes the soft-mode instability).
    gate_on_conservation : bool
        ``True`` reproduces the legacy stop test (SCF residual AND
        heat-flow conservation); ``False`` (default) stops on the SCF
        residual alone and reports conservation as a diagnostic.
    divergence_guard : bool
        ``True`` (default) aborts early -- returning the best iterate --
        when the residual blows up.

    Notes
    -----
    The ``eta_factor`` default is ``1.0`` (``eta ~ d_omega``): the
    ``0.05`` of earlier revisions under-resolves the propagator on the
    frequency grid (see ``verify_discretization``). Recover the
    physical ``eta -> 0`` limit by extrapolating a few ``eta_factor``
    values rather than running at ``eta ~ 0``.
    """
    if hilbert_retarded:
        retarded = "fft"
    if solver is None:
        solver = "anderson" if anderson_mixing else "linear"
    from phonon_inputs.convention import get_btd_blocks
    from phonon_inputs.separable import (
        build_supercell_mapping,
        build_realspace_fc3_matrices,
    )

    freqs_thz, dw_thz, eta_w, z2_arr, pos_mask, mid = build_frequency_grid(
        freq_range_thz, eta_factor=eta_factor)
    nfreq = len(freqs_thz)

    n_atoms = len(phonon.primitive.masses)
    n_dof = 3 * n_atoms
    N_D = n_slabs * n_dof

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(
        phonon, transport_direction)
    masses_super = phonon.supercell.masses
    n_super = len(masses_super)
    dim_sc = n_super * 3

    if M_stacked_override is not None:
        M_stacked = M_stacked_override
        fc3_raw = None
    else:
        fc3_raw = load_fc3_raw(fc3_hdf5)
        M_stacked = build_realspace_fc3_matrices(
            fc3_raw, n_atoms, masses_super, ref_sc_atoms)

    # Build the device-resolved FC3 vertex dict (multi-slab).
    phi_dev_blocks = build_device_fc3_blocks(
        M_stacked, prim_indices, slab_indices,
        n_atoms, n_slabs, vertex_cutoff=vertex_cutoff,
    )

    if verbose:
        if M_stacked_override is None:
            print(f"  FC3 raw shape: {fc3_raw.shape}")
        print(f"  Supercell atoms: {n_super}, dim_sc: {dim_sc}")
        print(f"  M_stacked norm: {np.linalg.norm(M_stacked):.4e}")
        print(f"  Phi device blocks: {len(phi_dev_blocks)} "
              f"(vertex_cutoff={vertex_cutoff})")
        print(f"  Device: {n_slabs} slab(s), {N_D} DOFs (finite, Gamma only)")
        print(f"  Cutoffs: sigma={sigma_cutoff}, vertex={vertex_cutoff}, "
              f"g={g_cutoff}; dc_handling={dc_handling}")

    T_L = temperature + delta_T / 2.0
    T_R = temperature - delta_T / 2.0

    if verbose:
        print(f"  Frequency grid: {nfreq} points, "
              f"{freqs_thz[0]:.2f} to {freqs_thz[-1]:.2f} THz")
        print(f"  Temperature: {temperature} K, delta_T: {delta_T} K")
        print(f"  eta_w = {eta_w:.4e} THz  (eta/d_omega = "
              f"{eta_w / dw_thz:.2f})")
        sg = " safeguarded" if (anderson_safeguard
                                and solver in ("anderson", "anderson+jfnk")
                                ) else ""
        print(f"  SCBA: max {max_scba_iter} iter, tol={scba_tol}, "
              f"mix={mixing}, solver={solver}{sg}")
        print(f"  Options: zero_mode_projection={zero_mode_projection}, "
              f"gate_on_conservation={gate_on_conservation}, "
              f"divergence_guard={divergence_guard}")
        print(f"  Retarded SE: {retarded}")

    H_00, H_01 = get_btd_blocks(
        phonon, (0.0, 0.0), transport_direction=transport_direction,
        conversion_factor=CONVERSION_THZ2)
    H_D = build_device_hamiltonian(H_00, H_01, n_slabs)

    if verbose:
        suffix = " (will update each SCBA iter)" if scattering_contacts else ""
        print(f"  Precomputing OBC self-energies (batched)...{suffix}")
    obc = compute_obc_batch(z2_arr, H_00, H_01, freqs_thz, T_L, T_R,
                            n_slabs=n_slabs)

    H_LD = np.zeros((n_dof, N_D), dtype=complex)
    H_LD[:, :n_dof] = H_01
    H_DR = np.zeros((N_D, n_dof), dtype=complex)
    H_DR[-n_dof:, :] = H_01

    trans_ballistic = np.zeros(nfreq)
    for iw, z2 in enumerate(z2_arr):
        trans_ballistic[iw] = ballistic_transmission_z2(
            z2, H_D, H_00, H_01, H_LD, H_DR)

    if verbose:
        print(f"  Ballistic max T: {trans_ballistic.max():.4f}")

    lattice = phonon.primitive.cell
    tidx = "xyz".index(transport_direction)
    perp_idx = [i for i in range(3) if i != tidx]
    a1 = lattice[perp_idx[0]]
    a2 = lattice[perp_idx[1]]
    A_c = np.linalg.norm(np.cross(a1, a2)) * 1e-20

    omega_rad = freqs_thz * THZ_TO_RAD
    n_bose_L = bose_full_axis(freqs_thz, T_L)
    n_bose_R = bose_full_axis(freqs_thz, T_R)
    spectral_J_ball = HBAR_SI * omega_rad * (n_bose_L - n_bose_R) * trans_ballistic
    J_ball_total = np.sum(spectral_J_ball[pos_mask]) * dw_thz * 1e12
    G_ball = J_ball_total / (A_c * delta_T) if delta_T != 0 else 0.0

    if verbose:
        print(f"  Ballistic thermal conductance: {G_ball:.2f} W/(m^2 K)")

    def _g_dict_from_dense(g_dense, *, max_offset):
        """Build {(K, K'): block} dict from a dense (nfreq, N_D, N_D) G.

        ``max_offset = None`` includes every (K, K') pair; otherwise
        keeps only ``|K - K'| <= max_offset``.
        """
        out: dict[tuple[int, int], np.ndarray] = {}
        for K in range(n_slabs):
            sK = slice(K * n_dof, (K + 1) * n_dof)
            for Kp in range(n_slabs):
                if max_offset is not None and abs(K - Kp) > max_offset:
                    continue
                sKp = slice(Kp * n_dof, (Kp + 1) * n_dof)
                out[(K, Kp)] = g_dense[:, sK, sKp]
        return out

    def se_kernel(G_less_dev_q, G_great_dev_q):
        Sigma_l_new = np.zeros((1, nfreq, N_D, N_D), dtype=complex)
        Sigma_g_new = np.zeros_like(Sigma_l_new)

        gl = _g_dict_from_dense(G_less_dev_q[0], max_offset=g_cutoff)
        gg = _g_dict_from_dense(G_great_dev_q[0], max_offset=g_cutoff)
        sl_blocks, sg_blocks = compute_phph_self_energy_finite_multi_slab(
            gl, gg, phi_dev_blocks, n_slabs,
            freqs_thz, dw_thz,
            sigma_cutoff=sigma_cutoff,
            g_cutoff=g_cutoff,
            dc_handling=dc_handling,
        )
        for (I, J), block in sl_blocks.items():
            sI = slice(I * n_dof, (I + 1) * n_dof)
            sJ = slice(J * n_dof, (J + 1) * n_dof)
            Sigma_l_new[0, :, sI, sJ] = block
        for (I, J), block in sg_blocks.items():
            sI = slice(I * n_dof, (I + 1) * n_dof)
            sJ = slice(J * n_dof, (J + 1) * n_dof)
            Sigma_g_new[0, :, sI, sJ] = block
        return Sigma_l_new, Sigma_g_new

    scba_result = scba_loop_dev(
        z2_arr=z2_arr, freqs_thz=freqs_thz, dw_thz=dw_thz,
        omega_rad=omega_rad, pos_mask=pos_mask,
        n_slabs=n_slabs, n_dof=n_dof, N_D=N_D,
        H_D_list=[H_D], obc_list=[obc], btd_blocks_list=[(H_00, H_01)],
        n_kpts=1,
        se_kernel=se_kernel,
        T_L=T_L, T_R=T_R,
        max_scba_iter=max_scba_iter, scba_tol=scba_tol,
        conservation_tol=conservation_tol,
        mixing=mixing, anderson_mixing=anderson_mixing,
        anderson_depth=anderson_depth,
        scattering_contacts=scattering_contacts,
        retarded=retarded,
        verbose=verbose,
        solver=solver,
        anderson_safeguard=anderson_safeguard,
        zero_mode_projection=zero_mode_projection,
        gate_on_conservation=gate_on_conservation,
        divergence_guard=divergence_guard,
        masses_primitive=phonon.primitive.masses,
    )

    spectral_J_L = scba_result["spectral_J_L"]
    spectral_J_R = scba_result["spectral_J_R"]
    Sigma_R = scba_result["Sigma_R"]
    Sigma_l = scba_result["Sigma_l"]
    Sigma_g = scba_result["Sigma_g"]
    conservation_err = scba_result["conservation_err"]
    convergence_history = scba_result["convergence_history"]

    spectral_J_anh = 0.5 * (spectral_J_L + spectral_J_R)
    J_anh_total = np.sum(spectral_J_anh[pos_mask]) * dw_thz * 1e12
    G_anh = J_anh_total / (A_c * delta_T) if delta_T != 0 else 0.0

    if verbose:
        print(f"  Anharmonic thermal conductance: {G_anh:.2f} W/(m^2 K)")
        print(f"  Heat flow conservation: {conservation_err:.4e}")

    return {
        "freqs_thz": freqs_thz[pos_mask],
        "omega_rad": freqs_thz[pos_mask] * THZ_TO_RAD,
        "transmission_ballistic": trans_ballistic[pos_mask],
        "spectral_heat_current_ballistic": spectral_J_ball[pos_mask],
        "spectral_heat_current": spectral_J_anh[pos_mask],
        "spectral_heat_current_L": spectral_J_L[pos_mask].copy(),
        "spectral_heat_current_R": spectral_J_R[pos_mask].copy(),
        "heat_current_ballistic": J_ball_total,
        "heat_current": J_anh_total,
        "thermal_conductance_ballistic": G_ball,
        "thermal_conductance_anharmonic": G_anh,
        "heat_flow_conservation": conservation_err,
        "delta_T": delta_T,
        "n_scba_iterations": len(convergence_history) + 1,
        "convergence_history": convergence_history,
        "scba_converged": scba_result.get("converged", True),
        "scba_residual": scba_result.get("scba_residual", float("nan")),
        "self_energy_retarded": Sigma_R[0, pos_mask],
        "self_energy_lesser": Sigma_l[0, pos_mask],
        "self_energy_greater": Sigma_g[0, pos_mask],
    }


def transmission_q(
    phonon,
    fc3_hdf5=None,
    q_mesh_transverse: tuple[int, int] = (4, 4),
    freq_range_thz: tuple[float, float, int] = (0.01, 16.0, 101),
    transport_direction: str = "x",
    eta_factor: float = 1.0,
    temperature: float = 300.0,
    delta_T: float = 10.0,
    max_scba_iter: int = 80,
    scba_tol: float = 1e-3,
    conservation_tol: float = 1e-3,
    mixing: float = 0.3,
    anderson_mixing: bool = False,
    anderson_depth: int = 8,
    n_slabs: int = 1,
    verbose: bool = True,
    M_stacked_override: np.ndarray = None,
    retarded: str = "half",
    scattering_contacts: bool = False,
    hilbert_retarded: bool = False,
) -> dict:
    """Anharmonic phonon transport with full q-dependent dense self-energy.

    Uses a Γ-centered q-mesh ``[0, 1/N, ..., (N-1)/N]`` for closure under
    subtraction (required for q-convolution).

    When ``q_mesh=(1,1)``, must reproduce :func:`transmission_finite`
    exactly. A regression check is run automatically in that case.
    """
    if hilbert_retarded:
        retarded = "pv"

    from phonon_inputs.convention import get_btd_blocks
    from phonon_inputs.separable import (
        build_supercell_mapping,
        build_realspace_fc3_matrices,
        build_gathering_matrix,
        build_q_diff_map,
    )

    freqs_thz, dw_thz, eta_w, z2_arr, pos_mask, mid = build_frequency_grid(
        freq_range_thz, eta_factor=eta_factor)
    nfreq = len(freqs_thz)

    n_atoms = len(phonon.primitive.masses)
    n_dof = 3 * n_atoms
    N_D = n_slabs * n_dof

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(
        phonon, transport_direction)
    masses_super = phonon.supercell.masses
    n_super = len(masses_super)
    dim_sc = n_super * 3

    if M_stacked_override is not None:
        M_stacked = M_stacked_override
        fc3_raw = None
    else:
        fc3_raw = load_fc3_raw(fc3_hdf5)
        M_stacked = build_realspace_fc3_matrices(
            fc3_raw, n_atoms, masses_super, ref_sc_atoms)

    if verbose:
        if M_stacked_override is None:
            print(f"  FC3 raw shape: {fc3_raw.shape}")
        print(f"  Supercell atoms: {n_super}, dim_sc: {dim_sc}")
        print(f"  M_stacked norm: {np.linalg.norm(M_stacked):.4e}")
        print(f"  Device: {n_slabs} slab(s), {N_D} DOFs per q-point")

    nkx, nky = q_mesh_transverse
    q_1d_x = [i / nkx for i in range(nkx)]
    q_1d_y = [j / nky for j in range(nky)]
    q_points = [(qx, qy) for qx in q_1d_x for qy in q_1d_y]
    n_kpts = len(q_points)
    q_diff_map = build_q_diff_map(nkx, nky)

    T_all_q = []
    for qx, qy in q_points:
        T = build_gathering_matrix(
            prim_indices, cell_frac, (qx, qy), n_atoms, transport_direction)
        T_all_q.append(T)

    T_L = temperature + delta_T / 2.0
    T_R = temperature - delta_T / 2.0

    if verbose:
        print(f"  q-mesh: {nkx}x{nky} = {n_kpts} (Gamma-centered)")
        print(f"  Frequency grid: {nfreq} points, "
              f"{freqs_thz[0]:.2f} to {freqs_thz[-1]:.2f} THz")
        print(f"  Temperature: {temperature} K, delta_T: {delta_T} K")
        print(f"  eta_w = {eta_w:.4e} THz")
        mix_str = (f"Anderson(depth={anderson_depth})" if anderson_mixing
                   else "linear")
        print(f"  SCBA: max {max_scba_iter} iter, tol={scba_tol}, "
              f"mix={mixing}, method={mix_str}")
        print(f"  Retarded SE: {retarded}")

    btd_blocks = []
    for qx, qy in q_points:
        H_00, H_01 = get_btd_blocks(
            phonon, (qx, qy), transport_direction=transport_direction,
            conversion_factor=CONVERSION_THZ2)
        btd_blocks.append((H_00, H_01))

    if verbose:
        suffix = " (will update each SCBA iter)" if scattering_contacts else ""
        print(f"  Precomputing OBC self-energies (batched)...{suffix}")
    obc_all = []
    H_D_all = []
    for iq, (H_00, H_01) in enumerate(btd_blocks):
        H_D = build_device_hamiltonian(H_00, H_01, n_slabs)
        H_D_all.append(H_D)
        obc = compute_obc_batch(z2_arr, H_00, H_01, freqs_thz, T_L, T_R,
                                n_slabs=n_slabs)
        obc_all.append(obc)

    trans_ballistic = np.zeros(nfreq)
    for iq, (H_00, H_01) in enumerate(btd_blocks):
        H_D = H_D_all[iq]
        H_LD = np.zeros((n_dof, N_D), dtype=complex)
        H_LD[:, :n_dof] = H_01
        H_DR = np.zeros((N_D, n_dof), dtype=complex)
        H_DR[-n_dof:, :] = H_01
        for iw, z2 in enumerate(z2_arr):
            trans_ballistic[iw] += ballistic_transmission_z2(
                z2, H_D, H_00, H_01, H_LD, H_DR)
    trans_ballistic /= n_kpts

    if verbose:
        print(f"  Ballistic max T: {trans_ballistic.max():.4f}")

    lattice = phonon.primitive.cell
    tidx = "xyz".index(transport_direction)
    perp_idx = [i for i in range(3) if i != tidx]
    a1 = lattice[perp_idx[0]]
    a2 = lattice[perp_idx[1]]
    A_c = np.linalg.norm(np.cross(a1, a2)) * 1e-20

    omega_rad = freqs_thz * THZ_TO_RAD
    n_bose_L = bose_full_axis(freqs_thz, T_L)
    n_bose_R = bose_full_axis(freqs_thz, T_R)
    spectral_J_ball = HBAR_SI * omega_rad * (n_bose_L - n_bose_R) * trans_ballistic
    J_ball_total = np.sum(spectral_J_ball[pos_mask]) * dw_thz * 1e12
    G_ball = J_ball_total / (A_c * delta_T) if delta_T != 0 else 0.0

    if verbose:
        print(f"  Ballistic thermal conductance: {G_ball:.2f} W/(m^2 K)")

    def se_kernel(G_lesser_slab, G_greater_slab):
        Sigma_l_new = np.zeros(
            (n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
        Sigma_g_new = np.zeros_like(Sigma_l_new)
        for l in range(n_slabs):
            sl_n, sg_n = compute_phph_self_energy_q_dense(
                G_lesser_slab[l], G_greater_slab[l],
                M_stacked, T_all_q, q_diff_map,
                n_atoms, n_kpts, freqs_thz, dw_thz)
            Sigma_l_new[l] = sl_n
            Sigma_g_new[l] = sg_n
        return Sigma_l_new, Sigma_g_new

    scba_result = scba_loop(
        z2_arr=z2_arr, freqs_thz=freqs_thz, dw_thz=dw_thz,
        omega_rad=omega_rad, pos_mask=pos_mask,
        n_slabs=n_slabs, n_dof=n_dof, N_D=N_D,
        H_D_list=H_D_all, obc_list=obc_all, btd_blocks_list=btd_blocks,
        n_kpts=n_kpts,
        se_kernel=se_kernel,
        T_L=T_L, T_R=T_R,
        max_scba_iter=max_scba_iter, scba_tol=scba_tol,
        conservation_tol=conservation_tol,
        mixing=mixing, anderson_mixing=anderson_mixing,
        anderson_depth=anderson_depth,
        scattering_contacts=scattering_contacts,
        retarded=retarded,
        verbose=verbose,
    )

    spectral_J_L = scba_result["spectral_J_L"]
    spectral_J_R = scba_result["spectral_J_R"]
    Sigma_R_q = scba_result["Sigma_R"]
    Sigma_l_q = scba_result["Sigma_l"]
    Sigma_g_q = scba_result["Sigma_g"]
    conservation_err = scba_result["conservation_err"]
    convergence_history = scba_result["convergence_history"]

    spectral_J_anh = 0.5 * (spectral_J_L + spectral_J_R)
    J_anh_total = np.sum(spectral_J_anh[pos_mask]) * dw_thz * 1e12
    G_anh = J_anh_total / (A_c * delta_T) if delta_T != 0 else 0.0

    if verbose:
        print(f"  Anharmonic thermal conductance: {G_anh:.2f} W/(m^2 K)")
        print(f"  Heat flow conservation: {conservation_err:.4e}")

    result = {
        "freqs_thz": freqs_thz[pos_mask],
        "omega_rad": freqs_thz[pos_mask] * THZ_TO_RAD,
        "transmission_ballistic": trans_ballistic[pos_mask],
        "spectral_heat_current_ballistic": spectral_J_ball[pos_mask],
        "spectral_heat_current": spectral_J_anh[pos_mask],
        "spectral_heat_current_L": spectral_J_L[pos_mask].copy(),
        "spectral_heat_current_R": spectral_J_R[pos_mask].copy(),
        "heat_current_ballistic": J_ball_total,
        "heat_current": J_anh_total,
        "thermal_conductance_ballistic": G_ball,
        "thermal_conductance_anharmonic": G_anh,
        "heat_flow_conservation": conservation_err,
        "delta_T": delta_T,
        "n_scba_iterations": len(convergence_history) + 1,
        "convergence_history": convergence_history,
        "self_energy_retarded": Sigma_R_q[:, :, pos_mask],
        "self_energy_lesser": Sigma_l_q[:, :, pos_mask],
        "self_energy_greater": Sigma_g_q[:, :, pos_mask],
    }

    if n_kpts == 1 and verbose:
        _check_q11_vs_finite(result)
    return result


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------


def _check_q11_vs_finite(res_q, rtol=5e-3, atol=1e-8):
    """Warn if q=(1,1) output is internally inconsistent.

    This checks the q-path result against itself (ballistic should match
    the analytic Γ-only result). For a full q-vs-finite check, run both
    entry points and call :func:`compare_q11_to_finite`.
    """
    G_ball = res_q["thermal_conductance_ballistic"]
    G_anh = res_q["thermal_conductance_anharmonic"]
    if G_ball != 0 and abs(G_anh) > 2 * abs(G_ball):
        warnings.warn(
            f"q=(1,1) sanity: G_anh ({G_anh:.2e}) > 2 * G_ball ({G_ball:.2e})"
        )


def compare_q11_to_finite(res_q, res_f, rtol=5e-3, atol=1e-8):
    """Verify that ``q=(1,1)`` matches the finite (Γ-only) solver.

    Raises ``AssertionError`` on mismatch.
    """
    keys = [
        "heat_current",
        "thermal_conductance_anharmonic",
        "thermal_conductance_ballistic",
        "heat_flow_conservation",
        "transmission_ballistic",
        "spectral_heat_current",
    ]
    for key in keys:
        a = np.asarray(res_q[key])
        b = np.asarray(res_f[key])
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            max_rel = float(np.max(np.abs(a - b) / (np.abs(b) + atol)))
            raise AssertionError(
                f"q=(1,1) vs finite mismatch in '{key}': "
                f"max rel err = {max_rel:.2e}"
            )
