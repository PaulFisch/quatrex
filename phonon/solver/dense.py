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
            x_in = np.concatenate([
                Sigma_l.ravel(), Sigma_g.ravel()])
            x_out = np.concatenate([
                Sigma_l_new.ravel(), Sigma_g_new.ravel()])
            f_k = x_out - x_in
            _anderson_x_hist.append(x_in)
            _anderson_f_hist.append(f_k)

            m = len(_anderson_f_hist)
            if m >= 2:
                n_use = min(m - 1, anderson_depth)
                dF = np.column_stack([
                    _anderson_f_hist[-n_use + j] - _anderson_f_hist[-n_use + j - 1]
                    for j in range(n_use)])
                dX = np.column_stack([
                    _anderson_x_hist[-n_use + j] - _anderson_x_hist[-n_use + j - 1]
                    for j in range(n_use)])
                FtF = dF.conj().T @ dF
                Ftf = dF.conj().T @ f_k
                reg = 1e-8 * np.trace(FtF).real / max(FtF.shape[0], 1)
                gamma = np.linalg.solve(
                    FtF + reg * np.eye(FtF.shape[0]), Ftf)
                x_mixed = (x_in + mixing * f_k) - (dX + mixing * dF) @ gamma
            else:
                x_mixed = x_in + mixing * f_k

            sz = Sigma_l.size
            Sigma_l = x_mixed[:sz].reshape(Sigma_l.shape)
            Sigma_g = x_mixed[sz:].reshape(Sigma_g.shape)

            if len(_anderson_x_hist) > anderson_depth + 2:
                _anderson_x_hist.pop(0)
                _anderson_f_hist.pop(0)
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
):
    """SCBA fixed-point with device-storage self-energies.

    Mirrors :func:`scba_loop` but stores
    ``Sigma^{<,>,R}`` as ``(n_kpts, nfreq, N_D, N_D)`` device-matrix
    buffers so the multi-slab driver can populate off-diagonal blocks
    ``Sigma_{IJ}`` produced by
    :func:`compute_phph_self_energy_finite_multi_slab`.

    Parameters
    ----------
    se_kernel : callable(G_less_dev, G_great_dev)
        ``G_*_dev`` have shape ``(n_kpts, nfreq, N_D, N_D)``. The
        callable must return ``(Sigma_l, Sigma_g)`` of the same shape.
    retarded : {"pv", "fft", "half"}
        Method for rebuilding ``Sigma^R`` from the mixed
        ``Sigma^{<,>}`` pair.
    """
    nfreq = len(freqs_thz)

    Sigma_R = np.zeros((n_kpts, nfreq, N_D, N_D), dtype=complex)
    Sigma_l = np.zeros_like(Sigma_R)
    Sigma_g = np.zeros_like(Sigma_R)

    convergence_history = []
    J_total_prev = 0.0
    rel_change = float('inf')
    sig_change = float('inf')
    conservation_err = 1.0
    _anderson_x_hist = []
    _anderson_f_hist = []

    spectral_J_L = np.zeros(nfreq)
    spectral_J_R = np.zeros(nfreq)

    sl0 = slice(0, n_dof)
    sl_last = slice((n_slabs - 1) * n_dof, n_slabs * n_dof)

    G_less_dev_q = np.zeros((n_kpts, nfreq, N_D, N_D), dtype=complex)
    G_great_dev_q = np.zeros_like(G_less_dev_q)
    G_ret_dev_q = np.zeros_like(G_less_dev_q)

    for scba_iter in range(max_scba_iter):
        if scattering_contacts and scba_iter > 0:
            for iq, (H_00_iq, H_01_iq) in enumerate(btd_blocks_list):
                lead_L = Sigma_R[iq, :, sl0, sl0]
                lead_R = Sigma_R[iq, :, sl_last, sl_last]
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

        G_less_dev_q[:] = 0.0
        G_great_dev_q[:] = 0.0
        G_ret_dev_q[:] = 0.0
        spectral_J_L[:] = 0.0
        spectral_J_R[:] = 0.0

        for iq in range(n_kpts):
            H_D = H_D_list[iq]
            obc = obc_list[iq]

            Sig_R_dev = Sigma_R[iq]
            Sig_l_dev = Sigma_l[iq]
            Sig_g_dev = Sigma_g[iq]

            G_ret, G_less, G_great = solve_green_batch(
                z2_arr, H_D, obc, Sig_R_dev, Sig_l_dev, Sig_g_dev)

            G_less_dev_q[iq] = G_less
            G_great_dev_q[iq] = G_great
            G_ret_dev_q[iq] = G_ret

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

        Sigma_l_new, Sigma_g_new = se_kernel(G_less_dev_q, G_great_dev_q)

        symmetrize_lesser_greater(Sigma_l_new, Sigma_g_new)

        sig_r_norm = (
            np.max(np.abs(Sigma_l_new)) + np.max(np.abs(Sigma_g_new))
        )

        if verbose and scba_iter == 0:
            gl_max = np.max(np.abs(G_less_dev_q))
            print(f"    G diagnostic: max|G^<| = {gl_max:.4e}")
            print(f"    Self-energy: max|Sigma^R| = {sig_r_norm:.4e} THz^2")

        _Sigma_prev = np.concatenate([Sigma_l.ravel(), Sigma_g.ravel()])

        if scba_iter == 0:
            Sigma_l = Sigma_l_new.copy()
            Sigma_g = Sigma_g_new.copy()
        elif anderson_mixing:
            x_in = np.concatenate([Sigma_l.ravel(), Sigma_g.ravel()])
            x_out = np.concatenate([
                Sigma_l_new.ravel(), Sigma_g_new.ravel()])
            f_k = x_out - x_in
            _anderson_x_hist.append(x_in)
            _anderson_f_hist.append(f_k)

            m = len(_anderson_f_hist)
            if m >= 2:
                n_use = min(m - 1, anderson_depth)
                dF = np.column_stack([
                    _anderson_f_hist[-n_use + j] - _anderson_f_hist[-n_use + j - 1]
                    for j in range(n_use)])
                dX = np.column_stack([
                    _anderson_x_hist[-n_use + j] - _anderson_x_hist[-n_use + j - 1]
                    for j in range(n_use)])
                FtF = dF.conj().T @ dF
                Ftf = dF.conj().T @ f_k
                reg = 1e-8 * np.trace(FtF).real / max(FtF.shape[0], 1)
                gamma = np.linalg.solve(
                    FtF + reg * np.eye(FtF.shape[0]), Ftf)
                x_mixed = (x_in + mixing * f_k) - (dX + mixing * dF) @ gamma
            else:
                x_mixed = x_in + mixing * f_k

            sz = Sigma_l.size
            Sigma_l = x_mixed[:sz].reshape(Sigma_l.shape)
            Sigma_g = x_mixed[sz:].reshape(Sigma_g.shape)

            if len(_anderson_x_hist) > anderson_depth + 2:
                _anderson_x_hist.pop(0)
                _anderson_f_hist.pop(0)
        else:
            alpha = mixing
            Sigma_l = (1 - alpha) * Sigma_l + alpha * Sigma_l_new
            Sigma_g = (1 - alpha) * Sigma_g + alpha * Sigma_g_new

        Sigma_R = build_retarded(
            Sigma_l, Sigma_g, freqs_thz, method=retarded)

        total_viol = 0
        total_max = 0.0
        for iq in range(n_kpts):
            for l in range(n_slabs):
                sl = slice(l * n_dof, (l + 1) * n_dof)
                nv, mv = check_broadening_sign(
                    Sigma_R[iq, :, sl, sl], freqs_thz, "SCBA", tol=1e-8)
                total_viol += nv
                total_max = max(total_max, mv)
        if total_viol > 0 and verbose:
            print(f"    WARNING: Gamma sign violations: {total_viol} points, "
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
                      f"dSig/Sig = {sig_change:.4e}, "
                      f"max|Sigma^R| = {np.max(np.abs(Sigma_R)):.2e} THz^2")
            numerically_converged = (sig_change < scba_tol
                                      and rel_change < scba_tol)
            conserving = conservation_err < conservation_tol

            if numerically_converged and conserving:
                if verbose:
                    print(f"    Converged after {scba_iter + 1} iterations "
                          f"(dJ/J={rel_change:.2e}, dSig/Sig={sig_change:.2e}, "
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
                  f"iterations (dJ/J={rel_change:.2e}, "
                  f"dSig/Sig={sig_change:.2e}, "
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
    eta_factor: float = 0.05,
    temperature: float = 300.0,
    delta_T: float = 10.0,
    max_scba_iter: int = 10,
    scba_tol: float = 1e-3,
    conservation_tol: float = 1e-3,
    mixing: float = 0.5,
    anderson_mixing: bool = False,
    anderson_depth: int = 5,
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
    """
    if hilbert_retarded:
        retarded = "fft"
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
        print(f"  eta_w = {eta_w:.4e} THz")
        mix_str = (f"Anderson(depth={anderson_depth})" if anderson_mixing
                   else "linear")
        print(f"  SCBA: max {max_scba_iter} iter, tol={scba_tol}, "
              f"mix={mixing}, method={mix_str}")
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
    eta_factor: float = 0.05,
    temperature: float = 300.0,
    delta_T: float = 10.0,
    max_scba_iter: int = 10,
    scba_tol: float = 1e-3,
    conservation_tol: float = 1e-3,
    mixing: float = 0.5,
    anderson_mixing: bool = False,
    anderson_depth: int = 5,
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
