"""Test fully symmetric CP decomposition of FC3.

The fully symmetric CP uses ONE shared mode per rank:
    Phi[i,j,k] = sum_r lam_r v_r[i] v_r[j] v_r[k]

No S3 permutation sum needed — the symmetry is built into the ansatz.
The self-energy kernel becomes a simple element-wise product of projected G's.

Tests:
1. Fitting quality vs PCP and SC-CP at various ranks
2. Polarization identity: convert PCP to fully symmetric
3. Self-energy kernel validation
"""

import sys
from pathlib import Path
import numpy as np
import time
import torch

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.pcp import (
    fit_pcp, fit_supercell_cp, _build_target,
    fourier_transform_supercell_cp,
    _compute_phph_self_energy_pcp,
)
from phonon_inputs.separable import (
    build_supercell_mapping, build_realspace_fc3_matrices,
    build_gathering_matrix, build_q_diff_map,
)
from phonon_inputs.anharmonic import _compute_phph_self_energy_q_dense
from phonon_inputs.constants import CONVERSION_FC3_THZ, HBAR_SI
import h5py


# ---- Fully symmetric CP forward model ----

def _symmetric_cp_forward_torch(v, lambdas, p2s_expanded, nat_prim, n_super, R):
    """Forward model: Phi[i,j,k,a,b,c] = sum_r lam_r v_r[p2s[i]*3+a] v_r[j*3+b] v_r[k*3+c].

    v : (R, n_super, 3)
    lambdas : (R,)
    p2s_expanded : (nat_prim,), long — maps prim atom -> supercell atom
    """
    ext = v[:, p2s_expanded, :]  # (R, nat_prim, 3) — cell-0 restriction
    wv = lambdas[:, None, None] * v  # (R, n_super, 3)

    fc3 = torch.einsum('ria, rjb, rkc -> ijkabc', ext, wv, v)
    # No S3 sum needed — it's already symmetric because ext comes from v!
    # But wait: ext = v restricted to cell-0 atoms, while the full tensor would
    # have v on all three legs. The compact form breaks the full symmetry.
    # We need to verify this gives the right FC3.
    return fc3


def _project_asr_symmetric(v):
    """ASR projection for symmetric CP modes.

    Requires sum_j v[r, j, beta] = 0 for each (r, beta).

    v : (R, n_super, 3)
    """
    return v - v.mean(dim=1, keepdim=True)


def _project_asr_grad_symmetric(grad):
    return (grad - grad.mean(dim=1, keepdim=True)).contiguous()


def fit_symmetric_cp(fc3_raw, phonon, R=24, max_iter=2000, verbose=True):
    """Fit fully symmetric CP: Phi = sum_r lam_r v_r^{otimes 3}.

    One shared mode per rank, no permutation sum.
    """
    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    p2s = torch.tensor(phonon.primitive.p2s_map.astype(np.int64), dtype=torch.long)

    target_np = _build_target(fc3_raw, phonon)
    target_norm = np.linalg.norm(target_np)

    if verbose:
        n_params = R * n_super * 3 + R
        n_target = nat_prim * n_super * n_super * 27
        print(f"  Symmetric CP: R={R}, params={n_params}, target={n_target}")

    target_t = torch.tensor(target_np / target_norm, dtype=torch.float64)

    rng = np.random.default_rng(42)
    scale = (1.0 / (R * n_super)) ** (1.0 / 3.0)

    v_init = rng.normal(0, scale, (R, n_super, 3))
    v_init -= v_init.mean(axis=1, keepdims=True)

    v_param = torch.tensor(v_init, dtype=torch.float64, requires_grad=True)
    lam_param = torch.tensor(np.ones(R, dtype=np.float64), requires_grad=True)

    best_err = float('inf')
    best_v = v_param.detach().clone()
    best_lam = lam_param.detach().clone()

    def forward():
        fc3 = _symmetric_cp_forward_torch(v_param, lam_param, p2s, nat_prim, n_super, R)
        return torch.sum((fc3 - target_t) ** 2), fc3

    def update_best(err_val):
        nonlocal best_err, best_v, best_lam
        if err_val < best_err:
            best_err = err_val
            best_v = v_param.detach().clone()
            best_lam = lam_param.detach().clone()

    # Phase 1: Adam
    adam_iters = min(max_iter * 3 // 4, 1500)
    optimizer = torch.optim.Adam([v_param, lam_param], lr=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-4)

    if verbose:
        print(f"  Phase 1: Adam ({adam_iters} iters)...")

    for it in range(1, adam_iters + 1):
        optimizer.zero_grad()
        loss, fc3_approx = forward()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            v_param.data = _project_asr_symmetric(v_param.data)

        if it % 200 == 0 or it == 1:
            with torch.no_grad():
                err = torch.sqrt(torch.sum((fc3_approx - target_t) ** 2)).item()
            update_best(err)
            if verbose:
                print(f"    iter {it:5d}: rel_err={err:.6e}")

    # Phase 2: L-BFGS
    lbfgs_iters = max_iter - adam_iters
    if verbose:
        print(f"  Phase 2: L-BFGS ({lbfgs_iters} iters)...")

    v_param.data.copy_(best_v)
    lam_param.data.copy_(best_lam)

    lbfgs = torch.optim.LBFGS([v_param, lam_param], lr=1.0, max_iter=20,
                                history_size=50, line_search_fn='strong_wolfe')

    for outer in range(max(1, lbfgs_iters // 20)):
        def closure():
            lbfgs.zero_grad()
            loss, _ = forward()
            loss.backward()
            if v_param.grad is not None:
                v_param.grad.data = _project_asr_grad_symmetric(v_param.grad.data)
            return loss
        lbfgs.step(closure)

        if (outer + 1) % 5 == 0 or outer == 0:
            with torch.no_grad():
                _, fc3_approx = forward()
                err = torch.sqrt(torch.sum((fc3_approx - target_t) ** 2)).item()
            update_best(err)
            if verbose:
                print(f"    L-BFGS step {outer+1}: rel_err={err:.6e}")

    v_modes = best_v.numpy()
    lambdas = best_lam.numpy() * target_norm

    # Sort by |lambda|
    order = np.argsort(-np.abs(lambdas))
    v_modes = v_modes[order]
    lambdas = lambdas[order]

    info = {'rel_err': best_err, 'target_norm': target_norm}
    if verbose:
        print(f"  Fit: rel_err={best_err:.6e}")

    return v_modes, lambdas, info


def fourier_transform_symmetric_cp(v_modes, lambdas, phonon, q_points,
                                     transport_direction='x'):
    """FT fully symmetric CP modes.

    Returns f_modes_all_q (for internal projection) and f_ext (cell-0 modes).
    Since the mode is shared across all legs, f_ext comes from the same v.
    """
    from phonon_inputs.separable import build_supercell_mapping, build_gathering_matrix

    nat_prim = len(phonon.primitive.masses)
    n_dof = nat_prim * 3
    R = len(lambdas)
    n_super = len(phonon.supercell.masses)
    n_kpts = len(q_points)
    p2s = phonon.primitive.p2s_map

    prim_indices, cell_frac, _, _ = build_supercell_mapping(phonon, transport_direction)

    v_flat = v_modes.reshape(R, n_super * 3)

    # Internal FT modes: f_r(q) = T(q) @ v_r
    # Shape: (n_kpts, R, n_dof) — note: only ONE mode set (not 3)
    f_int = np.zeros((n_kpts, R, n_dof), dtype=complex)
    for iq, (qx, qy) in enumerate(q_points):
        T_q = build_gathering_matrix(prim_indices, cell_frac,
                                     (qx, qy), nat_prim, transport_direction)
        f_int[iq] = v_flat @ T_q.T

    # External modes: v restricted to cell-0 atoms
    f_ext = np.zeros((R, n_dof))
    for kappa in range(nat_prim):
        j_sc = int(p2s[kappa])
        f_ext[:, kappa*3:kappa*3+3] = v_modes[:, j_sc, :]

    return f_int, f_ext


def compute_self_energy_symmetric_cp(
    G_lesser_q, G_greater_q,
    f_int, f_ext, lambdas,
    n_dof, n_kpts, omega_grid_thz, dw_thz,
    q_diff_map=None,
):
    """Self-energy kernel for fully symmetric CP.

    No S3 permutation sum — just element-wise products.

    Sigma[a,a'](q,w) = sum_{r,s} lam_r lam_s ext_r[a] ext_s*[a']
                        * (1/N_q) sum_{q'} conv(g_rs(q'), g_rs(q-q'))

    g_rs(q,w) = f_r(q)^T G(q,w) f_s*(q)  — scalar projection.
    """
    n_freq = len(omega_grid_thz)
    R = len(lambdas)

    n_low = max(0, int(np.round(omega_grid_thz[0] / dw_thz)))
    n_ext = n_low + n_freq
    n_fft = 2 * n_ext
    freq_sl = slice(n_low, n_low + n_freq)

    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi) / n_kpts

    def _pad(G_q):
        out = np.zeros((n_kpts, n_fft, n_dof, n_dof), dtype=complex)
        out[:, n_low:n_low + n_freq] = G_q
        return out

    GL = _pad(G_lesser_q)
    GG = _pad(G_greater_q)

    if q_diff_map is None:
        q_diff_map = np.array([[(i - j) % n_kpts for j in range(n_kpts)]
                                for i in range(n_kpts)])

    # Phase 1: Scalar projections g[r,s,q,w] = f_r^T G f_s*
    gL = np.zeros((R, R, n_kpts, n_fft), dtype=complex)
    gG = np.zeros_like(gL)

    for iq in range(n_kpts):
        f = f_int[iq]           # (R, n_dof)
        f_conj = np.conj(f)     # (R, n_dof)

        # g = f @ G @ f*^T, shape (R, R) per w
        Gfc_L = GL[iq] @ f_conj.T   # (n_fft, n_dof, R)
        Gfc_G = GG[iq] @ f_conj.T

        gL[:, :, iq, :] = np.einsum('rm, wmq -> rqw', f, Gfc_L)
        gG[:, :, iq, :] = np.einsum('rm, wmq -> rqw', f, Gfc_G)

    # Phase 2: FFT over omega
    gL_hat = np.fft.fft(gL, axis=-1)
    gG_hat = np.fft.fft(gG, axis=-1)

    # Phase 3: For each q_ext, product + IFFT + sum over q'
    Sigma_lesser = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
    Sigma_greater = np.zeros_like(Sigma_lesser)

    lam_outer = np.outer(lambdas, lambdas)

    for iq_ext in range(n_kpts):
        iq_diffs = q_diff_map[iq_ext]  # (n_kpts,)

        # g at q-q' for all q': gather
        gLd = gL_hat[:, :, iq_diffs, :]  # (R, R, n_kpts, n_fft)
        gGd = gG_hat[:, :, iq_diffs, :]

        # Product in Fourier domain: g_hat(q') * g_hat(q-q')
        # Single element-wise multiply! No S3 loop.
        prod_L = gL_hat * gLd  # (R, R, n_kpts, n_fft)
        prod_G = gG_hat * gGd

        # IFFT, extract freq range, sum over q'
        conv_L = np.sum(np.fft.ifft(prod_L, axis=-1)[:, :, :, freq_sl], axis=2)
        conv_G = np.sum(np.fft.ifft(prod_G, axis=-1)[:, :, :, freq_sl], axis=2)
        # Shape: (R, R, n_freq)

        # Phase 4: Accumulate Sigma = ext @ (lam*lam * C) @ ext^T
        CL = (lam_outer[:, :, None] * conv_L).transpose(2, 0, 1)  # (n_freq, R, R)
        CG = (lam_outer[:, :, None] * conv_G).transpose(2, 0, 1)

        Sigma_lesser[iq_ext] = prefactor * np.einsum('ra,wrs,sb->wab', f_ext, CL, f_ext)
        Sigma_greater[iq_ext] = prefactor * np.einsum('ra,wrs,sb->wab', f_ext, CG, f_ext)

    Sigma_retarded = 0.5 * (Sigma_greater - Sigma_lesser)
    return Sigma_lesser, Sigma_greater, Sigma_retarded


def main():
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"
    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    nat_prim = len(phonon.primitive.masses)
    n_dof = nat_prim * 3
    n_super = len(phonon.supercell.masses)

    # ---- Fitting comparison ----
    print("=" * 60)
    print("Fitting quality comparison")
    print("=" * 60)

    print(f"\n{'R/Nc':>6s} {'PCP':>12s} {'SC-CP':>12s} {'Sym-CP':>12s}")
    print("-" * 48)
    for rank in [4, 8, 12, 24, 48]:
        _, _, i1 = fit_pcp(fc3_raw, phonon, N_c=rank, max_iter=2000, verbose=False)
        _, _, i2 = fit_supercell_cp(fc3_raw, phonon, N_c=rank, max_iter=2000, verbose=False)
        _, _, i3 = fit_symmetric_cp(fc3_raw, phonon, R=rank, max_iter=2000, verbose=False)
        print(f"{rank:6d} {i1['rel_err']:12.4e} {i2['rel_err']:12.4e} {i3['rel_err']:12.4e}")

    # ---- Self-energy validation ----
    print("\n" + "=" * 60)
    print("Self-energy kernel validation (symmetric CP)")
    print("=" * 60)

    nk = 4
    q_points = [(i/nk, j/nk) for i in range(nk) for j in range(nk)]
    n_kpts = len(q_points)
    q_diff_map = build_q_diff_map(nk, nk)

    # Dense reference from raw FC3
    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(phonon)
    M_stacked = build_realspace_fc3_matrices(fc3_raw, nat_prim, phonon.supercell.masses, ref_sc_atoms)
    T_all = [build_gathering_matrix(prim_indices, cell_frac, q, nat_prim, 'x') for q in q_points]

    n_freq = 21
    freqs = np.linspace(1.0, 14.0, n_freq)
    dw = freqs[1] - freqs[0]

    rng = np.random.default_rng(42)
    G_lesser = rng.standard_normal((n_kpts, n_freq, n_dof, n_dof)) + \
               1j * rng.standard_normal((n_kpts, n_freq, n_dof, n_dof))
    G_greater = rng.standard_normal((n_kpts, n_freq, n_dof, n_dof)) + \
                1j * rng.standard_normal((n_kpts, n_freq, n_dof, n_dof))
    G_lesser = 0.5 * (G_lesser - np.conj(G_lesser.transpose(0, 1, 3, 2)))
    G_greater = 0.5 * (G_greater - np.conj(G_greater.transpose(0, 1, 3, 2)))

    print("\nDense reference...")
    t0 = time.time()
    SL_d, SG_d, SR_d = _compute_phph_self_energy_q_dense(
        G_lesser, G_greater, M_stacked, T_all, q_diff_map,
        nat_prim, n_kpts, freqs, dw)
    t_dense = time.time() - t0
    norm_d = np.linalg.norm(SR_d)
    print(f"  Time: {t_dense:.3f}s")

    for R in [24, 48, 96, 144]:
        print(f"\nSymmetric CP R={R}:")
        v_modes, lambdas, info = fit_symmetric_cp(
            fc3_raw, phonon, R=R, max_iter=2000, verbose=False)
        print(f"  fit err: {info['rel_err']:.4e}")

        f_int, f_ext = fourier_transform_symmetric_cp(
            v_modes, lambdas, phonon, q_points, transport_direction='x')

        t0 = time.time()
        SL_s, SG_s, SR_s = compute_self_energy_symmetric_cp(
            G_lesser, G_greater, f_int, f_ext, lambdas,
            n_dof, n_kpts, freqs, dw, q_diff_map=q_diff_map)
        t_sym = time.time() - t0

        err = np.linalg.norm(SR_s - SR_d) / norm_d
        print(f"  SE err:  {err:.4e}")
        print(f"  Time:    {t_sym:.3f}s ({t_dense/t_sym:.2f}x vs dense)")

    # Compare with SC-CP at same self-energy error level
    print("\n" + "=" * 60)
    print("Speed comparison at similar accuracy")
    print("=" * 60)

    for N_c, R_sym in [(4, 24), (8, 48), (24, 144)]:
        u, lam_u, info_u = fit_supercell_cp(fc3_raw, phonon, N_c=N_c, max_iter=2000, verbose=False)
        f_q, f_e = fourier_transform_supercell_cp(u, lam_u, phonon, q_points, transport_direction='x')
        t0 = time.time()
        SL_c, _, SR_c = _compute_phph_self_energy_pcp(
            G_lesser, G_greater, f_q, lam_u,
            n_dof, n_kpts, freqs, dw, q_diff_map=q_diff_map, f_ext=f_e)
        t_cp = time.time() - t0
        err_cp = np.linalg.norm(SR_c - SR_d) / norm_d

        v, lam_v, info_v = fit_symmetric_cp(fc3_raw, phonon, R=R_sym, max_iter=2000, verbose=False)
        fi, fe = fourier_transform_symmetric_cp(v, lam_v, phonon, q_points, transport_direction='x')
        t0 = time.time()
        SL_s, _, SR_s = compute_self_energy_symmetric_cp(
            G_lesser, G_greater, fi, fe, lam_v,
            n_dof, n_kpts, freqs, dw, q_diff_map=q_diff_map)
        t_sym = time.time() - t0
        err_sym = np.linalg.norm(SR_s - SR_d) / norm_d

        print(f"  SC-CP N_c={N_c:2d}: {t_cp:.3f}s  err={err_cp:.4e}  fit={info_u['rel_err']:.4e}")
        print(f"  Sym   R={R_sym:3d}: {t_sym:.3f}s  err={err_sym:.4e}  fit={info_v['rel_err']:.4e}")
        print()


if __name__ == "__main__":
    main()
