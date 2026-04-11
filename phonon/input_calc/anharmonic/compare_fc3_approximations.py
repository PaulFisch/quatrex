"""Compare FC3 tensor approximation methods.

For each method and rank, computes:
  1. Number of parameters
  2. Relative Frobenius-norm error ||Phi - Phi_approx||_F / ||Phi||_F

Methods:
  1. Truncated SVD (separable)
  2. Partially Symmetric CP (PSCP) — algebraic, S2 on internal legs
  3. Symmetric CP with 3 modes (SCP3) — optimization, S3
  4. Fully Symmetric CP (FSCP) — optimization, S3, single mode per rank

Requires:
  - fc3_prim/fc3.hdf5 (run fc3-reap first)
  - fc3_prim/phono3py_disp.yaml
"""

import sys
import time
import itertools
from pathlib import Path

import h5py
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent  # input_calc/
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.constants import CONVERSION_FC3_THZ
from phonon_inputs.separable import (
    build_supercell_mapping,
    build_realspace_fc3_matrices,
    decompose_fc3_supercell,
)
from phonon_inputs.pcp import (
    _build_target,
    _supercell_cp_forward_torch,
    _project_asr_supercell,
    _project_asr_grad_supercell,
    fit_supercell_cp,
)

S3_PERMS = list(itertools.permutations(range(3)))


# =====================================================================
# Method 1: Truncated SVD
# =====================================================================

def svd_approximation(M_stacked, rank):
    """Truncated SVD at given rank. Returns reconstructed M_stacked."""
    U, S, Vt = np.linalg.svd(M_stacked, full_matrices=False)
    R = min(rank, len(S))
    return U[:, :R] @ np.diag(S[:R]) @ Vt[:R, :]


def svd_n_params(rank, n_dof, dim_sc):
    """Parameter count: R left vectors of length n_dof*dim_sc + R right vectors of length dim_sc."""
    return rank * (n_dof * dim_sc + dim_sc)


# =====================================================================
# Method 2: Partially Symmetric CP (PSCP)
# =====================================================================

def pscp_decomposition(M_stacked, n_dof, dim_sc, R_max=None, threshold=1e-10):
    """Algebraic PSCP: SVD of (n_dof, dim_sc^2) + eigendecomposition.

    Returns
    -------
    d_ext : (R_total, n_dof)
    v_int : (R_total, dim_sc)
    norms : (R_total,) — |d_r| * |v_r|, for sorting/truncation
    """
    Phi = M_stacked.reshape(n_dof, dim_sc, dim_sc)

    Phi_flat = Phi.reshape(n_dof, dim_sc * dim_sc)
    U_ext, sigma, Vt = np.linalg.svd(Phi_flat, full_matrices=False)
    R_svd = len(sigma)

    all_d = []
    all_v = []
    all_norms = []

    for r in range(R_svd):
        if sigma[r] < threshold * sigma[0]:
            break
        W_r = (sigma[r] * Vt[r]).reshape(dim_sc, dim_sc)
        W_r = 0.5 * (W_r + W_r.T)

        eigenvalues, eigenvectors = np.linalg.eigh(W_r)
        order = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        for p in range(len(eigenvalues)):
            if np.abs(eigenvalues[p]) < threshold * np.abs(eigenvalues[0]):
                break
            d_rp = U_ext[:, r] * eigenvalues[p]
            v_rp = eigenvectors[:, p]
            all_d.append(d_rp)
            all_v.append(v_rp)
            all_norms.append(np.abs(eigenvalues[p]) * np.linalg.norm(U_ext[:, r]))

    d_ext = np.array(all_d)
    v_int = np.array(all_v)
    norms = np.array(all_norms)

    # Sort by contribution magnitude
    order = np.argsort(-norms)
    d_ext = d_ext[order]
    v_int = v_int[order]
    norms = norms[order]

    if R_max is not None and R_max < len(d_ext):
        d_ext = d_ext[:R_max]
        v_int = v_int[:R_max]
        norms = norms[:R_max]

    return d_ext, v_int, norms


def pscp_reconstruct(d_ext, v_int, n_dof, dim_sc):
    """Reconstruct M_stacked from PSCP decomposition."""
    M_approx = np.zeros((n_dof * dim_sc, dim_sc))
    for a in range(n_dof):
        weighted_v = d_ext[:, a:a+1] * v_int  # (R, dim_sc)
        M_a = weighted_v.T @ v_int  # (dim_sc, dim_sc)
        M_approx[a * dim_sc:(a + 1) * dim_sc, :] = M_a
    return M_approx


def pscp_n_params(rank, n_dof, dim_sc):
    return rank * (n_dof + dim_sc)


# =====================================================================
# Method 3: Symmetric CP with 3 modes (SCP3) — uses fit_supercell_cp
# =====================================================================

def scp3_reconstruct_M_stacked(u_modes, lambdas, phonon, target_norm, n_dof, dim_sc):
    """Reconstruct M_stacked from supercell CP modes."""
    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    p2s = torch.tensor(phonon.primitive.p2s_map.astype(np.int64), dtype=torch.long)
    N_c = len(lambdas)

    with torch.no_grad():
        u_t = torch.tensor(u_modes, dtype=torch.float64)
        lam_t = torch.tensor(lambdas / target_norm, dtype=torch.float64)
        fc3_mw = _supercell_cp_forward_torch(
            u_t, lam_t, p2s, nat_prim, n_super, N_c,
        ).numpy() * target_norm

    # Reshape (nat_prim, n_super, n_super, 3, 3, 3) -> M_stacked
    M = np.zeros((n_dof * dim_sc, dim_sc))
    for i_prim in range(nat_prim):
        for alpha in range(3):
            a = 3 * i_prim + alpha
            block = fc3_mw[i_prim, :, :, alpha, :, :]
            M[a * dim_sc:(a + 1) * dim_sc, :] = block.transpose(0, 2, 1, 3).reshape(dim_sc, dim_sc)
    return M


def scp3_n_params(N_c, dim_sc):
    return N_c * (3 * dim_sc + 1)


# =====================================================================
# Method 4: Fully Symmetric CP (FSCP)
# =====================================================================

def _fscp_forward_torch(v, lambdas, p2s_map, nat_prim, n_super, R):
    """Forward: Phi[i,j,k,a,b,c] = sum_r lam_r v_r[p2s[i],a] v_r[j,b] v_r[k,c]."""
    ext = v[:, p2s_map, :]  # (R, nat_prim, 3)
    wv = lambdas[:, None, None] * v  # (R, n_super, 3)
    return torch.einsum('ria, rjb, rkc -> ijkabc', ext, wv, v)


def fit_fscp(fc3_raw, phonon, R=24, max_iter=2000, verbose=True):
    """Fit fully symmetric CP: Phi = sum_r lam_r v_r^{otimes 3}."""
    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    p2s = torch.tensor(phonon.primitive.p2s_map.astype(np.int64), dtype=torch.long)

    target_np = _build_target(fc3_raw, phonon)
    target_norm = np.linalg.norm(target_np)
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
        fc3 = _fscp_forward_torch(v_param, lam_param, p2s, nat_prim, n_super, R)
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
        print(f"  FSCP fitting: R={R}, max_iter={max_iter}")

    for it in range(1, adam_iters + 1):
        optimizer.zero_grad()
        loss, fc3_approx = forward()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            v_param.data -= v_param.data.mean(dim=1, keepdim=True)

        if it % 200 == 0 or it == 1:
            with torch.no_grad():
                err = torch.sqrt(torch.sum((fc3_approx - target_t) ** 2)).item()
            update_best(err)
            if verbose:
                print(f"    iter {it:5d}: rel_err={err:.6e}")

    # Phase 2: L-BFGS
    lbfgs_iters = max_iter - adam_iters
    v_param.data.copy_(best_v)
    lam_param.data.copy_(best_lam)

    if verbose:
        print(f"  Phase 2: L-BFGS ({lbfgs_iters} iters)...")

    lbfgs = torch.optim.LBFGS(
        [v_param, lam_param], lr=1.0, max_iter=20,
        history_size=50, line_search_fn='strong_wolfe')

    for outer in range(max(1, lbfgs_iters // 20)):
        def closure():
            lbfgs.zero_grad()
            loss, _ = forward()
            loss.backward()
            if v_param.grad is not None:
                v_param.grad.data = (
                    v_param.grad.data - v_param.grad.data.mean(dim=1, keepdim=True)
                ).contiguous()
            return loss
        lbfgs.step(closure)

        if (outer + 1) % 5 == 0 or outer == 0:
            with torch.no_grad():
                _, fc3_approx = forward()
                err = torch.sqrt(torch.sum((fc3_approx - target_t) ** 2)).item()
            update_best(err)
            if verbose:
                print(f"    L-BFGS step {outer+1:4d}: rel_err={err:.6e}")

    v_modes = best_v.numpy()
    lambdas = best_lam.numpy() * target_norm

    order = np.argsort(-np.abs(lambdas))
    v_modes = v_modes[order]
    lambdas = lambdas[order]

    return v_modes, lambdas, {'rel_err': best_err, 'target_norm': target_norm}


def fscp_reconstruct_M_stacked(v_modes, lambdas, phonon, target_norm, n_dof, dim_sc):
    """Reconstruct M_stacked from FSCP modes."""
    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    p2s = torch.tensor(phonon.primitive.p2s_map.astype(np.int64), dtype=torch.long)
    R = len(lambdas)

    with torch.no_grad():
        v_t = torch.tensor(v_modes, dtype=torch.float64)
        lam_t = torch.tensor(lambdas / target_norm, dtype=torch.float64)
        fc3_mw = _fscp_forward_torch(v_t, lam_t, p2s, nat_prim, n_super, R).numpy() * target_norm

    M = np.zeros((n_dof * dim_sc, dim_sc))
    for i_prim in range(nat_prim):
        for alpha in range(3):
            a = 3 * i_prim + alpha
            block = fc3_mw[i_prim, :, :, alpha, :, :]
            M[a * dim_sc:(a + 1) * dim_sc, :] = block.transpose(0, 2, 1, 3).reshape(dim_sc, dim_sc)
    return M


def fscp_n_params(R, dim_sc):
    return R * (dim_sc + 1)


# =====================================================================
# Main comparison
# =====================================================================

def main():
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    n_dof = 3 * nat_prim
    dim_sc = 3 * n_super
    masses_super = phonon.supercell.masses

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(phonon)

    print(f"System: nat_prim={nat_prim}, n_super={n_super}")
    print(f"  n_dof={n_dof}, dim_sc={dim_sc}")
    print(f"  Full tensor entries: {n_dof * dim_sc**2}")
    print()

    # Build reference M_stacked
    M_stacked = build_realspace_fc3_matrices(fc3_raw, nat_prim, masses_super, ref_sc_atoms)
    M_norm = np.linalg.norm(M_stacked, 'fro')
    print(f"M_stacked shape: {M_stacked.shape}, Frobenius norm: {M_norm:.4e}")

    # Full SVD for rank analysis
    _, S_full, _ = np.linalg.svd(M_stacked, full_matrices=False)
    R_full = len(S_full)
    print(f"Full SVD rank: {R_full}")
    print(f"Singular values: {S_full}")
    print()

    # Full PSCP for rank analysis
    d_full, v_full, norms_full = pscp_decomposition(M_stacked, n_dof, dim_sc)
    R_pscp_full = len(d_full)
    print(f"Full PSCP rank: {R_pscp_full}")
    print()

    # ---- Collect results ----
    results = {
        'SVD': {'ranks': [], 'n_params': [], 'errors': []},
        'PSCP': {'ranks': [], 'n_params': [], 'errors': []},
        'SCP3': {'ranks': [], 'n_params': [], 'errors': []},
        'FSCP': {'ranks': [], 'n_params': [], 'errors': []},
    }

    # -- SVD at various ranks --
    print("=" * 60)
    print("Method 1: Truncated SVD")
    print("=" * 60)
    svd_ranks = sorted(set([1, 2, 3, 4, 6, 8, 12, 16, 24, R_full]))
    svd_ranks = [r for r in svd_ranks if r <= R_full]

    for R in svd_ranks:
        M_approx = svd_approximation(M_stacked, R)
        err = np.linalg.norm(M_stacked - M_approx, 'fro') / M_norm
        n_p = svd_n_params(R, n_dof, dim_sc)
        results['SVD']['ranks'].append(R)
        results['SVD']['n_params'].append(n_p)
        results['SVD']['errors'].append(err)
        print(f"  R={R:3d}: params={n_p:8d}, rel_err={err:.6e}")

    # -- PSCP at various ranks --
    print()
    print("=" * 60)
    print("Method 2: Partially Symmetric CP (PSCP)")
    print("=" * 60)
    pscp_ranks = sorted(set([1, 2, 4, 6, 8, 12, 18, 24, 36, 48, R_pscp_full]))
    pscp_ranks = [r for r in pscp_ranks if r <= R_pscp_full]

    for R in pscp_ranks:
        d_trunc = d_full[:R]
        v_trunc = v_full[:R]
        M_approx = pscp_reconstruct(d_trunc, v_trunc, n_dof, dim_sc)
        err = np.linalg.norm(M_stacked - M_approx, 'fro') / M_norm
        n_p = pscp_n_params(R, n_dof, dim_sc)
        results['PSCP']['ranks'].append(R)
        results['PSCP']['n_params'].append(n_p)
        results['PSCP']['errors'].append(err)
        print(f"  R={R:3d}: params={n_p:8d}, rel_err={err:.6e}")

    # -- SCP3 at various ranks --
    print()
    print("=" * 60)
    print("Method 3: Symmetric CP (3 modes)")
    print("=" * 60)
    scp3_ranks = [2, 4, 8, 16, 24]

    for N_c in scp3_ranks:
        print(f"\n  --- N_c={N_c} ---")
        t0 = time.time()
        u_modes, lambdas, info = fit_supercell_cp(
            fc3_raw, phonon, N_c=N_c, max_iter=2000, verbose=False)
        dt = time.time() - t0

        M_approx = scp3_reconstruct_M_stacked(
            u_modes, lambdas, phonon, info['target_norm'], n_dof, dim_sc)
        err = np.linalg.norm(M_stacked - M_approx, 'fro') / M_norm
        n_p = scp3_n_params(N_c, dim_sc)

        results['SCP3']['ranks'].append(N_c)
        results['SCP3']['n_params'].append(n_p)
        results['SCP3']['errors'].append(err)
        print(f"  N_c={N_c:3d}: params={n_p:8d}, rel_err={err:.6e} ({dt:.1f}s)")

    # -- FSCP at various ranks --
    print()
    print("=" * 60)
    print("Method 4: Fully Symmetric CP (1 mode)")
    print("=" * 60)
    fscp_ranks = [4, 8, 16, 24, 48]

    for R in fscp_ranks:
        print(f"\n  --- R={R} ---")
        t0 = time.time()
        v_modes, lambdas, info = fit_fscp(
            fc3_raw, phonon, R=R, max_iter=2000, verbose=False)
        dt = time.time() - t0

        M_approx = fscp_reconstruct_M_stacked(
            v_modes, lambdas, phonon, info['target_norm'], n_dof, dim_sc)
        err = np.linalg.norm(M_stacked - M_approx, 'fro') / M_norm
        n_p = fscp_n_params(R, dim_sc)

        results['FSCP']['ranks'].append(R)
        results['FSCP']['n_params'].append(n_p)
        results['FSCP']['errors'].append(err)
        print(f"  R={R:3d}: params={n_p:8d}, rel_err={err:.6e} ({dt:.1f}s)")

    # ---- Summary table ----
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<8} {'Rank':>6} {'Params':>10} {'Rel Error':>12}")
    print("-" * 40)
    for method in ['SVD', 'PSCP', 'SCP3', 'FSCP']:
        for i in range(len(results[method]['ranks'])):
            print(f"{method:<8} {results[method]['ranks'][i]:>6} "
                  f"{results[method]['n_params'][i]:>10} "
                  f"{results[method]['errors'][i]:>12.4e}")
        print("-" * 40)

    # ---- Plots ----
    fig_dir = script_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    colors = {'SVD': 'C0', 'PSCP': 'C1', 'SCP3': 'C2', 'FSCP': 'C3'}
    markers = {'SVD': 'o', 'PSCP': 's', 'SCP3': '^', 'FSCP': 'D'}

    # Plot 1: Error vs number of parameters
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['SVD', 'PSCP', 'SCP3', 'FSCP']:
        r = results[method]
        if r['n_params']:
            ax.semilogy(r['n_params'], r['errors'],
                        f'-{markers[method]}', color=colors[method],
                        label=method, markersize=6)
    ax.set_xlabel('Number of parameters')
    ax.set_ylabel('Relative Frobenius error')
    ax.set_title('FC3 approximation: error vs parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fc3_error_vs_params.pdf")
    fig.savefig(fig_dir / "fc3_error_vs_params.png", dpi=150)
    print(f"\nSaved: {fig_dir / 'fc3_error_vs_params.pdf'}")

    # Plot 2: Error vs rank
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['SVD', 'PSCP', 'SCP3', 'FSCP']:
        r = results[method]
        if r['ranks']:
            ax.semilogy(r['ranks'], r['errors'],
                        f'-{markers[method]}', color=colors[method],
                        label=method, markersize=6)
    ax.set_xlabel('Rank R')
    ax.set_ylabel('Relative Frobenius error')
    ax.set_title('FC3 approximation: error vs rank')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fc3_error_vs_rank.pdf")
    fig.savefig(fig_dir / "fc3_error_vs_rank.png", dpi=150)
    print(f"Saved: {fig_dir / 'fc3_error_vs_rank.pdf'}")

    # Plot 3: SVD singular value spectrum
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(np.arange(1, len(S_full) + 1), S_full / S_full[0], 'o-', markersize=4)
    ax.set_xlabel('Singular value index')
    ax.set_ylabel(r'$\sigma_r / \sigma_1$')
    ax.set_title('SVD singular value spectrum of M_stacked')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "svd_spectrum.pdf")
    fig.savefig(fig_dir / "svd_spectrum.png", dpi=150)
    print(f"Saved: {fig_dir / 'svd_spectrum.pdf'}")

    plt.close('all')


if __name__ == "__main__":
    main()
