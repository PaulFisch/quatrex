"""Permanent CP (PCP) decomposition of FC3 tensors.

Implements the tensor learning approach from Luo et al. (2025),
"Tensor Learning and Compression of N-phonon Interactions",
arXiv:2503.05913.

The PCP ansatz for 3-IFCs in real space (Eq. S1):

    Phi~[l1b1, l2b2, l3b3]^{a1,a2,a3}
        = sum_xi (lam_xi/3!) sum_{sigma in S3} sum_l
          prod_{i=1}^3 A_{sigma_i}^xi(l_i - l, b_i, a_i)

where l = primitive cell, b = basis atom, a = Cartesian direction.
A_i^xi(l, b, a) are the PCP modes (3 mode functions per rank xi).

The compressed 3-ph interactions in momentum space (Eq. 2):

    V~(Q1,Q2,Q3) = delta(sum qi)
        sum_xi (lam_xi/3!) sum_{sigma in S3} prod_i A_i^xi(Q_{sigma_i})

where A_i^xi(Q) = sum_{l,b,a} (e^Q_{ab}/sqrt(m_b)) e^{iq.r_l} A_i^xi(l,b,a).
"""

import itertools

import numpy as np
import torch

from .constants import CONVERSION_FC3_THZ, CONVERSION_THZ2, HBAR_SI, HBAR_EV, KB_EV, THZ_TO_RAD
from .anharmonic import (
    _build_device_hamiltonian,
    _compute_obc_self_energies,
    _solve_green_functions,
)

# All 6 permutations of (0, 1, 2)
S3_PERMS = list(itertools.permutations(range(3)))


# ---------------------------------------------------------------------------
# Supercell-to-primitive-cell mapping
# ---------------------------------------------------------------------------


def build_cell_mapping(phonon):
    """Map supercell atoms to (cell_index, prim_atom_index).

    Parameters
    ----------
    phonon : Phonopy

    Returns
    -------
    sc_to_cell : ndarray, shape (n_super,)
        Cell index for each supercell atom.
    sc_to_prim : ndarray, shape (n_super,)
        Primitive atom index for each supercell atom.
    cell_frac : ndarray, shape (n_super, 3)
        Fractional cell coordinates (integer-valued) for each supercell atom.
    n_cells : int
        Number of distinct cells in the supercell.
    sc_matrix : ndarray, shape (3,)
        Diagonal of the supercell matrix (assumed diagonal).
    """
    prim = phonon.primitive
    sc = phonon.supercell
    n_super = len(sc.masses)
    nat_prim = len(prim.masses)
    inv_cell = np.linalg.inv(prim.cell.T)

    sc_to_prim = np.zeros(n_super, dtype=int)
    cell_frac = np.zeros((n_super, 3), dtype=int)

    for s in range(n_super):
        for a in range(nat_prim):
            diff = sc.positions[s] - prim.positions[a]
            frac = inv_cell @ diff
            R_int = np.round(frac).astype(int)
            err = np.linalg.norm(diff - prim.cell.T @ R_int)
            if err < 0.1:
                sc_to_prim[s] = a
                cell_frac[s] = R_int
                break

    # Supercell matrix (diagonal assumed)
    sc_mat = np.diag(phonon.supercell_matrix).astype(int)
    n_cells = int(np.prod(sc_mat))

    def cell_to_idx(t):
        return ((t[0] % sc_mat[0]) * sc_mat[1] + (t[1] % sc_mat[1])) * sc_mat[2] + (t[2] % sc_mat[2])

    sc_to_cell = np.array([cell_to_idx(cell_frac[s]) for s in range(n_super)], dtype=int)

    return sc_to_cell, sc_to_prim, cell_frac, n_cells, sc_mat


def build_cell_diff_table(n_cells, sc_mat):
    """Build cell subtraction table with periodic boundary conditions.

    cell_diff[l1, l2] = index of (cell_l1 - cell_l2) mod supercell.

    Parameters
    ----------
    n_cells : int
    sc_mat : ndarray, shape (3,) — diagonal of supercell matrix

    Returns
    -------
    cell_diff : ndarray, shape (n_cells, n_cells), int
    idx_to_t : ndarray, shape (n_cells, 3), int
        Cell vector (t1,t2,t3) for each cell index.
    """
    n1, n2, n3 = sc_mat

    idx_to_t = np.zeros((n_cells, 3), dtype=int)
    for l in range(n_cells):
        t1 = l // (n2 * n3)
        t2 = (l // n3) % n2
        t3 = l % n3
        idx_to_t[l] = [t1, t2, t3]

    def t_to_idx(t):
        return ((t[0] % n1) * n2 + (t[1] % n2)) * n3 + (t[2] % n3)

    cell_diff = np.zeros((n_cells, n_cells), dtype=int)
    for l1 in range(n_cells):
        for l2 in range(n_cells):
            d = idx_to_t[l1] - idx_to_t[l2]
            cell_diff[l1, l2] = t_to_idx(d)

    return cell_diff, idx_to_t


# ---------------------------------------------------------------------------
# Index tables for advanced indexing
# ---------------------------------------------------------------------------


def _build_index_tables(sc_to_cell, sc_to_prim, cell_diff, nat_prim, n_super, n_cells):
    """Precompute index tables for the PCP forward model.

    Returns
    -------
    idx1_cell : (nat_prim, n_cells) — cell indices for leg 1
    idx_jk_cell : (n_super, n_cells) — cell indices for legs 2,3
    sc_to_prim : (n_super,) — prim atom for each supercell atom
    """
    idx1_cell = np.zeros((nat_prim, n_cells), dtype=np.int64)
    for b1 in range(nat_prim):
        for l in range(n_cells):
            idx1_cell[b1, l] = cell_diff[0, l]

    idx_jk_cell = np.zeros((n_super, n_cells), dtype=np.int64)
    for j in range(n_super):
        for l in range(n_cells):
            idx_jk_cell[j, l] = cell_diff[sc_to_cell[j], l]

    return idx1_cell, idx_jk_cell, sc_to_prim.astype(np.int64)


# ---------------------------------------------------------------------------
# PCP forward model (PyTorch)
# ---------------------------------------------------------------------------


def _pcp_forward_torch(A, lambdas, idx1_cell, idx_jk_cell, sc_to_prim,
                       nat_prim, n_super, n_cells, N_c):
    """Fully vectorized PCP forward model in PyTorch.

    A : (3, N_c, n_cells, nat_prim, 3)
    lambdas : (N_c,)
    Returns fc3 : (nat_prim, n_super, n_super, 3, 3, 3)
    """
    # Build mode tables via advanced indexing
    b1_range = torch.arange(nat_prim, device=A.device)
    mode1 = A[:, :, idx1_cell, b1_range[:, None], :]  # (3, N_c, nat_prim, n_cells, 3)

    mode_jk = A[:, :, idx_jk_cell, sc_to_prim[:, None].expand(-1, n_cells), :]
    # shape: (3, N_c, n_super, n_cells, 3)

    fc3 = torch.zeros(nat_prim, n_super, n_super, 3, 3, 3,
                       dtype=A.dtype, device=A.device)

    w = lambdas / 6.0  # (N_c,)

    for s1, s2, s3 in S3_PERMS:
        m1 = mode1[s1]   # (N_c, nat_prim, n_cells, 3)
        m2 = mode_jk[s2] # (N_c, n_super, n_cells, 3)
        m3 = mode_jk[s3] # (N_c, n_super, n_cells, 3)

        wm1 = w[:, None, None, None] * m1  # (N_c, nat_prim, n_cells, 3)

        # Contract over xi (rank) and l (cell shifts)
        fc3 += torch.einsum('xila, xjlb, xklc -> ijkabc', wm1, m2, m3)

    return fc3


# ---------------------------------------------------------------------------
# ASR projection
# ---------------------------------------------------------------------------


def _project_asr(A):
    """Project modes onto the acoustic sum rule subspace.

    The FC3 acoustic sum rule requires sum_j Phi(i,j,k) = 0 for all i,k.
    In the PCP parameterization, this is equivalent to:

        sum_{l, b} A_i^xi(l, b, alpha) = 0

    for each (leg i, rank xi, Cartesian alpha).

    Proof: summing the PCP forward model over supercell atom j,
    the pair (cell_j, prim_j) covers all (cell, atom) pairs exactly once
    as j varies. The cell shift cell_diff[c_j, l] is a bijection on cells
    for each fixed l. Therefore the sum collapses to
    sum_{l',b'} A_{s2}[l', b', alpha], which this projection zeroes out.

    The same argument holds for summing over i or k (all three legs),
    since the permanent symmetrization assigns each mode to each leg.

    Parameters
    ----------
    A : Tensor, shape (3, N_c, n_cells, nat_prim, 3)

    Returns
    -------
    A_proj : same shape, satisfying ASR exactly.
    """
    mean = A.mean(dim=(2, 3), keepdim=True)  # (3, N_c, 1, 1, 3)
    return A - mean


def _project_asr_grad(grad_A):
    """Project gradients onto ASR-tangent subspace.

    For projected gradient methods, the gradient must also lie in the
    constraint tangent space. The ASR constraint is linear
    (sum_{l,b} A = 0), so its tangent space is the same as the
    constraint subspace: vectors with zero mean over (l, b).

    Returns a contiguous tensor (required by L-BFGS which calls .view()).
    """
    return (grad_A - grad_A.mean(dim=(2, 3), keepdim=True)).contiguous()


# ---------------------------------------------------------------------------
# PCP fitting via torch (Adam + L-BFGS, ASR projection)
# ---------------------------------------------------------------------------


def _build_target(fc3_raw, phonon):
    """Build mass-weighted target FC3 in compact format.

    Returns
    -------
    target : ndarray, shape (nat_prim, n_super, n_super, 3, 3, 3)
    """
    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    masses = phonon.supercell.masses
    p2s = phonon.primitive.p2s_map
    is_compact = fc3_raw.shape[0] == nat_prim

    target = np.zeros((nat_prim, n_super, n_super, 3, 3, 3))
    for i_prim in range(nat_prim):
        fc3_idx = i_prim if is_compact else int(p2s[i_prim])
        m_i = masses[int(p2s[i_prim])]
        for j in range(n_super):
            m_j = masses[j]
            for k in range(n_super):
                m_k = masses[k]
                mass_factor = np.sqrt(m_i * m_j * m_k)
                target[i_prim, j, k] = fc3_raw[fc3_idx, j, k] / mass_factor * CONVERSION_FC3_THZ

    return target


def fit_pcp(fc3_raw, phonon, N_c=24, max_iter=2000, verbose=True):
    """Fit PCP decomposition to FC3 using ASR-projected optimization.

    Two-phase optimization with ASR enforced by projection (not penalty):
      Phase 1: Adam with cosine warm restarts and ASR projection after
               each step (Adam's momentum is robust to mild projection).
      Phase 2: L-BFGS operating in ASR-projected gradient space.
               Gradients are projected onto the ASR tangent plane BEFORE
               L-BFGS sees them, so the Hessian approximation stays valid
               within the constraint manifold.

    Parameters
    ----------
    fc3_raw : ndarray
        FC3 in compact (nat_prim, n_super, n_super, 3, 3, 3) or
        full (n_super, n_super, n_super, 3, 3, 3) format.
    phonon : Phonopy
    N_c : int
        PCP rank.
    max_iter : int
        Total optimization iterations (Adam + L-BFGS).
    verbose : bool

    Returns
    -------
    A_modes : ndarray, shape (3, N_c, n_cells, nat_prim, 3)
    lambdas : ndarray, shape (N_c,)
    info : dict
    """
    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)

    sc_to_cell, sc_to_prim_np, cell_frac_all, n_cells, sc_mat = build_cell_mapping(phonon)
    cell_diff, idx_to_t = build_cell_diff_table(n_cells, sc_mat)

    # Build target
    target_np = _build_target(fc3_raw, phonon)
    target_norm = np.linalg.norm(target_np)

    if verbose:
        print(f"  PCP fitting: N_c={N_c}, target norm={target_norm:.4e}")
        print(f"  Supercell: {n_super} atoms, {n_cells} cells, {nat_prim} prim atoms")
        n_params = 3 * N_c * n_cells * nat_prim * 3 + N_c
        n_target = nat_prim * n_super * n_super * 27
        print(f"  Parameters: {n_params}, target entries: {n_target}")

    # Normalize target for numerical stability
    target_t = torch.tensor(target_np / target_norm, dtype=torch.float64)

    # Precompute index tables
    idx1_cell_np, idx_jk_cell_np, sc_to_prim_i = _build_index_tables(
        sc_to_cell, sc_to_prim_np, cell_diff, nat_prim, n_super, n_cells)
    idx1_cell = torch.tensor(idx1_cell_np, dtype=torch.long)
    idx_jk_cell = torch.tensor(idx_jk_cell_np, dtype=torch.long)
    sc_to_prim_t = torch.tensor(sc_to_prim_i, dtype=torch.long)

    # --- Initialization ---
    # Scale so initial FC3 magnitude ~ O(1) (matching normalized target).
    # FC3 ~ N_c * lam * n_cells * A^3, with lam=1: A ~ (1/(N_c * n_cells))^{1/3}
    rng = np.random.default_rng(42)
    scale = (1.0 / (N_c * n_cells)) ** (1.0 / 3.0)

    A_init = rng.normal(0, scale, (3, N_c, n_cells, nat_prim, 3))
    A_init -= A_init.mean(axis=(2, 3), keepdims=True)  # enforce ASR from start

    A_param = torch.tensor(A_init, dtype=torch.float64, requires_grad=True)
    lam_param = torch.tensor(
        np.ones(N_c, dtype=np.float64),
        requires_grad=True,
    )

    best_err = float('inf')
    best_A = A_param.detach().clone()
    best_lam = lam_param.detach().clone()

    def forward():
        fc3_approx = _pcp_forward_torch(
            A_param, lam_param, idx1_cell, idx_jk_cell, sc_to_prim_t,
            nat_prim, n_super, n_cells, N_c,
        )
        return torch.sum((fc3_approx - target_t) ** 2), fc3_approx

    def update_best(err_val):
        nonlocal best_err, best_A, best_lam
        if err_val < best_err:
            best_err = err_val
            best_A = A_param.detach().clone()
            best_lam = lam_param.detach().clone()

    # --- Phase 1: Adam with ASR projection ---
    adam_iters = min(max_iter * 3 // 4, 1500)
    optimizer = torch.optim.Adam([A_param, lam_param], lr=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-4,
    )

    if verbose:
        print(f"  Phase 1: Adam ({adam_iters} iters, lr=0.02, cosine warm restarts)...")

    for it in range(1, adam_iters + 1):
        optimizer.zero_grad()
        loss, fc3_approx = forward()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Project modes back onto ASR subspace after each Adam step.
        # Adam's per-parameter momentum is tolerant of this mild projection.
        with torch.no_grad():
            A_param.data = _project_asr(A_param.data)

        if it % 100 == 0 or it == 1:
            with torch.no_grad():
                err = torch.sqrt(torch.sum((fc3_approx - target_t) ** 2)).item()
                asr_v = torch.max(torch.abs(A_param.sum(dim=(2, 3)))).item()
            update_best(err)
            if verbose:
                print(f"    iter {it:5d}: rel_err={err:.6e}, "
                      f"max_asr={asr_v:.2e}, lr={scheduler.get_last_lr()[0]:.2e}")

    # --- Phase 2: L-BFGS with projected gradients ---
    # Instead of projecting parameters post-step (which corrupts the
    # L-BFGS Hessian approximation), we project GRADIENTS onto the ASR
    # tangent plane inside the closure. This way L-BFGS builds a valid
    # inverse-Hessian approximation within the constraint manifold.
    lbfgs_iters = max_iter - adam_iters
    if verbose:
        print(f"  Phase 2: L-BFGS ({lbfgs_iters} iters, projected gradients)...")

    # Reset from best
    A_param.data.copy_(best_A)
    lam_param.data.copy_(best_lam)

    lbfgs = torch.optim.LBFGS(
        [A_param, lam_param],
        lr=1.0,
        max_iter=20,
        history_size=50,
        line_search_fn='strong_wolfe',
    )

    n_lbfgs_steps = max(1, lbfgs_iters // 20)
    for outer in range(n_lbfgs_steps):
        def closure():
            lbfgs.zero_grad()
            loss, _ = forward()
            loss.backward()
            # Project A gradient onto ASR tangent plane so L-BFGS
            # only explores the constraint manifold.
            if A_param.grad is not None:
                A_param.grad.data = _project_asr_grad(A_param.grad.data)
            return loss

        lbfgs.step(closure)

        if (outer + 1) % 5 == 0 or outer == 0:
            with torch.no_grad():
                loss_val, fc3_approx = forward()
                err = torch.sqrt(torch.sum((fc3_approx - target_t) ** 2)).item()
                asr_v = torch.max(torch.abs(A_param.sum(dim=(2, 3)))).item()
            update_best(err)
            if verbose:
                print(f"    L-BFGS step {outer+1:4d}: rel_err={err:.6e}, "
                      f"max_asr={asr_v:.2e}")

    # Use best parameters
    A_modes = best_A.numpy()
    lambdas = best_lam.numpy()

    # Rescale lambdas to undo target normalization
    lambdas *= target_norm

    # Sort by |lambda| descending
    order = np.argsort(-np.abs(lambdas))
    A_modes = A_modes[:, order]
    lambdas = lambdas[order]
    asr_violation = np.max(np.abs(np.sum(A_modes, axis=(2, 3))))

    # Verify ASR of reconstructed FC3
    with torch.no_grad():
        A_t = torch.tensor(A_modes, dtype=torch.float64)
        lam_t = torch.tensor(lambdas / target_norm, dtype=torch.float64)
        fc3_final = _pcp_forward_torch(
            A_t, lam_t, idx1_cell, idx_jk_cell, sc_to_prim_t,
            nat_prim, n_super, n_cells, N_c,
        )
        fc3_recon = fc3_final.numpy() * target_norm
    asr_fc3 = np.max(np.abs(np.sum(fc3_recon, axis=1)))  # sum over j

    info = {
        'rel_err': best_err,
        'asr_violation': asr_violation,
        'asr_fc3': asr_fc3,
        'n_iter': max_iter,
        'target_norm': target_norm,
        'n_cells': n_cells,
        'sc_mat': sc_mat,
        'sc_to_cell': sc_to_cell,
        'sc_to_prim': sc_to_prim_np,
        'cell_diff': cell_diff,
        'idx_to_t': idx_to_t,
    }

    if verbose:
        print(f"  PCP fit: rel_err={best_err:.6e}, asr_mode={asr_violation:.2e}, "
              f"asr_fc3={asr_fc3:.2e}")

    return A_modes, lambdas, info


def fit_pcp_greedy(fc3_raw, phonon, N_c=24, iters_per_rank=500, verbose=True):
    """Fit PCP decomposition greedily, one rank at a time.

    For each new rank, fit the residual (target minus current approximation).
    This avoids the difficult joint optimization over all ranks simultaneously
    and guarantees monotonic error decrease.

    Parameters
    ----------
    fc3_raw : ndarray
        FC3 tensor.
    phonon : Phonopy
    N_c : int
        Total PCP rank.
    iters_per_rank : int
        Adam iterations per rank.
    verbose : bool

    Returns
    -------
    A_modes : ndarray, shape (3, N_c, n_cells, nat_prim, 3)
    lambdas : ndarray, shape (N_c,)
    info : dict
    """
    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)

    sc_to_cell, sc_to_prim_np, cell_frac_all, n_cells, sc_mat = build_cell_mapping(phonon)
    cell_diff, idx_to_t = build_cell_diff_table(n_cells, sc_mat)

    target_np = _build_target(fc3_raw, phonon)
    target_norm = np.linalg.norm(target_np)
    target_t = torch.tensor(target_np / target_norm, dtype=torch.float64)

    idx1_cell_np, idx_jk_cell_np, sc_to_prim_i = _build_index_tables(
        sc_to_cell, sc_to_prim_np, cell_diff, nat_prim, n_super, n_cells)
    idx1_cell = torch.tensor(idx1_cell_np, dtype=torch.long)
    idx_jk_cell = torch.tensor(idx_jk_cell_np, dtype=torch.long)
    sc_to_prim_t = torch.tensor(sc_to_prim_i, dtype=torch.long)

    if verbose:
        print(f"  PCP greedy fitting: N_c={N_c}, target norm={target_norm:.4e}")
        print(f"  Supercell: {n_super} atoms, {n_cells} cells, {nat_prim} prim atoms")

    # Accumulate fitted modes
    all_A = []
    all_lam = []
    residual = target_t.clone()

    rng = np.random.default_rng(42)

    for rank_idx in range(N_c):
        residual_norm = torch.sqrt(torch.sum(residual ** 2)).item()
        if verbose and (rank_idx % max(1, N_c // 10) == 0 or rank_idx == 0):
            print(f"  Rank {rank_idx+1}/{N_c}: residual_norm={residual_norm:.6e}")

        if residual_norm < 1e-12:
            if verbose:
                print(f"  Converged at rank {rank_idx} (residual < 1e-12)")
            break

        # Initialize a single-rank PCP for the residual.
        # Try multiple random starts and keep the best (rank-1 landscape
        # is highly non-convex, so restarts are essential).
        n_restarts = 3
        best_loss = float('inf')
        best_A_r = None
        best_lam_r = None

        for restart in range(n_restarts):
            scale = (residual_norm / n_cells) ** (1.0 / 3.0)
            A_init = rng.normal(0, scale, (3, 1, n_cells, nat_prim, 3))
            A_init -= A_init.mean(axis=(2, 3), keepdims=True)

            A_r = torch.tensor(A_init, dtype=torch.float64, requires_grad=True)
            lam_r = torch.tensor([residual_norm], dtype=torch.float64, requires_grad=True)

            # Adam phase
            adam_its = iters_per_rank * 2 // 3
            opt = torch.optim.Adam([A_r, lam_r], lr=0.03)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=adam_its, eta_min=1e-4,
            )

            for it in range(adam_its):
                opt.zero_grad()
                fc3_r = _pcp_forward_torch(
                    A_r, lam_r, idx1_cell, idx_jk_cell, sc_to_prim_t,
                    nat_prim, n_super, n_cells, 1,
                )
                loss = torch.sum((fc3_r - residual) ** 2)
                loss.backward()
                opt.step()
                sched.step()
                with torch.no_grad():
                    A_r.data = _project_asr(A_r.data)

            # L-BFGS refinement
            lbfgs_r = torch.optim.LBFGS(
                [A_r, lam_r], lr=1.0, max_iter=20,
                history_size=20, line_search_fn='strong_wolfe',
            )
            for _ in range(iters_per_rank // 3 // 20 + 1):
                def closure_r():
                    lbfgs_r.zero_grad()
                    fc3_r = _pcp_forward_torch(
                        A_r, lam_r, idx1_cell, idx_jk_cell, sc_to_prim_t,
                        nat_prim, n_super, n_cells, 1,
                    )
                    l = torch.sum((fc3_r - residual) ** 2)
                    l.backward()
                    if A_r.grad is not None:
                        A_r.grad.data = _project_asr_grad(A_r.grad.data)
                    return l
                lbfgs_r.step(closure_r)

            with torch.no_grad():
                fc3_r = _pcp_forward_torch(
                    A_r, lam_r, idx1_cell, idx_jk_cell, sc_to_prim_t,
                    nat_prim, n_super, n_cells, 1,
                )
                final_loss = torch.sum((fc3_r - residual) ** 2).item()

            if final_loss < best_loss:
                best_loss = final_loss
                best_A_r = A_r.detach().clone()
                best_lam_r = lam_r.detach().clone()

        # Only subtract if we actually reduced the residual
        with torch.no_grad():
            fc3_fitted = _pcp_forward_torch(
                best_A_r, best_lam_r, idx1_cell, idx_jk_cell, sc_to_prim_t,
                nat_prim, n_super, n_cells, 1,
            )
            new_residual = residual - fc3_fitted
            new_norm = torch.sqrt(torch.sum(new_residual ** 2)).item()
            if new_norm < residual_norm:
                residual = new_residual
            else:
                if verbose:
                    print(f"    Rank {rank_idx+1}: no improvement, stopping greedy phase")
                break

        all_A.append(best_A_r.numpy())
        all_lam.append(best_lam_r.numpy()[0])

    # Assemble
    actual_rank = len(all_lam)
    A_modes = np.zeros((3, N_c, n_cells, nat_prim, 3))
    lambdas = np.zeros(N_c)
    for r in range(actual_rank):
        A_modes[:, r] = all_A[r][:, 0]
        lambdas[r] = all_lam[r]

    # Rescale lambdas
    lambdas *= target_norm

    # Sort by |lambda| descending
    order = np.argsort(-np.abs(lambdas))
    A_modes = A_modes[:, order]
    lambdas = lambdas[order]
    asr_violation = np.max(np.abs(np.sum(A_modes, axis=(2, 3))))

    # --- Joint refinement of all ranks ---
    if verbose:
        print(f"  Joint refinement (L-BFGS, {min(200, iters_per_rank)} steps)...")

    A_param = torch.tensor(A_modes, dtype=torch.float64, requires_grad=True)
    lam_param = torch.tensor(lambdas / target_norm, dtype=torch.float64, requires_grad=True)

    lbfgs = torch.optim.LBFGS(
        [A_param, lam_param], lr=1.0, max_iter=20,
        history_size=50, line_search_fn='strong_wolfe',
    )

    n_refine = min(200, iters_per_rank) // 20
    for outer in range(max(1, n_refine)):
        def closure():
            lbfgs.zero_grad()
            fc3_a = _pcp_forward_torch(
                A_param, lam_param, idx1_cell, idx_jk_cell, sc_to_prim_t,
                nat_prim, n_super, n_cells, N_c,
            )
            loss = torch.sum((fc3_a - target_t) ** 2)
            loss.backward()
            if A_param.grad is not None:
                A_param.grad.data = _project_asr_grad(A_param.grad.data)
            return loss
        lbfgs.step(closure)

    A_modes = A_param.detach().numpy()
    lambdas = lam_param.detach().numpy() * target_norm

    # Sort by |lambda| descending
    order = np.argsort(-np.abs(lambdas))
    A_modes = A_modes[:, order]
    lambdas = lambdas[order]
    asr_violation = np.max(np.abs(np.sum(A_modes, axis=(2, 3))))

    # Compute final reconstruction error
    with torch.no_grad():
        A_t = torch.tensor(A_modes, dtype=torch.float64)
        lam_t = torch.tensor(lambdas / target_norm, dtype=torch.float64)
        fc3_final = _pcp_forward_torch(
            A_t, lam_t, idx1_cell, idx_jk_cell, sc_to_prim_t,
            nat_prim, n_super, n_cells, N_c,
        )
        final_err = torch.sqrt(torch.sum((fc3_final - target_t) ** 2)).item()

    # Verify ASR of reconstructed FC3
    fc3_recon = fc3_final.numpy() * target_norm
    asr_fc3 = np.max(np.abs(np.sum(fc3_recon, axis=1)))  # sum over j
    if verbose:
        print(f"  FC3 ASR check: max|sum_j Phi(i,j,k)| = {asr_fc3:.2e}")

    info = {
        'rel_err': final_err,
        'asr_violation': asr_violation,
        'asr_fc3': asr_fc3,
        'n_iter': iters_per_rank * actual_rank,
        'actual_rank': actual_rank,
        'target_norm': target_norm,
        'n_cells': n_cells,
        'sc_mat': sc_mat,
        'sc_to_cell': sc_to_cell,
        'sc_to_prim': sc_to_prim_np,
        'cell_diff': cell_diff,
        'idx_to_t': idx_to_t,
    }

    if verbose:
        print(f"  PCP greedy fit: rel_err={final_err:.6e}, asr={asr_violation:.2e}, "
              f"actual_rank={actual_rank}")

    return A_modes, lambdas, info


# ---------------------------------------------------------------------------
# Fourier transform of PCP modes
# ---------------------------------------------------------------------------


def fourier_transform_pcp(A_modes, lambdas, phonon, q_perp_frac,
                           transport_direction='x', info=None):
    """Fourier-transform PCP modes to q-space.

    f_i^xi(q)[kappa*3+alpha] = sum_l exp(-2pi*i*q.R_l) A_i^xi(l, kappa, alpha)

    Note: mass weighting is already in the modes (from fitting against
    mass-weighted target). No additional 1/sqrt(m) is needed.

    Parameters
    ----------
    A_modes : ndarray, shape (3, N_c, n_cells, nat_prim, 3)
    lambdas : ndarray, shape (N_c,)
    phonon : Phonopy
    q_perp_frac : tuple (qx, qy) in fractional coords
    transport_direction : str
    info : dict from fit_pcp (contains cell mapping)

    Returns
    -------
    f_q : ndarray, shape (3, N_c, n_dof), complex
    """
    prim = phonon.primitive
    nat_prim = len(prim.masses)
    n_dof = nat_prim * 3
    N_c = len(lambdas)
    tidx = "xyz".index(transport_direction)
    perp_idx = [i for i in range(3) if i != tidx]

    q_full = np.zeros(3)
    q_full[perp_idx[0]] = q_perp_frac[0]
    q_full[perp_idx[1]] = q_perp_frac[1]

    if info is not None:
        idx_to_t = info['idx_to_t']
        n_cells = info['n_cells']
    else:
        sc_mat = np.diag(phonon.supercell_matrix).astype(int)
        n_cells = int(np.prod(sc_mat))
        _, idx_to_t = build_cell_diff_table(n_cells, sc_mat)

    # Phase for each cell: exp(-2pi*i * q . R_frac_cell)
    phases = np.exp(-2j * np.pi * idx_to_t @ q_full)  # (n_cells,)

    # FT: f_i^xi(q)[kappa*3+alpha] = sum_l phase[l] * A_i^xi(l, kappa, alpha)
    A_ft = np.sum(A_modes * phases[None, None, :, None, None], axis=2)  # (3, N_c, nat_prim, 3)
    f_q = A_ft.reshape(3, N_c, n_dof)  # (3, N_c, n_dof)

    return f_q


# ---------------------------------------------------------------------------
# PCP self-energy kernel
# ---------------------------------------------------------------------------


def _compute_phph_self_energy_pcp(
    G_lesser_q, G_greater_q,
    f_modes_all_q, lambdas,
    n_dof, n_kpts,
    omega_grid_thz, dw_thz,
    q_diff_map=None,
):
    """Compute phonon-phonon self-energy via PCP decomposition.

    WARNING: This kernel uses the PCP momentum-space factorization
    V = sum f(q_ext)*f(q')*f(q-q'), which requires q*N_supercell to be
    integer-valued. For q-meshes finer than the supercell grid, the
    factorization identity exp(-iq*R_{(t+l) mod N}) = exp(-iq*(R_t + R_l))
    breaks, giving ~50-80% vertex errors at non-commensurate q-points.
    Use pcp_anharmonic_transmission (which reconstructs FC3 and uses the
    dense FT kernel) instead.

    Projects G onto the PCP mode vectors to obtain scalar functions
    g_{ij}^{xi,xi'}(w;q), convolves pairs of scalars, and reconstructs
    the matrix self-energy from the external mode vectors.

    Uses matmul accumulation (Sigma[w] = F^H @ C[w] @ conj(F)) for
    ~5-15x speedup over the naive einsum on the accumulation step.

    Parameters
    ----------
    G_lesser_q, G_greater_q : ndarray, shape (n_kpts, n_freq, n_dof, n_dof)
    f_modes_all_q : ndarray, shape (n_kpts, 3, N_c, n_dof), complex
    lambdas : ndarray, shape (N_c,)
    n_dof, n_kpts : int
    omega_grid_thz : ndarray, shape (n_freq,)
    dw_thz : float
    q_diff_map : ndarray, shape (n_kpts, n_kpts), optional

    Returns
    -------
    Sigma_lesser, Sigma_greater, Sigma_retarded :
        ndarray, shape (n_kpts, n_freq, n_dof, n_dof)
    """
    n_freq = len(omega_grid_thz)
    N_c = len(lambdas)

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

    # Precompute scalar projections g[xi, xi', i, j, q, w]
    gL = np.zeros((N_c, N_c, 3, 3, n_kpts, n_fft), dtype=complex)
    gG = np.zeros_like(gL)

    for iq in range(n_kpts):
        f_flat = f_modes_all_q[iq].reshape(3 * N_c, n_dof)
        f_conj = np.conj(f_flat)

        Gf_L = GL[iq] @ f_flat.T
        Gf_G = GG[iq] @ f_flat.T

        g_flat_L = np.einsum('ia,waj->ijw', f_conj, Gf_L)
        g_flat_G = np.einsum('ia,waj->ijw', f_conj, Gf_G)

        gL[:, :, :, :, iq, :] = g_flat_L.reshape(3, N_c, 3, N_c, n_fft).transpose(1, 3, 0, 2, 4)
        gG[:, :, :, :, iq, :] = g_flat_G.reshape(3, N_c, 3, N_c, n_fft).transpose(1, 3, 0, 2, 4)

    gL_hat = np.fft.fft(gL, axis=-1)
    gG_hat = np.fft.fft(gG, axis=-1)

    Sigma_lesser = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
    Sigma_greater = np.zeros_like(Sigma_lesser)

    lam_outer = np.outer(lambdas, lambdas) / 36.0

    for iq_ext in range(n_kpts):
        for iq_prime in range(n_kpts):
            iq_diff = q_diff_map[iq_ext, iq_prime] if q_diff_map is not None else (iq_ext - iq_prime) % n_kpts

            gLp = gL_hat[:, :, :, :, iq_prime, :]
            gLd = gL_hat[:, :, :, :, iq_diff, :]
            gGp = gG_hat[:, :, :, :, iq_prime, :]
            gGd = gG_hat[:, :, :, :, iq_diff, :]

            # Sum products grouped by (s1, s1p)
            sum_prod_L = np.zeros((N_c, N_c, 3, 3, n_fft), dtype=complex)
            sum_prod_G = np.zeros_like(sum_prod_L)
            for s1, s2, s3 in S3_PERMS:
                for s1p, s2p, s3p in S3_PERMS:
                    sum_prod_L[:, :, s1, s1p, :] += gLp[:, :, s2, s2p, :] * gLd[:, :, s3, s3p, :]
                    sum_prod_G[:, :, s1, s1p, :] += gGp[:, :, s2, s2p, :] * gGd[:, :, s3, s3p, :]

            conv_all_L = np.fft.ifft(sum_prod_L, axis=-1)[:, :, :, :, freq_sl]
            conv_all_G = np.fft.ifft(sum_prod_G, axis=-1)[:, :, :, :, freq_sl]

            # Matmul accumulation: Sigma[w] = F^H @ C[w] @ conj(F)
            for s1 in range(3):
                f_s1 = f_modes_all_q[iq_ext, s1, :, :]    # (N_c, n_dof)
                for s1p in range(3):
                    f_s1p = f_modes_all_q[iq_ext, s1p, :, :]
                    f_s1p_conj = np.conj(f_s1p)

                    CL = (lam_outer[:, :, None] * conv_all_L[:, :, s1, s1p, :]).transpose(2, 0, 1)
                    CG = (lam_outer[:, :, None] * conv_all_G[:, :, s1, s1p, :]).transpose(2, 0, 1)

                    TL = CL @ f_s1p_conj   # (n_freq, N_c, n_dof)
                    TG = CG @ f_s1p_conj

                    Sigma_lesser[iq_ext] += prefactor * np.einsum('xa,wxy->way', f_s1, TL)
                    Sigma_greater[iq_ext] += prefactor * np.einsum('xa,wxy->way', f_s1, TG)

    Sigma_retarded = 0.5 * (Sigma_greater - Sigma_lesser)
    return Sigma_lesser, Sigma_greater, Sigma_retarded


# ---------------------------------------------------------------------------
# FC3 reconstruction from PCP modes
# ---------------------------------------------------------------------------


def reconstruct_fc3_from_pcp(A_modes, lambdas, phonon, pcp_info):
    """Reconstruct raw FC3 from fitted PCP modes.

    Runs the PCP forward model to obtain the mass-weighted compact FC3,
    then un-mass-weights to recover a raw-like FC3 in compact format
    (nat_prim, n_super, n_super, 3, 3, 3).

    Parameters
    ----------
    A_modes : ndarray, shape (3, N_c, n_cells, nat_prim, 3)
    lambdas : ndarray, shape (N_c,)
    phonon : Phonopy
    pcp_info : dict
        From fit_pcp / fit_pcp_greedy. Must contain 'target_norm'.

    Returns
    -------
    fc3_recon : ndarray, shape (nat_prim, n_super, n_super, 3, 3, 3)
        Un-mass-weighted FC3 in compact format, suitable for
        build_realspace_fc3_matrices.
    """
    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    masses = phonon.supercell.masses
    p2s = phonon.primitive.p2s_map

    sc_to_cell, sc_to_prim_np, _, n_cells, sc_mat = build_cell_mapping(phonon)
    cell_diff, _ = build_cell_diff_table(n_cells, sc_mat)
    idx1_cell, idx_jk_cell, sc_to_prim_i = _build_index_tables(
        sc_to_cell, sc_to_prim_np, cell_diff, nat_prim, n_super, n_cells)

    N_c = len(lambdas)
    target_norm = pcp_info['target_norm']

    with torch.no_grad():
        A_t = torch.tensor(A_modes, dtype=torch.float64)
        lam_t = torch.tensor(lambdas / target_norm, dtype=torch.float64)
        fc3_mw = _pcp_forward_torch(
            A_t, lam_t,
            torch.tensor(idx1_cell, dtype=torch.long),
            torch.tensor(idx_jk_cell, dtype=torch.long),
            torch.tensor(sc_to_prim_i, dtype=torch.long),
            nat_prim, n_super, n_cells, N_c,
        ).numpy() * target_norm

    # Un-mass-weight: fc3_mw = fc3_raw / sqrt(m_i * m_j * m_k) * CONVERSION
    # => fc3_raw = fc3_mw * sqrt(m_i * m_j * m_k) / CONVERSION
    fc3_recon = np.zeros_like(fc3_mw)
    for i_prim in range(nat_prim):
        m_i = masses[int(p2s[i_prim])]
        mass_jk = np.sqrt(m_i * masses[:, None] * masses[None, :])
        fc3_recon[i_prim] = fc3_mw[i_prim] * mass_jk[:, :, None, None, None] / CONVERSION_FC3_THZ

    return fc3_recon


# ---------------------------------------------------------------------------
# SCBA transport driver
# ---------------------------------------------------------------------------


def pcp_anharmonic_transmission(
    phonon,
    fc3_hdf5: str,
    q_mesh_transverse: tuple[int, int] = (4, 4),
    freq_range_thz: tuple[float, float, int] = (0.01, 16.0, 101),
    transport_direction: str = "x",
    eta_factor: float = 0.5,
    temperature: float = 300.0,
    delta_T: float = 10.0,
    max_scba_iter: int = 10,
    scba_tol: float = 0.01,
    mixing: float = 0.5,
    n_slabs: int = 1,
    pcp_rank: int = 24,
    pcp_max_iter: int = 2000,
    greedy: bool = False,
    verbose: bool = True,
) -> dict:
    """Anharmonic phonon transmission via PCP FC3 decomposition.

    Fits the PCP decomposition to FC3, reconstructs FC3 in real space,
    then uses the dense q-dependent self-energy kernel for the SCBA loop.

    Note: The PCP momentum-space factorization V = sum f(q_ext)*f(q')*f(q-q')
    is only valid when q*N_supercell is integer-valued (q commensurate with
    the supercell grid). For general q-meshes, the factorization identity
    exp(-iq*R_{(t+l) mod N}) = exp(-iq*(R_t + R_l)) breaks down.
    This function therefore reconstructs FC3 in real space and uses the
    correct dense Fourier transform for arbitrary q-points.

    Parameters
    ----------
    phonon : Phonopy
    fc3_hdf5 : str or Path
        Path to fc3.hdf5.
    pcp_rank : int
        PCP rank N_c.
    pcp_max_iter : int
        Max iterations for PCP fitting.
    greedy : bool
        Use greedy rank-1 fitting instead of joint optimization.

    Returns
    -------
    result : dict
    """
    import h5py
    from .convention import get_btd_blocks
    from .validation import _ballistic_transmission
    from .separable import (
        build_supercell_mapping,
        build_realspace_fc3_matrices,
        build_gathering_matrix,
        build_q_diff_map,
    )
    from .anharmonic import _compute_phph_self_energy_q_dense

    # --- Setup ---
    fmin, fmax, nfreq = freq_range_thz
    nfreq = int(nfreq)
    freqs_thz = np.linspace(fmin, fmax, nfreq)
    omega_sq_thz2 = freqs_thz ** 2
    dw_thz = freqs_thz[1] - freqs_thz[0]
    eta = dw_thz ** 2 * eta_factor

    n_atoms = len(phonon.primitive.masses)
    n_dof = 3 * n_atoms
    N_D = n_slabs * n_dof

    # --- Load FC3 and fit PCP ---
    with h5py.File(fc3_hdf5, "r") as f:
        fc3_raw = np.array(f["fc3"])

    if verbose:
        print(f"  FC3 raw shape: {fc3_raw.shape}")

    if greedy:
        A_modes, lambdas, pcp_info = fit_pcp_greedy(
            fc3_raw, phonon, N_c=pcp_rank,
            iters_per_rank=pcp_max_iter // pcp_rank,
            verbose=verbose,
        )
    else:
        A_modes, lambdas, pcp_info = fit_pcp(
            fc3_raw, phonon, N_c=pcp_rank,
            max_iter=pcp_max_iter, verbose=verbose,
        )

    if verbose:
        print(f"  PCP rank: {pcp_rank}, rel_err: {pcp_info['rel_err']:.4e}")

    # --- Reconstruct FC3 from PCP and build dense self-energy infrastructure ---
    fc3_recon = reconstruct_fc3_from_pcp(A_modes, lambdas, phonon, pcp_info)

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(
        phonon, transport_direction
    )
    masses_super = phonon.supercell.masses
    n_super = len(masses_super)
    dim_sc = n_super * 3

    M_stacked = build_realspace_fc3_matrices(
        fc3_recon, n_atoms, masses_super, ref_sc_atoms
    )

    if verbose:
        print(f"  Reconstructed FC3 -> M_stacked norm: {np.linalg.norm(M_stacked):.4e}")
        print(f"  Supercell atoms: {n_super}, dim_sc: {dim_sc}")
        print(f"  Device: {n_slabs} slab(s), {N_D} DOFs per q-point")

    # --- q-mesh ---
    nkx, nky = q_mesh_transverse
    q_1d_x = [i / nkx for i in range(nkx)]
    q_1d_y = [j / nky for j in range(nky)]
    q_points = [(qx, qy) for qx in q_1d_x for qy in q_1d_y]
    n_kpts = len(q_points)
    q_diff_map = build_q_diff_map(nkx, nky)

    # Build gathering matrices T(q) for each q-point
    T_all_q = []
    for qx, qy in q_points:
        T = build_gathering_matrix(
            prim_indices, cell_frac,
            (qx, qy), n_atoms, transport_direction,
        )
        T_all_q.append(T)

    # --- Bose-Einstein ---
    def bose_einstein(freq_thz_arr, T):
        hw = HBAR_EV * np.abs(freq_thz_arr) * THZ_TO_RAD
        x = hw / (KB_EV * T)
        n = np.zeros_like(x)
        valid = x > 1e-10
        n[valid] = 1.0 / (np.exp(x[valid]) - 1.0)
        return n

    T_L = temperature + delta_T / 2.0
    T_R = temperature - delta_T / 2.0
    n_bose_L = bose_einstein(freqs_thz, T_L)
    n_bose_R = bose_einstein(freqs_thz, T_R)

    if verbose:
        print(f"  q-mesh: {nkx}x{nky} = {n_kpts} (Gamma-centered)")
        print(f"  Frequency grid: {nfreq} points, {fmin:.2f} to {fmax:.2f} THz")
        print(f"  Temperature: {temperature} K, delta_T: {delta_T} K")
        print(f"  eta = {eta:.4e} THz^2")
        print(f"  SCBA: max {max_scba_iter} iter, tol={scba_tol}, mix={mixing}")

    # --- BTD blocks per q-point ---
    btd_blocks = []
    for qx, qy in q_points:
        H_00, H_01 = get_btd_blocks(
            phonon, (qx, qy), transport_direction=transport_direction,
            conversion_factor=CONVERSION_THZ2,
        )
        btd_blocks.append((H_00, H_01))

    # --- Ballistic transmission ---
    trans_ballistic = np.zeros(nfreq)
    for iq, (H_00, H_01) in enumerate(btd_blocks):
        H_D = _build_device_hamiltonian(H_00, H_01, n_slabs)
        H_LD = np.zeros((n_dof, N_D), dtype=complex)
        H_LD[:, :n_dof] = H_01
        H_DR = np.zeros((N_D, n_dof), dtype=complex)
        H_DR[-n_dof:, :] = H_01
        for iw, w2 in enumerate(omega_sq_thz2):
            trans_ballistic[iw] += _ballistic_transmission(
                w2, H_D, H_00, H_01, H_00, H_01, H_LD, H_DR, eta=eta
            )
    trans_ballistic /= n_kpts

    if verbose:
        print(f"  Ballistic max T: {trans_ballistic.max():.4f}")

    # Cross-sectional area
    lattice = phonon.primitive.cell
    tidx = "xyz".index(transport_direction)
    perp_idx = [i for i in range(3) if i != tidx]
    a1 = lattice[perp_idx[0]]
    a2 = lattice[perp_idx[1]]
    A_c = np.linalg.norm(np.cross(a1, a2)) * 1e-20

    omega_rad = freqs_thz * THZ_TO_RAD
    spectral_J_ball = HBAR_SI * omega_rad * (n_bose_L - n_bose_R) * trans_ballistic
    J_ball_total = np.sum(spectral_J_ball) * dw_thz * 1e12
    G_ball = J_ball_total / (A_c * delta_T)

    if verbose:
        print(f"  Ballistic thermal conductance: {G_ball:.2f} W/(m^2 K)")

    # --- SCBA with dense q-dependent self-energy from PCP-reconstructed FC3 ---
    Sigma_R_q = np.zeros((n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
    Sigma_l_q = np.zeros_like(Sigma_R_q)
    Sigma_g_q = np.zeros_like(Sigma_R_q)

    spectral_J_L = np.zeros(nfreq)
    spectral_J_R = np.zeros(nfreq)
    J_total_prev = 0.0
    conservation_err = 1.0

    convergence_history = []

    for scba_iter in range(max_scba_iter):
        G_lesser_slab_q = np.zeros((n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
        G_greater_slab_q = np.zeros_like(G_lesser_slab_q)
        spectral_J_L[:] = 0.0
        spectral_J_R[:] = 0.0

        for iq, (H_00, H_01) in enumerate(btd_blocks):
            H_D = _build_device_hamiltonian(H_00, H_01, n_slabs)
            for iw, w2 in enumerate(omega_sq_thz2):
                Sig_R_dev = np.zeros((N_D, N_D), dtype=complex)
                Sig_l_dev = np.zeros_like(Sig_R_dev)
                Sig_g_dev = np.zeros_like(Sig_R_dev)
                for sl_idx in range(n_slabs):
                    sl = slice(sl_idx * n_dof, (sl_idx + 1) * n_dof)
                    Sig_R_dev[sl, sl] = Sigma_R_q[sl_idx, iq, iw]
                    Sig_l_dev[sl, sl] = Sigma_l_q[sl_idx, iq, iw]
                    Sig_g_dev[sl, sl] = Sigma_g_q[sl_idx, iq, iw]

                obc = _compute_obc_self_energies(
                    w2, H_00, H_01, eta, n_bose_L[iw], n_bose_R[iw], n_slabs=n_slabs,
                )
                _, G_less, G_great = _solve_green_functions(
                    w2, H_D, obc, Sig_R_dev, Sig_l_dev, Sig_g_dev, eta,
                )

                for sl_idx in range(n_slabs):
                    sl = slice(sl_idx * n_dof, (sl_idx + 1) * n_dof)
                    G_lesser_slab_q[sl_idx, iq, iw] = G_less[sl, sl]
                    G_greater_slab_q[sl_idx, iq, iw] = G_great[sl, sl]

                sl0 = slice(0, n_dof)
                spectral_J_L[iw] += HBAR_SI * omega_rad[iw] * np.real(np.trace(
                    obc["Sigma_L_greater"][sl0, sl0] @ G_less[sl0, sl0]
                    - obc["Sigma_L_lesser"][sl0, sl0] @ G_great[sl0, sl0]
                ))
                sl_last = slice((n_slabs - 1) * n_dof, n_slabs * n_dof)
                spectral_J_R[iw] += HBAR_SI * omega_rad[iw] * np.real(np.trace(
                    obc["Sigma_R_lesser"][sl_last, sl_last] @ G_great[sl_last, sl_last]
                    - obc["Sigma_R_greater"][sl_last, sl_last] @ G_less[sl_last, sl_last]
                ))

        spectral_J_L /= n_kpts
        spectral_J_R /= n_kpts
        J_L_total = np.sum(spectral_J_L) * dw_thz * 1e12
        J_R_total = np.sum(spectral_J_R) * dw_thz * 1e12
        J_total = 0.5 * (J_L_total + J_R_total)
        J_denom = abs(J_L_total) + abs(J_R_total)
        conservation_err = abs(J_L_total - J_R_total) / J_denom if J_denom > 0 else 0.0

        # Dense q-dependent self-energy from PCP-reconstructed FC3
        Sigma_l_new = np.zeros_like(Sigma_l_q)
        Sigma_g_new = np.zeros_like(Sigma_g_q)
        Sigma_r_new = np.zeros_like(Sigma_R_q)

        for sl_idx in range(n_slabs):
            sl_n, sg_n, sr_n = _compute_phph_self_energy_q_dense(
                G_lesser_slab_q[sl_idx], G_greater_slab_q[sl_idx],
                M_stacked, T_all_q, q_diff_map,
                n_atoms, n_kpts, freqs_thz, dw_thz,
            )
            Sigma_l_new[sl_idx] = sl_n
            Sigma_g_new[sl_idx] = sg_n
            Sigma_r_new[sl_idx] = sr_n

        sig_r_norm = np.max(np.abs(Sigma_r_new))

        if scba_iter == 0:
            Sigma_l_q = Sigma_l_new
            Sigma_g_q = Sigma_g_new
            Sigma_R_q = Sigma_r_new
            if verbose:
                H_diag_norm = np.max([np.max(np.abs(np.diag(H_00))) for H_00, _ in btd_blocks])
                print(f"    Self-energy: max|Sigma^R| = {sig_r_norm:.4e} THz^2, "
                      f"|Sigma^R|/|H_00| = {sig_r_norm / H_diag_norm:.4e}")
                print(f"    SCBA iter 1: J_L = {J_L_total:.4e} W, J_R = {J_R_total:.4e} W")
        else:
            Sigma_l_q = (1 - mixing) * Sigma_l_q + mixing * Sigma_l_new
            Sigma_g_q = (1 - mixing) * Sigma_g_q + mixing * Sigma_g_new
            Sigma_R_q = (1 - mixing) * Sigma_R_q + mixing * Sigma_r_new

            rel_change = abs(J_total - J_total_prev) / abs(J_total_prev) if abs(J_total_prev) > 0 else 1.0
            convergence_history.append(rel_change)
            if verbose:
                print(f"    SCBA iter {scba_iter + 1}: J = {J_total:.4e} W, "
                      f"conservation = {conservation_err:.4e}, "
                      f"rel. change = {rel_change:.4e}, "
                      f"max|Sigma^R| = {sig_r_norm:.2e} THz^2")
            if conservation_err < scba_tol:
                break

        J_total_prev = J_total

    # Final results
    G_anh = J_total / (A_c * delta_T)
    spectral_J = 0.5 * (spectral_J_L + spectral_J_R)

    if verbose:
        print(f"  Anharmonic thermal conductance: {G_anh:.2f} W/(m^2 K)")
        print(f"  Heat flow conservation: {conservation_err:.4e}")

    return {
        "freqs_thz": freqs_thz,
        "omega_rad": freqs_thz * THZ_TO_RAD,
        "transmission_ballistic": trans_ballistic,
        "spectral_heat_current_ballistic": spectral_J_ball,
        "spectral_heat_current": spectral_J,
        "heat_current_ballistic": J_ball_total,
        "heat_current": J_total,
        "thermal_conductance_ballistic": G_ball,
        "thermal_conductance_anharmonic": G_anh,
        "heat_flow_conservation": conservation_err,
        "delta_T": delta_T,
        "n_scba_iterations": scba_iter + 1,
        "convergence_history": convergence_history,
        "self_energy_retarded": Sigma_R_q,
        "self_energy_lesser": Sigma_l_q,
        "self_energy_greater": Sigma_g_q,
        "pcp_info": pcp_info,
    }
