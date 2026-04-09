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
from scipy.optimize import minimize

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

    # Map (t1, t2, t3) -> linear cell index
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
# PCP forward model
# ---------------------------------------------------------------------------


def evaluate_pcp(A_modes, lambdas, sc_to_cell, sc_to_prim, cell_diff,
                 n_cells, nat_prim, n_super):
    """Evaluate PCP ansatz to reconstruct FC3 in compact format.

    Parameters
    ----------
    A_modes : ndarray, shape (3, N_c, n_cells, nat_prim, 3)
        PCP mode vectors. A_modes[leg, xi, cell, atom, alpha].
    lambdas : ndarray, shape (N_c,)
        PCP singular values.
    sc_to_cell, sc_to_prim : ndarray, shape (n_super,)
    cell_diff : ndarray, shape (n_cells, n_cells)
    n_cells, nat_prim, n_super : int

    Returns
    -------
    fc3_approx : ndarray, shape (nat_prim, n_super, n_super, 3, 3, 3)
    """
    N_c = len(lambdas)
    fc3 = np.zeros((nat_prim, n_super, n_super, 3, 3, 3))

    # Cell index for the first atom (at cell 0): cell_b1 = 0 for all b1
    cell_b1 = 0

    for xi in range(N_c):
        w = lambdas[xi] / 6.0  # lambda / 3!
        for sigma in S3_PERMS:
            s1, s2, s3 = sigma
            for l in range(n_cells):
                # Cell offsets: first atom at cell 0, shifted by -l
                c1 = cell_diff[cell_b1, l]  # = (0 - l) mod sc
                for b1 in range(nat_prim):
                    mode1 = A_modes[s1, xi, c1, b1, :]  # (3,)
                    for j in range(n_super):
                        c2 = cell_diff[sc_to_cell[j], l]
                        b2 = sc_to_prim[j]
                        mode2 = A_modes[s2, xi, c2, b2, :]  # (3,)
                        for k in range(n_super):
                            c3 = cell_diff[sc_to_cell[k], l]
                            b3 = sc_to_prim[k]
                            mode3 = A_modes[s3, xi, c3, b3, :]  # (3,)
                            # Outer product of 3 mode vectors
                            fc3[b1, j, k] += w * np.einsum('a,b,c->abc', mode1, mode2, mode3)

    return fc3


def _evaluate_pcp_vectorized(A_flat, lambdas, nat_prim, n_super, n_cells, N_c,
                              sc_to_cell, sc_to_prim, cell_diff):
    """Vectorized PCP forward model for optimization.

    Parameters
    ----------
    A_flat : ndarray, shape (3 * N_c * n_cells * nat_prim * 3,)
    lambdas : ndarray, shape (N_c,)
    Returns fc3_flat : ndarray, shape (nat_prim * n_super * n_super * 27,)
    """
    mode_size = n_cells * nat_prim * 3
    A = A_flat.reshape(3, N_c, n_cells, nat_prim, 3)

    # Precompute mode vectors for each (leg, xi, supercell_atom, cell_shift_l)
    # For atom j at cell c_j, shifted by l: cell = cell_diff[c_j, l]
    # mode_table[leg, xi, j, l, :] = A[leg, xi, cell_diff[sc_to_cell[j], l], sc_to_prim[j], :]
    # Shape: (3, N_c, n_super, n_cells, 3)
    mode_table = np.zeros((3, N_c, n_super, n_cells, 3))
    for j in range(n_super):
        cj = sc_to_cell[j]
        bj = sc_to_prim[j]
        for l in range(n_cells):
            cl = cell_diff[cj, l]
            mode_table[:, :, j, l, :] = A[:, :, cl, bj, :]

    # First atom: b1 at cell 0, shift -l means cell_diff[0, l]
    # mode1_table[leg, xi, b1, l, :] for first atom index
    mode1_table = np.zeros((3, N_c, nat_prim, n_cells, 3))
    for b1 in range(nat_prim):
        for l in range(n_cells):
            c1 = cell_diff[0, l]
            mode1_table[:, :, b1, l, :] = A[:, :, c1, b1, :]

    fc3 = np.zeros((nat_prim, n_super, n_super, 3, 3, 3))

    for xi in range(N_c):
        w = lambdas[xi] / 6.0
        for sigma in S3_PERMS:
            s1, s2, s3 = sigma
            # For each l, build the contribution:
            # fc3[b1, j, k, a, b, c] += w * mode1[s1,xi,b1,l,a] * mode_t[s2,xi,j,l,b] * mode_t[s3,xi,k,l,c]
            for l in range(n_cells):
                m1 = mode1_table[s1, xi, :, l, :]  # (nat_prim, 3)
                m2 = mode_table[s2, xi, :, l, :]    # (n_super, 3)
                m3 = mode_table[s3, xi, :, l, :]    # (n_super, 3)
                # fc3[b1, j, k, a, b, c] += w * m1[b1, a] * m2[j, b] * m3[k, c]
                fc3 += w * np.einsum('ia,jb,kc->ijkabc', m1, m2, m3)

    return fc3


# ---------------------------------------------------------------------------
# Loss function and gradient
# ---------------------------------------------------------------------------


def _pcp_loss_and_grad(x, target, nat_prim, n_super, n_cells, N_c,
                        sc_to_cell, sc_to_prim, cell_diff, asr_penalty,
                        sc_mat):
    """Compute PCP loss and analytic gradient.

    x = [A_flat (3*N_c*mode_dim), lambdas (N_c)]
    """
    mode_dim = n_cells * nat_prim * 3
    n_A = 3 * N_c * mode_dim
    A_flat = x[:n_A]
    lambdas = x[n_A:]
    A = A_flat.reshape(3, N_c, n_cells, nat_prim, 3)

    # Forward model
    fc3_approx = _evaluate_pcp_vectorized(A_flat, lambdas, nat_prim, n_super,
                                           n_cells, N_c, sc_to_cell, sc_to_prim, cell_diff)

    # Residual
    residual = fc3_approx - target
    loss = np.sum(residual**2)

    # ASR penalty: sum_{l,b} A_i^xi(l, b, alpha) = 0 for each (xi, i, alpha)
    asr_loss = 0.0
    if asr_penalty > 0:
        asr_sum = np.sum(A, axis=(2, 3))  # (3, N_c, 3) — sum over cells and atoms
        asr_loss = asr_penalty * np.sum(asr_sum**2)
        loss += asr_loss

    # ---- Gradient ----
    grad_A = np.zeros_like(A)
    grad_lam = np.zeros(N_c)

    # Precompute mode tables (same as forward)
    mode_table = np.zeros((3, N_c, n_super, n_cells, 3))
    for j in range(n_super):
        cj = sc_to_cell[j]
        bj = sc_to_prim[j]
        for l in range(n_cells):
            cl = cell_diff[cj, l]
            mode_table[:, :, j, l, :] = A[:, :, cl, bj, :]

    mode1_table = np.zeros((3, N_c, nat_prim, n_cells, 3))
    for b1 in range(nat_prim):
        for l in range(n_cells):
            c1 = cell_diff[0, l]
            mode1_table[:, :, b1, l, :] = A[:, :, c1, b1, :]

    # Gradient w.r.t. lambdas: dL/dlam_xi = (2/6) * sum_sigma sum_l
    #   sum_{b1,j,k,a,b,c} residual * mode1*mode2*mode3
    # This equals sum of element-wise product of residual with (fc3 from xi without lambda)
    for xi in range(N_c):
        fc3_xi = np.zeros_like(fc3_approx)
        for sigma in S3_PERMS:
            s1, s2, s3 = sigma
            for l in range(n_cells):
                m1 = mode1_table[s1, xi, :, l, :]
                m2 = mode_table[s2, xi, :, l, :]
                m3 = mode_table[s3, xi, :, l, :]
                fc3_xi += (1.0 / 6.0) * np.einsum('ia,jb,kc->ijkabc', m1, m2, m3)
        grad_lam[xi] = 2.0 * np.sum(residual * fc3_xi)

    # Gradient w.r.t. A modes:
    # For each (leg_p, xi), dL/dA_p^xi[l0, b0, alpha0] involves contracting
    # residual with the other two mode vectors at all positions where
    # A_p^xi contributes (through different sigma permutations and l-shifts).
    R = 2.0 * residual  # dL/dPhi = 2 * residual

    for xi in range(N_c):
        w = lambdas[xi] / 6.0
        for sigma in S3_PERMS:
            s1, s2, s3 = sigma
            for l in range(n_cells):
                m1 = mode1_table[s1, xi, :, l, :]  # (nat_prim, 3)
                m2 = mode_table[s2, xi, :, l, :]    # (n_super, 3)
                m3 = mode_table[s3, xi, :, l, :]    # (n_super, 3)

                # Gradient for leg s1 (appears as first index, i.e., b1 position):
                # dL/dA_{s1}^xi at cell_diff[0,l], atom b1, alpha
                # = w * R[b1,j,k,a,b,c] * m2[j,b] * m3[k,c]   summed over j,k,b,c
                # = w * einsum('ijkabc,jb,kc->ia', R, m2, m3)
                g1 = w * np.einsum('ijkabc,jb,kc->ia', R, m2, m3)
                c1 = cell_diff[0, l]
                for b1 in range(nat_prim):
                    grad_A[s1, xi, c1, b1, :] += g1[b1, :]

                # Gradient for leg s2 (appears as second index, j position):
                # dL/dA_{s2}^xi at cell_diff[sc_to_cell[j], l], sc_to_prim[j]
                # = w * R[b1,j,k,a,b,c] * m1[b1,a] * m3[k,c]  summed over b1,k,a,c
                g2 = w * np.einsum('ijkabc,ia,kc->jb', R, m1, m3)
                for j in range(n_super):
                    c2 = cell_diff[sc_to_cell[j], l]
                    b2 = sc_to_prim[j]
                    grad_A[s2, xi, c2, b2, :] += g2[j, :]

                # Gradient for leg s3 (appears as third index, k position):
                g3 = w * np.einsum('ijkabc,ia,jb->kc', R, m1, m2)
                for k in range(n_super):
                    c3 = cell_diff[sc_to_cell[k], l]
                    b3 = sc_to_prim[k]
                    grad_A[s3, xi, c3, b3, :] += g3[k, :]

    # ASR gradient
    if asr_penalty > 0:
        asr_sum = np.sum(A, axis=(2, 3))  # (3, N_c, 3)
        grad_A += 2.0 * asr_penalty * asr_sum[:, :, None, None, :]

    grad = np.concatenate([grad_A.ravel(), grad_lam])
    return loss, grad


# ---------------------------------------------------------------------------
# PyTorch PCP forward model and fitting
# ---------------------------------------------------------------------------


def _build_index_tables(sc_to_cell, sc_to_prim, cell_diff, nat_prim, n_super, n_cells):
    """Precompute index tables for the PCP forward model.

    Returns integer arrays that map (b1, j, k, l) to the appropriate
    (cell, prim_atom) indices for all three legs. These are used for
    advanced indexing in both NumPy and PyTorch.

    Returns
    -------
    idx1_cell : (nat_prim, n_cells) — cell indices for leg 1
    idx1_atom : (nat_prim, n_cells) — atom indices for leg 1 (just b1)
    idx_jk_cell : (n_super, n_cells) — cell indices for legs 2,3
    idx_jk_atom : (n_super,) — prim atom for each supercell atom
    """
    # Leg 1: atom b1 at cell 0, shift -l => cell_diff[0, l]
    idx1_cell = np.zeros((nat_prim, n_cells), dtype=np.int64)
    for b1 in range(nat_prim):
        for l in range(n_cells):
            idx1_cell[b1, l] = cell_diff[0, l]

    # Legs 2,3: atom j at cell c_j, shift => cell_diff[c_j, l]
    idx_jk_cell = np.zeros((n_super, n_cells), dtype=np.int64)
    for j in range(n_super):
        for l in range(n_cells):
            idx_jk_cell[j, l] = cell_diff[sc_to_cell[j], l]

    return idx1_cell, idx_jk_cell, sc_to_prim.astype(np.int64)


def _pcp_forward_torch(A, lambdas, idx1_cell, idx_jk_cell, sc_to_prim,
                       nat_prim, n_super, n_cells, N_c):
    """Fully vectorized PCP forward model in PyTorch.

    A : (3, N_c, n_cells, nat_prim, 3)
    lambdas : (N_c,)
    Returns fc3 : (nat_prim, n_super, n_super, 3, 3, 3)
    """
    import torch

    # Build mode tables via advanced indexing
    # mode1[leg, xi, b1, l, alpha] = A[leg, xi, idx1_cell[b1,l], b1, alpha]
    # shape: (3, N_c, nat_prim, n_cells, 3)
    b1_range = torch.arange(nat_prim, device=A.device)
    mode1 = A[:, :, idx1_cell, b1_range[:, None], :]  # (3, N_c, nat_prim, n_cells, 3)

    # mode_jk[leg, xi, j, l, alpha] = A[leg, xi, idx_jk_cell[j,l], sc_to_prim[j], alpha]
    # shape: (3, N_c, n_super, n_cells, 3)
    mode_jk = A[:, :, idx_jk_cell, sc_to_prim[:, None].expand(-1, n_cells), :]

    # Accumulate fc3 over permutations and cell shifts
    # fc3[b1, j, k, a, b, c] = sum_xi (lam_xi/6) sum_sigma sum_l
    #   mode_{s1}[xi, b1, l, a] * mode_{s2}[xi, j, l, b] * mode_{s3}[xi, k, l, c]
    #
    # For each permutation (s1, s2, s3) and each l:
    #   contribution = einsum('xia, xjb, xkc -> ijkabc', m1, m2, m3) summed over x=xi

    fc3 = torch.zeros(nat_prim, n_super, n_super, 3, 3, 3,
                       dtype=A.dtype, device=A.device)

    w = lambdas / 6.0  # (N_c,)

    for s1, s2, s3 in S3_PERMS:
        # Sum over l (cell shift) first, then do the triple outer product
        # m1_sum[xi, b1, a] = sum_l mode1[s1, xi, b1, l, a]  — NO, we need per-l products
        # Instead: for each l, accumulate the weighted outer product
        # Vectorize over l by doing a single einsum with l as a contraction index

        m1 = mode1[s1]   # (N_c, nat_prim, n_cells, 3)
        m2 = mode_jk[s2] # (N_c, n_super, n_cells, 3)
        m3 = mode_jk[s3] # (N_c, n_super, n_cells, 3)

        # Contract over l (cell shifts) and xi (ranks, with weights):
        # fc3[b1,j,k,a,b,c] += sum_xi w[xi] * sum_l m1[xi,b1,l,a] * m2[xi,j,l,b] * m3[xi,k,l,c]
        #
        # Reshape for einsum: contract xi and l simultaneously
        # w[xi]*m1[xi,b1,l,a] -> wm1[xi,b1,l,a]
        wm1 = w[:, None, None, None] * m1  # (N_c, nat_prim, n_cells, 3)

        # einsum('xila, xjlb, xklc -> ijkabc', wm1, m2, m3)
        # This is an outer product over (b1,a) x (j,b) x (k,c) contracted over (xi, l)
        fc3 += torch.einsum('xila, xjlb, xklc -> ijkabc', wm1, m2, m3)

    return fc3


def fit_pcp(fc3_raw, phonon, N_c=24, asr_penalty=10.0,
            max_iter=2000, verbose=True):
    """Fit PCP decomposition to FC3 via Adam optimizer (PyTorch).

    Parameters
    ----------
    fc3_raw : ndarray
        FC3 in compact (nat_prim, n_super, n_super, 3, 3, 3) or
        full (n_super, n_super, n_super, 3, 3, 3) format.
    phonon : Phonopy
    N_c : int
        PCP rank.
    asr_penalty : float
        Weight for ASR violation penalty.
    max_iter : int
        Number of Adam iterations.
    verbose : bool

    Returns
    -------
    A_modes : ndarray, shape (3, N_c, n_cells, nat_prim, 3)
    lambdas : ndarray, shape (N_c,)
    info : dict
    """
    import torch

    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    masses = phonon.supercell.masses

    sc_to_cell, sc_to_prim_np, cell_frac_all, n_cells, sc_mat = build_cell_mapping(phonon)
    cell_diff, idx_to_t = build_cell_diff_table(n_cells, sc_mat)

    # Handle compact vs full format
    is_compact = fc3_raw.shape[0] == nat_prim
    p2s = phonon.primitive.p2s_map

    # Build mass-weighted target FC3 in compact format
    target_np = np.zeros((nat_prim, n_super, n_super, 3, 3, 3))
    for i_prim in range(nat_prim):
        fc3_idx = i_prim if is_compact else int(p2s[i_prim])
        m_i = masses[int(p2s[i_prim])]
        for j in range(n_super):
            m_j = masses[j]
            for k in range(n_super):
                m_k = masses[k]
                mass_factor = np.sqrt(m_i * m_j * m_k)
                target_np[i_prim, j, k] = fc3_raw[fc3_idx, j, k] / mass_factor * CONVERSION_FC3_THZ

    target_norm = np.linalg.norm(target_np)
    if verbose:
        print(f"  PCP fitting: N_c={N_c}, target norm={target_norm:.4e}")
        print(f"  Supercell: {n_super} atoms, {n_cells} cells, {nat_prim} prim atoms")
        n_params = 3 * N_c * n_cells * nat_prim * 3 + N_c
        n_target = nat_prim * n_super * n_super * 27
        print(f"  Parameters: {n_params}, target entries: {n_target}, "
              f"compression: {n_target / N_c:.0f}x")

    # Normalize target
    target_np /= target_norm
    target_t = torch.tensor(target_np, dtype=torch.float64)

    # Precompute index tables
    idx1_cell_np, idx_jk_cell_np, sc_to_prim_i = _build_index_tables(
        sc_to_cell, sc_to_prim_np, cell_diff, nat_prim, n_super, n_cells)
    idx1_cell = torch.tensor(idx1_cell_np, dtype=torch.long)
    idx_jk_cell = torch.tensor(idx_jk_cell_np, dtype=torch.long)
    sc_to_prim_t = torch.tensor(sc_to_prim_i, dtype=torch.long)

    # Initialize parameters
    mode_dim = n_cells * nat_prim * 3
    rng = np.random.default_rng(42)
    scale = (1.0 / (N_c * mode_dim)) ** (1.0 / 3.0)

    A_param = torch.tensor(
        rng.normal(0, scale, (3, N_c, n_cells, nat_prim, 3)),
        dtype=torch.float64, requires_grad=True,
    )
    lam_param = torch.tensor(
        np.ones(N_c) / N_c,
        dtype=torch.float64, requires_grad=True,
    )

    # Two-phase optimization: Adam warm-up then L-BFGS refinement
    best_err = float('inf')
    best_A = A_param.detach().clone()
    best_lam = lam_param.detach().clone()

    def compute_loss():
        fc3_approx = _pcp_forward_torch(
            A_param, lam_param, idx1_cell, idx_jk_cell, sc_to_prim_t,
            nat_prim, n_super, n_cells, N_c,
        )
        loss = torch.sum((fc3_approx - target_t) ** 2)
        if asr_penalty > 0:
            asr_sum = A_param.sum(dim=(2, 3))
            loss = loss + asr_penalty * torch.sum(asr_sum ** 2)
        return loss, fc3_approx

    # Phase 1: Adam with high learning rate
    adam_iters = min(max_iter // 4, 500)
    optimizer = torch.optim.Adam([A_param, lam_param], lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=adam_iters, eta_min=1e-3,
    )

    if verbose:
        print(f"  Phase 1: Adam ({adam_iters} iters, lr=0.05)...")

    for it in range(1, adam_iters + 1):
        optimizer.zero_grad()
        loss, fc3_approx = compute_loss()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if it % 100 == 0 or it == 1:
            with torch.no_grad():
                err = torch.sqrt(torch.sum((fc3_approx - target_t) ** 2)).item()
                asr_v = torch.max(torch.abs(A_param.sum(dim=(2, 3)))).item()
            if err < best_err:
                best_err = err
                best_A = A_param.detach().clone()
                best_lam = lam_param.detach().clone()
            if verbose:
                print(f"    iter {it:5d}: rel_err={err:.4e}, "
                      f"max_asr={asr_v:.2e}")

    # Phase 2: L-BFGS for precise convergence
    lbfgs_iters = max_iter - adam_iters
    if verbose:
        print(f"  Phase 2: L-BFGS ({lbfgs_iters} iters)...")

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

    lbfgs_step = [0]

    for outer in range(lbfgs_iters // 20 + 1):
        def closure():
            lbfgs.zero_grad()
            loss, _ = compute_loss()
            loss.backward()
            return loss

        lbfgs.step(closure)
        lbfgs_step[0] += 1

        if (lbfgs_step[0]) % 5 == 0 or lbfgs_step[0] == 1:
            with torch.no_grad():
                loss_val, fc3_approx = compute_loss()
                err = torch.sqrt(torch.sum((fc3_approx - target_t) ** 2)).item()
                asr_v = torch.max(torch.abs(A_param.sum(dim=(2, 3)))).item()
            if err < best_err:
                best_err = err
                best_A = A_param.detach().clone()
                best_lam = lam_param.detach().clone()
            if verbose:
                print(f"    L-BFGS step {lbfgs_step[0]:4d}: rel_err={err:.4e}, "
                      f"max_asr={asr_v:.2e}")

    # Use best parameters
    A_modes = best_A.numpy()
    lambdas = best_lam.numpy()

    # Rescale lambdas
    lambdas *= target_norm

    # Sort by |lambda| descending
    order = np.argsort(-np.abs(lambdas))
    A_modes = A_modes[:, order]
    lambdas = lambdas[order]
    asr_violation = np.max(np.abs(np.sum(A_modes, axis=(2, 3))))

    info = {
        'rel_err': best_err,
        'asr_violation': asr_violation,
        'n_iter': max_iter,
        'success': best_err < 0.05,
        'message': 'Adam optimization',
        'target_norm': target_norm,
        'n_cells': n_cells,
        'sc_mat': sc_mat,
        'sc_to_cell': sc_to_cell,
        'sc_to_prim': sc_to_prim_np,
        'cell_diff': cell_diff,
        'idx_to_t': idx_to_t,
    }

    if verbose:
        print(f"  PCP fit: rel_err={best_err:.4e}, asr={asr_violation:.2e}")

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
    # R_frac_cell = idx_to_t[l] (in fractional coords of primitive cell)
    phases = np.exp(-2j * np.pi * idx_to_t @ q_full)  # (n_cells,)

    # FT: f_i^xi(q)[kappa*3+alpha] = sum_l phase[l] * A_i^xi(l, kappa, alpha)
    # A_modes shape: (3, N_c, n_cells, nat_prim, 3)
    # phases shape: (n_cells,) -> broadcast: (1, 1, n_cells, 1, 1)
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

    Projects G onto the PCP mode vectors to obtain scalar functions
    g_{ij}^{xi,xi'}(w;q), convolves pairs of scalars, and reconstructs
    the matrix self-energy from the external mode vectors.

    This is the exact PCP self-energy formula: the scalar projection
    happens BEFORE the frequency convolution, which is not equivalent
    to reconstructing Phi and using the dense kernel (that would be a
    different approximation).

    Uses matmul accumulation (Sigma[w] = F^H @ C[w] @ conj(F)) for
    ~5-15x speedup over the naive einsum on the accumulation step.

    Cost per (q,q') pair: O(9*N_c^2 * n_fft * log(n_fft)).
    Faster than dense when 9*N_c^2 < n_dof^4, i.e., small PCP rank.

    Parameters
    ----------
    G_lesser_q, G_greater_q : ndarray, shape (n_kpts, n_freq, n_dof, n_dof)
    f_modes_all_q : ndarray, shape (n_kpts, 3, N_c, n_dof), complex
        FT'd PCP modes at each q-point.
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

                    # C[w, xi, xi'] = lam_outer * conv[xi, xi', w]
                    CL = (lam_outer[:, :, None] * conv_all_L[:, :, s1, s1p, :]).transpose(2, 0, 1)
                    CG = (lam_outer[:, :, None] * conv_all_G[:, :, s1, s1p, :]).transpose(2, 0, 1)

                    # T[w, xi, b] = C[w] @ conj(f')
                    TL = CL @ f_s1p_conj   # (n_freq, N_c, n_dof)
                    TG = CG @ f_s1p_conj

                    # Sigma[w, a, b] += prefactor * f^T @ T[w]
                    Sigma_lesser[iq_ext] += prefactor * np.einsum('xa,wxy->way', f_s1, TL)
                    Sigma_greater[iq_ext] += prefactor * np.einsum('xa,wxy->way', f_s1, TG)

    Sigma_retarded = 0.5 * (Sigma_greater - Sigma_lesser)
    return Sigma_lesser, Sigma_greater, Sigma_retarded




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
    asr_penalty: float = 10.0,
    pcp_max_iter: int = 2000,
    verbose: bool = True,
) -> dict:
    """Anharmonic phonon transmission via PCP FC3 decomposition.

    Parameters
    ----------
    phonon : Phonopy
    fc3_hdf5 : str or Path
        Path to fc3.hdf5.
    pcp_rank : int
        PCP rank N_c.
    asr_penalty : float
        ASR penalty weight for PCP fitting.
    pcp_max_iter : int
        Max iterations for PCP fitting.

    Returns
    -------
    result : dict
    """
    import h5py
    from .convention import get_btd_blocks
    from .validation import _ballistic_transmission
    from .separable import build_supercell_mapping, build_q_diff_map

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

    A_modes, lambdas, pcp_info = fit_pcp(
        fc3_raw, phonon, N_c=pcp_rank,
        asr_penalty=asr_penalty, max_iter=pcp_max_iter, verbose=verbose,
    )

    if verbose:
        print(f"  PCP rank: {pcp_rank}, rel_err: {pcp_info['rel_err']:.4e}")
        print(f"  Device: {n_slabs} slab(s), {N_D} DOFs per q-point")

    # --- q-mesh ---
    nkx, nky = q_mesh_transverse
    q_1d_x = [i / nkx for i in range(nkx)]
    q_1d_y = [j / nky for j in range(nky)]
    q_points = [(qx, qy) for qx in q_1d_x for qy in q_1d_y]
    n_kpts = len(q_points)
    q_diff_map = build_q_diff_map(nkx, nky)

    # FT PCP modes for each q-point
    f_modes_all_q = np.zeros((n_kpts, 3, pcp_rank, n_dof), dtype=complex)
    for iq, (qx, qy) in enumerate(q_points):
        f_modes_all_q[iq] = fourier_transform_pcp(
            A_modes, lambdas, phonon, (qx, qy), transport_direction, info=pcp_info,
        )

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

    # --- SCBA ---
    Sigma_R_q = np.zeros((n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
    Sigma_l_q = np.zeros_like(Sigma_R_q)
    Sigma_g_q = np.zeros_like(Sigma_R_q)

    spectral_J_L = np.zeros(nfreq)
    spectral_J_R = np.zeros(nfreq)
    J_total_prev = 0.0

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

        # PCP self-energy
        Sigma_l_new = np.zeros_like(Sigma_l_q)
        Sigma_g_new = np.zeros_like(Sigma_g_q)
        Sigma_r_new = np.zeros_like(Sigma_R_q)

        for sl_idx in range(n_slabs):
            sl_n, sg_n, sr_n = _compute_phph_self_energy_pcp(
                G_lesser_slab_q[sl_idx], G_greater_slab_q[sl_idx],
                f_modes_all_q, lambdas,
                n_dof, n_kpts, freqs_thz, dw_thz,
                q_diff_map=q_diff_map,
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
