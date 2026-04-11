"""Prototype Block-CP decomposition for FC3 tensors.

Block-CP generalizes both PCP (rank-1 vectors) and SVD (full matrices):
  Phi[a,b,c] = sum_xi lambda_xi * F1_xi[a,:r] @ F2_xi[b,:r].T @ ...

Actually, the right formulation for FC3 in NEGF context is:

Given M_stacked (n_dof*dim_t, dim_t) from the SVD decomposition:
  M_stacked ≈ sum_r F_r ⊗ h_r  (SVD: F_r is n_dof x dim_t, h_r is dim_t)

The self-energy kernel contracts:
  v_s(w,q) = G(w,q) @ h_hat_s(q)    [n_dof-vector]

Key insight: the SVD cost is O(R^2) because we have R independent h_r vectors.
If we could group the h_r vectors into blocks that share structure, we reduce R.

Block-CP idea: instead of R independent (F_r, h_r) pairs, use N_b blocks
each containing r_b internal rank:
  M_stacked ≈ sum_{b=1}^{N_b} sum_{j=1}^{r_b} F_{b,j} ⊗ h_{b,j}

where within each block, the h_{b,j} share a common "basis" or structure.

Actually, let me think about this differently. The key bottleneck in the
separable kernel is the R^2 scaling in the inner loop. What if we could
find a decomposition where the R rank-1 terms have additional structure
that allows batching?

A simpler and more practical approach: **truncated SVD with block structure**.
The M_stacked has shape (n_dof * dim_t, dim_t). After SVD, we get R terms.
The cost per q-pair scales as R^2 because we need all (r,s) pairs.

What if we decompose M_stacked as a sum of B "blocks", where each block
has internal rank r_b, and the cross-terms between blocks are zero?

  M_stacked = sum_{b=1}^B U_b S_b V_b^T

where U_b (n_dof*dim_t, r_b), S_b (r_b, r_b), V_b (dim_t, r_b).
The total rank is sum(r_b) = R, but the cost is sum(r_b^2) instead of R^2.

This is essentially a block-diagonal structure in the SVD coordinate system.
If we can partition the R singular vectors into B groups where cross-group
interactions are small, we get speedup B with negligible accuracy loss.

Let me test this on the Si FC3.
"""

import sys
from pathlib import Path

import numpy as np

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.separable import (
    build_supercell_mapping,
    decompose_fc3_supercell,
)


def build_M_stacked(fc3_raw, nat_prim, masses_super, prim_indices,
                     slab_indices, ref_sc_atoms):
    """Build M_stacked matrix directly."""
    from phonon_inputs.separable import CONVERSION_FC3_THZ as CONV

    trans_atoms = np.where(slab_indices == 0)[0]
    n_trans = len(trans_atoms)
    dim_t = n_trans * 3
    n_dof = 3 * nat_prim
    n_atoms_super = len(masses_super)
    dim_full = n_atoms_super * 3

    filt_idx = np.sort(np.concatenate([3 * trans_atoms[:, None] + np.arange(3)], axis=0).ravel())

    M_stacked = np.zeros((n_dof * dim_t, dim_t))
    for i_prim in range(nat_prim):
        s_i = ref_sc_atoms[i_prim]
        m_i = masses_super[s_i]
        for alpha in range(3):
            a = 3 * i_prim + alpha
            block = fc3_raw[s_i, :, :, alpha, :, :]
            mat = block.transpose(0, 2, 1, 3).reshape(dim_full, dim_full)
            mat_filt = mat[np.ix_(filt_idx, filt_idx)]
            m_row = np.repeat(np.sqrt(masses_super[trans_atoms]), 3)
            m_col = np.repeat(np.sqrt(masses_super[trans_atoms]), 3)
            mat_filt = mat_filt / (np.sqrt(m_i) * m_row[:, None] * m_col[None, :])
            mat_filt = mat_filt * CONV
            M_stacked[a * dim_t:(a + 1) * dim_t, :] = mat_filt

    return M_stacked, trans_atoms


def analyze_block_structure(M_stacked, n_dof, dim_t):
    """Analyze whether SVD components have block structure."""
    print(f"\n=== Block structure analysis ===")
    print(f"M_stacked shape: {M_stacked.shape}")

    U, S, Vt = np.linalg.svd(M_stacked, full_matrices=False)
    R = len(S)
    print(f"Full rank: {R}")
    print(f"Singular values: {S[:10]}...")

    # The self-energy cost depends on the interaction between pairs (r, s).
    # Specifically, for each q-pair, the kernel computes:
    #   sum_{r,s} conv(v_r, u_s) * F_r^T
    # where v_s = G @ h_s_hat, u_r = G^T @ h_r_hat
    #
    # The cost is R^2 * n_dof^2 * n_fft per q-pair.
    # With block structure, we only need intra-block pairs.
    #
    # To find blocks: look at the coupling matrix between SVD components.
    # The "coupling" between r and s in the self-energy comes from the
    # Fourier-transformed left factors F_hat_r(q) and F_hat_s(q).
    # If F_r and F_s have small overlap, their cross-term is small.

    # Reshape U to get F_r matrices
    # U shape: (n_dof * dim_t, R)
    # Each column U[:,r] reshaped to (n_dof, dim_t) gives F_r
    F_matrices = U.reshape(n_dof, dim_t, R)  # (n_dof, dim_t, R)

    # Compute overlap matrix: O[r,s] = sum_a ||F_r[a,:] . F_s[a,:]||
    # This measures how much the left factors overlap
    # F_matrices[:,:,r] is F_r (n_dof, dim_t)
    # Inner product: sum_a (F_r[a,:] @ F_s[a,:].T) -> scalar
    # Actually let's compute: O[r,s] = ||sum_a F_r[a,:].T @ F_s[a,:]||_F
    # which is the Frobenius norm of the n_dof-contracted product

    print(f"\nComputing coupling matrix (R={R})...")
    # More directly: the contribution of (r,s) pair to the self-energy
    # is proportional to S[r] * S[s] * <F_r, F_s>
    # where <F_r, F_s> = sum_a trace(F_r[a]^T F_s[a])
    # But F_r are orthonormal columns of U, so <F_r, F_s> = delta_{rs}!
    # That means the coupling is ONLY through the Fourier transform.

    # Let me think again. The SVD gives M = U S V^T.
    # F_list[r] = U[:, r] reshaped to (n_dof, dim_t) (but unnormalized)
    # H = diag(S) @ V^T -> H[:, r] = S[r] * V[r, :]
    #
    # The self-energy sums over all (r,s) pairs.
    # The key quantity is the "interaction strength" between r and s.
    # In the q-domain, this depends on the Fourier transforms of H.
    #
    # Without Fourier transforms, the "coupling" is:
    # C[r,s] = sum_a F_r[a] @ F_s[a]^T (matrix in dim_t x dim_t)
    # For SVD, the U columns are orthogonal, so:
    # sum_{a,i} F_r[a,i] * F_s[a,i] = delta_{rs} (by orthogonality of U)
    # BUT the Fourier transform breaks this orthogonality!

    # So the block structure is q-dependent. At each q, the Fourier-
    # transformed factors may have different coupling patterns.

    # For a practical block-CP: group singular values by magnitude.
    # The cross-terms scale as S[r]*S[s], so if we keep the top R_eff
    # singular values and group them, the error is bounded.

    # Actually, the most practical approach is just to use a LOWER rank.
    # The SVD already gives the optimal low-rank approximation.
    # Block-CP doesn't help if U is already orthogonal.

    # NEW IDEA: Instead of SVD of M_stacked, what about a decomposition
    # that is low-rank in a DIFFERENT sense?
    #
    # Current: M_stacked = sum_r sigma_r * u_r ⊗ v_r
    #   -> Each term is rank-1 in (n_dof*dim_t) x (dim_t)
    #   -> F_r is n_dof x dim_t matrix (reshape of u_r)
    #   -> h_r is dim_t vector (v_r scaled by sigma_r)
    #
    # Block-CP: M_a = sum_b U_b^(a) S_b V_b^T
    #   -> Each M_a block (dim_t x dim_t) gets its own decomposition
    #   -> But we want SHARED right factors across a.
    #   -> This is exactly what the stacked SVD gives!

    # Let me try a DIFFERENT decomposition: Tucker on the 3-way tensor.
    # Reshape M_stacked to (n_dof, dim_t, dim_t) tensor, then Tucker decompose.

    print(f"\n=== Tucker decomposition of M(a, s2, s3) ===")
    M_tensor = M_stacked.reshape(n_dof, dim_t, dim_t)
    print(f"M_tensor shape: {M_tensor.shape}")

    # Tucker: M ≈ G x_1 A x_2 B x_3 C
    # where G is (r1, r2, r3) core tensor
    # A is (n_dof, r1), B is (dim_t, r2), C is (dim_t, r3)
    #
    # For NEGF: we need the (s2, s3) indices to remain as matrix indices
    # through the Green's function contraction.
    #
    # Tucker with r2 = r3 = r: the Green's function is contracted with
    # B and C (both dim_t x r), giving r-dimensional projections.
    # This is the Block-CP idea!

    # Mode-1 unfolding: M_(1) = M_tensor reshaped to (n_dof, dim_t^2)
    M_1 = M_tensor.reshape(n_dof, -1)
    U1, S1, _ = np.linalg.svd(M_1, full_matrices=False)
    print(f"Mode-1 rank: {np.sum(S1 > 1e-10 * S1[0])}")
    print(f"Mode-1 singular values: {S1}")

    # Mode-2 unfolding: M_(2) = M_tensor.transpose(1,0,2).reshape(dim_t, n_dof*dim_t)
    M_2 = M_tensor.transpose(1, 0, 2).reshape(dim_t, -1)
    U2, S2, _ = np.linalg.svd(M_2, full_matrices=False)
    print(f"\nMode-2 rank: {np.sum(S2 > 1e-10 * S2[0])}")
    print(f"Mode-2 singular values (first 10): {S2[:10]}")

    # Mode-3 unfolding: M_(3) = M_tensor.transpose(2,0,1).reshape(dim_t, n_dof*dim_t)
    M_3 = M_tensor.transpose(2, 0, 1).reshape(dim_t, -1)
    U3, S3, _ = np.linalg.svd(M_3, full_matrices=False)
    print(f"\nMode-3 rank: {np.sum(S3 > 1e-10 * S3[0])}")
    print(f"Mode-3 singular values (first 10): {S3[:10]}")

    # HOSVD: truncate each mode
    for r1 in [2, 4, 6]:
        for r23 in [4, 8, 12, 16, 24]:
            A = U1[:, :r1]  # (n_dof, r1)
            B = U2[:, :r23]  # (dim_t, r23)
            C = U3[:, :r23]  # (dim_t, r23)
            # Core tensor
            G_core = np.einsum('ia,ijk,jb,kc->abc', A, M_tensor, B, C)
            # Reconstruction
            M_recon = np.einsum('abc,ia,jb,kc->ijk', G_core, A, B, C)
            err = np.linalg.norm(M_tensor - M_recon) / np.linalg.norm(M_tensor)
            # Effective parameters: r1*n_dof + r23*dim_t*2 + r1*r23*r23
            n_params = r1 * n_dof + 2 * r23 * dim_t + r1 * r23 * r23
            # Self-energy cost would scale as r23^2 (like SVD with rank r23)
            # but only r1 terms in the outer sum (vs n_dof for dense)
            print(f"  Tucker r1={r1}, r23={r23}: err={err:.4e}, "
                  f"params={n_params}, SE cost ~ r1*r23^2 = {r1*r23**2}")

    # Compare with SVD at equivalent cost
    print(f"\n=== SVD at matched rank ===")
    for R in [2, 4, 6, 8, 12, 16, 24]:
        M_recon_svd = U[:, :R] @ np.diag(S[:R]) @ Vt[:R, :]
        err_svd = np.linalg.norm(M_stacked - M_recon_svd) / np.linalg.norm(M_stacked)
        # SE cost: R^2 per q-pair (main loop)
        print(f"  SVD R={R}: err={err_svd:.4e}, SE cost ~ R^2 = {R**2}")

    # === Practical Block-CP: shared right basis ===
    print(f"\n=== Shared-basis Block-CP ===")
    # Idea: use r right basis vectors (from V), but group the n_dof left
    # factors into blocks that only interact within their block.
    #
    # M_stacked = U @ diag(S) @ Vt
    # The columns of V span the right space.
    # The rows of U, when reshaped to (n_dof, dim_t, R), give the
    # per-DOF left factor contribution to each SVD component.
    #
    # If F_r[a] has support only on a subset of DOFs a, then
    # the self-energy contribution from r only involves those DOFs.
    # This doesn't reduce R^2 though.
    #
    # DIFFERENT ANGLE: The separable kernel's bottleneck is the
    # convolution over (r,s) pairs. What if we use a single shared
    # right basis V of dimension r, and decompose each M_a as:
    #   M_a ≈ L_a @ V^T  (L_a is dim_t x r, V is dim_t x r)
    #
    # Then v(w,q) = G(w,q) @ V_hat(q) gives r projections.
    # The convolution involves: conv(v_j, v_k) for j,k in 1..r
    # That's r^2 convolutions per q-pair.
    # Then: Sigma += sum_a L_a_hat(q) @ conv(j,k) @ L_a_hat(q)^H
    #
    # This is IDENTICAL to the stacked SVD with rank r!
    # The "shared V" is just V from the SVD.
    #
    # Key insight: Block-CP ≡ low-rank SVD for this problem.
    # The Tucker decomposition adds the mode-1 compression (r1 < n_dof),
    # but n_dof is already small (6 for Si).

    # For LARGE primitive cells (n_dof >> 6), the Tucker mode-1 compression
    # becomes valuable. Let me estimate for n_dof=48 (16-atom cell):
    print(f"\n=== Scaling estimates for large cells ===")
    for n_dof_est in [6, 18, 36, 48]:
        for R in [6, 12, 24]:
            svd_cost = R**2  # relative
            for r1 in [min(4, n_dof_est), min(8, n_dof_est)]:
                tucker_cost = r1 * R**2 / n_dof_est  # relative, normalized
                print(f"  n_dof={n_dof_est}, SVD R={R}: cost={svd_cost}, "
                      f"Tucker r1={r1}: cost_ratio={tucker_cost/svd_cost:.2f}x")


def test_tucker_transport():
    """Test Tucker decomposition for transport accuracy."""
    print(f"\n=== Tucker transport test ===")
    print("(Would need to implement Tucker kernel in separable.py)")
    print("For now, just comparing reconstruction errors.")


if __name__ == "__main__":
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    import h5py
    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(phonon)
    nat_prim = len(phonon.primitive.masses)
    masses_super = phonon.supercell.masses

    M_stacked, trans_atoms = build_M_stacked(
        fc3_raw, nat_prim, masses_super, prim_indices, slab_indices, ref_sc_atoms
    )
    n_dof = 3 * nat_prim
    dim_t = M_stacked.shape[1]

    analyze_block_structure(M_stacked, n_dof, dim_t)
    test_tucker_transport()
