"""Test double factorization of the separable FC3 decomposition.

Double factorization (DF, from quantum chemistry):
1. SVD of M_stacked: M = sum_r sigma_r * u_r ⊗ v_r  (rank R)
   -> F_r[a, d] = u_r reshaped, h_r[c] = sigma_r * v_r[c]

2. Secondary SVD of each F_r: F_r[a, d] = sum_k U^r[a,k] * lambda^r_k * W^r[d,k]
   -> F_r has its own internal rank R_r

If R_r << dim_t, then the self-energy kernel can exploit this:
  Instead of: G @ h_r  -> n_dof-vector  (R times)
  We get:     G @ W^r_k -> scalar       (sum_r R_r times)
  Then reconstruct via U^r factors.

The total cost becomes O(sum_r R_r) convolutions instead of O(R^2).

But WAIT: the current separable kernel's R^2 comes from needing all
(r,s) PAIRS in the convolution, not from the F_r side. Let me re-examine.

Actually the cost is:
  For each q-pair:
    For each (r,s): convolve v_r(w,q') * u_s(w,q-q')  -> one scalar conv
    Then accumulate: Sigma += F_hat_r @ conv_rs @ F_hat_s^H

So R^2 convolutions, each O(n_fft log n_fft), plus R^2 * n_dof^2 * n_freq
for the accumulation.

Double factorization of F_r would help the ACCUMULATION step:
  F_r @ conv_rs @ F_s^H = (U^r Lambda^r W^{rT}) @ conv_rs @ (W^s Lambda^s U^{sT})

This doesn't reduce R^2 convolutions, but could help with the n_dof^2 factor.

ALTERNATIVE approach: the R^2 comes from the fact that the left and right
FC3 vertices use DIFFERENT decompositions. What if we could find a
decomposition where the interaction matrix is (block-)diagonal?

Let me investigate several angles:
1. Internal rank of F_r matrices
2. Interaction structure between (r,s) pairs
3. Whether a rotation can block-diagonalize the cross-terms
4. Whether the PCP residual is low-rank
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


def analyze_double_factorization():
    """Analyze the internal rank structure of SVD factors."""
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    import h5py
    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(phonon)
    nat_prim = len(phonon.primitive.masses)
    masses_super = phonon.supercell.masses
    n_dof = 3 * nat_prim

    # Full SVD
    F_list, H, svals, trans_atoms = decompose_fc3_supercell(
        fc3_raw, nat_prim, masses_super, prim_indices, slab_indices,
        ref_sc_atoms, rank=None, tol=1e-15,
    )
    R = len(F_list)
    dim_t = H.shape[0]

    print(f"=== Double Factorization Analysis ===")
    print(f"n_dof={n_dof}, dim_t={dim_t}, R={R}")
    print(f"Singular values: {svals}")

    # --- 1. Internal rank of each F_r ---
    print(f"\n--- Internal rank of F_r matrices ---")
    print(f"Each F_r has shape ({n_dof}, {dim_t})")

    for r in range(min(R, 12)):
        F_r = F_list[r]  # shape (n_dof, dim_t)
        u, s, vt = np.linalg.svd(F_r, full_matrices=False)
        s_rel = s / s[0] if s[0] > 0 else s
        rank_99 = np.sum(np.cumsum(s**2) / np.sum(s**2) < 0.99) + 1
        rank_999 = np.sum(np.cumsum(s**2) / np.sum(s**2) < 0.999) + 1
        print(f"  F_{r}: svals = {s_rel[:6]}, rank(99%) = {rank_99}, rank(99.9%) = {rank_999}")

    # --- 2. What would DF buy us? ---
    print(f"\n--- DF cost analysis ---")
    print(f"Current separable kernel cost per q-pair:")
    print(f"  Convolutions: R^2 = {R**2}")
    print(f"  Accumulation: R^2 * n_dof^2 = {R**2 * n_dof**2}")
    print(f"  (n_fft and n_freq factors omitted)")

    # DF doesn't reduce convolutions, only accumulation
    # But the accumulation is typically not the bottleneck for small n_dof

    # --- 3. Interaction structure ---
    print(f"\n--- Cross-rank interaction structure ---")
    # The self-energy sum is:
    #   Sigma = sum_{r,s} F_hat_r @ conv(v_r, u_s) @ F_hat_s^H
    # Can we find (r,s) pairs that contribute negligibly?
    #
    # In real space, the "coupling" is:
    #   C_rs = sum_a F_r[a,:] @ F_s[a,:]^T  (dim_t x dim_t matrix)
    # For orthonormal U columns: C_rs = delta_{rs} * I
    # But in q-space, the Fourier transform breaks this.
    #
    # Alternative: look at singular value products sigma_r * sigma_s
    # The contribution of pair (r,s) is proportional to sigma_r * sigma_s
    #
    # If we keep only pairs where sigma_r * sigma_s > threshold:
    for frac in [0.01, 0.001, 0.0001]:
        threshold = frac * svals[0]**2
        n_pairs = sum(1 for r in range(R) for s in range(R)
                      if svals[r] * svals[s] > threshold)
        print(f"  Pairs with sigma_r*sigma_s > {frac:.0e}*sigma_max^2: "
              f"{n_pairs}/{R**2} ({n_pairs/R**2*100:.1f}%)")

    # --- 4. Can we reduce R^2 to R via symmetry? ---
    print(f"\n--- Exploiting Phi symmetry to reduce R^2 ---")
    # The self-energy diagram has TWO FC3 vertices.
    # For the left vertex: Phi_L = T(q') @ M_a @ T(q-q')^T
    # For the right vertex: Phi_R = conj(Phi_L) (Hermitian conjugate)
    #
    # So Phi_L and Phi_R share the SAME decomposition!
    # This means: conv(v_r, u_s) has the structure of the
    # left-right contraction of the SAME tensor, which is
    # like computing <Phi | G⊗G | Phi>.
    #
    # In the SVD basis: this is sum_{r,s} sigma_r * sigma_s * ...
    # which can potentially be rewritten as a single sum.
    #
    # Specifically: Phi_L[a,c,d] = sum_r sigma_r * F_r[a,c] * h_r[d]
    # The K-tensor involves: sum_{c,d} Phi_L[a,c,d] * K[c,d,f,e] * Phi_R[b,e,f]
    # = sum_{r,s} sigma_r * sigma_s * sum_c F_r[a,c] * sum_d h_r[d] *
    #   K[c,d,f,e] * h_s[e] * F_s[b,f]
    #
    # The inner contraction is: sum_d h_r[d] * K[c,d,f,e] * h_s[e]
    # = sum_d sum_e h_r[d] * IFFT{G[c,e] * G[d,f]}[w] * h_s[e]
    #
    # This gives a (c, f)-indexed matrix for each (r,s) pair.
    # Cannot avoid R^2 unless the h_r * K * h_s contraction has structure.

    # --- 5. Rank of the interaction kernel ---
    print(f"\n--- Rank of interaction kernel K ---")
    # The K-tensor at each frequency has shape (dim_t, dim_t, dim_t, dim_t)
    # which is too large to compute here. But we can check:
    # does K factorize as K[c,d,f,e] = sum_p A[c,f,p] * B[d,e,p]?
    # This would reduce R^2 to R * n_interp.
    #
    # K = IFFT{ G^<(q')[c,f] * G^<(q-q')[d,e] }
    # At each w: K_w = conv(G1, G2) where G1 has (c,f) indices and G2 (d,e)
    # The convolution IS a factorization: it's the Hadamard product in freq domain
    # followed by IFFT. So K_w = IFFT{ hat_G1[c,f] * hat_G2[d,e] }
    #
    # This already factorizes as a sum over frequencies:
    # K_w[c,d,f,e] = sum_{w'} G1[w',c,f] * G2[w-w',d,e]
    # = rank-n_fft factorization of the 4-tensor.
    #
    # But the stacked SVD already exploits this via the FFT.

    # --- 6. Alternative: reduce R via better decomposition ---
    print(f"\n--- Randomized range-finding for effective rank ---")
    # The reconstruction error at each rank
    from phonon_inputs.separable import CONVERSION_FC3_THZ as CONV

    # Build M_stacked
    trans_atoms_arr = np.where(slab_indices == 0)[0]
    n_trans = len(trans_atoms_arr)
    n_atoms_super = len(masses_super)
    dim_full = n_atoms_super * 3
    filt_idx = np.sort(np.concatenate([3 * trans_atoms_arr[:, None] + np.arange(3)], axis=0).ravel())

    M_stacked = np.zeros((n_dof * dim_t, dim_t))
    for i_prim in range(nat_prim):
        s_i = ref_sc_atoms[i_prim]
        m_i = masses_super[s_i]
        for alpha in range(3):
            a = 3 * i_prim + alpha
            block = fc3_raw[s_i, :, :, alpha, :, :]
            mat = block.transpose(0, 2, 1, 3).reshape(dim_full, dim_full)
            mat_filt = mat[np.ix_(filt_idx, filt_idx)]
            m_row = np.repeat(np.sqrt(masses_super[trans_atoms_arr]), 3)
            m_col = np.repeat(np.sqrt(masses_super[trans_atoms_arr]), 3)
            mat_filt = mat_filt / (np.sqrt(m_i) * m_row[:, None] * m_col[None, :])
            mat_filt = mat_filt * CONV
            M_stacked[a * dim_t:(a + 1) * dim_t, :] = mat_filt

    U_full, S_full, Vt_full = np.linalg.svd(M_stacked, full_matrices=False)

    # --- 7. Key idea: PAIR interaction matrix ---
    print(f"\n--- Pair interaction matrix (sigma_r * sigma_s) ---")
    # The contribution of (r,s) to the self-energy scales as sigma_r * sigma_s.
    # The matrix sigma_r * sigma_s = sigma ⊗ sigma is rank-1!
    # So we can write: sum_{r,s} sigma_r * sigma_s * (...) = (sum_r sigma_r * ...)^2
    # Wait, that's only true if the (...) part also factorizes.
    #
    # In fact, the self-energy IS a quadratic form in Phi:
    #   Sigma ~ Phi @ K @ Phi^H
    # where K involves the Green's functions.
    #
    # With the SVD, Phi = sum_r sigma_r * F_r ⊗ h_r, so:
    #   Sigma ~ sum_{r,s} sigma_r * sigma_s * (F_r @ K_rs @ F_s^H)
    # where K_rs = h_r^T @ K @ h_s (projected K-tensor).
    #
    # If we define: Phi_proj(w) = sum_r sigma_r * F_r * v_r(w)
    # where v_r(w) = sum_c h_r[c] * G[w, c, :]  (an n_dof-vector)
    # Then: Sigma ~ IFFT{ Phi_proj(w') ⊗ Phi_proj(w-w')^H }  -- NO!
    # The two G's come from different q-points.

    # --- 8. The REAL question: can we avoid R^2? ---
    print(f"\n--- Can we reduce from R^2 to R? ---")
    # The R^2 is fundamental: the self-energy involves TWO vertices,
    # and each vertex independently carries its own rank index.
    # To reduce to R, we'd need the cross-terms to vanish.
    #
    # This happens if: F_hat_r(q) and h_hat_r(q) are simultaneously
    # diagonal in some basis. But F and h come from the SVD of M_stacked,
    # where F encodes the "left" spatial structure and h the "right".
    #
    # The ONLY way to get R cost is to have a single-vertex
    # formulation: Sigma ~ sum_r (...) where each r contributes
    # independently. This is only possible if Phi @ K @ Phi^H
    # can be written as Phi @ (diagonal in SVD basis) @ Phi^H.
    #
    # This would require K projected into the SVD basis to be diagonal:
    # h_r^T @ K_w @ h_s = delta_{rs} * k_r(w)
    #
    # Is K diagonalizable in the h-basis?
    # K depends on G (which changes at each SCBA iteration), so we
    # can't pre-diagonalize. But maybe the dominant contributions come
    # from near-diagonal terms.

    # Test: compute the "diagonal dominance" of the projected K
    # For this we need actual Green's functions, which we don't have here.
    # Instead, compute the "structural" overlap:
    # O_rs = sum_d h_r[d] * h_s[d] = delta_rs (orthogonal by SVD)
    # So in real space, the h vectors are orthogonal.
    # In q-space (after Fourier transform), they may mix.

    # Let's check h-vector orthogonality structure:
    H_mat = H  # shape (dim_t, R)
    # H = diag(S) @ V, so h_r = S[r] * V[r, :] -> H[:, r]
    # Inner products:
    gram = H_mat.T @ H_mat  # (R, R) -- should be diag(S^2 * ...)
    gram_norm = gram / np.sqrt(np.diag(gram)[:, None] * np.diag(gram)[None, :])
    print(f"  H Gram matrix (normalized) off-diagonal max: {np.max(np.abs(gram_norm - np.diag(np.diag(gram_norm)))):.6f}")
    print(f"  (0 = perfectly orthogonal h vectors)")

    # The h vectors ARE orthogonal (V from SVD), so the Gram is diagonal.
    # The R^2 coupling comes from the FOURIER TRANSFORM mixing them.

    # --- 9. Estimate: at what R does the SVD error become acceptable? ---
    print(f"\n--- SVD error at practical ranks ---")
    cum_energy = np.cumsum(S_full**2) / np.sum(S_full**2)
    for thresh in [0.99, 0.999, 0.9999, 0.99999]:
        R_needed = np.searchsorted(cum_energy, thresh) + 1
        print(f"  {thresh*100:.3f}% energy: R = {R_needed}")

    # --- 10. Randomized sketch: does a random projection preserve transport? ---
    print(f"\n--- Summary ---")
    print(f"The R^2 scaling in the separable kernel is FUNDAMENTAL:")
    print(f"  - It arises from the two-vertex structure of the self-energy diagram")
    print(f"  - Each vertex independently contributes rank R")
    print(f"  - Cross-rank terms (r,s) with r≠s are coupled through the")
    print(f"    frequency convolution of differently-projected Green's functions")
    print(f"  - The h vectors are orthogonal in real space but mix under")
    print(f"    Fourier transform, so cross-terms don't vanish in q-space")
    print(f"")
    print(f"Possible reductions:")
    print(f"  1. Use lower R (accept truncation error) — already implemented")
    print(f"  2. Pair pruning: skip pairs where sigma_r * sigma_s < threshold")
    print(f"  3. Tucker mode-1 compression: reduce n_dof in accumulation step")
    print(f"  4. Batched convolution: current impl already uses FFT batching")
    print(f"  5. ISDF-style interpolation: replace dim_t with N_g grid points")


if __name__ == "__main__":
    analyze_double_factorization()
