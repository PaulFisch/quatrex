"""Investigate the ASR violation in the raw FC3 and its effect on transport.

Questions:
1. How large is the raw ASR violation?
2. What fraction of M_stacked lives in the ASR subspace?
3. Does cleaning the FC3 (enforcing ASR on the raw tensor) help?
4. Is the 68% a supercell-size artifact?
"""

import sys
from pathlib import Path

import numpy as np
import h5py

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.separable import (
    build_supercell_mapping,
    decompose_fc3_supercell,
    enforce_asr_fc3_matrices,
    reconstruction_error,
)


def investigate():
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(phonon)
    nat_prim = len(phonon.primitive.masses)
    masses_super = phonon.supercell.masses
    n_dof = 3 * nat_prim

    print(f"FC3 shape: {fc3_raw.shape}")
    print(f"nat_prim: {nat_prim}, n_dof: {n_dof}")
    print(f"Supercell atoms: {len(masses_super)}")

    # --- 1. Raw FC3 ASR violation ---
    # ASR: sum over second atom index should be zero
    # fc3_raw shape: (nat_prim, n_super, n_super, 3, 3, 3) or (n_super, n_super, n_super, 3, 3, 3)
    print(f"\n--- Raw FC3 ASR violation ---")
    if fc3_raw.shape[0] == nat_prim:
        # Compact format: (nat_prim, n_super, n_super, 3, 3, 3)
        n_super = fc3_raw.shape[1]
        print(f"Compact format: ({nat_prim}, {n_super}, {n_super}, 3, 3, 3)")
        # ASR: sum_j Phi(i, j, k, a, b, c) = 0 for all i, k, a, c
        asr_viol_2 = np.sum(fc3_raw, axis=1)  # sum over 2nd atom -> (nat_prim, n_super, 3, 3, 3)
        asr_viol_3 = np.sum(fc3_raw, axis=2)  # sum over 3rd atom -> (nat_prim, n_super, 3, 3, 3)
    else:
        n_super = fc3_raw.shape[0]
        print(f"Full format: ({n_super}, {n_super}, {n_super}, 3, 3, 3)")
        asr_viol_2 = np.sum(fc3_raw, axis=1)
        asr_viol_3 = np.sum(fc3_raw, axis=2)

    fc3_norm = np.linalg.norm(fc3_raw)
    viol_2_norm = np.linalg.norm(asr_viol_2)
    viol_3_norm = np.linalg.norm(asr_viol_3)
    print(f"||FC3||_F = {fc3_norm:.4e}")
    print(f"||sum_j FC3(i,j,k)|| / ||FC3|| = {viol_2_norm / fc3_norm:.4e}  (ASR on 2nd index)")
    print(f"||sum_k FC3(i,j,k)|| / ||FC3|| = {viol_3_norm / fc3_norm:.4e}  (ASR on 3rd index)")

    # Also check first index
    asr_viol_1 = np.sum(fc3_raw, axis=0)
    viol_1_norm = np.linalg.norm(asr_viol_1)
    print(f"||sum_i FC3(i,j,k)|| / ||FC3|| = {viol_1_norm / fc3_norm:.4e}  (ASR on 1st index)")

    # --- 2. M_stacked analysis ---
    print(f"\n--- M_stacked analysis ---")
    trans_atoms = np.where(slab_indices == 0)[0]
    n_trans = len(trans_atoms)
    dim_t = n_trans * 3
    print(f"Same-slab atoms: {n_trans}, dim_t: {dim_t}")

    # Build M_stacked manually to analyze
    CONVERSION_FC3_THZ = 9.648534e18  # from eV/Ang^3 to THz^5/2 * amu^3/2
    # Actually get it from the code
    from phonon_inputs.separable import CONVERSION_FC3_THZ as CONV

    # Get M_stacked via full-rank decomposition
    F_list, H, svals, ta = decompose_fc3_supercell(
        fc3_raw, nat_prim, masses_super, prim_indices, slab_indices,
        ref_sc_atoms, rank=None, tol=1e-15, enforce_asr=False,
    )
    R_full = len(F_list)
    print(f"Full SVD rank: {R_full} out of {dim_t}")
    print(f"Singular values: {svals}")

    # Now with ASR
    F_list_asr, H_asr, svals_asr, ta_asr = decompose_fc3_supercell(
        fc3_raw, nat_prim, masses_super, prim_indices, slab_indices,
        ref_sc_atoms, rank=None, tol=1e-15, enforce_asr=True,
    )
    R_asr = len(F_list_asr)
    print(f"\nFull SVD rank (with ASR): {R_asr} out of {dim_t}")
    print(f"Singular values (ASR): {svals_asr}")

    # Compare total energy in singular values
    total_energy = np.sum(svals**2)
    total_energy_asr = np.sum(svals_asr**2)
    print(f"\nTotal SV energy (no ASR): {total_energy:.4e}")
    print(f"Total SV energy (ASR):    {total_energy_asr:.4e}")
    print(f"Fraction removed by ASR:  {1 - total_energy_asr / total_energy:.4f}")
    print(f"  => ASR removes {(1 - total_energy_asr / total_energy)*100:.1f}% of M_stacked energy")

    # --- 3. Analyze the ASR-violating component ---
    print(f"\n--- ASR-violating component analysis ---")
    # Reconstruct M_stacked from SVD
    # M_stacked = U @ diag(S) @ Vt
    # We have F_list (left singular vectors reshaped) and H (right factor = S * V)
    # But let's just build M_stacked directly

    # Build M_stacked directly
    n_atoms_super = len(masses_super)
    dim_full = n_atoms_super * 3
    filt_idx = np.concatenate([3 * trans_atoms[:, None] + np.arange(3)], axis=0).ravel()
    filt_idx = np.sort(filt_idx)

    M_stacked = np.zeros((n_dof * dim_t, dim_t))
    for i_prim in range(nat_prim):
        s_i = ref_sc_atoms[i_prim]
        m_i = masses_super[s_i]
        for alpha in range(3):
            a = 3 * i_prim + alpha
            if fc3_raw.shape[0] == nat_prim:
                block = fc3_raw[i_prim, :, :, alpha, :, :]
            else:
                block = fc3_raw[s_i, :, :, alpha, :, :]
            mat = block.transpose(0, 2, 1, 3).reshape(dim_full, dim_full)
            mat_filt = mat[np.ix_(filt_idx, filt_idx)]
            m_row = np.repeat(np.sqrt(masses_super[trans_atoms]), 3)
            m_col = np.repeat(np.sqrt(masses_super[trans_atoms]), 3)
            mat_filt = mat_filt / (np.sqrt(m_i) * m_row[:, None] * m_col[None, :])
            mat_filt = mat_filt * CONV
            M_stacked[a * dim_t:(a + 1) * dim_t, :] = mat_filt

    M_norm = np.linalg.norm(M_stacked)
    print(f"||M_stacked||_F = {M_norm:.4e}")

    # Apply ASR projection
    M_asr = enforce_asr_fc3_matrices(M_stacked, nat_prim, trans_atoms, prim_indices)
    M_asr_norm = np.linalg.norm(M_asr)
    M_diff = M_stacked - M_asr
    M_diff_norm = np.linalg.norm(M_diff)

    print(f"||M_stacked (ASR projected)||_F = {M_asr_norm:.4e}")
    print(f"||M_stacked - M_asr||_F = {M_diff_norm:.4e}")
    print(f"Fraction in ASR subspace: {M_diff_norm / M_norm:.4f}")
    print(f"  => {M_diff_norm / M_norm * 100:.1f}% of M_stacked energy is in ASR-violating space")

    # --- 4. Check: is the ASR-violating part physical or numerical? ---
    print(f"\n--- ASR violation structure ---")
    # Build the projector P
    P = np.zeros((dim_t, n_dof))
    counts = np.zeros(nat_prim)
    for s_local, s_global in enumerate(trans_atoms):
        kappa = prim_indices[s_global]
        counts[kappa] += 1
    for s_local, s_global in enumerate(trans_atoms):
        kappa = prim_indices[s_global]
        w = 1.0 / np.sqrt(counts[kappa])
        for beta in range(3):
            P[s_local * 3 + beta, kappa * 3 + beta] = w
    PPt = P @ P.T
    print(f"Projector rank: {np.linalg.matrix_rank(PPt)}")
    print(f"dim_t: {dim_t}, n_dof: {n_dof}")
    print(f"PPt projects onto {np.linalg.matrix_rank(PPt)}/{dim_t} dimensional subspace")

    # Decompose M_stacked into ASR-compliant and ASR-violating parts
    # For each block M_a, the ASR-violating part is:
    #   M_viol = PPt @ M_a @ PPt + PPt @ M_a @ (I-PPt) + (I-PPt) @ M_a @ PPt
    # Let's compute the norms of each component
    I_minus = np.eye(dim_t) - PPt
    norms_pp = 0.0  # PPt M PPt (pure ASR)
    norms_pm = 0.0  # PPt M (I-PPt) (mixed)
    norms_mp = 0.0  # (I-PPt) M PPt (mixed)
    norms_mm = 0.0  # (I-PPt) M (I-PPt) (ASR-compliant)

    for a in range(n_dof):
        M_a = M_stacked[a * dim_t:(a + 1) * dim_t, :]
        norms_pp += np.linalg.norm(PPt @ M_a @ PPt)**2
        norms_pm += np.linalg.norm(PPt @ M_a @ I_minus)**2
        norms_mp += np.linalg.norm(I_minus @ M_a @ PPt)**2
        norms_mm += np.linalg.norm(I_minus @ M_a @ I_minus)**2

    total = norms_pp + norms_pm + norms_mp + norms_mm
    print(f"\nBlock decomposition of M_stacked energy:")
    print(f"  PPt M PPt  (pure ASR viol.):  {norms_pp/total*100:6.2f}%")
    print(f"  PPt M Q    (mixed, removed):  {norms_pm/total*100:6.2f}%")
    print(f"  Q M PPt    (mixed, removed):  {norms_mp/total*100:6.2f}%")
    print(f"  Q M Q      (ASR compliant):   {norms_mm/total*100:6.2f}%")
    print(f"  Total removed by (I-PPt)M(I-PPt): {(1 - norms_mm/total)*100:.2f}%")

    # --- 5. What if we only project one side? ---
    print(f"\n--- One-sided projection test ---")
    # Maybe two-sided is too aggressive. What about one-sided?
    # ASR requires: sum_j M_a[j, k] = 0 for all a, k
    # This is: PPt @ M_a = 0 (left-side constraint only)
    # One-sided: M_corrected = (I - PPt) @ M_a (preserves right side)

    M_left_only = np.zeros_like(M_stacked)
    for a in range(n_dof):
        M_a = M_stacked[a * dim_t:(a + 1) * dim_t, :]
        M_left_only[a * dim_t:(a + 1) * dim_t, :] = I_minus @ M_a

    M_left_norm = np.linalg.norm(M_left_only)
    print(f"||M_left_projected||_F / ||M|| = {M_left_norm / M_norm:.4f}")
    print(f"  => Left-only removes {(1 - M_left_norm / M_norm) * 100:.1f}%")

    # Check if FC3 from phono3py is supposed to satisfy ASR
    print(f"\n--- FC3 symmetry check ---")
    # Check permutation symmetry: Phi(i,j,k) = Phi(j,i,k) etc.
    if fc3_raw.shape[0] == nat_prim:
        print("FC3 in compact format — checking translational symmetry of M_stacked blocks")
        # For compact format, check: M_a[s2, s3] should equal M_a[s3, s2]
        sym_err = 0.0
        for a in range(n_dof):
            M_a = M_stacked[a * dim_t:(a + 1) * dim_t, :]
            sym_err += np.linalg.norm(M_a - M_a.T)**2
        sym_err = np.sqrt(sym_err) / M_norm
        print(f"  Symmetry error (M_a vs M_a^T): {sym_err:.4e}")


if __name__ == "__main__":
    investigate()
