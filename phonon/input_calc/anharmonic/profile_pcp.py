"""Profile PCP vs Dense vs Separable: operation counts and wall times.

Measures wall time of each major phase for the Si 2x2x2 parameters.
"""

import sys
import time
import itertools
from pathlib import Path

import numpy as np

S3_PERMS = list(itertools.permutations(range(3)))


def profile_dense_per_pair(n_dof, n_fft, n_freq):
    """Time one (q,q') iteration of the dense kernel."""
    rng = np.random.default_rng(0)
    GL_fft_qp = rng.normal(0, 1, (n_fft, n_dof, n_dof)).astype(complex)
    GL_fft_qd = rng.normal(0, 1, (n_fft, n_dof, n_dof)).astype(complex)
    Phi_L = rng.normal(0, 1, (n_dof, n_dof, n_dof)).astype(complex)
    Phi_R = rng.normal(0, 1, (n_dof, n_dof, n_dof)).astype(complex)
    Sigma = np.zeros((n_freq, n_dof, n_dof), dtype=complex)
    freq_sl = slice(0, n_freq)

    # Warmup
    product = GL_fft_qp[:, :, None, :, None] * GL_fft_qd[:, None, :, None, :]
    K = np.fft.ifft(product, axis=0)[freq_sl]
    temp = np.einsum('acd,wcdfe->wafe', Phi_L, K)
    Sigma += np.einsum('wafe,bef->wab', temp, Phi_R)

    # Timed
    N_rep = 20
    t0 = time.time()
    for _ in range(N_rep):
        product = GL_fft_qp[:, :, None, :, None] * GL_fft_qd[:, None, :, None, :]
        K = np.fft.ifft(product, axis=0)[freq_sl]
        temp = np.einsum('acd,wcdfe->wafe', Phi_L, K)
        Sigma += np.einsum('wafe,bef->wab', temp, Phi_R)
    return (time.time() - t0) / N_rep


def profile_separable_per_pair(n_dof, n_fft, n_freq, R):
    """Time one (q,q') iteration of optimized separable kernel."""
    rng = np.random.default_rng(0)
    F_qp = rng.normal(0, 1, (R, n_dof, n_dof)).astype(complex)
    F_diff_conj = rng.normal(0, 1, (R, n_dof, n_dof)).astype(complex)
    UL_hat = rng.normal(0, 1, (n_fft, n_dof, R)).astype(complex)
    VL_hat = rng.normal(0, 1, (n_fft, n_dof, R)).astype(complex)
    Sigma = np.zeros((n_freq, n_dof, n_dof), dtype=complex)
    freq_sl = slice(0, n_freq)

    # Warmup
    M1 = np.einsum('rac,wdr->wacd', F_qp, UL_hat)
    M2 = np.einsum('wcs,sbd->wcbd', VL_hat, F_diff_conj)
    conv_hat = np.einsum('wacd,wcbd->wab', M1, M2)
    Sigma += np.fft.ifft(conv_hat, axis=0)[freq_sl]

    N_rep = 20
    t0 = time.time()
    for _ in range(N_rep):
        M1 = np.einsum('rac,wdr->wacd', F_qp, UL_hat)
        M2 = np.einsum('wcs,sbd->wcbd', VL_hat, F_diff_conj)
        conv_hat = np.einsum('wacd,wcbd->wab', M1, M2)
        Sigma += np.fft.ifft(conv_hat, axis=0)[freq_sl]
    return (time.time() - t0) / N_rep


def profile_pcp_per_pair(n_dof, n_fft, n_freq, N_c):
    """Time one (q,q') iteration of the PCP kernel."""
    rng = np.random.default_rng(0)
    gLp = rng.normal(0, 1, (N_c, N_c, 3, 3, n_fft)).astype(complex)
    gLd = rng.normal(0, 1, (N_c, N_c, 3, 3, n_fft)).astype(complex)
    f_modes_ext = rng.normal(0, 1, (3, N_c, n_dof)).astype(complex)
    lam_outer = rng.normal(0, 1, (N_c, N_c))
    freq_sl = slice(0, n_freq)
    Sigma = np.zeros((n_freq, n_dof, n_dof), dtype=complex)

    # Warmup
    sum_prod = np.zeros((N_c, N_c, 3, 3, n_fft), dtype=complex)
    for s1, s2, s3 in S3_PERMS:
        for s1p, s2p, s3p in S3_PERMS:
            sum_prod[:, :, s1, s1p, :] += gLp[:, :, s2, s2p, :] * gLd[:, :, s3, s3p, :]
    conv_all = np.fft.ifft(sum_prod, axis=-1)[:, :, :, :, freq_sl]
    for s1 in range(3):
        for s1p in range(3):
            w = lam_outer[:, :, None] * conv_all[:, :, s1, s1p, :]
            Sigma += np.einsum('xyw,xa,yb->wab', w, f_modes_ext[s1], np.conj(f_modes_ext[s1p]))

    N_rep = 10
    t0 = time.time()
    for _ in range(N_rep):
        sum_prod = np.zeros((N_c, N_c, 3, 3, n_fft), dtype=complex)
        for s1, s2, s3 in S3_PERMS:
            for s1p, s2p, s3p in S3_PERMS:
                sum_prod[:, :, s1, s1p, :] += gLp[:, :, s2, s2p, :] * gLd[:, :, s3, s3p, :]
        conv_all = np.fft.ifft(sum_prod, axis=-1)[:, :, :, :, freq_sl]
        for s1 in range(3):
            for s1p in range(3):
                w = lam_outer[:, :, None] * conv_all[:, :, s1, s1p, :]
                Sigma += np.einsum('xyw,xa,yb->wab', w, f_modes_ext[s1], np.conj(f_modes_ext[s1p]))
    return (time.time() - t0) / N_rep


def profile_pcp_parts(n_dof, n_fft, n_freq, N_c):
    """Breakdown of PCP per-pair cost."""
    rng = np.random.default_rng(0)
    gLp = rng.normal(0, 1, (N_c, N_c, 3, 3, n_fft)).astype(complex)
    gLd = rng.normal(0, 1, (N_c, N_c, 3, 3, n_fft)).astype(complex)
    f_modes_ext = rng.normal(0, 1, (3, N_c, n_dof)).astype(complex)
    lam_outer = rng.normal(0, 1, (N_c, N_c))
    freq_sl = slice(0, n_freq)

    N_rep = 20

    # Part A: 36 products
    t0 = time.time()
    for _ in range(N_rep):
        sum_prod = np.zeros((N_c, N_c, 3, 3, n_fft), dtype=complex)
        for s1, s2, s3 in S3_PERMS:
            for s1p, s2p, s3p in S3_PERMS:
                sum_prod[:, :, s1, s1p, :] += gLp[:, :, s2, s2p, :] * gLd[:, :, s3, s3p, :]
    t_products = (time.time() - t0) / N_rep

    # Part B: IFFT
    t0 = time.time()
    for _ in range(N_rep):
        conv_all = np.fft.ifft(sum_prod, axis=-1)[:, :, :, :, freq_sl]
    t_ifft = (time.time() - t0) / N_rep

    # Part C: 9 einsums
    Sigma = np.zeros((n_freq, n_dof, n_dof), dtype=complex)
    t0 = time.time()
    for _ in range(N_rep):
        for s1 in range(3):
            for s1p in range(3):
                w = lam_outer[:, :, None] * conv_all[:, :, s1, s1p, :]
                Sigma += np.einsum('xyw,xa,yb->wab', w, f_modes_ext[s1], np.conj(f_modes_ext[s1p]))
    t_einsum = (time.time() - t0) / N_rep

    return t_products, t_ifft, t_einsum


def main():
    n_dof = 6
    n_freq = 101
    n_low = 8
    n_fft = 2 * (n_low + n_freq)

    print("=" * 70)
    print(f"Si 2x2x2: n_dof={n_dof}, n_freq={n_freq}, n_fft={n_fft}")
    print("=" * 70)

    # Per-pair costs
    print("\n--- Per (q,q') pair cost (one lesser channel) ---")

    t_dense = profile_dense_per_pair(n_dof, n_fft, n_freq)
    print(f"  Dense:        {t_dense*1000:.3f} ms/pair")

    for R in [6, 24, 48]:
        t_sep = profile_separable_per_pair(n_dof, n_fft, n_freq, R)
        print(f"  Sep R={R:<3d}:    {t_sep*1000:.3f} ms/pair")

    for N_c in [8, 24]:
        t_pcp = profile_pcp_per_pair(n_dof, n_fft, n_freq, N_c)
        print(f"  PCP N_c={N_c:<3d}:  {t_pcp*1000:.3f} ms/pair")

    # Total cost estimates for N_q=16 (4x4) and one SCBA iteration (× lesser + greater)
    n_kpts = 16
    n_pairs = n_kpts ** 2
    print(f"\n--- Estimated self-energy time (N_q={n_kpts}, ×2 for lesser+greater) ---")

    t = profile_dense_per_pair(n_dof, n_fft, n_freq)
    print(f"  Dense:        {2*n_pairs*t:.2f} s")

    for R in [6, 24, 48]:
        t = profile_separable_per_pair(n_dof, n_fft, n_freq, R)
        print(f"  Sep R={R:<3d}:    {2*n_pairs*t:.2f} s")

    for N_c in [8, 24]:
        t = profile_pcp_per_pair(n_dof, n_fft, n_freq, N_c)
        print(f"  PCP N_c={N_c:<3d}:  {2*n_pairs*t:.2f} s")

    # PCP breakdown
    print("\n--- PCP per-pair breakdown ---")
    for N_c in [8, 24]:
        tp, ti, te = profile_pcp_parts(n_dof, n_fft, n_freq, N_c)
        total = tp + ti + te
        print(f"  N_c={N_c}:")
        print(f"    36 products:  {tp*1000:.3f} ms ({tp/total*100:.0f}%)")
        print(f"    IFFT:         {ti*1000:.3f} ms ({ti/total*100:.0f}%)")
        print(f"    9 einsums:    {te*1000:.3f} ms ({te/total*100:.0f}%)")

    # PCP scaling
    print("\n--- PCP scaling with N_c ---")
    print(f"  {'N_c':>5} {'ms/pair':>10} {'ratio':>8} {'N_c^2 ratio':>12}")
    ref = None
    for N_c in [4, 8, 12, 16, 24, 32]:
        t = profile_pcp_per_pair(n_dof, n_fft, n_freq, N_c)
        if ref is None:
            ref = t
        ratio = t / ref
        expected = (N_c / 4) ** 2
        print(f"  {N_c:>5} {t*1000:>10.3f} {ratio:>8.1f}x     (N_c^2: {expected:.1f}x)")

    # Operation count comparison
    print("\n--- Theoretical operation counts per (q,q') pair ---")
    print(f"  Dense:")
    print(f"    Outer product:   n_fft × n_dof^4 = {n_fft} × {n_dof**4} = {n_fft * n_dof**4:,}")
    print(f"    IFFT:            n_fft × n_dof^4 × log2(n_fft) = {n_fft * n_dof**4 * np.log2(n_fft):,.0f}")
    print(f"    Einsum1 (acd,wcdfe->wafe): n_freq × n_dof^5 = {n_freq * n_dof**5:,}")
    print(f"    Einsum2 (wafe,bef->wab):   n_freq × n_dof^4 = {n_freq * n_dof**4:,}")
    print(f"    TOTAL: ~{n_fft * n_dof**4 + n_fft * n_dof**4 * np.log2(n_fft) + n_freq * n_dof**5 + n_freq * n_dof**4:,.0f}")

    print(f"  Separable R=24:")
    R = 24
    print(f"    M1 (rac,wdr->wacd):  n_fft × n_dof^3 × R = {n_fft * n_dof**3 * R:,}")
    print(f"    M2 (wcs,sbd->wcbd):  n_fft × n_dof^3 × R = {n_fft * n_dof**3 * R:,}")
    print(f"    S  (wacd,wcbd->wab): n_fft × n_dof^4 = {n_fft * n_dof**4:,}")
    print(f"    IFFT:                n_fft × n_dof^2 × log2(n_fft) = {n_fft * n_dof**2 * np.log2(n_fft):,.0f}")
    total_sep = 2 * n_fft * n_dof**3 * R + n_fft * n_dof**4 + n_fft * n_dof**2 * np.log2(n_fft)
    print(f"    TOTAL: ~{total_sep:,.0f}")

    print(f"  PCP N_c=24:")
    N_c = 24
    print(f"    36 products:  36 × N_c^2 × n_fft = {36 * N_c**2 * n_fft:,}")
    print(f"    IFFT:         N_c^2 × 9 × n_fft × log2(n_fft) = {N_c**2 * 9 * n_fft * np.log2(n_fft):,.0f}")
    print(f"    9 einsums (xyw,xa,yb->wab): 9 × N_c^2 × n_freq × n_dof = {9 * N_c**2 * n_freq * n_dof:,}")
    total_pcp = 36 * N_c**2 * n_fft + N_c**2 * 9 * n_fft * np.log2(n_fft) + 9 * N_c**2 * n_freq * n_dof
    print(f"    TOTAL: ~{total_pcp:,.0f}")
    print(f"    Ratio PCP/Dense: {total_pcp / (n_fft * n_dof**4 + n_fft * n_dof**4 * np.log2(n_fft) + n_freq * n_dof**5 + n_freq * n_dof**4):.1f}")

    # Memory comparison
    print("\n--- Memory for g_hat arrays (PCP) ---")
    for N_c in [8, 24]:
        mem = N_c**2 * 9 * n_kpts * n_fft * 16 / 1e6  # complex128
        print(f"  N_c={N_c}, N_q={n_kpts}: g_hat = {mem:.1f} MB")
        sum_prod_mem = N_c**2 * 9 * n_fft * 16 / 1e6
        print(f"    sum_prod per pair: {sum_prod_mem:.1f} MB")

    # The key insight: PCP vs Dense element count
    print("\n--- Why PCP is slow: element count per (q,q') pair ---")
    print(f"  Dense IFFT input:   n_fft × n_dof^4 = {n_fft * n_dof**4:,} elements")
    print(f"  PCP   IFFT input:   N_c^2 × 9 × n_fft (N_c=8)  = {8**2 * 9 * n_fft:,} elements")
    print(f"  PCP   IFFT input:   N_c^2 × 9 × n_fft (N_c=24) = {24**2 * 9 * n_fft:,} elements")
    print(f"  Sep   IFFT input:   n_fft × n_dof^2 = {n_fft * n_dof**2:,} elements")
    print()
    print(f"  Dense product:      n_fft × n_dof^4 = {n_fft * n_dof**4:,}")
    print(f"  PCP 36 products:    36 × N_c^2 × n_fft (N_c=8)  = {36 * 8**2 * n_fft:,}")
    print(f"  PCP 36 products:    36 × N_c^2 × n_fft (N_c=24) = {36 * 24**2 * n_fft:,}")


if __name__ == "__main__":
    main()
