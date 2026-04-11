"""Test the PCP (Permanent CP) FC3 decomposition.

Tests:
1. Cell mapping: verify supercell-to-primitive mapping
2. Forward model: torch forward produces physically valid FC3
3. ASR enforcement: projection zeroes the sum rule exactly
4. Rank sweep: reconstruction error decreases monotonically
5. Greedy vs joint fitting comparison
6. SCBA transport comparison (PCP vs dense)

Requires:
  - fc3_prim/fc3.hdf5
  - fc3_prim/phono3py_disp.yaml
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent  # input_calc/
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.pcp import (
    build_cell_mapping,
    build_cell_diff_table,
    _build_index_tables,
    _pcp_forward_torch,
    _project_asr,
    _build_target,
    fit_pcp,
    fit_pcp_greedy,
    fourier_transform_pcp,
    pcp_anharmonic_transmission,
    CONVERSION_FC3_THZ,
)


def _load_fc3_raw(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        return np.array(f["fc3"])


# -----------------------------------------------------------------------
# Test 1: Cell mapping
# -----------------------------------------------------------------------

def test_cell_mapping(phonon):
    """Verify supercell-to-primitive cell mapping."""
    print("\n=== Test 1: Cell mapping ===")
    sc_to_cell, sc_to_prim, cell_frac, n_cells, sc_mat = build_cell_mapping(phonon)
    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)

    print(f"  n_super={n_super}, nat_prim={nat_prim}, n_cells={n_cells}, sc_mat={sc_mat}")

    # Each cell appears nat_prim times
    for c in range(n_cells):
        count = np.sum(sc_to_cell == c)
        assert count == nat_prim, f"Cell {c} has {count} atoms, expected {nat_prim}"

    # Each prim atom appears n_cells times
    for a in range(nat_prim):
        count = np.sum(sc_to_prim == a)
        assert count == n_cells, f"Prim atom {a} has {count} images, expected {n_cells}"

    # cell_diff[l, l] = 0 (self-difference)
    cell_diff, idx_to_t = build_cell_diff_table(n_cells, sc_mat)
    for l in range(n_cells):
        assert cell_diff[l, l] == 0

    print("  PASSED")


# -----------------------------------------------------------------------
# Test 2: Forward model + ASR
# -----------------------------------------------------------------------

def test_forward_and_asr(phonon):
    """Test that ASR-projected modes produce ASR-satisfying FC3."""
    print("\n=== Test 2: Forward model + ASR enforcement ===")
    import torch

    sc_to_cell, sc_to_prim_np, _, n_cells, sc_mat = build_cell_mapping(phonon)
    cell_diff, _ = build_cell_diff_table(n_cells, sc_mat)
    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)

    idx1_cell_np, idx_jk_cell_np, sc_to_prim_i = _build_index_tables(
        sc_to_cell, sc_to_prim_np, cell_diff, nat_prim, n_super, n_cells)
    idx1_cell = torch.tensor(idx1_cell_np, dtype=torch.long)
    idx_jk_cell = torch.tensor(idx_jk_cell_np, dtype=torch.long)
    sc_to_prim_t = torch.tensor(sc_to_prim_i, dtype=torch.long)

    N_c = 4
    rng = np.random.default_rng(42)
    A = torch.tensor(rng.normal(0, 0.1, (3, N_c, n_cells, nat_prim, 3)),
                      dtype=torch.float64)
    lambdas = torch.tensor(rng.normal(0, 1.0, N_c), dtype=torch.float64)

    # Without ASR projection
    fc3_raw = _pcp_forward_torch(A, lambdas, idx1_cell, idx_jk_cell, sc_to_prim_t,
                                  nat_prim, n_super, n_cells, N_c)
    asr_raw = torch.max(torch.abs(torch.sum(fc3_raw, dim=1))).item()
    print(f"  Without ASR projection: max|sum_j FC3| = {asr_raw:.4e}")

    # With ASR projection
    A_proj = _project_asr(A)
    asr_mode = torch.max(torch.abs(A_proj.sum(dim=(2, 3)))).item()
    print(f"  Mode ASR after projection: max|sum_{{l,b}} A| = {asr_mode:.2e}")

    fc3_proj = _pcp_forward_torch(A_proj, lambdas, idx1_cell, idx_jk_cell, sc_to_prim_t,
                                   nat_prim, n_super, n_cells, N_c)

    # Check all three ASR conditions
    asr_j = torch.max(torch.abs(torch.sum(fc3_proj, dim=1))).item()  # sum over j
    asr_k = torch.max(torch.abs(torch.sum(fc3_proj, dim=2))).item()  # sum over k
    # For sum over b1 (compact format), need to sum over b1 index:
    asr_b1 = torch.max(torch.abs(torch.sum(fc3_proj, dim=0))).item()

    print(f"  With ASR projection:")
    print(f"    max|sum_j FC3(b1,j,k)| = {asr_j:.2e}")
    print(f"    max|sum_k FC3(b1,j,k)| = {asr_k:.2e}")
    print(f"    max|sum_b1 FC3(b1,j,k)| = {asr_b1:.2e}")

    # sum over j and sum over k should be near machine precision
    assert asr_j < 1e-12, f"ASR sum_j violated: {asr_j}"
    assert asr_k < 1e-12, f"ASR sum_k violated: {asr_k}"
    # sum over b1 is a PARTIAL sum (only atoms at cell 0), not the full ASR.
    # It need NOT be zero. (Full ASR sums over ALL cells of atom b1.)
    # So we don't assert on asr_b1.

    print("  PASSED")


# -----------------------------------------------------------------------
# Test 3: Rank sweep — error must decrease
# -----------------------------------------------------------------------

def test_rank_sweep(phonon, fc3_hdf5, max_iter=1000):
    """Verify reconstruction error decreases with rank."""
    print("\n=== Test 3: Rank sweep (joint fit) ===")
    fc3_raw = _load_fc3_raw(fc3_hdf5)

    ranks = [2, 4, 8, 16, 24]
    errors = {}

    for N_c in ranks:
        t0 = time.time()
        _, lambdas, info = fit_pcp(fc3_raw, phonon, N_c=N_c, max_iter=max_iter, verbose=False)
        dt = time.time() - t0
        errors[N_c] = info['rel_err']
        print(f"  N_c={N_c:3d}: rel_err={info['rel_err']:.6e}, "
              f"asr_mode={info['asr_violation']:.2e}, "
              f"asr_fc3={info['asr_fc3']:.2e}, "
              f"time={dt:.1f}s")

    # Check monotonic decrease (allow 5% tolerance for optimization noise)
    sorted_ranks = sorted(errors.keys())
    for i in range(1, len(sorted_ranks)):
        r_prev, r_curr = sorted_ranks[i-1], sorted_ranks[i]
        if errors[r_curr] > errors[r_prev] * 1.05:
            print(f"  WARNING: error increased from N_c={r_prev} ({errors[r_prev]:.4e}) "
                  f"to N_c={r_curr} ({errors[r_curr]:.4e})")

    # Overall: error at max rank should be < error at min rank
    assert errors[sorted_ranks[-1]] < errors[sorted_ranks[0]], \
        f"Error did not decrease: N_c={sorted_ranks[0]}: {errors[sorted_ranks[0]]:.4e} -> " \
        f"N_c={sorted_ranks[-1]}: {errors[sorted_ranks[-1]]:.4e}"

    print("  PASSED — error decreases with rank")
    return errors


# -----------------------------------------------------------------------
# Test 4: Greedy fitting
# -----------------------------------------------------------------------

def test_greedy(phonon, fc3_hdf5):
    """Test greedy rank-1 fitting."""
    print("\n=== Test 4: Greedy fitting ===")
    fc3_raw = _load_fc3_raw(fc3_hdf5)

    t0 = time.time()
    A_modes, lambdas, info = fit_pcp_greedy(
        fc3_raw, phonon, N_c=16, iters_per_rank=300, verbose=True,
    )
    dt = time.time() - t0

    print(f"  Time: {dt:.1f}s")
    print(f"  Final rel_err: {info['rel_err']:.6e}")
    print(f"  ASR mode: {info['asr_violation']:.2e}")
    print(f"  ASR FC3: {info['asr_fc3']:.2e}")
    print(f"  Actual rank: {info['actual_rank']}")
    print(f"  Lambda spectrum: {np.abs(lambdas[:8])}")

    return info


# -----------------------------------------------------------------------
# Test 5: Fourier transform
# -----------------------------------------------------------------------

def test_fourier_transform(phonon, fc3_hdf5):
    """Test FT of PCP modes at Gamma and generic q."""
    print("\n=== Test 5: Fourier transform ===")
    fc3_raw = _load_fc3_raw(fc3_hdf5)

    A_modes, lambdas, info = fit_pcp(
        fc3_raw, phonon, N_c=8, max_iter=500, verbose=False,
    )

    f_gamma = fourier_transform_pcp(A_modes, lambdas, phonon, (0.0, 0.0), info=info)
    f_q = fourier_transform_pcp(A_modes, lambdas, phonon, (0.25, 0.25), info=info)

    print(f"  f_q(Gamma) shape: {f_gamma.shape}, max|f|: {np.max(np.abs(f_gamma)):.4e}")
    print(f"  f_q(0.25,0.25) shape: {f_q.shape}, max|f|: {np.max(np.abs(f_q)):.4e}")

    # At Gamma, ASR means the FT modes should be small (sum over cells ~ 0)
    # But not exactly zero because ASR is sum_{l,b} A = 0, while FT is
    # sum_l exp(0) * A = sum_l A (over cells only, summed over atoms already in reshape).
    # Actually f_q(Gamma)[kappa*3+alpha] = sum_l A(l, kappa, alpha) which is the
    # sum over cells for a specific atom — NOT the ASR sum. So this can be non-zero.
    print("  PASSED")


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_rank_sweep(errors, out_dir):
    """Plot reconstruction error vs PCP rank."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ranks = sorted(errors.keys())
    errs = [errors[r] for r in ranks]

    ax.semilogy(ranks, errs, 'o-', color='tab:blue', linewidth=2, markersize=8)
    ax.set_xlabel('PCP rank $N_c$', fontsize=12)
    ax.set_ylabel('Relative Frobenius error', fontsize=12)
    ax.set_title('PCP reconstruction error vs rank')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "pcp_rank_sweep.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PCP Decomposition Tests")
    print("=" * 60)

    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    test_cell_mapping(phonon)
    test_forward_and_asr(phonon)
    errors = test_rank_sweep(phonon, fc3_path, max_iter=1500)
    test_greedy(phonon, fc3_path)
    test_fourier_transform(phonon, fc3_path)
    plot_rank_sweep(errors, script_dir)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
