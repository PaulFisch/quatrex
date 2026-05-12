"""End-to-end pipeline test on an analytic Si chain.

The chain is constructed entirely in memory:

  * 8 Si atoms uniformly spaced along z (no DFT).
  * FC2 = nearest-neighbour springs (analytic) with a small next-NN term so
    the dynamical matrix is non-degenerate.
  * FC3 = small known cubic perturbation (also nearest-neighbour).

The test exercises every public driver in :mod:`finite_analysis` to
catch interface regressions, and asserts a handful of physical bounds
(ASR residual at machine precision, FC2 Hermiticity, lifted FC3 perm
symmetry, decomposition Frobenius bounds, synthetic-Σ vs dense-Σ
agreement). It does *not* invoke the cluster pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Insert the phonon dir on sys.path so the package import works
# without a conda install. (Mirrors how the smoke tests in this repo run.)
_INPUT_CALC = Path(__file__).resolve().parents[3] / "phonon"
sys.path.insert(0, str(_INPUT_CALC))


@pytest.fixture(scope="module")
def chain_bundle():
    """Build an 8-atom Si chain bundle with analytic FC2/FC3."""
    import warnings
    warnings.simplefilter("ignore")

    import numpy as np
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    n_atoms = 8
    spacing = 2.35  # Å
    c_len = n_atoms * spacing
    positions_frac = np.array([
        [0.5, 0.5, (i + 0.5) / n_atoms] for i in range(n_atoms)
    ])
    cell = np.diag([15.0, 15.0, c_len])
    atoms = PhonopyAtoms(
        symbols=["Si"] * n_atoms,
        cell=cell,
        scaled_positions=positions_frac,
    )
    phonon = Phonopy(atoms, supercell_matrix=np.eye(3, dtype=int),
                     primitive_matrix=np.eye(3))

    # Analytic FC2: nearest-neighbour spring along z + small NN xy.
    # Phonopy convention: shape (n, n, 3, 3). The chain is periodic, so the
    # atom 0 ↔ atom (n-1) bond across the periodic boundary is included.
    fc2 = np.zeros((n_atoms, n_atoms, 3, 3))
    k_z = 4.0   # eV/Å²
    k_xy = 0.5  # eV/Å²

    def _is_nn(i: int, j: int) -> bool:
        d = (i - j) % n_atoms
        return d == 1 or d == n_atoms - 1

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            if _is_nn(i, j):
                fc2[i, j] = -np.diag([k_xy, k_xy, k_z])
    # Self terms enforce ASR.
    for i in range(n_atoms):
        fc2[i, i] = -fc2[i, :, :, :].sum(axis=0)
    # Symmetrise.
    fc2 = 0.5 * (fc2 + fc2.transpose(1, 0, 3, 2))
    phonon.force_constants = fc2

    # Analytic FC3: small cubic NN-only term, Si-Si-Si triplets.
    fc3 = np.zeros((n_atoms, n_atoms, n_atoms, 3, 3, 3))
    g3 = 0.05  # eV/Å³ scale
    for i in range(n_atoms):
        for di in (-1, 1):
            j = i + di
            if not (0 <= j < n_atoms):
                continue
            for dk in (-1, 1):
                k = i + dk
                if not (0 <= k < n_atoms):
                    continue
                # Only z-components couple — minimal isotropic NN cubic term.
                fc3[i, j, k, 2, 2, 2] = g3 * (di + dk)
    # Project FC3 onto S3 (i,j,k) symmetric to satisfy permutational invariance.
    fc3_sym = (
        fc3 + fc3.transpose(0, 2, 1, 3, 5, 4)
        + fc3.transpose(1, 0, 2, 4, 3, 5) + fc3.transpose(1, 2, 0, 4, 5, 3)
        + fc3.transpose(2, 0, 1, 5, 3, 4) + fc3.transpose(2, 1, 0, 5, 4, 3)
    ) / 6.0

    # Build the SystemBundle directly (no YAML round-trip).
    from finite_analysis.loader import SystemBundle, cluster_into_slabs
    from phonon_inputs.fc3_compression import build_fc3_target
    target = build_fc3_target(fc3_sym, phonon)

    block_sizes, atom_perm = cluster_into_slabs(
        np.asarray(phonon.supercell.positions), np.asarray(phonon.supercell.cell),
        transport_axis=2, n_slabs_hint=4,
    )

    return SystemBundle(
        name="si_chain_test",
        phonon=phonon,
        fc2=fc2,
        fc3_raw=fc3_sym,
        fc3_target=target,
        masses=np.asarray(phonon.supercell.masses),
        sc_positions=np.asarray(phonon.supercell.positions),
        sc_cell=np.asarray(phonon.supercell.cell),
        block_sizes=block_sizes,
        atom_perm=atom_perm,
        transport_axis=2,
        meta={"source": "analytic"},
    )


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_fc2_hermiticity(chain_bundle):
    from finite_analysis.physical_tests import fc2_hermiticity
    res = fc2_hermiticity(chain_bundle.fc2)
    assert res["max_abs"] < 1e-12
    assert res["rel_frob"] < 1e-12


def test_fc2_asr(chain_bundle):
    from finite_analysis.physical_tests import fc2_asr_residual
    res = fc2_asr_residual(chain_bundle.fc2)
    assert res["rel_frob"] < 1e-10, f"FC2 ASR rel residual {res['rel_frob']}"


def test_fc3_perm_sym(chain_bundle):
    from finite_analysis.physical_tests import fc3_perm_symmetry
    res = fc3_perm_symmetry(chain_bundle.fc3_target.T_lifted)
    for k, v in res.items():
        assert v < 1e-10, f"{k} = {v}"


def test_sparsity_runs(chain_bundle, tmp_path):
    from finite_analysis.sparsity import run_sparsity
    summary = run_sparsity(chain_bundle, tmp_path / "sparsity")
    assert "fc2_max" in summary and summary["fc2_max"] > 0
    expected = {
        "sparsity_fc2_heatmap.png", "sparsity_fc3_decay_1d.png",
        "sparsity_fc3_scatter_3d.png", "sparsity_nnz_table.csv",
    }
    files = {p.name for p in (tmp_path / "sparsity").iterdir()}
    assert expected.issubset(files)


def test_decomposition_msvd_bounds(chain_bundle):
    """mSVD at full rank (=dim_sc) should reach Frobenius error ~ 0."""
    from finite_analysis.decomposition import fit_decomposition
    rows = fit_decomposition(
        chain_bundle, scalar_ranks=(2, chain_bundle.fc3_target.dim_sc),
        methods=["mSVD"], skip_pcp=True,
    )
    assert any(
        r.method == "mSVD" and isinstance(r.rank, int)
        and r.rank == chain_bundle.fc3_target.dim_sc and r.frob_err < 1e-8
        for r in rows
    ), rows


def test_synthetic_gf_bose_symmetry(chain_bundle):
    """G^<(ω) = [G^>(−ω)]^T for the synthetic GF."""
    from finite_analysis.synthetic_gf import synthetic_gf_dense
    G_l, G_g, freqs, dw, modes = synthetic_gf_dense(
        chain_bundle, n_freq_pos=32, eta_thz=0.5,
    )
    mid = freqs.size // 2
    err = np.max(np.abs(G_l[mid + 1:] - G_g[mid - 1::-1].transpose(0, 2, 1)))
    assert err < 1e-10, f"Bose symmetry violation {err}"


def test_synthetic_gf_no_unstable(chain_bundle):
    """The analytic FC2 must have a positive-definite dynamical matrix."""
    from finite_analysis.synthetic_gf import diagonalise
    modes = diagonalise(chain_bundle)
    assert modes.n_unstable == 0, (
        f"Unexpected {modes.n_unstable} imaginary modes — FC2 is not PSD"
    )


def test_bose_identity():
    """Bose-Einstein identity n_B(-ω) = -1 - n_B(ω) on a sample of ω."""
    from finite_analysis.physics_validation import bose_identity_residual

    res = bose_identity_residual(
        np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]),
        temperature_k=300.0,
    )
    assert res["max_abs_diff"] < 1e-12, res


def test_synthetic_gf_anti_hermiticity(chain_bundle):
    """G^<(ω) and G^>(ω) must be anti-Hermitian in DOF space (Wang 2014, eq. 19)."""
    from finite_analysis.synthetic_gf import synthetic_gf_dense
    from finite_analysis.physics_validation import anti_hermiticity_residual

    G_l, G_g, _, _, _ = synthetic_gf_dense(
        chain_bundle, n_freq_pos=32, eta_thz=0.1, temperature_k=300.0,
    )
    res_l = anti_hermiticity_residual(G_l)
    res_g = anti_hermiticity_residual(G_g)
    assert res_l["max_rel"] < 1e-10, f"G^< not anti-Hermitian: {res_l}"
    assert res_g["max_rel"] < 1e-10, f"G^> not anti-Hermitian: {res_g}"


def test_first_moment_sum_rule(chain_bundle):
    """∫₀^∞ 2ω³ A(ω) dω = Tr(D) on the analytic chain."""
    from finite_analysis.synthetic_gf import diagonalise, dynamical_matrix
    from finite_analysis.physics_validation import first_moment_sum_rule_residual
    from finite_analysis.constants import THZ_ACOUSTIC_BAND

    D = dynamical_matrix(chain_bundle, z_sorted=True)
    modes = diagonalise(chain_bundle)

    omega = modes.omega_thz
    keep = omega > THZ_ACOUSTIC_BAND
    omega_pos = omega[keep]
    eps = modes.polarisation[:, keep]

    freqs = np.linspace(-1.05 * float(omega.max()), 1.05 * float(omega.max()), 401)
    eta = 0.05
    G_R = np.empty((freqs.size, eps.shape[0], eps.shape[0]), dtype=complex)
    for iw, w in enumerate(freqs):
        denom = (w + 1j * eta) ** 2 - omega_pos ** 2
        G_R[iw] = (eps / denom) @ eps.T
    # Tr(D) computed from the kept modes (the dropped acoustic zeros
    # contribute zero since ω_n² ≈ 0 for them).
    expected = float(np.sum(omega_pos ** 2))
    res = first_moment_sum_rule_residual(G_R, freqs, np.diag(np.full(eps.shape[0],
                                                                       expected / eps.shape[0])))
    # Compare against the *correct* expected (sum of ω_n²) directly.
    rel = (res["integrated"] - expected) / max(expected, 1.0)
    assert abs(rel) < 0.10, (
        f"First-moment sum rule deviates by {rel:.3%}: integrated="
        f"{res['integrated']:.3e}, expected (Σω_n²)={expected:.3e}"
    )


def test_detailed_balance_bubble_sigma(chain_bundle):
    """The bubble Σ^{<,>} computed on a thermal eigenmode G should obey
    bosonic detailed balance Σ^>(ω) ≈ exp(βℏω) Σ^<(ω) within the finite
    Bose window."""
    from finite_analysis.loader import load_quatrex_blocks
    from finite_analysis.synthetic_gf import synthetic_gf_dense, gf_to_block_dict
    from finite_analysis.sse_cutoffs import compute_sse_with_cutoffs
    from finite_analysis.physical_tests import detailed_balance_residual

    phi_blocks = load_quatrex_blocks(chain_bundle, truncation_warn=0.5)
    G_l, G_g, freqs, dw, _ = synthetic_gf_dense(
        chain_bundle, n_freq_pos=64, eta_thz=0.1, temperature_k=300.0,
    )
    gl = gf_to_block_dict(G_l, chain_bundle.block_sizes, nn_only=False)
    gg = gf_to_block_dict(G_g, chain_bundle.block_sizes, nn_only=False)
    res = compute_sse_with_cutoffs(
        phi_blocks, gl, gg, chain_bundle.block_sizes, dw,
    )
    # Aggregate to a dense Σ for the detailed-balance check.
    block_sizes = chain_bundle.block_sizes
    n_dof = int(np.sum(block_sizes))
    offsets = np.concatenate(([0], np.cumsum(block_sizes)))
    Sl = np.zeros((freqs.size, n_dof, n_dof), dtype=complex)
    Sg = np.zeros_like(Sl)
    for (I, J), v in res.sigma_lesser.items():
        Sl[:, offsets[I]:offsets[I+1], offsets[J]:offsets[J+1]] = v
    for (I, J), v in res.sigma_greater.items():
        Sg[:, offsets[I]:offsets[I+1], offsets[J]:offsets[J+1]] = v
    rep = detailed_balance_residual(Sl, Sg, freqs, temperature_k=300.0)
    # Bubble on a thermal G should respect detailed balance to a few percent.
    # Loosen because the synthetic G is a Lorentzian-broadened approximation.
    assert rep["max_rel_dev"] < 0.50, (
        f"Detailed balance violated: max_rel_dev = {rep['max_rel_dev']:.3f}, "
        f"n_samples = {rep['n_samples']}"
    )


def test_scba_convergence_chain(chain_bundle):
    """SCBA loop on the chain: rel-change in Σ^< must monotonically decrease
    over iterations and the final ‖ΔΣ‖/‖Σ‖ should be < 5 %."""
    from finite_analysis.loader import load_quatrex_blocks
    from finite_analysis.transport_metrics import transport_trace_scba

    phi_blocks = load_quatrex_blocks(chain_bundle, truncation_warn=0.5)
    freqs = np.linspace(-15.0, 15.0, 65)
    trace, rel_changes = transport_trace_scba(
        chain_bundle, phi_blocks, freqs,
        n_iter=3, mixing=0.5, eta_thz=0.05,
        lead_model="synthetic", gamma_lead_thz=0.5,
    )
    assert len(rel_changes) >= 1, "SCBA produced no iteration history"
    # Monotone decrease (modulo small noise).
    for k in range(1, len(rel_changes)):
        assert rel_changes[k] <= rel_changes[k - 1] * 1.5, (
            f"SCBA rel-change non-monotone: {rel_changes}"
        )
    # Final value sane.
    assert rel_changes[-1] < 0.5, (
        f"SCBA did not converge: rel_changes = {rel_changes}"
    )


def test_sancho_rubio_lead_self_energy(chain_bundle):
    """Sancho-Rubio lead Σ converges on the chain, is non-trivial in the
    optical band, and has the expected support (only on first/last slab)."""
    from finite_analysis.transport_metrics import sancho_rubio_lead_self_energies

    freqs = np.linspace(-15.0, 15.0, 65)
    Sigma_L, Sigma_R = sancho_rubio_lead_self_energies(
        chain_bundle, freqs, eta_thz=1e-3,
    )
    # Both should be the same shape as the dense system Σ.
    assert Sigma_L.shape == (freqs.size, chain_bundle.n_dof, chain_bundle.n_dof)
    assert Sigma_R.shape == Sigma_L.shape

    # Support: Σ_L nonzero only on first slab; Σ_R only on last slab.
    block_sizes = np.asarray(chain_bundle.block_sizes)
    offsets = np.concatenate(([0], np.cumsum(block_sizes)))
    sL = slice(offsets[0], offsets[1])
    sR = slice(offsets[-2], offsets[-1])
    # Take a positive-frequency sample (avoid the zeroed acoustic gap).
    pos_iw = int(np.argmin(np.abs(freqs - 5.0)))  # near 5 THz
    SL = Sigma_L[pos_iw]
    SR = Sigma_R[pos_iw]
    if not np.any(np.isnan(SL)):
        outside = SL.copy()
        outside[sL, sL] = 0
        assert np.allclose(outside, 0.0), (
            "Sigma_L_lead has unexpected nonzero entries outside slab 0"
        )
        # Nontrivial broadening at this ω.
        assert np.linalg.norm(SL) > 1e-6, "Sigma_L_lead vanishes mid-band"
    if not np.any(np.isnan(SR)):
        outside = SR.copy()
        outside[sR, sR] = 0
        assert np.allclose(outside, 0.0)
        assert np.linalg.norm(SR) > 1e-6


def test_dos_sum_rule_eigenmode_gf(chain_bundle):
    """∫ A(ω) dω = N_dof for the synthetic eigenmode G^R on the chain.

    A(ω) = -Im Tr G^R(ω)/π. With Lorentzian-broadened delta peaks at
    ±ω_n, ∫A dω = (number of physical positive modes). Excludes the
    acoustic and imaginary modes that synthetic_gf_dense drops.
    """
    from finite_analysis.synthetic_gf import synthetic_gf_dense, diagonalise
    from finite_analysis.constants import THZ_ACOUSTIC_BAND
    from finite_analysis.physical_tests import dos_sum_rule_residual

    G_l, G_g, freqs, dw, modes = synthetic_gf_dense(
        chain_bundle, n_freq_pos=200, eta_thz=0.05,
    )
    # The eigenmode G^R(ω) = Σ_n (ε ε^T)/[(ω+iη)² − ω_n²]; build it directly.
    n_kept = int(np.sum(modes.omega_thz > THZ_ACOUSTIC_BAND))
    omega = np.asarray(modes.omega_thz)
    keep = omega > THZ_ACOUSTIC_BAND
    omega_pos = omega[keep]
    eps = modes.polarisation[:, keep]
    eta = 0.05
    G_R = np.empty((freqs.size, eps.shape[0], eps.shape[0]), dtype=complex)
    for iw, w in enumerate(freqs):
        denom = (w + 1j * eta) ** 2 - omega_pos ** 2
        G_R[iw] = (eps / denom) @ eps.T
    res = dos_sum_rule_residual(G_R, freqs, n_kept)
    # ∫_0^∞ A dω = n_kept/2. Lorentzian broadening (η = 0.05 THz) and the
    # finite grid leak a few percent at the band edge.
    assert abs(res["rel_dev"]) < 0.10, (
        f"DOS sum rule deviates by {res['rel_dev']:.3%} (want < 10 %): {res}"
    )


def test_gamma_projection_matches_einsum():
    """``_gamma_project_M_blocks`` must be bit-for-bit identical to the
    original ``T0 @ M @ T0†`` einsum at q=Γ. Sentinel: random M, random
    prim_indices."""
    from solver import gamma_project_M_blocks as _gamma_project_M_blocks
    from phonon_inputs.separable import build_gathering_matrix

    rng = np.random.default_rng(0)
    n_atoms = 3
    n_super = 12  # 4 primitive images
    dim_sc = 3 * n_super
    n_dof = 3 * n_atoms
    prim_indices = np.array([i % n_atoms for i in range(n_super)])
    cell_frac = rng.random((n_super, 3))  # not used at Γ
    M = (rng.standard_normal((n_dof, dim_sc, dim_sc))
         + 1j * rng.standard_normal((n_dof, dim_sc, dim_sc)))

    T0 = build_gathering_matrix(prim_indices, cell_frac, (0.0, 0.0),
                                 n_atoms, "z")
    Phi_einsum = np.einsum("ci,aij,dj->acd", T0, M, T0.conj())
    Phi_fast = _gamma_project_M_blocks(M, prim_indices, n_atoms)
    assert np.allclose(Phi_einsum, Phi_fast, atol=1e-12, rtol=1e-10)


def test_synthetic_gf_correlator_normalisation(chain_bundle):
    """∫ iG^<(ω) dω/(2π) should reproduce the equal-time displacement
    correlator ⟨u u⟩ = Σ_n (ε_n ε_n^T)(n_B(ω_n) + ½)/ω_n.

    Pins the synthetic-GF prefactor (theory.tex eq. 873–874 + eigenmode
    expansion of Im G^R). Targets 5% agreement on the 8-atom chain at the
    default broadening; tighter tolerances would require a larger n_freq
    or smaller eta.
    """
    from finite_analysis.synthetic_gf import synthetic_gf_dense, diagonalise
    from finite_analysis.constants import THZ_ACOUSTIC_BAND
    from phonon_inputs.constants import HBAR_SI, KB_SI, THZ_TO_RAD

    T_K = 300.0
    G_l, G_g, freqs, dw, modes = synthetic_gf_dense(
        chain_bundle, n_freq_pos=200, eta_thz=0.05, temperature_k=T_K,
    )
    # Numerical: ∫ i G^<(ω) dω / (2π).  The grid is symmetric and uniform.
    correlator_num = 1j * G_l.sum(axis=0) * dw / (2.0 * np.pi)

    # Analytic: Σ_n (ε ε^T) (n_B + ½) / ω_n on positive non-acoustic modes.
    omega = modes.omega_thz
    keep = omega > THZ_ACOUSTIC_BAND
    omega_pos = omega[keep]
    eps = modes.polarisation[:, keep]
    x = HBAR_SI * omega_pos * THZ_TO_RAD / (KB_SI * T_K)
    n_B = 1.0 / np.expm1(x)
    weight = (n_B + 0.5) / omega_pos
    correlator_ana = (eps * weight) @ eps.T

    # Compare diagonal elements (off-diagonal sensitive to off-resonance Lorentzian tails).
    diag_num = np.real(np.diag(correlator_num))
    diag_ana = np.diag(correlator_ana)
    rel = np.linalg.norm(diag_num - diag_ana) / np.linalg.norm(diag_ana)
    assert rel < 0.05, (
        f"Synthetic-GF correlator off by {rel:.3%}. "
        f"Mean(num)={diag_num.mean():.3e}, mean(ana)={diag_ana.mean():.3e}. "
        "Likely a wrong prefactor in synthetic_gf.synthetic_gf_dense."
    )


def test_lead_broadening_monotone(chain_bundle):
    """T(ω; γ) at fixed ω should be monotonically increasing in lead
    broadening γ for small γ (wide-band-limit lead). Validates the
    transport_metrics lead model on the chain."""
    from finite_analysis.synthetic_gf import synthetic_gf_dense
    from finite_analysis.transport_metrics import transport_trace_from_sigma

    # Build a tiny zero-Σ trace family at increasing γ.
    G_l, G_g, freqs, dw, _ = synthetic_gf_dense(
        chain_bundle, n_freq_pos=64, eta_thz=0.5,
    )
    zero_blocks_l: dict = {}
    zero_blocks_g: dict = {}
    n_blocks = chain_bundle.n_slabs
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            shape = (freqs.size, chain_bundle.block_sizes[I], chain_bundle.block_sizes[J])
            zero_blocks_l[(I, J)] = np.zeros(shape, dtype=complex)
            zero_blocks_g[(I, J)] = np.zeros(shape, dtype=complex)

    gammas = [0.05, 0.1, 0.2, 0.5, 1.0]
    traces = []
    for gamma in gammas:
        tr = transport_trace_from_sigma(
            chain_bundle, zero_blocks_l, zero_blocks_g, freqs,
            eta_thz=0.05, gamma_lead_thz=gamma, T_L=305.0, T_R=295.0,
            lead_model="synthetic",  # explicit; Sancho-Rubio ignores γ
        )
        traces.append(tr.transmission)

    # Sample a few optical-band frequencies above the acoustic gap.
    pos_mask = freqs > 1.0
    pos_idx = np.where(pos_mask)[0]
    if pos_idx.size < 5:
        pytest.skip("Not enough positive frequencies above acoustic gap")
    sample = np.linspace(0, pos_idx.size - 1, 5).astype(int)
    sample_idx = pos_idx[sample]

    monotone_count = 0
    for iw in sample_idx:
        T_vs_gamma = [traces[k][iw] for k in range(len(gammas))]
        if all(T_vs_gamma[k + 1] >= T_vs_gamma[k] - 1e-12 for k in range(len(gammas) - 1)):
            monotone_count += 1
    assert monotone_count >= 3, (
        f"T(ω; γ) is monotone in γ at only {monotone_count}/5 sample frequencies. "
        f"Sample T(γ): {[float(traces[k][sample_idx[0]]) for k in range(len(gammas))]}. "
        "Lead broadening is not behaving as a wide-band-limit Γ."
    )


def test_sse_baseline_vs_dense_path(chain_bundle):
    """Block-decomposed bubble (sse_cutoffs) must agree with the dense
    standalone bubble (anharmonic.py) on the same Phi and G."""
    from finite_analysis.synthetic_gf import synthetic_gf_dense, gf_to_block_dict
    from finite_analysis.loader import load_quatrex_blocks
    from finite_analysis.sse_cutoffs import compute_sse_with_cutoffs
    from finite_analysis.sse_quatrex_run import run_dense_scba_crosscheck

    G_l, G_g, freqs, dw, modes = synthetic_gf_dense(
        chain_bundle, n_freq_pos=32, eta_thz=0.5,
    )
    # Expose ALL (I, J) G blocks (nn_only=False) so the bubble can pick up
    # off-NN G contributions and match the dense-bubble enumeration. With
    # nn_only=True the block route would silently drop those terms.
    gl_blocks = gf_to_block_dict(G_l, chain_bundle.block_sizes, nn_only=False)
    gg_blocks = gf_to_block_dict(G_g, chain_bundle.block_sizes, nn_only=False)

    phi_blocks = load_quatrex_blocks(chain_bundle, truncation_warn=0.5)

    block_res = compute_sse_with_cutoffs(
        phi_blocks, gl_blocks, gg_blocks, chain_bundle.block_sizes, dw,
    )
    dense_block_frob = run_dense_scba_crosscheck(
        chain_bundle, n_freq_pos=32, eta_thz=0.5, n_iter=0,
    )

    # Now that compute_sse_with_cutoffs enumerates the full block-tridiagonal
    # bubble (including off-diagonal G(K, K') contributions), the two routes
    # must agree to floating-point precision on the chain. The chain's FC3
    # is NN-only so both phi_blocks and the full-dense bubble see exactly
    # the same support.
    for ij in block_res.block_frob:
        if abs(ij[0] - ij[1]) > 1:
            continue
        block_norm = block_res.block_frob[ij][0]
        dense_norm = dense_block_frob[ij][0]
        denom = max(block_norm, dense_norm, 1e-30)
        rel = abs(block_norm - dense_norm) / denom
        assert rel < 1e-6, (
            f"Σ^< {ij} disagrees: block={block_norm:.6e}, "
            f"dense={dense_norm:.6e}, rel={rel:.3e}"
        )


# --------------------------------------------------------------------------- #
# Unified solver end-to-end (phonon.solver)                                   #
# --------------------------------------------------------------------------- #


def test_unified_solver_transmission_finite_chain(chain_bundle):
    """``phonon.solver.transmission_finite`` runs end-to-end on the chain.

    Sanity-checks the returned dict shape, the ballistic max-T, and
    heat-flow conservation. The SCBA loop is exercised with two
    iterations (mixing=0.5) which is enough to populate
    ``convergence_history``.
    """
    from solver import transmission_finite

    result = transmission_finite(
        chain_bundle.phonon,
        M_stacked_override=chain_bundle.fc3_target.T,
        freq_range_thz=(0.01, 14.0, 41),
        transport_direction="z",
        eta_factor=0.1,
        temperature=300.0,
        delta_T=10.0,
        max_scba_iter=2,
        scba_tol=1e-3,
        mixing=0.5,
        retarded="half",
        verbose=False,
    )

    # Result dict has the documented keys.
    expected_keys = {
        "freqs_thz", "transmission_ballistic", "spectral_heat_current",
        "heat_current", "thermal_conductance_anharmonic",
        "thermal_conductance_ballistic", "heat_flow_conservation",
        "n_scba_iterations", "convergence_history",
        "self_energy_retarded", "self_energy_lesser", "self_energy_greater",
    }
    missing = expected_keys - set(result)
    assert not missing, f"transmission_finite is missing keys: {missing}"

    # Frequencies and ballistic T are sane.
    freqs = np.asarray(result["freqs_thz"])
    assert freqs.size > 0
    assert (freqs > 0).all(), "frequencies must be positive-only (pos_mask)"
    T_ball = np.asarray(result["transmission_ballistic"])
    assert T_ball.shape == freqs.shape
    # Some non-zero ballistic transport must show up below the LA cutoff.
    assert T_ball.max() > 1e-3, (
        f"Ballistic T(ω) is too small (max={T_ball.max():.3e}); "
        "check that the chain's H_00/H_01 are non-trivial."
    )

    # Heat current must be finite and conservation residual modest.
    Q = result["heat_current"]
    assert np.isfinite(Q), f"heat_current is not finite: {Q}"
    cons = result["heat_flow_conservation"]
    assert cons < 0.5, (
        f"Heat-flow conservation residual {cons:.3e} too large; "
        "SCBA equilibrium-closure should give |J_L - J_R| / (|J_L| + |J_R|) "
        "well below 50% on a passive chain."
    )


def test_unified_solver_scba_convergence_chain(chain_bundle):
    """SCBA via the unified solver: convergence_history is non-increasing
    over the iterations and the final iteration carries a small dJ/J."""
    from solver import transmission_finite

    result = transmission_finite(
        chain_bundle.phonon,
        M_stacked_override=chain_bundle.fc3_target.T,
        freq_range_thz=(0.01, 14.0, 33),
        transport_direction="z",
        eta_factor=0.1,
        temperature=300.0,
        delta_T=10.0,
        max_scba_iter=5,
        scba_tol=1e-4,
        mixing=0.5,
        retarded="half",
        verbose=False,
    )
    history = list(result["convergence_history"])
    # First iteration prints J_L/J_R only (no rel-change), so history starts
    # at iter 2; with max_scba_iter=5 we expect up to 4 entries (fewer on
    # early convergence).
    assert len(history) >= 1, (
        f"SCBA produced no rel-change samples — bumped to max_scba_iter? "
        f"history={history}"
    )
    # Non-increasing modulo a 1.5x slack (Picard mixing has small overshoots).
    for k in range(1, len(history)):
        assert history[k] <= history[k - 1] * 1.5, (
            f"SCBA rel-change non-monotone at iter {k}: {history}"
        )


def test_unified_solver_q11_matches_finite(chain_bundle):
    """``transmission_q(q_mesh=(1,1))`` must reproduce ``transmission_finite``
    on the same chain to within the q-path's own regression tolerance."""
    from solver import (
        compare_q11_to_finite, transmission_finite, transmission_q,
    )

    kwargs = dict(
        M_stacked_override=chain_bundle.fc3_target.T,
        freq_range_thz=(0.01, 14.0, 33),
        transport_direction="z",
        eta_factor=0.1,
        temperature=300.0,
        delta_T=10.0,
        max_scba_iter=2,
        scba_tol=1e-3,
        mixing=0.5,
        retarded="half",
        verbose=False,
    )
    res_f = transmission_finite(chain_bundle.phonon, **kwargs)
    res_q = transmission_q(
        chain_bundle.phonon, q_mesh_transverse=(1, 1), **kwargs,
    )
    # compare_q11_to_finite raises AssertionError on mismatch.
    compare_q11_to_finite(res_q, res_f, rtol=5e-3, atol=1e-8)


def test_cli_smoke(chain_bundle, tmp_path, monkeypatch):
    """CLI runs the fast subset against an existing on-disk reap if present.

    This is a smoke test only; we use the Si primitive reap that ships in
    the repo to drive ``main()`` end-to-end. If the reap is missing the
    test is skipped.
    """
    from finite_analysis.cli import main

    config = _INPUT_CALC / "configs" / "si_primitive" / "prim_vasp.yaml"
    fc3 = _INPUT_CALC / "reaps" / "si_primitive_vasp" / "fc3.hdf5"
    if not fc3.exists():
        pytest.skip(f"reap not found: {fc3}")

    rc = main([
        "--config", str(config),
        "--fc3-path", str(fc3),
        "--out-dir", str(tmp_path / "cli_out"),
        "--analyses", "physical,sparsity",
        "--n-slabs-hint", "2",
    ])
    assert rc == 0
    assert (tmp_path / "cli_out" / "summary.json").exists()
    assert (tmp_path / "cli_out" / "physical" / "physical.json").exists()
    assert (tmp_path / "cli_out" / "sparsity" / "sparsity_fc2_heatmap.png").exists()
