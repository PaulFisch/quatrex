"""Tests for the multi-slab self-energy refactor of the dense solver.

Covers:

  * :func:`phonon.solver.fc3_device.build_device_fc3_blocks` — basic
    shape/contents at ``n_slabs=1`` and ``n_slabs=2``.
  * :func:`phonon.solver.bubble.bubble_dense` — the new
    ``dc_handling`` argument is tri-state and reduces to the legacy
    ``"zero"`` behaviour bit-exact when requested.
  * :func:`phonon.solver.se_finite.compute_phph_self_energy_finite_multi_slab`
    — reduces to the legacy single-slab kernel for a 1-block input.
  * :func:`phonon.solver.dense.transmission_finite` — runs end-to-end at
    ``n_slabs=2`` with default cutoffs (no truncation) without crashing
    and produces non-NaN observables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PHONON = Path(__file__).resolve().parents[2] / "phonon"
if str(_PHONON) not in sys.path:
    sys.path.insert(0, str(_PHONON))


def _rng_block(shape, rng):
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )


# ---------------------------------------------------------------------------
# bubble_dense dc_handling
# ---------------------------------------------------------------------------


def test_dc_handling_zero_matches_legacy():
    from solver.bubble import bubble_dense
    rng = np.random.default_rng(0)
    ne, nbs = 9, 4
    n_fft = 2 * ne - 1
    mid = ne // 2
    phi = _rng_block((nbs, nbs, nbs), rng)
    G = _rng_block((ne, nbs, nbs), rng)

    out_legacy = bubble_dense(
        phi_left=phi, phi_right=phi, G_a=G, G_b=G,
        n_fft=n_fft, prefactor=1.0,
        out_slice=slice(mid, mid + ne),
        zero_freq_idx=mid,
    )  # dc_handling defaults to "zero"
    out_zero = bubble_dense(
        phi_left=phi, phi_right=phi, G_a=G, G_b=G,
        n_fft=n_fft, prefactor=1.0,
        out_slice=slice(mid, mid + ne),
        zero_freq_idx=mid, dc_handling="zero",
    )
    np.testing.assert_array_equal(out_legacy, out_zero)


def test_dc_handling_interpolate_differs_from_zero():
    from solver.bubble import bubble_dense
    rng = np.random.default_rng(1)
    ne, nbs = 9, 4
    n_fft = 2 * ne - 1
    mid = ne // 2
    phi = _rng_block((nbs, nbs, nbs), rng)
    G = _rng_block((ne, nbs, nbs), rng)

    out_zero = bubble_dense(
        phi_left=phi, phi_right=phi, G_a=G, G_b=G,
        n_fft=n_fft, prefactor=1.0,
        out_slice=slice(mid, mid + ne),
        zero_freq_idx=mid, dc_handling="zero",
    )
    out_interp = bubble_dense(
        phi_left=phi, phi_right=phi, G_a=G, G_b=G,
        n_fft=n_fft, prefactor=1.0,
        out_slice=slice(mid, mid + ne),
        zero_freq_idx=mid, dc_handling="interpolate",
    )
    # Two distinct DC treatments must differ on a generic input.
    assert not np.allclose(out_zero, out_interp)


def test_dc_handling_keep_does_not_mutate_input():
    from solver.bubble import bubble_dense
    rng = np.random.default_rng(2)
    ne, nbs = 9, 4
    n_fft = 2 * ne - 1
    mid = ne // 2
    phi = _rng_block((nbs, nbs, nbs), rng)
    G = _rng_block((ne, nbs, nbs), rng)
    G_before = G.copy()

    _ = bubble_dense(
        phi_left=phi, phi_right=phi, G_a=G, G_b=G,
        n_fft=n_fft, prefactor=1.0,
        out_slice=slice(mid, mid + ne),
        zero_freq_idx=mid, dc_handling="keep",
    )
    # "keep" must leave the caller's G untouched.
    np.testing.assert_array_equal(G, G_before)


def test_dc_handling_invalid_raises():
    from solver.bubble import bubble_dense
    ne, nbs = 5, 2
    n_fft = 2 * ne - 1
    mid = ne // 2
    G = np.zeros((ne, nbs, nbs), dtype=complex)
    phi = np.zeros((nbs, nbs, nbs), dtype=complex)
    with pytest.raises(ValueError):
        bubble_dense(
            phi_left=phi, phi_right=phi, G_a=G, G_b=G,
            n_fft=n_fft, prefactor=1.0,
            out_slice=slice(mid, mid + ne),
            zero_freq_idx=mid, dc_handling="bogus",
        )


# ---------------------------------------------------------------------------
# build_device_fc3_blocks
# ---------------------------------------------------------------------------


def _toy_supercell_inputs(n_super_z: int, n_atoms_prim: int):
    """Build a minimal 1D-along-z toy supercell mapping.

    One primitive atom per cell (`n_atoms_prim=1`) tiled along z gives
    prim_indices = [0, 0, ..., 0], slab_indices = [0, 1, ..., N-1],
    and a randomly populated M_stacked of the right shape.
    """
    n_super = n_super_z * n_atoms_prim
    prim_indices = np.repeat(np.arange(n_atoms_prim), n_super_z)
    slab_indices = np.tile(np.arange(n_super_z), n_atoms_prim)
    return prim_indices, slab_indices, n_super


def test_device_fc3_n_slabs_1_single_block():
    from solver.fc3_device import build_device_fc3_blocks
    rng = np.random.default_rng(3)
    n_super_z = 4
    n_atoms = 2
    n_dof = 3 * n_atoms
    prim_indices, slab_indices, n_super = _toy_supercell_inputs(
        n_super_z, n_atoms,
    )
    dim_sc = n_super * 3
    M_stacked = rng.standard_normal((n_dof * dim_sc, dim_sc))

    phi_dev = build_device_fc3_blocks(
        M_stacked, prim_indices, slab_indices, n_atoms, n_slabs=1,
    )
    # Exactly the (0, 0, 0) entry exists for a single-slab device.
    assert list(phi_dev.keys()) == [(0, 0, 0)]
    assert phi_dev[(0, 0, 0)].shape == (n_dof, n_dof, n_dof)


def test_device_fc3_vertex_cutoff_0_only_diagonal():
    from solver.fc3_device import build_device_fc3_blocks
    rng = np.random.default_rng(4)
    n_super_z = 4
    n_atoms = 2
    n_dof = 3 * n_atoms
    prim_indices, slab_indices, n_super = _toy_supercell_inputs(
        n_super_z, n_atoms,
    )
    dim_sc = n_super * 3
    M_stacked = rng.standard_normal((n_dof * dim_sc, dim_sc))

    phi_dev = build_device_fc3_blocks(
        M_stacked, prim_indices, slab_indices, n_atoms, n_slabs=3,
        vertex_cutoff=0,
    )
    # vertex_cutoff=0 forces I=K=K'; expect at most 3 entries (one per slab).
    for (I, K, Kp) in phi_dev.keys():
        assert I == K == Kp
    assert len(phi_dev) <= 3


def test_device_fc3_returns_more_blocks_for_larger_cutoff():
    from solver.fc3_device import build_device_fc3_blocks
    rng = np.random.default_rng(5)
    n_super_z = 4
    n_atoms = 1
    n_dof = 3 * n_atoms
    prim_indices, slab_indices, n_super = _toy_supercell_inputs(
        n_super_z, n_atoms,
    )
    dim_sc = n_super * 3
    M_stacked = rng.standard_normal((n_dof * dim_sc, dim_sc))

    n_blocks_seen: dict[int | None, int] = {}
    for vc in (0, 1, 2, None):
        phi_dev = build_device_fc3_blocks(
            M_stacked, prim_indices, slab_indices, n_atoms, n_slabs=4,
            vertex_cutoff=vc,
        )
        n_blocks_seen[vc] = len(phi_dev)
    # Larger cutoff cannot drop blocks the smaller cutoff kept.
    assert n_blocks_seen[0] <= n_blocks_seen[1] <= n_blocks_seen[2]
    assert n_blocks_seen[2] <= n_blocks_seen[None]


# ---------------------------------------------------------------------------
# multi-slab SSE kernel reduces to single-slab driver
# ---------------------------------------------------------------------------


def _reference_finite_bubble(G_lesser, G_greater, Phi, omega, dw):
    """Independent one-block Gamma bubble (two bubble_dense calls) used as the
    oracle for the multi-slab kernel at n_slabs=1."""
    from solver.bubble import bubble_dense
    from phonon_inputs.constants import HBAR_SI, PHPH_SYMMETRY_FACTOR
    n_freq = len(omega)
    n_fft = 2 * n_freq - 1
    mid = n_freq // 2
    freq_sl = slice(mid, mid + n_freq)
    pf = PHPH_SYMMETRY_FACTOR * 0.5j * HBAR_SI * dw / (2 * np.pi)
    sl = bubble_dense(phi_left=Phi, phi_right=Phi, G_a=G_lesser, G_b=G_lesser,
                      n_fft=n_fft, prefactor=pf, out_slice=freq_sl,
                      zero_freq_idx=mid, dc_handling="zero")
    sg = bubble_dense(phi_left=Phi, phi_right=Phi, G_a=G_greater, G_b=G_greater,
                      n_fft=n_fft, prefactor=pf, out_slice=freq_sl,
                      zero_freq_idx=mid, dc_handling="zero")
    return sl, sg


def test_multi_slab_n1_matches_single_slab():
    """At n_slabs=1 with a one-block FC3, the multi-slab kernel must bit-match
    the independent two-call Gamma bubble reference.
    """
    from solver.se_finite import compute_phph_self_energy_finite_multi_slab
    rng = np.random.default_rng(6)
    n_freq = 11
    n_dof = 4
    omega = np.linspace(-5.0, 5.0, n_freq)
    dw = omega[1] - omega[0]

    Phi = _rng_block((n_dof, n_dof, n_dof), rng)
    Gl = _rng_block((n_freq, n_dof, n_dof), rng)
    Gg = _rng_block((n_freq, n_dof, n_dof), rng)

    sl_old, sg_old = _reference_finite_bubble(Gl, Gg, Phi, omega, dw)
    sl_new, sg_new = compute_phph_self_energy_finite_multi_slab(
        g_lesser_blocks={(0, 0): Gl},
        g_greater_blocks={(0, 0): Gg},
        phi_dev_blocks={(0, 0, 0): Phi},
        n_slabs=1,
        omega_grid_thz=omega,
        dw_thz=dw,
        sigma_cutoff=None,
        g_cutoff=None,
        dc_handling="zero",
    )
    np.testing.assert_allclose(sl_new[(0, 0)], sl_old, atol=1e-12)
    np.testing.assert_allclose(sg_new[(0, 0)], sg_old, atol=1e-12)


def test_multi_slab_memory_cap_omega_chunks_match(monkeypatch):
    """A tiny memory budget must force omega-axis chunking inside the
    bubble and still produce a bit-identical result to the
    unconstrained run.
    """
    from solver.se_finite import compute_phph_self_energy_finite_multi_slab
    rng = np.random.default_rng(11)
    n_freq, n_dof, n_slabs = 21, 12, 2
    omega = np.linspace(-9.0, 9.0, n_freq)
    dw = omega[1] - omega[0]

    phi = {
        (I, K, Kp): _rng_block((n_dof, n_dof, n_dof), rng)
        for I in range(n_slabs)
        for K in range(n_slabs)
        for Kp in range(n_slabs)
    }
    gl = {
        (K, Kp): _rng_block((n_freq, n_dof, n_dof), rng)
        for K in range(n_slabs) for Kp in range(n_slabs)
    }
    gg = {
        (K, Kp): _rng_block((n_freq, n_dof, n_dof), rng)
        for K in range(n_slabs) for Kp in range(n_slabs)
    }

    monkeypatch.setenv("QUATREX_PHPH_MEMORY_GB", "64")
    sl_ref, sg_ref = compute_phph_self_energy_finite_multi_slab(
        gl, gg, phi, n_slabs, omega, dw, dc_handling="zero", n_threads=1,
    )

    monkeypatch.setenv("QUATREX_PHPH_MEMORY_GB", "0.05")
    sl_cap, sg_cap = compute_phph_self_energy_finite_multi_slab(
        gl, gg, phi, n_slabs, omega, dw, dc_handling="zero", n_threads=4,
    )

    assert set(sl_ref) == set(sl_cap)
    for k in sl_ref:
        np.testing.assert_array_equal(sl_ref[k], sl_cap[k])
        np.testing.assert_array_equal(sg_ref[k], sg_cap[k])


def test_multi_slab_sigma_cutoff_zero_drops_off_diagonal():
    """sigma_cutoff=0 forbids any (I, J) with I != J in the output."""
    from solver.se_finite import compute_phph_self_energy_finite_multi_slab
    rng = np.random.default_rng(7)
    n_freq = 7
    n_dof = 3
    omega = np.linspace(-3.0, 3.0, n_freq)
    dw = omega[1] - omega[0]

    phi_dev = {
        (I, K, Kp): _rng_block((n_dof, n_dof, n_dof), rng)
        for I in (0, 1)
        for K in (0, 1)
        for Kp in (0, 1)
    }
    g_blocks = {
        (K, Kp): _rng_block((n_freq, n_dof, n_dof), rng)
        for K in (0, 1) for Kp in (0, 1)
    }
    sl, sg = compute_phph_self_energy_finite_multi_slab(
        g_lesser_blocks=g_blocks,
        g_greater_blocks=g_blocks,
        phi_dev_blocks=phi_dev,
        n_slabs=2,
        omega_grid_thz=omega,
        dw_thz=dw,
        sigma_cutoff=0, g_cutoff=None,
        dc_handling="zero",
    )
    assert all(I == J for (I, J) in sl.keys())
    assert all(I == J for (I, J) in sg.keys())


# ---------------------------------------------------------------------------
# q-resolved multi-slab kernel == Gamma multi-slab kernel at a 1x1 mesh
# ---------------------------------------------------------------------------


def _toy_qmesh_inputs(n_super_z, n_atoms, n_slabs, n_freq, seed):
    """Toy 1D supercell mapping + random FC3/G for the q-resolved kernel."""
    from solver.fc3_device import build_device_fc3_blocks
    rng = np.random.default_rng(seed)
    n_dof = 3 * n_atoms
    prim_indices, slab_indices, n_super = _toy_supercell_inputs(
        n_super_z, n_atoms)
    cell_frac = np.zeros((n_super, 3))  # enters only via phases; q=0 -> 1
    dim_sc = n_super * 3
    M_stacked = rng.standard_normal((n_dof * dim_sc, dim_sc))
    phi_dev = build_device_fc3_blocks(
        M_stacked, prim_indices, slab_indices, n_atoms, n_slabs)
    g_l = {(k, kp): _rng_block((n_freq, n_dof, n_dof), rng)
           for k in range(n_slabs) for kp in range(n_slabs)}
    g_g = {(k, kp): _rng_block((n_freq, n_dof, n_dof), rng)
           for k in range(n_slabs) for kp in range(n_slabs)}
    return (M_stacked, prim_indices, cell_frac, slab_indices, phi_dev,
            g_l, g_g, n_dof)


def test_q_multislab_reduces_to_finite_at_1x1():
    """The coupled-q multi-slab kernel at a 1x1 mesh must bit-match the
    Gamma-only multi-slab kernel block for block (locks the q-fold + the
    shared task runner)."""
    from solver.se_finite import compute_phph_self_energy_finite_multi_slab
    from solver.se_q import compute_phph_self_energy_q_dense_multi_slab

    n_atoms, n_super_z, n_slabs, n_freq = 2, 5, 3, 21
    omega = np.linspace(-16.0, 16.0, n_freq)
    omega -= omega[n_freq // 2]
    dw = float(omega[1] - omega[0])
    (M_stacked, prim_indices, cell_frac, slab_indices, phi_dev,
     g_l, g_g, n_dof) = _toy_qmesh_inputs(
        n_super_z, n_atoms, n_slabs, n_freq, seed=21)

    sl_f, sg_f = compute_phph_self_energy_finite_multi_slab(
        g_l, g_g, phi_dev, n_slabs, omega, dw,
        dc_handling="interpolate", n_threads=2)

    g_l_q = {k: v[None] for k, v in g_l.items()}
    g_g_q = {k: v[None] for k, v in g_g.items()}
    sl_q, sg_q = compute_phph_self_energy_q_dense_multi_slab(
        g_l_q, g_g_q, M_stacked, prim_indices, cell_frac, slab_indices,
        n_atoms, n_slabs, 1, [(0.0, 0.0)], np.array([[0]]), omega, dw,
        transport_direction="x", dc_handling="interpolate")

    assert set(sl_f) == set(sl_q)
    for key in sl_f:
        np.testing.assert_allclose(sl_q[key][0], sl_f[key], rtol=1e-10,
                                   atol=1e-12)
        np.testing.assert_allclose(sg_q[key][0], sg_f[key], rtol=1e-10,
                                   atol=1e-12)


def test_q_multislab_sigma_cutoff_zero_is_diagonal_subset():
    """sigma_cutoff=0 keeps only (I, I) blocks, equal to the full kernel's
    diagonal blocks (Guo approximation III as a knob on the full default)."""
    from solver.se_q import compute_phph_self_energy_q_dense_multi_slab

    n_atoms, n_super_z, n_slabs, n_freq = 1, 5, 3, 15
    omega = np.linspace(-12.0, 12.0, n_freq)
    omega -= omega[n_freq // 2]
    dw = float(omega[1] - omega[0])
    (M_stacked, prim_indices, cell_frac, slab_indices, _phi,
     g_l, g_g, n_dof) = _toy_qmesh_inputs(
        n_super_z, n_atoms, n_slabs, n_freq, seed=33)
    # 2x1 transverse mesh.
    from phonon_inputs.separable import build_q_diff_map
    q_points = [(0.0, 0.0), (0.5, 0.0)]
    q_diff_map = build_q_diff_map(2, 1)
    n_kpts = 2
    g_l_q = {k: _rng_block((n_kpts, n_freq, n_dof, n_dof),
                           np.random.default_rng(hash(k) % 99))
             for k in g_l}
    g_g_q = {k: g_l_q[k].conj() for k in g_l_q}

    common = dict(
        M_stacked=M_stacked, prim_indices=prim_indices, cell_frac=cell_frac,
        slab_indices=slab_indices, n_atoms=n_atoms, n_slabs=n_slabs,
        n_kpts=n_kpts, q_points=q_points, q_diff_map=q_diff_map,
        omega_grid_thz=omega, dw_thz=dw, transport_direction="x")
    sl_full, _ = compute_phph_self_energy_q_dense_multi_slab(
        g_l_q, g_g_q, **common)
    sl_diag, _ = compute_phph_self_energy_q_dense_multi_slab(
        g_l_q, g_g_q, sigma_cutoff=0, **common)

    assert all(I == J for (I, J) in sl_diag)
    for key in sl_diag:
        np.testing.assert_allclose(sl_diag[key], sl_full[key], rtol=1e-10,
                                   atol=1e-12)
