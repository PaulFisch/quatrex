# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The spatially analytic Green-function tail: the invariants, not the numbers.

Companion to ``phonon/docs/spatial_analytic_tail.md``. The programme asks
whether the long-range spatial part of ``G`` can be carried by complex-band
modes and whether that recovers a transport-relevant part of the cubic
self-energy. What belongs HERE is the half of that with an analytic or
combinatorial answer -- pencil degree and root counts, the exactness of the
shell decomposition, the anchor/rank rule, the validity window of the analytic
contraction. What belongs in ``phonon/studies/_spatial_tail_*.py`` is anything
whose threshold is a property of a bed.

The division matters because the sibling file already has one test on the wrong
side of it (``test_the_pin_grows_over_the_lengths_...`` asserts
``0.08 < fracs[7] < 0.20`` on a 1-DOF chain), and that pattern should not be
multiplied.

Everything is at eta = 0. Beds come from ``phonon/solver/toy_models.py`` and
``phonon/studies/_spatial_bed.py``, promoted out of ``test_spatial_modal.py``
so the studies and the tests measure the same object; the promotion is asserted
bit-equal below rather than assumed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# ``phonon/`` is not a package, so its subpackages are imported by putting it on
# the path -- the same route ``test_spatial_modal.py`` takes.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "phonon"))

from solver.toy_models import (                                  # noqa: E402
    bulk_green_blocks, chain2_band_top, chain2_system_blocks, gapped_chain,
    gapped_chain_green, gapped_chain_root, neighbour_cubic_vertex)
from studies import _spatial_bed as bed                          # noqa: E402

from quatrex.phonon.spatial_fit import (                         # noqa: E402
    geometric_block_sum, geometric_factor, modal_fit,
    prune_by_amplitude, tail_amplitudes)
from quatrex.phonon.spatial_modes import (                       # noqa: E402
    band_range_cells, bloch_modes, bloch_modes_poly, nevp_residual)

BAND_TOP_2 = chain2_band_top()


# --- the promotion is bit-equal -------------------------------------------- #
#
# These beds were module-private helpers in test_spatial_modal.py, and the
# published numbers in phonon/docs/ were measured with them. A promotion that
# changed a random draw order or a quadrature default would move those numbers
# silently, so it is pinned rather than trusted.

def test_the_promoted_vertex_reproduces_the_original_draw_order():
    """The draw sits INSIDE the bounds check, so the sequence depends on
    ``n_cell``. Reordering it would move the pin-versus-length table."""
    for n_cell in (4, 7, 12, 16):
        rng = np.random.default_rng(5)
        want = np.zeros((n_cell,) * 3)
        for i in range(n_cell):
            for a in (i - 1, i, i + 1):
                for b in (i - 1, i, i + 1):
                    if 0 <= a < n_cell and 0 <= b < n_cell:
                        want[i, a, b] = rng.normal()
        want = (want + want.transpose(0, 2, 1)) / 2.0
        assert np.array_equal(neighbour_cubic_vertex(n_cell, seed=5), want)


def test_the_promoted_chain_reproduces_its_own_quadrature_and_root():
    r"""``G(n) = G(0) lambda^n`` on the gapped chain, root against quadrature.

    Two independent computations: the root solves the pencil, the blocks come
    from a periodic-trapezoid Brillouin-zone sum. Agreement is the statement
    that the modal form is the bulk Green function and not a fit to it.
    """
    for omega in (0.2, 0.6, 0.9):
        g = gapped_chain_green(omega, 6)
        lam = gapped_chain_root(omega)
        assert abs(lam) < 1.0
        for n in (1, 3, 5):
            assert g[n] == pytest.approx(g[0] * lam ** n, rel=1e-10)


def test_the_quadrature_reference_is_converged_before_it_is_believed():
    """Guards the trap the sibling file records: a reference that stopped
    decaying because it had reached its own noise floor."""
    coarse = gapped_chain_green(0.9, 8, n_k=1024)
    fine = gapped_chain_green(0.9, 8, n_k=4096)
    for n in (4, 6, 8):
        assert abs(coarse[n] - fine[n]) < 1e-12 * abs(fine[n])


# --- the pencil: degree, batching, residual -------------------------------- #

@pytest.mark.parametrize("m,b", [(1, 3), (2, 3), (3, 2), (2, 5)])
def test_a_degree_2m_pencil_returns_2mb_roots_that_solve_it(m, b):
    r"""The generalisation the programme needs, and the count that pins it.

    Once the output pin is removed ``Sigma^R`` has range ``M = 2p + b > 1``, so
    the spatial recurrence is ``sum_{n=-M}^{M} a_n lambda^n = 0``. A quadratic
    solved against a degree-``2M`` reference measures the wrong operator.
    """
    rng = np.random.default_rng(m * 10 + b)
    blocks = [rng.normal(size=(b, b)) + 1j * rng.normal(size=(b, b))
              for _ in range(2 * m + 1)]
    modes = bloch_modes_poly(blocks, residual=True)

    finite = np.isfinite(modes.lam) & (np.abs(modes.lam) > 1e-12)
    assert modes.lam.size == 2 * m * b
    assert finite.sum() == 2 * m * b, "a root collapsed to 0 or infinity"
    assert modes.residual.max() < 1e-8, (
        f"worst NEVP residual {modes.residual.max():.2e}")


def test_an_even_number_of_coefficient_blocks_is_refused():
    with pytest.raises(ValueError, match="odd number"):
        bloch_modes_poly([np.eye(2)] * 4)


def test_non_square_blocks_are_refused():
    with pytest.raises(ValueError, match="square"):
        bloch_modes(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)))


def test_batching_is_the_loop_element_by_element():
    """The whole point of batching is that it changes nothing. A frequency
    sweep is one ``eig`` call over a stacked pencil, and every slice must equal
    what the single-pencil path returns for that slice."""
    rng = np.random.default_rng(11)
    nw, b = 6, 4
    blocks = [rng.normal(size=(nw, b, b)) + 1j * rng.normal(size=(nw, b, b))
              for _ in range(3)]
    batched = bloch_modes(blocks[1], blocks[2], blocks[0], residual=True)
    assert batched.lam.shape == (nw, 2 * b)
    assert batched.vecs.shape == (nw, b, 2 * b)

    for iw in range(nw):
        one = bloch_modes(blocks[1][iw], blocks[2][iw], blocks[0][iw],
                          residual=True)
        assert np.array_equal(one.lam, batched.lam[iw])
        assert np.array_equal(one.vecs, batched.vecs[iw])
        assert np.array_equal(one.residual, batched.residual[iw])


def test_the_band_range_reduces_over_modes_and_not_over_the_batch():
    """A global reduction would report the worst frequency's range at every
    frequency, which reads as a converged answer and is not one."""
    chain = gapped_chain()
    ws = np.array([1.6, 2.5, 3.4])
    a_ii = (ws ** 2)[:, None, None] * np.ones((1, 1)) - chain.h00[None]
    a_ij = np.broadcast_to(-chain.h01, a_ii.shape).copy()
    sigma = -1j * np.array([0.2, 0.6, 1.8])[:, None, None]

    got = band_range_cells(a_ii + sigma, a_ij, a_ij)
    assert np.shape(got) == (3,)
    for iw in range(3):
        one = band_range_cells(a_ii[iw] + sigma[iw], a_ij[iw], a_ij[iw])
        assert got[iw] == pytest.approx(one, rel=1e-12)
    assert got[0] > got[1] > got[2], "more damping must mean a shorter range"


def test_the_residual_separates_a_degenerate_coupling_and_never_drops_a_root():
    r"""A rank-deficient inter-cell coupling makes the pencil degenerate; the
    residual is what says so.

    It is reported as a MASK, not applied. The mode COUNT is itself a
    diagnostic -- ``2Mb`` finite roots, ``Mb`` inside the disc -- and a solver
    that silently returned fewer would destroy it.
    """
    d00 = np.array([[2.6, -0.4], [-0.4, 2.2]])
    d01_bad = np.array([[-0.9, 0.0], [0.0, 0.0]])          # rank 1 on purpose
    assert np.linalg.matrix_rank(d01_bad) == 1
    a_ii = 3.0 * np.eye(2) - d00
    modes = bloch_modes(a_ii, -d01_bad, -d01_bad.conj().T, residual=True)

    assert modes.lam.size == 4, "roots were dropped, not masked"
    good = modes.converged(tol=1e-8)
    assert good.sum() < 4, "this bed is meant to produce unconverged roots"
    assert good.sum() >= 1, "every root spurious: the bed is degenerate, not testing"


def test_converged_refuses_to_answer_when_no_residual_was_computed():
    """An all-true mask would read as 'everything converged'."""
    modes = bloch_modes(*chain2_system_blocks(1.05 * BAND_TOP_2))
    with pytest.raises(ValueError, match="no residual"):
        modes.converged()


def test_the_residual_is_normalised_so_a_root_near_zero_is_not_free():
    r"""Without the ``/|lambda|`` the residual of a tiny root vanishes with it
    and every dead mode looks perfectly converged."""
    rng = np.random.default_rng(3)
    blocks = [rng.normal(size=(3, 3)) for _ in range(3)]
    modes = bloch_modes_poly(blocks, residual=True)
    raw = nevp_residual(blocks, modes.lam, modes.vecs, normalise=False)
    small = np.argmin(np.abs(modes.lam))
    assert raw[small] < modes.residual[small]


# --- the representation: fit, anchor, prune, geometric sums ---------------- #

def _series(omega, anchors=(1, 2), n_max=12):
    modes = bloch_modes(*chain2_system_blocks(omega))
    keep = np.abs(modes.lam) < 1.0 - 1e-12
    blocks = bulk_green_blocks(*chain2_system_blocks(omega), n_max, n_k=2048)
    return modal_fit(modes.vecs[:, keep], modes.lam[keep], blocks,
                     anchors), blocks


def test_the_full_rank_fit_extrapolates_to_every_distance_it_never_saw():
    r""":math:`G(n) = V\,\mathrm{diag}(\lambda^n)\,C`, fitted at ``n = 1, 2``."""
    omega = 1.05 * BAND_TOP_2
    s, exact = _series(omega)
    for n in range(1, 13):
        rel = np.linalg.norm(s.block(n) - exact[n]) / np.linalg.norm(exact[n])
        assert rel < 1e-12, f"n={n}: {rel:.3e}"
    # and it is extrapolation, not a fit sitting on flat data
    fall = np.linalg.norm(exact[12]) / np.linalg.norm(exact[3])
    assert fall < 1e-2


def test_a_fit_with_fewer_equations_than_modes_is_refused():
    """``lstsq`` would return a minimum-norm solution that extrapolates badly
    and reports no error at all."""
    omega = 1.05 * BAND_TOP_2
    modes = bloch_modes(*chain2_system_blocks(omega))
    blocks = bulk_green_blocks(*chain2_system_blocks(omega), 4)
    with pytest.raises(ValueError, match="cannot determine"):
        modal_fit(modes.vecs, modes.lam, blocks, (1,))       # 4 modes, 2 rows


def test_pruning_after_a_full_fit_beats_refitting_a_truncated_basis():
    r"""The anchor/rank rule, as an inequality rather than a number.

    Truncating the mode set and THEN fitting pushes the dropped modes' weight
    onto the survivors, because the design matrix still sees them in the data.
    Fitting everything and dropping afterwards does not.
    """
    rng = np.random.default_rng(3)
    lam = np.array([0.62, 0.31, 1e-3])            # two long-range, one dead
    vecs = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    coef = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
    blocks = {n: vecs @ np.diag(lam ** n) @ coef for n in range(0, 16)}

    full = modal_fit(vecs, lam, blocks, (1, 2))
    pruned, keep = prune_by_amplitude(full, 5, rank=2)
    assert keep.sum() == 2 and not keep[np.argmin(np.abs(lam))]

    refit = modal_fit(vecs[:, keep], lam[keep], blocks, (1, 2))

    def err(series, n):
        return (np.linalg.norm(series.block(n) - blocks[n])
                / np.linalg.norm(blocks[n]))

    for n in (5, 8, 12):
        assert err(pruned, n) < 1e-9, f"prune-after-fit failed at n={n}"
        assert err(pruned, n) < 1e-3 * err(refit, n), (
            f"n={n}: pruned {err(pruned, n):.2e} vs refit {err(refit, n):.2e}")


def test_the_tail_amplitude_ranks_by_contribution_and_not_by_decay_rate():
    r"""A mode with ``|lambda|`` near one is still irrelevant if its
    coefficient is tiny, and pruning on ``|lambda|`` alone would keep it."""
    lam = np.array([0.95, 0.40])
    vecs = np.eye(2, dtype=complex)
    coef = np.array([[1e-8, 0.0], [1.0, 0.0]], dtype=complex)
    from quatrex.phonon.spatial_fit import ModalSeries
    s = ModalSeries(lam=lam, vecs=vecs, coef=coef, anchor=(1,))

    amp = tail_amplitudes(s, 3)
    assert amp[1] > amp[0], "the long-range mode is the negligible one here"
    kept, keep = prune_by_amplitude(s, 3, rank=1)
    assert keep[1] and not keep[0]


def test_prune_by_amplitude_needs_exactly_one_criterion():
    s, _ = _series(1.05 * BAND_TOP_2)
    with pytest.raises(ValueError, match="exactly one"):
        prune_by_amplitude(s, 4)
    with pytest.raises(ValueError, match="exactly one"):
        prune_by_amplitude(s, 4, rank=1, tol=0.1)


@pytest.mark.parametrize("n0,n1", [(1, 9), (3, 25), (2, 200)])
def test_a_range_of_blocks_sums_without_materialising_one(n0, n1):
    s, _ = _series(1.05 * BAND_TOP_2)
    closed = geometric_block_sum(s, n0, n1)
    direct = sum(s.block(n) for n in range(n0, n1))
    assert np.linalg.norm(closed - direct) < 1e-12 * np.linalg.norm(direct)


@pytest.mark.parametrize("k", [1, 2, 5, 37, 200])
def test_the_geometric_factor_is_continuous_through_one(k):
    r"""At ``zeta = 1`` the closed form is 0/0 and the limit is ``k``. Checked
    as a LIMIT, so a solver may cross the branch without a jump."""
    exact_at_one = float(k)
    assert geometric_factor(np.array(1.0 + 0j), k) == pytest.approx(
        exact_at_one, rel=1e-12)
    for eps in (1e-3, 1e-5, 1e-7, 1e-9, 1e-11):
        for u in (eps, -eps, 1j * eps):
            got = complex(geometric_factor(np.array(1.0 + u), k))
            want = complex(np.sum((1.0 + u) ** np.arange(k)))
            assert got == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_the_geometric_series_branch_refuses_a_range_it_cannot_resolve():
    """``|k u| << 1`` is a precondition, not a hope."""
    with pytest.raises(ValueError, match="zeta-1"):
        geometric_factor(np.array(1.0 + 1e-7j), 10 ** 8)


# --- E0: the geometry of the bubble ---------------------------------------- #

N_CELL = 12
W_GRID = np.linspace(0.0, 0.90, 14)
VERTEX_REACH = 1


@pytest.mark.parametrize("band", [0, 1, 2, 3])
def test_the_sigma_support_law_is_two_p_plus_band(band):
    r"""``supp(Sigma) = {|I-J| <= 2p + b}``, with ``p`` the vertex reach.

    Restated here on the promoted bed so the shell decomposition below has a
    pinned premise; the sibling file pins the same law on its own bed.
    """
    _, banded, _ = bed.legs(W_GRID, band, N_CELL)
    s = bed.ring(neighbour_cubic_vertex(N_CELL), banded, banded, W_GRID)
    d = bed.out_distance(N_CELL)
    reach = max(dd for dd in range(N_CELL)
                if np.abs(s[:, d == dd]).max() > 1e-13 * np.abs(s).max())
    assert reach == 2 * VERTEX_REACH + band


def test_the_shell_decomposition_is_exactly_additive():
    r"""``Sigma`` is BILINEAR in ``G``, so splitting the legs by distance shell
    gives ``Sigma_R = sum_{m,m'} Sigma_R^{(m,m')}`` exactly.

    This is what makes the decomposition a decomposition. A ``g_cutoff`` sweep
    gives only the partial sums, and the partial sums are cumulative -- raising
    the band changes blocks that already existed, through interference -- so a
    difference of two of them is not the contribution of a shell.
    """
    exact, _, _ = bed.legs(W_GRID, N_CELL, N_CELL)
    phi = neighbour_cubic_vertex(N_CELL)
    full = bed.ring(phi, exact, exact, W_GRID)
    shells, labels = bed.ring_by_shell(phi, exact, exact, W_GRID)

    assert len(labels) == shells.shape[0] == shells.shape[1]
    rel = (np.abs(shells.sum(axis=(0, 1)) - full).max()
           / np.abs(full).max())
    assert rel < 1e-14, f"shells do not sum to the ring: {rel:.3e}"


def test_a_bin_set_that_does_not_partition_the_distances_is_refused():
    exact, _, _ = bed.legs(W_GRID[:3], N_CELL, N_CELL)
    with pytest.raises(ValueError, match="partition"):
        bed.ring_by_shell(neighbour_cubic_vertex(N_CELL), exact, exact,
                          W_GRID[:3], bins=[(0, 1), (1, N_CELL)])


def test_an_output_at_separation_R_samples_only_legs_within_2p_of_it():
    r"""The finer geometry statement: with external separation ``R`` the two
    internal Green-function links carry ``R - 2p <= r <= R + 2p``.

    Read straight off the shell keys, which is what makes the decomposition
    worth having: a leg-distance shell outside that window must contribute
    exactly nothing to that output distance, whatever its weight.
    """
    exact, _, _ = bed.legs(W_GRID, N_CELL, N_CELL)
    phi = neighbour_cubic_vertex(N_CELL)
    bins = [(m, m) for m in range(N_CELL)]
    shells, _ = bed.ring_by_shell(phi, exact, exact, W_GRID, bins=bins)
    d = bed.out_distance(N_CELL)
    scale = np.abs(shells).max()

    for r_out in range(N_CELL):
        block = np.abs(shells[..., d == r_out]).max(axis=(2, 3))   # (m, m')
        lo, hi = r_out - 2 * VERTEX_REACH, r_out + 2 * VERTEX_REACH
        outside = [(m, mp) for m in range(N_CELL) for mp in range(N_CELL)
                   if not (lo <= m <= hi) or not (lo <= mp <= hi)]
        for m, mp in outside:
            assert block[m, mp] < 1e-13 * scale, (
                f"R={r_out} took weight from legs ({m}, {mp}), outside "
                f"[{lo}, {hi}]")
        inside = block[max(lo, 0):hi + 1, max(lo, 0):hi + 1]
        assert inside.max() > 1e-13 * scale, f"R={r_out} took nothing at all"


def test_the_reblocking_arm_is_a_mask_and_the_block_count_is_what_drives_it():
    r"""Reblocking changes the partition, not the physics, and a dense Dyson
    solve has no block-tridiagonal restriction -- so what reblocking would
    discard is exactly a mask on the same ``Sigma``.

    Combinatorial rather than numerical at the ends: with two blocks the
    largest possible ``|I-J|`` is 1 and a tridiagonal restriction discards
    nothing on ANY bed.
    """
    sigma = bed.long_bed(12)
    fracs = [bed.discarded(sigma, m) for m in (1, 2, 3, 4, 6)]
    assert all(a >= b for a, b in zip(fracs, fracs[1:])), fracs
    assert fracs[0] > 0.0
    assert bed.discarded(sigma, 6) == 0.0        # 2 blocks: nothing to discard
    for n_cell, m in ((6, 3), (8, 4)):
        assert bed.discarded(bed.long_bed(n_cell), m) == 0.0


# --- the exponents that come from the DATA, not from the operator ---------- #
#
# spatial_modes solves the pencil the device defines. These get the exponents by
# fitting the sequence, which is the only route open for G^{<,>}: the Keldysh
# object is not the resolvent of anything, and whether it is a sum of
# exponentials in the separation at all is a measurement.

from quatrex.phonon.spatial_hankel import (                      # noqa: E402
    cluster_exponents, directional_exponents, matrix_pencil, numerical_rank,
    semiseparable_fit, singular_spectrum)


def _same_set(got, want, tol=1e-8):
    """Compared as a SET. A conjugate pair has equal real parts, so any sort
    tie-breaks on the imaginary part by last-digit noise and can hand back the
    two in either order -- which says nothing about the solver."""
    got, want = np.asarray(got), np.asarray(want)
    if got.size != want.size:
        return False
    return (all(np.min(np.abs(got - w)) < tol for w in want)
            and all(np.min(np.abs(want - g)) < tol for g in got))


def _planted(xi, residues, n=24):
    return [sum(a * z ** k for a, z in zip(residues, xi)) for k in range(n)]


def test_a_scalar_sequence_of_r_exponentials_has_hankel_rank_r():
    rng = np.random.default_rng(0)
    xi = np.array([0.8 + 0.2j, 0.55 - 0.3j, 0.2])
    a = rng.normal(size=3) + 1j * rng.normal(size=3)
    seq = _planted(xi, a)
    assert numerical_rank(seq, 1e-10) == 3
    sv = singular_spectrum(seq)
    assert sv[2] > 1e-6 and sv[3] < 1e-12, "the rank cliff is not where it says"


def test_a_block_sequence_has_hankel_rank_r_times_the_residue_rank():
    r"""The correction that matters: ``rank H = sum_p rank(A_p)``.

    Reading a block-Hankel rank as an exponent count overstates it by the block
    size, and the whole programme turns on an exponent count.
    """
    rng = np.random.default_rng(1)
    b, r = 2, 4
    xi = np.array([0.9 * np.exp(0.3j), 0.9 * np.exp(-0.3j), 0.4, 0.15])

    full = [rng.normal(size=(b, b)) + 1j * rng.normal(size=(b, b))
            for _ in range(r)]
    assert numerical_rank(_planted(xi, full), 1e-10) == r * b

    rank_one = [np.outer(rng.normal(size=b) + 0j, rng.normal(size=b))
                for _ in range(r)]
    assert numerical_rank(_planted(xi, rank_one), 1e-10) == r


def test_the_pencil_recovers_planted_exponents_and_their_multiplicity():
    rng = np.random.default_rng(1)
    b, r = 2, 4
    xi = np.array([0.9 * np.exp(0.3j), 0.9 * np.exp(-0.3j), 0.4, 0.15])
    res = [rng.normal(size=(b, b)) + 1j * rng.normal(size=(b, b))
           for _ in range(r)]
    seq = _planted(xi, res)

    est = matrix_pencil(seq)
    assert est.rank == r * b, "the matrix rank is the multiplicity-counted one"
    assert est.n_exponents() == r
    uniq, mult = cluster_exponents(est.xi)
    assert _same_set(uniq, xi, tol=1e-9)
    assert list(mult) == [b] * r
    assert est.rel_error(seq).max() < 1e-10


def test_the_pencil_degrades_monotonically_as_noise_is_added():
    """Zero noise is not a demanding test on its own; the interesting property
    is that the estimate does not fall off a cliff."""
    rng = np.random.default_rng(4)
    xi = np.array([0.85, 0.45 + 0.2j, 0.45 - 0.2j])
    a = rng.normal(size=3) + 1j * rng.normal(size=3)
    clean = np.asarray(_planted(xi, a, n=30))
    scale = np.abs(clean).max()

    prev = 0.0
    for level in (0.0, 1e-10, 1e-7, 1e-4):
        noisy = clean + level * scale * (rng.normal(size=clean.shape)
                                         + 1j * rng.normal(size=clean.shape))
        est = matrix_pencil(noisy, rank=3)
        err = max(np.min(np.abs(est.xi - z)) for z in xi)
        assert err >= prev * 0.5, f"error fell as noise grew: {err} < {prev}"
        assert err < max(1e-9, 300.0 * level), (
            f"noise {level:.0e} gave exponent error {err:.2e}")
        prev = err


def test_a_toeplitz_object_and_a_semiseparable_one_are_distinguishable():
    r"""The structural question, decided without any fitting choice.

    ``M_{IJ}`` that is a function of ``J - I`` alone gives the SAME exponents
    along ``I`` and along ``J``. One with a residue ``(mu nu)^I`` -- which is
    what a source-resolved Keldysh derivation produces -- does not, and that
    difference is what says the ``Sigma_R = C A^{R-1} B`` ansatz is the wrong
    class rather than merely underranked.
    """
    n = 14
    lam = np.array([0.75 + 0.1j, 0.35])
    amp = np.array([1.0 + 0.0j, 0.4 - 0.2j])
    toe = np.array([[sum(amp * lam ** abs(j - i)) for j in range(n)]
                    for i in range(n)])
    d = directional_exponents(toe, rank=2, anchor=3, span=8)
    assert _same_set(d["along_J"].xi, d["along_I"].xi, tol=1e-7)

    mu = np.array([0.7 + 0.1j, 0.3])
    nu = np.array([0.6 - 0.2j, 0.25])
    coef = np.array([[1.0, 0.5], [-0.3, 0.8]], dtype=complex)
    sep = np.array([[sum(coef[a, c] * mu[a] ** i * nu[c] ** j
                         for a in range(2) for c in range(2))
                     for j in range(n)] for i in range(n)])
    d2 = directional_exponents(sep, rank=2, anchor=3, span=8)
    assert not _same_set(d2["along_J"].xi, d2["along_I"].xi, tol=1e-3)


def test_the_semiseparable_fit_reproduces_the_class_it_is_written_for():
    n = 12
    mu = np.array([0.7 + 0.1j, 0.3])
    nu = np.array([0.6 - 0.2j, 0.25])
    coef = np.array([[1.0, 0.5], [-0.3, 0.8]], dtype=complex)
    mat = np.array([[sum(coef[a, c] * mu[a] ** i * nu[c] ** j
                         for a in range(2) for c in range(2))
                     for j in range(n)] for i in range(n)])

    fit = semiseparable_fit(mat, n_dof=1, rank=2)
    assert _same_set(fit.mu, mu, tol=1e-9)
    assert _same_set(fit.nu, nu, tol=1e-9)
    assert fit.rel_error(mat, 1) < 1e-10


def test_the_semiseparable_fit_cross_validates_on_cells_it_never_saw():
    """A fit that only reproduces its own training cells has proved nothing."""
    n = 16
    mu = np.array([0.72, 0.28 + 0.15j])
    nu = np.array([0.66, 0.31 - 0.1j])
    coef = np.array([[1.0, -0.4], [0.7, 0.25]], dtype=complex)
    mat = np.array([[sum(coef[a, c] * mu[a] ** i * nu[c] ** j
                         for a in range(2) for c in range(2))
                     for j in range(n)] for i in range(n)])

    train = list(range(2, 10))
    held = list(range(10, 16))
    fit = semiseparable_fit(mat, n_dof=1, rank=2, cells=train)
    assert fit.rel_error(mat, 1, cells=train) < 1e-10
    assert fit.rel_error(mat, 1, cells=held) < 1e-8, "no generalisation"


def test_the_pencil_refuses_a_sequence_too_short_to_shift():
    with pytest.raises(ValueError, match="at least 2"):
        matrix_pencil([np.eye(2)])
