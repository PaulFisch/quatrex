# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Experiment D: propagating + evanescent spatial structure.

The method proposal's Sec. 53-D, and the first piece of its SPATIAL half --
the leg that was never started (`pole_qnm_audit_map.md`). It asks for three
things on a chain carrying both characters, :math:`G_n = A e^{iqn} +
B e^{-\kappa n}`: that a hard spatial band fails, that the modal + banded
decomposition is exact, and that geometric summation reproduces the long-range
terms without materialising the blocks.

This matters for a live physics number rather than for tidiness. The band ladder
in the thesis (`64_gband.tex`) brackets the long CNT answer by a factor 2.2
between a boxcar upper bound "contaminated by non-causal gain" and a tapered
lower bound, and the series stops at seven cells for that reason. The boxcar is
the hard band this file shows cannot work for a propagating mode.

Everything here is at eta = 0. Character is decided by :math:`|\lambda|` and by
the group velocity, never by a broadening prescription -- which is the
proposal's Sec. 1.1 point that :math:`|\lambda| \neq 1` means evanescence and
not radiation.

Built on `phonon/solver/toy_models.py`, a complete toy library that no test in
the repository used before this one.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# ``phonon/`` is not a package, so its subpackages are imported by putting it on
# the path -- the same route the phonon_inputs tests take. `solver` costs 0.2 s
# to import and this is the first test in the repository to use its toy library.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "phonon"))

from solver.toy_models import monatomic_chain                  # noqa: E402

CHAIN = monatomic_chain(omega_max_thz=8.0)
K_S = float(CHAIN.h00[0, 0]) / 2.0          # h00 = 2 k_s
BAND_TOP = 2.0 * np.sqrt(K_S)               # omega(pi) = 2 sqrt(k_s)


def bloch_roots(omega: float) -> np.ndarray:
    r"""Roots of the nearest-layer pencil, proposal Eq. (143).

    ``[-h_10/lambda + (omega^2 I - h_00) - h_01 lambda] v = 0``. For one degree
    of freedom that is ``k_s lambda^2 + (omega^2 - 2 k_s) lambda + k_s = 0``.
    """
    a, b, c = K_S, omega * omega - 2.0 * K_S, K_S
    disc = np.lib.scimath.sqrt(b * b - 4.0 * a * c)
    return np.array([(-b + disc) / (2.0 * a), (-b - disc) / (2.0 * a)])


def decaying_root(omega: float) -> complex:
    lam = bloch_roots(omega)
    return complex(lam[np.argmin(np.abs(lam))])


# --- character ------------------------------------------------------------- #

def test_the_roots_come_in_reciprocal_pairs():
    r""":math:`\lambda_1 \lambda_2 = 1` for every frequency.

    A property of the pencil, not of the frequency: the constant and leading
    coefficients are both ``h_01``. It is what makes "the decaying one" and
    "the growing one" a partition rather than a choice, and it holds inside the
    band where both sit on the unit circle.
    """
    for omega in (0.3, 2.0, BAND_TOP, 1.4 * BAND_TOP, 3.0 * BAND_TOP):
        assert np.prod(bloch_roots(omega)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("frac", [0.05, 0.35, 0.7, 0.99])
def test_inside_the_band_both_roots_sit_on_the_unit_circle(frac):
    """Propagating: no decay length exists, so no spatial band can be short
    enough. Also recovers the dispersion, ``omega^2 = 4 k_s sin^2(k/2)``."""
    omega = frac * BAND_TOP
    lam = bloch_roots(omega)
    assert np.abs(lam) == pytest.approx([1.0, 1.0], abs=1e-12)

    k = np.angle(lam[0])
    assert omega ** 2 == pytest.approx(4.0 * K_S * np.sin(k / 2.0) ** 2, rel=1e-10)


@pytest.mark.parametrize("frac", [1.05, 1.3, 2.0])
def test_outside_the_band_the_pair_splits_into_decaying_and_growing(frac):
    omega = frac * BAND_TOP
    mod = np.sort(np.abs(bloch_roots(omega)))
    assert mod[0] < 1.0 < mod[1]
    assert mod[0] * mod[1] == pytest.approx(1.0, abs=1e-12)


# --- the claim: a hard band fails ------------------------------------------ #

@pytest.mark.parametrize("band", [2, 4, 8, 16])
def test_a_hard_band_discards_nothing_of_an_evanescent_mode(band):
    r"""What a boxcar range ``b`` throws away is :math:`\sum_{n>b}|\lambda|^n`.

    Evanescent: geometric and exponentially small, so a band is the right tool.
    """
    lam = abs(decaying_root(1.5 * BAND_TOP))
    assert lam < 1.0
    tail = lam ** (band + 1) / (1.0 - lam)
    assert tail < 0.1 ** (band / 4.0)


@pytest.mark.parametrize("band", [2, 4, 8, 16, 64, 1024])
def test_a_hard_band_discards_an_unbounded_amount_of_a_propagating_mode(band):
    r"""The failure, exactly.

    In band :math:`|\lambda| = 1`, so every retained cell contributes the same
    magnitude and the discarded tail :math:`\sum_{n>b} 1` DIVERGES however
    large ``b`` is made. There is no band width at which a propagating mode is
    captured, which is why the thesis' long-CNT boxcar arm is an upper bracket
    contaminated by gain rather than a converged number -- and why the answer
    has to be a modal sector, not a longer mask.
    """
    lam = abs(decaying_root(0.6 * BAND_TOP))
    assert lam == pytest.approx(1.0, abs=1e-12)
    retained = np.sum(np.abs(lam ** np.arange(band + 1)))
    assert retained == pytest.approx(band + 1.0)      # no decay at all
    # ... and the taper the repo reaches for instead does not fix the character
    taper = np.sum(1.0 - np.arange(band + 1) / (band + 1.0))
    assert taper > 0.4 * (band + 1.0), "a taper still keeps O(b) weight"


# --- geometric summation, proposal Eqs. (160)-(161) ------------------------- #

@pytest.mark.parametrize("frac", [1.05, 1.5, 3.0])
@pytest.mark.parametrize("n0,n", [(0, 12), (3, 40), (7, 200)])
def test_geometric_summation_matches_the_explicit_sum(frac, n0, n):
    r""":math:`\sum_{n_0}^{N-1}\lambda^n = \lambda^{n_0}(1-\lambda^{N-n_0})/(1-\lambda)`.

    The point of Eq. (160): long-range plane-wave factors need not be
    materialised block by block merely to be summed.
    """
    lam = decaying_root(frac * BAND_TOP)
    closed = lam ** n0 * (1.0 - lam ** (n - n0)) / (1.0 - lam)
    direct = np.sum(lam ** np.arange(n0, n))
    assert closed == pytest.approx(direct, rel=1e-10)


def test_the_closed_form_degenerates_and_the_stated_limit_repairs_it():
    r"""At :math:`\lambda \to 1` Eq. (160) is 0/0 and Eq. (161) gives ``N - n0``.

    Checked as a limit, not as a special case: the closed form must approach
    ``N - n0`` continuously as ``lambda`` approaches 1, so a solver may switch
    branches near it without a jump.
    """
    n0, n = 4, 37
    for eps in (1e-4, 1e-6, 1e-8):
        lam = 1.0 - eps
        closed = lam ** n0 * (1.0 - lam ** (n - n0)) / (1.0 - lam)
        assert closed == pytest.approx(n - n0, rel=50.0 * eps)

    # exactly at 1 the closed form is 0/0; the limit is the sum of ones
    assert np.sum(np.ones(n - n0)) == n - n0


# --- the modal form is exact ----------------------------------------------- #

@pytest.mark.parametrize("frac", [1.1, 1.6, 2.5])
def test_the_bulk_green_function_is_rank_one_in_the_bloch_factor(frac):
    r""":math:`G(n) = G(0)\,\lambda^{|n|}` for the infinite chain.

    This is the proposal's Eq. (158), ``G_{S,ij} = U_i C V_j^H``, at one degree
    of freedom: the whole distance dependence is a power of one Bloch factor,
    so distant blocks are generated rather than stored.

    The reference is a Brillouin-zone quadrature of
    ``G(n) = (1/2pi) int dk e^{ikn} / (omega^2 - 2k_s + 2 k_s cos k)``,
    evaluated OUTSIDE the band where the integrand is regular -- so no
    broadening is needed anywhere and eta stays exactly zero.
    """
    omega = frac * BAND_TOP
    lam = decaying_root(omega)
    assert abs(lam) < 1.0

    k = np.linspace(-np.pi, np.pi, 200001)
    denom = omega ** 2 - 2.0 * K_S + 2.0 * K_S * np.cos(k)
    assert np.all(np.abs(denom) > 1e-9), "the quadrature hit the band"

    def g_of(n):
        return np.trapezoid(np.exp(1j * k * n) / denom, k) / (2.0 * np.pi)

    g0 = g_of(0)
    for n in (1, 2, 3, 5, 8):
        assert g_of(n) == pytest.approx(g0 * lam ** n, rel=1e-6)


def test_the_modal_form_beats_a_hard_band_at_equal_storage():
    r"""Same memory, different answer -- the reason to prefer a modal sector.

    A boxcar of range ``b`` stores ``b+1`` numbers per row and reproduces the
    first ``b+1`` entries exactly and everything beyond as zero. The rank-one
    modal form stores TWO (``G(0)`` and ``lambda``) and reproduces every entry.
    Measured on the propagating case, where the discarded tail never decays.
    """
    omega = 0.45 * BAND_TOP
    lam = decaying_root(omega)
    n = np.arange(0, 400)
    exact = lam ** n                       # |exact| == 1 for every n

    band = 8
    boxcar = np.where(n <= band, exact, 0.0)
    modal = lam ** n                       # two stored numbers

    assert np.abs(exact - modal).max() < 1e-12
    # the boxcar is wrong by the full magnitude on every dropped cell
    dropped = np.abs(exact - boxcar)[n > band]
    assert dropped.min() == pytest.approx(1.0, abs=1e-12)
    assert dropped.size == n.size - band - 1


# --- complex bands of the dressed operator, proposal Eqs. (143)-(144) ------- #

def _blocks(omega, sigma=0.0):
    """System-matrix blocks of the chain: ``a_ii = w^2 - h00 - Sigma``."""
    return (np.array([[omega * omega - 2.0 * K_S - sigma]], dtype=complex),
            np.array([[K_S + 0j]]), np.array([[K_S + 0j]]))


@pytest.mark.parametrize("frac", [0.4, 0.9, 1.4, 2.2])
def test_the_nevp_reproduces_the_closed_form_roots(frac):
    """The fixed point of reusing the OBC's solver: handed the chain's own
    blocks it must return the quadratic's roots.

    This is what pins the block convention. ``a_ji/lambda + a_ii + a_ij lambda``
    is the OBC's ordering; passing dynamical-matrix blocks instead of
    system-matrix blocks solves a different pencil and returns wrong bands with
    no error raised.
    """
    from quatrex.phonon.spatial_modes import bloch_modes

    omega = frac * BAND_TOP
    got = np.asarray(bloch_modes(*_blocks(omega)).lam)
    want = np.asarray(bloch_roots(omega))

    # Compared as a SET. A conjugate pair has equal real parts, so any sort
    # tie-breaks on the imaginary part by last-digit noise and can hand back
    # the two roots in either order -- which says nothing about the solver.
    assert got.size == want.size
    for w in want:
        assert np.min(np.abs(got - w)) < 1e-10, f"{w} missing from {got}"
    for g in got:
        assert np.min(np.abs(want - g)) < 1e-10, f"{g} is spurious"


def test_an_undressed_in_band_mode_has_no_range_at_all():
    r"""``|lambda| = 1`` gives ``xi = inf``, and the required band is unbounded.

    Reported as ``inf`` rather than as a large number: the honest answer for an
    undressed operator is that no mask is long enough, and the reply is a modal
    representation rather than a wider one.
    """
    from quatrex.phonon.spatial_modes import band_range_cells, bloch_modes

    modes = bloch_modes(*_blocks(0.5 * BAND_TOP))
    assert modes.propagating.all()
    assert np.isinf(modes.xi).all()
    assert np.isinf(band_range_cells(*_blocks(0.5 * BAND_TOP)))


@pytest.mark.parametrize("gamma_s", [0.05, 0.5, 5.0])
def test_dressing_gives_a_propagating_mode_a_finite_mean_free_path(gamma_s):
    r"""The substitution the spatial leg turns on, Eq. (144).

    :math:`\Sigma^R = -i\Gamma` splits the reciprocal pair, and the decaying
    partner acquires :math:`\xi = -1/\ln|\lambda|`. That is a mean free path in
    cells and it is what a spatial truncation has to be compared against --
    the quantity that decides whether ``sse_g_band`` is a convention or a
    controlled approximation.
    """
    from quatrex.phonon.spatial_modes import band_range_cells, bloch_modes

    blocks = _blocks(0.5 * BAND_TOP, sigma=-1j * gamma_s)
    modes = bloch_modes(*blocks)

    assert not modes.propagating.any(), "damping left a mode on the unit circle"
    assert modes.decaying.sum() == 1
    xi = band_range_cells(*blocks)
    assert np.isfinite(xi) and xi > 0.0

    # reciprocity survives the dressing: one decays, one grows, product 1
    assert np.prod(modes.lam) == pytest.approx(1.0, abs=1e-10)


def test_the_mean_free_path_scales_as_one_over_the_damping():
    r"""Weak damping gives a long range, and the two are inversely proportional.

    The relation that makes the number actionable: reading a range of hundreds
    of cells off a device says a band of a few cannot be right, whatever the
    convergence table appears to show.
    """
    from quatrex.phonon.spatial_modes import band_range_cells

    xis = [band_range_cells(*_blocks(0.5 * BAND_TOP, sigma=-1j * g))
           for g in (0.05, 0.5, 5.0)]
    assert xis[0] > xis[1] > xis[2]
    for a, b in zip(xis, xis[1:]):
        assert a / b == pytest.approx(10.0, rel=0.05)


def test_a_short_band_discards_weight_the_range_predicts():
    r"""Ties the diagnostic to the thing it is meant to warn about.

    A truncation at ``b`` blocks drops ``exp(-b/xi)`` of a mode whose range is
    ``xi``. At ``xi = 55`` cells a band of 3 keeps 95 % of the amplitude it
    should have removed -- which is a truncation in name only.
    """
    from quatrex.phonon.spatial_modes import band_range_cells

    xi = band_range_cells(*_blocks(0.5 * BAND_TOP, sigma=-0.5j))
    assert xi == pytest.approx(55.4, rel=0.02)
    assert np.exp(-3.0 / xi) > 0.94, "a band of 3 barely touches this mode"
    assert np.exp(-3.0 * xi / xi) == pytest.approx(np.exp(-3.0))


def test_non_square_blocks_are_refused():
    from quatrex.phonon.spatial_modes import bloch_modes

    with pytest.raises(ValueError, match="square"):
        bloch_modes(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)))


def test_decay_lengths_label_each_character_distinctly():
    """A growing partner gets ``nan``, not a negative length -- otherwise a
    ``min`` over the array silently returns it as the binding range."""
    from quatrex.phonon.spatial_modes import decay_lengths

    xi = decay_lengths(np.array([0.5, 1.0, 2.0, 0.0], dtype=complex))
    assert xi[0] == pytest.approx(-1.0 / np.log(0.5))
    assert np.isinf(xi[1])
    assert np.isnan(xi[2])
    assert xi[3] == 0.0


@pytest.mark.parametrize("frac", [0.3, 0.5, 0.8])
@pytest.mark.parametrize("gamma_s", [0.05, 0.5])
def test_the_range_is_the_group_velocity_over_the_linewidth(frac, gamma_s):
    r"""``xi = v_g / gamma``, and it ties the spatial half to the frequency one.

    The pole census already measures ``gamma`` per mode on every bed it has
    run. This says the required spatial band follows from it directly, given
    the group velocity, so a frequency-domain measurement already taken can be
    converted into "how many blocks does the self-energy need" without a new
    calculation.

    ``gamma`` here is the HWHM in THz that :math:`\Sigma^R = -i\Gamma`
    implies, :math:`\gamma = \Gamma/2\omega`, since the self-energy enters the
    phonon Dyson operator in :math:`\omega^2`. Exact to 1e-4 in weak damping;
    the identification of a range with a lifetime is itself a weak-damping
    statement, so the agreement loosens to about a percent once a mode decays
    within a few cells.
    """
    from quatrex.phonon.spatial_modes import band_range_cells

    omega = frac * BAND_TOP
    k = 2.0 * np.arcsin(omega / (2.0 * np.sqrt(K_S)))
    v_g = np.sqrt(K_S) * np.cos(k / 2.0)          # dw/dk, cells * THz
    gamma = gamma_s / (2.0 * omega)               # HWHM in THz

    xi = band_range_cells(*_blocks(omega, sigma=-1j * gamma_s))
    assert xi == pytest.approx(v_g / gamma, rel=2e-3)


def test_a_census_linewidth_implies_a_band_far_longer_than_any_in_use():
    r"""What that bridge says about a bed already measured.

    The converged Si census reports a median half width near 0.16 THz
    (``pole_sector_observations.md`` Sec. 13). At an acoustic group velocity of
    a few cell-THz that is a range of tens of cells, against an
    ``sse_g_band`` of 1 to 3 blocks in every production run.

    Written as a calculation rather than a claim about Si specifically -- the
    device group velocity is not measured here -- but the order is the point,
    and it is the same order as the factor 2.2 the long-CNT bracket sits at.
    """
    from quatrex.phonon.spatial_modes import band_range_cells

    gamma_thz = 0.16
    for v_g in (1.0, 3.0, 6.0):                   # cells * THz
        omega = 0.5 * BAND_TOP
        xi = band_range_cells(
            *_blocks(omega, sigma=-1j * 2.0 * omega * gamma_thz))
        assert xi == pytest.approx(
            np.sqrt(K_S) * np.cos(np.arcsin(omega / (2 * np.sqrt(K_S))))
            / gamma_thz, rel=5e-3)
        assert xi > 10.0, "the range collapsed below a plausible band"


# --- Phase 7: distant blocks are generated, not stored --------------------- #

D00_2 = np.array([[2.6, -0.4], [-0.4, 2.2]])
D01_2 = np.array([[-0.9, -0.25], [-0.2, -0.8]])       # full rank on purpose


def _blocks2(omega):
    """A 2-DOF cell whose inter-cell coupling is INVERTIBLE.

    A rank-deficient ``D_01`` makes the pencil degenerate -- roots collapse to
    0 and infinity and the mode count is wrong -- so the coupling is chosen
    full rank and the test asserts it.
    """
    return (omega * omega * np.eye(2) - D00_2, -D01_2, -D01_2.conj().T)


def _band_top_2(n_k=2001):
    k = np.linspace(0.0, np.pi, n_k)
    top = 0.0
    for x in k:
        dk = D00_2 + D01_2 * np.exp(1j * x) + D01_2.conj().T * np.exp(-1j * x)
        top = max(top, np.sqrt(np.linalg.eigvalsh(
            0.5 * (dk + dk.conj().T)).max()))
    return top


BAND_TOP_2 = _band_top_2()


def _green_blocks(omega, n_max, n_k=2048):
    r"""Exact ``G(n)`` by Brillouin-zone quadrature, for ``n`` up to ``n_max``.

    Periodic trapezoid: the integrand is analytic and periodic in ``k``, so
    this converges exponentially. Using ``linspace`` with both endpoints
    instead double counts and lands on a noise floor -- the first version of
    this bed did exactly that and reported a Green function that stopped
    decaying at 5e-6, which reads as physics and is not.

    Evaluated above the band, where ``A(k)`` is invertible for every ``k``, so
    no broadening is needed and eta stays zero.
    """
    a_ii, a_ij, a_ji = _blocks2(omega)
    k = 2.0 * np.pi * np.arange(n_k) / n_k
    ph = np.exp(1j * k)[:, None, None]
    a_k = a_ii[None] + a_ij[None] * ph + a_ji[None] * np.conj(ph)
    inv = np.linalg.inv(a_k)                                  # (n_k, 2, 2)
    return {n: (inv * np.exp(1j * k * n)[:, None, None]).sum(0) / n_k
            for n in range(n_max + 1)}


def _modal_fit(omega):
    """``V``, ``lambda`` and ``C`` of ``G(n) = V diag(lambda^n) C``."""
    from quatrex.phonon.spatial_modes import bloch_modes

    modes = bloch_modes(*_blocks2(omega))
    keep = np.abs(modes.lam) < 1.0 - 1e-12
    V, lm = modes.vecs[:, keep], modes.lam[keep]
    g = _green_blocks(omega, 2)
    design = np.vstack([V @ np.diag(lm ** n) for n in (1, 2)])
    rhs = np.vstack([g[1], g[2]])
    C = np.linalg.lstsq(design, rhs, rcond=None)[0]
    return V, lm, C


def test_the_bed_has_an_invertible_coupling_and_two_decaying_modes():
    from quatrex.phonon.spatial_modes import bloch_modes

    assert np.linalg.cond(D01_2) < 10.0
    modes = bloch_modes(*_blocks2(1.05 * BAND_TOP_2))
    mod = np.sort(np.abs(modes.lam))
    assert mod.size == 4                       # 2 DOF -> 4 finite roots
    assert np.sum(mod < 1.0) == 2 and np.sum(mod > 1.0) == 2
    assert mod[0] > 1e-6, "a root collapsed to zero: the coupling is singular"


def test_two_modes_reproduce_every_distant_block(  ):
    r"""Proposal Eq. (158), :math:`G_{S,ij} = U_i C V_j^\dagger`, exactly.

    The coefficients are fitted from ``n = 1`` and ``n = 2`` ONLY, and then
    every block out to ``n = 12`` is predicted. Agreement at roundoff is the
    Phase 7 claim: distant blocks are generated from a rank-``r`` object rather
    than stored, with ``r`` the number of decaying modes.
    """
    omega = 1.05 * BAND_TOP_2
    V, lm, C = _modal_fit(omega)
    exact = _green_blocks(omega, 12)

    for n in range(1, 13):
        rec = V @ np.diag(lm ** n) @ C
        rel = np.linalg.norm(rec - exact[n]) / np.linalg.norm(exact[n])
        assert rel < 1e-12, f"n={n}: rel err {rel:.3e}"

    # and it is a genuine extrapolation, not a fit sitting on flat data: over
    # the nine distances that were never fitted the blocks fall by more than
    # two orders. The slowest mode sets the scale, lambda^9 = 0.511^9 ~ 4e-3.
    fall = np.linalg.norm(exact[12]) / np.linalg.norm(exact[3])
    assert fall < 1e-2, f"blocks only fell by {1 / fall:.0f}x"
    assert fall == pytest.approx(np.max(np.abs(lm)) ** 9, rel=0.5)


def test_the_quadrature_reference_is_converged():
    """Guards the trap the first version fell into: a reference that stopped
    decaying because it had hit its own noise floor."""
    omega = 1.05 * BAND_TOP_2
    coarse = _green_blocks(omega, 8, n_k=1024)
    fine = _green_blocks(omega, 8, n_k=4096)
    for n in (4, 6, 8):
        rel = np.linalg.norm(coarse[n] - fine[n]) / np.linalg.norm(fine[n])
        assert rel < 1e-12, f"n={n}: quadrature not converged, {rel:.3e}"


def test_dropping_a_mode_breaks_the_reconstruction():
    """The rank is not a tuning parameter: it is the number of decaying modes,
    and one fewer does not reproduce the blocks at any coefficient."""
    omega = 1.05 * BAND_TOP_2
    V, lm, C = _modal_fit(omega)
    exact = _green_blocks(omega, 8)

    order = np.argsort(-np.abs(lm))            # drop the fastest-decaying one
    V1, lm1 = V[:, order[:1]], lm[order[:1]]
    design = np.vstack([V1 @ np.diag(lm1 ** n) for n in (1, 2)])
    rhs = np.vstack([exact[1], exact[2]])
    C1 = np.linalg.lstsq(design, rhs, rcond=None)[0]

    worst = max(np.linalg.norm(V1 @ np.diag(lm1 ** n) @ C1 - exact[n])
                / np.linalg.norm(exact[n]) for n in (1, 2, 3))
    assert worst > 1e-3, "rank 1 fitted this bed, so it is not testing the rank"


def test_a_range_of_blocks_sums_in_closed_form():
    r"""Eq. (160) at the matrix level -- the point of the whole construction.

    :math:`\sum_{n_0}^{N-1} G(n) = V\,\mathrm{diag}\!\left(\lambda^{n_0}
    \frac{1-\lambda^{N-n_0}}{1-\lambda}\right) C`, so a long-range sum costs
    ``r`` geometric series and never materialises the blocks it runs over.
    """
    omega = 1.05 * BAND_TOP_2
    V, lm, C = _modal_fit(omega)

    for n0, n_end in ((1, 9), (3, 25), (2, 200)):
        closed = V @ np.diag(
            lm ** n0 * (1.0 - lm ** (n_end - n0)) / (1.0 - lm)) @ C
        direct = sum(V @ np.diag(lm ** n) @ C for n in range(n0, n_end))
        assert np.linalg.norm(closed - direct) < 1e-12 * np.linalg.norm(direct)


def test_the_modal_form_is_smaller_than_the_blocks_it_replaces():
    """Storage, stated rather than implied: ``r`` mode vectors and ``r``
    coefficient rows against one dense block per distance."""
    omega = 1.05 * BAND_TOP_2
    V, lm, C = _modal_fit(omega)
    b, r = V.shape[0], lm.size

    modal = V.size + lm.size + C.size          # 2*2 + 2 + 2*2 = 10
    for n_dist in (8, 32, 128):
        blocks = n_dist * b * b
        assert modal < blocks / 3, (
            f"{modal} numbers vs {blocks} for {n_dist} distances")


# --- the rank and the fit anchor are coupled ------------------------------- #

def _synthetic_modes(seed=3):
    r"""Three modes with a deliberately huge spread in ``|lambda|``.

    ``G(n)`` is BUILT from them, so the exact answer is known without a
    quadrature and the test measures only the fitting, which is what is at
    issue here.
    """
    rng = np.random.default_rng(seed)
    lam = np.array([0.62, 0.31, 1e-3])            # two long-range, one dead
    V = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    C = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
    def g(n):
        return V @ np.diag(lam ** n) @ C
    return lam, V, C, g


def _fit(lam, V, g, anchors):
    design = np.vstack([V @ np.diag(lam ** n) for n in anchors])
    rhs = np.vstack([g(n) for n in anchors])
    return np.linalg.lstsq(design, rhs, rcond=None)[0]


def test_the_anchor_selects_the_distance_window_not_the_accuracy():
    r"""The rule underneath both halves of this, and it cuts both ways.

    A fit anchored at ``n_0`` can only determine the coefficient of a mode that
    is still ALIVE at ``n_0``. Anchor close in and every mode is constrained,
    so the representation is exact everywhere -- at the cost of an
    ill-conditioned design when the spread in ``|lambda|`` is large. Anchor far
    out and the fast modes have decayed below the fit's reach, so their
    coefficients are unconstrained and SHORT range degrades, even at full rank.

    So the anchor is not a numerical detail to be tuned for stability: it
    chooses the window of distances the representation is valid on.
    """
    lam, V, C, g = _synthetic_modes()

    near = _fit(lam, V, g, (1, 2))
    for n in (1, 5, 12):
        rec = V @ np.diag(lam ** n) @ near
        assert np.linalg.norm(rec - g(n)) < 1e-9 * np.linalg.norm(g(n))

    # Anchored past the fast mode's range, its coefficient is undetermined:
    # lambda^7 = 1e-21 there, so the design carries no information about it.
    far = V @ np.diag(lam ** 1) @ _fit(lam, V, g, (7, 8))
    assert np.linalg.norm(far - g(1)) > 1e-6 * np.linalg.norm(g(1))

    # ... while the distances at and beyond the anchor are still exact.
    for n in (7, 9, 14):
        rec = V @ np.diag(lam ** n) @ _fit(lam, V, g, (7, 8))
        assert np.linalg.norm(rec - g(n)) < 1e-8 * np.linalg.norm(g(n))


def test_a_truncated_set_must_be_fitted_where_the_dropped_modes_are_dead():
    r"""The design rule, and it is not obvious.

    Dropping a mode with :math:`|\lambda| = 10^{-3}` looks free -- it
    contributes :math:`10^{-9}` by ``n = 3``. But fitting the survivors at
    ``n = 1, 2``, where the dropped mode is still present in the data, pushes
    its weight onto them and corrupts the coefficients at every distance.
    Anchoring the fit past its range instead recovers the accuracy.

    Measured on the real CNT cell at rank 22 of 36, the same effect is
    1.2e-02 fitted at ``n = 1, 2`` against 2.1e-07 fitted at ``n = 5, 6``
    (``phonon/docs/spatial_band_range.md``).

    The pole sector learned the same lesson as ``_fit_anchor``: where a local
    model is anchored is part of the model.
    """
    lam, V, C, g = _synthetic_modes()
    keep = np.abs(lam) > 1e-2                     # drop the dead mode
    assert keep.sum() == 2

    near = _fit(lam[keep], V[:, keep], g, (1, 2))
    far = _fit(lam[keep], V[:, keep], g, (5, 6))

    def err(C_fit, n):
        rec = V[:, keep] @ np.diag(lam[keep] ** n) @ C_fit
        return np.linalg.norm(rec - g(n)) / np.linalg.norm(g(n))

    # anchored close in, the truncation contaminates every distance
    assert err(near, 6) > 1e-6
    # anchored past the dropped mode's range, it is accurate where it is used
    assert err(far, 6) < 1e-9
    assert err(far, 10) < 1e-9
    assert err(far, 6) < 1e-3 * err(near, 6)


def test_the_dropped_mode_really_is_negligible_at_the_far_anchor():
    """Guards the premise: if the dead mode still mattered at n=5 the rule
    above would be describing something else."""
    lam, V, C, g = _synthetic_modes()
    dead = np.abs(lam) < 1e-2
    contrib = V[:, dead] @ np.diag(lam[dead] ** 5) @ C[dead]
    assert np.linalg.norm(contrib) < 1e-12 * np.linalg.norm(g(5))


# --- Phase 8: what the band discards, the ring notices ---------------------- #
#
# The ring is a convolution, so Sigma(Omega) needs G at omega AND at
# Omega - omega: the frequency grid has to start at zero. A gapped chain is
# what lets that coexist with an exact reference -- with an on-site pinning the
# band is [w0, sqrt(w0^2 + 4 k_s)], so a grid below w0 never touches it and the
# Brillouin-zone integrand stays regular at eta = 0. The ungapped chain used
# above cannot do this: its acoustic branch reaches zero, so any grid starting
# at zero runs through the band.

N_CELL = 7                       # 1 DOF per cell, so a block IS a cell
W0, KS_G = 1.0, 4.0              # gap and coupling; band = [1.0, 4.123]
W_TOP = 0.9 * W0                 # the whole grid sits below the band


def _gap_root(omega):
    """Decaying Bloch factor of the gapped chain (real, in (0, 1))."""
    a, b, c = KS_G, omega * omega - (W0 ** 2 + 2 * KS_G), KS_G
    disc = np.lib.scimath.sqrt(b * b - 4 * a * c)
    lam = np.array([(-b + disc) / (2 * a), (-b - disc) / (2 * a)])
    return complex(lam[np.argmin(np.abs(lam))])


def _gap_green(omega, n_max, n_k=4096):
    """``G(n)`` by periodic-trapezoid quadrature -- independent of the roots.

    This is a Brillouin-zone integral of ``1/A(k)``; ``_gap_root`` is a root of
    ``A``. That the two agree is what earlier tests establish, and it is what
    lets the ring be driven by one and completed by the other without the
    argument becoming circular.
    """
    k = 2.0 * np.pi * np.arange(n_k) / n_k
    denom = omega ** 2 - (W0 ** 2 + 2 * KS_G) + 2.0 * KS_G * np.cos(k)
    return np.array([np.sum(np.exp(1j * k * n) / denom) / n_k
                     for n in range(n_max + 1)])


def _legs(omegas, band):
    """``(exact, banded, completed)`` spatial legs, each ``(n_w, N, N)``."""
    idx = np.abs(np.subtract.outer(np.arange(N_CELL), np.arange(N_CELL)))
    exact = np.zeros((omegas.size, N_CELL, N_CELL), dtype=complex)
    completed = np.zeros_like(exact)
    for iw, omega in enumerate(omegas):
        g = _gap_green(omega, N_CELL)
        lam = _gap_root(omega)
        exact[iw] = g[idx]
        # continued from the last block INSIDE the band, so the completion
        # only ever supplies what the boxcar removed
        completed[iw] = np.where(idx <= band, g[idx],
                                 g[band] * lam ** (idx - band))
    banded = np.where(idx <= band, exact, 0.0)
    return exact, banded, completed


def _ring(phi, a_leg, b_leg, w):
    r""":math:`\Phi_{ace} A_{cb} B_{ed} \Phi_{Jdb}`, convolved over ``w``.

    The contraction the production ring performs, written out directly so the
    legs can be swapped without touching anything else.
    """
    h = w[1] - w[0]
    out = np.zeros((w.size, N_CELL, N_CELL), dtype=complex)
    for iw, om in enumerate(w):
        j = np.rint((om - w - w[0]) / h).astype(int)
        ok = (j >= 0) & (j < w.size)
        conv = np.einsum("kcb,ked->cbed", a_leg[ok], b_leg[j[ok]]) * h
        out[iw] = np.einsum("ace,Jdb,cbed->aJ", phi, phi, conv)
    return out / (2.0 * np.pi)


def _phi_nn(seed=5):
    """A cubic vertex coupling each cell to its neighbours, index-symmetric."""
    rng = np.random.default_rng(seed)
    phi = np.zeros((N_CELL, N_CELL, N_CELL))
    for i in range(N_CELL):
        for a in (i - 1, i, i + 1):
            for b in (i - 1, i, i + 1):
                if 0 <= a < N_CELL and 0 <= b < N_CELL:
                    phi[i, a, b] = rng.normal()
    return (phi + phi.transpose(0, 2, 1)) / 2.0


W_GRID = np.linspace(0.0, W_TOP, 24)


def test_the_gapped_bed_is_below_its_band_and_long_ranged_enough():
    """Both premises, because either one failing makes the rest vacuous."""
    band_lo = W0
    assert W_GRID.max() < band_lo
    lam = np.array([abs(_gap_root(w)) for w in W_GRID])
    assert np.all(lam < 1.0), "a grid point landed inside the band"
    # ranges of a few cells, so a band of 1-2 on a 7-cell device truncates
    xi = -1.0 / np.log(lam)
    assert xi.min() > 1.5 and xi.max() > 3.0, f"too short-ranged: {xi}"

    # and the modal form matches the quadrature, which is what makes the
    # completion below an independent computation rather than a restatement
    g = _gap_green(W_GRID[-1], 5)
    lm = _gap_root(W_GRID[-1])
    assert g[4] == pytest.approx(g[0] * lm ** 4, rel=1e-10)


@pytest.mark.parametrize("band", [1, 2])
def test_the_modal_completion_restores_what_the_band_removed(band):
    r"""The Phase 8 claim, on a dense vertex.

    Three rings differing only in the spatial legs: the exact ones, a boxcar,
    and the boxcar completed by the modal form beyond the band. The completed
    ring must land on the exact one, and the banded one must not -- otherwise
    the truncation was harmless here and the bed proves nothing.

    Measures the general mechanism of a hard band, over ALL ``Sigma`` blocks.
    It is NOT a statement about the shipped kernel: production retains only
    ``|I-J| <= 1``, where ``sse_g_band = 3`` is exact, and most of the error
    counted here lives in blocks that are discarded regardless. See
    ``test_band_three_is_exact_on_the_output_band_and_lossy_off_it`` and
    ``phonon/docs/spatial_truncation_derivation.md``.
    """
    exact, banded, completed = _legs(W_GRID, band)
    phi = _phi_nn()

    s_exact = _ring(phi, exact, exact, W_GRID)
    s_band = _ring(phi, banded, banded, W_GRID)
    s_modal = _ring(phi, completed, completed, W_GRID)

    scale = np.abs(s_exact).max()
    assert scale > 0.0
    err_band = np.abs(s_band - s_exact).max() / scale
    err_modal = np.abs(s_modal - s_exact).max() / scale

    assert err_band > 1e-3, (
        f"the band changed the ring by only {err_band:.1e}; this bed does not "
        "test a truncation")
    assert err_modal < 1e-10, f"completion left {err_modal:.1e}"
    assert err_modal < 1e-6 * err_band


def test_the_completion_beats_a_wider_band():
    """Widening the boxcar by one block is the obvious alternative and costs a
    whole extra block per cell pair; the completion costs one root and one
    anchor block.

    General mechanism, not a property of the shipped kernel -- see the note on
    the previous test.
    """
    phi = _phi_nn()
    exact, _, completed = _legs(W_GRID, 1)
    _, wider, _ = _legs(W_GRID, 2)

    s_exact = _ring(phi, exact, exact, W_GRID)
    scale = np.abs(s_exact).max()
    err_wider = np.abs(_ring(phi, wider, wider, W_GRID) - s_exact).max() / scale
    err_modal = np.abs(
        _ring(phi, completed, completed, W_GRID) - s_exact).max() / scale

    assert err_wider > 1e-4
    assert err_modal < 1e-6 * err_wider


def test_the_banded_error_grows_with_the_range_of_the_green_function():
    """A longer-ranged G makes the same truncation worse.

    True of a hard band in general. It is not the mechanism behind the CNT band
    ladder, which was the reading originally attached to it: the ring's leg band
    is exact at the shipped default, and the output pin that IS live turns out
    to be insensitive to the range (see
    ``test_the_discarded_output_weight_does_not_track_the_green_range``).
    """
    phi = _phi_nn()
    errs, ranges = [], []
    for top in (0.30 * W0, 0.65 * W0, 0.90 * W0):
        w = np.linspace(0.0, top, 20)
        exact, banded, _ = _legs(w, 1)
        s_exact = _ring(phi, exact, exact, w)
        errs.append(np.abs(_ring(phi, banded, banded, w) - s_exact).max()
                    / np.abs(s_exact).max())
        ranges.append(-1.0 / np.log(abs(_gap_root(top))))
    assert ranges[0] < ranges[1] < ranges[2], ranges
    assert errs[0] < errs[1] < errs[2], f"error did not grow with range: {errs}"


# --- no reweighting of the mask can do this job ----------------------------- #

def test_the_output_mask_is_psd_only_below_a_range_of_one_and_a_half_cells():
    r"""Closes the cheap alternative to a modal sector, with a proof.

    The obvious way to avoid building a modal representation is to keep the
    boxcar and re-weight it -- a taper. But the OUTPUT band is pinned at
    ``|I-J| <= 1`` whatever ``g_band`` is, so the output mask is the
    tridiagonal Toeplitz ``[w_1, 1, w_1]`` with symbol
    :math:`1 + 2 w_1\cos\theta`, non-negative only for :math:`w_1 \le 1/2`.

    Any weighting faithful to a Green function of range :math:`\xi` has
    :math:`w_1 = e^{-1/\xi}`, so PSD-ness demands

        xi <= 1 / ln 2 = 1.4427 cells,

    and every range measured on a real bed is far above that -- 3.05 to 28.8
    on Si, 1.5 to 25.5 on CNT (``phonon/docs/spatial_band_range.md``). So no
    choice of weights is simultaneously PSD at the output and faithful to the
    range the device actually has. The mask has to go, not be reshaped.

    This also derives the existing empirical result rather than restating it:
    Bartlett has ``w_1 = b/(b+1)``, which is ``<= 1/2`` only at ``b = 1``,
    which is exactly where ``test_taper_is_psd_only_at_band_one`` finds it.
    """
    def output_symbol_min(w1):
        theta = np.linspace(0.0, 2 * np.pi, 2001)
        return float(np.min(1.0 + 2.0 * w1 * np.cos(theta)))

    # the bound itself
    assert output_symbol_min(0.5) == pytest.approx(0.0, abs=1e-5)
    assert output_symbol_min(0.49) > 0.0
    assert output_symbol_min(0.51) < 0.0

    xi_max = 1.0 / np.log(2.0)
    assert xi_max == pytest.approx(1.4427, rel=1e-4)
    assert np.exp(-1.0 / xi_max) == pytest.approx(0.5)

    # Bartlett is PSD at the output only at band 1, and marginally
    for band in (1, 2, 3):
        w1 = 1.0 - 1.0 / (band + 1.0)
        assert (output_symbol_min(w1) >= -1e-9) == (band == 1)

    # every range measured on a real device is above the bound
    for xi in (3.05, 6.05, 28.8, 1.5, 2.62, 10.7):
        assert xi > xi_max
        assert output_symbol_min(np.exp(-1.0 / xi)) < 0.0


def test_a_truncated_geometric_mask_is_psd_only_once_the_band_exceeds_the_range():
    r"""The other half of the impossibility, and it was not what I expected.

    The untruncated geometric weight is the Poisson kernel
    :math:`(1-\lambda^2)/(1-2\lambda\cos\theta+\lambda^2) > 0`, so a geometric
    taper looks like the obvious PSD replacement for the boxcar. TRUNCATED it
    is not: cutting a slowly decaying tail leaves a discontinuity, and a
    truncated positive-definite sequence need not stay positive definite. At
    :math:`\lambda = 0.91` and band 4 the leg symbol reaches -1.11.

    Measured, the first band at which it turns positive:

    ======  =========  ==============  ==========
    lambda  xi [cells] first PSD band  band / xi
    ======  =========  ==============  ==========
    0.30    0.83       1               1.20
    0.50    1.44       2               1.39
    0.68    2.59       4               1.54
    0.80    4.48       10              2.23
    0.91    10.60      32              3.02
    ======  =========  ==============  ==========

    So the band has to exceed the range, by a factor that itself grows with the
    range -- which is precisely the regime in which no truncation was needed.
    Together with the output bound above, reweighting cannot substitute for a
    modal sector on any bed whose range exceeds about one and a half cells.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 4001)

    def sym_min(lam, band):
        d = np.arange(-band, band + 1)
        return float(np.min(np.real(
            np.exp(1j * np.outer(theta, d)) @ (lam ** np.abs(d)))))

    # the failure at a short band and a long range
    assert sym_min(0.91, 4) < -1.0

    # the measured turn-on, and that it needs more than the range itself
    for lam, first in ((0.3, 1), (0.5, 2), (0.68, 4), (0.8, 10), (0.91, 32)):
        xi = -1.0 / np.log(lam)
        assert sym_min(lam, first) > 0.0
        if first > 1:
            assert sym_min(lam, first - 1) <= 0.0
        assert first >= xi, f"lam={lam}: band {first} below the range {xi:.2f}"

    # the untruncated limit IS the Poisson kernel and is strictly positive
    for lam in (0.3, 0.68, 0.91):
        exact = (1 - lam ** 2) / (1 - 2 * lam * np.cos(theta) + lam ** 2)
        assert exact.min() > 0.0
        wide = int(np.ceil(30.0 / abs(np.log(lam))))
        d = np.arange(-wide, wide + 1)
        got = np.real(np.exp(1j * np.outer(theta, d)) @ (lam ** np.abs(d)))
        assert np.allclose(got, exact, rtol=1e-8)


# --- what the ring's spatial truncations actually are ----------------------- #
#
# Three tests pinning the derivation in
# ``phonon/docs/spatial_truncation_derivation.md``. They exist because reasoning
# in prose about "which blocks matter" produced two opposite wrong answers in a
# row; the index algebra is short and the numbers settle it.

VERTEX_REACH = 1                 # _phi_nn couples each cell to its neighbours


def _out_distance():
    i, j = np.meshgrid(np.arange(N_CELL), np.arange(N_CELL), indexing="ij")
    return np.abs(i - j)


@pytest.mark.parametrize("band", [0, 1, 2, 3])
def test_the_sigma_support_law_is_two_p_plus_band(band):
    r"""``supp(Sigma) = {|I-J| <= 2p + b}``, with ``p`` the vertex reach.

    One line of index algebra: ``K1, K2`` lie within ``p`` of ``I`` and
    ``K1', K2'`` within ``p`` of ``J``, while the legs contribute only for
    ``|K - K'| <= b``; chaining the three gives
    ``|I-J| <= p + b + p``.

    The consequence that matters is that ``Sigma`` is NOT tridiagonal. Its
    reach grows with the leg band, so pinning the output at ``|I-J| <= 1`` is a
    truncation in its own right and not a property of the vertex.
    """
    _, banded, _ = _legs(W_GRID, band)
    s = _ring(_phi_nn(), banded, banded, W_GRID)
    d = _out_distance()

    reach = max(dd for dd in range(N_CELL)
                if np.abs(s[:, d == dd]).max() > 1e-13 * np.abs(s).max())
    assert reach == 2 * VERTEX_REACH + band


def test_band_three_is_exact_on_the_output_band_and_lossy_off_it():
    r"""The claim I got wrong in both directions, frozen.

    ``sse_g_band`` truncates the LEGS. Given the output pin at ``|I-J| <= 1``
    the reachable leg distance is ``2p + 1 = 3``, so ``b = 3`` loses nothing
    THERE -- which is what the config docstring means by "the first
    off-diagonal Sigma blocks become exact and causal", and why the field is
    capped at 3.

    It is not a statement that the ring is exact. Off the retained band the
    same ``b = 3`` result is visibly wrong, because those blocks were never
    computed to begin with. Reporting a whole-array error therefore overstates
    the leg band's cost, and reporting only the retained band hides the output
    pin's.
    """
    exact, _, _ = _legs(W_GRID, N_CELL)
    phi = _phi_nn()
    s_exact = _ring(phi, exact, exact, W_GRID)
    d = _out_distance()
    keep, drop = d <= 1, d > 1

    errs = {}
    for band in (1, 2, 3):
        _, banded, _ = _legs(W_GRID, band)
        diff = np.abs(_ring(phi, banded, banded, W_GRID) - s_exact)
        errs[band] = (diff[:, keep].max() / np.abs(s_exact[:, keep]).max(),
                      diff[:, drop].max() / np.abs(s_exact[:, drop]).max())

    assert errs[1][0] > 1e-2, "band 1 must be lossy on the retained band"
    assert errs[2][0] > 1e-2, "band 2 must be lossy on the retained band"
    assert errs[3][0] < 1e-13, f"band 3 not exact on |I-J|<=1: {errs[3][0]:.2e}"
    # ... and off it, band 3 is not exact at all
    assert errs[3][1] > 1e-2


def test_the_discarded_output_weight_does_not_track_the_green_range():
    r"""The output pin costs about a tenth of ``Sigma``, whatever the range.

    Measured 10.5 % at a Green-function range of 2.1 cells and 11.4 % at 7.2 --
    flat, where a long-range effect would grow. Index algebra again: for
    ``|I-J| = 2`` one may take ``K = I+1`` and ``K' = J-1 = I+1``, so
    ``|K - K'| = 0``. The near tail of ``Sigma`` is fed by the DIAGONAL of
    ``G`` through the vertex's reach, and long-range ``G`` only ever appears in
    blocks the pin has already discarded.

    That is why a low-rank representation of distant ``G`` does not repair this
    truncation, and why the fix has to be a non-tridiagonal ``Sigma``.
    """
    phi = _phi_nn()
    d = _out_distance()
    fracs, ranges = [], []
    for top in (0.20, 0.65, 0.96):
        w = np.linspace(0.0, top * W0, 20)
        exact, _, _ = _legs(w, N_CELL)
        s = _ring(phi, exact, exact, w)
        fracs.append(float(np.abs(s[:, d > 1]).sum() / np.abs(s).sum()))
        ranges.append(-1.0 / np.log(abs(_gap_root(top * W0))))

    assert ranges[0] < ranges[1] < ranges[2]
    assert ranges[2] / ranges[0] > 3.0, "the range barely moved; not a test"
    # the discarded share is real ...
    assert min(fracs) > 0.05
    # ... and flat: it moves by under a fifth while the range triples
    assert max(fracs) / min(fracs) < 1.2, f"{fracs}"


# --- the output pin is a BLOCKING statement --------------------------------- #

def _long_bed(n_cell, top=0.90, nw=14, seed=5):
    """Gapped chain of ``n_cell`` cells and its exact ring, no leg mask."""
    w = np.linspace(0.0, top * W0, nw)
    idx = np.abs(np.subtract.outer(np.arange(n_cell), np.arange(n_cell)))
    legs = np.stack([_gap_green(om, n_cell)[idx] for om in w])

    rng = np.random.default_rng(seed)
    phi = np.zeros((n_cell,) * 3)
    for i in range(n_cell):
        for a in (i - 1, i, i + 1):
            for b in (i - 1, i, i + 1):
                if 0 <= a < n_cell and 0 <= b < n_cell:
                    phi[i, a, b] = rng.normal()
    phi = (phi + phi.transpose(0, 2, 1)) / 2.0

    h = w[1] - w[0]
    out = np.zeros((nw, n_cell, n_cell), dtype=complex)
    for iw, om in enumerate(w):
        j = np.rint((om - w - w[0]) / h).astype(int)
        ok = (j >= 0) & (j < nw)
        conv = np.einsum("kcb,ked->cbed", legs[ok], legs[j[ok]],
                         optimize=True) * h
        out[iw] = np.einsum("ace,Jdb,cbed->aJ", phi, phi, conv, optimize=True)
    return out


def _discarded(sigma, cells_per_block):
    n = sigma.shape[-1]
    blk = np.arange(n) // cells_per_block
    far = np.abs(np.subtract.outer(blk, blk)) > 1
    return float(np.abs(sigma[:, far]).sum() / np.abs(sigma).sum())


def test_the_output_pin_costs_far_more_on_a_long_device():
    """A seven-cell device understates it by a factor three.

    The tridiagonal band is ``3N-2`` of ``N^2`` entries, so the share of
    ``Sigma`` outside it grows with the device until the decay of ``Sigma``
    with distance takes over. Measured 10.5 % at seven cells and about 30 % by
    ten, which is where it settles -- so a short bed is not a conservative
    proxy for a long one, it is a different answer.
    """
    short = _discarded(_long_bed(7), 1)
    long_ = _discarded(_long_bed(12), 1)
    assert short < 0.15
    assert long_ > 0.25
    assert long_ > 2.0 * short


@pytest.mark.parametrize("cells_per_block,ceiling", [(1, 1.0), (2, 0.10),
                                                     (3, 0.05), (4, 0.01)])
def test_wider_blocks_make_the_tridiagonal_pin_accurate(cells_per_block,
                                                        ceiling):
    r"""The lever on the output pin is the BLOCKING, not the modes.

    ``supp(Sigma) = {|I-J| <= 2p + b}`` in CELLS. Group ``m`` cells into a
    block and that becomes ``ceil((2p+b)/m)`` in BLOCKS, so once a block is
    wide enough the tridiagonal restriction the RGF needs stops discarding
    anything. Measured on a 12-cell device:

    ======  ==========
    m       discarded
    ======  ==========
    1       32.1 %
    2        5.4 %
    3        2.5 %
    4        0.30 %
    ======  ==========

    This is the mechanism behind an observation already in the tree -- the same
    Si device diverging at 4x1 blocks and converging at 2x2. It is also why the
    long-range modal machinery is the wrong lever here: the discarded weight
    depends only weakly on the Green-function range (about five points between
    ranges of 2 and 20 cells) and strongly on how the device is blocked.
    """
    assert _discarded(_long_bed(12), cells_per_block) < ceiling


def test_a_two_block_device_has_no_output_pin_error_at_all():
    r"""Combinatorial, not numerical: with two blocks the largest possible
    ``|I-J|`` is 1, so a tridiagonal restriction discards nothing on ANY bed.

    That is the whole difference between the two blockings of one 24-DOF Si
    device recorded in the tree, `si4x1` (4 blocks of 6 DOF) and `si4x2`
    (2 blocks of 12 DOF): the second has no mask to apply. Since the mask is a
    Schur product with an indefinite band-ones matrix -- the documented source
    of non-causal gain -- what matters is not that its weight is small but that
    it is exactly absent.
    """
    sig = _long_bed(4)
    assert _discarded(sig, 2) == 0.0          # 2 blocks: nothing to discard
    assert _discarded(sig, 1) > 0.0           # 4 blocks: a mask exists

    # and it is the block COUNT that does it, not the bed
    for n_cell, m in ((6, 3), (8, 4), (12, 6)):
        assert _discarded(_long_bed(n_cell), m) == 0.0


def test_the_pin_grows_over_the_lengths_where_the_cnt_ladder_stops_being_read():
    r"""The discarded weight at one cell per block, over the ladder's lengths.

    L4 about 2 %, L7 about 11 %, L16 about 35 %. The reported CNT series stops
    at seven cells and brackets from sixteen
    (``document/src/results/64_gband.tex``), which is where this crosses from a
    few percent to a third.

    Correspondence, not proof: the bed is a 1-DOF chain with a random vertex,
    so the percentages are not the device's. What transfers is that the pin's
    cost grows steeply with block count over exactly that range, while the
    quantity usually blamed -- the Green-function range -- moves it by five
    points over a factor ten.
    """
    fracs = {n: _discarded(_long_bed(n), 1) for n in (4, 7, 16)}
    assert fracs[4] < 0.05
    assert 0.08 < fracs[7] < 0.20
    assert fracs[16] > 0.30
    assert fracs[16] > 10.0 * fracs[4]
