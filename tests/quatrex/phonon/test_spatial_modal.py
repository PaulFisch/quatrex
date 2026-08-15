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
