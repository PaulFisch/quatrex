# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The sign and linewidth conventions the pole sector rests on.

Every gate in the sector is a function of the pole's HALF width, and the
tree carries both conventions in prose. The exact line-weight error is
:math:`2/(e^{2\pi\gamma/h}-1)` -- exponential in its argument -- so feeding it a
full width instead of a half width does not mis-rank a pole slightly, it
mis-ranks it by the square. These tests pin the convention against a physical
requirement (a passive mode has positive spectral weight, and its Lorentzian is
half its peak one half-width off resonance) rather than against a docstring.

Audit checklist T0.
"""
import numpy as np
import pytest

from quatrex.phonon.pole_nevp import PoleSolution


def _sol(z):
    """A PoleSolution carrying nothing but a location."""
    return PoleSolution(z=complex(z), r=np.ones(1), l=np.ones(1),
                        eps_nep=0.0, kappa=1.0, converged=True, iterations=0)


def _spectral(omega, z, residue=1.0):
    r""":math:`A(\omega) = -2\,\Im\,[R/(\omega - z)]`, the spectral function of
    one simple pole of the retarded function."""
    return -2.0 * np.imag(residue / (omega - z))


@pytest.mark.parametrize("omega_pole,gamma", [(3.0, 0.25), (11.7, 1e-3),
                                              (0.4, 0.4)])
def test_a_passive_pole_carries_positive_spectral_weight(omega_pole, gamma):
    r"""The convention is :math:`e^{-i\omega t}`, so a decaying mode sits at
    :math:`z = \Omega - i\gamma` with :math:`\gamma > 0`."""
    z = complex(omega_pole, -gamma)
    sol = _sol(z)
    assert sol.is_passive
    assert sol.gamma_hwhm > 0.0

    w = np.linspace(omega_pole - 20 * gamma, omega_pole + 20 * gamma, 401)
    assert np.all(_spectral(w, z) > 0.0)

    # The mirrored pole is the counterfactual: same mode, wrong half plane.
    assert not _sol(np.conj(z)).is_passive
    assert np.all(_spectral(w, np.conj(z)) < 0.0)


def test_gamma_is_the_half_width_and_Gamma_is_the_full_width():
    """Half maximum is reached one ``gamma_hwhm`` off resonance, so the full
    width at half maximum is twice it. This is the factor of two that a
    resolution gate must not absorb."""
    omega_pole, gamma = 5.0, 0.3
    sol = _sol(complex(omega_pole, -gamma))

    assert sol.omega_pole == pytest.approx(omega_pole)
    assert sol.gamma_hwhm == pytest.approx(gamma)
    assert sol.Gamma_fwhm == pytest.approx(2.0 * gamma)

    peak = _spectral(omega_pole, sol.z)
    for side in (-1.0, +1.0):
        half = _spectral(omega_pole + side * sol.gamma_hwhm, sol.z)
        assert half == pytest.approx(0.5 * peak, rel=1e-12)

    # ... and the full width spans the two half-maximum points.
    lo = omega_pole - 0.5 * sol.Gamma_fwhm
    hi = omega_pole + 0.5 * sol.Gamma_fwhm
    assert hi - lo == pytest.approx(sol.Gamma_fwhm)
    assert _spectral(lo, sol.z) == pytest.approx(0.5 * peak, rel=1e-12)


def test_the_named_widths_agree_with_the_raw_imaginary_part():
    """``gamma_hwhm`` is exactly ``-Im z``: the names are a label on the
    existing convention, not a new one."""
    rng = np.random.default_rng(0)
    for _ in range(16):
        z = complex(rng.uniform(0.1, 20.0), -rng.uniform(1e-6, 2.0))
        sol = _sol(z)
        assert sol.gamma_hwhm == -z.imag
        assert sol.Gamma_fwhm == -2.0 * z.imag
        assert sol.omega_pole == z.real


def test_the_resolution_gate_consumes_the_half_width():
    r"""``leg_weight_error`` is :math:`2/(e^{2\pi h^{-1}\gamma \cdot
    2\pi/...}-1)` in the variable :math:`r = h/\gamma`, with ``gamma`` the
    HALF width."""
    from quatrex.core.config import PoleSectorConfig
    from quatrex.phonon.pole_sector import PoleSector

    h = 0.125
    freqs = np.arange(0.0, 20.0 + h, h)
    sec = PoleSector(PoleSectorConfig(), freqs)
    assert sec.h == pytest.approx(h)

    for eps in (1e-2, 5e-2, 1e-1):
        r_star = 2.0 * np.pi / np.log1p(2.0 / eps)
        gamma = h / r_star
        assert sec.leg_weight_error(gamma) == pytest.approx(eps, rel=1e-12)

        # The same line described by its FULL width is off by the factor of two.
        assert sec.leg_weight_error(2.0 * gamma) != pytest.approx(eps, rel=1e-3)


def test_a_non_positive_width_is_refused_rather_than_ranked():
    """A root on or above the real axis has no line-weight to speak of, and the
    gate returns ``inf`` so it can never read as resolved."""
    from quatrex.core.config import PoleSectorConfig
    from quatrex.phonon.pole_sector import PoleSector

    sec = PoleSector(PoleSectorConfig(), np.arange(0.0, 10.0, 0.1))
    assert np.isinf(sec.leg_weight_error(0.0))
    assert np.isinf(sec.leg_weight_error(-1.0))


# --- finite-support line weight, audit Eq. (14) ----------------------------- #

def _sector(h=0.125, top=20.0):
    from quatrex.core.config import PoleSectorConfig
    from quatrex.phonon.pole_sector import PoleSector
    return PoleSector(PoleSectorConfig(), np.arange(0.0, top + h, h))


@pytest.mark.parametrize("gamma", [0.166, 0.0849, 0.0252])
def test_finite_support_matches_the_infinite_formula_where_the_line_is_unresolved(
        gamma):
    """Where the pole's own discretisation error dominates, the finite
    statement reduces to the infinite one. Checked with the line ON a node,
    which is where the infinite formula attains its worst case."""
    sec = _sector()
    centre = 10.0                       # 80 * h exactly, i.e. on a node
    assert sec.leg_weight_error_finite(gamma, centre) == pytest.approx(
        sec.leg_weight_error(gamma), rel=2e-2)


def test_a_resolved_line_floors_on_the_endpoint_term_not_on_the_pole():
    r"""The two formulas stop agreeing once the grid carries the line."""
    sec = _sector()
    inf_, fin = sec.leg_weight_error(0.4), sec.leg_weight_error_finite(0.4, 10.0)
    assert inf_ < 1e-8
    assert fin < 1e-5
    assert fin > 10.0 * inf_


def test_widening_the_support_drives_the_two_together():
    """The gap between them IS the truncated tail, so it shrinks as the grid
    covers more of the line."""
    gamma, centre = 0.0849, 10.0
    gaps = []
    for top in (20.0, 40.0, 80.0):
        sec = _sector(top=top)
        gaps.append(abs(sec.leg_weight_error_finite(gamma, centre)
                        - sec.leg_weight_error(gamma)))
    assert gaps[0] > gaps[1] > gaps[2]


def test_a_broad_line_at_the_edge_loses_weight_the_interior_test_cannot_see():
    """``leg_weight_error`` calls a broad line perfectly carried wherever it
    sits. At the support boundary half its Lorentzian is off the end of the
    grid, and only the finite statement reports it."""
    sec = _sector()
    gamma = 0.4
    interior = sec.leg_weight_error(gamma)
    edge = sec.leg_weight_error_finite(gamma, 19.9)
    assert interior < 1e-6                      # "nothing to fix here"
    assert edge > 1e-4                          # ... but there is
    assert edge > 100.0 * interior


def test_a_narrow_line_at_the_edge_is_relatively_better_carried():
    r"""The inversion the finite statement exists to expose."""
    sec = _sector()
    gamma = 0.0252
    assert sec.leg_weight_error(gamma) > 0.5            # badly under-resolved
    assert sec.leg_weight_error_finite(gamma, 19.9) < 0.1


def test_finite_support_refuses_a_non_positive_width():
    sec = _sector()
    assert np.isinf(sec.leg_weight_error_finite(0.0, 10.0))
    assert np.isinf(sec.leg_weight_error_finite(-1.0, 10.0))


# --- the census must not be written by every rank --------------------------- #

def test_the_census_prints_on_one_rank_only(monkeypatch, capsys):
    """Four ranks writing the same report to one stdout interleave mid-word."""
    import quatrex.phonon.pole_sector as ps

    rows = [{"z": 3 - 0.01j, "gamma": 0.01, "separation": 1.0, "chi": 0.01,
             "q_omega": 0.04, "leg_weight_error": 12.0, "E_finite": 0.9,
             "gamma_sens_anh": 0.009, "passive": True, "eps_z": 1e-12,
             "eps_nep": 1e-12, "eps_left": 1e-12, "kappa": 1.0,
             "iterations": 3, "trust_radius": 1.0, "refused": None}]

    monkeypatch.setattr(ps, "_report_rank", lambda: 0)
    ps.PoleSector._report_census(rows)
    assert "pole census" in capsys.readouterr().out

    for rank in (1, 2, 3):
        monkeypatch.setattr(ps, "_report_rank", lambda r=rank: r)
        ps.PoleSector._report_census(rows)
        assert capsys.readouterr().out == "", f"rank {rank} wrote to stdout"


def test_an_empty_census_is_also_silent_off_rank_zero(monkeypatch, capsys):
    """The no-candidates line is a print too, and was outside the guard in the
    first version of this fix."""
    import quatrex.phonon.pole_sector as ps

    monkeypatch.setattr(ps, "_report_rank", lambda: 2)
    ps.PoleSector._report_census([])
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(ps, "_report_rank", lambda: 0)
    ps.PoleSector._report_census([])
    assert "no candidates" in capsys.readouterr().out


# --- the shipped resolution gate is the exact one --------------------------- #

def test_the_exact_line_weight_gate_is_the_default():
    """It defaulted to 0 -- the legacy ratio -- until 2026-08-15."""
    from quatrex.core.config import PoleSectorConfig

    cfg = PoleSectorConfig()
    assert cfg.leg_weight_tol > 0.0, "the legacy ratio gate is shipping again"
    assert cfg.leg_weight_tol == pytest.approx(0.05)
    # 0 stays reachable, so pre-2026-08-15 runs still reproduce.
    assert PoleSectorConfig(leg_weight_tol=0.0).leg_weight_tol == 0.0


def test_the_default_tolerance_splits_the_population_where_the_census_did():
    r"""0.05 promotes the tail and leaves the median mode on the grid."""
    from quatrex.core.config import PoleSectorConfig
    from quatrex.phonon.pole_sector import PoleSector

    tol = PoleSectorConfig().leg_weight_tol
    h = 0.25                                    # the census grid, wmax 35 / 140
    sec = PoleSector(PoleSectorConfig(), np.arange(0.0, 35.0 + h, h))
    assert sec.h == pytest.approx(h)

    # Invert E_leg^max to the gamma each census figure corresponds to.
    def gamma_of(err):
        return h / (2.0 * np.pi / np.log1p(2.0 / err))

    assert sec.leg_weight_error(gamma_of(0.0314)) <= tol      # median: refused
    assert sec.leg_weight_error(gamma_of(3.56)) > tol         # tail: promoted


def test_hysteresis_still_derives_from_the_new_default():
    """``leg_weight_tol_out = 0`` means tol/3, and the out-tolerance must be
    SMALLER: this gate refuses a pole the grid already resolves, so staying in
    the sector is the easier condition, not the harder one."""
    from quatrex.core.config import PoleSectorConfig

    cfg = PoleSectorConfig()
    assert cfg.leg_weight_tol_out == 0.0
    effective_out = cfg.leg_weight_tol / 3.0
    assert effective_out < cfg.leg_weight_tol


# --- lead band edges: the branch-point gate ---------------------------------- #

def test_lead_band_edges_match_the_monatomic_chain_closed_form():
    r"""``omega(k)^2 = 4 k_s sin^2(k/2)``, so the band is ``[0, 2 sqrt(k_s)]``.

    The one lead whose dispersion is elementary, used so the sampler is pinned
    against algebra rather than against itself.
    """
    from quatrex.phonon.pole_sector import lead_band_edges

    for k_s in (1.0, 4.0, 17.5):
        edges = lead_band_edges(np.array([[2.0 * k_s]]),
                                np.array([[-k_s]]), np.array([[-k_s]]))
        assert edges == pytest.approx([0.0, 2.0 * np.sqrt(k_s)], abs=1e-9)


def test_lead_band_edges_finds_both_branches_of_a_diatomic_chain():
    r"""Two masses give an acoustic and an optical branch with a gap between
    them, so a correct sampler returns FOUR edges, not two."""
    from quatrex.phonon.pole_sector import lead_band_edges

    k_s, m1, m2 = 1.0, 1.0, 3.0
    d00 = np.array([[2 * k_s / m1, -k_s / np.sqrt(m1 * m2)],
                    [-k_s / np.sqrt(m1 * m2), 2 * k_s / m2]])
    d01 = np.array([[0.0, -k_s / np.sqrt(m1 * m2)], [0.0, 0.0]])
    edges = lead_band_edges(d00, d01, d01.T.conj())

    assert edges.size == 4, f"expected two branches, got edges {edges}"
    acoustic_top, optical_bottom = edges[1], edges[2]
    assert acoustic_top < optical_bottom, "no gap: the branches were merged"
    # Closed form: the zone-boundary pair is sqrt(2k/m2), sqrt(2k/m1).
    assert acoustic_top == pytest.approx(np.sqrt(2 * k_s / m2), rel=1e-6)
    assert optical_bottom == pytest.approx(np.sqrt(2 * k_s / m1), rel=1e-6)


def test_lead_band_edges_refuses_a_stack_instead_of_a_block():
    """A leading frequency axis reaching this would silently make a 'band edge'
    per frequency. Refuse rather than average."""
    from quatrex.phonon.pole_sector import lead_band_edges

    d = np.zeros((7, 2, 2))
    with pytest.raises(ValueError, match="one square block"):
        lead_band_edges(d, d, d)


def test_the_band_edge_gate_is_off_by_default_and_bites_when_supplied():
    """It was inert in production for the whole campaign: neither solver call
    site passed ``band_edges``, so ``edge_factor`` never fired and the
    proposal's "do not force band-edge continua into isolated poles" was
    unenforced. Default stays off because switching it on changes which poles
    are promoted."""
    from quatrex.core.config import PoleSectorConfig
    from quatrex.phonon.pole_sector import PoleSector

    assert PoleSectorConfig().band_edges == "none"

    freqs = np.arange(0.0, 20.0, 0.25)
    gamma = 0.05
    sol = _sol(complex(9.0, -gamma))
    sol.dz_est = 0.0 + 0.0j

    blind = PoleSector(PoleSectorConfig(enabled=True), freqs)
    # An edge one half-width away: inside edge_factor * gamma of the pole.
    seeing = PoleSector(PoleSectorConfig(enabled=True), freqs,
                        band_edges=np.array([9.0 + gamma]))

    assert "band edge" not in (blind.screen(sol, False, 5.0) or "")
    why = seeing.screen(sol, False, 5.0)
    assert why is not None and "band edge" in why, why


def test_the_solver_derivation_is_cached_off_switchable_and_squeezes():
    """``PhononSolver._band_edges_for`` without standing up a solver.

    The default is ``"none"``, so nothing in the suite would otherwise execute
    this path -- which is exactly how it stayed inert through the campaign in
    the first place. Bound to a stand-in carrying only the attributes it uses.
    """
    from quatrex.core.config import PoleSectorConfig
    from quatrex.phonon.solver import PhononSolver

    k_s = 4.0
    d00 = np.array([[2.0 * k_s]])
    d01 = np.array([[-k_s]])

    class _Stub:
        block_sections = 1
        dynamical_matrix = object()

        def __init__(self, mode, leading=()):
            self._pole_cfg = PoleSectorConfig(enabled=True, band_edges=mode)
            self.calls = 0
            self._leading = leading

        def _pole_blocks(self, matrix, index_slice=()):
            self.calls += 1
            def _b(a):
                return a.reshape(self._leading + a.shape)
            return {(0, 0): _b(d00), (0, 1): _b(d01),
                    (1, 0): _b(d01.T), (1, 1): _b(d00)}

    off = _Stub("none")
    assert PhononSolver._band_edges_for(off) is None
    assert off.calls == 0, "the off switch must not do the work"

    on = _Stub("lead")
    edges = PhononSolver._band_edges_for(on)
    assert edges == pytest.approx([0.0, 2.0 * np.sqrt(k_s)], abs=1e-9)

    # Cached per q: D does not move during the SCBA.
    PhononSolver._band_edges_for(on)
    PhononSolver._band_edges_for(on)
    assert on.calls == 1, f"recomputed {on.calls} times; should be cached"
    # ... and a different q is a different entry.
    PhononSolver._band_edges_for(on, index_slice=(3, 1))
    assert on.calls == 2

    # A leading singleton stack must be squeezed, not refused.
    stacked = _Stub("lead", leading=(1,))
    assert PhononSolver._band_edges_for(stacked) == pytest.approx(
        [0.0, 2.0 * np.sqrt(k_s)], abs=1e-9)


def test_a_single_block_device_has_no_periodic_layer_to_homogenise():
    from quatrex.core.config import PoleSectorConfig
    from quatrex.phonon.solver import PhononSolver

    class _Stub:
        block_sections = 1
        dynamical_matrix = object()
        _pole_cfg = PoleSectorConfig(enabled=True, band_edges="lead")

        def _pole_blocks(self, matrix, index_slice=()):
            return {(0, 0): np.array([[1.0]])}

    assert PhononSolver._band_edges_for(_Stub()) is None
