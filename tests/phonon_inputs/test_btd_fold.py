"""The folded BT export must be exact at Gamma and the zone boundary.

The transport .mat holds only (H00, H01, H01^dag). When the fit mesh
resolves couplings beyond one cell (n_qz >= 4), the old export dropped
the n >= 2 Fourier coefficients, breaking the acoustic sum rule of the
emitted matrix (the d5a/d11a corruption: Gamma translations at
+-0.15-0.65 THz, partly imaginary, instead of exact zeros).

Test system: a monatomic chain with FIRST- and SECOND-neighbour springs
(k2 != 0 makes the n=2 coefficient nonzero), fitted exactly on a
[1,1,4] supercell. Analytic dispersion:

    w^2(q) = (2 k1 (1 - cos q) + 2 k2 (1 - cos 2q)) / m
"""
import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")
from phonopy import Phonopy  # noqa: E402
from phonopy.structure.atoms import PhonopyAtoms  # noqa: E402

from phonon_inputs.convention import (  # noqa: E402
    get_btd_blocks,
    get_btd_blocks_folded,
)

K1, K2 = 10.0, 1.5  # eV/A^2; k2 != 0 -> nonzero n=2 coefficient
A = 3.0             # chain period, Angstrom


def _chain_phonon() -> Phonopy:
    """1-atom chain along z, supercell [1,1,4], 1st+2nd-neighbour FC2."""
    unit = PhonopyAtoms(
        symbols=["Si"],
        cell=np.diag([15.0, 15.0, A]),
        scaled_positions=[[0.0, 0.0, 0.0]],
    )
    phonon = Phonopy(unit, supercell_matrix=np.diag([1, 1, 4]),
                     primitive_matrix=np.eye(3))
    n = 4  # supercell atoms (all periodic images along z)
    fc2 = np.zeros((n, n, 3, 3))
    # Longitudinal springs along z with minimum-image neighbour distance.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = (j - i) % n
            d = d - n if d > n // 2 else d
            k = {1: K1, -1: K1, 2: K2, -2: K2}.get(d, 0.0)
            fc2[i, j, 2, 2] += -k
    # On a 4-supercell the +-2 images coincide: the loop above visits the
    # d=2 pair once per (i,j) but both +2 and -2 map to the same j, so
    # double the second-neighbour entry to count both bonds.
    for i in range(n):
        j = (i + 2) % n
        fc2[i, j, 2, 2] += -K2
    # Acoustic sum rule: on-site balances the row.
    for i in range(n):
        fc2[i, i] = -np.sum(fc2[i], axis=0) + fc2[i, i]
    phonon.force_constants = fc2
    return phonon


def _w2_analytic(q_frac: float) -> float:
    """Analytic w^2(q) in the same units the conversion factor maps to."""
    th = 2.0 * np.pi * q_frac
    m = 28.0855  # Si amu
    return (2 * K1 * (1 - np.cos(th)) + 2 * K2 * (1 - np.cos(2 * th))) / m


@pytest.fixture(scope="module")
def chain():
    return _chain_phonon()


def _dk(h00, h01, q_frac):
    ph = np.exp(2j * np.pi * q_frac)
    return h00 + h01 * ph + h01.conj().T * np.conj(ph)


def test_folded_exact_at_gamma_and_boundary(chain):
    h00, h01, rep = get_btd_blocks_folded(
        chain, (0.0, 0.0), transport_direction="z", n_qz=4,
        conversion_factor=1.0,
    )
    assert 2 in rep["fold_norms"] and rep["fold_norms"][2] > 0.01
    for q in (0.0, 0.5):  # Gamma AND zone boundary: fold is exact
        w2 = np.linalg.eigvalsh(_dk(h00, h01, q))
        # longitudinal branch = the largest eigenvalue (transverse are 0)
        assert np.isclose(w2[-1], _w2_analytic(q), rtol=1e-10, atol=1e-12), (
            f"q={q}: {w2[-1]} != {_w2_analytic(q)}"
        )
        # acoustic sum rule at Gamma: all eigenvalues >= 0, zero at q=0
        if q == 0.0:
            assert abs(w2).max() < 1e-12


def test_unfolded_breaks_gamma(chain):
    """The OLD export (dropped n=2): Gamma is off by the k2 weight."""
    h00, h01 = get_btd_blocks(
        chain, (0.0, 0.0), transport_direction="z", n_qz=4,
        conversion_factor=1.0,
    )
    w2 = np.linalg.eigvalsh(_dk(h00, h01, 0.0))
    assert abs(w2).max() > 1e-3, (
        "expected the truncated export to violate the acoustic sum rule"
    )


def test_folded_midzone_error_bounded(chain):
    h00, h01, rep = get_btd_blocks_folded(
        chain, (0.0, 0.0), transport_direction="z", n_qz=4,
        conversion_factor=1.0,
    )
    for q in (0.1, 0.25, 0.4):
        w2 = np.linalg.eigvalsh(_dk(h00, h01, q))
        err = abs(w2[-1] - _w2_analytic(q))
        assert err <= rep["midzone_bound"] + 1e-12, (
            f"q={q}: error {err} exceeds bound {rep['midzone_bound']}"
        )
