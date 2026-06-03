"""Static, real, Hermitian anharmonic self-energies (loop + tadpole).

These are the two ``omega``-independent diagrams that renormalise the
phonon *frequencies* (they shift, they do not broaden -- only the cubic
bubble :mod:`phonon.solver.se_finite` carries an imaginary part / linewidth):

  * **quartic loop** ``Sigma_L = 1/2 Phi4 : <uu>``  (Hartree contraction of
    FC4 with the displacement *variance*; needs FC4),
  * **cubic tadpole** ``Sigma_T = Phi3 : <u>`` with
    ``<u> = -Phi_eff^{-1} (1/2 Phi3 : <uu>)`` (FC3 contracted with the static
    mean displacement; needs only FC3, and is ~0 for a consistently relaxed
    symmetric crystal).

Both are added to the dynamical matrix, ``Phi_eff = Phi + Sigma_L + Sigma_T``,
re-Hermitized and ASR-projected, inside the single SCBA loop. See the brief
(Tadano & Tsuneyuki 2018; Paulatto et al. 2015) and the project plan.

Units / conventions (pinned once -- this is where the bugs live)
----------------------------------------------------------------
The solver works in the dense reference's mass-weighted ``THz^2`` space
(``phonon.solver.toy_models`` / ``grids``): ``G^R = [(omega+i eta)^2 I - D]^{-1}``
with ``D`` the mass-weighted dynamical matrix [THz^2] and
``G^< = -i n_B A``, ``A = i(G^R - G^A)``.

The equal-time **mass-weighted** displacement correlation
``<w_a w_b>`` (``w = sqrt(m) u``, m in amu, u in Angstrom, so ``<ww>`` is in
``amu * Angstrom^2``) is obtained from the lesser Green's function by

    <w_a w_b> = UU_PREFACTOR * sum_omega  i G^<_{ab}(omega) * dw / (2 pi)

summed over the **full symmetric** frequency axis (both signs of omega; the two
halves supply the ``n_B`` and ``n_B+1`` pieces). ``UU_PREFACTOR`` is fixed by
requiring the equilibrium limit to reproduce the analytic mode sum
``<w_a w_b> = sum_s (hbar / 2 Omega_s)(2 n_s + 1) e_{a,s} e*_{b,s}`` (with
``Omega_s = omega_s[THz] * THZ_TO_RAD`` the angular frequency). It comes out
near unity (~1.01) -- a useful sanity check. See ``tests/.../test_static_se.py``.

Keeping ``<ww>`` in ``amu*Angstrom^2`` and the mass-weighted FC4 in
``eV / Angstrom^4 / amu^2`` makes ``Sigma_L`` come out in ``eV/Angstrom^2/amu``,
which converts to ``THz^2`` with the *same* ``CONVERSION_THZ2`` the FC2 -> Phi
path uses.
"""

from __future__ import annotations

import numpy as np

from phonon_inputs.constants import (
    AMU_KG,
    CONVERSION_FC3_THZ,
    CONVERSION_THZ2,
    HBAR_SI,
    THZ_TO_RAD,
)

#: Converts ``sum_omega i G^<_{ab} dw/(2 pi)`` (omega in THz) into the
#: mass-weighted correlation ``<w_a w_b>`` in ``amu * Angstrom^2``.
#: ``hbar / (THZ_TO_RAD * AMU_KG * 1e-20) ~ 1.01``.
UU_PREFACTOR = HBAR_SI / (THZ_TO_RAD * AMU_KG * 1e-20)


def equal_time_uu(g_lesser, dw_thz, *, freq_axis=-3, average_axes=None):
    """Equal-time mass-weighted displacement correlation ``<w_a w_b>``.

    Parameters
    ----------
    g_lesser : complex ndarray
        Lesser Green's function ``G^<(omega)`` on the **full symmetric**
        frequency grid, mass-weighted (THz^2 convention). Shape
        ``(..., nfreq, N, N)`` with the frequency axis at ``freq_axis``
        and the two matrix axes last. A leading transverse-q axis (and any
        other batch axis to be averaged) may be reduced via ``average_axes``.
    dw_thz : float
        Frequency-grid spacing ``Delta omega`` [THz].
    freq_axis : int
        Axis of ``g_lesser`` holding the frequency samples (default ``-3``).
    average_axes : int | tuple[int] | None
        Axes to average over after the frequency integral (e.g. a transverse
        Brillouin-zone q-mesh: ``<uu> = (1/N_q) sum_q ...``). ``None`` = no
        averaging.

    Returns
    -------
    uu : real ndarray, shape ``(..., N, N)``
        Real symmetric ``<w_a w_b>`` in ``amu * Angstrom^2``. (Imaginary part
        is round-off; it is dropped.)

    Notes
    -----
    ``i G^<`` equals ``n_B(omega) A(omega)`` (real, symmetric) on the grid, so
    summing it over the full axis is a real Riemann integral. Reuse the same
    ``G^<`` already built for the bubble ``Sigma_B^<`` -- do *not* recompute it.
    """
    g = np.asarray(g_lesser)
    integrand = 1j * g                       # = n_B A  (real up to round-off)
    uu = np.sum(integrand, axis=freq_axis) * (dw_thz / (2.0 * np.pi))
    uu = uu * UU_PREFACTOR
    if average_axes is not None:
        uu = np.mean(uu, axis=average_axes)
    uu = 0.5 * (uu + np.swapaxes(uu, -1, -2).conj())   # Hermitize / symmetrize
    return uu.real


def equilibrium_uu_modesum(dyn_matrix, temperature, *, eps_thz2=1e-12):
    """Analytic equilibrium ``<w_a w_b>`` from the harmonic mode sum.

    ``<w_a w_b> = sum_s (hbar / 2 Omega_s)(2 n_s + 1) e_{a,s} e*_{b,s}`` in
    ``amu * Angstrom^2``, with ``Omega_s = omega_s[THz] * THZ_TO_RAD`` and the
    sum over modes with ``omega_s^2 > eps_thz2`` (rigid/zero modes excluded).

    This is the ground-truth reference for :func:`equal_time_uu`; it is *not*
    used in the solver loop (there ``<uu>`` is always taken from ``G^<``).

    Parameters
    ----------
    dyn_matrix : (N, N) array
        Mass-weighted dynamical matrix ``D`` [THz^2] (Hermitian).
    temperature : float
        Temperature [K].
    eps_thz2 : float
        Modes with ``omega_s^2`` below this (THz^2) are treated as rigid and
        dropped from the sum.
    """
    from .grids import bose_full_axis

    d = 0.5 * (np.asarray(dyn_matrix) + np.asarray(dyn_matrix).conj().T)
    w2, evecs = np.linalg.eigh(d)
    keep = w2 > eps_thz2
    w2 = w2[keep]
    evecs = evecs[:, keep]
    omega_thz = np.sqrt(w2)
    n_s = bose_full_axis(omega_thz, temperature)
    # hbar / (2 Omega_s) in amu*Angstrom^2, Omega_s = omega_thz * THZ_TO_RAD.
    zp = HBAR_SI / (2.0 * omega_thz * THZ_TO_RAD * AMU_KG * 1e-20)
    weight = zp * (2.0 * n_s + 1.0)          # (n_modes,)
    uu = (evecs * weight[None, :]) @ evecs.conj().T
    return 0.5 * (uu + uu.conj().T).real


# ---------------------------------------------------------------------------
# Static self-energy contraction kernels (loop + tadpole)
# ---------------------------------------------------------------------------
#
# Unit bridge (derived; see module docstring + the project memory). Unlike the
# cubic *bubble* vertex (``M_stacked`` carries ``CONVERSION_FC3_THZ`` because the
# bubble is ``Phi3^2 G G`` with an explicit hbar*dw prefactor), the loop and
# tadpole renormalise FC2 *directly* -- they produce an ``omega^2`` object from
# ``Phi4 : <uu>`` resp. ``Phi3 : <u>``. Both therefore use the SAME FC2 -> Phi
# conversion ``CONVERSION_THZ2``, fed:
#   * mass-weighted FCs (each leg divided by sqrt(mass), no THz factor), and
#   * ``<ww>`` in amu*Angstrom^2 (:func:`equal_time_uu`).
# A single mode m, omega0, on-site quartic g4 [eV/A^4] then gives the textbook
# self-consistent-phonon shift ``Omega^2 = omega0^2 + 1/2 CONVERSION_THZ2
# (g4/m^2) <w^2>`` -- stiffening for g4 > 0.


def sigma_loop(fc4_mw, uu):
    """Quartic loop self-energy ``Sigma_L = 1/2 Phi4 : <uu>`` [THz^2].

    Parameters
    ----------
    fc4_mw : ndarray, shape (N, N, N, N)
        **Mass-weighted** FC4: ``fc4_raw_{abcd} / sqrt(m_a m_b m_c m_d)`` in
        ``eV / Angstrom^4 / amu^2`` (the four legs are the device DOFs ``a, b``
        kept and ``c, d`` contracted; the tensor must be symmetric in ``c, d``).
    uu : ndarray, shape (N, N)
        Equal-time correlation ``<w_c w_d>`` in ``amu * Angstrom^2`` from
        :func:`equal_time_uu`.

    Returns
    -------
    sigma_L : real ndarray, shape (N, N)
        Static, real, symmetric loop self-energy [THz^2], to be added to the
        dynamical matrix ``Phi``.
    """
    sig = 0.5 * CONVERSION_THZ2 * np.einsum("abcd,cd->ab", fc4_mw, uu)
    return 0.5 * (sig + sig.conj().T).real


def tadpole_source(fc3_mw, uu):
    """Thermally-dressed force source ``s_a = 1/2 Phi3_{acd} <u_c u_d>``.

    Returned in **mass-weighted** units (``eV / Angstrom / sqrt(amu)``), i.e.
    ``s_a = 1/2 sum_cd fc3_mw_{acd} <w_c w_d>`` with ``fc3_mw`` divided by
    ``sqrt(m_a m_c m_d)``. This is the quantity whose vanishing defines the
    finite-T relaxed structure (the tadpole source / force-balance residual).
    """
    return 0.5 * np.einsum("acd,cd->a", fc3_mw, uu)


def mean_force(fc3_mw, uu, masses_dof, f0_ev_per_a=None):
    """Quantum-statistical average force ``<F_a>`` [eV/Angstrom] (diagnostic).

    ``<F_a> = F0_a + 1/2 sum_cd Phi3_{acd} <u_c u_d>``. Used only as the
    equilibrium force-balance check (brief §3.6): on a consistently relaxed
    symmetric reference ``<F>`` is below the DFT force tolerance and its
    symmetry-forbidden components vanish. Out of equilibrium a nonzero ``<F>``
    is physics (thermoelastic strain), not a bug -- so call this on the
    equilibrium reference only.

    ``masses_dof`` is the per-DOF mass [amu] (length N, atom masses repeated x3)
    used to undo the mass-weighting of :func:`tadpole_source`. ``f0_ev_per_a``
    is the static reference force (default 0 = relaxed).
    """
    s = tadpole_source(fc3_mw, uu)               # eV/A/sqrt(amu)
    f = s * np.sqrt(np.asarray(masses_dof))      # -> eV/A
    if f0_ev_per_a is not None:
        f = f + np.asarray(f0_ev_per_a)
    return f


def mean_displacement(fc3_mw, uu, phi_eff, *, optical_projector=None):
    """Static mean displacement ``<w> = -CONVERSION_THZ2 Phi_eff^{-1} s`` (mass-weighted).

    Solves the anharmonic force balance ``Phi_eff <w> = -CONVERSION_THZ2 s`` with
    ``s = tadpole_source`` and ``Phi_eff`` the (THz^2) effective dynamical
    matrix. Only the **optical** intermediate may enter as a self-energy; the
    acoustic / rigid-translation (``T_A``) part is singular and must be handled
    by cell relaxation, so it is projected out via ``optical_projector`` (the
    ``Q = I - T T^T`` from :func:`phonon.solver.zero_modes.build_translation_projector`,
    or the dynamical zero-mode projector). Returns ``<w>`` in ``sqrt(amu)*Angstrom``.
    """
    s = tadpole_source(fc3_mw, uu)               # (N,)
    rhs = -CONVERSION_THZ2 * s
    if optical_projector is not None:
        rhs = optical_projector @ rhs
    d = 0.5 * (np.asarray(phi_eff) + np.asarray(phi_eff).conj().T)
    w_mean = np.linalg.solve(d, rhs)
    if optical_projector is not None:
        w_mean = optical_projector @ w_mean
    return w_mean


def sigma_tadpole(fc3_mw, w_mean):
    """Cubic tadpole self-energy ``Sigma_T = Phi3 : <u>`` [THz^2].

    ``Sigma_T_{ab} = CONVERSION_THZ2 sum_c fc3_mw_{abc} <w_c>`` with ``fc3_mw``
    divided by ``sqrt(m_a m_b m_c)`` and ``<w_c>`` from
    :func:`mean_displacement`. Static, real, symmetric; added to ``Phi``.
    ~0 for a consistently relaxed symmetric crystal (then ``<w> ~ 0``).
    """
    sig = CONVERSION_THZ2 * np.einsum("abc,c->ab", fc3_mw, w_mean)
    return 0.5 * (sig + sig.conj().T).real


# ---------------------------------------------------------------------------
# Device-level assembly + SCBA hook
# ---------------------------------------------------------------------------


def assemble_device_fc3_tensor(phi_blocks, n_slabs, n_dof):
    """Dense device FC3 tensor ``Phi3_dev[A, B, C]`` (A,B,C device DOFs).

    ``phi_blocks`` is the ``{(I, K, K'): Phi[n_dof, n_dof, n_dof]}`` dict from
    :func:`phonon.solver.fc3_device.build_device_fc3_blocks` (leg ``a`` in slab
    ``I``, ``b`` in slab ``K``, ``c`` in slab ``K'``), carrying the bubble
    mass-weighting (``CONVERSION_FC3_THZ``). The returned dense tensor is in the
    **same** units; divide by ``CONVERSION_FC3_THZ`` for the loop/tadpole
    convention (see :func:`device_fc3_mass_weighted`).
    """
    N_D = n_slabs * n_dof
    tensor = np.zeros((N_D, N_D, N_D), dtype=float)
    for (I, K, Kp), blk in phi_blocks.items():
        tensor[I * n_dof:(I + 1) * n_dof,
               K * n_dof:(K + 1) * n_dof,
               Kp * n_dof:(Kp + 1) * n_dof] = np.asarray(blk).real
    return tensor


def device_fc3_mass_weighted(phi_blocks, n_slabs, n_dof):
    """Device FC3 in the loop/tadpole mass-weighting (``Phi3_dev / CONVERSION_FC3_THZ``).

    The bubble device blocks carry ``CONVERSION_FC3_THZ`` on top of the bare
    ``1/sqrt(m_a m_b m_c)`` mass-weighting; the loop/tadpole instead bridge to
    THz^2 with ``CONVERSION_THZ2``, so they need the FC3 with the mass-weighting
    only. Dividing the assembled device tensor by ``CONVERSION_FC3_THZ`` yields
    exactly that. Returns ``(N_D, N_D, N_D)``.
    """
    return assemble_device_fc3_tensor(phi_blocks, n_slabs, n_dof) / CONVERSION_FC3_THZ


def build_static_self_energy_hook(
    *, dw_thz, n_dof, n_slabs,
    fc3_dev_mw=None, fc4_dev_mw=None,
    use_loop=False, use_tadpole=False,
    optical_projector=None,
):
    """Build the per-iteration static self-energy hook for the SCBA loop.

    Returns ``hook(G_less_dev_q, sigma_static_current, H_D_list) ->
    Sigma_static`` with ``Sigma_static`` shape ``(n_kpts, N_D, N_D)``. Each call
    forms ``<uu>`` from the current device ``G^<`` and returns
    ``Sigma_L (+ Sigma_T)`` to be added to the dynamical matrix. Hermitization +
    ASR projection are applied by the caller (``scba_loop``).

    Restricted to the Gamma device (``n_kpts == 1``) for now; the q-resolved
    static self-energy (folding ``<uu>(q)`` with the transverse Bloch phases) is
    future work and raises if ``G_less_dev_q`` carries more than one q-point.

    Parameters
    ----------
    fc3_dev_mw : (N_D, N_D, N_D) array, optional
        Loop/tadpole mass-weighted device FC3 (:func:`device_fc3_mass_weighted`).
        Required when ``use_tadpole``.
    fc4_dev_mw : (N_D, N_D, N_D, N_D) array, optional
        Loop/tadpole mass-weighted device FC4. Required when ``use_loop``.
    optical_projector : (N_D, N_D) array, optional
        ``Q`` projecting out the rigid-translation / acoustic intermediate for
        the tadpole (only the optical part ``T_O`` may enter as a self-energy).
    """
    if use_tadpole and fc3_dev_mw is None:
        raise ValueError("use_tadpole=True requires fc3_dev_mw")
    if use_loop and fc4_dev_mw is None:
        raise ValueError("use_loop=True requires fc4_dev_mw (FC4 not available)")
    N_D = n_slabs * n_dof

    def hook(g_less_dev_q, sigma_static_current, h_d_list):
        if g_less_dev_q.shape[0] != 1:
            raise NotImplementedError(
                "static self-energy (loop/tadpole) is implemented for the "
                "Gamma device (n_kpts=1) only; got "
                f"n_kpts={g_less_dev_q.shape[0]}")
        uu = equal_time_uu(g_less_dev_q[0], dw_thz)        # (N_D, N_D)
        sig = np.zeros((N_D, N_D), dtype=float)
        if use_loop:
            sig = sig + sigma_loop(fc4_dev_mw, uu)
        if use_tadpole:
            phi_eff = (np.asarray(h_d_list[0]) + sigma_static_current[0]).real
            w_mean = mean_displacement(
                fc3_dev_mw, uu, phi_eff, optical_projector=optical_projector)
            sig = sig + sigma_tadpole(fc3_dev_mw, w_mean)
        return sig[np.newaxis].astype(complex)             # (1, N_D, N_D)

    return hook
