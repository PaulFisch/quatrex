"""Spectral function ``A(q, omega)`` and decomposed renormalised bands.

Visualises how each anharmonic self-energy renormalises the phonon band diagram
(brief Task 3): the static loop ``Sigma_L`` and tadpole ``Sigma_T`` shift
frequencies, the cubic bubble ``Sigma_B(q, omega)`` shifts *and* broadens. The
primary object is the (q, omega)-resolved spectral function

    A(q, omega) = -1/pi Im Tr G^R(q, omega),
    G^R = [(omega + i eta)^2 I - D(q) - Sigma_L - Sigma_T - Sigma_B(q, omega)]^{-1}

with **no contact self-energy** (bulk / intrinsic renormalisation). Peak
position = renormalised frequency, peak width = linewidth (bubble Im part).

The array-based core (this module) is decoupled from phonopy and unit-tested on
synthetic dynamical matrices; :func:`dynamical_matrix_at_q` /
:func:`dynamical_matrix_qpath` are thin phonopy bridges that build ``D(q)`` in
the dense solver's mass-weighted THz^2 convention (reusing
:func:`phonon.phonon_inputs.convention.gauge_transform_A_to_B`).

All matrices are mass-weighted, THz^2; frequencies THz (negative = imaginary /
soft mode, plotted below zero).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Band frequencies from a (renormalised) dynamical matrix
# ---------------------------------------------------------------------------


def frequencies_from_dynamical(dyn):
    """Signed branch frequencies ``sign(w2) sqrt(|w2|)`` [THz].

    ``dyn`` is a Hermitian dynamical matrix [THz^2], shape ``(N, N)`` or a stack
    ``(nq, N, N)``. Negative eigenvalues (soft / unstable modes) are returned as
    *negative* frequencies (the convention for plotting an imaginary mode below
    the axis), not dropped.
    """
    dyn = np.asarray(dyn)
    herm = 0.5 * (dyn + np.conj(np.swapaxes(dyn, -1, -2)))
    w2 = np.linalg.eigvalsh(herm)
    return np.sign(w2) * np.sqrt(np.abs(w2))


def _hermitize(m):
    return 0.5 * (np.asarray(m) + np.conj(np.swapaxes(np.asarray(m), -1, -2)))


def _broadcast_static(sigma, nq, N):
    """Normalise a static self-energy to ``(nq, N, N)`` (or None)."""
    if sigma is None:
        return None
    sigma = np.asarray(sigma)
    if sigma.shape == (N, N):
        return np.broadcast_to(sigma, (nq, N, N))
    if sigma.shape == (nq, N, N):
        return sigma
    raise ValueError(
        f"static self-energy shape {sigma.shape} != (N,N) or (nq,N,N) "
        f"with N={N}, nq={nq}")


# ---------------------------------------------------------------------------
# Quasiparticle bands (collapse each spectral blob to its peak)
# ---------------------------------------------------------------------------


def quasiparticle_bands(dyn_q, *, sigma_static=None):
    """Static quasiparticle bands ``Omega_n(q) = sqrt(eig(D + Re Sigma_static))``.

    Exact for the static (loop + tadpole) shifts. ``dyn_q`` is ``(nq, N, N)``;
    ``sigma_static`` is ``(N, N)`` or ``(nq, N, N)`` or ``None`` (bare bands).
    Returns ``(nq, N)`` signed frequencies (sorted ascending per q).
    """
    dyn_q = np.asarray(dyn_q)
    nq, N, _ = dyn_q.shape
    sig = _broadcast_static(sigma_static, nq, N)
    eff = dyn_q if sig is None else dyn_q + sig.real
    return frequencies_from_dynamical(eff)


def decomposition_bands(dyn_q, *, sigma_loop=None, sigma_tadpole=None):
    """Decomposed band sets: bare -> +loop -> +loop+tadpole.

    Returns a dict of ``(nq, N)`` band arrays:
      * ``"bare"``          : ``sqrt(eig D(q))``
      * ``"loop"``          : ``sqrt(eig (D + Sigma_L))``      (if ``sigma_loop``)
      * ``"loop_tadpole"``  : ``sqrt(eig (D + Sigma_L + Sigma_T))`` (if both)

    The bubble contribution is the spectral function :func:`spectral_function_qw`
    on top of the static ``Phi_eff`` (shift *and* width), not a band line.
    """
    dyn_q = np.asarray(dyn_q)
    nq, N, _ = dyn_q.shape
    out = {"bare": frequencies_from_dynamical(dyn_q)}
    sl = _broadcast_static(sigma_loop, nq, N)
    st = _broadcast_static(sigma_tadpole, nq, N)
    if sl is not None:
        out["loop"] = frequencies_from_dynamical(dyn_q + sl.real)
        if st is not None:
            out["loop_tadpole"] = frequencies_from_dynamical(
                dyn_q + sl.real + st.real)
    elif st is not None:
        out["tadpole"] = frequencies_from_dynamical(dyn_q + st.real)
    return out


# ---------------------------------------------------------------------------
# (q, omega) spectral function
# ---------------------------------------------------------------------------


def spectral_function_qw(
    dyn_q, omega_grid_thz, eta_w_thz, *,
    sigma_static=None, sigma_b=None, return_trace=False,
):
    """Spectral function ``A(q, omega) = -1/pi Im Tr G^R(q, omega)``.

    Parameters
    ----------
    dyn_q : (nq, N, N) array
        Dynamical matrices ``D(q)`` [THz^2] along the q-path.
    omega_grid_thz : (nw,) array
        Positive frequency grid [THz].
    eta_w_thz : float
        Lorentzian half-width on ``omega`` (``z2 = (omega + i eta)^2``). Use a
        small value (a few times the grid spacing) so sharp modes are visible.
    sigma_static : (N,N) or (nq,N,N), optional
        Static loop + tadpole self-energy ``Sigma_L + Sigma_T`` [THz^2]. Added to
        every ``D(q)``.
    sigma_b : (nq, nw, N, N) complex, optional
        Cubic bubble retarded self-energy ``Sigma_B(q, omega)`` [THz^2] on the
        same q / omega grid (the only source of linewidth). ``None`` = static
        spectral function (delta-like peaks broadened only by ``eta``).
    return_trace : bool
        If True, also return the per-mode (diagonal) spectral weight
        ``A_diag[q, omega, N]`` for branch-resolved plots.

    Returns
    -------
    A : (nq, nw) real array
        ``-1/pi Im Tr G^R`` (>= 0 for omega > 0). Peaks at the renormalised
        branch frequencies; widths are the bubble linewidths.
    A_diag : (nq, nw, N) real array, optional
        Per-DOF diagonal spectral weight (only if ``return_trace``).
    """
    dyn_q = np.asarray(dyn_q)
    nq, N, _ = dyn_q.shape
    omega = np.asarray(omega_grid_thz, dtype=float)
    nw = omega.shape[0]
    z2 = (omega + 1j * eta_w_thz) ** 2          # (nw,)
    eye = np.eye(N)
    sig_s = _broadcast_static(sigma_static, nq, N)

    A = np.zeros((nq, nw))
    A_diag = np.zeros((nq, nw, N)) if return_trace else None
    for iq in range(nq):
        d_eff = _hermitize(dyn_q[iq])
        if sig_s is not None:
            d_eff = d_eff + sig_s[iq]
        # M(omega) = z2 I - D_eff - Sigma_B(omega), batched over omega.
        M = z2[:, None, None] * eye[None] - d_eff[None]
        if sigma_b is not None:
            M = M - np.asarray(sigma_b[iq])
        G = np.linalg.inv(M)                     # (nw, N, N)
        A[iq] = -(1.0 / np.pi) * np.imag(np.trace(G, axis1=-2, axis2=-1))
        if return_trace:
            A_diag[iq] = -(1.0 / np.pi) * np.imag(
                np.diagonal(G, axis1=-2, axis2=-1))
    if return_trace:
        return A, A_diag
    return A


def mode_projected_spectral(dyn_q, omega_grid_thz, eta_w_thz, *,
                            sigma_static=None, sigma_b=None):
    """Mode-projected spectral function A_s(q, omega) (eq:spectral_mode).

    Projects the retarded resolvent onto the HARMONIC eigenvectors e_s(q)
    of (D + Sigma_static)(q):

        A_s(q, omega) = -1/pi Im [e_s^H G^R(q, omega) e_s],

    the per-branch Lorentzian whose peak is the renormalised frequency
    and whose width the anharmonic linewidth (eq:lifetime). Same inputs
    and units as :func:`spectral_function_qw`.

    Returns
    -------
    A_s : (nq, nw, N) real array
        Branch-projected spectral weight (branches sorted by the
        harmonic eigenvalue).
    omega_s : (nq, N) real array
        The reference (static-renormalised) branch frequencies [THz].
    """
    dyn_q = np.asarray(dyn_q)
    nq, N, _ = dyn_q.shape
    omega = np.asarray(omega_grid_thz, dtype=float)
    nw = omega.shape[0]
    z2 = (omega + 1j * eta_w_thz) ** 2
    eye = np.eye(N)
    sig_s = _broadcast_static(sigma_static, nq, N)

    A_s = np.zeros((nq, nw, N))
    omega_s = np.zeros((nq, N))
    for iq in range(nq):
        d_eff = _hermitize(dyn_q[iq])
        if sig_s is not None:
            d_eff = d_eff + sig_s[iq]
        w2, ev = np.linalg.eigh(d_eff)
        omega_s[iq] = np.sqrt(np.clip(w2.real, 0.0, None))
        M = z2[:, None, None] * eye[None] - d_eff[None]
        if sigma_b is not None:
            M = M - np.asarray(sigma_b)[iq]
        G = np.linalg.inv(M)                        # (nw, N, N)
        proj = np.einsum("as,wab,bs->ws", ev.conj(), G, ev)
        A_s[iq] = -proj.imag / np.pi
    return A_s, omega_s


def onshell_quasiparticle_bands(
    dyn_q, sigma_b_diag_func, *, sigma_static=None, max_iter=50, tol=1e-6,
):
    """On-shell quasiparticle frequencies including the bubble shift.

    Solves ``Omega_n(q) = sqrt(omega2_n0(q) + Re Sigma_nn(q, Omega_n))`` per
    branch by fixed-point iteration, where ``omega2_n0`` are the eigenvalues of
    ``D(q) + Re Sigma_static`` and ``Sigma_nn`` is the bubble self-energy
    projected on branch ``n``. Valid where ``hbar Omega >> |Sigma|`` (sharp
    modes); unreliable for soft/overdamped modes -- read those off the spectral
    function instead.

    ``sigma_b_diag_func(iq, omega_thz) -> (N,)`` returns the branch-diagonal
    ``Re Sigma_B`` at ``(q_iq, omega)`` (the caller supplies it from the
    eigenbasis of ``D + Re Sigma_static``). Returns ``(nq, N)``.
    """
    dyn_q = np.asarray(dyn_q)
    nq, N, _ = dyn_q.shape
    sig_s = _broadcast_static(sigma_static, nq, N)
    bands = np.zeros((nq, N))
    for iq in range(nq):
        d_eff = _hermitize(dyn_q[iq])
        if sig_s is not None:
            d_eff = d_eff + sig_s[iq].real
        w2_0 = np.linalg.eigvalsh(d_eff)
        for n in range(N):
            omega = np.sqrt(abs(w2_0[n]))
            for _ in range(max_iter):
                shift = sigma_b_diag_func(iq, omega)[n]
                new2 = w2_0[n] + shift
                new = np.sign(new2) * np.sqrt(abs(new2))
                if abs(new - omega) < tol:
                    omega = new
                    break
                omega = abs(new)
            bands[iq, n] = np.sign(w2_0[n] + sigma_b_diag_func(iq, omega)[n]) * omega
    return bands


# ---------------------------------------------------------------------------
# Device / per-region projection (nonequilibrium)
# ---------------------------------------------------------------------------


def region_projected_phi_eff(phi_eff_device, n_dof, layer):
    """Project the device ``Phi_eff`` onto one principal layer and diagonalise.

    Out of equilibrium ``Sigma_L``/``Sigma_T`` are position-dependent, so there
    is no single global device band structure; renormalised "bands" are a
    per-region concept (brief §4.5). Returns the ``(n_dof, n_dof)`` diagonal
    block for slab ``layer`` and its signed branch frequencies.
    """
    sl = slice(layer * n_dof, (layer + 1) * n_dof)
    block = _hermitize(np.asarray(phi_eff_device)[sl, sl])
    return block, frequencies_from_dynamical(block)


# ---------------------------------------------------------------------------
# phonopy bridge: D(q) in the dense-solver convention
# ---------------------------------------------------------------------------


def band_renormalization_bundle(
    dyn_q, omega_grid_thz, eta_w_thz, *,
    q_distance=None, sigma_loop=None, sigma_tadpole=None, sigma_b=None,
):
    """Assemble the full band-renormalisation output in one call.

    Combines the decomposition band sets (bare -> +loop -> +loop+tadpole) and
    the ``A(q, omega)`` heat-map (static ``Phi_eff`` + optional bubble) into a
    dict ready for :func:`phonon.postproc.io.save_spectral` (pass it as
    ``save_spectral(path, **bundle)``). ``dyn_q`` is ``(nq, N, N)``;
    ``sigma_loop``/``sigma_tadpole`` are the static self-energies [THz^2]
    (``(N,N)`` or ``(nq,N,N)``); ``sigma_b`` is the optional bubble
    ``(nq, nw, N, N)``.
    """
    dyn_q = np.asarray(dyn_q)
    nq, N, _ = dyn_q.shape
    sl = _broadcast_static(sigma_loop, nq, N)
    st = _broadcast_static(sigma_tadpole, nq, N)
    static_total = None
    if sl is not None:
        static_total = sl.real.copy()
    if st is not None:
        static_total = st.real.copy() if static_total is None else static_total + st.real

    bands = decomposition_bands(dyn_q, sigma_loop=sigma_loop,
                                sigma_tadpole=sigma_tadpole)
    A = spectral_function_qw(dyn_q, omega_grid_thz, eta_w_thz,
                             sigma_static=static_total, sigma_b=sigma_b)
    if q_distance is None:
        q_distance = np.arange(nq, dtype=float)
    return {"q_distance": np.asarray(q_distance, dtype=float),
            "omega_grid_thz": np.asarray(omega_grid_thz, dtype=float),
            "A": A, "bands": bands}


def dynamical_matrix_at_q(phonon, q_frac, *, conversion_factor=None):
    """Mass-weighted dynamical matrix ``D(q)`` [THz^2] at a 3-vector ``q_frac``.

    Reuses the dense solver's gauge convention
    (:func:`phonon.phonon_inputs.convention.gauge_transform_A_to_B`) so the
    spectral function is consistent with ``get_btd_blocks`` / the transport
    solver. ``conversion_factor`` defaults to ``constants.CONVERSION_THZ2``.
    """
    from phonon_inputs.constants import CONVERSION_THZ2
    from phonon_inputs.convention import gauge_transform_A_to_B

    if conversion_factor is None:
        conversion_factor = CONVERSION_THZ2
    tau = phonon.primitive.scaled_positions
    q = np.asarray(q_frac, dtype=float)
    d_a = phonon.get_dynamical_matrix_at_q(q)
    d_b = gauge_transform_A_to_B(d_a, q, tau)
    return d_b * conversion_factor


def dynamical_matrix_qpath(phonon, q_path, *, conversion_factor=None):
    """Stack of ``D(q)`` [THz^2] over a list/array of fractional q-points.

    Returns ``(nq, N, N)`` complex. Build ``q_path`` (and high-symmetry tick
    positions) with your usual band-path helper; pass the same path used for the
    bare bands so the decomposition overlays line up.
    """
    return np.stack([
        dynamical_matrix_at_q(phonon, q, conversion_factor=conversion_factor)
        for q in np.asarray(q_path, dtype=float)
    ])
