"""Dense reference NEGF/SSE solver for anharmonic phonon transport.

Public entry points:

  * :func:`transmission` -- the single SCBA driver. Solves a finite
    n_slabs-cell device with an optional Gamma-centered transverse q-mesh;
    q_mesh=(1,1) is the Gamma-only finite device, larger meshes are the
    transversely-periodic problem. Computes the full off-diagonal
    self-energy by default, with cutoff knobs for the approximations.
  * :func:`transmission_finite` -- wrapper for q_mesh=(1,1).
  * :func:`transmission_q` -- wrapper for a transverse q-mesh (keeps the
    historical q-path defaults retarded="half", no zero-mode projection).
  * :func:`scba_loop` -- the shared SCBA fixed-point loop (callable for
    custom drivers that supply their own self-energy kernel and OBC).
  * :func:`gamma_project_M_blocks` -- Gamma-only supercell-to-primitive
    projection of the FC3 vertex.
  * :func:`compare_q11_to_finite` -- regression check ensuring
    transmission_q(q_mesh=(1,1)) reproduces transmission_finite.

Implements the self-consistent Born approximation for 3-phonon scattering
following Guo et al., Phys. Rev. B 102, 195412 (2020). The bubble integrand
(diagonal block) is

    Sigma^<(w) = (i hbar / 2) sum_{c,d,e,f} Phi_{a c d}
        integral dw'/(2 pi) G^<_{cf}(w') G^<_{de}(w - w') Phi_{b e f}.

Internal units: THz^2 for dynamical matrices and self-energies. The SCBA loop
stores self-energies as device matrices (n_kpts, nfreq, N_D, N_D), so
off-diagonal slab blocks are represented directly.

Sigma^R reconstruction (the ``retarded`` argument):
  - "half" -- Sigma^R = (Sigma^> - Sigma^<) / 2 (no Kramers-Kronig).
  - "pv"   -- singularity-subtracted principal-value integral (O(nfreq^2)).
  - "fft"  -- FFT Hilbert transform (O(nfreq log nfreq)).
"""

from __future__ import annotations

import warnings

import numpy as np

from phonon_inputs.constants import CONVERSION_THZ2, HBAR_SI, THZ_TO_RAD
from phonon_inputs.convention import get_btd_blocks

from .diagnostics import (
    check_broadening_sign,
    check_full_axis_symmetry,
    symmetrize_lesser_greater,
)
from .grids import bose_full_axis, build_frequency_grid
from .leads import (
    ballistic_transmission_z2,
    build_device_hamiltonian,
    build_device_hamiltonian_massprofile,
    compute_obc_batch,
    solve_green_batch,
)
from .fc3_device import build_device_fc3_blocks
from .retarded import build_retarded
from .se_finite import compute_phph_self_energy
from .se_q import _build_folded_vertices
from .static_se import (
    build_static_self_energy_hook,
    device_fc3_mass_weighted,
)
from .causality import (
    causality_diagnostic,
    dynamical_stability_diagnostic,
    enforce_causality_psd,
)
from .zero_modes import (
    build_dynamical_zero_mode_projector,
    build_translation_projector,
    project_self_energy,
)


# ---------------------------------------------------------------------------
# Frequency-grid safety: fmax >= 2 * omega_max
# ---------------------------------------------------------------------------


def _device_omega_max(H_00, H_01):
    """Largest phonon frequency (in THz) of the periodic Gamma-point
    dynamical matrix ``H_00 + H_01 + H_01^\\dagger``. Used to size the
    bubble-convolution frequency grid: the 3-phonon convolution has
    support ``[-2 omega_max, 2 omega_max]``.
    """
    dyn = H_00 + H_01 + H_01.conj().T
    dyn = 0.5 * (dyn + dyn.conj().T)
    eigs = np.linalg.eigvalsh(dyn)
    return float(np.sqrt(max(eigs.max().real, 0.0)))


def _ensure_fmax(freq_range_thz, H_00, H_01, *, name, auto_extend,
                 margin=1.05, verbose=False):
    """Return a (possibly extended) ``freq_range_thz`` whose ``fmax``
    safely contains the 3-phonon bubble's convolution support.

    If the caller's ``fmax`` is below ``margin * 2 * omega_max`` the
    function -- by default -- transparently extends it (and rescales
    ``nfreq_pos`` to keep ``d_omega`` roughly constant). Setting
    ``auto_extend=False`` instead emits a ``RuntimeWarning`` and
    returns the caller's range unchanged. The verbose path prints a
    one-line confirmation either way.
    """
    fmin, fmax, n_pos = freq_range_thz
    fmin = float(fmin)
    fmax = float(fmax)
    n_pos = int(n_pos)
    omega_max = _device_omega_max(H_00, H_01)
    fmax_req = float(margin) * 2.0 * omega_max
    if fmax >= fmax_req:
        if verbose:
            print(f"  {name}: omega_max={omega_max:.1f} THz, "
                  f"fmax={fmax:.1f} THz >= {fmax_req:.1f} THz (OK)")
        return (fmin, fmax, n_pos)
    if not auto_extend:
        warnings.warn(
            f"{name}: freq_range fmax={fmax:.1f} THz < "
            f"{margin:g}*2*omega_max={fmax_req:.1f} THz. The 3-phonon "
            f"bubble convolution will be truncated/aliased; Sigma will "
            f"pick up large spurious contributions and SCBA will likely "
            f"diverge. Pass fmax >= {fmax_req:.0f} THz or leave "
            f"auto_extend_fmax=True.",
            RuntimeWarning, stacklevel=3)
        return (fmin, fmax, n_pos)
    # Auto-extend, preserving the user's requested d_omega.
    dw = fmax / max(n_pos, 1)
    new_fmax = float(np.ceil(fmax_req))
    new_n_pos = int(np.ceil(new_fmax / dw))
    if verbose:
        print(f"  {name}: auto-extending fmax {fmax:.1f} -> {new_fmax:.1f} "
              f"THz (2*omega_max={2 * omega_max:.1f}); "
              f"nfreq_pos {n_pos} -> {new_n_pos} (d_omega preserved)")
    else:
        warnings.warn(
            f"{name}: auto-extended freq_range fmax "
            f"{fmax:.1f} -> {new_fmax:.1f} THz (2*omega_max="
            f"{2 * omega_max:.1f}); nfreq_pos {n_pos} -> {new_n_pos}.",
            stacklevel=3)
    return (fmin, new_fmax, new_n_pos)


# ---------------------------------------------------------------------------
# FC3 loading
# ---------------------------------------------------------------------------


def load_fc3_raw(fc3_source):
    """Load raw FC3 array from various sources.

    Parameters
    ----------
    fc3_source : str, Path, ndarray, or dict
        - str/Path: HDF5 file with an ``"fc3"`` dataset.
        - ndarray:  raw FC3 passed directly.
        - dict:     must contain ``"fc3"`` (raw array) or ``"ph3"``
          (phono3py object whose ``.fc3`` attribute is the raw array).
    """
    if isinstance(fc3_source, np.ndarray):
        return fc3_source
    if isinstance(fc3_source, dict):
        if "fc3" in fc3_source:
            return np.asarray(fc3_source["fc3"])
        if "ph3" in fc3_source:
            return np.asarray(fc3_source["ph3"].fc3)
        raise ValueError(
            "fc3 dict has neither 'fc3' nor 'ph3' key. "
            "Pass the fc3.hdf5 path or the raw fc3 array instead. "
            f"Available keys: {list(fc3_source.keys())}"
        )
    import h5py
    with h5py.File(str(fc3_source), "r") as f:
        return np.array(f["fc3"])


# ---------------------------------------------------------------------------
# Stabilized Anderson mixing
# ---------------------------------------------------------------------------


def _anderson_mix(x_in, x_out, x_hist, f_hist, prev_fnorm, *,
                  depth, beta, reg=1e-3, step_cap=2.0):
    """Stabilized Anderson (Pulay/DIIS) mixing step.

     This variant adds three safeguards:

      1. restart -- the history is cleared whenever the residual norm grows
      2. regularization -- the normal equations are damped at a
         relative ``reg``
      3. step cap -- an Anderson step that overshoots the damped
         linear step by more than ``step_cap`` is rejected in favour of
         that linear step.
    """
    f_k = x_out - x_in
    fnorm = float(np.linalg.norm(f_k))
    if fnorm > prev_fnorm:
        x_hist.clear()
        f_hist.clear()
    x_hist.append(x_in)
    f_hist.append(f_k)
    x_lin = x_in + beta * f_k
    m = len(f_hist)
    if m >= 2:
        n_use = min(m - 1, depth)
        dF = np.column_stack([
            f_hist[-n_use + j] - f_hist[-n_use + j - 1]
            for j in range(n_use)])
        dX = np.column_stack([
            x_hist[-n_use + j] - x_hist[-n_use + j - 1]
            for j in range(n_use)])
        FtF = dF.conj().T @ dF
        reg_abs = reg * np.trace(FtF).real / max(FtF.shape[0], 1)
        try:
            gamma = np.linalg.solve(
                FtF + reg_abs * np.eye(FtF.shape[0]), dF.conj().T @ f_k)
            x_and = (x_in + beta * f_k) - (dX + beta * dF) @ gamma
        except np.linalg.LinAlgError:
            x_and = x_lin
        lin_step = np.linalg.norm(x_lin - x_in) + 1e-300
        x_mixed = (x_lin
                   if np.linalg.norm(x_and - x_in) > step_cap * lin_step
                   else x_and)
    else:
        x_mixed = x_lin
    if len(x_hist) > depth + 2:
        x_hist.pop(0)
        f_hist.pop(0)
    return x_mixed, fnorm


# ---------------------------------------------------------------------------
# Safeguarded Anderson acceleration
# ---------------------------------------------------------------------------

def _block_weight_vector(sig_l, sig_g, n_slabs, n_dof, *, floor_rel=1e-6):
    """Per-entry weights placing every (I, J) slab-pair block on an
    equal footing in the safeguarded Anderson least-squares.

    The fixed-point vector ``[Sigma^<, Sigma^>]`` is dominated in
    2-norm by the large diagonal blocks; the small far-off-diagonal
    blocks are then invisible to both the secant least-squares and the
    restart test. Weighting each block by ``1/sqrt(||block|| + floor)``
    rescales them to comparable size. Computed once from the first
    bubble output and held fixed.

    Returns a real vector of length ``sig_l.size + sig_g.size`` (the
    weights are shared between the ``Sigma^<`` and ``Sigma^>`` halves
    of the fixed-point vector).
    """
    n_kpts, nfreq, N_D, _ = sig_l.shape
    mag = np.abs(sig_l) + np.abs(sig_g)
    w = np.ones((n_kpts, nfreq, N_D, N_D))
    block_norms = []
    for I in range(n_slabs):
        sI = slice(I * n_dof, (I + 1) * n_dof)
        for J in range(n_slabs):
            sJ = slice(J * n_dof, (J + 1) * n_dof)
            block_norms.append(float(np.linalg.norm(mag[:, :, sI, sJ])))
    scale = max(block_norms) if block_norms else 1.0
    floor = floor_rel * scale + 1e-300
    for I in range(n_slabs):
        sI = slice(I * n_dof, (I + 1) * n_dof)
        for J in range(n_slabs):
            sJ = slice(J * n_dof, (J + 1) * n_dof)
            bn = float(np.linalg.norm(mag[:, :, sI, sJ]))
            w[:, :, sI, sJ] = 1.0 / np.sqrt(bn + floor)
    w_flat = w.ravel()
    return np.concatenate([w_flat, w_flat])


class _AndersonAccelerator:
    """Safeguarded Anderson (Pulay/DIIS) acceleration.

    Fixes the failure mode of :func:`_anderson_mix`, which cleared its
    entire history on *any* residual uptick: on a noisy residual (the
    d5a multi-slab SCBA near 1e-4) that collapses Anderson to a bare
    linear step every iteration, which then diverges:

      * history is *kept* across residual upticks -- Anderson is
        allowed to be non-monotone;
      * the least-squares is solved by a truncated SVD (``lstsq`` with
        ``rcond``), filtering ill-conditioned / linearly-dependent
        secant directions instead of discarding the whole history;
      * an over-long extrapolation step is replaced by the damped
        linear step *for that iteration only*;
      * the history is restarted only on genuine stagnation -- after
        ``stagnation`` consecutive iterations with no new best
        residual;
      * the residual is measured in a block-weighted norm so every
        (I, J) self-energy block is resolved.

    ``step(x_in, x_out)`` returns ``(x_mixed, fnorm)``.
    """

    def __init__(self, depth, beta, *, weights=None, cond_max=1e8,
                 step_cap=2.0, stagnation=5):
        self.depth = int(depth)
        self.beta = float(beta)
        self.weights = weights
        self.cond_max = float(cond_max)
        self.step_cap = float(step_cap)
        self.stagnation = int(stagnation)
        self.x_hist: list = []
        self.f_hist: list = []
        self.best_fnorm = float("inf")
        self.n_since_best = 0

    def fnorm(self, f):
        if self.weights is None:
            return float(np.linalg.norm(f))
        return float(np.linalg.norm(self.weights * f))

    def _truncate(self):
        while len(self.x_hist) > self.depth + 2:
            self.x_hist.pop(0)
            self.f_hist.pop(0)

    def step(self, x_in, x_out):
        f_k = x_out - x_in
        fnorm = self.fnorm(f_k)

        if fnorm < self.best_fnorm:
            self.best_fnorm = fnorm
            self.n_since_best = 0
        else:
            self.n_since_best += 1
        # Stagnation-only restart: NOT triggered by a single uptick.
        if self.n_since_best >= self.stagnation:
            self.x_hist.clear()
            self.f_hist.clear()
            self.n_since_best = 0

        self.x_hist.append(x_in)
        self.f_hist.append(f_k)

        x_lin = x_in + self.beta * f_k
        m = len(self.f_hist)
        if m < 2:
            return x_lin, fnorm

        n_use = min(m - 1, self.depth)
        dF = np.column_stack([
            self.f_hist[-n_use + j] - self.f_hist[-n_use + j - 1]
            for j in range(n_use)])
        dX = np.column_stack([
            self.x_hist[-n_use + j] - self.x_hist[-n_use + j - 1]
            for j in range(n_use)])

        if self.weights is None:
            dF_w, f_w = dF, f_k
        else:
            dF_w = dF * self.weights[:, None]
            f_w = f_k * self.weights

        try:
            gamma, *_ = np.linalg.lstsq(dF_w, f_w, rcond=1.0 / self.cond_max)
        except np.linalg.LinAlgError:
            self._truncate()
            return x_lin, fnorm

        x_and = x_lin - (dX + self.beta * dF) @ gamma

        lin_step = np.linalg.norm(x_lin - x_in) + 1e-300
        x_mixed = (x_lin
                   if np.linalg.norm(x_and - x_in) > self.step_cap * lin_step
                   else x_and)
        self._truncate()
        return x_mixed, fnorm


def _run_jfnk(x0_l, x0_g, x0_s, scba_step, scba_tol, max_iter, verbose):
    """Jacobian-free Newton-Krylov solve of the SCBA fixed point.

    Wraps ``residual(x) = scba_step(x) - x`` (with ``x`` the augmented vector
    ``[Sigma^<, Sigma^>, Sigma_static]``; ``x0_s=None`` reduces it to the bare
    bubble pair) and hands it to :func:`scipy.optimize.newton_krylov`.

    Returns ``(sig_l, sig_g, sig_static, residual)``; ``residual`` is ``None``
    when SciPy is unavailable.
    """
    try:
        from scipy.optimize import newton_krylov
    except ImportError:
        if verbose:
            print("    JFNK requested but SciPy is unavailable; skipping")
        return x0_l, x0_g, x0_s, None

    shape = x0_l.shape
    sz = x0_l.size
    static_on = x0_s is not None
    s_shape = x0_s.shape if static_on else None

    def _unpack(xc):
        sl = xc[:sz].reshape(shape)
        sg = xc[sz:2 * sz].reshape(shape)
        ss = xc[2 * sz:].reshape(s_shape) if static_on else None
        return sl, sg, ss

    def _pack(sl, sg, ss):
        parts = [sl.ravel(), sg.ravel()]
        if static_on:
            parts.append(ss.ravel())
        return np.concatenate(parts)

    def residual_real(xr):
        xc = np.ascontiguousarray(xr).view(np.complex128)
        sl, sg, ss = _unpack(xc)
        sl_n, sg_n, ss_n, _ = scba_step(sl, sg, ss)
        res = _pack(sl_n, sg_n, ss_n) - xc
        return np.ascontiguousarray(res).view(np.float64)

    x0c = _pack(x0_l, x0_g, x0_s)
    x0r = np.ascontiguousarray(x0c).view(np.float64)
    f_tol = max(scba_tol * (float(np.linalg.norm(x0c)) + 1e-300), 1e-300)
    newton_iter = min(int(max_iter), 40)

    try:
        sol_r = newton_krylov(residual_real, x0r, f_tol=f_tol,
                              maxiter=newton_iter, verbose=bool(verbose))
        sol_c = np.ascontiguousarray(sol_r).view(np.complex128)
    except Exception as exc:  # noqa: BLE001 - scipy NoConvergence
        if exc.__class__.__name__ != "NoConvergence" or not exc.args:
            raise
        sol_c = np.ascontiguousarray(
            np.asarray(exc.args[0]).ravel()).view(np.complex128)
        if verbose:
            print("    JFNK: newton_krylov did not meet f_tol; "
                  "using best iterate")

    sl, sg, ss = _unpack(sol_c)
    fr = residual_real(np.ascontiguousarray(
        _pack(sl, sg, ss)).view(np.float64))
    fc = np.ascontiguousarray(fr).view(np.complex128)
    resid = float(np.linalg.norm(fc) / (np.linalg.norm(sol_c) + 1e-300))
    return sl, sg, ss, resid


# ---------------------------------------------------------------------------
# SCBA loop
# ---------------------------------------------------------------------------


def scba_loop(
    *,
    z2_arr, freqs_thz, dw_thz, omega_rad, pos_mask,
    n_slabs, n_dof, N_D,
    H_D_list, obc_list, btd_blocks_list,
    n_kpts,
    se_kernel,
    T_L, T_R,
    max_scba_iter, scba_tol, conservation_tol,
    mixing, anderson_mixing, anderson_depth,
    anderson_step_cap=None,
    scattering_contacts,
    retarded,
    verbose,
    solver=None,
    anderson_safeguard=True,
    zero_mode_projection=True,
    gate_on_conservation=False,
    divergence_guard=True,
    divergence_factor=10.0,
    divergence_patience=4,
    masses_primitive=None,
    causality_projection=False,
    track_diagnostics=False,
    static_se_hook=None,
    static_mixing=None,
    loop_propagator="loop_only",
    stage_loop_first=False,
    stage_max_iter=20,
    stage_tol=1e-3,
):
    """SCBA fixed-point with device-storage self-energies.

    Stores ``Sigma^{<,>,R}`` as ``(n_kpts, nfreq, N_D, N_D)`` device-matrix
    buffers so the self-energy can carry off-diagonal blocks ``Sigma_{IJ}``,
    for both the Gamma-only (n_kpts=1) and q-resolved (n_kpts>1) paths.

    The per-iteration map ``Sigma -> G -> Sigma`` is isolated in the
    nested ``_scba_step``; the outer driver applies one of several
    fixed-point accelerators to it.

    Parameters
    ----------
    se_kernel : callable(G_less_dev, G_great_dev)
        ``G_*_dev`` have shape ``(n_kpts, nfreq, N_D, N_D)``. Returns
        ``(Sigma_l, Sigma_g)`` of the same shape.
    retarded : {"pv", "fft", "half"}
        Method for rebuilding ``Sigma^R`` from the mixed
        ``Sigma^{<,>}`` pair.
    solver : {"linear", "anderson", "jfnk", "anderson+jfnk"}
        Fixed-point accelerator. ``"anderson+jfnk"`` runs safeguarded
        Anderson and, if it fails to converge, restarts a
        Jacobian-free Newton-Krylov solve from the best Anderson
        iterate.
    anderson_safeguard : bool
        ``True`` uses the safeguarded :class:`_AndersonAccelerator`;
        ``False`` keeps the legacy hard-restart :func:`_anderson_mix`.
    zero_mode_projection : bool
        Project ``Sigma^{<,>}`` onto the translation-free subspace
        every iteration (a discrete acoustic sum rule on the
        self-energy). Requires ``masses_primitive``.
    gate_on_conservation : bool
        ``True`` reproduces the legacy combined stop test (SCF residual
        AND heat-flow conservation). ``False`` (default) stops on the
        SCF residual alone and reports conservation as a diagnostic.
    divergence_guard : bool
        Abort early -- returning the best iterate -- once the residual
        exceeds ``divergence_factor`` x its running minimum for
        ``divergence_patience`` consecutive iterations.
    masses_primitive : (n_atoms,) array, optional
        Per-atom primitive-cell masses; required when
        ``zero_mode_projection`` is set.
    static_se_hook : callable, optional
        ``hook(G_less_dev_q, Sigma_static, H_D_list) -> Sigma_static_new``
        (shape ``(n_kpts, N_D, N_D)``): the static, real, Hermitian loop +
        tadpole self-energy built from the current device ``G^<`` (see
        :func:`phonon.solver.static_se.build_static_self_energy_hook`). When
        set, the dynamical matrix used in the Dyson solve becomes
        ``Phi_eff = H_D + Sigma_static`` (re-Hermitized, ASR-projected), and
        ``Sigma_static`` is updated from ``G^<`` each iteration -- the single
        SCP + bubble self-consistency loop. ``Sigma_static`` is carried inside
        the fixed-point vector ``[Sigma^<, Sigma^>, Sigma_static]``, so it is
        accelerated jointly with the bubble by whichever ``solver`` is selected
        (``linear``, ``anderson``, ``jfnk``, ``anderson+jfnk``).
    static_mixing : float, optional
        Under-relaxation for the loop-only *staging* pre-loop (defaults to
        ``mixing``); the main loop mixes the whole augmented vector.
    anderson_step_cap : float, optional
        Cap on the Anderson extrapolation length (multiples of the linear
        step). ``None`` -> ``8.0`` (the value that converges the slow,
        large-magnitude loop/tadpole mode in a few iterations; the
        easily-converged bubble is insensitive to it in the available tests).
        Pass a tighter value (e.g. the historical ``2.0``) for a strongly
        coupled bubble near the SCBA breakdown that needs a conservative step.
    stage_loop_first : bool
        Converge the loop/tadpole-only sub-problem (bubble off) before enabling
        the bubble, so the bubble is built on a stable ``Phi_eff`` (brief §3.5).
    stage_max_iter, stage_tol : int, float
        Iteration cap / relative tolerance for the loop-only staging pre-loop.
    """
    if solver is None:
        solver = "anderson" if anderson_mixing else "linear"
    if solver not in ("linear", "anderson", "jfnk", "anderson+jfnk"):
        raise ValueError(
            f"Unknown solver={solver!r}. Use 'linear', 'anderson', "
            "'jfnk', or 'anderson+jfnk'.")

    nfreq = len(freqs_thz)
    sl0 = slice(0, n_dof)
    sl_last = slice((n_slabs - 1) * n_dof, n_slabs * n_dof)
    shape = (n_kpts, nfreq, N_D, N_D)

    Q_trans = None
    if zero_mode_projection:
        # Use the dynamical projector built from H_00 + H_01 + H_01^T
        # when the periodic blocks are available -- this catches every
        # near-zero cell mode (the three cartesian translations plus
        # any soft cell mode such as the wire-twist) instead of only
        # the rigid translations. For a 1-D Si nanowire with C4
        # symmetry the twist mode at q=0 leaves the translation
        # projector untouched but feeds the same IR / acoustic
        # instability that destabilises SCBA. Fall back to the
        # mass-weighted translation projector when btd_blocks_list is
        # empty (toy harnesses that build H_D directly).
        if btd_blocks_list:
            h00_g, h01_g = btd_blocks_list[0]
            Q_trans = build_dynamical_zero_mode_projector(
                h00_g, h01_g, n_slabs)
        else:
            if masses_primitive is None:
                raise ValueError(
                    "zero_mode_projection=True needs either "
                    "btd_blocks_list (dynamical projector) or "
                    "masses_primitive (translation-only fallback)")
            n_atoms_prim = len(np.asarray(masses_primitive))
            if n_dof % n_atoms_prim != 0:
                raise ValueError(
                    f"n_dof={n_dof} is not a multiple of "
                    f"len(masses_primitive)={n_atoms_prim}")
            Q_trans = build_translation_projector(
                masses_primitive, n_slabs, n_cart=n_dof // n_atoms_prim)

    # Static (loop + tadpole) self-energy: an FC2-like renormalisation
    # Phi_eff = H_D + Sigma_static, carried as part of the fixed-point vector
    # x = [Sigma^<, Sigma^>, Sigma_static] so the bubble accelerators
    # (safeguarded Anderson / JFNK) drive it JOINTLY with the bubble -- the
    # conserving SCP + bubble scheme -- instead of slow linear mixing. The map
    # stays stationary because Sigma_static is recomputed from the current
    # iterate's G inside the step (not lagged side-state).
    static_on = static_se_hook is not None
    static_resid = 0.0
    Sigma_static = (np.zeros((n_kpts, N_D, N_D), dtype=complex)
                    if static_on else None)
    if static_on and loop_propagator not in ("loop_only", "full_G"):
        raise ValueError(
            "loop_propagator must be 'loop_only' or 'full_G', got "
            f"{loop_propagator!r}")
    static_mix = mixing if static_mixing is None else static_mixing
    _sz_lg = n_kpts * nfreq * N_D * N_D            # size of one Sigma^{<,>} half
    _sz_s = n_kpts * N_D * N_D if static_on else 0

    def _pack(sl, sg, ss):
        parts = [sl.ravel(), sg.ravel()]
        if static_on:
            parts.append(ss.ravel())
        return np.concatenate(parts)

    def _unpack(x):
        sl = x[:_sz_lg].reshape(shape)
        sg = x[_sz_lg:2 * _sz_lg].reshape(shape)
        ss = (x[2 * _sz_lg:].reshape((n_kpts, N_D, N_D))
              if static_on else None)
        return sl, sg, ss

    def _apply_static_asr(sig):
        """Hermitize + ASR-project the static self-energy (per q)."""
        sig = 0.5 * (sig + np.conj(np.swapaxes(sig, -1, -2)))
        if Q_trans is not None:
            sig = Q_trans @ sig @ Q_trans
        return sig

    # Buffers reused by every _scba_step call.
    G_less_dev_q = np.zeros(shape, dtype=complex)
    G_great_dev_q = np.zeros_like(G_less_dev_q)
    G_ret_dev_q = np.zeros_like(G_less_dev_q)
    _G_less_loop_only = (np.zeros(shape, dtype=complex)
                         if static_on else None)
    _spec_L = np.zeros(nfreq)
    _spec_R = np.zeros(nfreq)

    def _loop_only_glesser(sig_static):
        """Device ``G^<`` with the bubble off (only ``Phi_eff = H_D +
        sig_static`` + leads).

        Feeds the loop/tadpole ``<uu>`` for the standard SCP-then-bubble scheme
        (``loop_propagator='loop_only'``), excluding the bubble broadening.
        """
        zero_se = np.zeros((nfreq, N_D, N_D), dtype=complex)
        for iq in range(n_kpts):
            _, g_less, _ = solve_green_batch(
                z2_arr, H_D_list[iq] + sig_static[iq], obc_list[iq],
                zero_se, zero_se, zero_se)
            _G_less_loop_only[iq] = g_less
        return _G_less_loop_only

    def _g_for_static(sig_static):
        """The propagator feeding the static self-energy's ``<uu>``."""
        if loop_propagator == "full_G":
            return G_less_dev_q
        return _loop_only_glesser(sig_static)

    def _scba_step(sig_l, sig_g, sig_static=None):
        """One SCBA pass: input ``[Sigma^{<,>}, Sigma_static]`` -> the
        bubble + static self-energy of the resulting ``G``. Pure map; the
        inputs are not mutated. ``Phi_eff = H_D + sig_static`` enters the
        Dyson solve, and ``sig_static_new`` is recomputed from the current
        ``G`` (Hermitized + ASR-projected) so the whole map is a stationary
        fixed point the accelerators can drive. Returns
        ``(sig_l_new, sig_g_new, sig_static_new, info)`` (``sig_static_new``
        is ``None`` when no static hook is set).
        """
        Sig_R = build_retarded(sig_l, sig_g, freqs_thz, method=retarded)
        if causality_projection:
            # Project Sigma^R onto the causal manifold (Gamma_Sigma PSD
            # for omega > 0) before the Dyson solve. Stabilises the loop
            # on systems where the discretised bubble + per-iteration
            # projection / symmetrisation leak negative eigenvalues into
            # Gamma_Sigma (the "epic divergence" failure mode after the
            # twist projector is already in place).
            Sig_R = enforce_causality_psd(Sig_R, freqs_thz)

        G_less_dev_q[:] = 0.0
        G_great_dev_q[:] = 0.0
        G_ret_dev_q[:] = 0.0
        _spec_L[:] = 0.0
        _spec_R[:] = 0.0

        for iq in range(n_kpts):
            H_eff_iq = (H_D_list[iq] if sig_static is None
                        else H_D_list[iq] + sig_static[iq])
            G_ret, G_less, G_great = solve_green_batch(
                z2_arr, H_eff_iq, obc_list[iq],
                Sig_R[iq], sig_l[iq], sig_g[iq])
            G_less_dev_q[iq] = G_less
            G_great_dev_q[iq] = G_great
            G_ret_dev_q[iq] = G_ret

            obc = obc_list[iq]
            SLg_Gl = obc["Sigma_L_greater"][:, sl0, sl0] @ G_less[:, sl0, sl0]
            SLl_Gg = obc["Sigma_L_lesser"][:, sl0, sl0] @ G_great[:, sl0, sl0]
            _spec_L[:] += HBAR_SI * omega_rad * np.real(
                np.trace(SLg_Gl - SLl_Gg, axis1=-2, axis2=-1))

            SRl_Gg = obc["Sigma_R_lesser"][:, sl_last, sl_last] @ G_great[:, sl_last, sl_last]
            SRg_Gl = obc["Sigma_R_greater"][:, sl_last, sl_last] @ G_less[:, sl_last, sl_last]
            _spec_R[:] += HBAR_SI * omega_rad * np.real(
                np.trace(SRl_Gg - SRg_Gl, axis1=-2, axis2=-1))

        spec_L = _spec_L / n_kpts
        spec_R = _spec_R / n_kpts
        J_L_total = np.sum(spec_L[pos_mask]) * dw_thz * 1e12
        J_R_total = np.sum(spec_R[pos_mask]) * dw_thz * 1e12
        J_total = 0.5 * (J_L_total + J_R_total)
        J_denom = abs(J_L_total) + abs(J_R_total)
        conservation_err = (abs(J_L_total - J_R_total) / J_denom
                            if J_denom > 0 else 0.0)

        sig_l_new, sig_g_new = se_kernel(G_less_dev_q, G_great_dev_q)
        symmetrize_lesser_greater(sig_l_new, sig_g_new)
        if Q_trans is not None:
            # Discrete acoustic sum rule: strip the rigid-translation
            # component so the bubble cannot renormalise the zero modes.
            sig_l_new = project_self_energy(sig_l_new, Q_trans)
            sig_g_new = project_self_energy(sig_g_new, Q_trans)

        sig_static_new = None
        if static_on:
            # Static loop + tadpole self-energy from the current G (Hermitized,
            # ASR-projected). loop_only feeds it the bubble-free G.
            sig_static_new = _apply_static_asr(
                static_se_hook(_g_for_static(sig_static), sig_static,
                               H_D_list))

        info = {
            "Sigma_R": Sig_R,
            "spectral_J_L": spec_L,
            "spectral_J_R": spec_R,
            "J_L_total": J_L_total,
            "J_R_total": J_R_total,
            "J_total": J_total,
            "conservation_err": conservation_err,
        }
        if track_diagnostics:
            info["causality"] = causality_diagnostic(Sig_R, freqs_thz)
            info["stability"] = dynamical_stability_diagnostic(
                H_D_list[0], Sig_R, freqs_thz)
        return sig_l_new, sig_g_new, sig_static_new, info

    def _update_scattering_obc(Sig_R):
        """Refresh the lead OBC self-energies from the current Sigma^R."""
        for iq, (H_00_iq, H_01_iq) in enumerate(btd_blocks_list):
            lead_L = Sig_R[iq, :, sl0, sl0]
            lead_R = Sig_R[iq, :, sl_last, sl_last]
            try:
                obc_try = compute_obc_batch(
                    z2_arr, H_00_iq, H_01_iq, freqs_thz, T_L, T_R,
                    n_slabs=n_slabs,
                    lead_sigma_r_L=lead_L, lead_sigma_r_R=lead_R)
                if not np.any(np.isnan(obc_try["Sigma_L_R"])):
                    obc_list[iq] = obc_try
                elif verbose:
                    print(f"    WARNING: scattering-contact OBC "
                          f"diverged for iq={iq}, keeping previous")
            except RuntimeError:
                if verbose:
                    print(f"    WARNING: scattering-contact OBC "
                          f"failed for iq={iq}, keeping previous")

    def _broadening_report(Sig_R):
        total_viol = 0
        total_max = 0.0
        for iq in range(n_kpts):
            for l in range(n_slabs):
                sl = slice(l * n_dof, (l + 1) * n_dof)
                nv, mv = check_broadening_sign(
                    Sig_R[iq, :, sl, sl], freqs_thz, "SCBA", tol=1e-8)
                total_viol += nv
                total_max = max(total_max, mv)
        return total_viol, total_max

    convergence_history = []

    # --- loop/tadpole-only staging: converge Phi_eff with the bubble off so
    #     the bubble is later built on a stable (stiffened) reference. -------
    if static_on and stage_loop_first:
        for stage_iter in range(stage_max_iter):
            sig_new = _apply_static_asr(static_se_hook(
                _loop_only_glesser(Sigma_static), Sigma_static, H_D_list))
            num = float(np.linalg.norm(sig_new - Sigma_static))
            den = float(np.linalg.norm(Sigma_static)) + 1e-300
            Sigma_static[:] = ((1 - static_mix) * Sigma_static
                               + static_mix * sig_new)
            srel = num / den
            if verbose:
                print(f"    loop-only stage iter {stage_iter + 1}: "
                      f"||dSigma_static||/||Sigma_static|| = {srel:.4e}, "
                      f"max|Sigma_static| = {np.max(np.abs(Sigma_static)):.3e}")
            if srel < stage_tol:
                break

    # --- iteration 0: lowest-order Born (Sigma = 0 -> bare G -> bubble) -
    Sigma_l, Sigma_g, Sigma_static, info = _scba_step(
        np.zeros(shape, dtype=complex), np.zeros(shape, dtype=complex),
        Sigma_static)
    if verbose:
        gl_max = np.max(np.abs(G_less_dev_q))
        sig0 = np.max(np.abs(Sigma_l)) + np.max(np.abs(Sigma_g))
        print(f"    G diagnostic: max|G^<| = {gl_max:.4e}")
        print(f"    Self-energy: max|Sigma_new| = {sig0:.4e} THz^2")
        print(f"    SCBA iter 1: J_L = {info['J_L_total']:.4e} W, "
              f"J_R = {info['J_R_total']:.4e} W")

    use_anderson = solver in ("anderson", "anderson+jfnk")

    accel = None
    if use_anderson and anderson_safeguard:
        weights = _block_weight_vector(Sigma_l, Sigma_g, n_slabs, n_dof)
        if static_on:
            # Put Sigma_static on a comparable footing in the least-squares;
            # floor relative to the bubble magnitude so a (near-)zero static
            # block does not get a blown-up weight.
            lg_scale = float(np.linalg.norm(Sigma_l) + np.linalg.norm(Sigma_g))
            s_scale = float(np.linalg.norm(Sigma_static))
            w_s = 1.0 / np.sqrt(s_scale + 1e-6 * lg_scale + 1e-300)
            weights = np.concatenate([weights, np.full(_sz_s, w_s)])
        # Anderson step-length cap (multiples of the linear step). The slow,
        # large-magnitude loop/tadpole mode needs steps many times the linear
        # step (~1/(1-rate)); an 8x cap converges it in a few iterations
        # (toy: 35 -> 3 iters) and leaves the (easily-converged) bubble
        # unaffected in the tests/sweeps available. Default 8x for both paths;
        # pass a tighter cap (e.g. the historical 2x) for a strongly-coupled
        # bubble near the SCBA breakdown if it needs a more conservative step.
        step_cap = 8.0 if anderson_step_cap is None else anderson_step_cap
        accel = _AndersonAccelerator(anderson_depth, mixing, weights=weights,
                                     step_cap=step_cap)
    _ax_hist, _af_hist, _aprev = [], [], float("inf")

    best_resid = float("inf")
    best_l = Sigma_l.copy()
    best_g = Sigma_g.copy()
    best_s = Sigma_static.copy() if static_on else None
    converged = False
    diagnostics_history = []
    if track_diagnostics:
        # Record iter-0 diagnostics so the trajectory starts at the
        # lowest-order Born point.
        diagnostics_history.append({
            "iter": 0,
            "resid": float("nan"),
            "max_abs_Sigma_R": float(np.max(np.abs(info["Sigma_R"]))),
            "conservation_err": float(info["conservation_err"]),
            "J_total": float(info["J_total"]),
            "causality": info.get("causality", {}),
            "stability": info.get("stability", {}),
        })

    if solver in ("linear", "anderson", "anderson+jfnk"):
        J_total_prev = info["J_total"]
        min_resid = float("inf")
        n_diverging = 0
        for scba_iter in range(1, max_scba_iter):
            if scattering_contacts:
                _update_scattering_obc(info["Sigma_R"])

            Sigma_l_out, Sigma_g_out, Sigma_s_out, info = _scba_step(
                Sigma_l, Sigma_g, Sigma_static)

            x_in = _pack(Sigma_l, Sigma_g, Sigma_static)
            x_out = _pack(Sigma_l_out, Sigma_g_out, Sigma_s_out)
            f = x_out - x_in
            resid = float(np.linalg.norm(f)
                          / (np.linalg.norm(x_in) + 1e-300))
            if static_on:
                static_resid = (
                    float(np.linalg.norm(Sigma_s_out - Sigma_static))
                    / (float(np.linalg.norm(Sigma_static)) + 1e-300))
            convergence_history.append(resid)
            if track_diagnostics:
                diagnostics_history.append({
                    "iter": scba_iter,
                    "resid": resid,
                    "max_abs_Sigma_R":
                        float(np.max(np.abs(info["Sigma_R"]))),
                    "conservation_err": float(info["conservation_err"]),
                    "J_total": float(info["J_total"]),
                    "causality": info.get("causality", {}),
                    "stability": info.get("stability", {}),
                })

            if verbose and scba_iter <= 3:
                max_l_err = 0.0
                max_r_err = 0.0
                for iq in range(n_kpts):
                    for l in range(n_slabs):
                        sl = slice(l * n_dof, (l + 1) * n_dof)
                        le, re = check_full_axis_symmetry(
                            G_ret_dev_q[iq, :, sl, sl],
                            G_less_dev_q[iq, :, sl, sl],
                            G_great_dev_q[iq, :, sl, sl], freqs_thz)
                        max_l_err = max(max_l_err, le)
                        max_r_err = max(max_r_err, re)
                if max_l_err > 1e-3 or max_r_err > 1e-3:
                    print(f"    Symmetry: |G^<+G^>(-)|={max_l_err:.2e}, "
                          f"|G^R-G^R(-)*|={max_r_err:.2e}")

            if solver == "linear":
                x_mixed = x_in + mixing * f
            elif accel is not None:
                x_mixed, _ = accel.step(x_in, x_out)
            else:
                x_mixed, _aprev = _anderson_mix(
                    x_in, x_out, _ax_hist, _af_hist, _aprev,
                    depth=anderson_depth, beta=mixing)

            Sigma_l, Sigma_g, Sigma_static = _unpack(x_mixed)

            J_total = info["J_total"]
            conservation_err = info["conservation_err"]
            rel_change = (abs(J_total - J_total_prev)
                          / (abs(J_total_prev) + 1e-30))
            J_total_prev = J_total

            if resid < best_resid:
                best_resid = resid
                best_l = Sigma_l.copy()
                best_g = Sigma_g.copy()
                if static_on:
                    best_s = Sigma_static.copy()

            if verbose:
                viol, vmax = _broadening_report(info["Sigma_R"])
                msg = (f"    SCBA iter {scba_iter + 1}: J = {J_total:.4e} W, "
                       f"conservation = {conservation_err:.4e}, "
                       f"resid = {resid:.4e}, dJ/J = {rel_change:.4e}, "
                       f"max|Sigma^R| = {np.max(np.abs(info['Sigma_R'])):.2e}")
                if viol > 0:
                    msg += f"  [Gamma sign viol: {viol} pts, max {vmax:.2e}]"
                print(msg)

            numerically_converged = resid < scba_tol
            conserving = conservation_err < conservation_tol
            if numerically_converged and (
                    conserving or not gate_on_conservation):
                converged = True
                if verbose:
                    print(f"    Converged after {scba_iter + 1} iterations "
                          f"(resid={resid:.2e}, dJ/J={rel_change:.2e}, "
                          f"conservation={conservation_err:.2e})")
                break
            if numerically_converged and gate_on_conservation:
                if verbose:
                    print(f"    Numerically converged but conservation NOT "
                          f"satisfied ({conservation_err:.2e} > "
                          f"{conservation_tol:.2e}), continuing...")

            min_resid = min(min_resid, resid)
            if divergence_guard and scba_iter > 2:
                if resid > divergence_factor * min_resid:
                    n_diverging += 1
                else:
                    n_diverging = 0
                if n_diverging >= divergence_patience:
                    if verbose:
                        print(f"    Divergence guard: resid {resid:.2e} "
                              f"exceeds {divergence_factor:g}x running min "
                              f"{min_resid:.2e}; aborting at iter "
                              f"{scba_iter + 1}")
                    break
        else:
            if verbose:
                print(f"  WARNING: SCBA did not converge after "
                      f"{max_scba_iter} iterations "
                      f"(best resid={best_resid:.2e})")
    else:
        # Pure JFNK: seed from the lowest-order Born iterate.
        best_l, best_g = Sigma_l.copy(), Sigma_g.copy()
        best_s = Sigma_static.copy() if static_on else None

    # --- JFNK (pure, or fallback when Anderson did not converge) --------
    if solver in ("jfnk", "anderson+jfnk") and not converged:
        if verbose:
            why = ("pure JFNK solve" if solver == "jfnk"
                   else f"Anderson stalled (best resid {best_resid:.2e}); "
                        f"JFNK fallback")
            print(f"    {why}")
        jl, jg, js, jresid = _run_jfnk(
            best_l, best_g, best_s, _scba_step, scba_tol, max_scba_iter,
            verbose)
        if jresid is not None:
            convergence_history.append(jresid)
            if jresid <= best_resid:
                best_l, best_g, best_s, best_resid = jl, jg, js, jresid
            converged = best_resid < scba_tol
            if verbose:
                print(f"    JFNK final resid = {jresid:.4e} "
                      f"({'converged' if converged else 'not converged'})")

    # --- consistent final evaluation for the returned self-energy ------
    Sigma_l, Sigma_g, Sigma_static = best_l, best_g, best_s
    _, _, _, info = _scba_step(Sigma_l, Sigma_g, Sigma_static)
    Sigma_R = info["Sigma_R"]

    return {
        "spectral_J_L": info["spectral_J_L"],
        "spectral_J_R": info["spectral_J_R"],
        "Sigma_R": Sigma_R,
        "Sigma_l": Sigma_l,
        "Sigma_g": Sigma_g,
        "conservation_err": info["conservation_err"],
        "convergence_history": convergence_history,
        "diagnostics_history": diagnostics_history,
        "converged": converged,
        "scba_residual": best_resid,
        "Sigma_static": (None if Sigma_static is None
                         else Sigma_static.copy()),
        "static_residual": static_resid,
    }


# ---------------------------------------------------------------------------
# Gamma-only supercell->primitive projection of the FC3 vertex
# ---------------------------------------------------------------------------


def gamma_project_M_blocks(
    M_blocks: np.ndarray, prim_indices: np.ndarray, n_atoms: int,
) -> np.ndarray:
    """Project ``M_blocks`` onto the primitive cell at q=Gamma.

    Equivalent to ``np.einsum('ci,aij,dj->acd', T0, M_blocks, T0.conj())``
    where ``T0 = build_gathering_matrix(..., q=(0, 0), ...)`` -- at Gamma the
    matrix is real, integer-valued (each column has exactly one nonzero
    equal to 1), so the einsum reduces to summing supercell-image
    rows/cols grouped by ``prim_indices``. We do that fold directly with
    two ``np.add.at`` reductions, avoiding the ``(n_dof, dim_sc)`` dense
    ``T0`` (~99 % zeros at Gamma).
    """
    n_super = len(prim_indices)
    n_dof = 3 * n_atoms
    dim_sc = 3 * n_super
    assert M_blocks.shape == (n_dof, dim_sc, dim_sc)

    M_r = M_blocks.reshape(n_dof, n_super, 3, dim_sc)
    M_left = np.zeros((n_dof, n_atoms, 3, dim_sc), dtype=M_blocks.dtype)
    np.add.at(M_left, (slice(None), prim_indices, slice(None), slice(None)), M_r)

    M2 = M_left.reshape(n_dof, n_atoms, 3, n_super, 3)
    M_proj = np.zeros((n_dof, n_atoms, 3, n_atoms, 3), dtype=M_blocks.dtype)
    np.add.at(M_proj, (slice(None), slice(None), slice(None), prim_indices, slice(None)), M2)

    return M_proj.reshape(n_dof, n_dof, n_dof)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
#
# A single driver, transmission(), computes anharmonic phonon transport on a
# finite n_slabs-cell device with an optional transverse q-mesh. The two named
# wrappers select the historical defaults:
#
#   * transmission_finite -> q_mesh=(1,1)            (Gamma-only device)
#   * transmission_q      -> q_mesh=q_mesh_transverse (transversely periodic)
#
# Both compute the FULL off-diagonal self-energy by default; the cutoff knobs
# (sigma_cutoff / vertex_cutoff / g_cutoff) turn on the documented
# approximations (sigma_cutoff=0 is Guo's diagonal-block approximation III).


def _build_q_points(q_mesh):
    """Gamma-centered transverse q-mesh and its difference map.

    Returns ``(q_points, q_diff_map, n_kpts)``. ``q_mesh=(1,1)`` gives the
    single Gamma point, reducing the q-resolved path to the finite device.
    """
    from phonon_inputs.separable import build_q_diff_map
    nkx, nky = q_mesh
    q_1d_x = [i / nkx for i in range(nkx)]
    q_1d_y = [j / nky for j in range(nky)]
    q_points = [(qx, qy) for qx in q_1d_x for qy in q_1d_y]
    q_diff_map = build_q_diff_map(nkx, nky)
    return q_points, q_diff_map, len(q_points)


def _load_mass_weighted_fc3(phonon, fc3_hdf5, M_stacked_override,
                            transport_direction, *, enforce_asr, vertex_scale,
                            verbose):
    """Mass-weighted real-space FC3 plus the supercell mapping.

    Applies the Gamma-translation ASR projection (``enforce_asr``) and the
    vertex rescaling (``vertex_scale``). Returns
    ``(M_stacked, mapping, fc3_raw)`` where ``mapping`` is
    ``(prim_indices, cell_frac, slab_indices, ref_sc_atoms)``.
    """
    from phonon_inputs.separable import (
        build_supercell_mapping,
        build_realspace_fc3_matrices,
        enforce_asr_fc3_matrices,
    )
    n_atoms = len(phonon.primitive.masses)
    mapping = build_supercell_mapping(phonon, transport_direction)
    prim_indices, _cell_frac, _slab_indices, ref_sc_atoms = mapping

    if M_stacked_override is not None:
        M_stacked = M_stacked_override
        fc3_raw = None
    else:
        fc3_raw = load_fc3_raw(fc3_hdf5)
        M_stacked = build_realspace_fc3_matrices(
            fc3_raw, n_atoms, phonon.supercell.masses, ref_sc_atoms)

    if enforce_asr:
        # Project the FC3 onto the Gamma-translation null space on both device
        # legs before it enters the vertex. Hiphive enforces only the leg-1
        # translational ASR by construction; the leg-j/k legs keep a ~0.8
        # relative violation on open-wire FC3, which couples the vertex into
        # the acoustic/twist subspace and stiffens the small-eta SCBA fixed
        # point. Projection is linear, so it commutes with vertex_scale.
        norm_before = float(np.linalg.norm(M_stacked))
        M_stacked = enforce_asr_fc3_matrices(M_stacked, n_atoms, prim_indices)
        if verbose:
            norm_after = float(np.linalg.norm(M_stacked))
            print(f"  FC3 ASR projection: ||M||_F {norm_before:.4e} -> "
                  f"{norm_after:.4e} (dropped "
                  f"{1.0 - norm_after / norm_before:.1%})")
    if vertex_scale != 1.0:
        M_stacked = M_stacked * float(vertex_scale)
    return M_stacked, mapping, fc3_raw


def _load_device_fc4_mass_weighted(
    fc4_hdf5, prim_indices, slab_indices, n_atoms, n_slabs, masses_super,
    *, vertex_cutoff=None,
):
    """Device FC4 tensor ``Phi4_dev[A,B,C,D]`` in loop mass-weighting.

    Reads the compact-reference sparse FC4 written by the hiPhive reap
    (datasets ``fc4_atoms`` ``(M, 4)`` supercell atom indices with leg-1 a
    reference atom, and ``fc4_values`` ``(M, 3, 3, 3, 3)`` in eV/Angstrom^4),
    then folds it onto the device with
    :func:`phonon.solver.fc4_device.build_device_fc4_tensor` (mass-weighted by
    ``1/sqrt(m)`` per leg, NO THz conversion -- the loop applies
    ``CONVERSION_THZ2``). The supercell atom indexing must match the solver's
    supercell mapping (same phonopy supercell ordering as the FC3 path).
    """
    import h5py

    from .fc4_device import build_device_fc4_tensor

    with h5py.File(fc4_hdf5, "r") as f:
        if "fc4_atoms" not in f or "fc4_values" not in f:
            raise KeyError(
                f"{fc4_hdf5!r} has no compact-reference FC4 (expected datasets "
                "'fc4_atoms' (M,4) and 'fc4_values' (M,3,3,3,3) from the "
                "hiPhive FC4 reap).")
        atoms = np.asarray(f["fc4_atoms"][:], dtype=int)
        values = np.asarray(f["fc4_values"][:], dtype=float)

    fc4_sparse = {tuple(int(a) for a in atoms[m]): values[m]
                  for m in range(atoms.shape[0])}
    return build_device_fc4_tensor(
        fc4_sparse, prim_indices, slab_indices, masses_super,
        n_atoms, n_slabs, vertex_cutoff=vertex_cutoff)


def _setup_devices(phonon, q_points, transport_direction, n_slabs,
                   z2_arr, freqs_thz, T_L, T_R, mass_profile):
    """Per-q device Hamiltonian, btd blocks, and OBC self-energies.

    With ``mass_profile`` set, each slab carries its own mass (Si force
    constants + per-slab mass, leads stay pure lead-material) via
    :func:`build_device_hamiltonian_massprofile`.
    """
    m_lead = None
    if mass_profile is not None:
        mass_profile = list(mass_profile)
        if len(mass_profile) != n_slabs:
            raise ValueError(
                f"mass_profile length {len(mass_profile)} != n_slabs {n_slabs}")
        m_lead = float(phonon.primitive.masses[0])

    btd_blocks, H_D_all, obc_all = [], [], []
    for qx, qy in q_points:
        H_00, H_01 = get_btd_blocks(
            phonon, (qx, qy), transport_direction=transport_direction,
            conversion_factor=CONVERSION_THZ2)
        btd_blocks.append((H_00, H_01))
        if mass_profile is None:
            H_D = build_device_hamiltonian(H_00, H_01, n_slabs)
        else:
            # un-mass-weight the uniform-lead blocks (K = H * m_lead) then
            # re-weight per slab inside build_device_hamiltonian_massprofile.
            H_D = build_device_hamiltonian_massprofile(
                H_00 * m_lead, H_01 * m_lead, mass_profile)
        H_D_all.append(H_D)
        obc_all.append(compute_obc_batch(
            z2_arr, H_00, H_01, freqs_thz, T_L, T_R, n_slabs=n_slabs))
    return btd_blocks, H_D_all, obc_all


def _ballistic_transmission(z2_arr, btd_blocks, H_D_all, n_dof, N_D):
    """q-averaged ballistic (Caroli) transmission on the frequency grid."""
    trans = np.zeros(len(z2_arr))
    for (H_00, H_01), H_D in zip(btd_blocks, H_D_all):
        H_LD = np.zeros((n_dof, N_D), dtype=complex)
        H_LD[:, :n_dof] = H_01
        H_DR = np.zeros((N_D, n_dof), dtype=complex)
        H_DR[-n_dof:, :] = H_01
        for iw, z2 in enumerate(z2_arr):
            trans[iw] += ballistic_transmission_z2(
                z2, H_D, H_00, H_01, H_LD, H_DR)
    trans /= len(btd_blocks)
    return trans


def _cross_section_area(phonon, transport_direction):
    """Transverse cell area in m^2 (for per-area conductance)."""
    lattice = phonon.primitive.cell
    tidx = "xyz".index(transport_direction)
    perp = [i for i in range(3) if i != tidx]
    a1, a2 = lattice[perp[0]], lattice[perp[1]]
    return np.linalg.norm(np.cross(a1, a2)) * 1e-20


def _device_g_blocks(g_dev, n_slabs, n_dof, max_offset, *, has_q_axis=True):
    """Slice device G ``(n_kpts, nfreq, N_D, N_D)`` into ``{(K, K'): block}``
    within ``|K - K'| <= max_offset`` (each block keeps the leading q axis)."""
    out = {}
    for K in range(n_slabs):
        sK = slice(K * n_dof, (K + 1) * n_dof)
        for Kp in range(n_slabs):
            if max_offset is not None and abs(K - Kp) > max_offset:
                continue
            sKp = slice(Kp * n_dof, (Kp + 1) * n_dof)
            out[(K, Kp)] = (g_dev[:, :, sK, sKp] if has_q_axis
                            else g_dev[:, sK, sKp])
    return out


def _scatter_blocks(dst, blocks, n_dof):
    """Place ``{(I, J): block}`` into the device matrix ``dst[..., I, J]``."""
    for (I, J), blk in blocks.items():
        sI = slice(I * n_dof, (I + 1) * n_dof)
        sJ = slice(J * n_dof, (J + 1) * n_dof)
        dst[..., sI, sJ] = blk


def transmission(
    phonon,
    fc3_hdf5=None,
    *,
    q_mesh: tuple[int, int] = (1, 1),
    n_slabs: int = 1,
    freq_range_thz: tuple[float, float, int] = (0.01, 16.0, 101),
    transport_direction: str = "x",
    eta_factor: float = 1.0,
    temperature: float = 300.0,
    delta_T: float = 10.0,
    max_scba_iter: int = 80,
    scba_tol: float = 1e-3,
    conservation_tol: float = 1e-3,
    mixing: float = 0.3,
    anderson_mixing: bool = False,
    anderson_depth: int = 8,
    anderson_step_cap: float | None = None,
    verbose: bool = True,
    M_stacked_override: np.ndarray = None,
    retarded: str = "fft",
    scattering_contacts: bool = False,
    hilbert_retarded: bool = False,
    sigma_cutoff: int | None = None,
    vertex_cutoff: int | None = None,
    g_cutoff: int | None = None,
    dc_handling: str = "interpolate",
    solver: str | None = None,
    anderson_safeguard: bool = True,
    zero_mode_projection: bool = True,
    gate_on_conservation: bool = False,
    divergence_guard: bool = True,
    auto_extend_fmax: bool = True,
    fmax_margin: float = 1.05,
    causality_projection: bool = False,
    track_diagnostics: bool = False,
    vertex_scale: float = 1.0,
    enforce_asr: bool = False,
    legacy_prefactor: bool = False,
    mass_profile: list | None = None,
    loop: bool = False,
    tadpole: bool = False,
    fc4_hdf5: str | None = None,
    loop_propagator: str = "loop_only",
    stage_loop_first: bool = False,
    static_mixing: float | None = None,
) -> dict:
    """Reference anharmonic phonon transport (NEGF + SCBA, dense).

    Solves the self-consistent Born approximation for 3-phonon scattering on a
    finite ``n_slabs``-cell device with an optional Gamma-centered transverse
    q-mesh. ``q_mesh=(1,1)`` is the finite (Gamma-only) device; larger meshes
    are the transversely-periodic problem with crystal-momentum conservation in
    the bubble. The FULL off-diagonal self-energy is computed by default.

    Approximation knobs (all off by default):

    sigma_cutoff
        Maximum ``|I - J|`` for produced Sigma blocks. ``None`` keeps every
        block; ``0`` is Guo's diagonal-block approximation (III).
    vertex_cutoff
        Maximum slab-distance retained in the FC3 vertex (approximation II).
    g_cutoff
        Maximum ``|K - K'|`` for G blocks used in the inner bubble sum.
    dc_handling
        Treatment of the omega = 0 sample of G before the bubble FFT
        (``"interpolate"`` default, ``"zero"`` legacy, ``"keep"``).

    Numerics:

    retarded : {"fft", "pv", "half"}
        How Sigma^R is rebuilt from the mixed Sigma^{<,>} (FFT Hilbert,
        principal value, or anti-Hermitian half).
    solver : {None, "linear", "anderson", "jfnk", "anderson+jfnk"}
        SCBA fixed-point accelerator; ``None`` derives it from
        ``anderson_mixing``.
    zero_mode_projection, gate_on_conservation, divergence_guard,
    anderson_safeguard, causality_projection
        See :func:`scba_loop`.
    enforce_asr
        Project the FC3 onto the Gamma-translation null space before the
        vertex (stabilises the small-eta SCBA on soft-mode wires).
    legacy_prefactor
        ``True`` restores the 4x-too-large Luisier self-energy prefactor.
    mass_profile : list of float, optional
        Per-slab mass (length ``n_slabs``) for mass-mismatch heterostructures;
        leads stay pure lead-material.

    Returns a dict of spectral / integrated heat currents, ballistic and
    anharmonic conductances, convergence diagnostics, and the converged
    self-energies. The ``eta_factor`` default ``1.0`` (eta ~ d_omega) keeps the
    propagator resolved on the grid; recover eta -> 0 by extrapolating a few
    ``eta_factor`` values rather than running at eta ~ 0.
    """
    if hilbert_retarded:
        retarded = "fft"
    if solver is None:
        solver = "anderson" if anderson_mixing else "linear"

    n_atoms = len(phonon.primitive.masses)
    n_dof = 3 * n_atoms
    N_D = n_slabs * n_dof

    q_points, q_diff_map, n_kpts = _build_q_points(q_mesh)

    # Auto-extend fmax to cover the 3-phonon convolution support
    # [-2*omega_max, 2*omega_max]; a too-small fmax aliases the bubble and
    # destabilises SCBA. Sized off the Gamma-point dynamical matrix.
    H_00_g, H_01_g = get_btd_blocks(
        phonon, (0.0, 0.0), transport_direction=transport_direction,
        conversion_factor=CONVERSION_THZ2)
    name = "transmission_finite" if n_kpts == 1 else "transmission_q"
    freq_range_thz = _ensure_fmax(
        freq_range_thz, H_00_g, H_01_g, name=name,
        auto_extend=auto_extend_fmax, margin=fmax_margin, verbose=verbose)
    freqs_thz, dw_thz, eta_w, z2_arr, pos_mask, mid = build_frequency_grid(
        freq_range_thz, eta_factor=eta_factor)
    nfreq = len(freqs_thz)

    M_stacked, mapping, fc3_raw = _load_mass_weighted_fc3(
        phonon, fc3_hdf5, M_stacked_override, transport_direction,
        enforce_asr=enforce_asr, vertex_scale=vertex_scale, verbose=verbose)
    prim_indices, cell_frac, slab_indices, _ref = mapping

    T_L = temperature + delta_T / 2.0
    T_R = temperature - delta_T / 2.0

    btd_blocks, H_D_all, obc_all = _setup_devices(
        phonon, q_points, transport_direction, n_slabs,
        z2_arr, freqs_thz, T_L, T_R, mass_profile)

    trans_ballistic = _ballistic_transmission(
        z2_arr, btd_blocks, H_D_all, n_dof, N_D)
    A_c = _cross_section_area(phonon, transport_direction)

    omega_rad = freqs_thz * THZ_TO_RAD
    n_bose_L = bose_full_axis(freqs_thz, T_L)
    n_bose_R = bose_full_axis(freqs_thz, T_R)
    spectral_J_ball = (HBAR_SI * omega_rad * (n_bose_L - n_bose_R)
                       * trans_ballistic)
    J_ball_total = np.sum(spectral_J_ball[pos_mask]) * dw_thz * 1e12
    G_ball = J_ball_total / (A_c * delta_T) if delta_T != 0 else 0.0

    if verbose:
        print(f"  Device: {n_slabs} slab(s), {N_D} DOFs, q-mesh "
              f"{q_mesh[0]}x{q_mesh[1]} = {n_kpts} point(s)")
        print(f"  Frequency grid: {nfreq} points, {freqs_thz[0]:.2f} to "
              f"{freqs_thz[-1]:.2f} THz; eta_w = {eta_w:.4e} THz "
              f"(eta/d_omega = {eta_w / dw_thz:.2f})")
        print(f"  Temperature: {temperature} K, delta_T: {delta_T} K")
        print(f"  Cutoffs: sigma={sigma_cutoff}, vertex={vertex_cutoff}, "
              f"g={g_cutoff}; dc_handling={dc_handling}; retarded={retarded}")
        print(f"  SCBA: max {max_scba_iter} iter, tol={scba_tol}, "
              f"mix={mixing}, solver={solver}")
        print(f"  Ballistic max T: {trans_ballistic.max():.4f}; "
              f"G_ball = {G_ball:.2f} W/(m^2 K)")

    sym = 1.0 if legacy_prefactor else None  # None -> correct 1/4 default

    # Build the device vertex once (it does not depend on G, so it is reused
    # across SCBA iterations). Gamma uses the real device blocks; a transverse
    # mesh uses the q-folded blocks for every (iq1, iq2) the bubble couples.
    if n_kpts == 1:
        vertices = {(0, 0): build_device_fc3_blocks(
            M_stacked, prim_indices, slab_indices, n_atoms, n_slabs,
            vertex_cutoff=vertex_cutoff)}
    else:
        vertices = _build_folded_vertices(
            M_stacked, prim_indices, cell_frac, slab_indices, n_atoms, n_slabs,
            n_kpts, q_points, q_diff_map, transport_direction,
            vertex_cutoff=vertex_cutoff)

    def se_kernel(G_less_dev_q, G_great_dev_q):
        sig_l = np.zeros((n_kpts, nfreq, N_D, N_D), dtype=complex)
        sig_g = np.zeros_like(sig_l)
        gl = _device_g_blocks(G_less_dev_q, n_slabs, n_dof, g_cutoff,
                              has_q_axis=True)
        gg = _device_g_blocks(G_great_dev_q, n_slabs, n_dof, g_cutoff,
                              has_q_axis=True)
        sl_b, sg_b = compute_phph_self_energy(
            gl, gg, vertices, n_slabs, n_kpts, q_diff_map, freqs_thz, dw_thz,
            sigma_cutoff=sigma_cutoff, g_cutoff=g_cutoff,
            dc_handling=dc_handling, symmetry_factor=sym)
        _scatter_blocks(sig_l, sl_b, n_dof)
        _scatter_blocks(sig_g, sg_b, n_dof)
        return sig_l, sig_g

    # Static loop/tadpole self-energy (renormalises Phi_eff = Phi + Sigma_L +
    # Sigma_T inside the same SCBA loop). Gamma device (n_kpts=1) only for now.
    static_se_hook = None
    if loop or tadpole:
        if n_kpts != 1:
            raise NotImplementedError(
                "loop/tadpole self-energies are implemented for the Gamma "
                f"device (q_mesh=(1,1)) only; got n_kpts={n_kpts}.")
        fc4_dev_mw = None
        if loop:
            if fc4_hdf5 is None:
                raise ValueError("loop=True requires fc4_hdf5")
            fc4_dev_mw = _load_device_fc4_mass_weighted(
                fc4_hdf5, prim_indices, slab_indices, n_atoms, n_slabs,
                phonon.supercell.masses, vertex_cutoff=vertex_cutoff)
        fc3_dev_mw = device_fc3_mass_weighted(vertices[(0, 0)], n_slabs, n_dof)
        optical_projector = build_dynamical_zero_mode_projector(
            H_00_g, H_01_g, n_slabs)
        static_se_hook = build_static_self_energy_hook(
            dw_thz=dw_thz, n_dof=n_dof, n_slabs=n_slabs,
            fc3_dev_mw=fc3_dev_mw, fc4_dev_mw=fc4_dev_mw,
            use_loop=loop, use_tadpole=tadpole,
            optical_projector=optical_projector)
        # The static loop/tadpole self-energy is carried in the fixed-point
        # vector; linear mixing converges it slowly, so default to the
        # safeguarded Anderson accelerator (with the loosened static step cap)
        # unless the caller chose a solver explicitly.
        if solver is None and not anderson_mixing:
            solver = "anderson"
        if verbose:
            print(f"  Static self-energy: loop={loop}, tadpole={tadpole}, "
                  f"propagator={loop_propagator}, "
                  f"stage_loop_first={stage_loop_first}, solver={solver}")

    scba_result = scba_loop(
        z2_arr=z2_arr, freqs_thz=freqs_thz, dw_thz=dw_thz,
        omega_rad=omega_rad, pos_mask=pos_mask,
        n_slabs=n_slabs, n_dof=n_dof, N_D=N_D,
        H_D_list=H_D_all, obc_list=obc_all, btd_blocks_list=btd_blocks,
        n_kpts=n_kpts, se_kernel=se_kernel, T_L=T_L, T_R=T_R,
        max_scba_iter=max_scba_iter, scba_tol=scba_tol,
        conservation_tol=conservation_tol,
        mixing=mixing, anderson_mixing=anderson_mixing,
        anderson_depth=anderson_depth, anderson_step_cap=anderson_step_cap,
        scattering_contacts=scattering_contacts, retarded=retarded,
        verbose=verbose, solver=solver, anderson_safeguard=anderson_safeguard,
        zero_mode_projection=zero_mode_projection,
        gate_on_conservation=gate_on_conservation,
        divergence_guard=divergence_guard,
        masses_primitive=phonon.primitive.masses,
        causality_projection=causality_projection,
        track_diagnostics=track_diagnostics,
        static_se_hook=static_se_hook,
        static_mixing=static_mixing,
        loop_propagator=loop_propagator,
        stage_loop_first=stage_loop_first,
    )

    spectral_J_L = scba_result["spectral_J_L"]
    spectral_J_R = scba_result["spectral_J_R"]
    Sigma_R = scba_result["Sigma_R"]
    Sigma_l = scba_result["Sigma_l"]
    Sigma_g = scba_result["Sigma_g"]
    conservation_err = scba_result["conservation_err"]
    convergence_history = scba_result["convergence_history"]

    spectral_J_anh = 0.5 * (spectral_J_L + spectral_J_R)
    J_anh_total = np.sum(spectral_J_anh[pos_mask]) * dw_thz * 1e12
    G_anh = J_anh_total / (A_c * delta_T) if delta_T != 0 else 0.0

    if verbose:
        print(f"  Anharmonic conductance: {G_anh:.2f} W/(m^2 K); "
              f"heat-flow conservation: {conservation_err:.4e}")

    # Finite path keeps its historical self-energy shape (no q axis); the
    # q-resolved path carries the leading q axis.
    if n_kpts == 1:
        se_r = Sigma_R[0, pos_mask]
        se_l = Sigma_l[0, pos_mask]
        se_g = Sigma_g[0, pos_mask]
    else:
        se_r = Sigma_R[:, pos_mask]
        se_l = Sigma_l[:, pos_mask]
        se_g = Sigma_g[:, pos_mask]

    result = {
        "freqs_thz": freqs_thz[pos_mask],
        "omega_rad": freqs_thz[pos_mask] * THZ_TO_RAD,
        "transmission_ballistic": trans_ballistic[pos_mask],
        "spectral_heat_current_ballistic": spectral_J_ball[pos_mask],
        "spectral_heat_current": spectral_J_anh[pos_mask],
        "spectral_heat_current_L": spectral_J_L[pos_mask].copy(),
        "spectral_heat_current_R": spectral_J_R[pos_mask].copy(),
        "heat_current_ballistic": J_ball_total,
        "heat_current": J_anh_total,
        "thermal_conductance_ballistic": G_ball,
        "thermal_conductance_anharmonic": G_anh,
        "heat_flow_conservation": conservation_err,
        "delta_T": delta_T,
        "n_scba_iterations": len(convergence_history) + 1,
        "convergence_history": convergence_history,
        "scba_converged": scba_result.get("converged", True),
        "scba_residual": scba_result.get("scba_residual", float("nan")),
        "diagnostics_history": scba_result.get("diagnostics_history", []),
        "self_energy_retarded": se_r,
        "self_energy_lesser": se_l,
        "self_energy_greater": se_g,
    }

    if static_se_hook is not None:
        # Converged static (loop+tadpole) self-energy [THz^2] and the effective
        # dynamical matrix Phi_eff = Phi + Sigma_static (Gamma device).
        sig_static = scba_result.get("Sigma_static")
        result["sigma_static"] = None if sig_static is None else sig_static[0]
        result["phi_eff"] = (None if sig_static is None
                             else (H_D_all[0] + sig_static[0]))
        result["static_residual"] = scba_result.get(
            "static_residual", float("nan"))

    if n_kpts == 1 and verbose:
        _check_q11_vs_finite(result)
    return result


def transmission_finite(phonon, fc3_hdf5=None, **kwargs) -> dict:
    """Gamma-only finite device: :func:`transmission` with ``q_mesh=(1,1)``."""
    return transmission(phonon, fc3_hdf5=fc3_hdf5, q_mesh=(1, 1), **kwargs)


def transmission_q(
    phonon,
    fc3_hdf5=None,
    q_mesh_transverse: tuple[int, int] = (4, 4),
    *,
    retarded: str = "half",
    zero_mode_projection: bool = False,
    **kwargs,
) -> dict:
    """Transversely-periodic device: :func:`transmission` on ``q_mesh``.

    Keeps the historical q-path defaults (``retarded="half"``, no zero-mode
    projection). ``q_mesh_transverse=(1,1)`` reproduces
    :func:`transmission_finite`; pass ``sigma_cutoff=0`` for Guo's cheap
    slab-diagonal approximation (III).
    """
    return transmission(
        phonon, fc3_hdf5=fc3_hdf5, q_mesh=q_mesh_transverse,
        retarded=retarded, zero_mode_projection=zero_mode_projection,
        **kwargs)


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------


def _check_q11_vs_finite(res_q, rtol=5e-3, atol=1e-8):
    """Warn if q=(1,1) output is internally inconsistent.

    This checks the q-path result against itself (ballistic should match
    the analytic Gamma-only result). For a full q-vs-finite check, run both
    entry points and call :func:`compare_q11_to_finite`.
    """
    G_ball = res_q["thermal_conductance_ballistic"]
    G_anh = res_q["thermal_conductance_anharmonic"]
    if G_ball != 0 and abs(G_anh) > 2 * abs(G_ball):
        warnings.warn(
            f"q=(1,1) sanity: G_anh ({G_anh:.2e}) > 2 * G_ball ({G_ball:.2e})"
        )


def compare_q11_to_finite(res_q, res_f, rtol=5e-3, atol=1e-8):
    """Verify that ``q=(1,1)`` matches the finite (Gamma-only) solver.

    Raises ``AssertionError`` on mismatch.
    """
    keys = [
        "heat_current",
        "thermal_conductance_anharmonic",
        "thermal_conductance_ballistic",
        "heat_flow_conservation",
        "transmission_ballistic",
        "spectral_heat_current",
    ]
    for key in keys:
        a = np.asarray(res_q[key])
        b = np.asarray(res_f[key])
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            max_rel = float(np.max(np.abs(a - b) / (np.abs(b) + atol)))
            raise AssertionError(
                f"q=(1,1) vs finite mismatch in '{key}': "
                f"max rel err = {max_rel:.2e}"
            )
