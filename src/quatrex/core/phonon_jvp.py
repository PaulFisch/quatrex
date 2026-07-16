# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Exact analytic Jacobian-vector products for the phonon SCBA map.

The SCBA fixed-point map is ``F(x) = postop(S(D(x)))`` with
``x = [Sigma^<, Sigma^>, Sigma^R]``: the Dyson solve ``D`` (RGF selected
solve at fixed contacts), the bubble SSE ``S`` (homogeneous R-quadratic
in the Green's function), and the driver post-op
``Sigma^R += 0.5*(Sigma^< - Sigma^>)``. Its exact directional
derivative therefore needs no finite differencing:

* Dyson half (frozen G, fixed contact self-energies)::

      dG^R      = G^R dSigma^R G^R
      dG^{<,>}  = G^R dSigma^{<,>} G^A + G^R dSigma^R G^{<,>}
                  + G^{<,>} (dSigma^R)^H G^A

* bubble half via the polarisation identity of a quadratic map::

      S'(G) dG = S(G + dG) - S(G) - S(dG)

  i.e. two extra calls to the UNMODIFIED production kernel per
  direction (``S(G)`` is the SSE output the driver already has).

The RGF selected solve is not the plain congruence for arbitrary
inputs: it reads only the diagonal + upper Sigma^{<,>} blocks
(substituting ``Sigma_ji -> -Sigma_ij^H``) and projects its outputs
onto the skew-hermitian banded subspace. That subspace is invariant
under the map and contains the residual, so the Newton-Krylov solve is
run inside it: every direction's ``dSigma^{<,>}`` is projected onto it,
where the plain dense identities coincide with the implemented map
(verified in ``phonon/studies/_jvp_validate.py``).

The frozen ``G^R, G^{<,>}`` are reconstructed densely per rank-local
frequency once per Newton step (the systems this targets are small,
N <= a few hundred DOF); each JVP's Dyson part is then a handful of
dense GEMMs per frequency, negligible against the two bubble calls.
"""

from __future__ import annotations

import numpy as np

from qttools import xp
from qttools.comm import comm

__all__ = ["PhononJVP"]


class PhononJVP:
    """Exact JVP context for the phonon SCBA, owned by the SCBA driver.

    Built lazily at the first Newton step (the mixer exists before the
    phonon solver does). ``prepare()`` freezes the Green's function at
    the current iterate; ``apply(dx)`` returns ``J_F dx`` for a
    flattened complex direction in the mixer's vector layout.

    Parameters
    ----------
    scba : SCBA
        The driver. Must be a phonon simulation with a stationary,
        contact-independent map (see the guard list in ``__init__``).
    recon_check_tol : float
        Relative tolerance for the reconstruction self-check of the
        frozen dense G against the solver's actual RGF output.
    """

    def __init__(self, scba, recon_check_tol: float = 1e-8) -> None:
        config = scba.config
        ph = config.phonon

        # ---- guards: the map must be iteration-independent and the
        # contacts Sigma-independent for the frozen-G linearisation. ----
        def _forbid(cond: bool, what: str) -> None:
            if cond:
                raise NotImplementedError(
                    f"mixing_method='newton' (exact JVP) requires {what}."
                )

        _forbid(config.simulation_type != "phonon",
                "simulation_type == 'phonon'")
        _forbid(xp.__name__ != "numpy", "the numpy backend")
        _forbid(comm.block.size > 1, "block_comm_size == 1")
        _forbid(int(np.prod([k for k in config.device.kpoint_grid
                             if k > 1])) > 1, "a Gamma-only device (nq == 1)")
        _forbid(int(getattr(ph, "sse_ramp_iterations", 0)) > 0,
                "sse_ramp_iterations == 0 (the ramp counter advances per "
                "kernel call and would desynchronise the polarisation "
                "evaluations)")
        _forbid(int(getattr(ph, "sse_fold_verify_iterations", 0)) > 0,
                "sse_fold_verify_iterations == 0")
        _forbid(int(getattr(ph, "eta_ramp_iterations", 0)) > 0,
                "eta_ramp_iterations == 0 (frozen A)")
        _forbid(int(getattr(ph, "eta_obc_ramp_iterations", 0)) > 0,
                "eta_obc_ramp_iterations == 0 (frozen contacts)")
        _forbid(int(getattr(ph, "eta_ir_floor_ramp_iterations", 0)) > 0,
                "eta_ir_floor_ramp_iterations == 0 (frozen A)")
        _forbid(bool(getattr(ph, "buttiker_probe", False)),
                "buttiker_probe == false (G-dependent probe source is not "
                "differentiated)")
        _forbid(bool(getattr(ph, "scp_tadpole", False)),
                "scp_tadpole == false (the static tadpole mutates per "
                "kernel call and is not quadratic in G)")
        _forbid(bool(getattr(ph, "obc_scattering_contacts", False)),
                "obc_scattering_contacts == false (Sigma-independent leads)")
        _forbid(str(ph.solver.algorithm) != "rgf",
                "phonon.solver.algorithm == 'rgf' (the 'inv' path has "
                "different off-band Jacobian semantics)")

        self._scba = scba
        self._solver = scba.phonon_solver
        self._sse = scba._phonon_phonon_interaction.sigma_phonon_phonon
        self._data = scba.data
        self.recon_check_tol = float(recon_check_tol)

        g = self._data.g_lesser
        rows = getattr(g, "rows", None)
        cols = getattr(g, "cols", None)
        if rows is None or cols is None:
            raise NotImplementedError(
                "mixing_method='newton' needs the sparsity rows/cols "
                "(DSDBCOO-style) to gather/scatter banded blocks."
            )
        self._rows = np.asarray(rows)
        self._cols = np.asarray(cols)

        self._block_sizes = np.asarray(self._solver.block_sizes, dtype=int)
        self._block_offsets = np.hstack(([0], np.cumsum(self._block_sizes)))
        self._nb = int(self._block_sizes.size)
        self._N = int(self._block_offsets[-1])

        blk_r = np.searchsorted(self._block_offsets, self._rows,
                                side="right") - 1
        blk_c = np.searchsorted(self._block_offsets, self._cols,
                                side="right") - 1
        # Only the block-tridiagonal band of Sigma enters the Dyson solve
        # (system-matrix subtraction and the RGF source reads); the d2
        # pattern slots (present when sse_g_band = 2) carry J == 0.
        self._bt_mask = np.abs(blk_r - blk_c) <= 1

        # Dense block-tridiagonal dynamical matrix (what _btd_subtract
        # actually subtracts), assembled once.
        D = self._solver.dynamical_matrix
        Dd = np.zeros((self._N, self._N), dtype=np.complex128)
        for i in range(self._nb):
            for j in range(max(0, i - 1), min(self._nb, i + 2)):
                blk = np.asarray(D.blocks[i, j])
                blk = blk.reshape((-1,) + blk.shape[-2:])[0]
                Dd[self._sl(i), self._sl(j)] = blk
        self._D_dense = Dd

        # Scratch buffers: two kernel inputs (G pattern) and three kernel
        # outputs (Sigma pattern) -- reused across all JVPs.
        dsdbsparse_type = config.compute.dsdbsparse_type

        def _scratch():
            m = dsdbsparse_type.empty_like(g)
            m.allocate_data()
            m.data[:] = 0.0
            return m

        self._in_l, self._in_g = _scratch(), _scratch()
        self._out = (_scratch(), _scratch(), _scratch())

        # Frozen per-Newton-step state.
        self._GR = None
        self._GA = None
        self._GL = None
        self._GG = None
        self._s_base = None      # (S_l, S_g, S_r_kk) flat complex
        self._g_l_flat = None
        self._g_g_flat = None
        self._n_local = 0
        self._nnz = int(self._rows.size)

    # ------------------------------------------------------------------
    # small dense-block helpers
    # ------------------------------------------------------------------
    def _sl(self, i: int) -> slice:
        return slice(int(self._block_offsets[i]),
                     int(self._block_offsets[i + 1]))

    def _to_dense(self, flat: np.ndarray, bt_only: bool) -> np.ndarray:
        """(n_local, nnz) pattern data -> dense (n_local, N, N)."""
        out = np.zeros((flat.shape[0], self._N, self._N),
                       dtype=np.complex128)
        if bt_only:
            m = self._bt_mask
            out[:, self._rows[m], self._cols[m]] = flat[:, m]
        else:
            out[:, self._rows, self._cols] = flat
        return out

    def _to_flat(self, dense: np.ndarray) -> np.ndarray:
        """Dense (n_local, N, N) -> (n_local, nnz) pattern data."""
        return dense[:, self._rows, self._cols]

    def _sub_lower(self, dense: np.ndarray) -> np.ndarray:
        """RGF input substitution: lower blocks <- -(upper)^H, diag kept."""
        out = dense.copy()
        for i in range(self._nb):
            si = self._sl(i)
            for j in range(i + 1, min(self._nb, i + 2)):
                sj = self._sl(j)
                out[:, sj, si] = -out[:, si, sj].conj().swapaxes(-2, -1)
        return out

    def _skew_project(self, dense: np.ndarray) -> np.ndarray:
        """Project onto the skew-hermitian banded subspace: diagonal blocks
        0.5*(d - d^H), every lower block mirrored from its upper."""
        out = dense.copy()
        for i in range(self._nb):
            si = self._sl(i)
            d = out[:, si, si]
            out[:, si, si] = 0.5 * (d - d.conj().swapaxes(-2, -1))
            for j in range(i + 1, self._nb):
                sj = self._sl(j)
                out[:, sj, si] = -out[:, si, sj].conj().swapaxes(-2, -1)
        return out

    # ------------------------------------------------------------------
    # per-Newton-step preparation
    # ------------------------------------------------------------------
    def prepare(self) -> float:
        """Freeze the dense Green's function at the current iterate.

        Reads the driver's buffers directly: ``sigma_*_prev`` hold the
        iterate x, ``sigma_*`` hold the raw map value F-components, and
        ``g_lesser/g_greater`` hold D(x) from the solve of this
        iteration. Returns the reconstruction self-check residual.
        """
        data = self._data
        solver = self._solver
        n_local = int(np.asarray(data.sigma_lesser.data).shape[0])
        self._n_local = n_local
        N = self._N

        w = np.asarray(solver.local_frequencies, dtype=float)
        z2 = w**2 + 2j * float(solver.eta) * np.abs(w)
        if getattr(solver, "_ir_floor_diag", None) is not None:
            z2 = z2 + np.asarray(solver._ir_floor_diag)

        sr = self._to_dense(np.asarray(data.sigma_retarded_hermitian_prev
                                       .data), bt_only=True)
        A = -sr
        A -= self._D_dense[None]
        idx = np.arange(N)
        A[:, idx, idx] += z2[:, None]

        obc = solver.obc_blocks
        b0 = self._sl(0)
        bN = self._sl(self._nb - 1)
        if obc.retarded[0] is not None:
            A[:, b0, b0] -= np.asarray(obc.retarded[0])
        if obc.retarded[-1] is not None:
            A[:, bN, bN] -= np.asarray(obc.retarded[-1])

        GR = np.linalg.inv(A) if n_local else A.copy()
        GA = GR.conj().swapaxes(-2, -1)

        def _source(buf, key):
            S = self._sub_lower(self._to_dense(np.asarray(buf.data),
                                               bt_only=True))
            corner0 = getattr(obc, key)[0]
            cornerN = getattr(obc, key)[-1]
            if corner0 is not None:
                S[:, b0, b0] += np.asarray(corner0)
            if cornerN is not None:
                S[:, bN, bN] += np.asarray(cornerN)
            return S

        GL = GR @ _source(data.sigma_lesser_prev, "lesser") @ GA
        GG = GR @ _source(data.sigma_greater_prev, "greater") @ GA

        self._GR, self._GA, self._GL, self._GG = GR, GA, GL, GG

        # Reconstruction self-check against the solver's actual output
        # (catches any forgotten A-term before it corrupts a Newton step).
        self._g_l_flat = np.asarray(data.g_lesser.data).copy()
        self._g_g_flat = np.asarray(data.g_greater.data).copy()
        num2 = den2 = 0.0
        for dense, flat in ((GL, self._g_l_flat), (GG, self._g_g_flat)):
            got = self._to_flat(self._skew_project(dense))
            num2 += float(np.linalg.norm(got - flat) ** 2)
            den2 += float(np.linalg.norm(flat) ** 2)
        # Global relative norm (a rank with pathological frequencies must
        # not silently poison the Krylov space).
        from quatrex.core.mpi_linalg import allreduce_sum, get_comm
        mpi_comm, op_sum = get_comm()
        buf = allreduce_sum(mpi_comm, op_sum,
                            np.array([num2, den2], dtype=np.float64))
        recon = float(np.sqrt(buf[0] / max(buf[1], 1e-300)))
        if recon > self.recon_check_tol:
            raise RuntimeError(
                f"PhononJVP reconstruction self-check failed: dense frozen "
                f"G differs from the RGF output by rel {recon:.3e} "
                f"(tol {self.recon_check_tol:.1e}). The reassembled system "
                "matrix does not match the solver's."
            )

        # S(G_frozen): the raw kernel output is exactly what sits in the
        # sigma buffers, except that the driver has already added the
        # skew part 0.5*(S_l - S_g) into sigma_retarded_hermitian.
        s_l = np.asarray(data.sigma_lesser.data).ravel().copy()
        s_g = np.asarray(data.sigma_greater.data).ravel().copy()
        s_r = np.asarray(data.sigma_retarded_hermitian.data).ravel().copy()
        s_r -= 0.5 * (s_l - s_g)
        self._s_base = (s_l, s_g, s_r)
        return recon

    # ------------------------------------------------------------------
    # Jacobian-vector product
    # ------------------------------------------------------------------
    def apply(self, dx: np.ndarray) -> np.ndarray:
        """Return ``J_F dx`` for a flat complex direction ``dx`` in the
        mixer layout ``[dSigma^<, dSigma^>, dSigma^R]`` (rank-local)."""
        if self._GR is None:
            raise RuntimeError("PhononJVP.apply() before prepare().")
        n_local, nnz = self._n_local, self._nnz
        size = n_local * nnz
        dl = dx[:size].reshape(n_local, nnz)
        dg = dx[size:2 * size].reshape(n_local, nnz)
        dr = dx[2 * size:].reshape(n_local, nnz)

        # Dyson half: projected onto the invariant skew subspace, plain
        # dense identities, RGF output projections.
        dl_d = self._skew_project(self._to_dense(dl, bt_only=True))
        dg_d = self._skew_project(self._to_dense(dg, bt_only=True))
        dr_d = self._to_dense(dr, bt_only=True)
        dr_dH = dr_d.conj().swapaxes(-2, -1)

        GR, GA, GL, GG = self._GR, self._GA, self._GL, self._GG
        GRdr = GR @ dr_d
        dGl = GR @ dl_d @ GA + GRdr @ GL + GL @ dr_dH @ GA
        dGg = GR @ dg_d @ GA + GRdr @ GG + GG @ dr_dH @ GA
        dGl_flat = self._to_flat(self._skew_project(dGl))
        dGg_flat = self._to_flat(self._skew_project(dGg))

        # Bubble half: polarisation identity, two production-kernel calls.
        s1 = self._kernel(self._g_l_flat + dGl_flat,
                          self._g_g_flat + dGg_flat)
        s2 = self._kernel(dGl_flat, dGg_flat)

        s_l0, s_g0, s_r0 = self._s_base
        dS_l = s1[0] - s_l0 - s2[0]
        dS_g = s1[1] - s_g0 - s2[1]
        dS_r = s1[2] - s_r0 - s2[2]
        # Driver post-op on the deltas (scba.py: Sigma^R += 0.5*(S_l-S_g)).
        dF_r = dS_r + 0.5 * (dS_l - dS_g)
        return np.concatenate([dS_l, dS_g, dF_r])

    def _kernel(self, gl_flat: np.ndarray, gg_flat: np.ndarray):
        """One production bubble evaluation on scratch buffers."""
        self._in_l.data[:] = xp.asarray(
            gl_flat.reshape(self._in_l.data.shape))
        self._in_g.data[:] = xp.asarray(
            gg_flat.reshape(self._in_g.data.shape))
        for m in self._out:
            m.data[:] = 0.0
        self._sse.compute(self._in_l, self._in_g, out=self._out)
        return [np.asarray(m.data).ravel().copy() for m in self._out]
