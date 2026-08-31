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

* bubble half: the bubble is a homogeneous quadratic map, so its
  derivative is the mixed-leg (2PI-kernel, cut-line) contraction::

      S'(G) dG = B(dG, G) + B(G, dG)

  evaluated by ``SigmaPhononPhonon.compute_linearized`` (the default
  ``"bilinear"`` route -- no subtraction of large terms, uniformly
  exact to rounding), or equivalently via the polarisation identity::

      S'(G) dG = S(G + dG) - S(G) - S(dG)

  through three calls to the unmodified production kernel (the
  ``"polarization"`` route, kept as an independent cross-check; it
  loses ~|S(G)|/|cross| digits on very small directions).

The RGF selected solve is not the plain congruence for arbitrary
inputs: it reads only the diagonal + upper Sigma^{<,>} blocks
(substituting ``Sigma_ji -> -Sigma_ij^H``) and projects its outputs
onto the skew-hermitian banded subspace. That subspace is invariant
under the map and contains the residual, so the Newton-Krylov solve is
run inside it: every direction's ``dSigma^{<,>}`` is projected onto it,
where the plain dense identities coincide with the implemented map
(verified in ``phonon/studies/_jvp_validate.py``).

The frozen ``G^R, G^{<,>}`` are reconstructed densely per rank-local
frequency and transverse momentum once per Newton step (the systems this
targets are small, N <= a few hundred DOF).  The frozen reconstruction remains
on the host because it is performed once per Newton step.  Its matrices are
then copied once to the active array backend, so every JVP's dense Dyson
products and the following production bubble remain on the GPU when CuPy is
selected.
"""

from __future__ import annotations

import numpy as np

from qttools import xp
from qttools.comm import comm
from qttools.utils.gpu_utils import get_any_location

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
    jvp_form : {"bilinear", "polarization"}
        Evaluation route of the bubble derivative (see module
        docstring). The bilinear route needs the symmetry fast paths
        off; otherwise it falls back to polarization with a notice.
    """

    def __init__(self, scba, recon_check_tol: float = 1e-8,
                 jvp_form: str = "bilinear") -> None:
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
        _forbid(comm.block.size > 1, "block_comm_size == 1")
        _forbid(int(getattr(ph, "sse_fold_verify_iterations", 0)) > 0,
                "sse_fold_verify_iterations == 0")
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
        _forbid(getattr(g, "q_section_offsets", None) is not None,
                "replicated transverse-q storage (q_comm_size == 1 or "
                "q_distributed == false)")
        self._stack_shape = tuple(int(n) for n in g.data.shape[:-1])
        self._n_batch = int(np.prod(self._stack_shape, dtype=np.int64))
        self._nq = int(np.prod(g.global_stack_shape[1:], dtype=np.int64))

        if jvp_form not in ("bilinear", "polarization"):
            raise ValueError(f"Unknown jvp_form={jvp_form!r}.")
        self._bilinear_supported = (
            self._nq == 1
            and getattr(self._sse, "_vfactors", None) is None
            and not bool(getattr(ph, "sse_greater_from_lesser", False))
            and not bool(getattr(ph, "sse_hermitian_pairs", False))
        )
        if jvp_form == "bilinear" and not self._bilinear_supported:
            # compute_linearized deliberately implements only the plain dense
            # Gamma path.  The polarisation identity calls the unmodified
            # production kernel, so it already covers coupled q, factored
            # vertices and the symmetry fast paths without duplicating their
            # product rules here.
            reasons = []
            if self._nq != 1:
                reasons.append(f"coupled q (nq={self._nq})")
            if getattr(self._sse, "_vfactors", None) is not None:
                reasons.append("factored vertices")
            if bool(getattr(ph, "sse_greater_from_lesser", False)):
                reasons.append("greater-from-lesser")
            if bool(getattr(ph, "sse_hermitian_pairs", False)):
                reasons.append("Hermitian-pair fast path")
            if comm.rank == 0:
                print(
                    "PhononJVP: " + ", ".join(reasons)
                    + " -> falling back to the polarization JVP route.",
                    flush=True,
                )
            jvp_form = "polarization"
        self.jvp_form = jvp_form

        rows = getattr(g, "rows", None)
        cols = getattr(g, "cols", None)
        if rows is None or cols is None:
            raise NotImplementedError(
                "mixing_method='newton' needs the sparsity rows/cols "
                "(DSDBCOO-style) to gather/scatter banded blocks."
            )
        self._rows = self._host(rows)
        self._cols = self._host(cols)
        self._rows_device = xp.asarray(self._rows)
        self._cols_device = xp.asarray(self._cols)

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
        self._bt_mask_device = xp.asarray(self._bt_mask)
        # The RGF writes G only on its output band (= sse_g_band: 1 =
        # block-tridiagonal, 2 = + second off-diagonal, 3 = + third); pattern
        # slots beyond it -- present when the cutoff makes the pattern
        # block-dense -- stay zero in the production buffers and must stay
        # zero in the JVP's dG too.
        out_band = int(getattr(self._solver, "_gf_band", 1))
        self._g_mask = np.abs(blk_r - blk_c) <= out_band
        self._g_mask_device = xp.asarray(self._g_mask)

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
        self._GR_device = None
        self._GA_device = None
        self._GL_device = None
        self._GG_device = None
        self._s_base = None      # (S_l, S_g, S_r_kk) flat complex
        self._g_l_flat = None
        self._g_g_flat = None
        self._g_l_flat_device = None
        self._g_g_flat_device = None
        # Backwards-compatible name used by the validation harness.  It now
        # counts every local stack element, not frequencies alone.
        self._n_local = self._n_batch
        self._nnz = int(self._rows.size)

    # ------------------------------------------------------------------
    # small dense-block helpers
    # ------------------------------------------------------------------
    def _sl(self, i: int) -> slice:
        return slice(int(self._block_offsets[i]),
                     int(self._block_offsets[i + 1]))

    @staticmethod
    def _host(a) -> np.ndarray:
        """Return a host view/copy of a NumPy or CuPy array."""
        return np.asarray(get_any_location(
            a, "numpy", use_pinned_memory=(xp.__name__ == "cupy")))

    def _stack_block(self, block) -> np.ndarray:
        """Broadcast a block's leading axes over this rank-local stack."""
        a = self._host(block)
        shape = self._stack_shape + a.shape[-2:]
        try:
            return np.broadcast_to(a, shape).reshape(
                (self._n_batch,) + a.shape[-2:])
        except ValueError as exc:
            raise RuntimeError(
                f"Cannot broadcast block shape {a.shape} over local JVP "
                f"stack {self._stack_shape}."
            ) from exc

    def _dynamical_matrix_dense(self) -> np.ndarray:
        """Dense q-resolved BTD dynamical matrix over the local stack."""
        out = np.zeros((self._n_batch, self._N, self._N),
                       dtype=np.complex128)
        D = self._solver.dynamical_matrix
        for i in range(self._nb):
            for j in range(max(0, i - 1), min(self._nb, i + 2)):
                out[:, self._sl(i), self._sl(j)] = self._stack_block(
                    D.blocks[i, j])
        return out

    def _to_dense(self, flat: np.ndarray, bt_only: bool, *,
                  device: bool = False) -> np.ndarray:
        """(*stack, nnz) pattern data -> dense (n_batch, N, N)."""
        lib = xp if device else np
        rows = self._rows_device if device else self._rows
        cols = self._cols_device if device else self._cols
        flat = lib.asarray(flat).reshape((-1, self._nnz))
        out = lib.zeros((flat.shape[0], self._N, self._N),
                        dtype=lib.complex128)
        if bt_only:
            m = self._bt_mask_device if device else self._bt_mask
            out[:, rows[m], cols[m]] = flat[:, m]
        else:
            out[:, rows, cols] = flat
        return out

    def _to_flat(self, dense: np.ndarray, *, device: bool = False) -> np.ndarray:
        """Dense (n_local, N, N) -> (n_local, nnz) pattern data."""
        rows = self._rows_device if device else self._rows
        cols = self._cols_device if device else self._cols
        return dense[:, rows, cols]

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
        stack_shape = tuple(int(n) for n in data.sigma_lesser.data.shape[:-1])
        if stack_shape != self._stack_shape:
            raise RuntimeError(
                "PhononJVP stack shape changed after construction: "
                f"{self._stack_shape} -> {stack_shape}."
            )
        n_local = self._n_batch
        self._n_local = n_local
        N = self._N

        w = self._host(solver.local_frequencies).astype(float, copy=False)
        z2 = w**2
        if not self._stack_shape or self._stack_shape[0] != w.size:
            raise RuntimeError(
                f"Local frequency count {w.size} is incompatible with JVP "
                f"stack shape {self._stack_shape}."
            )
        z2 = np.broadcast_to(
            z2.reshape((w.size,) + (1,) * (len(self._stack_shape) - 1)),
            self._stack_shape,
        ).reshape(n_local)

        sr = self._to_dense(self._host(
            data.sigma_retarded_hermitian_prev.data), bt_only=True)
        A = -sr
        A -= self._dynamical_matrix_dense()
        idx = np.arange(N)
        A[:, idx, idx] += z2[:, None]

        obc = solver.obc_blocks
        b0 = self._sl(0)
        bN = self._sl(self._nb - 1)
        if obc.retarded[0] is not None:
            A[:, b0, b0] -= self._stack_block(obc.retarded[0])
        # A one-block device stores the SUM of the two reservoir self-energies
        # in the sole OBC slot (PhononSolver._compute_obc).  Index 0 and -1
        # then alias and must not be applied twice.
        if self._nb > 1 and obc.retarded[-1] is not None:
            A[:, bN, bN] -= self._stack_block(obc.retarded[-1])

        GR = np.linalg.inv(A) if n_local else A.copy()
        GA = GR.conj().swapaxes(-2, -1)

        def _source(buf, key):
            S = self._sub_lower(self._to_dense(self._host(buf.data),
                                               bt_only=True))
            corner0 = getattr(obc, key)[0]
            cornerN = getattr(obc, key)[-1]
            if corner0 is not None:
                S[:, b0, b0] += self._stack_block(corner0)
            if self._nb > 1 and cornerN is not None:
                S[:, bN, bN] += self._stack_block(cornerN)
            return S

        GL = GR @ _source(data.sigma_lesser_prev, "lesser") @ GA
        GG = GR @ _source(data.sigma_greater_prev, "greater") @ GA

        self._GR, self._GA, self._GL, self._GG = GR, GA, GL, GG
        self._GR_device = xp.asarray(GR)
        self._GA_device = xp.asarray(GA)
        self._GL_device = xp.asarray(GL)
        self._GG_device = xp.asarray(GG)

        # Reconstruction self-check against the solver's actual output
        # (catches any forgotten A-term before it corrupts a Newton step).
        self._g_l_flat = self._host(data.g_lesser.data).reshape(
            n_local, self._nnz).copy()
        self._g_g_flat = self._host(data.g_greater.data).reshape(
            n_local, self._nnz).copy()
        self._g_l_flat_device = xp.asarray(self._g_l_flat)
        self._g_g_flat_device = xp.asarray(self._g_g_flat)
        num2 = den2 = 0.0
        gm = self._g_mask
        for dense, flat in ((GL, self._g_l_flat), (GG, self._g_g_flat)):
            got = self._to_flat(self._skew_project(dense))
            num2 += float(np.linalg.norm(got[:, gm] - flat[:, gm]) ** 2)
            den2 += float(np.linalg.norm(flat[:, gm]) ** 2)
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
        s_l = self._host(data.sigma_lesser.data).ravel().copy()
        s_g = self._host(data.sigma_greater.data).ravel().copy()
        s_r = self._host(data.sigma_retarded_hermitian.data).ravel().copy()
        s_r -= 0.5 * (s_l - s_g)
        self._s_base = (s_l, s_g, s_r)
        return recon

    # ------------------------------------------------------------------
    # Jacobian-vector product
    # ------------------------------------------------------------------
    def apply(self, dx: np.ndarray, form: str | None = None) -> np.ndarray:
        """Return ``J_F dx`` for a flat complex direction ``dx`` in the
        mixer layout ``[dSigma^<, dSigma^>, dSigma^R]`` (rank-local).

        ``form`` overrides the configured JVP route for this call
        (used by the A/B validation to compare both)."""
        if self._GR is None:
            raise RuntimeError("PhononJVP.apply() before prepare().")
        form = self.jvp_form if form is None else form
        if form not in ("bilinear", "polarization"):
            raise ValueError(f"Unknown JVP form {form!r}.")
        if form == "bilinear" and not self._bilinear_supported:
            raise NotImplementedError(
                "The bilinear JVP is available only for Gamma-only dense "
                "vertices without symmetry fast paths; use polarization."
            )
        n_local, nnz = self._n_local, self._nnz
        size = n_local * nnz
        device = xp.__name__ == "cupy"
        work = xp.asarray(dx) if device else np.asarray(dx)
        dl = work[:size].reshape(n_local, nnz)
        dg = work[size:2 * size].reshape(n_local, nnz)
        dr = work[2 * size:].reshape(n_local, nnz)

        # Dyson half: projected onto the invariant skew subspace, plain
        # dense identities, RGF output projections.
        dl_d = self._skew_project(self._to_dense(
            dl, bt_only=True, device=device))
        dg_d = self._skew_project(self._to_dense(
            dg, bt_only=True, device=device))
        dr_d = self._to_dense(dr, bt_only=True, device=device)
        dr_dH = dr_d.conj().swapaxes(-2, -1)

        if device:
            GR, GA = self._GR_device, self._GA_device
            GL, GG = self._GL_device, self._GG_device
            g_l_flat = self._g_l_flat_device
            g_g_flat = self._g_g_flat_device
            g_mask = self._g_mask_device
        else:
            GR, GA, GL, GG = self._GR, self._GA, self._GL, self._GG
            g_l_flat = self._g_l_flat
            g_g_flat = self._g_g_flat
            g_mask = self._g_mask
        GRdr = GR @ dr_d
        dGl = GR @ dl_d @ GA + GRdr @ GL + GL @ dr_dH @ GA
        dGg = GR @ dg_d @ GA + GRdr @ GG + GG @ dr_dH @ GA
        dGl_flat = self._to_flat(
            self._skew_project(dGl), device=device) * g_mask
        dGg_flat = self._to_flat(
            self._skew_project(dGg), device=device) * g_mask

        # Bubble half.
        if form == "bilinear":
            # Mixed-leg cross through compute_linearized: one kernel call,
            # frozen legs read straight from the driver's live G buffers.
            dS_l, dS_g, dS_r = self._kernel_linearized(dGl_flat, dGg_flat)
        else:
            # Polarisation identity: two production-kernel calls plus the
            # cached S(G) from the driver's own SSE output.
            s1 = self._kernel(g_l_flat + dGl_flat,
                              g_g_flat + dGg_flat)
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
        return [self._host(m.data).ravel().copy() for m in self._out]

    def _kernel_linearized(self, dgl_flat: np.ndarray,
                           dgg_flat: np.ndarray):
        """One mixed-leg linearized bubble evaluation on scratch buffers."""
        self._in_l.data[:] = xp.asarray(
            dgl_flat.reshape(self._in_l.data.shape))
        self._in_g.data[:] = xp.asarray(
            dgg_flat.reshape(self._in_g.data.shape))
        for m in self._out:
            m.data[:] = 0.0
        self._sse.compute_linearized(
            self._data.g_lesser, self._data.g_greater,
            self._in_l, self._in_g, out=self._out,
        )
        return [self._host(m.data).ravel().copy() for m in self._out]
