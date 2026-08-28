# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Passive rational auxiliary states for the phonon SCBA.

This module is the production-side algebra needed by the enriched-frequency
proposal in ``phonon/docs/phph_acceleration_review.md``.  A rational Keldysh
channel is stored once as

.. math::

   \Sigma^x(\omega)=+i L(\omega I-K)^{-1}Q^x
       (\omega I-K)^{-H}L^\dagger,\qquad x\in\{<,>\},

with lower-half-plane poles in ``K`` and positive-semidefinite carrier matrices
``Q``.  This is Quatrex's occupation-positive convention
``-i Sigma^{<,>} >= 0``.  Its retarded partner is not fitted independently.  It is reconstructed
from ``Q^>-Q^<`` so that ``Sigma^R-Sigma^A=Sigma^>-Sigma^<`` holds as a rational
identity, including between frequency-grid points.

Two properties make this useful rather than merely another pole sampler.

* The convolution of two passive clusters is another passive cluster.  Its
  generator is the Kronecker sum and its source is a sum of PSD Kronecker
  products.  The output sum pole is therefore carried as a state; it is never
  sampled back onto the coarse grid.
* A channel whose physical coupling is supported on the owner cell and its two
  neighbours can be inserted into the existing block-tridiagonal RGF by adding
  the auxiliary states to the owner block.  Eliminating those states produces
  physical self-energy shells out to distance two while the augmented operator
  remains block tridiagonal.  This is precisely the shell the current hard
  output pin loses.

The first property is complete and used by the reduced/Si rank studies.  The
second is exposed by :class:`LocalAuxiliaryChannel` and the dense assembly
oracle below.  The DSDB/RGF adapter is intentionally kept separate: an SCBA
driver must mix/adapt the rational state in a fixed basis, not replace it
outside the existing mixer.  Wiring an un-mixed sidecar would change the fixed
point and is refused in the production solver until that state mixer exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qttools import NDArray

__all__ = [
    "PassiveClusterState",
    "RationalKeldyshChannel",
    "LocalAuxiliaryChannel",
    "lyapunov_gramian",
    "passive_bubble_channel",
    "assemble_augmented_dense",
    "physical_schur_complement",
    "LocalAuxiliaryRGF",
    "GlobalAuxiliaryWoodbury",
]


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _dagger(a: NDArray) -> NDArray:
    return np.asarray(a).conj().swapaxes(-2, -1)


def _hermitian(a: NDArray) -> NDArray:
    a = np.asarray(a, dtype=complex)
    return 0.5 * (a + _dagger(a))


def _relative_psd_floor(a: NDArray) -> float:
    a = _hermitian(a)
    scale = max(float(np.linalg.norm(a, 2)), 1.0)
    return float(np.linalg.eigvalsh(a).min() / scale)


def lyapunov_gramian(poles: NDArray, source: NDArray) -> NDArray:
    r"""Frequency integral of a diagonal passive realization.

    For ``D(w)=diag(1/(w-z_a))``,

    .. math::

       W=\int D(\omega)Q D(\omega)^\dagger\frac{d\omega}{2\pi},
       \qquad W_{ab}=\frac{-i Q_{ab}}{z_a-z_b^*}.

    ``W`` is a controllability Gramian and is PSD whenever ``Q`` is PSD.  The
    explicit formula avoids a dense Lyapunov solve and is also the factor that
    closes the cluster--cluster convolution below.
    """
    z = np.asarray(_host(poles), dtype=complex).reshape(-1)
    q = _hermitian(_host(source))
    if q.shape != (z.size, z.size):
        raise ValueError(
            f"source has shape {q.shape}, expected {(z.size, z.size)}")
    if np.any(np.imag(z) >= 0.0):
        raise ValueError("passive poles must lie strictly in the lower half plane")
    gap = z[:, None] - z.conj()[None, :]
    w = _hermitian((-1j * q) / gap)
    if _relative_psd_floor(w) < -5e-11:
        raise ValueError("the pole/source pair produced a non-PSD Gramian")
    return w


@dataclass(frozen=True)
class PassiveClusterState:
    r"""A coherent cluster used as one Keldysh bubble leg.

    ``+i U D Q D^H U^H`` is the stored lesser/greater quantity.  The source
    matrices are Hermitian carriers (without the ``+i`` Keldysh factor).
    """

    poles: NDArray
    coupling: NDArray
    q_lesser: NDArray
    q_greater: NDArray

    def __post_init__(self) -> None:
        z = np.asarray(_host(self.poles), dtype=complex).reshape(-1)
        u = np.asarray(_host(self.coupling), dtype=complex)
        ql = _hermitian(_host(self.q_lesser))
        qg = _hermitian(_host(self.q_greater))
        if u.ndim != 2 or u.shape[1] != z.size:
            raise ValueError("coupling must have shape (n_dof, n_poles)")
        if ql.shape != (z.size, z.size) or qg.shape != ql.shape:
            raise ValueError("q_lesser/q_greater must be square in pole space")
        if np.any(np.imag(z) >= 0.0):
            raise ValueError("passive poles must lie strictly below the real axis")
        if _relative_psd_floor(ql) < -5e-11:
            raise ValueError("q_lesser is not positive semidefinite")
        if _relative_psd_floor(qg) < -5e-11:
            raise ValueError("q_greater is not positive semidefinite")
        object.__setattr__(self, "poles", z)
        object.__setattr__(self, "coupling", u)
        object.__setattr__(self, "q_lesser", ql)
        object.__setattr__(self, "q_greater", qg)

    @property
    def rank(self) -> int:
        return int(self.poles.size)

    @property
    def n_dof(self) -> int:
        return int(self.coupling.shape[0])

    def carrier(self, omega: NDArray | float, greater: bool = False) -> NDArray:
        """Hermitian carrier ``U D Q D^H U^H``."""
        w = np.asarray(omega, dtype=complex)
        flat = w.reshape(-1)
        d = 1.0 / (flat[:, None] - self.poles[None, :])
        q = self.q_greater if greater else self.q_lesser
        modal = np.einsum("wa,ab,wb->wab", d, q, d.conj(), optimize=True)
        out = np.einsum(
            "ia,wab,jb->wij", self.coupling, modal,
            self.coupling.conj(), optimize=True)
        return out.reshape(w.shape + (self.n_dof, self.n_dof))

    def keldysh(self, omega: NDArray | float, greater: bool = False) -> NDArray:
        return 1j * self.carrier(omega, greater=greater)


def _retarded_right(poles: NDArray, left: NDArray, q_delta: NDArray) -> NDArray:
    r"""Right residues of the causal partner of ``+i L D Q_delta D^H L^H``."""
    z = np.asarray(poles, dtype=complex).reshape(-1)
    l = np.asarray(left, dtype=complex)
    q = _hermitian(q_delta)
    gap = z[:, None] - z.conj()[None, :]
    coeff = (1j * q) / gap
    # R[a,j] = sum_b coeff[a,b] conj(L[j,b]).
    return coeff @ l.conj().T


@dataclass(frozen=True)
class RationalKeldyshChannel(PassiveClusterState):
    r"""Passive lesser/greater channel plus its causally matched retarded part."""

    right: NDArray | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        right = self.right
        if right is None:
            right = _retarded_right(
                self.poles, self.coupling, self.q_greater - self.q_lesser)
        right = np.asarray(_host(right), dtype=complex)
        if right.shape != (self.rank, self.n_dof):
            raise ValueError(
                f"right has shape {right.shape}, expected "
                f"{(self.rank, self.n_dof)}")
        object.__setattr__(self, "right", right)

    def retarded(self, omega: NDArray | float) -> NDArray:
        w = np.asarray(omega, dtype=complex)
        flat = w.reshape(-1)
        d = 1.0 / (flat[:, None] - self.poles[None, :])
        out = np.einsum(
            "ia,wa,aj->wij", self.coupling, d, self.right, optimize=True)
        return out.reshape(w.shape + (self.n_dof, self.n_dof))

    def spectral_identity_error(self, omega: NDArray) -> float:
        sr = self.retarded(omega)
        delta = self.keldysh(omega, greater=True) - self.keldysh(omega)
        got = sr - _dagger(sr)
        return float(np.linalg.norm(got - delta)
                     / max(np.linalg.norm(delta), 1e-300))


def passive_bubble_channel(
    vertex: NDArray,
    leg_a: PassiveClusterState,
    leg_b: PassiveClusterState | None = None,
    *,
    carrier_prefactor: float = 1.0,
) -> RationalKeldyshChannel:
    r"""Exact cluster--cluster bubble as one Kronecker-sum auxiliary channel.

    ``vertex[m,i,j]`` is the same dense cubic vertex ordering used by
    :func:`quatrex.phonon.pole_bubble.modal_vertex`.  The returned state has
    poles ``z_a+z_b`` and physical coupling
    ``vertex @ (U_a tensor U_b)``.  For either Keldysh component,

    .. math::

       Q_{ab}=W_a\otimes Q_b + Q_a\otimes W_b,

    where ``W`` is :func:`lyapunov_gramian`.  Both terms are PSD Kronecker
    products, hence positivity is structural and no eigenvalue clipping is
    involved.
    """
    if leg_b is None:
        leg_b = leg_a
    phi = np.asarray(_host(vertex), dtype=complex)
    if phi.ndim != 3 or phi.shape[1] != leg_a.n_dof \
            or phi.shape[2] != leg_b.n_dof:
        raise ValueError(
            "vertex must have shape (n_out, leg_a.n_dof, leg_b.n_dof)")
    pref = float(carrier_prefactor)
    if pref < 0.0:
        raise ValueError("carrier_prefactor must be non-negative")

    za, zb = leg_a.poles, leg_b.poles
    poles = (za[:, None] + zb[None, :]).reshape(-1)
    left = np.einsum(
        "mij,ia,jb->mab", phi, leg_a.coupling, leg_b.coupling,
        optimize=True).reshape(phi.shape[0], -1)

    def one(qa, qb):
        wa = lyapunov_gramian(za, qa)
        wb = lyapunov_gramian(zb, qb)
        q = (np.einsum("ac,bd->abcd", wa, qb, optimize=True)
             + np.einsum("ac,bd->abcd", qa, wb, optimize=True))
        # (a,b,c,d) above means row state (a,b), column state (c,d).
        return pref * _hermitian(q.reshape(poles.size, poles.size))

    ql = one(leg_a.q_lesser, leg_b.q_lesser)
    qg = one(leg_a.q_greater, leg_b.q_greater)
    return RationalKeldyshChannel(poles, left, ql, qg)


@dataclass(frozen=True)
class LocalAuxiliaryChannel:
    r"""A rational channel assigned to one augmented transport block.

    The physical coupling may touch ``owner-1``, ``owner`` and ``owner+1``.
    Eliminating the state can therefore generate a physical self-energy through
    distance two, while the augmented matrix itself remains block tridiagonal.
    Wider support is refused: it needs the sequentially semiseparable extension
    described in ``conserving_long_range_tail.md`` rather than a hidden dense
    block.
    """

    channel: RationalKeldyshChannel
    owner: int
    block_sizes: NDArray
    support_tol: float = 1e-12

    def __post_init__(self) -> None:
        sizes = np.asarray(_host(self.block_sizes), dtype=int).reshape(-1)
        if not (0 <= int(self.owner) < sizes.size):
            raise ValueError("owner is outside the transport-cell range")
        if int(sizes.sum()) != self.channel.n_dof:
            raise ValueError("block_sizes do not span the channel's physical DOFs")
        off = np.concatenate(([0], np.cumsum(sizes)))
        allowed = np.zeros(int(off[-1]), dtype=bool)
        for i in range(max(0, int(self.owner) - 1),
                       min(sizes.size, int(self.owner) + 2)):
            allowed[off[i]:off[i + 1]] = True
        l_bad = np.linalg.norm(self.channel.coupling[~allowed])
        r_bad = np.linalg.norm(self.channel.right[:, ~allowed])
        scale = max(np.linalg.norm(self.channel.coupling),
                    np.linalg.norm(self.channel.right), 1.0)
        if max(l_bad, r_bad) > float(self.support_tol) * scale:
            raise ValueError(
                "auxiliary coupling reaches beyond owner +/- 1; use an SSS "
                "spatial extension instead of the local augmented RGF")
        object.__setattr__(self, "block_sizes", sizes)

    @property
    def augmented_block_sizes(self) -> NDArray:
        out = self.block_sizes.copy()
        out[int(self.owner)] += self.channel.rank
        return out


def assemble_augmented_dense(
    physical_operator: NDArray,
    sigma_lesser: NDArray,
    sigma_greater: NDArray,
    local: LocalAuxiliaryChannel,
    omega: float,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    r"""Dense oracle for the local augmented RGF assembly.

    Returns ``(M_aug, Sigma_l_aug, Sigma_g_aug, physical_indices)``.  This is
    used for parity tests and for the frozen Si gate.  Production assembly uses
    exactly the same block placement but writes DSDB blocks and calls the
    existing RGF; keeping this oracle independent makes a wrong block offset
    visible.
    """
    a = np.asarray(_host(physical_operator), dtype=complex)
    sl = np.asarray(_host(sigma_lesser), dtype=complex)
    sg = np.asarray(_host(sigma_greater), dtype=complex)
    n = local.channel.n_dof
    if a.shape != (n, n) or sl.shape != a.shape or sg.shape != a.shape:
        raise ValueError("physical operator and Keldysh sources must be (n,n)")

    sizes = local.block_sizes
    off = np.concatenate(([0], np.cumsum(sizes)))
    aug_sizes = local.augmented_block_sizes
    aoff = np.concatenate(([0], np.cumsum(aug_sizes)))
    na = int(aoff[-1])
    out_a = np.zeros((na, na), dtype=complex)
    out_l = np.zeros_like(out_a)
    out_g = np.zeros_like(out_a)
    physical = []
    for i, b in enumerate(sizes):
        physical.extend(range(int(aoff[i]), int(aoff[i] + b)))
        pi = slice(int(off[i]), int(off[i + 1]))
        ai = slice(int(aoff[i]), int(aoff[i] + b))
        for j, c in enumerate(sizes):
            pj = slice(int(off[j]), int(off[j + 1]))
            aj = slice(int(aoff[j]), int(aoff[j] + c))
            out_a[ai, aj] = a[pi, pj]
            out_l[ai, aj] = sl[pi, pj]
            out_g[ai, aj] = sg[pi, pj]

    o = int(local.owner)
    aux = slice(int(aoff[o] + sizes[o]), int(aoff[o + 1]))
    out_a[aux, aux] = np.diag(float(omega) - local.channel.poles)
    out_l[aux, aux] = 1j * local.channel.q_lesser
    out_g[aux, aux] = 1j * local.channel.q_greater
    phys = np.asarray(physical, dtype=int)
    out_a[np.ix_(phys, np.arange(aux.start, aux.stop))] -= \
        local.channel.coupling
    out_a[np.ix_(np.arange(aux.start, aux.stop), phys)] -= \
        local.channel.right
    return out_a, out_l, out_g, phys


def physical_schur_complement(
    physical_operator: NDArray,
    channel: RationalKeldyshChannel,
    omega: float,
) -> NDArray:
    """``A - Sigma_aux^R`` evaluated without the augmented states."""
    a = np.asarray(_host(physical_operator), dtype=complex)
    d = 1.0 / (float(omega) - channel.poles)
    return a - (channel.coupling * d[None, :]) @ channel.right


def _block_band_pattern(block_sizes: NDArray, band: int):
    """Backend-native full-block sparsity pattern through ``band``."""
    from qttools import sparse, xp

    sizes = np.asarray(block_sizes, dtype=int)
    off = np.concatenate(([0], np.cumsum(sizes)))
    rows, cols = [], []
    for i in range(sizes.size):
        for j in range(max(0, i - int(band)),
                       min(sizes.size, i + int(band) + 1)):
            rr, cc = np.meshgrid(
                np.arange(off[i], off[i + 1]),
                np.arange(off[j], off[j + 1]), indexing="ij")
            rows.append(rr.ravel()); cols.append(cc.ravel())
    row = xp.asarray(np.concatenate(rows))
    col = xp.asarray(np.concatenate(cols))
    return sparse.coo_matrix(
        (xp.ones(row.size, dtype=float), (row, col)),
        shape=(int(off[-1]), int(off[-1])))


class LocalAuxiliaryRGF:
    r"""Adapter from one local rational channel to the existing RGF.

    The adapter enlarges only the owner's transport block, fills the rational
    state in the augmented diagonal block, pads the contact blocks with zeros,
    calls the unmodified :class:`qttools.greens_function_solver.RGF`, and copies
    the physical selected blocks back to the caller's DSDB buffers.

    The present adapter deliberately supports a frequency stack with an
    optional *replicated* transverse-q tail only: one channel is broadcast to
    every q.  A real film has a different pole set at every q and needs the
    padded q-resolved state container measured by the Si study.  Passing a
    q-distributed DSDB object is refused rather than pairing a channel with the
    wrong momentum.

    This solves the Dyson/Keldysh equations for a supplied rational channel. It
    does not decide how an SCBA mixer updates that channel.  Replacing a mixed
    grid self-energy by an un-mixed sidecar changes the fixed point, so callers
    must keep the rational basis fixed and mix its small coefficients before
    installing it.
    """

    def __init__(self, local: LocalAuxiliaryChannel, frequencies: NDArray, *,
                 max_batch_size: int = 100, n_offdiagonals: int = 1):
        from qttools.greens_function_solver import RGF

        self.local = local
        self.frequencies = np.asarray(_host(frequencies), dtype=float).reshape(-1)
        self.n_offdiagonals = int(n_offdiagonals)
        if self.n_offdiagonals not in (1, 2, 3):
            raise ValueError("LocalAuxiliaryRGF supports 1--3 selected off-diagonals")
        self._rgf = RGF(max_batch_size=max_batch_size)
        self._cache = None

    def _allocate(self, template):
        aug_sizes = self.local.augmented_block_sizes
        typ = type(template)
        stack = tuple(int(x) for x in template.global_stack_shape)
        a = typ.from_sparray(
            _block_band_pattern(aug_sizes, 1), block_sizes=aug_sizes,
            global_stack_shape=stack)
        sl = typ.empty_like(a); sg = typ.empty_like(a)
        sl.allocate_data(); sg.allocate_data()
        out_pattern = _block_band_pattern(aug_sizes, self.n_offdiagonals)
        proto = typ.from_sparray(
            out_pattern, block_sizes=aug_sizes, global_stack_shape=stack)
        xl = typ.empty_like(proto); xg = typ.empty_like(proto)
        xr = typ.empty_like(proto)
        xl.allocate_data(); xg.allocate_data(); xr.allocate_data()
        self._cache = a, sl, sg, xl, xg, xr

    @staticmethod
    def _pad_obc(obc, aug_sizes, physical_sizes):
        from qttools import xp
        from qttools.greens_function_solver.solver import OBCBlocks

        if obc is None:
            return None
        out = OBCBlocks(num_blocks=len(aug_sizes))
        for name in ("retarded", "lesser", "greater"):
            src, dst = getattr(obc, name), getattr(out, name)
            for i, block in enumerate(src):
                if block is None:
                    continue
                b, ba = int(physical_sizes[i]), int(aug_sizes[i])
                padded = xp.zeros(block.shape[:-2] + (ba, ba), dtype=block.dtype)
                padded[..., :b, :b] = block
                dst[i] = padded
        return out

    def selected_solve(self, a, sigma_lesser, sigma_greater, out, *,
                       obc_blocks=None, return_retarded: bool = True,
                       return_current: bool = False):
        """Run the existing selected RGF on the augmented realization."""
        from qttools import xp
        from qttools.comm import comm

        if comm.block.size != 1:
            raise NotImplementedError(
                "LocalAuxiliaryRGF currently requires block_comm_size == 1")
        for m in (a, sigma_lesser, sigma_greater, *out):
            if m.distribution_state != "stack":
                raise ValueError("LocalAuxiliaryRGF requires DSDB stack state")
        if getattr(a, "q_section_offsets", None) is not None:
            raise NotImplementedError(
                "q-distributed auxiliary channels require the q-resolved "
                "padded state container")
        if int(a.shape[0]) != self.frequencies.size:
            raise ValueError(
                f"frequency slice has {a.shape[0]} entries, expected "
                f"{self.frequencies.size}")
        if self._cache is None:
            self._allocate(a)
        aa, al, ag, xl, xg, xr = self._cache
        for m in self._cache:
            m.data[:] = 0.0

        sizes = self.local.block_sizes
        aug = self.local.augmented_block_sizes
        off = np.concatenate(([0], np.cumsum(sizes)))
        owner = int(self.local.owner)
        rank = self.local.channel.rank

        # Physical blocks.  The Dyson operator and Keldysh source consumed by
        # RGF are block tridiagonal; farther entries in the shared G pattern are
        # output slots and are intentionally not copied into the operator.
        for i, bi in enumerate(sizes):
            for j in range(max(0, i - 1), min(sizes.size, i + 2)):
                bj = int(sizes[j])
                ba = aa.blocks[i, j]
                bl = al.blocks[i, j]
                bg = ag.blocks[i, j]
                ba[..., :bi, :bj] = a.blocks[i, j]
                bl[..., :bi, :bj] = sigma_lesser.blocks[i, j]
                bg[..., :bi, :bj] = sigma_greater.blocks[i, j]
                aa.blocks[i, j] = ba
                al.blocks[i, j] = bl
                ag.blocks[i, j] = bg

        # The rational state is frequency local and broadcast over replicated
        # q axes.  Its physical maps may touch owner +/- 1, which are exactly
        # augmented neighbouring blocks.
        c = self.local.channel
        aux = slice(int(sizes[owner]), int(sizes[owner]) + rank)
        stack_ndim = len(aa.shape[:-2])
        w = xp.asarray(self.frequencies).reshape(
            (self.frequencies.size,) + (1,) * (stack_ndim - 1) + (1,))
        poles = xp.asarray(c.poles).reshape((1,) * stack_ndim + (rank,))
        diag = w - poles
        ii = xp.arange(rank)
        ba = aa.blocks[owner, owner]
        bl = al.blocks[owner, owner]
        bg = ag.blocks[owner, owner]
        ba[..., aux, aux] = 0.0
        # Index the augmented block directly.  Chaining ``[..., aux, aux]``
        # and then integer-array diagonal indexing produces a temporary under
        # NumPy/CuPy, leaving the stored auxiliary pivot zero and singular.
        ba[..., int(sizes[owner]) + ii,
           int(sizes[owner]) + ii] = diag
        bl[..., aux, aux] = 1j * xp.asarray(c.q_lesser)
        bg[..., aux, aux] = 1j * xp.asarray(c.q_greater)
        aa.blocks[owner, owner] = ba
        al.blocks[owner, owner] = bl
        ag.blocks[owner, owner] = bg
        for p in range(max(0, owner - 1), min(sizes.size, owner + 2)):
            ps = slice(int(off[p]), int(off[p + 1]))
            bp = int(sizes[p])
            if p == owner:
                boo = aa.blocks[owner, owner]
                boo[..., :bp, aux] -= xp.asarray(c.coupling[ps])
                boo[..., aux, :bp] -= xp.asarray(c.right[:, ps])
                aa.blocks[owner, owner] = boo
                continue
            bpo = aa.blocks[p, owner]
            bop = aa.blocks[owner, p]
            bpo[..., :bp, aux] -= xp.asarray(c.coupling[ps])
            bop[..., aux, :bp] -= xp.asarray(c.right[:, ps])
            aa.blocks[p, owner] = bpo
            aa.blocks[owner, p] = bop

        aug_obc = self._pad_obc(obc_blocks, aug, sizes)
        current = self._rgf.selected_solve(
            aa, al, ag, out=(xl, xg, xr), obc_blocks=aug_obc,
            return_retarded=True, return_current=return_current,
            n_offdiagonals=self.n_offdiagonals)

        # Return only physical selected blocks.  Output matrices may use a
        # symmetry-compressed pattern; writing the stored half is sufficient,
        # while the non-symmetric case receives both directions here.
        ol, og, *orr = out
        for i, bi in enumerate(sizes):
            for j in range(max(0, i - self.n_offdiagonals),
                           min(sizes.size, i + self.n_offdiagonals + 1)):
                bj = int(sizes[j])
                ol.blocks[i, j] = xl.blocks[i, j][..., :bi, :bj]
                og.blocks[i, j] = xg.blocks[i, j][..., :bi, :bj]
                if return_retarded and orr:
                    orr[0].blocks[i, j] = xr.blocks[i, j][..., :bi, :bj]
        return current


def _btd_blocks_with_obc(a, source, obc_blocks, component: str):
    """Dense BTD block lists, including the RGF contact convention."""
    blocks = []
    contact = None if obc_blocks is None else getattr(obc_blocks, component)
    for i in range(a.num_blocks):
        row = []
        for j in range(max(0, i - 1), min(a.num_blocks, i + 2)):
            value = (a if component == "retarded" else source).blocks[i, j]
            if i == j and contact is not None and contact[i] is not None:
                value = (value - contact[i] if component == "retarded"
                         else value + contact[i])
            row.append((j, value))
        blocks.append(row)
    diag = [dict(row)[i] for i, row in enumerate(blocks)]
    upper = [dict(blocks[i])[i + 1] for i in range(a.num_blocks - 1)]
    lower = [dict(blocks[i + 1])[i] for i in range(a.num_blocks - 1)]
    return diag, upper, lower


class GlobalAuxiliaryWoodbury:
    r"""Exact global low-rank auxiliary update around the production RGF.

    A propagating film mode is not local to one transport cell.  For such a
    channel, forcing its coupling into an enlarged local block would either be
    wrong or amount to reblocking the whole device.  This adapter instead uses

    .. math::

       (A-LD R)^{-1}=G_0+G_0L[(\omega I-K)-RG_0L]^{-1}RG_0,

    where all actions of ``G_0`` are multiple-RHS solves with the existing
    block-tridiagonal factorization.  The smooth Keldysh source is updated by
    the same identity and the auxiliary source enters by congruence.  No dense
    physical inverse or long-range self-energy matrix is formed.

    The channel is presently fixed and broadcast over any replicated q axes.
    A q-resolved SCBA uses a padded/direct-sum channel container, which is kept
    behind the Si rank gate because its raw rank may be prohibitive.  Internal
    interface currents for a global self-energy need a cut-aware auxiliary
    current operator; until that is implemented this adapter returns the two
    exact lead currents and ``NaN`` internally, matching the existing dense
    inverse solver's documented convention rather than inventing a current.
    """

    def __init__(self, channel: RationalKeldyshChannel, frequencies: NDArray,
                 *, max_batch_size: int = 100, n_offdiagonals: int = 1):
        from qttools.greens_function_solver import RGF

        self.channel = channel
        self.frequencies = np.asarray(_host(frequencies), dtype=float).reshape(-1)
        self.max_batch_size = max(1, int(max_batch_size))
        self.n_offdiagonals = int(n_offdiagonals)
        if self.n_offdiagonals not in (1, 2, 3):
            raise ValueError(
                "GlobalAuxiliaryWoodbury supports 1--3 selected off-diagonals")
        self._rgf = RGF(max_batch_size=self.max_batch_size)

    @staticmethod
    def _source_correction(fac, p_blocks, x, y, q, xp):
        """Low-rank correction to ``G0 P G0^H`` for one Keldysh source."""
        from quatrex.phonon.btd_linalg import btd_matvec

        yh = y.conj().swapaxes(-2, -1)
        p_yh = btd_matvec(*p_blocks, yh)
        c = fac.solve(p_yh)                       # G0 P Y^H
        h = y @ p_yh                              # Y P Y^H
        return c, h, xp.asarray(q)

    def selected_solve(self, a, sigma_lesser, sigma_greater, out, *,
                       obc_blocks=None, return_retarded: bool = True,
                       return_current: bool = False):
        from qttools import xp
        from qttools.comm import comm
        from quatrex.phonon.btd_linalg import BTDFactorization

        if comm.block.size != 1:
            raise NotImplementedError(
                "GlobalAuxiliaryWoodbury requires block_comm_size == 1")
        for m in (a, sigma_lesser, sigma_greater, *out):
            if m.distribution_state != "stack":
                raise ValueError(
                    "GlobalAuxiliaryWoodbury requires DSDB stack state")
        if getattr(a, "q_section_offsets", None) is not None:
            raise NotImplementedError(
                "q-distributed auxiliary channels need a q-resolved state")
        if int(a.shape[0]) != self.frequencies.size:
            raise ValueError(
                f"frequency slice has {a.shape[0]} entries, expected "
                f"{self.frequencies.size}")
        if int(sum(a.block_sizes)) != self.channel.n_dof:
            raise ValueError("auxiliary coupling does not span the physical device")

        # Established selected result for A and the smooth source.  The update
        # below touches only selected output blocks, so the default path stays
        # the numerical oracle rather than being reimplemented here.
        self._rgf.selected_solve(
            a, sigma_lesser, sigma_greater, out=out,
            obc_blocks=obc_blocks, return_retarded=return_retarded,
            return_current=False, n_offdiagonals=self.n_offdiagonals)

        ar = _btd_blocks_with_obc(a, None, obc_blocks, "retarded")
        pl = _btd_blocks_with_obc(a, sigma_lesser, obc_blocks, "lesser")
        pg = _btd_blocks_with_obc(a, sigma_greater, obc_blocks, "greater")
        fac = BTDFactorization.factorize(*ar)
        stack = tuple(int(k) for k in a.shape[:-2])
        n, r = self.channel.n_dof, self.channel.rank
        left = xp.broadcast_to(
            xp.asarray(self.channel.coupling), stack + (n, r)).copy()
        right_h = xp.broadcast_to(
            xp.asarray(self.channel.right).conj().T, stack + (n, r)).copy()
        x = fac.solve(left)                         # G0 L
        y = fac.solve_hermitian(right_h).conj().swapaxes(-2, -1)  # R G0

        wshape = (self.frequencies.size,) + (1,) * (len(stack) - 1) + (1,)
        w = xp.asarray(self.frequencies).reshape(wshape)
        poles = xp.asarray(self.channel.poles).reshape((1,) * len(stack) + (r,))
        # Schur pivot is D^{-1} - R G0 L.  ``y`` already contains R G0,
        # whereas ``x`` contains G0 L; multiplying y@x would insert G0 twice.
        s = -(y @ left)
        ii = xp.arange(r)
        s[..., ii, ii] += w - poles
        t = xp.linalg.inv(s)
        z = x @ t

        c_l, h_l, q_l = self._source_correction(
            fac, pl, x, y, self.channel.q_lesser, xp)
        c_g, h_g, q_g = self._source_correction(
            fac, pg, x, y, self.channel.q_greater, xp)
        off = np.concatenate(([0], np.cumsum(np.asarray(a.block_sizes, int))))
        ol, og, *orr = out

        def correction(zi, zj, ci, cj, h, q):
            zjh = zj.conj().swapaxes(-2, -1)
            cjh = cj.conj().swapaxes(-2, -1)
            return (-zi @ cjh + ci @ zjh + zi @ h @ zjh
                    + 1j * zi @ q @ zjh)

        for i in range(a.num_blocks):
            si = slice(int(off[i]), int(off[i + 1]))
            for j in range(max(0, i - self.n_offdiagonals),
                           min(a.num_blocks, i + self.n_offdiagonals + 1)):
                sj = slice(int(off[j]), int(off[j + 1]))
                dl = correction(z[..., si, :], z[..., sj, :],
                                c_l[..., si, :], c_l[..., sj, :], h_l, q_l)
                dg = correction(z[..., si, :], z[..., sj, :],
                                c_g[..., si, :], c_g[..., sj, :], h_g, q_g)
                if i == j:
                    dl = 0.5 * (dl - dl.conj().swapaxes(-2, -1))
                    dg = 0.5 * (dg - dg.conj().swapaxes(-2, -1))
                ol.blocks[i, j] = ol.blocks[i, j] + dl
                og.blocks[i, j] = og.blocks[i, j] + dg
                # Existing RGF semantics expose retarded diagonal blocks only.
                if return_retarded and orr and i == j:
                    dr = z[..., si, :] @ y[..., :, sj]
                    orr[0].blocks[i, i] = orr[0].blocks[i, i] + dr

        if not return_current:
            return None
        current = xp.full(stack + (a.num_blocks + 1,), xp.nan,
                          dtype=a.dtype)
        if obc_blocks is None:
            return current
        for slot, block, sign in ((0, 0, 1.0), (-1, a.num_blocks - 1, -1.0)):
            if obc_blocks.lesser[block] is None \
                    or obc_blocks.greater[block] is None:
                continue
            gl = ol.blocks[block, block]
            gg = og.blocks[block, block]
            current[..., slot] = sign * xp.trace(
                obc_blocks.greater[block] @ gl
                - gg @ obc_blocks.lesser[block], axis1=-2, axis2=-1)
        return current
