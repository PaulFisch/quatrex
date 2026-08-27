"""Beds for the spatially analytic Green-function tail programme.

Two kinds, and they answer different questions.

**Analytic beds** (:func:`legs`, :func:`ring`, :func:`long_bed`) are the gapped
1-DOF chain: exact complex bands, exact ``G`` by Brillouin-zone quadrature,
exact ring, and a grid that sits below the band so everything is regular at
``eta = 0``. Every invariant is checked here first, because a wrong answer on
this bed is a bug and a wrong answer on a device is an open question. The
physics half lives in ``phonon/solver/toy_models.py``; what is here is the
scaffolding a self-energy experiment needs on top of it.

**Frozen device beds** (:func:`build_frozen`, :class:`FrozenBed`) are a real
stored device -- ``dynamical_matrix.mat`` + ``fc3_blocks.hdf5`` -- driven to an
SCBA fixed point by the dense reference solver, with the whole dense
``(G^R, G^<, G^>, Sigma)`` tuple kept. Production never writes ``G`` (only
``Sigma``, via ``QX_SAVE_SIGMA``) and its RGF returns at most three
off-diagonals, so a question about long-range ``G`` cannot be asked of a
production snapshot at all. The dense reference has no such limit: every block
distance is present and ``solve_green_batch`` inverts a DENSE Dyson operator,
so a self-energy that is not block-tridiagonal is solved exactly.

Traps this module exists to not fall into, each guarded by an assert:

* ``fmax >= 2 omega_max``. The stored ``phonon_energies.npy`` grids sit at
  ``fmax ~ omega_max`` -- half what the 3-phonon convolution needs -- so they
  are NOT reused; the grid is rebuilt and ``_ensure_fmax`` must not fire.
* the SYMMETRIC axis. ``se_finite`` takes ``mid = n_freq // 2`` and the DC
  regularisation acts on that bin; on a positive-only grid ``mid`` points
  mid-band and the kernel silently kills a physical sample.
* ``eta = 0`` (CLAUDE.md), which means ``z2[mid] = 0`` exactly -- so nothing is
  inverted there -- and which is also why the OBC is the spectral (NEVP) one:
  Sancho-Rubio has nothing to damp it in-band and raises.
* ``build_device_hamiltonian`` silently ignores the ``d10`` it is handed, so
  ``d10 == d01^dagger`` is checked before it is called.
* ``load_device_fc3`` would reject any block count but the stored one and
  defaults to ``nn_only=True``, dropping exactly the triplets under study; the
  vertex is tiled in memory from the stored offsets instead, through the
  bit-exact translation gate of ``_tile_device_inputs``.

Run:  QTX_ARRAY_MODULE=numpy python phonon/studies/_spatial_bed.py --bed cluster/cnt_cal --cells 8
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("QTX_ARRAY_MODULE", "numpy")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "phonon")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = ROOT / "phonon/studies/out/spatial_bed"


# ---------------------------------------------------------------------------
# Analytic bed: the gapped chain, its legs and its ring
# ---------------------------------------------------------------------------

W0, KS_G = 1.0, 4.0                     # band [1.0, 4.123]; grids sit below w0


def out_distance(n_cell: int) -> np.ndarray:
    """``|I - J|`` for every output block pair."""
    i, j = np.meshgrid(np.arange(n_cell), np.arange(n_cell), indexing="ij")
    return np.abs(i - j)


def legs(omegas, band: int, n_cell: int, *, w0: float = W0, k_s: float = KS_G,
         n_k: int = 4096):
    """``(exact, banded, completed)`` spatial legs, each ``(n_w, N, N)``.

    ``completed`` continues from the last block INSIDE the band with the modal
    factor, so the completion only ever supplies what the boxcar removed --
    which is what makes it an approximation of ``G`` and not an additive
    correction on top of one.
    """
    from solver.toy_models import gapped_chain_green, gapped_chain_root

    omegas = np.asarray(omegas, float)
    idx = out_distance(n_cell)
    exact = np.zeros((omegas.size, n_cell, n_cell), dtype=complex)
    completed = np.zeros_like(exact)
    for iw, omega in enumerate(omegas):
        g = gapped_chain_green(omega, n_cell, w0=w0, k_s=k_s, n_k=n_k)
        lam = gapped_chain_root(omega, w0=w0, k_s=k_s)
        exact[iw] = g[idx]
        completed[iw] = np.where(idx <= band, g[idx],
                                 g[band] * lam ** (idx - band))
    banded = np.where(idx <= band, exact, 0.0)
    return exact, banded, completed


def ring(phi, a_leg, b_leg, w, *, normalise: bool = True) -> np.ndarray:
    r"""``Phi_{ace} A_{cb} B_{ed} Phi_{Jdb}``, convolved over ``w``.

    The contraction the production ring performs, written out directly so the
    legs can be swapped without touching anything else. ``normalise`` applies
    the ``1/2pi`` of the frequency measure; it is a global constant, so every
    ratio is independent of it, and it is exposed only because the two callers
    this was promoted from disagreed about it.
    """
    w = np.asarray(w, float)
    n_cell = np.shape(a_leg)[-1]
    h = w[1] - w[0]
    out = np.zeros((w.size, n_cell, n_cell), dtype=complex)
    for iw, om in enumerate(w):
        j = np.rint((om - w - w[0]) / h).astype(int)
        ok = (j >= 0) & (j < w.size)
        conv = np.einsum("kcb,ked->cbed", a_leg[ok], b_leg[j[ok]],
                         optimize=True) * h
        out[iw] = np.einsum("ace,Jdb,cbed->aJ", phi, phi, conv, optimize=True)
    return out / (2.0 * np.pi) if normalise else out


def ring_by_shell(phi, a_leg, b_leg, w, *, bins=None, normalise: bool = True):
    r"""The ring, decomposed EXACTLY by the leg-distance shell of each leg.

    ``Sigma`` is bilinear in ``G``, so splitting ``G = sum_m G^(m)`` by shell
    ``m = |K - K'|`` gives an additive decomposition

        Sigma_R = sum_{m, m'} Sigma_R^{(m, m')} ,

    of which a ``g_cutoff`` sweep is only the partial sums. This answers the
    question a sweep cannot -- WHICH leg distance feeds which output distance
    -- and it costs one pass instead of one per band.

    ``bins`` groups shells to bound the output size; the default keeps 0..3
    separate and lumps the rest, which is where the production band sits.
    Returns ``(sigma_shell, bin_labels)`` with ``sigma_shell`` of shape
    ``(n_bins, n_bins, n_w, N, N)``; summing both bin axes reproduces
    :func:`ring` exactly.
    """
    n_cell = np.shape(a_leg)[-1]
    idx = out_distance(n_cell)
    if bins is None:
        bins = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, n_cell)]
    labels = [f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in bins]
    masks = [((idx >= lo) & (idx <= hi)) for lo, hi in bins]
    covered = np.sum(masks, axis=0)
    if not np.all(covered == 1):
        raise ValueError("ring_by_shell: bins must partition 0..N-1 exactly")

    a_parts = [np.where(m, a_leg, 0.0) for m in masks]
    b_parts = [np.where(m, b_leg, 0.0) for m in masks]
    nb = len(bins)
    out = np.zeros((nb, nb, np.size(w), n_cell, n_cell), dtype=complex)
    for m in range(nb):
        for mp in range(nb):
            out[m, mp] = ring(phi, a_parts[m], b_parts[mp], w,
                              normalise=normalise)
    return out, labels


def long_bed(n_cell: int, top: float = 0.90, nw: int = 14, seed: int = 5,
             *, normalise: bool = False) -> np.ndarray:
    """Gapped chain of ``n_cell`` cells and its exact ring, no leg mask."""
    from solver.toy_models import neighbour_cubic_vertex

    w = np.linspace(0.0, top * W0, nw)
    exact, _, _ = legs(w, n_cell, n_cell)
    return ring(neighbour_cubic_vertex(n_cell, seed=seed), exact, exact, w,
                normalise=normalise)


def discarded(sigma, cells_per_block: int) -> float:
    """Share of ``|Sigma|`` a tridiagonal pin at ``m`` cells per block drops.

    This IS the reblocking arm: reblocking changes the partition, not the
    physics, and a dense Dyson solve has no block-tridiagonal restriction to
    begin with, so masking is the exact statement of what reblocking would
    discard.
    """
    n = np.shape(sigma)[-1]
    blk = np.arange(n) // cells_per_block
    far = np.abs(np.subtract.outer(blk, blk)) > 1
    return float(np.abs(sigma[..., far]).sum() / np.abs(sigma).sum())


# ---------------------------------------------------------------------------
# Frozen device bed
# ---------------------------------------------------------------------------


def transport_blocks(bed_dir: Path, tdir: int, q=(0.0, 0.0)):
    """``(d00, d01, d10)`` at one transverse ``q``, from ``dynamical_matrix.mat``.

    The stored keys are real-space cell offsets on ALL three axes, so the
    transverse ones are Fourier-summed with ``exp(i q.n)`` -- reading them as
    momentum indices is the error recorded in ``spatial_band_range.md``. At
    ``q = 0`` this reduces to the plain sum that ``_bubble_positivity`` and
    ``cm_channel`` both take.

    Raises if the transport coupling reaches past nearest cells, which every
    downstream assumption here depends on.
    """
    import re

    from scipy.io import loadmat

    raw = loadmat(str(Path(bed_dir) / "dynamical_matrix.mat"))
    perp = [a for a in range(3) if a != tdir]
    acc: dict[int, np.ndarray] = {}
    for key, val in raw.items():
        if key.startswith("__"):
            continue
        off = [int(x) for x in re.findall(r"-?\d+", key)]
        n_t = off[tdir]
        if abs(n_t) > 1:
            raise ValueError(
                f"transport_blocks: transport-axis offset {n_t} in "
                f"{bed_dir}; the nearest-cell (OBC superblock) convention is "
                "assumed everywhere downstream.")
        phase = np.exp(1j * (q[0] * off[perp[0]] + q[1] * off[perp[1]]))
        blk = np.asarray(val, dtype=complex) * phase
        acc[n_t] = blk if n_t not in acc else acc[n_t] + blk

    d00, d01, d10 = acc[0], acc[1], acc[-1]
    scale = max(np.linalg.norm(d00), 1e-300)
    if not np.allclose(d00, d00.conj().T, atol=1e-9 * scale):
        raise ValueError("transport_blocks: D00 is not Hermitian")
    if not np.allclose(d10, d01.conj().T, atol=1e-9 * scale):
        raise ValueError("transport_blocks: D10 != D01^dagger")
    return d00, d01, d10


def tile_fc3(bed_dir: Path, n_cells: int) -> dict:
    """``{(I, K, K'): Phi}`` for ``n_cells``, tiled from the stored offsets.

    Goes through ``_tile_device_inputs``, whose ``bulk_offsets`` refuses a
    source that is not translation-invariant -- so a tiled bed is bit-exactly
    the interior of the stored one, or the build fails. Nothing is written to
    disk: ``load_device_fc3`` would reject any block count but the stored one
    and would apply its ``nn_only`` truncation on the way in.
    """
    from studies._tile_device_inputs import bulk_offsets, load_blocks, tile

    return tile(bulk_offsets(load_blocks(Path(bed_dir))), int(n_cells))


def vertex_reach(phi_blocks: dict) -> int:
    r"""``p``: how far the vertex reaches FROM ITS OUTPUT CELL.

    ``max(|I-K|, |I-K'|)`` and deliberately not ``max(..., |K-K'|)``. It is the
    former that enters the support law, by the chain

        |I - J| <= |I - K1| + |K1 - K1'| + |K1' - J| <= p + b + p,

    so ``supp(Sigma) = {|I-J| <= 2p + b}`` with ``p`` bounding the distance from
    the vertex's own cell to its legs. The separation between the two legs is
    bounded by ``2p`` as a consequence and is not an independent reach: the
    nearest-neighbour shell of a real FC3 has ``|K-K'| = 2`` triplets absent
    only because the force constants have none, while ``|I-K| = 1`` is what the
    supercell resolves.
    """
    return max(max(abs(k - i), abs(kp - i)) for (i, k, kp) in phi_blocks)


def spectral_obc(freqs_thz, d00, d01, d10, n_slabs, t_left, t_right):
    """The ``obc`` dict ``solve_green_batch`` wants, from the spectral OBC.

    Not ``leads.compute_obc_batch``: that is Sancho-Rubio, whose decimation has
    nothing to damp it at ``eta = 0`` in-band and which raises on the residual
    check. The spectral (NEVP mode-matching) solver selects the retarded branch
    by group velocity and decay and is exact at ``eta = 0`` -- the same choice
    ``_bubble_positivity._ballistic_g`` and ``cm_channel._spectral_obc`` make.

    Built once, undressed, and frozen: pass ``scattering_contacts=False`` to
    the SCBA so nothing re-enters Sancho-Rubio behind your back.
    """
    from qttools.boundary_conditions.obc import Spectral
    from qttools.nevp import Full

    from solver.grids import boson_contact_self_energies_from_gamma

    obc_solver = Spectral(nevp=Full(), block_sections=1)
    ws = np.asarray(freqs_thz, float)
    z2 = (ws * ws).astype(complex)                  # eta = 0 EXACTLY
    nd = d00.shape[0]
    eye = np.eye(nd)
    m_00 = z2[:, None, None] * eye - d00[None]
    m_01 = np.broadcast_to(-d01, m_00.shape).copy()
    m_10 = np.broadcast_to(-d10, m_00.shape).copy()

    def flip(a):
        return np.flip(a, axis=(-2, -1))

    sig_l = m_10 @ obc_solver(m_00, m_01, m_10, "left") @ m_01
    sig_r = m_01 @ flip(
        obc_solver(flip(m_00), flip(m_10), flip(m_01), "right")) @ m_10

    n_d = n_slabs * nd
    nfreq = ws.size
    z = np.zeros((nfreq, n_d, n_d), dtype=complex)
    sl0, sl_last = slice(0, nd), slice((n_slabs - 1) * nd, n_slabs * nd)
    sigma_l_r, sigma_r_r = z.copy(), z.copy()
    gamma_l, gamma_r = z.copy(), z.copy()
    sigma_l_r[:, sl0, sl0] = sig_l
    sigma_r_r[:, sl_last, sl_last] = sig_r
    gamma_l[:, sl0, sl0] = 1j * (sig_l - sig_l.conj().transpose(0, 2, 1))
    gamma_r[:, sl_last, sl_last] = 1j * (sig_r - sig_r.conj().transpose(0, 2, 1))

    ll, lg = boson_contact_self_energies_from_gamma(gamma_l, ws, t_left)
    rl, rg = boson_contact_self_energies_from_gamma(gamma_r, ws, t_right)
    return {
        "Sigma_L_R": sigma_l_r, "Sigma_L_lesser": ll, "Sigma_L_greater": lg,
        "Sigma_R_R": sigma_r_r, "Sigma_R_lesser": rl, "Sigma_R_greater": rg,
        "Gamma_L": gamma_l, "Gamma_R": gamma_r,
    }


def lead_edges(d00, d01, d10) -> np.ndarray:
    """Band extrema of the periodic lead, in THz."""
    from quatrex.phonon.pole_sector import lead_band_edges

    return np.asarray(lead_band_edges(d00, d01, d10))


def _clear_band_edges(nfreq_pos, fmax, d00, d01, d10, *, margin=0.01,
                      name="", verbose=False, max_tries=512,
                      edge_min: float = 1e-3):
    """Nudge ``nfreq_pos`` until no grid sample lands ON a lead band edge.

    What is being avoided is an EXACT hit, where the group velocity vanishes,
    the eta = 0 surface Green's function has no imaginary part to regularise
    it, and the Dyson solve raises "singular matrix". A sample merely near an
    edge is a resolution question, not a singularity, so the margin is small
    and a failure to clear every edge is a warning rather than a refusal --
    a device with a dozen branches has a dozen edges and clearing all of them
    by a wide margin is not always possible.

    Edges below ``edge_min`` are ignored: that is the acoustic zero at Gamma,
    which sits at omega = 0, is excluded from every physical integral by
    ``pos_mask``, and is regularised by ``dc_handling`` before the bubble FFT.

    Returns ``(nfreq_pos, fmax)``. ``fmax`` is held and only the sample count
    moves, so the aliasing gate above stays satisfied by construction.
    """
    edges = lead_edges(d00, d01, d10)
    edges = np.unique(edges[(edges > edge_min) & (edges <= fmax)])
    if edges.size == 0:
        return int(nfreq_pos), float(fmax)
    best_n, best_gap = int(nfreq_pos), -1.0
    for k in range(max_tries):
        n = int(nfreq_pos) + k
        dw = fmax / n
        gap = float(np.min(np.abs(edges / dw - np.round(edges / dw))))
        if gap > best_gap:
            best_n, best_gap = n, gap
        if gap >= margin:
            if k and verbose:
                print(f"  {name}: nfreq_pos {nfreq_pos} -> {n} to clear a lead "
                      f"band edge (closest now {gap:.4f} dw)")
            return n, float(fmax)
    warnings.warn(
        f"{name}: no sample count in [{nfreq_pos}, {nfreq_pos + max_tries}) "
        f"clears every lead band edge by {margin} dw; using {best_n}, whose "
        f"closest approach is {best_gap:.2e} dw. Edges: "
        f"{np.array2string(edges, precision=4)}.", stacklevel=2)
    return best_n, float(fmax)


@dataclass
class FrozenBed:
    """One frozen SCBA state, with every block distance present."""

    name: str
    n_slabs: int
    n_dof: int
    freqs_thz: np.ndarray
    dw_thz: float
    pos_mask: np.ndarray
    mid: int
    d00: np.ndarray
    d01: np.ndarray
    d10: np.ndarray
    h_d: np.ndarray
    obc: dict
    phi: dict
    g_retarded: np.ndarray
    g_lesser: np.ndarray
    g_greater: np.ndarray
    sigma_lesser: np.ndarray
    sigma_greater: np.ndarray
    sigma_retarded: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_d(self) -> int:
        return self.n_slabs * self.n_dof

    @property
    def p(self) -> int:
        return vertex_reach(self.phi)

    def blocks(self, mat, max_offset=None) -> dict:
        """``{(K, K'): (nfreq, b, b)}`` from a dense device matrix."""
        from solver.dense import _device_g_blocks

        return _device_g_blocks(mat, self.n_slabs, self.n_dof, max_offset,
                                has_q_axis=False)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name, "n_slabs": self.n_slabs, "n_dof": self.n_dof,
            "freqs_thz": self.freqs_thz, "dw_thz": self.dw_thz,
            "pos_mask": self.pos_mask, "mid": self.mid,
            "d00": self.d00, "d01": self.d01, "d10": self.d10, "h_d": self.h_d,
            "g_retarded": self.g_retarded, "g_lesser": self.g_lesser,
            "g_greater": self.g_greater, "sigma_lesser": self.sigma_lesser,
            "sigma_greater": self.sigma_greater,
            "sigma_retarded": self.sigma_retarded,
            "phi_keys": np.array(sorted(self.phi), dtype=np.int64),
            "meta": np.array(repr(self.meta)),
        }
        for k, v in self.obc.items():
            payload[f"obc_{k}"] = v
        for key in sorted(self.phi):
            payload["phi_%d_%d_%d" % key] = self.phi[key]
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: Path) -> "FrozenBed":
        d = np.load(Path(path), allow_pickle=False)
        obc = {k[4:]: d[k] for k in d.files if k.startswith("obc_")}
        phi = {tuple(int(x) for x in key): d["phi_%d_%d_%d" % tuple(key)]
               for key in d["phi_keys"]}
        import ast
        return cls(
            name=str(d["name"]), n_slabs=int(d["n_slabs"]),
            n_dof=int(d["n_dof"]), freqs_thz=d["freqs_thz"],
            dw_thz=float(d["dw_thz"]), pos_mask=d["pos_mask"],
            mid=int(d["mid"]), d00=d["d00"], d01=d["d01"], d10=d["d10"],
            h_d=d["h_d"], obc=obc, phi=phi,
            g_retarded=d["g_retarded"], g_lesser=d["g_lesser"],
            g_greater=d["g_greater"], sigma_lesser=d["sigma_lesser"],
            sigma_greater=d["sigma_greater"],
            sigma_retarded=d["sigma_retarded"],
            meta=ast.literal_eval(str(d["meta"])),
        )


def make_se_kernel(phi, n_slabs, n_dof, freqs_thz, dw_thz, *,
                   sigma_cutoff=None, g_cutoff=None,
                   dc_handling="interpolate", n_threads=None):
    """``se_kernel(G^<, G^>) -> (Sigma^<, Sigma^>)`` for :func:`scba_loop`.

    Mirrors ``dense.transmission``'s own closure, so a frozen state built here
    is the one the production reference driver would have built, and only the
    two spatial cutoffs are ours to move.
    """
    from solver.dense import _device_g_blocks, _scatter_blocks
    from solver.se_finite import compute_phph_self_energy_finite_multi_slab

    n_d = n_slabs * n_dof
    nfreq = len(freqs_thz)

    def se_kernel(g_less_dev_q, g_great_dev_q):
        sig_l = np.zeros((1, nfreq, n_d, n_d), dtype=complex)
        sig_g = np.zeros_like(sig_l)
        gl = _device_g_blocks(g_less_dev_q[0], n_slabs, n_dof, g_cutoff,
                              has_q_axis=False)
        gg = _device_g_blocks(g_great_dev_q[0], n_slabs, n_dof, g_cutoff,
                              has_q_axis=False)
        sl_b, sg_b = compute_phph_self_energy_finite_multi_slab(
            gl, gg, phi, n_slabs, freqs_thz, dw_thz,
            sigma_cutoff=sigma_cutoff, g_cutoff=g_cutoff,
            dc_handling=dc_handling, n_threads=n_threads)
        _scatter_blocks(sig_l[0], sl_b, n_dof)
        _scatter_blocks(sig_g[0], sg_b, n_dof)
        return sig_l, sig_g

    return se_kernel


def freeze(d00, d01, d10, phi, n_cells: int, *, name: str,
           nfreq_pos: int | None = None, dw_thz: float | None = None,
           fmax_margin: float = 1.05,
           t_left: float = 305.0, t_right: float = 295.0,
           max_scba_iter: int = 60, scba_tol: float = 1e-3,
           mixing: float = 0.3, solver: str = "anderson",
           anderson_mixing: bool = True, anderson_depth: int = 8,
           divergence_guard: bool = True, edge_margin: float = 0.01,
           zero_mode_projection: bool = False,
           sigma_cutoff=None, g_cutoff=None, extra_meta: dict | None = None,
           n_threads=None, verbose: bool = True) -> FrozenBed:
    """Drive an assembled device to an SCBA fixed point, keep the dense state.

    ``eta = 0`` throughout, asserted. The grid is SIZED here rather than read
    from a stored ``phonon_energies.npy``: those sit at ``fmax ~ omega_max``,
    half what the 3-phonon convolution needs, and an aliased bubble would make
    every tail number an artefact.

    Front-ends: :func:`build_frozen` for a stored device directory,
    :func:`build_frozen_chain` for a :class:`~solver.toy_models.ToyModel`. Both
    reach the SAME code path, so a wiring bug shows on the analytic chain --
    where the answer is known -- instead of on a device, where it would read as
    physics.
    """
    from solver.dense import _device_omega_max, _ensure_fmax, scba_loop
    from solver.grids import build_frequency_grid
    from solver.leads import build_device_hamiltonian
    from phonon_inputs.constants import THZ_TO_RAD

    d00 = np.asarray(d00, dtype=complex)
    d01 = np.asarray(d01, dtype=complex)
    d10 = np.asarray(d10, dtype=complex)
    n_dof = d00.shape[0]

    # build_device_hamiltonian sets H_10 = H_01^dagger and ignores d10.
    if not np.allclose(d10, d01.conj().T, atol=1e-9 * np.linalg.norm(d01)):
        raise ValueError("build_frozen: d10 != d01^dagger")
    h_d = build_device_hamiltonian(d00, d01, n_cells)

    # The grid is SIZED here, not read from phonon_energies.npy: the stored
    # grids sit at fmax ~ omega_max, half what the convolution support
    # [-2 w_max, 2 w_max] needs, and an aliased bubble makes every tail number
    # an artefact. _ensure_fmax's own auto-extend preserves d_omega rather than
    # the sample count, which is not what a fixed-cost sweep wants, so the top
    # is set directly and the result is then GATED by the same function with
    # auto_extend off -- a warning there is a hard failure.
    omega_max = _device_omega_max(d00, d01)
    fmax = float(np.ceil(fmax_margin * 2.0 * omega_max))
    if dw_thz is not None:
        nfreq_pos = int(np.ceil(fmax / float(dw_thz)))
        fmax = nfreq_pos * float(dw_thz)
    elif nfreq_pos is None:
        nfreq_pos = 300
    nfreq_pos = int(nfreq_pos)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ensure_fmax((0.0, fmax, nfreq_pos), d00, d01, name=name,
                     auto_extend=False, margin=fmax_margin)
    if caught:
        raise AssertionError(
            f"freeze: aliasing gate fired -- {caught[0].message}")

    # A grid sample landing exactly ON a lead band edge is singular at eta = 0:
    # the group velocity vanishes there, so the surface Green's function has no
    # imaginary part to regularise it and the Dyson solve raises "singular
    # matrix". Measure zero in principle and routine in practice, because fmax
    # and the edges are both round numbers -- the gapped chain's edge at 1.0 THz
    # is hit exactly by fmax = 9, nfreq_pos = 180. Nudge the sample count until
    # every sample clears every edge; dw moves by well under a percent.
    nfreq_pos, fmax = _clear_band_edges(nfreq_pos, fmax, d00, d01, d10,
                                        margin=edge_margin, name=name,
                                        verbose=verbose)

    freqs_thz, dw, eta_w, z2_arr, pos_mask, mid = build_frequency_grid(
        (0.0, fmax, nfreq_pos), eta_factor=0.0)
    assert eta_w == 0.0, "eta must be exactly zero"

    obc = spectral_obc(freqs_thz, d00, d01, d10, n_cells, t_left, t_right)

    se_kernel = make_se_kernel(phi, n_cells, n_dof, freqs_thz, dw,
                               sigma_cutoff=sigma_cutoff, g_cutoff=g_cutoff,
                               n_threads=n_threads)
    res = scba_loop(
        z2_arr=z2_arr, freqs_thz=freqs_thz, dw_thz=dw,
        omega_rad=freqs_thz * THZ_TO_RAD, pos_mask=pos_mask,
        n_slabs=n_cells, n_dof=n_dof, N_D=n_cells * n_dof,
        H_D_list=[h_d], obc_list=[obc], btd_blocks_list=[(d00, d01)],
        n_kpts=1, se_kernel=se_kernel, T_L=t_left, T_R=t_right,
        max_scba_iter=max_scba_iter, scba_tol=scba_tol,
        conservation_tol=1e-3, mixing=mixing,
        anderson_mixing=anderson_mixing, anderson_depth=anderson_depth,
        scattering_contacts=False, retarded="fft",
        verbose=verbose, solver=solver, causality_projection=False,
        zero_mode_projection=zero_mode_projection,
        divergence_guard=divergence_guard,
        return_greens=True,
    )

    meta = {
        "fmax_thz": fmax, "nfreq_pos": int(nfreq_pos), "eta": 0.0,
        "t_left": t_left, "t_right": t_right,
        "converged": bool(res["converged"]),
        "scba_residual": float(res["scba_residual"]),
        "conservation_err": float(res["conservation_err"]),
        "sigma_cutoff": sigma_cutoff, "g_cutoff": g_cutoff,
        "vertex_reach_p": vertex_reach(phi),
        "symmetry_factor": "PHPH_SYMMETRY_FACTOR (see constants.py)",
        "solver": solver, "mixing": mixing, "max_scba_iter": max_scba_iter,
        "zero_mode_projection": zero_mode_projection,
    }
    meta.update(extra_meta or {})
    return FrozenBed(
        name=name, n_slabs=n_cells, n_dof=n_dof, freqs_thz=freqs_thz,
        dw_thz=dw, pos_mask=pos_mask, mid=mid, d00=d00, d01=d01, d10=d10,
        h_d=h_d, obc=obc, phi=phi,
        g_retarded=res["G_retarded"][0], g_lesser=res["G_lesser"][0],
        g_greater=res["G_greater"][0], sigma_lesser=res["Sigma_l"][0],
        sigma_greater=res["Sigma_g"][0], sigma_retarded=res["Sigma_R"][0],
        meta=meta)


def build_frozen(bed_dir, n_cells: int, *, name: str | None = None,
                 tdir: int = 2, q=(0.0, 0.0), **kw) -> FrozenBed:
    """:func:`freeze` on a stored device directory."""
    bed_dir = Path(bed_dir)
    name = name or f"{bed_dir.name}_L{n_cells}"
    d00, d01, d10 = transport_blocks(bed_dir, tdir, q=q)
    phi = tile_fc3(bed_dir, n_cells)
    kw.setdefault("extra_meta", {})
    kw["extra_meta"] = dict(kw["extra_meta"],
                            bed=str(bed_dir), tdir=tdir,
                            q=tuple(float(x) for x in q))
    return freeze(d00, d01, d10, phi, n_cells, name=name, **kw)


def build_frozen_chain(model, n_cells: int, *, name: str | None = None,
                       cubic: float = 0.0, reach: int = 1, seed: int = 5,
                       **kw) -> FrozenBed:
    """:func:`freeze` on a toy chain, with a reach-``p`` random cubic vertex.

    The vertex is the one the analytic-bed measurements use
    (:func:`solver.toy_models.neighbour_cubic_vertex`) lifted to ``n_dof``
    per cell, so a 1-DOF chain reproduces those numbers exactly and a
    multi-DOF cell exercises the matrix-valued path.
    """
    d00 = np.asarray(model.h00, dtype=complex)
    d01 = np.asarray(model.h01, dtype=complex)
    nd = d00.shape[0]
    rng = np.random.default_rng(seed)
    phi: dict = {}
    base: dict = {}
    for i in range(n_cells):
        for a in range(i - reach, i + reach + 1):
            for b in range(i - reach, i + reach + 1):
                if not (0 <= a < n_cells and 0 <= b < n_cells):
                    continue
                key = (a - i, b - i)
                if key not in base:
                    raw = rng.normal(size=(nd, nd, nd))
                    base[key] = cubic * (raw + raw.transpose(0, 2, 1)) / 2.0
                phi[(i, a, b)] = base[key]
    return freeze(d00, d01, d01.conj().T, phi, n_cells,
                  name=name or f"{model.name}_L{n_cells}",
                  extra_meta={"toy": model.name, "cubic": cubic,
                              "reach": reach, "seed": seed}, **kw)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bed", required=True, type=Path,
                    help="stored device dir with dynamical_matrix.mat + "
                         "fc3_blocks.hdf5")
    ap.add_argument("--cells", type=int, required=True)
    ap.add_argument("--axis", default="z", choices=["x", "y", "z"],
                    help="transport direction, matching the bed's config")
    ap.add_argument("--q", type=float, nargs=2, default=(0.0, 0.0),
                    help="transverse q; NOT Gamma-only for a film -- name the "
                         "slice beside every number it produces")
    ap.add_argument("--nfreq-pos", type=int, default=300)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--solver", default="anderson",
                    choices=["linear", "anderson", "jfnk", "anderson+jfnk"])
    ap.add_argument("--mixing", type=float, default=0.2)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--tl", type=float, default=305.0)
    ap.add_argument("--tr", type=float, default=295.0)
    ap.add_argument("--zero-mode-projection", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)

    bed = build_frozen(
        a.bed, a.cells, name=a.name, tdir="xyz".index(a.axis), q=tuple(a.q),
        nfreq_pos=a.nfreq_pos, max_scba_iter=a.max_iter, solver=a.solver,
        anderson_mixing=a.solver != "linear", mixing=a.mixing, scba_tol=a.tol,
        t_left=a.tl, t_right=a.tr, n_threads=a.threads,
        zero_mode_projection=a.zero_mode_projection, verbose=True)
    out = a.out or (OUT / f"{bed.name}.npz")
    bed.save(out)
    print(f"\n{bed.name}: {bed.n_slabs} cells x {bed.n_dof} dof, "
          f"{bed.freqs_thz.size} freqs (dw={bed.dw_thz:.4f} THz, "
          f"fmax={bed.meta['fmax_thz']:.1f}), p={bed.p}")
    print(f"  converged={bed.meta['converged']} "
          f"resid={bed.meta['scba_residual']:.3e} "
          f"conservation={bed.meta['conservation_err']:.3e}")
    print(f"  a pure-tail block needs N >= R + 2(p + m_edge) + 1; at p="
          f"{bed.p} and m_edge=2 this bed reaches R <= {bed.n_slabs - 2*(bed.p+2) - 1}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
