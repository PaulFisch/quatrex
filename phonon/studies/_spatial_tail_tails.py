"""Does long-range G generate a transport-relevant part of Sigma?

The first gate of the spatially analytic tail programme, and a stop: if the
long-propagation part of the self-energy is negligible in the OBSERVABLE and
not merely in a block norm, the modal route is aimed at something the transport
does not need and the programme ends here.

Three measurements, on one frozen SCBA state.

**The shell decomposition.** ``Sigma`` is bilinear in ``G``, so splitting the
legs by distance shell ``m = |K - K'|`` gives ``Sigma_R = sum_{m,m'}
Sigma_R^{(m,m')}`` exactly. A ``g_cutoff`` sweep gives only the partial sums,
and they are cumulative -- raising the band changes blocks that already
existed, through interference -- so a difference of two of them is not the
contribution of a shell. The sweep is also uninformative beyond ``R > 2p+3``,
where the ``b = 3`` reference is identically zero and the discarded share is 1
for free.

**The 2x2 observable factorial.** The dense Dyson operator is inverted, not
recursed, so a self-energy that is not block-tridiagonal is solved exactly and
all four arms cost one solve each:

    A  pin |I-J|<=1, band 3      the production kernel
    B  pin |I-J|<=1, band full   long-range G through the RETAINED blocks
    C  no pin,       band 3      the vertex-near tail alone
    D  no pin,       band full   the reference

``C -> D`` is the gate. It asks the programme's own question -- given that the
pin has already been removed, does widening ``G`` move a current? -- and
nothing in the tree has ever measured it. ``|D - C - B + A|`` is the
interaction: if it is small the two truncations are separable and the tree's
existing one-at-a-time measurements compose.

**First-order sensitivity.** A block norm is refused as the criterion (the
proposal's Sec. 11): a coherent long-range term can be small in Frobenius norm
and large in the current. ``dSigma^R`` is built through the SAME
``build_retarded`` as the reference -- every branch is linear in
``Sigma^> - Sigma^<``, so the difference may be pushed through and no
Kramers-Kronig mismatch enters -- and propagated as ``dG^R = G^R dSigma^R G^R``
with the three-term ``dG^<``. The contact terms of ``Sigma^<_tot`` are
included; leaving them out is the easiest way to get a wrong answer, since they
dominate the source.

Run:
    QTX_ARRAY_MODULE=numpy python phonon/studies/_spatial_tail_tails.py \
        --bed phonon/studies/out/spatial_bed/si16_lin.npz --out cluster/sptail/
    QTX_ARRAY_MODULE=numpy python phonon/studies/_spatial_tail_tails.py --chain
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QTX_ARRAY_MODULE", "numpy")

# A long cluster run whose stdout is a pipe is block-buffered, so `tortin.py
# tail` shows an empty log for the whole run and there is no progress signal at
# all. The dense solver's own prints do not flush.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):        # pragma: no cover - odd stdout
    pass

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "phonon")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from studies._spatial_bed import FrozenBed, OUT                  # noqa: E402

DEFAULT_BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5)]


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


def shell_bins_for(n_slabs: int, bins=None):
    """Contiguous bins covering every leg distance the device can carry."""
    bins = list(DEFAULT_BINS if bins is None else bins)
    top = bins[-1][1]
    if top < n_slabs - 1:
        bins.append((top + 1, n_slabs - 1))
    return [(lo, min(hi, n_slabs - 1)) for lo, hi in bins if lo <= n_slabs - 1]


def sigma_dense(bed: FrozenBed, *, sigma_cutoff=None, g_cutoff=None,
                shell_bins=None, n_threads=None):
    """``(Sigma^<, Sigma^>)`` as dense ``(nfreq, N_D, N_D)``, plus the shells."""
    from solver.dense import _scatter_blocks
    from solver.se_finite import compute_phph_self_energy_finite_multi_slab

    gl = bed.blocks(bed.g_lesser, g_cutoff)
    gg = bed.blocks(bed.g_greater, g_cutoff)
    shells_out = {} if shell_bins is not None else None
    sl_b, sg_b = compute_phph_self_energy_finite_multi_slab(
        gl, gg, bed.phi, bed.n_slabs, bed.freqs_thz, bed.dw_thz,
        sigma_cutoff=sigma_cutoff, g_cutoff=g_cutoff, n_threads=n_threads,
        shell_bins=shell_bins, shells_out=shells_out)

    def densify(blocks):
        out = np.zeros((bed.freqs_thz.size, bed.n_d, bed.n_d), dtype=complex)
        _scatter_blocks(out, blocks, bed.n_dof)
        return out

    sl, sg = densify(sl_b), densify(sg_b)
    shells = None
    if shells_out is not None:
        shells = {k: (densify(a), densify(b)) for k, (a, b) in shells_out.items()}
    return sl, sg, shells


def decompressed_blocks(bed: FrozenBed, mat, band: int, *, rank=None,
                        eps: float = 1e-3, m_edge: int = 2):
    r"""``{(K, K'): block}`` with ``|K-K'| > band`` supplied by a modal fit.

    The proposal's Eq. (44) hybrid, on the Keldysh object directly: exact inside
    the band, exponential continuation outside, and piecewise rather than
    additive so the two never double count. The exponents come from
    block-ESPRIT on the interior sequence -- route A of the proposal's Sec. 8,
    the cheapest and the one that does not preserve the matrix sign structure,
    which is why the congruence route is measured beside it.

    One fit per frequency, taken at an interior anchor, and reused at every
    cell pair: that IS the translation-invariance assumption, and
    ``eps_Toeplitz`` is what says how good it is.
    """
    from quatrex.phonon.spatial_hankel import matrix_pencil

    nd, n = bed.n_dof, bed.n_slabs
    anchor = bed.p + m_edge
    span = max(3, n - 2 * anchor)
    out = {}
    n_fitted = 0
    fits = []
    for iw in range(bed.freqs_thz.size):
        seq = [mat[iw, anchor * nd:(anchor + 1) * nd,
                   (anchor + r) * nd:(anchor + r + 1) * nd]
               for r in range(span)]
        if np.abs(np.asarray(seq)).max() <= 0.0:
            fits.append(None)
            continue
        try:
            fits.append(matrix_pencil(seq, rank=rank, eps=eps))
            n_fitted += 1
        except np.linalg.LinAlgError:
            fits.append(None)

    for k in range(n):
        for kp in range(n):
            d = abs(k - kp)
            blk = np.empty((bed.freqs_thz.size, nd, nd), dtype=complex)
            if d <= band:
                blk[:] = mat[:, k * nd:(k + 1) * nd, kp * nd:(kp + 1) * nd]
            else:
                for iw, f in enumerate(fits):
                    blk[iw] = (mat[iw, k * nd:(k + 1) * nd,
                                   kp * nd:(kp + 1) * nd] if f is None
                               else f.block(d))
            out[(k, kp)] = blk
    return out, n_fitted


def hybrid_retarded(bed: FrozenBed, band: int, *, rank=None, eps: float = 1e-3,
                    m_edge: int = 2, accept: float = 1e-2):
    r"""``G~^R``: exact inside ``band``, modal continuation outside.

    The proposal's Eq. (44), on the RETARDED function -- which is the object the
    spatial modal picture actually represents, and which the rank measurement
    says needs 2-3 exponentials where ``G^<`` needs 4-5. That gap is the whole
    reason route B exists.

    The model is used only where the model FITS. ``accept`` bounds the fit's own
    reconstruction error on the sequence it was fitted to; above it the exact
    blocks are kept at that frequency and the sample is counted as refused.
    Without that guard the substitution is applied at band edges, where the
    modal range is hundreds of cells and nothing has decayed, and the far blocks
    come back wrong by two orders -- which measures the bed's damping, not the
    representation.

    Returns ``(G~^R, stats)``. The far-block error is reported against the exact
    blocks it replaces, as a median and a worst case, so a congruence result can
    be attributed to the reconstruction rather than to the congruence.
    """
    from quatrex.phonon.spatial_hankel import ExponentialSeries, matrix_pencil

    nd, n = bed.n_dof, bed.n_slabs
    dropped_w = kept_w = 0.0
    anchor_i = bed.p + m_edge
    span = max(3, n - 2 * anchor_i)
    out = np.array(bed.g_retarded, copy=True)
    n_fitted = n_refused = 0
    errs = []
    for iw in range(bed.freqs_thz.size):
        seq = [bed.g_retarded[iw, anchor_i * nd:(anchor_i + 1) * nd,
                              (anchor_i + r) * nd:(anchor_i + r + 1) * nd]
               for r in range(span)]
        if np.abs(np.asarray(seq)).max() <= 0.0:
            continue
        try:
            fit = matrix_pencil(seq, rank=rank, eps=eps)
        except np.linalg.LinAlgError:
            n_refused += 1
            continue
        if float(fit.rel_error(seq).max()) > accept:
            n_refused += 1
            continue
        # An outward continuation must not grow. The pencil legitimately
        # returns |xi| > 1 -- in a two-terminal device that branch carries the
        # wave from the FAR contact -- but using it to extrapolate away from the
        # anchor diverges, and the far blocks come back wrong by a factor 40
        # rather than by a few percent. Dropped, with the weight it carried
        # recorded so the drop is visible.
        keep = np.abs(fit.xi) <= 1.0 + 1e-9
        if not keep.any():
            n_refused += 1
            continue
        dropped_w += float(np.linalg.norm(fit.residues[~keep]))
        kept_w += float(np.linalg.norm(fit.residues[keep]))
        fit = ExponentialSeries(xi=fit.xi[keep],
                                residues=fit.residues[keep],
                                spectrum=fit.spectrum)
        n_fitted += 1
        for k in range(n):
            for kp in range(n):
                d = abs(k - kp)
                if d <= band:
                    continue
                ref = bed.g_retarded[iw, k * nd:(k + 1) * nd,
                                     kp * nd:(kp + 1) * nd]
                blk = fit.block(d)
                out[iw, k * nd:(k + 1) * nd, kp * nd:(kp + 1) * nd] = blk
                den = np.linalg.norm(ref)
                if den > 0.0:
                    errs.append(float(np.linalg.norm(blk - ref) / den))
    stats = {"n_fitted": n_fitted, "n_refused": n_refused,
             "growing_weight": dropped_w / (dropped_w + kept_w + 1e-300),
             "far_block_err": float(np.max(errs)) if errs else 0.0,
             "far_block_err_median": float(np.median(errs)) if errs else 0.0}
    return out, stats


def sigma_congruence(bed: FrozenBed, band: int, *, rank=None, eps=1e-3,
                     m_edge: int = 2, n_threads=None):
    r"""Route B: modal ``G~^R``, then ``G~^{<,>} = G~^R Sigma_tot^{<,>} G~^A``.

    ``G~^A`` is literally ``G~^R`` conjugate-transposed, so ``i G~^{<,>}`` is a
    congruence of ``i Sigma_tot^{<,>}`` and inherits its sign structure exactly.
    A direct exponential fit of ``G^<`` has no such guarantee, which is the
    proposal's Sec. 9 point and the reason to measure the two side by side.

    ``Sigma_tot`` carries the CONTACT terms as well as the anharmonic source;
    they dominate it, and leaving them out is the standard way to get this
    wrong.
    """
    from solver.dense import _device_g_blocks, _scatter_blocks
    from solver.se_finite import compute_phph_self_energy_finite_multi_slab

    g_r, stats = hybrid_retarded(bed, band, rank=rank, eps=eps, m_edge=m_edge)
    g_a = g_r.conj().transpose(0, 2, 1)
    sig_l_tot = (bed.obc["Sigma_L_lesser"] + bed.obc["Sigma_R_lesser"]
                 + bed.sigma_lesser)
    sig_g_tot = (bed.obc["Sigma_L_greater"] + bed.obc["Sigma_R_greater"]
                 + bed.sigma_greater)
    g_l = g_r @ sig_l_tot @ g_a
    g_g = g_r @ sig_g_tot @ g_a

    gl = _device_g_blocks(g_l, bed.n_slabs, bed.n_dof, None, has_q_axis=False)
    gg = _device_g_blocks(g_g, bed.n_slabs, bed.n_dof, None, has_q_axis=False)
    sl_b, sg_b = compute_phph_self_energy_finite_multi_slab(
        gl, gg, bed.phi, bed.n_slabs, bed.freqs_thz, bed.dw_thz,
        sigma_cutoff=None, g_cutoff=None, n_threads=n_threads)

    def densify(blocks):
        o = np.zeros((bed.freqs_thz.size, bed.n_d, bed.n_d), dtype=complex)
        _scatter_blocks(o, blocks, bed.n_dof)
        return o

    keldysh_err = float(np.abs(g_l - bed.g_lesser).max()
                        / (np.abs(bed.g_lesser).max() + 1e-300))
    return densify(sl_b), densify(sg_b), dict(
        stats, keldysh_err=keldysh_err,
        psd_lesser=psd_negative_weight(g_l, bed.pos_mask),
        psd_lesser_exact=psd_negative_weight(bed.g_lesser, bed.pos_mask))


def psd_negative_weight(mat, pos_mask, *, stride: int = 17) -> float:
    r"""Worst share of ``i M``'s spectrum that is negative, over positive
    frequencies.

    Two things it is easy to get wrong here, and both give a number near 1 that
    reads as a catastrophic failure.

    ``i M`` and not ``-i M``: this tree sets ``Sigma^< = -i n Gamma``, so ``i``
    is the positive convention (the proposal's is the other one).

    And POSITIVE frequencies only. The symmetric axis carries the
    ``Sigma^<(-w) <-> Sigma^>(w)`` image, where the Bose factor changes sign, so
    ``i G^<`` is negative-definite for ``w < 0`` by construction. Sweeping the
    whole axis reports ~0.999 negative weight for an exact, perfectly physical
    ``G^<``.

    Weighted by spectral mass rather than maximised over frequency. The
    unweighted maximum is taken BELOW the band, where the contacts inject
    nothing, ``G^<`` is sourced entirely by the anharmonic ``Sigma_s^<`` -- which
    is itself not positive -- and the negative fraction is 1.000 on an amplitude
    four orders below the band's. That is a true statement about a negligible
    part of the object and it drowns the question being asked.

    Sampled rather than swept: an ``eigvalsh`` at every frequency costs more
    than the bubble it is checking.
    """
    idx = np.where(np.asarray(pos_mask))[0][::max(1, stride)]
    neg = tot = 0.0
    for iw in idx:
        h = 1j * mat[iw]
        h = 0.5 * (h + h.conj().T)
        ev = np.linalg.eigvalsh(h)
        neg += float(np.sum(np.abs(ev[ev < 0.0])))
        tot += float(np.sum(np.abs(ev)))
    return neg / (tot + 1e-300)


def sigma_decompressed(bed: FrozenBed, band: int, *, rank=None, eps=1e-3,
                       m_edge: int = 2, n_threads=None):
    """The wide bubble with far legs supplied modally instead of stored."""
    from solver.dense import _scatter_blocks
    from solver.se_finite import compute_phph_self_energy_finite_multi_slab

    gl, n_l = decompressed_blocks(bed, bed.g_lesser, band, rank=rank, eps=eps,
                                  m_edge=m_edge)
    gg, n_g = decompressed_blocks(bed, bed.g_greater, band, rank=rank, eps=eps,
                                  m_edge=m_edge)
    sl_b, sg_b = compute_phph_self_energy_finite_multi_slab(
        gl, gg, bed.phi, bed.n_slabs, bed.freqs_thz, bed.dw_thz,
        sigma_cutoff=None, g_cutoff=None, n_threads=n_threads)

    def densify(blocks):
        o = np.zeros((bed.freqs_thz.size, bed.n_d, bed.n_d), dtype=complex)
        _scatter_blocks(o, blocks, bed.n_dof)
        return o

    return densify(sl_b), densify(sg_b), (n_l, n_g)


def block_mask(bed: FrozenBed, sigma, cells_per_block: int):
    r"""``Sigma`` with ``|I//m - J//m| > 1`` zeroed: what reblocking discards.

    Reblocking changes the partition, not the physics, and the dense Dyson solve
    has no block-tridiagonal restriction, so masking IS the exact statement of
    the reblocked arm. Regenerating a device would be a null comparison.
    """
    nd, n = bed.n_dof, bed.n_slabs
    out = np.zeros_like(sigma)
    blk = np.arange(n) // int(cells_per_block)
    for i in range(n):
        for j in range(n):
            if abs(blk[i] - blk[j]) <= 1:
                out[..., i * nd:(i + 1) * nd, j * nd:(j + 1) * nd] = \
                    sigma[..., i * nd:(i + 1) * nd, j * nd:(j + 1) * nd]
    return out


def sigma_semiseparable(bed: FrozenBed, sigma_l, sigma_g, rank: int):
    r"""Arm S: the reference ``Sigma`` carried as a bidirectional semiseparable
    operator of state size ``rank`` per direction, then decompressed.

    This is the cost table's missing accuracy cell. Arms E and F fit the GREEN
    function and rebuild ``Sigma`` from it; this one compresses ``Sigma``
    itself, which is what an augmented Dyson would actually carry, and it is
    the only arm whose block width ``d + r+ + r-`` is a number the RGF cost
    depends on.

    The compression is applied per frequency and then projected onto the
    anti-Hermitian subspace with ``(M - M^H)/2``. That projection was put in
    to repair a truncation and measures as a no-op: ``_sss_realisation``
    obtains the upper triangle by flipping the cell order and running the same
    algorithm, so the two truncated SVDs cut conjugate-related subspaces and
    ``Sigma^< = -(Sigma^<)^H`` survives a rank cut to roundoff. It is kept as a
    guard -- the relation is what makes ``i Sigma^<`` Hermitian and lets
    positivity be a statement about a spectrum, and it costs one subtraction --
    but the structure is preserved by the realisation, not by this line. See
    ``test_the_realisation_preserves_hermitian_symmetry_under_truncation``.
    """
    from quatrex.phonon.spatial_operator import SemiSepOperator

    nd, nw = bed.n_dof, sigma_l.shape[0]
    out = []
    widths = []
    for mat in (sigma_l, sigma_g):
        acc = np.empty_like(mat)
        for iw in range(nw):
            op = SemiSepOperator.from_dense(mat[iw], nd, rank=rank)
            widths.append(op.augmented_block)
            d = op.to_dense()
            acc[iw] = 0.5 * (d - d.conj().T)
        out.append(acc)
    anti = max(float(np.abs(m + m.conj().transpose(0, 2, 1)).max()
                     / max(np.abs(m).max(), 1e-300))
               for m in (sigma_l, sigma_g))
    err = max(float(np.abs(a - m).max() / max(np.abs(m).max(), 1e-300))
              for a, m in zip(out, (sigma_l, sigma_g)))
    return out[0], out[1], {"rank": rank, "sigma_err": err,
                            "aug_width": int(max(widths)),
                            "ref_antiherm": anti}


def solve_arm(bed: FrozenBed, sigma_l, sigma_g):
    """One frozen Dyson/Keldysh solve and its observables."""
    from phonon_inputs.constants import THZ_TO_RAD
    from solver.leads import solve_green_batch
    from solver.observables import (meir_wingreen_spectra, slab_absorption,
                                    sumrule_D_omega)
    from solver.retarded import build_retarded

    sigma_r = build_retarded(sigma_l, sigma_g, bed.freqs_thz, method="fft")
    z2 = (bed.freqs_thz.astype(complex)) ** 2
    g_r, g_l, g_g = solve_green_batch(z2, bed.h_d, bed.obc,
                                      sigma_r, sigma_l, sigma_g)
    spec_l, spec_r = meir_wingreen_spectra(g_l, g_g, bed.obc, bed.freqs_thz,
                                           bed.n_dof, bed.n_slabs)
    p_abs = slab_absorption(sigma_l, sigma_g, g_l, g_g, bed.freqs_thz,
                            bed.n_dof, bed.n_slabs)
    j_s_spec = p_abs.sum(axis=-1)
    scale = bed.dw_thz * 1e12
    return {
        "sigma_retarded": sigma_r, "g_retarded": g_r,
        "g_lesser": g_l, "g_greater": g_g,
        "spec_L": spec_l, "spec_R": spec_r, "p_abs": p_abs,
        "J_L": float(np.sum(spec_l[bed.pos_mask]) * scale),
        "J_R": float(np.sum(spec_r[bed.pos_mask]) * scale),
        "J_s": float(np.sum(j_s_spec[bed.pos_mask]) * scale),
        "D": float(np.sum(sumrule_D_omega(spec_l, spec_r, j_s_spec)[bed.pos_mask])
                   * scale),
        "P_slab": p_abs[bed.pos_mask].sum(axis=0) * scale,
        "omega_rad": bed.freqs_thz * THZ_TO_RAD,
    }


def first_order(bed: FrozenBed, ref, d_sigma_l, d_sigma_g):
    r"""``dJ`` from ``dSigma`` by the linearised Dyson/Keldysh relations.

    ``build_retarded`` is linear in ``Sigma^> - Sigma^<`` on every branch, so
    the difference goes through it directly and the reference's own
    Kramers-Kronig reconstruction is reused rather than approximated.
    """
    from solver.observables import (meir_wingreen_spectra, slab_absorption,
                                    sumrule_D_omega)
    from solver.retarded import build_retarded

    d_sig_r = build_retarded(d_sigma_l, d_sigma_g, bed.freqs_thz, method="fft")
    g_r, g_a = ref["g_retarded"], ref["g_retarded"].conj().transpose(0, 2, 1)
    d_g_r = g_r @ d_sig_r @ g_r
    d_g_a = d_g_r.conj().transpose(0, 2, 1)

    sig_l_tot = (bed.obc["Sigma_L_lesser"] + bed.obc["Sigma_R_lesser"]
                 + ref["sigma_lesser"])
    sig_g_tot = (bed.obc["Sigma_L_greater"] + bed.obc["Sigma_R_greater"]
                 + ref["sigma_greater"])
    d_g_l = (d_g_r @ sig_l_tot @ g_a + g_r @ d_sigma_l @ g_a
             + g_r @ sig_l_tot @ d_g_a)
    d_g_g = (d_g_r @ sig_g_tot @ g_a + g_r @ d_sigma_g @ g_a
             + g_r @ sig_g_tot @ d_g_a)

    d_spec_l, d_spec_r = meir_wingreen_spectra(
        d_g_l, d_g_g, bed.obc, bed.freqs_thz, bed.n_dof, bed.n_slabs)
    # slab_absorption is bilinear: d(Sigma G) = dSigma G + Sigma dG.
    d_p = (slab_absorption(d_sigma_l, d_sigma_g, ref["g_lesser"],
                           ref["g_greater"], bed.freqs_thz, bed.n_dof,
                           bed.n_slabs)
           + slab_absorption(ref["sigma_lesser"], ref["sigma_greater"],
                             d_g_l, d_g_g, bed.freqs_thz, bed.n_dof,
                             bed.n_slabs))
    d_js = d_p.sum(axis=-1)
    scale = bed.dw_thz * 1e12
    return {
        "dJ_L": float(np.sum(d_spec_l[bed.pos_mask]) * scale),
        "dJ_R": float(np.sum(d_spec_r[bed.pos_mask]) * scale),
        "dJ_s": float(np.sum(d_js[bed.pos_mask]) * scale),
        "dD": float(np.sum(sumrule_D_omega(d_spec_l, d_spec_r, d_js)[bed.pos_mask])
                    * scale),
        "dP_slab": d_p[bed.pos_mask].sum(axis=0) * scale,
    }


# ---------------------------------------------------------------------------
# spatial statistics
# ---------------------------------------------------------------------------


def discarded_fraction(bed: FrozenBed, sigma, cells_per_block: int) -> float:
    """Share of ``|Sigma|`` the reblocked pin drops, as a weight."""
    nd, n = bed.n_dof, bed.n_slabs
    blk = np.arange(n) // int(cells_per_block)
    num = den = 0.0
    for i in range(n):
        for j in range(n):
            w = float(np.abs(sigma[..., i * nd:(i + 1) * nd,
                                   j * nd:(j + 1) * nd]).sum())
            den += w
            if abs(blk[i] - blk[j]) > 1:
                num += w
    return num / (den + 1e-300)


def block_of(mat, bed, i, j):
    nd = bed.n_dof
    return mat[..., i * nd:(i + 1) * nd, j * nd:(j + 1) * nd]


def admissible_i(bed, r_out, m_edge: int):
    """Interior anchors for output separation ``r_out``.

    Two edge effects, not one: the vertex loses its ``(alpha, beta)`` terms
    within ``p`` of an end, and the OBC matches the UNDRESSED lead while the
    interior is dressed, so ``G`` near a contact carries the growing branch
    too. A pure-tail block exists only when ``N >= R + 2(p + m_edge) + 1``.
    """
    lo = bed.p + m_edge
    hi = bed.n_slabs - 1 - bed.p - m_edge - r_out
    return list(range(lo, hi + 1))


def distance_profile(sigma, bed, *, m_edge: int = 0, weight=None, freqs=None):
    """``{R: (mean ||Sigma_{I,I+R}||_F, n_anchors)}`` over interior anchors.

    ``freqs`` restricts the frequency samples the norm is taken over. That is
    not a detail: a device whose modal range varies by two orders across the
    band has its integrated profile set by whichever frequency is least damped,
    and on the gapped chain that is the band-bottom sample, where the range is
    600 cells and no tail is resolved at all. The proposal asks for the split
    frequency-resolved AND frequency-integrated for this reason.
    """
    sl = slice(None) if freqs is None else freqs
    sig = sigma[sl]
    w = np.ones(sig.shape[0]) if weight is None else np.asarray(weight)[sl]
    prof = {}
    for r_out in range(bed.n_slabs):
        idx = admissible_i(bed, r_out, m_edge)
        if not idx:
            continue
        vals = [np.linalg.norm(block_of(sig, bed, i, i + r_out)
                               * w[:, None, None]) for i in idx]
        prof[r_out] = (float(np.mean(vals)), len(idx))
    return prof


def modal_range_by_frequency(bed, *, m_edge: int = 2):
    """Longest decaying modal range at each frequency, in cells.

    The quantity that says whether a bed can show a tail at all: a range longer
    than the device means ``G`` has not decayed anywhere in it.
    """
    from quatrex.phonon.spatial_modes import bloch_modes

    nd, anchor = bed.n_dof, bed.p + m_edge
    out = np.full(bed.freqs_thz.size, np.nan)
    for iw in np.where(bed.pos_mask)[0]:
        w = float(bed.freqs_thz[iw])
        s = bed.sigma_retarded[iw]
        d = s[anchor * nd:(anchor + 1) * nd, anchor * nd:(anchor + 1) * nd]
        o = s[anchor * nd:(anchor + 1) * nd,
              (anchor + 1) * nd:(anchor + 2) * nd]
        try:
            m = bloch_modes((w * w) * np.eye(nd) - bed.d00 - d,
                            -bed.d01 - o, -bed.d10 - o.conj().T)
        except np.linalg.LinAlgError:
            continue
        lam = np.abs(np.asarray(m.lam))
        lam = lam[(lam > 0.0) & (lam < 1.0)]
        if lam.size:
            out[iw] = float(np.max(-1.0 / np.log(lam)))
    return out


def toeplitz_residual(sigma, bed, *, m_edge: int = 0):
    r"""``eps_Toeplitz(R)``: is the interior tail a function of ``R`` alone?

    Expected to FAIL on the Keldysh part, and derivably so: the source-resolved
    ``G^<`` carries a residue ``(lambda_m lambda_n^*)^I``, which makes ``Sigma``
    semiseparable rather than Toeplitz. The number is reported because that
    prediction is what it tests.
    """
    out = {}
    for r_out in range(bed.n_slabs):
        idx = admissible_i(bed, r_out, m_edge)
        if len(idx) < 2:
            continue
        blocks = np.stack([block_of(sigma, bed, i, i + r_out) for i in idx])
        mean = blocks.mean(axis=0)
        num = np.sqrt(np.sum(np.abs(blocks - mean) ** 2))
        den = np.sqrt(np.sum(np.abs(blocks) ** 2))
        out[r_out] = float(num / (den + 1e-300))
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run(bed: FrozenBed, *, m_edge: int = 2, n_threads=None, verbose=True):
    bins = shell_bins_for(bed.n_slabs)
    arms = {}
    shells = None
    for tag, (sc, gc) in (("A", (1, 3)), ("B", (1, None)),
                          ("C", (None, 3)), ("D", (None, None))):
        sb = bins if tag == "D" else None
        sl, sg, sh = sigma_dense(bed, sigma_cutoff=sc, g_cutoff=gc,
                                 shell_bins=sb, n_threads=n_threads)
        if sh is not None:
            shells = sh
        arm = solve_arm(bed, sl, sg)
        arm.update(sigma_lesser=sl, sigma_greater=sg,
                   sigma_cutoff=sc, g_cutoff=gc)
        arms[tag] = arm
        if verbose:
            print(f"  arm {tag} (sigma_cutoff={sc}, g_cutoff={gc}): "
                  f"J_L={arm['J_L']:.6e} J_R={arm['J_R']:.6e} "
                  f"J_s={arm['J_s']:.3e} D={arm['D']:.3e}", flush=True)

    # The modal-extended arm: far legs supplied by an exponential fit rather
    # than stored, everything else identical to D.
    try:
        sl_m, sg_m, n_fit = sigma_decompressed(bed, 3, m_edge=m_edge,
                                               n_threads=n_threads)
        arm = solve_arm(bed, sl_m, sg_m)
        arm.update(sigma_lesser=sl_m, sigma_greater=sg_m,
                   sigma_cutoff=None, g_cutoff="modal>3", n_fitted=n_fit)
        arms["E"] = arm
        if verbose:
            print(f"  arm E (modal legs beyond band 3, {n_fit[0]} frequencies "
                  f"fitted): J_L={arm['J_L']:.6e} D={arm['D']:.3e}", flush=True)
    except Exception as exc:                       # pragma: no cover
        print(f"  arm E failed: {type(exc).__name__}: {exc}", flush=True)

    # Arm F, route B: the congruence. The modal continuation is applied to G^R
    # -- the object the modal picture represents, and the one that needs 2-3
    # exponentials where G^< needs 4-5 -- and G^{<,>} is then REBUILT rather
    # than fitted, so positivity is structural.
    try:
        sl_c, sg_c, info_c = sigma_congruence(bed, 3, m_edge=m_edge,
                                              n_threads=n_threads)
        arm = solve_arm(bed, sl_c, sg_c)
        arm.update(sigma_lesser=sl_c, sigma_greater=sg_c,
                   sigma_cutoff=None, g_cutoff="congr>3", **info_c)
        arms["F"] = arm
        if verbose:
            print(f"  arm F (congruence, modal G^R beyond band 3): "
                  f"J_L={arm['J_L']:.6e}  {info_c['n_fitted']} fitted / "
                  f"{info_c['n_refused']} refused  far-block err "
                  f"{info_c['far_block_err_median']:.2e} med / "
                  f"{info_c['far_block_err']:.2e} max  G^< err "
                  f"{info_c['keldysh_err']:.2e}  neg-weight "
                  f"{info_c['psd_lesser']:.2e} vs exact "
                  f"{info_c['psd_lesser_exact']:.2e}", flush=True)
    except Exception as exc:                       # pragma: no cover
        print(f"  arm F failed: {type(exc).__name__}: {exc}", flush=True)

    ref = arms["D"]
    # The reblocked arms: the same reference Sigma, masked at m cells per block.
    for m_cells in (2, 3, 4):
        if bed.n_slabs // m_cells < 2:
            continue
        msk_l = block_mask(bed, ref["sigma_lesser"], m_cells)
        msk_g = block_mask(bed, ref["sigma_greater"], m_cells)
        arm = solve_arm(bed, msk_l, msk_g)
        arm.update(sigma_lesser=msk_l, sigma_greater=msk_g,
                   sigma_cutoff=f"blk{m_cells}", g_cutoff=None,
                   discarded=discarded_fraction(bed, ref["sigma_lesser"],
                                                m_cells))
        arms[f"R{m_cells}"] = arm
        if verbose:
            print(f"  arm R{m_cells} (pin at {m_cells} cells/block, "
                  f"{arm['discarded'] * 100:.1f} % of |Sigma| discarded): "
                  f"J_L={arm['J_L']:.6e}", flush=True)

    # Arm S: Sigma itself carried semiseparably. Ranks chosen to bracket the
    # measured quasiseparable rank of Sigma on this bed, so the table shows
    # what the accuracy costs rather than one point of it.
    for rk in (2, 4, 6, 8):
        try:
            sl_s, sg_s, info_s = sigma_semiseparable(
                bed, ref["sigma_lesser"], ref["sigma_greater"], rk)
            arm = solve_arm(bed, sl_s, sg_s)
            arm.update(sigma_lesser=sl_s, sigma_greater=sg_s,
                       sigma_cutoff=f"sss{rk}", g_cutoff=None, **info_s)
            arms[f"S{rk}"] = arm
            if verbose:
                print(f"  arm S{rk} (semiseparable Sigma, rank {rk}/direction, "
                      f"augmented block {info_s['aug_width']} vs 2-cell "
                      f"reblock {2 * bed.n_dof}): J_L={arm['J_L']:.6e}  "
                      f"|dSigma|/|Sigma|={info_s['sigma_err']:.2e}", flush=True)
        except Exception as exc:                   # pragma: no cover
            print(f"  arm S{rk} failed: {type(exc).__name__}: {exc}", flush=True)

    scale = abs(ref["J_L"]) + 1e-300
    others = [t for t in arms if t != "D"]
    eps = {t: {k: abs(arms[t][k] - ref[k]) / scale
               for k in ("J_L", "J_R", "J_s")} for t in others}
    interaction = abs(ref["J_L"] - arms["C"]["J_L"] - arms["B"]["J_L"]
                      + arms["A"]["J_L"]) / scale

    lin = {t: first_order(bed, ref,
                          ref["sigma_lesser"] - arms[t]["sigma_lesser"],
                          ref["sigma_greater"] - arms[t]["sigma_greater"])
           for t in others}

    prof = distance_profile(ref["sigma_lesser"], bed, m_edge=m_edge)
    toe_s = toeplitz_residual(ref["sigma_lesser"], bed, m_edge=m_edge)
    toe_g = toeplitz_residual(bed.g_lesser, bed, m_edge=m_edge)

    # Frequency-resolved: keep only the samples whose own modal range is short
    # enough for the device to show a tail. Reported beside the integrated
    # profile, never instead of it.
    xi = modal_range_by_frequency(bed, m_edge=m_edge)
    resolved = np.where(np.isfinite(xi) & (xi < bed.n_slabs / 2.0))[0]
    prof_res = (distance_profile(ref["sigma_lesser"], bed, m_edge=m_edge,
                                 freqs=resolved) if resolved.size else None)

    shell_norm = None
    if shells is not None:
        nb = len(bins)
        shell_norm = np.zeros((nb, nb, bed.n_slabs))
        for (m, mp), (sl_s, _) in shells.items():
            for r_out in range(bed.n_slabs):
                idx = admissible_i(bed, r_out, 0)
                if not idx:
                    continue
                shell_norm[m, mp, r_out] = np.mean(
                    [np.linalg.norm(block_of(sl_s, bed, i, i + r_out))
                     for i in idx])

    return {"arms": arms, "eps": eps, "interaction": interaction, "lin": lin,
            "profile": prof, "profile_resolved": prof_res, "xi": xi,
            "n_resolved": int(resolved.size),
            "toeplitz_sigma": toe_s, "toeplitz_g": toe_g,
            "shell_norm": shell_norm, "shell_bins": bins, "m_edge": m_edge}


def report(bed: FrozenBed, res: dict) -> None:
    bins, ref = res["shell_bins"], res["arms"]["D"]
    print(f"\n{bed.name}: {bed.n_slabs} cells x {bed.n_dof} dof, p={bed.p}, "
          f"{bed.freqs_thz.size} freqs (dw={bed.dw_thz:.4f} THz)")
    print(f"  frozen state: converged={bed.meta.get('converged')} "
          f"resid={bed.meta.get('scba_residual'):.2e} "
          f"conservation={bed.meta.get('conservation_err'):.2e}")
    floor = max(3.0 * abs(bed.meta.get("conservation_err", 0.0)),
                float(bed.meta.get("scba_residual", 0.0)))
    print(f"  PRE-REGISTERED negligibility floor: "
          f"max(3*conservation, scba_tol) = {floor:.2e}")

    print("\n  the 2x2 factorial, relative to arm D (reference)")
    print("    arm  sigma_cut  g_cut |  eps(J_L)   eps(J_R)   eps(J_s)  | "
          "first-order dJ_L/J_L")
    labels = {"A": "production pin", "B": "pin, wide G",
              "C": "no pin, band 3",
              "E": "modal legs beyond band 3 (direct fit of G^<)",
              "F": "congruence: modal G^R, G^< rebuilt",
              "R2": "reblock 2 cells/block", "R3": "reblock 3",
              "R4": "reblock 4"}
    for t in [k for k in ("A", "B", "C", "E", "F", "R2", "R3", "R4")
              if k in res["eps"]]:
        lbl = labels[t]
        e, dl = res["eps"][t], res["lin"][t]
        print(f"    {t:3s} {str(res['arms'][t]['sigma_cutoff']):>9} "
              f"{str(res['arms'][t]['g_cutoff']):>6} | "
              f"{e['J_L']:.3e}  {e['J_R']:.3e}  {e['J_s']:.3e}  | "
              f"{dl['dJ_L'] / (abs(ref['J_L']) + 1e-300):.3e}   ({lbl})")
    print(f"    C -> D is the GATE: {res['eps']['C']['J_L']:.3e} against a "
          f"floor of {floor:.2e}  -> "
          f"{'ABOVE (continue)' if res['eps']['C']['J_L'] > floor else 'BELOW (stop)'}")
    print(f"    interaction |D-C-B+A|/|J_L| = {res['interaction']:.3e} "
          "(small = the two truncations are separable)")

    xi = res["xi"]
    fin = xi[np.isfinite(xi)]
    if fin.size:
        print(f"\n  modal range over the band: min {fin.min():.2f}  median "
              f"{np.median(fin):.2f}  max {fin.max():.2f} cells, device "
              f"{bed.n_slabs}\n  -> {res['n_resolved']} of {fin.size} samples "
              f"have a range under half the device and can show a tail")
    print(f"\n  Sigma^< by output distance (interior anchors, m_edge="
          f"{res['m_edge']})")
    print("    R  | n_anch | ||Sigma_R||  all freq | resolved freq only | "
          "eps_Toep(Sig) | eps_Toep(G)")
    pr = res.get("profile_resolved")
    for r_out, (val, n) in sorted(res["profile"].items()):
        ts = res["toeplitz_sigma"].get(r_out)
        tg = res["toeplitz_g"].get(r_out)
        rv = pr.get(r_out, (float("nan"), 0))[0] if pr else float("nan")
        print(f"    {r_out:2d} | {n:6d} | {val:20.4e} | {rv:18.4e} | "
              f"{'      n/a' if ts is None else f'{ts:13.4f}'} | "
              f"{'    n/a' if tg is None else f'{tg:.4f}'}")

    if res["shell_norm"] is not None:
        labels = [f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in bins]
        print("\n  EXACT shell decomposition of Sigma^<: share of ||Sigma_R|| by "
              "leg-distance pair")
        tot = res["shell_norm"].sum(axis=(0, 1))
        for r_out in range(min(bed.n_slabs, 9)):
            if tot[r_out] <= 0:
                continue
            row = res["shell_norm"][:, :, r_out] / tot[r_out]
            top = np.dstack(np.unravel_index(
                np.argsort(-row, axis=None)[:3], row.shape))[0]
            desc = "  ".join(f"({labels[m]},{labels[mp]}) {row[m, mp]:.3f}"
                             for m, mp in top)
            print(f"    R={r_out}: {desc}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bed", type=Path, default=None,
                    help="frozen bed npz from _spatial_bed.py")
    ap.add_argument("--chain", action="store_true",
                    help="build the analytic gapped chain instead")
    ap.add_argument("--dof", type=int, default=1,
                    help="DOF per cell; >1 routes through the multi-DOF "
                         "builder, which is the axis the cost of an augmented "
                         "block turns on (see spatial_representation.md Sec. 18)")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--mixing", type=float, default=0.2)
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--cells", type=int, default=16)
    ap.add_argument("--cubic", type=float, default=3e17,
                    help="toy vertex scale; the bubble prefactor carries the "
                         "SI hbar, so an O(1) vertex gives Sigma ~ 1e-33 THz^2")
    ap.add_argument("--nfreq-pos", type=int, default=120)
    ap.add_argument("--m-edge", type=int, default=2)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    if a.chain and a.dof > 1:
        from studies._spatial_rank_scaling import build
        bed = build("multi", a.dof, a.cells, a.cubic, nfreq_pos=a.nfreq_pos,
                    max_scba_iter=a.max_iter, solver="linear",
                    anderson_mixing=False, mixing=a.mixing, scba_tol=1e-8,
                    divergence_guard=False, n_threads=a.threads, verbose=True,
                    name=f"multi{a.dof}_L{a.cells}")
    elif a.chain:
        from studies._spatial_bed import build_frozen_chain
        from solver.toy_models import gapped_chain
        bed = build_frozen_chain(gapped_chain(), a.cells, cubic=a.cubic,
                                 nfreq_pos=a.nfreq_pos,
                                 max_scba_iter=a.max_iter,
                                 solver="linear", anderson_mixing=False,
                                 mixing=a.mixing, scba_tol=1e-8,
                                 divergence_guard=False, verbose=False,
                                 name=f"chain_L{a.cells}")
    elif a.bed is not None:
        bed = FrozenBed.load(a.bed)
    else:
        ap.error("give --bed or --chain")

    if a.save is not None:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        bed.save(a.save)
        print(f"  saved bed to {a.save}", flush=True)

    res = run(bed, m_edge=a.m_edge, n_threads=a.threads)
    report(bed, res)

    out = a.out or (OUT.parent / "spatial_tail" / f"{bed.name}_tails.npz")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, name=bed.name, n_dof=bed.n_dof, n_slabs=bed.n_slabs,
        eps=np.array(repr(res["eps"])), lin=np.array(repr(res["lin"])),
        interaction=res["interaction"],
        profile=np.array(repr(res["profile"])),
        toeplitz_sigma=np.array(repr(res["toeplitz_sigma"])),
        toeplitz_g=np.array(repr(res["toeplitz_g"])),
        shell_norm=(res["shell_norm"] if res["shell_norm"] is not None
                    else np.zeros(0)),
        shell_bins=np.array(res["shell_bins"]),
        arm_names=np.array(sorted(res["arms"])),
        J=np.array([[res["arms"][t]["J_L"], res["arms"][t]["J_R"],
                     res["arms"][t]["J_s"], res["arms"][t]["D"]]
                    for t in sorted(res["arms"])]),
        meta=np.array(repr(bed.meta)))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
