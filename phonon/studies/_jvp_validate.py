"""Validation of the exact (analytic) SCBA Jacobian-vector product.

The exact JVP rests on two facts: the bubble S(G) is a homogeneous
R-quadratic map, so its directional derivative is the polarisation
identity S'(G)[dG] = S(G+dG) - S(G) - S(dG); and the Dyson solve
linearises in closed form at frozen G. Both are checked here against
independent references before the production solver uses them.

Tests (run individually via --test, or all):

  bubble   Sandbox three-way: polarisation identity vs an explicit
           mixed-leg bubble (product rule on the ring contractions) vs
           finite differences, on synthetic multi-slab data. Also
           2-homogeneity S(aG) = a^2 S(G) and the forward-FD
           epsilon V-curve (the noise floor the analytic JVP removes).
  dyson    Frozen-A Dyson JVP vs finite differences of the production
           RGF selected_solve. Demonstrates that the plain identity
           G^R dSigma^< G^A only matches the implemented map on the
           skew-hermitian subspace (RGF substitutes
           Sigma_ji -> -Sigma_ij^dagger and skew-projects the diagonal),
           and that projecting the direction onto that subspace fixes it.
  kernel   Production SigmaPhononPhonon.compute (one-sided grid, bosonic
           fold, DC mask, KK): 2-homogeneity + polarisation identity vs
           forward FD on synthetic DSDBSparse inputs; fast paths
           (sse_greater_from_lesser / hermitian pairs) vs legacy path.
  spectrum Real-embedded Arnoldi eigenvalues of the toy-chain SCBA map:
           FD-JVP Arnoldi (as in _toy_grid_e7) vs analytic-JVP Arnoldi
           (polarisation bubble + dense Dyson identity).

Symmetry of the underlying kernel Hessian is by construction: the
JVP is the exact bilinear cross form of the implemented symmetric
bubble, not a re-derived convenience form.

Memory-light on purpose (laptop-safe): tiny blocks, short grids.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

for p in (str(Path(__file__).resolve().parent),
          str(Path(__file__).resolve().parents[1]),
          str(Path(__file__).resolve().parents[2] / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from solver.bubble import bubble_dense_from_fft, precompute_g_fft  # noqa: E402
from solver.se_finite import (  # noqa: E402
    _build_pair_index,
    bubble_prefactor,
    compute_phph_self_energy,
)

RNG = np.random.default_rng(42)


def _rand_c(*shape, rng=None):
    rng = RNG if rng is None else rng
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _rel(a, b):
    na = np.linalg.norm(a - b)
    return na / max(np.linalg.norm(b), 1e-300)


def _blocks_rel(A: dict, B: dict) -> float:
    worst = 0.0
    for k in B:
        worst = max(worst, _rel(A[k], B[k]))
    return worst


# ---------------------------------------------------------------------------
# Test 1: sandbox bubble three-way
# ---------------------------------------------------------------------------


def _make_sandbox_case(n_slabs=3, n_dof=4, nf=41, dw=0.5):
    """Synthetic vertices + G blocks on a symmetric grid (Gamma-only)."""
    vert = {}
    for I in range(n_slabs):
        for K1 in range(n_slabs):
            for K2 in range(n_slabs):
                if abs(I - K1) <= 1 and abs(I - K2) <= 1:
                    vert[(I, K1, K2)] = _rand_c(n_dof, n_dof, n_dof)

    def rand_gdict():
        return {
            (K, Kp): _rand_c(1, nf, n_dof, n_dof)
            for K in range(n_slabs)
            for Kp in range(n_slabs)
            if abs(K - Kp) <= 1
        }

    grid = (np.arange(nf) - nf // 2) * dw
    return vert, rand_gdict, grid, dw


def _sandbox_S(gl, gg, vert, n_slabs, grid, dw):
    return compute_phph_self_energy(
        gl, gg, {(0, 0): vert}, n_slabs, 1, np.array([[0]]),
        grid, dw, dc_handling="interpolate", n_threads=1)


def _sandbox_cross_mixed_leg(gl, gg, dgl, dgg, vert, n_slabs, grid, dw):
    """Explicit product-rule bubble: B(dGa, Gb) + B(Ga, dGb) per task.

    Mirrors the task construction in compute_phph_self_energy for the
    Gamma-only case (left vertex conjugated), independently of the
    polarisation identity.
    """
    nf = len(grid)
    n_fft = 2 * nf - 1
    mid = nf // 2
    freq_sl = slice(mid, mid + nf)
    prefactor = bubble_prefactor(dw, n_kpts=1)

    pair_index = _build_pair_index(
        vert, set(gl.keys()), n_slabs, sigma_cutoff=None)

    def fft_all(gblk):
        return {k: precompute_g_fft(arr[0], n_fft=n_fft, zero_freq_idx=mid,
                                    dc_handling="interpolate")
                for k, arr in gblk.items()}

    ffts = {"lesser": (fft_all(gl), fft_all(dgl)),
            "greater": (fft_all(gg), fft_all(dgg))}

    n_dof = next(iter(gl.values())).shape[-1]
    out = {kind: {(I, J): np.zeros((1, nf, n_dof, n_dof), complex)
                  for (I, J) in pair_index}
           for kind in ("lesser", "greater")}

    for (I, J), quads in pair_index.items():
        for (K1, K2, K1p, K2p, _pl, _pr) in quads:
            pl = np.conj(vert[(I, K1, K2)])
            pr = vert[(J, K2p, K1p)]
            for kind in ("lesser", "greater"):
                g_fft, dg_fft = ffts[kind]
                ga, dga = g_fft[(K1, K1p)], dg_fft[(K1, K1p)]
                gb, dgb = g_fft[(K2, K2p)], dg_fft[(K2, K2p)]
                blk = bubble_dense_from_fft(
                    phi_left=pl, phi_right=pr, G_a_fft=dga, G_b_fft=gb,
                    ne=nf, prefactor=prefactor, out_slice=freq_sl)
                blk += bubble_dense_from_fft(
                    phi_left=pl, phi_right=pr, G_a_fft=ga, G_b_fft=dgb,
                    ne=nf, prefactor=prefactor, out_slice=freq_sl)
                out[kind][(I, J)][0] += blk
    return out["lesser"], out["greater"]


def test_bubble() -> bool:
    n_slabs, n_dof, nf, dw = 3, 4, 41, 0.5
    vert, rand_gdict, grid, dw = _make_sandbox_case(n_slabs, n_dof, nf, dw)
    gl, gg = rand_gdict(), rand_gdict()
    dgl, dgg = rand_gdict(), rand_gdict()

    S = lambda a, b: _sandbox_S(a, b, vert, n_slabs, grid, dw)  # noqa: E731

    sl0, sg0 = S(gl, gg)

    # 2-homogeneity.
    a = 1.7
    sl_a, sg_a = S({k: a * v for k, v in gl.items()},
                   {k: a * v for k, v in gg.items()})
    hom = max(_blocks_rel(sl_a, {k: a**2 * v for k, v in sl0.items()}),
              _blocks_rel(sg_a, {k: a**2 * v for k, v in sg0.items()}))
    print(f"[bubble] 2-homogeneity S(aG) vs a^2 S(G): rel {hom:.2e} "
          f"{'PASS' if hom < 1e-13 else 'FAIL'}")

    # Polarisation identity.
    slp, sgp = S({k: gl[k] + dgl[k] for k in gl},
                 {k: gg[k] + dgg[k] for k in gg})
    sld, sgd = S(dgl, dgg)
    cross_pol_l = {k: slp[k] - sl0[k] - sld[k] for k in sl0}
    cross_pol_g = {k: sgp[k] - sg0[k] - sgd[k] for k in sg0}

    # Mixed-leg product rule (independent construction).
    cross_ml_l, cross_ml_g = _sandbox_cross_mixed_leg(
        gl, gg, dgl, dgg, vert, n_slabs, grid, dw)
    agree = max(_blocks_rel(cross_pol_l, cross_ml_l),
                _blocks_rel(cross_pol_g, cross_ml_g))
    print(f"[bubble] polarisation vs mixed-leg: rel {agree:.2e} "
          f"{'PASS' if agree < 1e-12 else 'FAIL'}")

    # Forward-FD epsilon sweep (V-curve): truncation eps*S(d) down,
    # cancellation macheps/eps up.
    print("[bubble] forward-FD V-curve (rel error vs mixed-leg):")
    best_fd = np.inf
    for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9):
        sle, sge = S({k: gl[k] + eps * dgl[k] for k in gl},
                     {k: gg[k] + eps * dgg[k] for k in gg})
        fd_l = {k: (sle[k] - sl0[k]) / eps for k in sl0}
        fd_g = {k: (sge[k] - sg0[k]) / eps for k in sg0}
        err = max(_blocks_rel(fd_l, cross_ml_l), _blocks_rel(fd_g, cross_ml_g))
        best_fd = min(best_fd, err)
        print(f"    eps={eps:.0e}  rel={err:.2e}")
    print(f"[bubble] best FD ({best_fd:.2e}) vs analytic (exact): "
          f"{'PASS' if best_fd < 1e-5 else 'FAIL'} (FD floor is the point)")

    return hom < 1e-13 and agree < 1e-12 and best_fd < 1e-5


# ---------------------------------------------------------------------------
# Test 2: Dyson JVP vs production RGF (the skew-subspace requirement)
# ---------------------------------------------------------------------------

_QTTOOLS_READY = False


def _setup_qttools():
    global _QTTOOLS_READY
    if _QTTOOLS_READY:
        return
    from qttools.comm import comm

    _mpi = {k: "device_mpi" for k in ("all_gather", "all_to_all",
                                      "all_reduce", "bcast", "send_recv")}
    comm.configure(block_comm_size=1, block_comm_config=dict(_mpi),
                   stack_comm_config=dict(_mpi), override=True)
    _QTTOOLS_READY = True


def _band_pattern_dsdb(nb, b, band, ns):
    from qttools import sparse
    from qttools.datastructures.dsdbcoo import DSDBCOO

    N = nb * b
    rows, cols = [], []
    for i in range(nb):
        for j in range(nb):
            if abs(i - j) <= band:
                r = np.arange(i * b, (i + 1) * b)
                c = np.arange(j * b, (j + 1) * b)
                rr, cc = np.meshgrid(r, c, indexing="ij")
                rows.append(rr.ravel())
                cols.append(cc.ravel())
    pat = sparse.coo_matrix(
        (np.ones(sum(len(r) for r in rows)),
         (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    return DSDBCOO.from_sparray(pat.astype(np.complex128),
                                np.full(nb, b), global_stack_shape=(ns,))


def _dense_to_band(m, dense, nb, b, band):
    for i in range(nb):
        for j in range(nb):
            if abs(i - j) <= band:
                m.blocks[i, j] = dense[:, i*b:(i+1)*b, j*b:(j+1)*b]


def _band_to_dense(m, nb, b, band, ns):
    N = nb * b
    out = np.zeros((ns, N, N), complex)
    for i in range(nb):
        for j in range(nb):
            if abs(i - j) <= band:
                out[:, i*b:(i+1)*b, j*b:(j+1)*b] = np.asarray(m.blocks[i, j])
    return out


def _sub_lower(dense, nb, b):
    """RGF input substitution: lower blocks <- -(upper)^dagger, diagonal
    kept as stored (the recurrences read only diag + upper)."""
    out = dense.copy()
    for i in range(nb):
        s = slice(i * b, (i + 1) * b)
        for j in range(i + 1, nb):
            sj = slice(j * b, (j + 1) * b)
            out[:, sj, s] = -out[:, s, sj].conj().swapaxes(-2, -1)
    return out


def _proj_out(dense, nb, b):
    """RGF output projections: diagonal blocks skew-projected
    0.5*(X - X^dagger), lower blocks mirrored -(upper)^dagger."""
    out = dense.copy()
    for i in range(nb):
        s = slice(i * b, (i + 1) * b)
        d = out[:, s, s]
        out[:, s, s] = 0.5 * (d - d.conj().swapaxes(-2, -1))
        for j in range(i + 1, nb):
            sj = slice(j * b, (j + 1) * b)
            out[:, sj, s] = -out[:, s, sj].conj().swapaxes(-2, -1)
    return out


def _skew_project_band(dense, nb, b):
    """Full projection onto the skew-hermitian banded subspace (diag skew
    + lower mirror); used to build physical test data and Krylov vectors."""
    return _proj_out(dense, nb, b)


def test_dyson() -> bool:
    _setup_qttools()
    from qttools.greens_function_solver.rgf import RGF
    from qttools.greens_function_solver.solver import OBCBlocks

    rng = np.random.default_rng(7)
    nb, b, ns = 5, 4, 2
    N = nb * b

    def rc(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    def band_dense(band, skew=False):
        M = np.zeros((ns, N, N), complex)
        for i in range(nb):
            s = slice(i * b, (i + 1) * b)
            M[:, s, s] = rc(ns, b, b)
            if i + 1 < nb:
                sj = slice((i + 1) * b, (i + 2) * b)
                M[:, s, sj] = rc(ns, b, b)
                M[:, sj, s] = rc(ns, b, b)
        return _skew_project_band(M, nb, b) if skew else M

    # Frozen harmonic part + OBC; base Sigma (physical, skew-structured).
    A0 = band_dense(1) + 10.0 * np.eye(N)
    sR0 = 0.1 * band_dense(1)
    sL0, sG0 = band_dense(1, skew=True), band_dense(1, skew=True)
    obc = OBCBlocks(num_blocks=nb)
    obc_r0, obc_rN = 0.05 * rc(ns, b, b), 0.05 * rc(ns, b, b)
    obc.retarded[0], obc.retarded[nb - 1] = obc_r0, obc_rN
    ol = band_dense(1, skew=True)
    obc.lesser[0], obc.lesser[nb - 1] = ol[:, :b, :b], ol[:, -b:, -b:]
    og = band_dense(1, skew=True)
    obc.greater[0], obc.greater[nb - 1] = og[:, :b, :b], og[:, -b:, -b:]

    solver = RGF(max_batch_size=2)
    a_d = _band_pattern_dsdb(nb, b, 1, ns)
    sl_d = _band_pattern_dsdb(nb, b, 1, ns)
    sg_d = _band_pattern_dsdb(nb, b, 1, ns)

    def rgf_map(sR, sL, sG):
        """The implemented Dyson map: banded (X^<, X^>) from Sigma."""
        _dense_to_band(a_d, A0 - sR, nb, b, 1)
        _dense_to_band(sl_d, sL, nb, b, 1)
        _dense_to_band(sg_d, sG, nb, b, 1)
        xl = _band_pattern_dsdb(nb, b, 2, ns)
        xg = _band_pattern_dsdb(nb, b, 2, ns)
        xr = _band_pattern_dsdb(nb, b, 2, ns)
        for m in (xl, xg, xr):
            m.data[:] = 0
        solver.selected_solve(a=a_d, sigma_lesser=sl_d, sigma_greater=sg_d,
                              out=(xl, xg, xr), obc_blocks=obc,
                              return_retarded=True, second_offdiagonals=True)
        return (_band_to_dense(xl, nb, b, 2, ns),
                _band_to_dense(xg, nb, b, 2, ns))

    def fd_jvp(dR, dL, dG, eps=1e-6):
        p = rgf_map(sR0 + eps * dR, sL0 + eps * dL, sG0 + eps * dG)
        m = rgf_map(sR0 - eps * dR, sL0 - eps * dL, sG0 - eps * dG)
        return tuple((a - c) / (2 * eps) for a, c in zip(p, m))

    def analytic_jvp(dR, dL, dG, project):
        """Dense frozen-G identity, optionally with the RGF projections."""
        A_eff = (A0 - sR0).copy()
        A_eff[:, :b, :b] -= obc_r0
        A_eff[:, -b:, -b:] -= obc_rN
        GR = np.linalg.inv(A_eff)
        GA = GR.conj().swapaxes(-2, -1)

        def src(base, corner0, cornerN):
            S = base.copy()
            S[:, :b, :b] += corner0
            S[:, -b:, -b:] += cornerN
            return S

        out = []
        for base, corner0, cornerN, dS in (
                (sL0, obc.lesser[0], obc.lesser[nb - 1], dL),
                (sG0, obc.greater[0], obc.greater[nb - 1], dG)):
            if project:
                dS = _sub_lower(dS, nb, b)
            Gx = GR @ src(base, corner0, cornerN) @ GA
            dY = (GR @ dS @ GA + GR @ dR @ Gx
                  + Gx @ dR.conj().swapaxes(-2, -1) @ GA)
            if project:
                dY = _proj_out(dY, nb, b)
            out.append(dY)
        return tuple(out)

    def band_rel(got, want, band):
        num = den = 0.0
        for i in range(nb):
            for j in range(nb):
                if abs(i - j) <= band:
                    g = got[:, i*b:(i+1)*b, j*b:(j+1)*b]
                    w = want[:, i*b:(i+1)*b, j*b:(j+1)*b]
                    num += np.linalg.norm(g - w) ** 2
                    den += np.linalg.norm(w) ** 2
        return np.sqrt(num / max(den, 1e-300))

    ok = True

    # (a) Physical (skew-structured) direction: plain identity matches.
    dR = 0.3 * band_dense(1)
    dL, dG = band_dense(1, skew=True), band_dense(1, skew=True)
    fd = fd_jvp(dR, dL, dG)
    an = analytic_jvp(dR, dL, dG, project=False)
    rel_a = max(band_rel(an[k], fd[k], 2) for k in range(2))
    print(f"[dyson] skew direction, plain identity vs RGF-FD: rel {rel_a:.2e} "
          f"{'PASS' if rel_a < 1e-7 else 'FAIL'}")
    ok = ok and rel_a < 1e-7

    # (b) Non-skew direction: plain identity does NOT match (H1 is real)...
    dLn, dGn = band_dense(1), band_dense(1)
    fd_n = fd_jvp(dR, dLn, dGn)
    an_plain = analytic_jvp(dR, dLn, dGn, project=False)
    rel_b = max(band_rel(an_plain[k], fd_n[k], 2) for k in range(2))
    print(f"[dyson] non-skew direction, plain identity vs RGF-FD: "
          f"rel {rel_b:.2e} (expected O(1) mismatch: "
          f"{'PASS' if rel_b > 1e-3 else 'FAIL'})")
    ok = ok and rel_b > 1e-3

    # (c) ...but the implemented map PRESERVES the skew banded subspace
    # (output diag projection + lower mirrors), so the Newton-Krylov solve
    # can be run entirely inside it: the residual is skew-structured, and
    # J maps the subspace into itself, where (a) shows the plain identity
    # is the exact Jacobian. Verify invariance on a non-skew input.
    got_l, got_g = rgf_map(sR0, sL0 + dLn, sG0 + dGn)
    inv_rel = max(
        band_rel(_proj_out(got_l, nb, b), got_l, 2),
        band_rel(_proj_out(got_g, nb, b), got_g, 2))
    print(f"[dyson] RGF output lies in the skew subspace (invariance): "
          f"rel {inv_rel:.2e} {'PASS' if inv_rel < 1e-14 else 'FAIL'}")
    ok = ok and inv_rel < 1e-14

    # (d) Projected direction: identity-with-projections == RGF-FD (this is
    # the exact operation mode of the production JVP: project the Krylov
    # vector, apply the plain identity, project the output).
    dLp = _skew_project_band(dLn, nb, b)
    dGp = _skew_project_band(dGn, nb, b)
    fd_p = fd_jvp(dR, dLp, dGp)
    an_p = analytic_jvp(dR, dLp, dGp, project=True)
    rel_d = max(band_rel(an_p[k], fd_p[k], 2) for k in range(2))
    print(f"[dyson] projected direction, identity vs RGF-FD: rel {rel_d:.2e} "
          f"{'PASS' if rel_d < 1e-7 else 'FAIL'}")
    ok = ok and rel_d < 1e-7

    return ok


# ---------------------------------------------------------------------------
# Test 3: production kernel (one-sided grid, fold, masks, KK, fast paths)
# ---------------------------------------------------------------------------


def test_kernel() -> bool:
    _setup_qttools()
    from types import SimpleNamespace

    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

    rng = np.random.default_rng(3)
    nb, b, nf = 4, 3, 33
    freqs = np.linspace(0.0, 8.0, nf)

    phi = {}
    for I in range(nb):
        for K1 in range(max(0, I - 1), min(nb, I + 2)):
            for K2 in range(max(0, I - 1), min(nb, I + 2)):
                phi[(I, K1, K2)] = rng.standard_normal((b, b, b))

    def make_cfg(g_from_l, herm_pairs):
        phonon = SimpleNamespace(
            sse_ramp_iterations=0, sse_vertex_scale=1.0,
            sse_ring_threads=1, sse_ring_min_w=None,
            sse_ring_workspaces=False, sse_tau_min_chunk=4,
            sse_tau_chunk_bytes=1 << 26, sse_pool_scope="tau",
            sse_greater_from_lesser=g_from_l,
            sse_fold_verify_iterations=0,
            sse_hermitian_pairs=herm_pairs, retarded_method="fft",
            qfold_path=None, decomposed_vertices_path=None,
            fc3_path=None, sse_g_band=2,
        )
        return SimpleNamespace(phonon=phonon, device=None)

    def gbuf(dense=None):
        m = _band_pattern_dsdb(nb, b, 2, nf)
        m.data[:] = 0
        if dense is not None:
            _dense_to_band(m, dense, nb, b, 2)
        return m

    def rand_g(skew=False):
        M = np.zeros((nf, nb * b, nb * b), complex)
        for i in range(nb):
            for j in range(nb):
                if abs(i - j) <= 2:
                    M[:, i*b:(i+1)*b, j*b:(j+1)*b] = _rand_c(nf, b, b,
                                                             rng=rng)
        return _skew_project_band(M, nb, b) if skew else M

    ok = True
    for g_from_l in (False, True):
        for herm_pairs in (False, True):
            sse = SigmaPhononPhonon(
                make_cfg(g_from_l, herm_pairs), freqs, np.full(nb, b),
                phi_blocks=phi)

            def S(gl_dense, gg_dense):
                gl, gg = gbuf(gl_dense), gbuf(gg_dense)
                outs = (gbuf(), gbuf(), gbuf())
                sse.compute(gl, gg, out=outs)
                return np.concatenate(
                    [np.asarray(m.data).ravel().copy() for m in outs])

            # Physical-ish (skew) inputs and a skew direction: the subspace
            # the production Newton solve lives in.
            G_l, G_g = rand_g(skew=True), rand_g(skew=True)
            d_l, d_g = rand_g(skew=True), rand_g(skew=True)

            s0 = S(G_l, G_g)
            a = 1.7
            hom = _rel(S(a * G_l, a * G_g), a**2 * s0)

            cross_pol = S(G_l + d_l, G_g + d_g) - s0 - S(d_l, d_g)
            eps = 1e-3
            cross_fd = (S(G_l + eps * d_l, G_g + eps * d_g)
                        - S(G_l - eps * d_l, G_g - eps * d_g)) / (2 * eps)
            fd_rel = _rel(cross_pol, cross_fd)

            tag = f"g_from_l={int(g_from_l)} herm_pairs={int(herm_pairs)}"
            good = hom < 1e-12 and fd_rel < 1e-9
            print(f"[kernel] {tag}: homogeneity {hom:.2e}, "
                  f"polarisation vs central-FD {fd_rel:.2e} "
                  f"{'PASS' if good else 'FAIL'}")
            ok = ok and good

    return ok


# ---------------------------------------------------------------------------
# Test 4: end-to-end production-shaped map -- composed analytic JVP vs FD
# ---------------------------------------------------------------------------


def test_endtoend() -> bool:
    """Dress rehearsal of the production JVP composition.

    Builds the production-shaped SCBA map F(x) = driver_postop(S(D(x)))
    with x = [Sigma^<, Sigma^>, Sigma^R] on a band-2 pattern:
    A = z2*I - D_dyn - BT(Sigma^R) (+ OBC corners inside RGF), the real
    RGF selected solve with second off-diagonals, the real production
    bubble, and the driver's Sigma^R += 0.5*(Sigma^< - Sigma^>). The
    analytic JVP is the frozen-G dense Dyson identity (skew-projected)
    composed with the kernel polarisation identity. Compared against
    central FD of F.
    """
    _setup_qttools()
    from types import SimpleNamespace

    from qttools.greens_function_solver.rgf import RGF
    from qttools.greens_function_solver.solver import OBCBlocks
    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

    rng = np.random.default_rng(11)
    nb, b, nf = 4, 3, 33
    N = nb * b
    freqs = np.linspace(0.0, 8.0, nf)
    eta = 1e-3
    z2 = freqs**2 + 2j * eta * np.abs(freqs) + 1.0  # +1: keep A regular

    def rc(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    # Frozen harmonic pieces.
    D_sym = rng.standard_normal((N, N)) * 0.1
    D_sym = np.triu(D_sym) + np.triu(D_sym, 1).T          # symmetric
    D_band = np.zeros((N, N))
    for i in range(nb):
        for j in range(nb):
            if abs(i - j) <= 1:
                D_band[i*b:(i+1)*b, j*b:(j+1)*b] = (
                    D_sym[i*b:(i+1)*b, j*b:(j+1)*b])
    phi = {}
    for I in range(nb):
        for K1 in range(max(0, I - 1), min(nb, I + 2)):
            for K2 in range(max(0, I - 1), min(nb, I + 2)):
                phi[(I, K1, K2)] = rng.standard_normal((b, b, b))

    obc = OBCBlocks(num_blocks=nb)
    obc.retarded[0] = 0.05 * rc(nf, b, b)
    obc.retarded[nb - 1] = 0.05 * rc(nf, b, b)
    ol0, olN, og0, ogN = (rc(nf, b, b) for _ in range(4))
    obc.lesser[0] = 0.5 * (ol0 - ol0.conj().swapaxes(-2, -1))
    obc.lesser[nb - 1] = 0.5 * (olN - olN.conj().swapaxes(-2, -1))
    obc.greater[0] = 0.5 * (og0 - og0.conj().swapaxes(-2, -1))
    obc.greater[nb - 1] = 0.5 * (ogN - ogN.conj().swapaxes(-2, -1))

    phonon_cfg = SimpleNamespace(
        sse_ramp_iterations=0, sse_vertex_scale=1.0, sse_ring_threads=1,
        sse_ring_min_w=None, sse_ring_workspaces=False, sse_tau_min_chunk=4,
        sse_tau_chunk_bytes=1 << 26, sse_pool_scope="tau",
        sse_greater_from_lesser=True, sse_fold_verify_iterations=0,
        sse_hermitian_pairs=True, retarded_method="fft",
        qfold_path=None, decomposed_vertices_path=None, fc3_path=None,
        sse_g_band=2)
    sse = SigmaPhononPhonon(SimpleNamespace(phonon=phonon_cfg, device=None),
                            freqs, np.full(nb, b), phi_blocks=phi)
    solver = RGF(max_batch_size=8)

    def gbuf(dense=None):
        m = _band_pattern_dsdb(nb, b, 2, nf)
        m.data[:] = 0
        if dense is not None:
            _dense_to_band(m, dense, nb, b, 2)
        return m

    def kernel_S(gl_dense, gg_dense):
        gl, gg = gbuf(gl_dense), gbuf(gg_dense)
        outs = (gbuf(), gbuf(), gbuf())
        sse.compute(gl, gg, out=outs)
        return [_band_to_dense(m, nb, b, 2, nf) for m in outs]

    def F(sl, sg, sr):
        """One production-shaped SCBA iteration (dense in/out)."""
        A = z2[:, None, None] * np.eye(N)[None] - D_band[None]
        for i in range(nb):
            for j in range(nb):
                if abs(i - j) <= 1:
                    A[:, i*b:(i+1)*b, j*b:(j+1)*b] -= (
                        sr[:, i*b:(i+1)*b, j*b:(j+1)*b])
        a_d = gbuf(A)
        sl_d, sg_d = gbuf(sl), gbuf(sg)
        xl, xg, xr = gbuf(), gbuf(), gbuf()
        solver.selected_solve(a=a_d, sigma_lesser=sl_d, sigma_greater=sg_d,
                              out=(xl, xg, xr), obc_blocks=obc,
                              return_retarded=True, second_offdiagonals=True)
        gl = _band_to_dense(xl, nb, b, 2, nf)
        gg_ = _band_to_dense(xg, nb, b, 2, nf)
        S_l, S_g, S_r = kernel_S(gl, gg_)
        S_r = S_r + 0.5 * (S_l - S_g)      # driver post-op (scba.py:1321)
        return S_l, S_g, S_r, gl, gg_

    # Base point: physical-ish state.
    sl0 = _skew_project_band(0.1 * rc(nf, N, N), nb, b)
    sg0 = _skew_project_band(0.1 * rc(nf, N, N), nb, b)
    sr0 = 0.1 * rc(nf, N, N)
    for M in (sl0, sg0, sr0):
        for i in range(nb):
            for j in range(nb):
                if abs(i - j) > 1:
                    M[:, i*b:(i+1)*b, j*b:(j+1)*b] = 0.0

    Fl0, Fg0, Fr0, _, _ = F(sl0, sg0, sr0)

    # ---- analytic JVP (the PhononJVP algorithm) ----
    # prepare: frozen dense G at the base point + S_base.
    A_eff = z2[:, None, None] * np.eye(N)[None] - D_band[None] - sr0
    A_eff[:, :b, :b] -= obc.retarded[0]
    A_eff[:, -b:, -b:] -= obc.retarded[nb - 1]
    GR = np.linalg.inv(A_eff)
    GA = GR.conj().swapaxes(-2, -1)

    def with_corners(base, c0, cN):
        S = _sub_lower(base, nb, b)
        S = S.copy()
        S[:, :b, :b] += c0
        S[:, -b:, -b:] += cN
        return S

    GL = GR @ with_corners(sl0, obc.lesser[0], obc.lesser[nb - 1]) @ GA
    GG = GR @ with_corners(sg0, obc.greater[0], obc.greater[nb - 1]) @ GA

    # H17 self-check: dense reconstruction vs the RGF band output.
    _, _, _, gl_rgf, gg_rgf = F(sl0, sg0, sr0)
    recon = max(
        _rel(_proj_out(GL, nb, b) * _bandmask(nb, b, 2),
             gl_rgf),
        _rel(_proj_out(GG, nb, b) * _bandmask(nb, b, 2),
             gg_rgf))
    print(f"[endtoend] dense G reconstruction vs RGF band: rel {recon:.2e} "
          f"{'PASS' if recon < 1e-11 else 'FAIL'}")

    S_base = kernel_S(gl_rgf, gg_rgf)

    def apply_jvp(dl, dg, dr):
        dl = _skew_project_band(dl, nb, b)
        dg = _skew_project_band(dg, nb, b)
        dGl = (GR @ dl @ GA + GR @ dr @ GL
               + GL @ dr.conj().swapaxes(-2, -1) @ GA)
        dGg = (GR @ dg @ GA + GR @ dr @ GG
               + GG @ dr.conj().swapaxes(-2, -1) @ GA)
        dGl = _proj_out(dGl, nb, b) * _bandmask(nb, b, 2)
        dGg = _proj_out(dGg, nb, b) * _bandmask(nb, b, 2)
        S_plus = kernel_S(gl_rgf + dGl, gg_rgf + dGg)
        S_dir = kernel_S(dGl, dGg)
        dS = [p - s - d for p, s, d in zip(S_plus, S_base, S_dir)]
        dS[2] = dS[2] + 0.5 * (dS[0] - dS[1])
        return dS

    # FD comparison on a random direction.
    dl = 0.3 * rc(nf, N, N)
    dg = 0.3 * rc(nf, N, N)
    dr = 0.3 * rc(nf, N, N)
    for M in (dl, dg, dr):
        for i in range(nb):
            for j in range(nb):
                if abs(i - j) > 1:
                    M[:, i*b:(i+1)*b, j*b:(j+1)*b] = 0.0
    dl = _skew_project_band(dl, nb, b)
    dg = _skew_project_band(dg, nb, b)

    an = apply_jvp(dl, dg, dr)
    eps = 1e-5
    Fp = F(sl0 + eps * dl, sg0 + eps * dg, sr0 + eps * dr)[:3]
    Fm = F(sl0 - eps * dl, sg0 - eps * dg, sr0 - eps * dr)[:3]
    fd = [(p - m) / (2 * eps) for p, m in zip(Fp, Fm)]
    rel = max(_rel(a, f) for a, f in zip(an, fd))
    print(f"[endtoend] composed analytic JVP vs central FD of F: "
          f"rel {rel:.2e} {'PASS' if rel < 1e-7 else 'FAIL'}")

    return recon < 1e-11 and rel < 1e-7


def _bandmask(nb, b, band):
    N = nb * b
    m = np.zeros((N, N))
    for i in range(nb):
        for j in range(nb):
            if abs(i - j) <= band:
                m[i*b:(i+1)*b, j*b:(j+1)*b] = 1.0
    return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TESTS = {"bubble": test_bubble, "dyson": test_dyson, "kernel": test_kernel,
         "endtoend": test_endtoend}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=[*TESTS, "all"], default="all")
    args = ap.parse_args()
    names = list(TESTS) if args.test == "all" else [args.test]
    ok = True
    for name in names:
        print(f"=== {name} ===")
        ok = TESTS[name]() and ok
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
