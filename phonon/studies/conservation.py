"""Conservation gates: bubble energy-balance replica + vertex S3 symmetry.

Two machine-precision gates that together prove the discrete 3-phonon bubble
is exactly energy-conserving:

* :func:`replica_check` -- a standalone replica of the production discrete
  bubble pipeline (zero-based grid, ``n_fft = 2*ne-1`` linear convolution,
  index-reversal fold with the bosonic TRANSPOSE continuation
  ``G^<(-w) = G^>(w)^T``, S3-symmetric vertex, transpose-paired trace).
  Evaluates ``P_in = sum_w w Tr[Sigma^<(w) G^>(w)]`` against
  ``P_out = sum_w w Tr[Sigma^>(w) G^<(w)]`` in float64 AND float128: the
  residual scales down with precision iff the bubble is exactly conserving
  and the production ~1e-6 floor is pure floating-point accumulation.

* :func:`s3_violation` -- reads a device FC3 blocks HDF5 (``fc3_blocks``)
  and measures the worst/mean violation of full S3 permutation symmetry
  using the correct GROUP transport: 2-cycle key-permutations pair with the
  same numpy transpose, while the 3-cycle key-perm (1,2,0) pairs with the
  INVERSE transpose (2,0,1) and vice versa. Missing orbit members count as
  zero. The bubble conserves energy at machine eps iff this is ~eps.

CLI::

    python -m phonon.studies conservation run [--fc3 PATH ...]
"""

from __future__ import annotations

import argparse

import numpy as np

#: replica dimensions (small random dense system)
N = 6        # orbitals per cell
NE = 41      # frequency bins, zero-based grid
DW = 0.5

#: PASS thresholds
REPLICA_TOL_F64 = 1e-12
S3_TOL = 1e-12

#: S3 group transport: (key permutation, numpy transpose). The 2-cycles are
#: self-paired; the 3-cycles pair with their inverse transpose.
GROUP = [((0, 1, 2), (0, 1, 2)), ((1, 0, 2), (1, 0, 2)),
         ((0, 2, 1), (0, 2, 1)), ((2, 1, 0), (2, 1, 0)),
         ((1, 2, 0), (2, 0, 1)), ((2, 0, 1), (1, 2, 0))]


# ---------------------------------------------------------------------------
# Discrete-pipeline replica (from balance_roundoff_test.py)
# ---------------------------------------------------------------------------


def _bubble_balance(cdtype, rng):
    fdtype = np.float64 if cdtype == np.complex128 else np.longdouble
    w = (np.arange(NE) * DW).astype(fdtype)

    def rnd(*s):
        return (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(
            np.complex128).astype(cdtype)

    # Independent random G^<, G^> on the one-sided grid; production zeroes
    # the w=0 bin of G^<,> (DC fix) -- mirror that.
    gl = rnd(NE, N, N)
    gg = rnd(NE, N, N)
    gl[0] = 0
    gg[0] = 0

    # S3-symmetric 3-body vertex Phi_{i,k,l} (single cell, all legs alike).
    phi = rnd(N, N, N).real.astype(fdtype)   # real, like FC3
    phi = (phi + phi.transpose(0, 2, 1) + phi.transpose(1, 0, 2)
           + phi.transpose(1, 2, 0) + phi.transpose(2, 0, 1)
           + phi.transpose(2, 1, 0)) / 6.0
    phi = phi.astype(cdtype)

    nfft = 2 * NE - 1

    def pad(x):
        out = np.zeros((nfft, N, N), dtype=cdtype)
        out[:NE] = x
        return out

    def rev(x):
        # index-reversal fold of the TRANSPOSED array: G(-w) = G(w)^T.
        # Production layout (sse_phonon_phonon): data[0] = X[0],
        # data[1:] = X_padded[:0:-1] -> negative frequencies live at the
        # TOP of the n_fft buffer (data[nfft-m] = G(m)^T).
        xt = x.transpose(0, 2, 1)
        out = np.zeros((nfft, N, N), dtype=cdtype)
        out[0] = xt[0]
        for m in range(1, NE):
            out[nfft - m] = xt[m]
        return out

    def conv(a, b):
        # exact linear convolution over the padded axis, truncated to [0, NE)
        out = np.zeros((NE, N, N, N, N), dtype=cdtype)
        for n in range(NE):
            for m in range(nfft):
                k = n - m
                if k < 0:
                    k += nfft          # circular index = linear conv w/ pad
                out[n] += np.einsum("ab,cd->abcd", a[m], b[k])
        return out

    def contract(pair):
        # Sigma_{ij}(w) = sum Phi_{ikl} G_{kk'} G_{ll'} Phi_{jk'l'}
        return np.einsum("ikl,wkKlL,jKL->wij", phi, pair, phi)

    gl_p, gg_p = pad(gl), pad(gg)
    gl_r, gg_r = rev(gl), rev(gg)

    sig_l = contract(conv(gl_p, gl_p) + conv(gl_p, gg_r) + conv(gg_r, gl_p))
    sig_g = contract(conv(gg_p, gg_p) + conv(gg_p, gl_r) + conv(gl_r, gg_p))

    # transpose-paired weighted traces
    p_in = np.sum(w[:, None, None] * sig_l * gg.transpose(0, 2, 1))
    p_out = np.sum(w[:, None, None] * sig_g * gl.transpose(0, 2, 1))
    resid = abs(p_in - p_out) / max(abs(p_in) + abs(p_out), 1e-300)
    return p_in, p_out, resid


def replica_check() -> dict:
    """Run the discrete-pipeline replica in float64 and float128.

    Returns ``{dtype_name: {"p_in", "p_out", "resid"}}`` -- the residual must
    be < ~1e-12 in float64 and scale down further in float128 for the bubble
    to be exactly conserving (roundoff-only floor).
    """
    results = {}
    for cd, name in ((np.complex128, "complex128"),
                     (np.clongdouble, "clongdouble")):
        rng = np.random.default_rng(7)   # same draw for both precisions
        p_in, p_out, resid = _bubble_balance(cd, rng)
        results[name] = {"p_in": complex(p_in), "p_out": complex(p_out),
                         "resid": float(resid)}
    return results


# ---------------------------------------------------------------------------
# Device-vertex S3 gate (from plain_truncation_vertex.py)
# ---------------------------------------------------------------------------


def _s3_violation_blocks(pb: dict) -> tuple[float, float]:
    """Worst/mean S3 violation of a ``{(i,j,k): array}`` block dict.

    Correct group transport (GROUP above); missing orbit members count as
    zero blocks.
    """
    worst, tn, td = 0.0, 0.0, 0.0
    for key, v in pb.items():
        for kp, tr in GROUP[1:]:
            ik = tuple(key[i] for i in kp)
            img = pb.get(ik)
            x = img.transpose(tr) if img is not None else np.zeros_like(v)
            d = np.abs(v - x).max()
            worst = max(worst, d / (np.abs(v).max() + 1e-300))
            tn += d
            td += np.abs(v).max() + 1e-300
    return worst, tn / td


def s3_violation(fc3_blocks_path) -> dict:
    """Worst/mean S3 violation of a device FC3 blocks HDF5 file.

    ``fc3_blocks_path`` is an ``fc3_blocks.hdf5`` written by
    ``phonon_inputs.quatrex_writer.write_fc3_blocks`` (group ``fc3_blocks``
    with ``i_j_k`` slab-key datasets).
    """
    import h5py

    with h5py.File(fc3_blocks_path, "r") as f:
        g = f["fc3_blocks"]
        pb = {tuple(int(x) for x in k.split("_")): np.asarray(g[k])
              for k in g.keys()}
    worst, mean = _s3_violation_blocks(pb)
    return {"worst": float(worst), "mean": float(mean),
            "n_blocks": len(pb)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def sigma_convention(config_path) -> dict:
    """Sign-convention/positivity gate of the production SSE feedback.

    Solves the ballistic G of ``config_path``, evaluates the 3-phonon SSE
    once, and checks that Sigma^≷ follow the SOLVER convention
    (-i Sigma^≷ >= 0 per omega > 0, like the lead injection
    sigma^≷ = +i n(+1) gamma and like -iG^≷ >= 0). A violation here is the
    anti-dissipative feedback that made the SCBA diverge at any coupling
    (2026-06-12); the bubble energy balance is sign-blind and cannot see it.

    Returns {"g_worst", "sl_worst", "sg_worst"}: worst relative negative
    eigenvalue over all omega > 0 (0.0 = clean).
    """
    from quatrex.core.config import parse_config, setup_context

    cfg = parse_config(config_path)
    setup_context(cfg)
    from quatrex.core.scba import SCBA

    scba = SCBA(cfg)
    ph = scba.subsystems["phonon"]
    d = scba.data
    for solver in scba.subsystems.values():
        solver.solve(d.sigma_lesser, d.sigma_greater,
                     d.sigma_retarded_hermitian,
                     out=(d.g_lesser, d.g_greater, d.g_retarded))
    for m in (d.g_lesser, d.g_greater):
        m.dtranspose(discard=False)
    for m in (d.sigma_lesser, d.sigma_greater, d.sigma_retarded_hermitian):
        m.dtranspose(discard=True)
    scba._compute_interactions()
    for m in (d.sigma_lesser, d.sigma_greater, d.sigma_retarded_hermitian):
        m.dtranspose(discard=False)
    for m in (d.g_lesser, d.g_greater):
        m.dtranspose(discard=False)

    w = np.abs(np.asarray(ph.local_frequencies, float))
    rows = np.asarray(d.g_lesser.rows)
    cols = np.asarray(d.g_lesser.cols)
    n = int(max(rows.max(), cols.max())) + 1

    def worst_negative(matrix):
        worst = 0.0
        for i in range(len(w)):
            if w[i] <= 0.0:
                continue
            dense = np.zeros((n, n), complex)
            dense[rows, cols] = np.asarray(matrix.data)[i]
            m = -1j * dense
            m = 0.5 * (m + m.conj().T)
            ev = np.linalg.eigvalsh(m)
            worst = max(worst, -float(ev.min()) / (float(np.abs(ev).max()) + 1e-300))
        return worst

    return {"g_worst": max(worst_negative(d.g_lesser), worst_negative(d.g_greater)),
            "sl_worst": worst_negative(d.sigma_lesser),
            "sg_worst": worst_negative(d.sigma_greater)}


def run(argv) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phonon.studies conservation run",
        description=("Energy-conservation gates: discrete bubble replica "
                     "(always) + S3 symmetry of device FC3 blocks "
                     "(per --fc3 path) + SSE sign-convention/positivity "
                     "(per --convention config)."),
    )
    parser.add_argument(
        "--fc3", nargs="+", default=[], metavar="PATH",
        help="fc3_blocks.hdf5 file(s) to gate on S3 symmetry "
             "(default: none -- only the replica check runs)")
    parser.add_argument(
        "--convention", default=None, metavar="CONFIG_TOML",
        help="quatrex config of a small cell: run the single-shot SSE "
             "sign-convention/positivity gate (-i Sigma^≷ >= 0)")
    args = parser.parse_args(argv)

    failures = 0

    print(f"[replica] discrete bubble pipeline, N={N} NE={NE} DW={DW}")
    rep = replica_check()
    for name, r in rep.items():
        print(f"  {name:12s}: P_in={r['p_in']:.12e} "
              f"P_out={r['p_out']:.12e} resid={r['resid']:.3e}")
    ok = rep["complex128"]["resid"] < REPLICA_TOL_F64
    failures += not ok
    print(f"  replica {'PASS' if ok else 'FAIL'} "
          f"(float64 resid {rep['complex128']['resid']:.3e} "
          f"{'<' if ok else '>='} {REPLICA_TOL_F64:g})")

    if not args.fc3:
        print("[s3] no --fc3 paths given; pass fc3_blocks.hdf5 file(s) to "
              "gate device vertices.")
    for path in args.fc3:
        res = s3_violation(path)
        ok = res["worst"] < S3_TOL
        failures += not ok
        print(f"[s3] {path}: worst={res['worst']:.3e} mean={res['mean']:.3e} "
              f"({res['n_blocks']} blocks) -> {'PASS' if ok else 'FAIL'} "
              f"(tol {S3_TOL:g})")

    if args.convention:
        res = sigma_convention(args.convention)
        ok = max(res.values()) < 1e-10
        failures += not ok
        print(f"[convention] {args.convention}: worst rel-negative eigenvalue "
              f"G^≷ {res['g_worst']:.3e}, Sigma^< {res['sl_worst']:.3e}, "
              f"Sigma^> {res['sg_worst']:.3e} -> {'PASS' if ok else 'FAIL'}")

    print(f"[conservation] {'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 0 if failures == 0 else 1


def plot(argv) -> int:
    print("conservation: nothing to plot (gates are textual PASS/FAIL).")
    return 0
