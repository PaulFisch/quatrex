"""Subcell positivity of the hybrid Green function (review Sec. 11-12).
"""

from __future__ import annotations

import numpy as np

from quatrex.phonon.experimental.pole import pole_local
from quatrex.phonon.experimental.pole.pole_keldysh import (
    PoleCluster,
    pole_keldysh,
    pole_retarded,
)


# ---------------------------------------------------------------- utilities

def gauss_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights on [-1/2, 1/2] (one cell of width 1)."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * x, 0.5 * w


def eigs(mats: np.ndarray, sign: float = -1.0) -> np.ndarray:
    """Eigenvalues of the Hermitian part of ``sign * i * M``."""
    herm = sign * 1j * np.asarray(mats)
    herm = 0.5 * (herm + np.conj(herm).swapaxes(-2, -1))
    return np.linalg.eigvalsh(herm)


def psd_metric(mats: np.ndarray, sign: float = -1.0,
               scale: float | None = None) -> dict:
    """Worst normalised eigenvalue of ``sign * i * M`` over a stack.

    Dense analogue of ``pole_audit.psd_residual`` for the small beds here.
    ``scale`` MUST be supplied when several stacks are to be compared: the
    normalisation has to be one global number, never per-stack. A per-stack
    scale makes the numerically empty tails look like failures and makes
    separate rows incommensurable -- the trap recorded in the docstring of
    ``psd_residual``.
    """
    ev = eigs(mats, sign)
    if scale is None:
        scale = float(np.max(ev)) if ev.size else 1.0
    scale = scale if scale > 0 else 1.0
    return {"worst": float(np.min(ev)) / scale, "scale": scale}


# ------------------------------------------------------------ the hybrid bed

def make_bed(
    gamma: float = 0.01,
    h: float = 0.25,
    n_dof: int = 3,
    w_pole: float = 4.0,
    seed: int = 0,
    offset: float = 0.0,
) -> dict:
    """A frozen hybrid iterate with one narrow promoted pole.

    ``G^R`` carries the narrow pole plus two WIDE background poles, so the
    remainder ``R = G - P`` is smooth, as the method assumes. ``G^<`` is built
    by congruence ``G^R S G^A`` with ``S`` PSD, so it is PSD at every
    frequency -- the bed cannot fail the test for a trivial reason.

    ``offset`` shifts the pole off the cell centre in units of ``h``; the
    damage depends on where inside its cell the pole sits.
    """
    rng = np.random.default_rng(seed)
    grid = np.arange(0.0, 12.0, h)

    def _vec():
        return rng.normal(size=(n_dof,)) + 1j * rng.normal(size=(n_dof,))

    # Narrow (promoted) pole, plus wide background.
    z_narrow = np.array([w_pole + offset * h - 1j * gamma])
    u_n = _vec().reshape(n_dof, 1)
    cluster = PoleCluster(z=z_narrow, u=u_n, v=u_n, label="narrow")

    z_bg = np.array([2.5 - 0.9j, 7.5 - 1.2j])
    u_bg = np.stack([_vec(), _vec()], axis=1)

    a0 = rng.normal(size=(n_dof, n_dof)) + 1j * rng.normal(size=(n_dof, n_dof))
    a1 = 0.30 * (rng.normal(size=(n_dof, n_dof))
                 + 1j * rng.normal(size=(n_dof, n_dof)))

    def sigma_lesser(w):
        """Keldysh source, SMOOTH in omega and with -i*Sigma^< >= 0 always.

        The PSD object is -i*Sigma^<, not Sigma^< itself, hence the explicit
        i. The source is deliberately omega-DEPENDENT: freezing it per cell
        must be an approximation, or the congruence route below would be
        exact for free and the comparison would prove nothing.
        """
        w = np.atleast_1d(np.asarray(w, float))
        amp = a0[None] + np.tanh(w / 6.0)[:, None, None] * a1[None]
        return 1j * (amp @ np.conj(amp).swapaxes(-2, -1))

    def g_retarded(w):
        w = np.atleast_1d(np.asarray(w, float))
        gr = pole_retarded(w, cluster)
        d = 1.0 / (w.reshape(-1, 1) - z_bg.reshape(1, -1))
        return gr + np.einsum("ia,wa,ja->wij", u_bg, d, np.conj(u_bg))

    def pole_retarded_only(w):
        return pole_retarded(np.atleast_1d(np.asarray(w, float)), cluster)

    def g_lesser(w):
        gr = g_retarded(w)
        ga = np.conj(gr).swapaxes(-2, -1)
        return gr @ sigma_lesser(w) @ ga      # PSD by congruence, exact

    # Pole sector's own source: frozen AT THE POLE, which is what
    # pole_bridge.source_at_poles does in production.
    w_p = float(np.real(cluster.z[0]))
    s_proj = np.conj(cluster.v).T @ sigma_lesser(w_p)[0] @ cluster.v

    def p_lesser(w):
        return pole_keldysh(np.atleast_1d(np.asarray(w, float)), cluster,
                            s_proj)

    return dict(grid=grid, h=h, cluster=cluster, gamma=gamma,
                g_lesser=g_lesser, p_lesser=p_lesser, s_proj=s_proj,
                g_retarded=g_retarded, pole_retarded=pole_retarded_only,
                sigma_lesser=sigma_lesser, n_dof=n_dof)


def subcell_scan(bed: dict, n_sub: int = 24, over: float = 1.0,
                 cells: int = 3) -> dict:
    """Centre-PSD vs subcell-PSD of ``G_h = P + R_k`` near the pole.

    ``over`` scales ``P`` in the SUBTRACTION only, so ``over > 1`` is the
    deliberately over-subtracted negative control: it must drive the metric
    clearly negative, per the rule that a bed has to show the wrong relation
    failing by a large margin.
    """
    grid, h = bed["grid"], bed["h"]
    g_lesser, p_lesser = bed["g_lesser"], bed["p_lesser"]

    k0 = int(np.argmin(np.abs(grid - float(np.real(bed["cluster"].z[0])))))
    ks = [k for k in range(k0 - cells, k0 + cells + 1)
          if 0 <= k < grid.size]

    centres = np.array([grid[k] for k in ks])
    g_c = g_lesser(centres)
    p_c = p_lesser(centres)
    r_cells = g_c - over * p_c               # the stored, frozen remainder

    x, _ = gauss_nodes(n_sub)
    sub_all = [over * p_lesser(grid[k] + h * x) + r_cells[i]
               for i, k in enumerate(ks)]

    # ONE global scale for every row, centre and subcell alike.
    scale = float(max(eigs(np.concatenate(sub_all)).max(),
                      eigs(g_c).max()))
    rows = [dict(k=k, w=float(grid[k]),
                 centre=psd_metric(g_c[i][None], scale=scale)["worst"],
                 subcell=psd_metric(sub_all[i], scale=scale)["worst"])
            for i, k in enumerate(ks)]
    return dict(rows=rows, scale=scale,
                eps_psd_subcell=psd_metric(np.concatenate(sub_all),
                                           scale=scale)["worst"],
                eps_psd_centre=psd_metric(g_c, scale=scale)["worst"])


def congruence_scan(bed: dict, n_sub: int = 24, cells: int = 3) -> dict:
    """Direct Keldysh subtraction vs congruence reconstruction (review Eq. 7).

    DIRECT (production today) freezes the Keldysh remainder:

        G_dir(w) = P^<(w) + R_k,      R_k = G^<(w_k) - P^<(w_k)

    CONGRUENCE freezes the RETARDED remainder and the source instead, then
    rebuilds the Keldysh function as a congruence:

        G_R(w) = P^R(w) + R_k^R,      R_k^R = G^R(w_k) - P^R(w_k)
        G_con(w) = G_R(w) Sigma_k^< G_R(w)^H

    Both are exact at the cell centre. The difference is structural: for
    congruence, -i G_con = G_R (-i Sigma_k^<) G_R^H is PSD for ANY G_R, so
    positivity no longer depends on the reconstruction being accurate. It is
    also the better approximation, because what gets frozen is the SOURCE,
    which is smooth, rather than the RESPONSE, which carries the pole.
    """
    grid, h = bed["grid"], bed["h"]
    g_lesser, p_lesser = bed["g_lesser"], bed["p_lesser"]
    g_ret, p_ret, sig = bed["g_retarded"], bed["pole_retarded"], bed["sigma_lesser"]

    k0 = int(np.argmin(np.abs(grid - float(np.real(bed["cluster"].z[0])))))
    ks = [k for k in range(k0 - cells, k0 + cells + 1) if 0 <= k < grid.size]
    centres = np.array([grid[k] for k in ks])

    r_les = g_lesser(centres) - p_lesser(centres)     # frozen Keldysh remainder
    r_ret = g_ret(centres) - p_ret(centres)           # frozen retarded remainder
    s_cell = sig(centres)                             # frozen source

    x, _ = gauss_nodes(n_sub)
    direct, cong, exact = [], [], []
    for i, k in enumerate(ks):
        w = grid[k] + h * x
        direct.append(p_lesser(w) + r_les[i])
        gr = p_ret(w) + r_ret[i]
        cong.append(gr @ s_cell[i] @ np.conj(gr).swapaxes(-2, -1))
        exact.append(g_lesser(w))

    scale = float(max(eigs(np.concatenate(direct)).max(),
                      eigs(np.concatenate(cong)).max(),
                      eigs(np.concatenate(exact)).max()))

    def _err(a, b):
        num = np.linalg.norm(a - b, axis=(-2, -1))
        den = np.linalg.norm(b, axis=(-2, -1))
        return float(np.max(num / np.maximum(den, 1e-300)))

    rows = []
    for i, k in enumerate(ks):
        rows.append(dict(
            k=k, w=float(grid[k]),
            psd_dir=psd_metric(direct[i], scale=scale)["worst"],
            psd_con=psd_metric(cong[i], scale=scale)["worst"],
            err_dir=_err(direct[i], exact[i]),
            err_con=_err(cong[i], exact[i])))
    return dict(rows=rows,
                psd_dir=psd_metric(np.concatenate(direct), scale=scale)["worst"],
                psd_con=psd_metric(np.concatenate(cong), scale=scale)["worst"],
                err_dir=_err(np.concatenate(direct), np.concatenate(exact)),
                err_con=_err(np.concatenate(cong), np.concatenate(exact)))



# ------------------------------------------------- E. local finite-cell bubble

def _gl_panels(f, a, b, n_panels=300, n_nodes=40):
    """Composite Gauss-Legendre: the reference the closed forms are judged on."""
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    edges = np.linspace(a, b, n_panels + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    u = (0.5 * (hi - lo) * x[None] + 0.5 * (hi + lo)).ravel()
    wt = (0.5 * (hi - lo) * w[None]).ravel()
    return np.einsum("s,s...->...", wt, f(u))


def make_ring_bed(gamma: float, n: int = 33, h: float = 0.3,
                  n_dof: int = 4, seed: int = 5) -> dict:
    """A bed whose legs are PSD at EVERY frequency, with two narrow poles.

    ``-i G^< = A A^H`` with ``A`` rational makes the leg a congruence, so
    positivity holds off grid by construction and any negative eigenvalue in
    the bubble is the quadrature's doing rather than the model's. Flattening
    the product into simple poles is exact, which is what lets the local model
    represent the leg without approximation and isolates the quadrature.
    """
    rng = np.random.default_rng(seed)
    freqs = np.arange(n) * h
    cells = (11, 21)
    z = np.array([freqs[cells[0]] + 0.13 * h - 1j * gamma * h,
                  freqs[cells[1]] - 1j * gamma * h,
                  freqs[6] - 2.5j * h])
    a = np.stack([rng.normal(size=(n_dof, n_dof))
                  + 1j * rng.normal(size=(n_dof, n_dof)) for _ in z])
    c0 = rng.normal(size=(n_dof, n_dof)) + 1j * rng.normal(size=(n_dof, n_dof))

    def amp(s):
        s = np.asarray(s, complex)
        return c0[None] + np.einsum("sp,pij->sij",
                                    1.0 / (s[:, None] - z[None]), a)

    def minus_i_g(s):
        return amp(s) @ np.conj(amp(np.conj(s))).swapaxes(-2, -1)

    gap = z[:, None] - np.conj(z)[None, :]
    outer = np.einsum("pij,qkj->pqik", a, np.conj(a))
    zeta = np.concatenate([z, np.conj(z)])
    res = np.concatenate([
        np.stack([a[p] @ np.conj(c0).T + (outer[p] / gap[p, :, None, None]).sum(0)
                  for p in range(len(z))]),
        np.stack([c0 @ np.conj(a[q]).T - (outer[:, q] / gap[:, q, None, None]).sum(0)
                  for q in range(len(z))])])
    return dict(freqs=freqs, h=h, leg=minus_i_g, cells=cells,
                zeta=zeta, residues=res, n_dof=n_dof)


def ring_scan(bed: dict, rho_min: float = 0.0, radius: int = 1,
              poly_order: int = 2) -> dict:
    """Ring vs ring + local correction, against the exact cell integrals.

    The reference integrates the SAME cell decomposition the ring sums over, so
    what is measured is the quadrature and nothing else. Comparing against an
    integral over a different support would fold in the finite-window question,
    which ``pair_convolution``'s docstring already puts at sub-percent and
    which this method does not change.
    """
    freqs, h, leg = bed["freqs"], bed["h"], bed["leg"]
    n, nd = len(freqs), bed["n_dof"]
    g = leg(freqs)
    ring = np.zeros((n, nd, nd), complex)
    exact = np.zeros_like(ring)
    for m in range(n):
        for k in range(n):
            l = m - k
            if not 0 <= l < n:
                continue
            ring[m] += h * (g[k] * g[l]) / (2 * np.pi)
            exact[m] += _gl_panels(lambda u: leg(u) * leg(freqs[m] - u),
                                   freqs[k] - h / 2, freqs[k] + h / 2) / (2 * np.pi)
    delta, report = pole_local.correct_spectrum(
        freqs, g, bed["cells"], bed["zeta"], bed["residues"],
        bilinear=lambda x, y: x * y, radius=radius,
        poly_order=poly_order, rho_min=rho_min)
    # ONE global scale for every stack, per psd_metric's own warning.
    scale = float(np.abs(np.linalg.eigvalsh(
        0.5 * (exact + np.conj(exact).swapaxes(-2, -1)))).max())

    def psd(x):
        # psd_metric measures sign*i*M, the convention for Sigma; the bubble of
        # two -iG legs is ALREADY the PSD object, so the extra i undoes it.
        return psd_metric(1j * x, sign=-1.0, scale=scale)["worst"]

    err = lambda x: float(np.abs(x - exact).max() / np.abs(exact).max())
    return dict(err_ring=err(ring), err_corr=err(ring + delta),
                psd_exact=psd(exact), psd_ring=psd(ring),
                psd_corr=psd(ring + delta), report=report)


def _report_ring(title: str, res: dict) -> None:
    r = res["report"]
    print(f"    {title:<26} {res['err_ring']:10.2e} {res['err_corr']:10.2e} "
          f"{res['err_ring'] / max(res['err_corr'], 1e-300):9.0f} "
          f"{res['psd_ring']:+11.3e} {res['psd_corr']:+11.3e} "
          f"{r['n_corrected']:6d} {r['n_refused_rho']:5d}")


def _report_congruence(title: str, res: dict) -> None:
    print(f"\n--- {title}")
    print(f"    {'cell':>5} {'omega':>8} {'PSD direct':>12} {'PSD congr':>12}"
          f" {'err direct':>12} {'err congr':>12}")
    for r in res["rows"]:
        print(f"    {r['k']:5d} {r['w']:8.3f} {r['psd_dir']:12.3e} "
              f"{r['psd_con']:12.3e} {r['err_dir']:12.3e} {r['err_con']:12.3e}")
    print(f"    GLOBAL  PSD direct = {res['psd_dir']:+.3e}   "
          f"congruence = {res['psd_con']:+.3e}")
    print(f"            err direct = {res['err_dir']:.3e}    "
          f"congruence = {res['err_con']:.3e}")


def _report(title: str, res: dict) -> None:
    print(f"\n--- {title}")
    print(f"    {'cell':>5} {'omega':>8} {'centre PSD':>12} {'subcell PSD':>13}")
    for r in res["rows"]:
        print(f"    {r['k']:5d} {r['w']:8.3f} {r['centre']:12.3e} "
              f"{r['subcell']:13.3e}")
    print(f"    GLOBAL  eps_PSD centre = {res['eps_psd_centre']:+.3e}   "
          f"subcell = {res['eps_psd_subcell']:+.3e}")


def main() -> None:
    h = 0.25
    print("Subcell positivity of G_h = P(w) + R_k  (review Sec. 11-12)")
    print(f"cell width h = {h}, 24 Gauss-Legendre nodes per cell, eta = 0")

    print("\n=== A. faithful reconstruction (over = 1.0) ===")
    for gamma in (0.10, 0.02, 0.005):
        bed = make_bed(gamma=gamma, h=h)
        res = subcell_scan(bed, over=1.0)
        _report(f"2*gamma/h = {2*gamma/h:.3f}  (gamma = {gamma})", res)

    print("\n=== B. pole off-centre within its cell (gamma = 0.005) ===")
    for off in (0.0, 0.25, 0.5):
        bed = make_bed(gamma=0.005, h=h, offset=off)
        res = subcell_scan(bed, over=1.0)
        _report(f"pole offset = {off:.2f} h", res)

    print("\n=== C. NEGATIVE CONTROL: over-subtracted G_PP (over = 1.5) ===")
    bed = make_bed(gamma=0.005, h=h)
    res = subcell_scan(bed, over=1.5)
    _report("over = 1.5 -- must be clearly negative", res)

    print("\n=== D. CONGRUENCE RECONSTRUCTION vs direct subtraction ===")
    for gamma in (0.10, 0.02, 0.005):
        bed = make_bed(gamma=gamma, h=h)
        res = congruence_scan(bed)
        _report_congruence(f"2*gamma/h = {2*gamma/h:.3f}", res)


    print("\n=== E. LOCAL FINITE-CELL BUBBLE vs the ring's rectangle rule ===")
    print("    legs PSD at every frequency by congruence; the reference")
    print("    integrates the SAME cells the ring sums over.")
    print(f"    {'':<26} {'err ring':>10} {'err corr':>10} {'gain':>9} "
          f"{'PSD ring':>11} {'PSD corr':>11} {'corr':>6} {'refus':>5}")
    for gamma in (0.4, 0.1, 0.02, 0.005, 0.001):
        bed = make_ring_bed(gamma=gamma)
        _report_ring(f"gamma/h = {gamma:.3f}", ring_scan(bed))

    print("\n    radius of the corrected set (gamma/h = 0.02):")
    print("    the rectangle rule fails across the pole's TAIL, not only in")
    print("    the cell holding it.")
    bed = make_ring_bed(gamma=0.02)
    for radius in (0, 1, 2, 3):
        _report_ring(f"radius = {radius}", ring_scan(bed, radius=radius))

    print("\n    output-resolution gate (gamma/h = 0.02): refusing the")
    print("    unresolved pairs costs the whole gain, because the refused set")
    print("    IS the combination frequencies where the ring is worst.")
    for rho_min in (0.0, 1.0):
        _report_ring(f"rho_min = {rho_min:.1f}", ring_scan(bed, rho_min=rho_min))
    rho = ring_scan(bed)["report"]["rho_out"]
    finite = rho[np.isfinite(rho)]
    print(f"    rho_out over same-half-plane pole pairs: min {finite.min():.3f}, "
          f"max {finite.max():.3f}")
    print("    (pairings across the half planes are inf: they make no output")
    print("     pole, so a zero width there is an absent feature not a sharp one)")

if __name__ == "__main__":
    main()
