"""Vertex-element check: |Phi_{lambda lambda' lambda''}|^2, quatrex vs
phono3py.

Run::
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "phonon") not in sys.path:
    sys.path.insert(0, str(REPO / "phonon"))

WORK = REPO / "phonon" / "reaps" / "si_primitive_work"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_ph3(mesh, *, r0_average: bool, work: Path = WORK):
    """Load the checked-in Si primitive FC3 and initialise the ph-ph kernel."""
    import phono3py

    ph3 = phono3py.load(
        str(work / "phono3py.yaml"),
        fc3_filename=str(work / "fc3.hdf5"),
        fc2_filename=str(work / "fc2.hdf5"),
        log_level=0,
        make_r0_average=r0_average,
    )
    ph3.mesh_numbers = list(mesh)
    ph3.init_phph_interaction()
    return ph3


def run_at_grid_point(itr, grid_point: int):
    """``Interaction.run`` at one grid point.

    ``Interaction.run`` guards its phonon solve with ``if self._phonon_all_done``
    (phono3py 3.29), which does *not* fire on a fresh object -- the C kernel then
    reads a ``phonon_done`` mask of zeros and returns an all-zero
    ``interaction_strength``. Solving explicitly first is what makes it non-zero.
    """
    itr.set_grid_point(grid_point)
    itr.run_phonon_solver()
    itr.run()
    strength = np.array(itr.interaction_strength, copy=True)
    if strength.max() <= 0.0:
        raise RuntimeError("interaction_strength is identically zero")
    return strength


# ---------------------------------------------------------------------------
# the contraction, written once and reused by every stage
# ---------------------------------------------------------------------------


def phi_bare(fc3_reciprocal, eigvecs, *, mass_weight=None):
    """Bare mode-projected vertex ``Phi_{b1 b2 b3}`` for one q-triplet.

    ``fc3_reciprocal`` is ``(nat, nat, nat, 3, 3, 3)`` in *primitive* atom
    indices; ``eigvecs`` is a 3-tuple of ``(3 nat, nband)`` column-eigenvector
    matrices, one per leg.  ``mass_weight`` is ``sqrt(m_i m_j m_k)`` divided out
    when the tensor is not already mass-weighted (phono3py's is not, ours is).

    "Bare" = no ``1/sqrt(f1 f2 f3)``.  phono3py folds that factor into
    ``ReciprocalToNormal``; it belongs to the golden-rule mode representation,
    not to the vertex, and in the NEGF bubble the same factors come out of the
    Green's functions instead.
    """
    nat = fc3_reciprocal.shape[0]
    e1, e2, e3 = eigvecs
    tensor = np.asarray(fc3_reciprocal, dtype=complex)
    if mass_weight is not None:
        tensor = tensor / mass_weight[..., None, None, None]
    flat = tensor.transpose(0, 3, 1, 4, 2, 5).reshape(3 * nat, 3 * nat, 3 * nat)
    return np.einsum("abc,ai,bj,ck->ijk", flat, e1, e2, e3, optimize=True)


def _mass_weight(masses):
    m = np.asarray(masses, dtype=float)
    return np.sqrt(m[:, None, None] * m[None, :, None] * m[None, None, :])


# ---------------------------------------------------------------------------
# S2/S3: the code's real-space vertex
# ---------------------------------------------------------------------------


def code_realspace_fc3(ph3):
    """The code's mass-weighted real-space FC3, as a phono3py-shaped tensor.

    ``build_realspace_fc3_matrices`` returns ``M_stacked`` with rows only for
    the primitive atoms at cell (0,0,0); phono3py's Fourier transform reads only
    those rows (``fc3[p2s_map[i], j, k]``), so the remaining rows stay zero.
    """
    from phonon.phonon_inputs.separable import (
        build_realspace_fc3_matrices,
        build_supercell_mapping,
    )

    prim_indices, cell_frac, _, ref_sc_atoms = build_supercell_mapping(ph3)
    nat_prim = len(ph3.primitive)
    n_super = len(ph3.supercell)
    dim_sc = 3 * n_super
    m_stacked = build_realspace_fc3_matrices(
        ph3.fc3, nat_prim, ph3.supercell.masses, ref_sc_atoms
    )
    tensor = np.zeros((n_super, n_super, n_super, 3, 3, 3))
    for i_prim in range(nat_prim):
        s_i = ref_sc_atoms[i_prim]
        for alpha in range(3):
            a = 3 * i_prim + alpha
            block = m_stacked[a * dim_sc : (a + 1) * dim_sc, :]
            tensor[s_i, :, :, alpha, :, :] = block.reshape(
                n_super, 3, n_super, 3
            ).transpose(0, 2, 1, 3)
    return tensor, prim_indices, cell_frac, ref_sc_atoms


def code_reciprocal(tensor, prim_indices, cell_frac, ref_sc_atoms, nat_prim,
                    q2, q3, *, sign2=-1.0, sign3=-1.0):
    """Fourier-transform the code's way: ``T(q2) M T(q3)^T``.

    ``build_gathering_matrix`` phases are ``exp(-2 pi i q . R_cell)`` on the two
    contracted legs, with the external leg unphased -- exactly
    ``solver/se_q.py::_qfold_device_blocks``. ``sign2``/``sign3`` expose the
    exponent sign so the caller can search the convention instead of assuming it.
    """
    n_super = len(prim_indices)
    ph2 = np.exp(sign2 * 2j * np.pi * (cell_frac @ np.asarray(q2, float)))
    ph3_ = np.exp(sign3 * 2j * np.pi * (cell_frac @ np.asarray(q3, float)))
    out = np.zeros((nat_prim, nat_prim, nat_prim, 3, 3, 3), dtype=complex)
    for i_prim in range(nat_prim):
        s_i = ref_sc_atoms[i_prim]
        block = np.asarray(tensor[s_i], dtype=complex)  # (ns, ns, 3, 3, 3)
        phased = np.einsum("jkabc,j,k->jkabc", block, ph2, ph3_, optimize=True)
        for kappa in range(nat_prim):
            mj = prim_indices == kappa
            for kappa2 in range(nat_prim):
                mk = prim_indices == kappa2
                out[i_prim, kappa, kappa2] = phased[np.ix_(mj, mk)].sum(axis=(0, 1))
    return out



def basis_gauge(scaled_positions, qs):
    """Per-atom phases converting the code's cell convention to phonopy's.

    Writing ``v(s, i) = R_s + r_kappa - r_i`` for phonopy's shortest vector and
    dropping the image average, phono3py's reciprocal FC3 factorises as

        Phi^p3p_{ijk} = e^{2 pi i q1 . r_i} e^{2 pi i q2 . r_j}
                        e^{2 pi i q3 . r_k} * Phi^cell_{ijk}(+q2, +q3),

    where ``Phi^cell`` sums lattice translations only. Returns the three
    per-atom phase vectors.
    """
    r = np.asarray(scaled_positions, dtype=float)
    return [np.exp(2j * np.pi * (r @ np.asarray(q, float))) for q in qs]


def cell_dynamical_matrix(ph3, prim_indices, cell_frac, ref_sc_atoms, q):
    """D(q) in the code's convention: lattice-translation phases only.

    Convention B is ``exp(+2 pi i q . R)`` (``phonon_inputs/convention.py``),
    which is what the device Hamiltonian is built in. The *vertex* legs carry
    the conjugate phase (``build_gathering_matrix``,
    ``se_q._qfold_device_blocks``: ``exp(-2 pi i q . R)``) because they are
    contracted legs; that is a leg-orientation convention, and whether it is
    globally consistent inside the bubble is not something this script tests.
    At q commensurate with the FC supercell the two signs coincide anyway.

    Comparing these eigenvalues with phonopy's is a direct test of whether the
    single-image cell sum represents the supercell FC2 at that ``q``.
    """
    from phonon.phonon_inputs.constants import CONVERSION_THZ2

    fc2 = np.asarray(ph3.fc2)
    masses = np.asarray(ph3.primitive.masses, dtype=float)
    nat = len(masses)
    ph = np.exp(2j * np.pi * (cell_frac @ np.asarray(q, float)))
    d = np.zeros((3 * nat, 3 * nat), dtype=complex)
    for i in range(nat):
        s_i = ref_sc_atoms[i]
        for j in range(nat):
            sel = prim_indices == j
            blk = np.einsum("sab,s->ab", fc2[s_i][sel], ph[sel])
            d[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = blk / np.sqrt(
                masses[i] * masses[j]
            )
    d = 0.5 * (d + d.conj().T)
    return d * CONVERSION_THZ2



def image_convention_audit(ph3, prim_indices, cell_frac, transport_direction="x"):
    """Where the vertex fold and the device Hamiltonian disagree on which image.

    ``get_btd_blocks_folded`` builds H from phonopy's dynamical matrix, whose
    Fourier sum uses the *shortest vectors* ``R + tau_kappa - tau_i``, averaged
    over ties. The vertex fold (``se_q._qfold_device_blocks``,
    ``separable.build_gathering_matrix``) uses one wrapped cell index per atom,
    with no basis offset and no tie average. Both live in convention B, so the
    gauge is not the issue -- the *image* is.

    Counts, over (supercell atom, reference primitive atom) pairs:

    * pairs whose shortest-vector images differ in their TRANSVERSE components,
      where phonopy averages and the fold does not;
    * pairs with an unambiguous transverse image that the fold nevertheless
      places elsewhere.

    Pair counts are an upper bound on what can go wrong, not an error: the
    weight actually carried by those pairs is set by the FC cutoff. The
    FC-weighted number is what S4 measures for the FC2 of this bed.
    """
    prim = ph3.primitive
    svecs, multi = prim.get_smallest_vectors()
    tau = np.asarray(prim.scaled_positions)
    tidx = "xyz".index(transport_direction)
    perp = [i for i in range(3) if i != tidx]
    n_super, nat = len(prim_indices), len(prim)
    tie = mism = total = 0
    for s in range(n_super):
        for i in range(nat):
            m, adr = int(multi[s, i, 0]), int(multi[s, i, 1])
            r = svecs[adr : adr + m] + tau[i] - tau[prim_indices[s]]
            r_perp = r[:, perp]
            total += 1
            if not np.allclose(r_perp, r_perp[0], atol=1e-6):
                tie += 1
            elif not np.allclose(r_perp[0], cell_frac[s][perp], atol=1e-6):
                mism += 1
    return {"pairs": total, "transversely_degenerate": tie, "wrapped_elsewhere": mism}


# ---------------------------------------------------------------------------
# report helpers
# ---------------------------------------------------------------------------


def _summarise(name, ratio, keep, *, expected=1.0):
    r = ratio[keep]
    if r.size == 0:
        print(f"  {name}: no entries above the magnitude floor")
        return
    med = float(np.median(r))
    spread = float(r.max() / r.min()) if r.min() > 0 else np.inf
    print(
        f"  {name}: n={r.size:6d}  median={med:.6f}  "
        f"min={r.min():.6f}  max={r.max():.6f}  max/min={spread:.4f}  "
        f"|median-{expected:g}|={abs(med - expected):.3e}"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mesh", type=int, nargs=3, default=[4, 4, 4])
    p.add_argument("--grid-point", type=int, default=5)
    p.add_argument("--floor", type=float, default=1e-6,
                   help="skip band triplets below this fraction of the "
                        "largest |Phi|^2 (symmetry-forbidden elements)")
    p.add_argument("--max-triplets", type=int, default=8)
    p.add_argument("--gate", action="store_true",
                   help="exit non-zero unless the two exact identities hold "
                        "(S2 everywhere, S3+gauge at commensurate q). Run it "
                        "with --mesh 2 2 2 --grid-point 1: that mesh is "
                        "commensurate with the 2x2x2 FC supercell.")
    args = p.parse_args(argv)

    from phonon.phonon_inputs.constants import (
        CONVERSION_FC3_THZ,
        CONVERSION_THZ2,
    )
    from phonon.phonon_inputs.convention import gauge_transform_A_to_B
    from phono3py.phonon3.real_to_reciprocal import RealToReciprocal

    print(f"mesh {args.mesh}  grid point {args.grid_point}")
    print(f"CONVERSION_FC3_THZ = {CONVERSION_FC3_THZ:.10e}")

    # ---- S0: phono3py C vs phono3py python ------------------------------
    print("\nS0  harness gate (phono3py C vs its own python reference)")
    dev = {}
    for r0 in (False, True):
        ph3 = load_ph3(args.mesh, r0_average=r0)
        itr = ph3.phph_interaction
        c = run_at_grid_point(itr, args.grid_point)
        itr.run(lang="Python")
        py = np.array(itr.interaction_strength, copy=True)
        m = c > c.max() * args.floor
        dev[r0] = float(np.abs(c[m] - py[m]).max() / c.max())
        print(f"  make_r0_average={str(r0):5s}  max rel dev C vs python = {dev[r0]:.3e}")
    if dev[False] > 1e-12:
        print("  GATE FAILED: the python reference does not reproduce the C kernel")
        return 1
    print(f"  -> gate passed. phono3py's r0 average is worth {dev[True]:.2%} in "
          f"|Phi|^2; the code does not apply it.")

    # the r0=False object is the one whose convention the code shares
    ph3 = load_ph3(args.mesh, r0_average=False)
    itr = ph3.phph_interaction
    strength = run_at_grid_point(itr, args.grid_point)
    triplets = np.array(itr.get_triplets_at_q()[0])
    freqs, eigvecs, _ = itr.get_phonons()
    addresses = itr.bz_grid.addresses
    unit_conv = itr.unit_conversion_factor
    mesh = np.array(itr.mesh_numbers)
    nat_prim = len(ph3.primitive)
    masses = ph3.primitive.masses
    mw = _mass_weight(masses)

    n_use = min(args.max_triplets, len(triplets))
    print(f"\n{len(triplets)} triplets at this grid point; using the first {n_use}")

    tensor_code, prim_indices, cell_frac, ref_sc_atoms = code_realspace_fc3(ph3)
    r2r_p3p = RealToReciprocal(ph3.fc3, ph3.primitive, mesh)
    r2r_code = RealToReciprocal(tensor_code, ph3.primitive, mesh)

    ratios_s1, ratios_s2 = [], []
    keep_s1, keep_s2 = [], []
    variants = (
        "as used (cell phases exp(-2pi i q.R), no basis gauge)",
        "cell phases exp(+2pi i q.R), no basis gauge",
        "cell phases exp(+2pi i q.R) + basis gauge",
    )
    ratios_s3 = {v: [] for v in variants}
    keep_s3 = []

    for t in range(n_use):
        gt = triplets[t]
        addr = addresses[gt]
        qs = addr.astype(float) / mesh
        ev = [eigvecs[g] for g in gt]
        f = [freqs[g] for g in gt]

        # phono3py reference, bare
        r2r_p3p.run(addr)
        phi_p3p = phi_bare(r2r_p3p.get_fc3_reciprocal(), ev, mass_weight=mw)
        ref = np.abs(phi_p3p) ** 2

        # phono3py's own answer, undone back to bare, as the S1 gate
        fff = f[0][:, None, None] * f[1][None, :, None] * f[2][None, None, :]
        bare_from_strength = strength[t] * fff / unit_conv

        floor = ref.max() * args.floor
        k = ref > floor
        keep_s1.append(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios_s1.append(bare_from_strength / np.where(k, ref, np.nan))

        # S2: the code's real-space tensor, phono3py's Fourier transform
        r2r_code.run(addr)
        phi_code = phi_bare(r2r_code.get_fc3_reciprocal(), ev)  # already mass-weighted
        s2 = np.abs(phi_code) ** 2
        keep_s2.append(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios_s2.append(s2 / np.where(k, ref, np.nan) / CONVERSION_FC3_THZ ** 2)

        # S3: the code's Fourier transform, same tensor, same eigenvectors
        ks3 = s2 > s2.max() * args.floor
        keep_s3.append(ks3)
        g1, g2, g3 = basis_gauge(ph3.primitive.scaled_positions, qs)
        for name in variants:
            sign = -1.0 if "exp(-2pi" in name else 1.0
            rec = code_reciprocal(
                tensor_code, prim_indices, cell_frac, ref_sc_atoms, nat_prim,
                qs[1], qs[2], sign2=sign, sign3=sign,
            )
            if "basis gauge" in name and "no basis" not in name:
                rec = rec * (g1[:, None, None, None, None, None]
                             * g2[None, :, None, None, None, None]
                             * g3[None, None, :, None, None, None])
            s3 = np.abs(phi_bare(rec, ev)) ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios_s3[name].append(s3 / np.where(ks3, s2, np.nan))

    k1 = np.concatenate([x.ravel() for x in keep_s1])
    k2 = np.concatenate([x.ravel() for x in keep_s2])
    k3 = np.concatenate([x.ravel() for x in keep_s3])
    print("\nS1  our contraction vs phono3py's interaction_strength (expect 1)")
    _summarise("bare |Phi|^2 ratio",
               np.concatenate([x.ravel() for x in ratios_s1]), k1)
    print("\nS2  NORMALISATION: code tensor / phono3py tensor, divided by "
          "CONVERSION_FC3_THZ^2 (expect 1)")
    _summarise("normalised ratio",
               np.concatenate([x.ravel() for x in ratios_s2]), k2)
    print("\nS3  PHASE CONVENTION: code Fourier transform / phono3py Fourier "
          "transform, same tensor (expect 1 for the matching convention)")
    for name, vals in ratios_s3.items():
        _summarise(name, np.concatenate([x.ravel() for x in vals]), k3)

    # ---- S4: is the code's cell-only Fourier sum exact at these q? --------
    print("\nS4  the same question one order down, eigenvector-free: phonon "
          "frequencies from the code's cell-convention D(q) vs phonopy's")
    seen = set()
    for t in range(n_use):
        for g in triplets[t]:
            if g in seen:
                continue
            seen.add(g)
            q = addresses[g].astype(float) / mesh
            d = cell_dynamical_matrix(
                ph3, prim_indices, cell_frac, ref_sc_atoms, q
            )
            w2 = np.linalg.eigvalsh(d)
            w = np.sign(w2) * np.sqrt(np.abs(w2))
            ref_w = np.sort(freqs[g])
            dev = float(np.abs(np.sort(w) - ref_w).max())
            # ... and the same comparison at matrix level, in the code's own
            # convention B, against the object the device H is actually built
            # from (convention.get_btd_blocks_folded -> gauge_transform_A_to_B).
            d_a = ph3.phph_interaction.dynamical_matrix
            d_a.run(q)
            d_b = gauge_transform_A_to_B(
                np.array(d_a.dynamical_matrix), q,
                np.asarray(ph3.primitive.scaled_positions),
            ) * CONVERSION_THZ2
            mat = float(np.abs(d - d_b).max() / (np.abs(d_b).max() + 1e-30))
            comm = np.allclose(np.mod(q * 2.0, 1.0), 0.0, atol=1e-9)
            print(f"  q={np.array2string(q, precision=3):22s} "
                  f"{'commensurate' if comm else 'INcommensurate':14s} "
                  f"max |dw| = {dev:.3e} THz   rel |dD_B| = {mat:.3e}")

    # ---- S5: which images the two sides pick ----------------------------
    aud = image_convention_audit(ph3, prim_indices, cell_frac)
    n = aud["pairs"]
    print("\nS5  image convention: H (phonopy shortest vectors, tie-averaged) "
          "vs the vertex fold (one wrapped cell index)")
    print(f"  (supercell atom, reference atom) pairs: {n}")
    print(f"  transversely-degenerate images (H averages, the fold does not): "
          f"{aud['transversely_degenerate']} ({aud['transversely_degenerate']/n:.1%})")
    print(f"  unambiguous but the fold places them elsewhere:                 "
          f"{aud['wrapped_elsewhere']} ({aud['wrapped_elsewhere']/n:.1%})")
    print("  pair counts bound what can differ; the FC-weighted size of it is "
          "the rel |dD_B| column of S4.")

    if args.gate:
        s2_dev = float(np.abs(
            np.concatenate([x.ravel() for x in ratios_s2])[k2] - 1.0).max())
        gauged = np.concatenate(
            [x.ravel() for x in ratios_s3["cell phases exp(+2pi i q.R) + basis gauge"]]
        )
        s3_dev = float(np.abs(gauged[k3] - 1.0).max())
        commensurate = all(
            np.allclose(np.mod(addresses[g].astype(float) / mesh * 2.0, 1.0), 0.0,
                        atol=1e-9)
            for t in range(n_use) for g in triplets[t]
        )
        print(f"\nGATE  S2 max dev {s2_dev:.3e} (< 1e-12 required)")
        ok = s2_dev < 1e-12
        if commensurate:
            print(f"      S3 max dev {s3_dev:.3e} (< 1e-10 required, "
                  f"commensurate mesh)")
            ok = ok and s3_dev < 1e-10
        else:
            print("      S3 not gated: this mesh is not commensurate with the "
                  "FC supercell, where the two sides legitimately pick "
                  "different images")
        print("      " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
