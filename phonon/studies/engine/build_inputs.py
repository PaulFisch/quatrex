"""Build PRODUCTION quatrex phonon-transport inputs for a benchmark system.

    ``qfold_vertices.npz``. Writes the above plus ``qfold_vertices.npz`` and a
Usage:
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

# Repo layout: <root>/phonon and <root> both importable (phonon_inputs.*,
# phonon.*, quatrex.*).
_ROOT = Path(__file__).resolve().parents[3]
_PHON = _ROOT / "phonon"
for _p in (_ROOT, _PHON):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import h5py
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonon_inputs.convention import get_btd_blocks_folded
from phonon_inputs.constants import CONVERSION_THZ2
from phonon_inputs.separable import (
    build_supercell_mapping,
    build_realspace_fc3_matrices,
    build_q_diff_map,
)
from phonon.solver.fc3_device import build_device_fc3_blocks
from phonon_inputs.quatrex_writer import write_fc3_blocks, write_structure_xyz

# Gamma-only (k==1) finite-device FC3 reaps. build_cnt() is species-agnostic, so
# the multi-species SiNW (Si+H) wires flow through it unchanged.
CNT_DIR = {
    "cnt33": _PHON / "configs/cnt/fc3_hiphive_cnt33_vasp",
    "cnt80": _PHON / "configs/cnt/fc3_hiphive_cnt80_vasp",
    "sinw_d5a": _PHON / "configs/sinw/fc3_hiphive_sinw100_d5a_sc4_vasp",
    "sinw_d11a": _PHON / "configs/sinw/fc3_hiphive_sinw100_d11a_vasp",
    # SrTiO3 (5-atom cubic perovskite, [3,3,3]) -- Gamma-only finite slab,
    # best-effort strong-anharmonicity transport (the bulk is transversely
    # periodic; this is a finite-cross-section device).
    "srtio3": _PHON / "configs/perovskite/fc3_hiphive_srtio3_333_vasp",
}


def _key_str(k):
    """scipy.savemat real-space key, matching the production .mat loader."""
    return f"[{k[0]}, {k[1]}, {k[2]}]"


def build_cnt(system, ncells, tdir, nfreq, fmax, emin, out):
    """Gamma-only (k==1) CNT device. Port of /tmp/build_cnt_inputs.py."""
    D = CNT_DIR[system]
    meta = json.load(open(D / "hiphive_meta.json"))
    prim = meta["primitive"]
    scm = meta["supercell"]
    ti = "xyz".index(tdir)
    n_qz = int(scm[ti] if not isinstance(scm, dict) else scm["xyz"[ti]])
    unit = PhonopyAtoms(
        symbols=prim["symbols"],
        cell=np.array(prim["cell"]),
        scaled_positions=np.array(prim["scaled_positions"]),
    )
    phonon = Phonopy(unit, supercell_matrix=np.diag(scm), primitive_matrix=np.eye(3))
    with h5py.File(D / "fc3.hdf5", "r") as f:
        fc2 = f["fc2"][...]
        fc3_raw = f["fc3"][...]
    phonon.force_constants = fc2
    nat = len(phonon.primitive.masses)
    nd = 3 * nat
    print(
        f"{system}: prim {nat} atoms, ndof/cell={nd}, supercell {scm} "
        f"(n_qz={n_qz}), transport={tdir}, ncells={ncells}, "
        f"grid {emin}-{fmax}/{nfreq}",
        flush=True,
    )

    # Beyond-nearest-cell Fourier coefficients (n_qz >= 4) are FOLDED
    # into H00: exact at Gamma (acoustic sum rule, twist, optical onset)
    # and at the zone boundary; without the fold the emitted BT matrix
    # has shifted / imaginary translational modes (the d5a/d11a
    # corruption, 2026-07: Gamma [-0.27, -0.15, ...] instead of zeros).
    h00, h01, fold = get_btd_blocks_folded(
        phonon, (0.0, 0.0), transport_direction=tdir, n_qz=n_qz,
        conversion_factor=CONVERSION_THZ2,
    )
    print(
        f"||H00||={np.linalg.norm(h00):.1f} ||H01||={np.linalg.norm(h01):.1f} "
        f"THz^2; H00 herm err {np.abs(h00 - h00.conj().T).max():.1e}; "
        f"folded n>=2 coeffs {fold['fold_norms']} "
        f"(midzone bound {fold['midzone_bound']:.2f} THz^2)",
        flush=True,
    )
    # Dispersion sanity: SIGNED frequencies -- imaginary modes must fail
    # loudly, not be clipped away (the old np.clip hid the truncation
    # corruption).
    fr = []
    for kz in np.linspace(0, 0.5, 21):
        hk = h00 + h01 * np.exp(2j * np.pi * kz) + h01.conj().T * np.exp(-2j * np.pi * kz)
        w2 = np.linalg.eigvalsh(hk)
        fr.append(np.sign(w2) * np.sqrt(np.abs(w2)))
    fr = np.array(fr)
    print(f"dispersion: min {fr.min():.4f}, max {fr.max():.3f} THz; "
          f"lowest-5 @Gamma (signed): {np.round(fr[0, :5], 4)}", flush=True)
    if fr.min() < -1e-3:
        raise SystemExit(
            f"IMAGINARY modes in the emitted BT dispersion (min "
            f"{fr.min():.4f} THz): the export is corrupt -- refusing to "
            "write inputs."
        )

    def key(n):
        k = [0, 0, 0]
        k[ti] = n
        return _key_str(k)

    savemat(str(out / "dynamical_matrix.mat"), {
        key(0): h00.astype(complex),
        key(1): h01.astype(complex),
        key(-1): h01.conj().T.astype(complex),
    })
    np.save(out / "phonon_energies.npy", np.linspace(emin, fmax, nfreq).astype(float))

    prim_idx, _cf, slab_idx, ref = build_supercell_mapping(phonon, tdir)
    M = build_realspace_fc3_matrices(
        fc3_raw, nat, phonon.supercell.masses, ref,
    )
    phi = build_device_fc3_blocks(M, prim_idx, slab_idx, nat, ncells)
    phi = {k: np.ascontiguousarray(v.astype(complex)) for k, v in phi.items()}
    write_fc3_blocks(phi, np.array([nd] * ncells), out / "fc3_blocks.hdf5", units="THz^2")
    write_structure_xyz(phonon.primitive, out / "structure.xyz")
    print(f"fc3 device: {len(phi)} blocks, off-diag "
          f"{sum(1 for k in phi if k[1] != k[2])}, "
          f"max|Phi|={max(np.abs(v).max() for v in phi.values()):.3e}", flush=True)


def _load_bulk_si(fc3_subdir):
    """Bulk-Si phonopy (FD FC2) + the FC3 path. Port of build_sifilm."""
    from phono3py import load as phono3py_load
    d = _PHON / fc3_subdir
    with h5py.File(d / "fc2.hdf5", "r") as f:
        fc2 = f["force_constants"][:]
    ph3 = phono3py_load(phono3py_yaml=str(d / "phono3py.yaml"), produce_fc=False, log_level=0)
    cell = PhonopyAtoms(symbols=ph3.unitcell.symbols, cell=ph3.unitcell.cell,
                        scaled_positions=ph3.unitcell.scaled_positions)
    phonon = Phonopy(cell, supercell_matrix=ph3.supercell_matrix, primitive_matrix=np.eye(3))
    phonon.force_constants = fc2
    return phonon, str(d / "fc3.hdf5")


def _load_bulk_hiphive(fc3_dir):
    """Bulk phonopy from a hiphive-format reap (hiphive_meta.json +
    fc3.hdf5 carrying both fc2 and fc3 datasets) -- the CNT reap
    format, lattice-agnostic (hexagonal MoS2, monoclinic TiS3, ...)."""
    d = Path(fc3_dir)
    if not d.is_absolute():
        d = _PHON / d
    meta = json.load(open(d / "hiphive_meta.json"))
    prim = meta["primitive"]
    unit = PhonopyAtoms(
        symbols=prim["symbols"],
        cell=np.array(prim["cell"]),
        scaled_positions=np.array(prim["scaled_positions"]),
    )
    phonon = Phonopy(unit, supercell_matrix=np.diag(meta["supercell"]),
                     primitive_matrix=np.eye(3))
    with h5py.File(d / "fc3.hdf5", "r") as f:
        phonon.force_constants = f["fc2"][...]
    return phonon, str(d / "fc3.hdf5")


def _load_bulk_film(fc3_subdir):
    """Dispatch on the reap format: hiphive_meta.json marks the CNT/
    hiphive layout; otherwise the phono3py (bulk-Si) layout."""
    d = Path(fc3_subdir)
    if not d.is_absolute():
        d = _PHON / d
    if (d / "hiphive_meta.json").exists():
        return _load_bulk_hiphive(d)
    return _load_bulk_si(fc3_subdir)


def _decompose_film_vertices(M_stacked, prim_idx, cell_frac, slab_idx, nat,
                             q_points, q_diff_map, nk, tdir, ranks, ansatz,
                             cache_dir, out, dense_vertices=None,
                             masses_super=None, support="full",
                             fit_kwargs=None, cache_label=None):
    """Fit the bulk FC3 (cached per (ansatz, rank, tensor-hash)), gather the
    per-(offset, q) device factor arrays and write
    ``decomposed_vertices[_r{R}].npz``. Self-check: reconstruct sample folded
    blocks from the factors against the dense qfold entries (or a directly
    folded sample pair when the dense dict is not built) -- pins the phase
    conventions; agreement is bounded by the fit rel_err."""
    from phonon.phonon_inputs.fc3_factor_device import (
        build_device_factor_arrays,
        fit_film_fc3_factors,
    )
    from phonon.solver.se_q import _qfold_device_blocks
    from quatrex.phonon.vertex_factors import VertexFactors, save_decomposed

    n_super = len(prim_idx)
    support_pairs = None
    if support == "dense":
        # (dK, dK') offset pairs the DENSE FC3 populates (zero blocks
        # dropped) -- the factored path otherwise manufactures vertex
        # blocks over the full offs x offs window (support asymmetry).
        phi_off = build_device_fc3_blocks(
            M_stacked, prim_idx, slab_idx, nat, 1, return_offsets=True)
        support_pairs = sorted((int(a), int(b)) for (a, b) in phi_off)
        print(f"[decompose] dense support: {len(support_pairs)} offset "
              f"pairs {support_pairs}", flush=True)
    paths = []
    for rank in ranks:
        export = fit_film_fc3_factors(
            M_stacked, nat, n_super, rank, ansatz=ansatz, cache_dir=cache_dir,
            masses_super=masses_super,
            cache_label=cache_label, **(fit_kwargs or {}))
        arrays = build_device_factor_arrays(
            export, prim_idx, cell_frac, slab_idx, nat, q_points, tdir)
        vf = VertexFactors(
            D=arrays["D"], lambdas=arrays["lambdas"],
            offsets=arrays["offsets"], UB=arrays["UB"], UC=arrays["UC"],
            q_diff_map=np.asarray(q_diff_map, dtype=np.int64),
            nk_shape=(nk, nk), ansatz=ansatz,
            meta={**arrays["meta"], "fc3_rank": int(rank),
                  **({"support_pairs": support_pairs}
                     if support_pairs is not None else {})},
        )
        rel_err = float(vf.meta.get("rel_err", np.nan))

        # Phase-convention self-check on sample (q1, q2) pairs.
        n_kpts = len(q_points)
        pos = vf.offset_index()
        num2 = den2 = 0.0
        off_cls: dict = {}
        for (iq1, iq2) in {(0, 0), (1, min(2, n_kpts - 1)),
                           (n_kpts - 1, min(3, n_kpts - 1))}:
            if dense_vertices is not None and (iq1, iq2) in dense_vertices:
                sample = dense_vertices[(iq1, iq2)]
            else:
                sample = _qfold_device_blocks(
                    M_stacked, prim_idx, cell_frac, slab_idx, nat,
                    int(slab_idx.max()) + 1, q_points[iq1], q_points[iq2],
                    tdir)
            for (I, K, Kp), dense_blk in sample.items():
                dK, dKp = K - I, Kp - I
                if dK not in pos or dKp not in pos:
                    continue
                rec = vf.reconstruct_block(iq1, iq2, dK, dKp)
                _e2 = float(np.linalg.norm(rec - dense_blk) ** 2)
                _d2 = float(np.linalg.norm(dense_blk) ** 2)
                num2 += _e2
                den2 += _d2
                oc = off_cls.setdefault((dK, dKp), [0.0, 0.0])
                oc[0] += _e2
                oc[1] += _d2
        # NORM-WEIGHTED aggregate: a global-Frobenius fit leaves O(1)
        # RELATIVE errors on weak blocks by construction (MoS2: the
        # cross-slab vertex carries ~0.5% of the diagonal weight), so
        # the old max-single-block-relative gate misfires exactly when
        # the fit is honest. The aggregate tracks rel_err for a
        # convention-correct chain; a phase/gauge break shows up as
        # aggregate >> rel_err.
        agg = float(np.sqrt(num2 / max(den2, 1e-300)))
        per_off = {k: float(np.sqrt(e2 / max(d2, 1e-300)))
                   for k, (e2, d2) in sorted(off_cls.items())}
        vf.meta = {
            **vf.meta,
            "qfold_sample_aggregate_rel_err": agg,
            "qfold_sample_per_offset_rel_err": {
                f"{a},{b}": value for (a, b), value in per_off.items()
            },
        }
        print(f"[decompose r{rank}] fit rel_err={rel_err:.4f}; sample "
              f"aggregate rel err={agg:.4f}; per-offset "
              f"{ {k: round(v, 3) for k, v in per_off.items()} }", flush=True)
        weak = [k for k, v in per_off.items() if v > 0.5]
        if weak:
            print(f"[decompose r{rank}] WARNING: offset classes {weak} are "
                  "fit-noise dominated (weak-block physics NOT captured at "
                  "this rank -- cf. the ARDR cross-gap pruning).", flush=True)
        if agg > 1.5 * max(rel_err, 1e-12) + 1e-10:
            raise RuntimeError(
                f"decomposed-vertex self-check FAILED at rank {rank}: sample "
                f"aggregate rel err {agg:.3e} >> fit rel_err {rel_err:.3e} "
                "-- phase-convention mismatch, not fit error.")

        tag = "" if rank == max(ranks) else f"_r{rank}"
        path = out / f"decomposed_vertices{tag}.npz"
        save_decomposed(path, vf)
        print(f"{path.name}: R={vf.rank} ansatz={ansatz} "
              f"({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        paths.append(path)
    return paths


def build_sifilm(nslabs, nk, tdir, nfreq, fmax, emin, fc3_subdir, out, nproc=1,
                 decompose_ranks=(), decompose_ansatz="INDSCAL",
                 decompose_support="full",
                 decompose_only=False, factorised_only=False,
                 harmonic_only=False, decompose_fit_kwargs=None,
                 decompose_cache_label=None):
    """Transversely-periodic (k>1) Si film. Port of /tmp/build_sifilm_inputs.py."""
    assert nk % 2 == 1, "use ODD nk (clean Gamma-centered IDFT)"
    phonon, fc3_path = _load_bulk_film(fc3_subdir)
    nat = len(phonon.primitive.masses)
    nd = 3 * nat
    tidx = "xyz".index(tdir)
    perp = [i for i in range(3) if i != tidx]
    print(f"Si film: prim {nat} atoms, nd/cell={nd}, transport={tdir}, perp={perp}, "
          f"nk={nk}x{nk}, n_slabs={nslabs}, fc3={fc3_subdir}", flush=True)

    q_1d = np.arange(nk) / nk
    q_points = [(qa, qb) for qa in q_1d for qb in q_1d]
    q_diff_map = build_q_diff_map(nk, nk)
    n_kpts = nk * nk

    if decompose_only:
        # Add the factor file(s) to an ALREADY-BUILT geometry dir without
        # redoing the O(nk^2) dense fold (or touching any other input file).
        prim_idx, cell_frac, slab_idx, ref_sc = build_supercell_mapping(
            phonon, tdir)
        with h5py.File(fc3_path, "r") as f:
            fc3 = f["fc3"][:]
        M_stacked = build_realspace_fc3_matrices(
            fc3, nat, phonon.supercell.masses, ref_sc)
        _decompose_film_vertices(
            M_stacked, prim_idx, cell_frac, slab_idx, nat, q_points,
            q_diff_map, nk, tdir, decompose_ranks, decompose_ansatz,
            Path(fc3_path).parent, out, dense_vertices=None,
            masses_super=np.asarray(phonon.supercell.masses, dtype=float),
            support=decompose_support, fit_kwargs=decompose_fit_kwargs,
            cache_label=decompose_cache_label)
        return

    H00 = np.zeros((n_kpts, nd, nd), complex)
    H01 = np.zeros((n_kpts, nd, nd), complex)
    for iq, (qa, qb) in enumerate(q_points):
        # Folded variant: a no-op at the default n_qz=3 (no n>=2
        # coefficients exist), future-proof for wider transport meshes.
        h00, h01, _fold = get_btd_blocks_folded(
            phonon, (qa, qb), transport_direction=tdir,
            conversion_factor=CONVERSION_THZ2)
        H00[iq], H01[iq] = h00, h01
    print(f"dense H(q): ||H00(G)||={np.linalg.norm(H00[0]):.1f} "
          f"||H01(G)||={np.linalg.norm(H01[0]):.1f} THz^2; "
          f"H00 herm err {np.abs(H00[0] - H00[0].conj().T).max():.1e}", flush=True)

    rng = range(-(nk // 2), nk // 2 + 1)
    cells = [(cy, cz) for cy in rng for cz in rng]
    qarr = np.array(q_points)
    mats = {}
    for (cy, cz) in cells:
        ph = np.exp(-2j * np.pi * (cy * qarr[:, 0] + cz * qarr[:, 1])) / n_kpts
        D00 = np.tensordot(ph, H00, axes=(0, 0))
        Dp1 = np.tensordot(ph, H01, axes=(0, 0))
        Dm1 = np.tensordot(ph, np.conj(np.transpose(H01, (0, 2, 1))), axes=(0, 0))

        def key(ix):
            k = [0, 0, 0]
            k[tidx] = ix
            k[perp[0]] = cy
            k[perp[1]] = cz
            return (k[0], k[1], k[2])

        mats[key(0)] = D00.astype(complex)
        mats[key(1)] = Dp1.astype(complex)
        mats[key(-1)] = Dm1.astype(complex)

    def fold(ix, qa, qb):
        acc = np.zeros((nd, nd), complex)
        for (cy, cz) in cells:
            kk = [0, 0, 0]
            kk[tidx] = ix
            kk[perp[0]] = cy
            kk[perp[1]] = cz
            acc += mats[(kk[0], kk[1], kk[2])] * np.exp(2j * np.pi * (cy * qa + cz * qb))
        return acc

    worst = 0.0
    for iq in sorted({0, min(1, n_kpts - 1), n_kpts // 3, n_kpts - 1}):
        qa, qb = q_points[iq]
        e00 = np.abs(fold(0, qa, qb) - H00[iq]).max() / (np.abs(H00[iq]).max() + 1e-30)
        e01 = np.abs(fold(1, qa, qb) - H01[iq]).max() / (np.abs(H01[iq]).max() + 1e-30)
        worst = max(worst, e00, e01)
    print(f"IDFT round-trip self-check: worst rel err = {worst:.2e}", flush=True)
    assert worst < 1e-10, "round-trip FAILED"

    savemat(str(out / "dynamical_matrix.mat"),
            {_key_str(k): v for k, v in mats.items()})
    print(f"dynamical_matrix.mat: {len(mats)} blocks "
          f"({len(cells)} transverse cells x 3 transport)", flush=True)

    if harmonic_only:
        # The production driver constructs the phonon-phonon object before
        # QX_BALLISTIC removes it. Supply a shape-correct zero vertex so a
        # genuinely harmonic geometry does not need an expensive FC3 fold or
        # factorisation merely to pass that initialisation boundary.
        zero_blocks = {
            (cell, cell, cell): np.zeros((nd, nd, nd), dtype=complex)
            for cell in range(nslabs)
        }
        write_fc3_blocks(
            zero_blocks,
            np.array([nd] * nslabs),
            out / "fc3_blocks.hdf5",
            units="THz^2",
        )
        np.savez_compressed(
            out / "decomposed_vertices.npz",
            format_version=np.int64(1),
            D=np.zeros((nd, 1), dtype=float),
            lambdas=np.zeros(1, dtype=float),
            offsets=np.zeros(1, dtype=np.int64),
            UB=np.zeros((1, n_kpts, nd, 1), dtype=complex),
            UC=np.zeros((1, n_kpts, nd, 1), dtype=complex),
            q_diff_map=np.asarray(q_diff_map, dtype=np.int64),
            nk_shape=np.asarray((nk, nk), dtype=np.int64),
            ansatz="INDSCAL",
            meta=np.array(
                {"harmonic_placeholder": True, "n_dof": nd}, dtype=object
            ),
        )
        write_structure_xyz(phonon.primitive, out / "structure.xyz")
        np.save(
            out / "phonon_energies.npy",
            np.linspace(emin, fmax, nfreq).astype(float),
        )
        shift = round(0.5 - 0.5 / nk, 10)
        np.save(out / "kshift.npy", np.array(shift))
        print(
            "harmonic-only input: wrote shape-correct zero FC3 placeholders; "
            "run only with QX_BALLISTIC=1",
            flush=True,
        )
        return

    from phonon.solver.se_q import (
        _build_folded_vertices, _qfold_device_blocks,
    )
    from quatrex.phonon.qfold import save_qfold

    prim_idx, cell_frac, slab_idx, ref_sc = build_supercell_mapping(phonon, tdir)
    with h5py.File(fc3_path, "r") as f:
        fc3 = f["fc3"][:]
    M_stacked = build_realspace_fc3_matrices(fc3, nat, phonon.supercell.masses, ref_sc)
    vertices = None
    if factorised_only:
        if not decompose_ranks:
            raise ValueError(
                "factorised_only requires at least one decomposition rank")
        print("skipping the dense q-folded vertex; writing primitive factors "
              "and the Gamma FC3 oracle only", flush=True)
    else:
        print(f"building {n_kpts}^2-pair folded vertices "
              f"(n_slabs={nslabs}, nproc={nproc})...", flush=True)
        vertices = _build_folded_vertices(
            M_stacked, prim_idx, cell_frac, slab_idx, nat,
            nslabs, n_kpts, q_points, q_diff_map, tdir, nproc=nproc)
        print(f"  folded vertex pairs: {len(vertices)}", flush=True)
        save_qfold(out / "qfold_vertices.npz", vertices, q_diff_map, (nk, nk))

    if decompose_ranks:
        _decompose_film_vertices(
            M_stacked, prim_idx, cell_frac, slab_idx, nat, q_points,
            q_diff_map, nk, tdir, decompose_ranks, decompose_ansatz,
            Path(fc3_path).parent, out, dense_vertices=vertices,
            masses_super=np.asarray(phonon.supercell.masses, dtype=float),
            support=decompose_support, fit_kwargs=decompose_fit_kwargs,
            cache_label=decompose_cache_label)

    gamma_blocks = (
        vertices[(0, 0)] if vertices is not None
        else _qfold_device_blocks(
            M_stacked, prim_idx, cell_frac, slab_idx, nat, nslabs,
            q_points[0], q_points[0], tdir)
    )
    gamma = {k: np.ascontiguousarray(v.astype(complex))
             for k, v in gamma_blocks.items()}
    write_fc3_blocks(gamma, np.array([nd] * nslabs), out / "fc3_blocks.hdf5", units="THz^2")
    print(f"fc3_blocks.hdf5: {len(gamma)} Gamma device blocks", flush=True)
    write_structure_xyz(phonon.primitive, out / "structure.xyz")
    np.save(out / "phonon_energies.npy", np.linspace(emin, fmax, nfreq).astype(float))
    print(f"phonon_energies.npy: {nfreq} pts {emin}..{fmax} THz", flush=True)

    shift = round(0.5 - 0.5 / nk, 10)
    np.save(out / "kshift.npy", np.array(shift))
    print(f"REQUIRED config: kpoint_grid with {nk} on the two perp axes {perp}, "
          f"kpoint_shift={shift} on those axes, transport={tdir}, "
          f"num_transport_cells={nslabs}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", required=True,
                   choices=["cnt33", "cnt80", "sinw_d5a", "sinw_d11a",
                            "srtio3", "sifilm", "mos2film"])
    p.add_argument("-L", "--ncells", type=int, default=2, help="CNT transport cells")
    p.add_argument("--nslabs", type=int, default=5, help="film layers along transport")
    p.add_argument("--nk", type=int, default=8, help="film transverse mesh (odd)")
    p.add_argument("--tdir", default=None, help="transport axis (CNT: z, film: x)")
    p.add_argument("--nfreq", type=int, default=None)
    p.add_argument("--fmax", type=float, default=None)
    p.add_argument("--emin", type=float, default=0.0,
                   help="grid start (0.0 = w0=0, best heat-flow conservation)")
    p.add_argument("--fc3-subdir", default="reaps/si_big_hiphive",
                   help="film bulk-Si FC reap (default the 5^3 hiphive)")
    # The FC3 vertex is used raw (plain truncation). The former ASR
    # projection was removed 2026-06-12: it was leg-asymmetric (broke the
    # vertex S3 symmetry -> energy conservation) and over-strong (suppressed
    # linewidths ~4-5x vs phono3py); hiphive fits are ASR-exact raw.
    p.add_argument("--out", required=True)
    p.add_argument("--nproc", type=int, default=1,
                   help="parallel workers for the O(nk^2) folded-vertex build")
    p.add_argument("--decompose-support", default="full",
                   choices=["full", "dense"],
                   help="factor offset window: 'dense' restricts the "
                        "generated vertex blocks to the (dK, dK') pairs "
                        "the dense FC3 populates (kills the invented "
                        "out-of-support weight); 'full' is legacy")
    p.add_argument("--decompose-ranks", default="",
                   help="comma-separated CP ranks; fit the bulk FC3 and write "
                        "decomposed_vertices[_r{R}].npz next to the qfold "
                        "(highest rank gets the untagged name; truncate at "
                        "runtime via sse_vertex_rank)")
    p.add_argument("--decompose-ansatz", default="INDSCAL",
                   choices=["INDSCAL", "CP", "S2CP"],
                   help="factorisation ansatz (INDSCAL = shared contracted "
                        "legs; S2CP = paired CP with exact contracted-leg "
                        "symmetry)")
    p.add_argument("--decompose-restarts", type=int, default=None,
                   help="override the production factor-fit restart count; "
                        "requires --decompose-cache-label so a reduced study "
                        "fit cannot masquerade as the default fit")
    p.add_argument("--decompose-max-iter", type=int, default=None,
                   help="override ALS iterations per restart; requires "
                        "--decompose-cache-label")
    p.add_argument("--decompose-lbfgs-iters", type=int, default=None,
                   help="override CP/INDSCAL L-BFGS iterations; requires "
                        "--decompose-cache-label")
    p.add_argument("--decompose-cache-label", default=None,
                   help="cache namespace recorded in factor metadata for a "
                        "non-default fit schedule")
    p.add_argument("--decompose-only", action="store_true",
                   help="only add the factor file(s) to an already-built "
                        "geometry dir (skips the dense O(nk^2) fold and all "
                        "other input files)")
    p.add_argument("--factorised-only", "--factorized-only",
                   action="store_true",
                   help="build a new film geometry and independently fitted "
                        "factors without materialising the dense O(nk^4) "
                        "q-folded vertex; retain a Gamma FC3 oracle")
    p.add_argument("--harmonic-only", action="store_true",
                   help="write the FC2 geometry plus shape-correct zero FC3 "
                        "placeholders for QX_BALLISTIC=1; skip every real "
                        "FC3 contraction and factorisation")
    a = p.parse_args()
    if a.decompose_only and a.factorised_only:
        p.error("--decompose-only and --factorised-only are mutually exclusive")
    if a.harmonic_only and (
        a.decompose_only or a.factorised_only or a.decompose_ranks
    ):
        p.error("--harmonic-only cannot be combined with decomposition options")
    fit_kwargs = {
        key: value for key, value in (
            ("n_restarts", a.decompose_restarts),
            ("max_iter", a.decompose_max_iter),
            ("lbfgs_iters", a.decompose_lbfgs_iters),
        ) if value is not None
    }
    if fit_kwargs and not a.decompose_cache_label:
        p.error("non-default factor-fit controls require "
                "--decompose-cache-label")
    if (a.decompose_ansatz not in ("INDSCAL", "CP", "S2CP")
            and any(key in fit_kwargs for key in ("max_iter", "lbfgs_iters"))):
        p.error("the selected factor ansatz does not accept ALS/L-BFGS "
                "iteration controls")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.system in ("cnt33", "cnt80", "sinw_d5a", "sinw_d11a", "srtio3"):
        tdir = a.tdir or "z"
        if a.system in ("cnt33", "cnt80"):
            nfreq = a.nfreq or 161
            fmax = a.fmax or 55.0
        elif a.system == "srtio3":  # perovskite optical max ~24 THz
            nfreq = a.nfreq or 121
            fmax = a.fmax or 26.0
        else:  # SiNW: Si optical max ~15.5 THz -> tighter window than the CNT's 55
            nfreq = a.nfreq or 101
            fmax = a.fmax or 18.0
        emin = a.emin if a.emin is not None else 0.0
        build_cnt(a.system, a.ncells, tdir, nfreq, fmax, emin, out)
    else:
        if a.system == "mos2film":
            # 2H-MoS2 cross-plane (kappa_z): transport along c, hexagonal
            # transverse q; spectrum tops at ~14.1 THz. The reap must be
            # the [4,4,3]-rematerialized one (phonon/studies/
            # _mos2_film_reap.py) -- the [4,4,1] fit cannot separate
            # H00/H01 along c.
            tdir = a.tdir or "z"
            fc3_subdir = (a.fc3_subdir
                          if a.fc3_subdir != "reaps/si_big_hiphive"
                          else "../cluster/mos2_film_reap")
            fmax = a.fmax or 16.0
        else:
            tdir = a.tdir or "x"
            fc3_subdir = a.fc3_subdir
            fmax = a.fmax or 15.0
        nfreq = a.nfreq or 121
        ranks = tuple(int(r) for r in a.decompose_ranks.split(",") if r)
        build_sifilm(a.nslabs, a.nk, tdir, nfreq, fmax, a.emin, fc3_subdir,
                     out, a.nproc, decompose_ranks=ranks,
                     decompose_ansatz=a.decompose_ansatz,
                     decompose_support=a.decompose_support,
                     decompose_only=a.decompose_only,
                     factorised_only=a.factorised_only,
                     harmonic_only=a.harmonic_only,
                     decompose_fit_kwargs=fit_kwargs,
                     decompose_cache_label=a.decompose_cache_label)
    print(f"inputs -> {out}", flush=True)


if __name__ == "__main__":
    main()
