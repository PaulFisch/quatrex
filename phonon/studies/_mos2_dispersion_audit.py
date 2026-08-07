"""Dispersion audit of a built device: is omega(q) linear around omega=0?

Paul's question (a) of the MoS2 conservation audit: non-linear
dispersion near omega = 0 changes the near-DC phase space AND the
opening exponent of the lead broadening Gamma(omega) ~ omega^p. The
whole infrared treatment (and the CM-channel derivation,
phonon/docs/ir_residue_derivation.md) assumes p = 1, which holds only
for a LINEAR acoustic branch. A quadratic (flexural/membrane) branch
gives p = 1/2 and breaks that assumption.

Builds H(q_perp, q_z) from a production ``dynamical_matrix.mat`` in the
exact production conventions:
  transverse   H_t(q_perp) = sum_c D[c, t] exp(+2i pi c . q_perp)
               (phonon/studies/engine/build_inputs.py:386 `fold`,
                src/quatrex/device/inputs.py:424 `_assemble_kpoint`)
  transport    H(q_perp, q_z) = H_0 + H_+1 e^{+2i pi q_z}
                                    + H_-1 e^{-2i pi q_z}
               (build_inputs.py:124, the CNT dispersion self-check)
Frequencies are SIGNED, sign(w2)*sqrt(|w2|) via
phonon.postproc.spectral.frequencies_from_dynamical, so soft/imaginary
modes are reported, not clipped (make_grid._modes_from_dyn clips and
must not be used here).

Reports, per system:
  1. hermiticity / D(-q) = D(q)* / reality gates on the input blocks
  2. Gamma-point spectrum (3 acoustic zeros expected)
  3. log-log exponent alpha in omega ~ q^alpha for the lowest branches,
     in-plane and along transport
  4. imaginary-mode census on the ACTUALLY SAMPLED transverse mesh
     (and on a fine mesh, separating "unstable model" from "unsampled
     instability")
  5. the sound velocities and the implied lead-opening exponent

Run:  QTX_ARRAY_MODULE=numpy python phonon/studies/_mos2_dispersion_audit.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "phonon"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.postproc.spectral import frequencies_from_dynamical  # noqa: E402

OUT = ROOT / "phonon/studies/out/mos2_conservation"

# (label, matfile, transport axis index, transverse mesh n per axis)
SYSTEMS = [
    ("MoS2 film L3 (15-block build)",
     "cluster/mos2film_L3_nk5_ls/dynamical_matrix.mat", 2, 5),
    ("MoS2 film L3 (SCP fc2)",
     "cluster/mos2film_L3_nk5_scp/dynamical_matrix.mat", 2, 5),
    ("MoS2 film L3 (mos2f3 build)",
     "cluster/mos2f3/dynamical_matrix.mat", 2, 5),
    ("Si film (control, converges)",
     "cluster/sifilm_nk9r/dynamical_matrix.mat", 0, 9),
    ("CNT33 L4 (control, converges)",
     "phonon/studies/out/anderson_test/cnt33_L4_inputs/dynamical_matrix.mat",
     2, 1),
]


def load_blocks(path: Path, t_axis: int):
    """{transport offset: {(c_perp): block}} from an offset-keyed .mat."""
    raw = loadmat(str(path))
    perp = [i for i in range(3) if i != t_axis]
    blocks: dict[int, dict[tuple, np.ndarray]] = {}
    for key, val in raw.items():
        if key.startswith("__"):
            continue
        off = [int(x) for x in re.findall(r"-?\d+", key)]
        blocks.setdefault(off[t_axis], {})[(off[perp[0]], off[perp[1]])] = val
    return blocks


def h_of_q(blocks, q_perp, q_z):
    """H(q_perp, q_z), production phase conventions."""
    ht = {}
    for t, cells in blocks.items():
        acc = None
        for c, mat in cells.items():
            ph = np.exp(2j * np.pi * (c[0] * q_perp[0] + c[1] * q_perp[1]))
            acc = mat * ph if acc is None else acc + mat * ph
        ht[t] = acc
    h = ht[0].copy()
    for t, mat in ht.items():
        if t:
            h = h + mat * np.exp(2j * np.pi * t * q_z)
    return h


def gates(blocks, label):
    """Hermiticity, D(-q)=D(q)*, reality of the real-space blocks."""
    scale = np.linalg.norm(blocks[0][(0, 0)])
    # real-space reality
    im = max(np.abs(m.imag).max() for cells in blocks.values()
             for m in cells.values())
    # H(q) hermitian and H(-q) = H(q)^* over a random q sample
    rng = np.random.default_rng(0)
    herm = neg = 0.0
    for _ in range(12):
        qp = rng.uniform(-0.5, 0.5, 2)
        qz = float(rng.uniform(-0.5, 0.5))
        h = h_of_q(blocks, qp, qz)
        hm = h_of_q(blocks, -qp, -qz)
        herm = max(herm, np.abs(h - h.conj().T).max())
        neg = max(neg, np.abs(hm - h.conj()).max())
    print(f"  gates: real-space max|Im| = {im:.2e} (|D00| {scale:.3e}); "
          f"max|H(q)-H(q)^H| = {herm:.2e}; max|H(-q)-H(q)*| = {neg:.2e}")
    return {"realspace_max_imag": float(im), "hermiticity": float(herm),
            "q_negation": float(neg), "scale": float(scale)}


def exponents(blocks, direction, nbranch=4, label=""):
    """log-log slope of omega vs |q| along `direction` (frac. coords)."""
    ss = np.geomspace(1e-4, 2e-2, 12)
    freqs = []
    for s in ss:
        qp = np.array(direction[:2]) * s
        qz = direction[2] * s
        w = frequencies_from_dynamical(h_of_q(blocks, qp, qz))
        freqs.append(np.sort(w)[:nbranch] if w.min() < 0 else np.sort(w)[:nbranch])
    freqs = np.array(freqs)
    out = []
    for b in range(nbranch):
        y = np.abs(freqs[:, b])
        m = y > 1e-12
        if m.sum() < 4:
            out.append(float("nan"))
            continue
        sl = np.polyfit(np.log(ss[m]), np.log(y[m]), 1)[0]
        out.append(float(sl))
    return out, freqs


def census(blocks, nk, label):
    """Imaginary modes on the sampled Gamma-centered mesh and on a fine one."""
    def scan(qs, nz):
        worst = 0.0
        n_imag = 0
        for qp in qs:
            for qz in np.linspace(0.0, 0.5, nz):
                w = frequencies_from_dynamical(h_of_q(blocks, qp, qz))
                worst = min(worst, float(w.min()))
                n_imag += int((w < -1e-6).sum())
        return worst, n_imag

    q1 = np.arange(nk) / nk
    sampled = [(a, b) for a in q1 for b in q1]
    w_s, n_s = scan(sampled, 5)
    fine = [(a, b) for a in np.linspace(0, 0.5, 11)
            for b in np.linspace(0, 0.5, 11)]
    w_f, n_f = scan(fine, 9)
    print(f"  imaginary-mode census: sampled {nk}x{nk} mesh -> min omega "
          f"{w_s:+.4f} THz ({n_s} imaginary); fine 11x11 -> {w_f:+.4f} THz "
          f"({n_f} imaginary)")
    return {"sampled_min_thz": w_s, "sampled_n_imag": n_s,
            "fine_min_thz": w_f, "fine_n_imag": n_f}


def main():
    report = {}
    for label, rel, t_axis, nk in SYSTEMS:
        path = ROOT / rel
        if not path.exists():
            print(f"\n=== {label}: MISSING {rel}")
            continue
        print(f"\n=== {label}  [{rel}]  transport axis {t_axis}")
        blocks = load_blocks(path, t_axis)
        r = {"file": rel, "transport_axis": t_axis}
        r["gates"] = gates(blocks, label)

        w_g = frequencies_from_dynamical(h_of_q(blocks, (0.0, 0.0), 0.0))
        w_g = np.sort(w_g)
        print(f"  Gamma spectrum (THz), lowest 8: "
              f"{np.round(w_g[:8], 4).tolist()}")
        r["gamma_lowest"] = w_g[:8].tolist()

        for name, d in (("in-plane (1,0,0)", (1.0, 0.0, 0.0)),
                        ("in-plane (1,1,0)", (1.0, 1.0, 0.0)),
                        ("transport (0,0,1)", (0.0, 0.0, 1.0))):
            al, fr = exponents(blocks, d, label=label)
            print(f"  alpha in omega~q^alpha, {name:18s}: "
                  f"{np.round(al, 3).tolist()}   "
                  f"(omega at |q|=2e-2: {np.round(fr[-1], 4).tolist()})")
            r[f"alpha_{name.split()[0]}_{d}"] = al
        r["census"] = census(blocks, nk, label)
        report[label] = r

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dispersion_audit.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT / 'dispersion_audit.json'}")


if __name__ == "__main__":
    main()
