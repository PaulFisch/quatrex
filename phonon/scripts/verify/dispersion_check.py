#\!/usr/bin/env python
"""Phonon dispersion along the transport axis for a SiNW config.

Checks for genuine imaginary/soft modes (the structural cause of the SCBA
instability) and reports the lowest branches near Gamma and a channel count.
Usage: dispersion_check.py <config.yaml> [transport_axis=2]
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")
import numpy as np
from phonon.finite_analysis.loader import load_system

cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    _REPO / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml"
axis = int(sys.argv[2]) if len(sys.argv) > 2 else 2
bundle = load_system(cfg, validate=False, transport_axis=axis)
ph = bundle.phonon
nat = ph.primitive.masses.shape[0]
print(f"config={cfg.name}  prim atoms={nat}  branches={3*nat}  axis={axis}")

# q-path Gamma -> zone boundary along the transport reciprocal axis
qdir = np.zeros(3); qdir[axis] = 0.5
nq = 41
qs = np.linspace(0, 1, nq)[:, None] * qdir[None, :]
freqs = []
for q in qs:
    ph.run_qpoints([q])
    freqs.append(ph.get_qpoints_dict()["frequencies"][0])  # THz
freqs = np.array(freqs)             # (nq, 3*nat)
fmin = freqs.min()
# phonopy reports imaginary modes as negative frequencies
n_imag_gamma = int(np.sum(freqs[0] < -1e-4))
n_imag_any = int(np.sum(freqs < -1e-4))
print(f"global min frequency = {fmin:.5f} THz")
print(f"imaginary modes (<-1e-4 THz): at Gamma={n_imag_gamma}, anywhere={n_imag_any}")
print("lowest 8 branches at Gamma (THz):",
      np.round(np.sort(freqs[0])[:8], 5))
qb = freqs[nq // 8]   # small but nonzero q
print(f"lowest 8 branches at q={qs[nq//8][axis]:.3f}*ZB (THz):",
      np.round(np.sort(qb)[:8], 5))
print("acoustic slopes near Gamma (branch 0..3 at q1):",
      np.round(np.sort(freqs[1])[:4], 5))
# channel count proxy: number of branches with freq below a few thresholds,
# averaged over q (rough density of propagating states)
for thr in (1.0, 2.0, 5.0, 10.0, 15.0):
    nch = np.mean(np.sum((freqs > 0) & (freqs < thr), axis=1))
    print(f"  mean #branches with 0<f<{thr:>4} THz = {nch:.2f}")
