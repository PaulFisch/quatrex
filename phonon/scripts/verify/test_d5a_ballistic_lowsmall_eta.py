#\!/usr/bin/env python
"""Push eta small on d5a ballistic + compare to mode-counting channel number."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")
import numpy as np
from phonon.finite_analysis.loader import load_system
from phonon.solver.dense import transmission_finite

cfg = _REPO / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml"
bundle = load_system(cfg, validate=False, transport_axis=2)
ph = bundle.phonon
fc3 = Path(bundle.meta.get("fc3_path", "")).expanduser().resolve()

# mode-counting: mean number of bands present in [w-dw, w+dw] over the BZ
nq = 400
freqs_band = []
for q in np.linspace(-0.5, 0.5, nq):
    ph.run_qpoints([[0, 0, q]])
    freqs_band.append(ph.get_qpoints_dict()["frequencies"][0])
freqs_band = np.array(freqs_band)            # (nq, 63)
def bands_at(w, dw=0.3):
    return np.mean(np.sum((freqs_band > w-dw) & (freqs_band < w+dw), axis=1))

grid = np.array([0.5, 1, 2, 3, 5, 8, 12, 16])
res_cache = {}
for ef in (0.1, 0.03, 0.01):
    r = transmission_finite(ph, fc3_hdf5=str(fc3),
        freq_range_thz=(0.01, 18.0, 81), transport_direction="z",
        temperature=300.0, delta_T=10.0, n_slabs=2,
        eta_factor=ef, vertex_scale=0.0, max_scba_iter=1,
        auto_extend_fmax=False, zero_mode_projection=True, verbose=False)
    res_cache[ef] = (np.asarray(r["freqs_thz"]), np.asarray(r["transmission_ballistic"]),
                     r["thermal_conductance_ballistic"])
f01 = res_cache[0.1][0]
print("freq  bands_at   ballT(.1)  ballT(.03) ballT(.01)")
for w in grid:
    i = np.argmin(np.abs(f01 - w))
    row = [f"{w:>4}", f"{bands_at(w):>8.2f}"]
    for ef in (0.1, 0.03, 0.01):
        row.append(f"{res_cache[ef][1][i]:>10.3f}")
    print("  ".join(row))
for ef in (0.1, 0.03, 0.01):
    fa, Ta, Ga = res_cache[ef]
    print(f"eta_f={ef}: maxT={Ta.max():.3f}  G_ball={Ga:.4e}")
