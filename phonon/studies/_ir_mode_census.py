"""A-priori IR-resolvability census for the CNT device (theory eq:grid_resolution).

Ballistic device spectral function on a fine grid (eta = 1e-4,
d_omega = 1e-3 THz) for L in {2, 4, 10}: does the infrared structure
(slow twist/flexural standing-wave features) sharpen below the production
grid spacing (55/180 = 0.3056 THz) as the device grows?

Memory-light by construction: explicit chunked loop over positive
frequencies (<= 64 x N_D x N_D dense at a time), Sancho-Rubio per
frequency on the 36-DOF bulk blocks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks, peak_widths

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _toy_grid_study as T  # noqa: E402  (path bootstrap for solver.*)
from solver.leads import sancho_rubio  # noqa: E402

HERE = Path(__file__).resolve().parent
CEN = HERE / "out/anderson_test/census"
DW_PROD = 55.0 / 180.0        # production grid spacing (ne = 181)
DW_FINE = 55.0 / 360.0        # ne = 361 candidate
ETA = 1e-4
W_LO, W_HI, DW = 0.02, 3.5, 1e-3
W_PEAK_MIN = 0.15             # exclude the acoustic omega->0 divergence
CHUNK = 64


def device_dos(h00, h01, L, freqs):
    n = h00.shape[0]
    N_D = L * n
    h_d = np.zeros((N_D, N_D), complex)
    for i in range(L):
        h_d[i * n:(i + 1) * n, i * n:(i + 1) * n] = h00
        if i + 1 < L:
            h_d[i * n:(i + 1) * n, (i + 1) * n:(i + 2) * n] = h01
            h_d[(i + 1) * n:(i + 2) * n, i * n:(i + 1) * n] = h01.conj().T
    dos = np.empty(len(freqs))
    eye = np.eye(N_D)
    for s in range(0, len(freqs), CHUNK):
        wk = freqs[s:s + CHUNK]
        z2 = (wk + 1j * ETA) ** 2
        A = np.empty((len(wk), N_D, N_D), complex)
        for i, z in enumerate(z2):
            g_L = sancho_rubio(z, h00, h01)
            g_R = sancho_rubio(z, h00, h01.conj().T)
            A[i] = z * eye - h_d
            A[i, :n, :n] -= h01.conj().T @ g_L @ h01
            A[i, -n:, -n:] -= h01 @ g_R @ h01.conj().T
        G = np.linalg.inv(A)
        dos[s:s + CHUNK] = (-2.0 * wk / np.pi
                            * np.trace(G, axis1=1, axis2=2).imag) / L
    return dos


def main() -> int:
    d = loadmat(CEN / "dyn_L4.mat")
    h00 = np.ascontiguousarray(d["[0, 0, 0]"])
    h01 = np.ascontiguousarray(d["[0, 0, 1]"])
    w2 = np.linalg.eigvalsh(h00 + h01 + h01.conj().T).real
    print(f"bulk Gamma-point omega^2 range: {w2.min():.3f} .. {w2.max():.1f}"
          f"  -> omega_max ~ {np.sqrt(max(w2.max(), 0)):.2f}", flush=True)

    freqs = np.arange(W_LO, W_HI, DW)
    m = freqs >= W_PEAK_MIN
    results = {}
    for L in (2, 4, 10):
        dos = device_dos(h00, h01, L, freqs)
        band = dos[m]
        pk, _ = find_peaks(band, prominence=0.02 * band.max())
        w_res = peak_widths(band, pk, rel_height=0.5)
        fwhm = w_res[0] * DW
        pos = freqs[m][pk]
        sub_prod = fwhm < DW_PROD
        sub_fine = fwhm < DW_FINE
        results[L] = dict(freqs=freqs.tolist(), dos=dos.tolist(),
                          peak_pos=pos.tolist(), peak_fwhm=fwhm.tolist())
        print(f"L={L:2d}: {len(pk)} IR peaks in ({W_PEAK_MIN}, {W_HI}) THz; "
              f"{int(sub_prod.sum())} narrower than prod dw={DW_PROD:.4f}; "
              f"{int(sub_fine.sum())} narrower than ne361 dw={DW_FINE:.4f}; "
              f"min FWHM={fwhm.min() if fwhm.size else np.nan:.4f} "
              f"median={np.median(fwhm) if fwhm.size else np.nan:.4f}",
              flush=True)
        with np.printoptions(precision=3, suppress=True):
            print(f"   sub-prod peaks (THz): {pos[sub_prod]}", flush=True)
            print(f"   their FWHM (THz):     {fwhm[sub_prod]}", flush=True)
    (CEN / "census.json").write_text(json.dumps(
        {str(k): v for k, v in results.items()},
        default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"saved {CEN}/census.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
