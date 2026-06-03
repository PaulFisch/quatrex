"""I/O for the spectral-function / band-renormalisation output.

Saves the (q, omega) spectral function, the decomposed band sets and the
per-branch quasiparticle frequencies / linewidths to a plotting-friendly
container (``.npz`` or HDF5), and emits a small matplotlib reference script
(heat-map of ``A(q, omega)`` + overlaid quasiparticle lines + decomposition).
"""

from __future__ import annotations

import numpy as np


def save_spectral(
    path, *, q_distance, omega_grid_thz, A=None,
    tick_positions=None, tick_labels=None,
    bands=None, linewidths=None, **extra,
):
    """Write the spectral / band-decomposition data to ``.npz`` or ``.h5``.

    Parameters
    ----------
    path : str
        Output path; ``.h5`` / ``.hdf5`` -> HDF5, else ``.npz``.
    q_distance : (nq,) array
        Cumulative distance along the q-path (the heat-map x-axis).
    omega_grid_thz : (nw,) array
        Frequency axis [THz].
    A : (nq, nw) array, optional
        Spectral function heat-map.
    tick_positions, tick_labels : optional
        High-symmetry tick locations (in ``q_distance`` units) and labels.
    bands : dict[str, (nq, N)], optional
        Decomposition band sets ("bare", "loop", "loop_tadpole", ...).
    linewidths : (nq, N), optional
        Per-branch FWHM [THz].
    **extra
        Any further named arrays to store (e.g. ``Omega_n``).
    """
    data = {"q_distance": np.asarray(q_distance),
            "omega_grid_thz": np.asarray(omega_grid_thz)}
    if A is not None:
        data["A"] = np.asarray(A)
    if tick_positions is not None:
        data["tick_positions"] = np.asarray(tick_positions)
    if tick_labels is not None:
        data["tick_labels"] = np.asarray(tick_labels, dtype=object)
    if bands is not None:
        for name, arr in bands.items():
            data[f"band_{name}"] = np.asarray(arr)
    if linewidths is not None:
        data["linewidths"] = np.asarray(linewidths)
    for k, v in extra.items():
        data[k] = np.asarray(v)

    path = str(path)
    if path.endswith((".h5", ".hdf5")):
        import h5py
        with h5py.File(path, "w") as f:
            for k, v in data.items():
                if v.dtype == object:
                    f.create_dataset(k, data=[s.encode() for s in v])
                else:
                    f.create_dataset(k, data=v)
    else:
        np.savez(path, **data)
    return path


#: Reference matplotlib script: heat-map of A(q, omega) with the on-shell
#: quasiparticle line overlaid and the decomposition bands. Write it next to the
#: data with :func:`write_reference_plot_script` and run ``python plot_*.py``.
REFERENCE_PLOT_SCRIPT = r'''#!/usr/bin/env python
"""Reference plot for the phonon spectral function + band renormalisation.

Usage:  python plot_spectral.py spectral.npz [out.pdf]
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

d = np.load(sys.argv[1], allow_pickle=True)
out = sys.argv[2] if len(sys.argv) > 2 else "spectral.pdf"

q = d["q_distance"]
w = d["omega_grid_thz"]
fig, ax = plt.subplots(figsize=(7, 5))

if "A" in d.files:
    ax.pcolormesh(q, w, d["A"].T, shading="auto", cmap="magma")

styles = {"bare": ("--", "w", "harmonic"),
          "loop": ("-", "tab:cyan", "+ loop"),
          "loop_tadpole": ("-", "tab:green", "+ loop + tadpole"),
          "tadpole": ("-", "tab:orange", "+ tadpole")}
for name, (ls, c, lab) in styles.items():
    key = f"band_{name}"
    if key in d.files:
        b = d[key]
        for n in range(b.shape[1]):
            ax.plot(q, b[:, n], ls, color=c, lw=0.8,
                    label=lab if n == 0 else None)

if "tick_positions" in d.files:
    ax.set_xticks(d["tick_positions"])
    if "tick_labels" in d.files:
        ax.set_xticklabels([s for s in d["tick_labels"]])
    for x in d["tick_positions"]:
        ax.axvline(x, color="0.5", lw=0.5)

ax.set_ylabel("frequency (THz)")
ax.set_xlabel("wave vector")
ax.set_ylim(bottom=0)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=200)
print("wrote", out)
'''


def write_reference_plot_script(path):
    """Write :data:`REFERENCE_PLOT_SCRIPT` to ``path``."""
    with open(path, "w") as f:
        f.write(REFERENCE_PLOT_SCRIPT)
    return path
