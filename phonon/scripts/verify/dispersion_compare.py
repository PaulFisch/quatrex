"""Bulk-Si phonon dispersion: DFPT (ph.x linear response) vs FD (phono3py+symfc).

DFPT frequencies are read directly from ph.out (computed natively at the 2x2x2
q-mesh: Gamma and the L/X zone points) -- this avoids any FC2 supercell-ordering
mismatch. FD bands are computed with phonopy from the reaps fc2.hdf5. Produces a
high-symmetry frequency table + a band plot (FD lines, DFPT markers).
"""
import re
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, "/usr/scratch/mont-fort11/pfischill/quatrex/phonon")
from finite_analysis.plot_style import set_publication_style  # noqa: E402
set_publication_style()
import phonopy  # noqa: E402
from phonopy.structure.atoms import PhonopyAtoms  # noqa: E402

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
OUTS = [_REPO / "phonon/scripts/out/fc_compare",
        _REPO / "document/fig/transport_sweeps"]
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)

# ---- parse DFPT ph.out: {q-tuple: sorted freqs (THz)} -------------------
ph_out = (_REPO / "phonon/configs/si_primitive/dfpt/ph.out").read_text().splitlines()
dfpt = {}
qcur = None
for ln in ph_out:
    m = re.search(r"q = \(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)", ln)
    if m:
        qcur = tuple(round(float(x), 3) for x in m.groups())
        dfpt.setdefault(qcur, [])
    m2 = re.search(r"freq \(.*\) =\s*([-\d.]+)\s*\[THz\]", ln)
    if m2 and qcur is not None:
        dfpt[qcur].append(float(m2.group(1)))
dfpt = {q: sorted(set(f)) if False else sorted(f) for q, f in dfpt.items() if f}
gamma_q = (0.0, 0.0, 0.0)
# L = all-equal magnitude (the (+-0.354,+-0.354,+-0.354) block)
L_q = next((q for q in dfpt if len({abs(x) for x in q}) == 1 and abs(q[0]) > 0.1), None)
dfpt_gamma = np.array(dfpt.get(gamma_q, []))
dfpt_L = np.array(dfpt.get(L_q, [])) if L_q else np.array([])

# ---- FD bands via phonopy ----------------------------------------------
prim = PhonopyAtoms(
    symbols=["Si", "Si"],
    cell=[[0.0, 2.7331, 2.7331], [2.7331, 0.0, 2.7331], [2.7331, 2.7331, 0.0]],
    scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
with h5py.File(_REPO / "phonon/reaps/si_primitive_work/fc2.hdf5", "r") as f:
    fd_fc2 = np.array(f["force_constants"])
ph = phonopy.Phonopy(prim, supercell_matrix=np.diag([2, 2, 2]), primitive_matrix="auto")
ph.force_constants = fd_fc2

from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
band_path = [[[0, 0, 0], [0.5, 0, 0.5], [0.375, 0.375, 0.75]],
             [[0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
labels = ["$\\Gamma$", "X", "K", "$\\Gamma$", "L"]
qpts, conn = get_band_qpoints_and_path_connections(band_path, npoints=61)
ph.run_band_structure(qpts, path_connections=conn, labels=labels)
bs = ph.get_band_structure_dict()

# FD freqs at Gamma and L
ph.run_qpoints([[0, 0, 0], [0.5, 0.5, 0.5]])
fdq = ph.get_qpoints_dict()["frequencies"]
fd_gamma, fd_L = np.sort(fdq[0]), np.sort(fdq[1])

# ---- table --------------------------------------------------------------
print("=== Bulk Si phonon frequencies (THz): DFPT (ph.x) vs FD (symfc) ===")
print(f"Gamma optical  DFPT: {np.round(dfpt_gamma[dfpt_gamma>1],2)}  "
      f"FD: {np.round(fd_gamma[fd_gamma>1],2)}")
print(f"L point        DFPT: {np.round(np.sort(dfpt_L),2)}")
print(f"               FD:   {np.round(fd_L,2)}")
if dfpt_gamma.size and fd_gamma.size:
    g_dfpt = dfpt_gamma[dfpt_gamma > 1].mean()
    g_fd = fd_gamma[fd_gamma > 1].mean()
    print(f"Gamma-optical mean: DFPT {g_dfpt:.2f}  FD {g_fd:.2f}  "
          f"diff {100*abs(g_dfpt-g_fd)/g_fd:.1f}%")

# ---- plot: FD bands + DFPT markers at Gamma and L ----------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for seg_d, seg_f in zip(bs["distances"], bs["frequencies"]):
    for b in range(seg_f.shape[1]):
        ax.plot(seg_d, seg_f[:, b], color="#1f77b4", lw=1.0,
                label="FD (symfc)" if (b == 0 and seg_d is bs["distances"][0]) else None)
xG0 = bs["distances"][0][0]
xL = bs["distances"][-1][-1]
ax.plot([xG0] * len(dfpt_gamma), dfpt_gamma, "o", color="#d62728", ms=5,
        label="DFPT (ph.x)")
ax.plot([xL] * len(dfpt_L), dfpt_L, "o", color="#d62728", ms=5)
ax.set_ylabel("frequency (THz)")
ax.set_xticks([xG0, xL])
ax.set_xticklabels(["$\\Gamma$", "L"])
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
for o in OUTS:
    fig.savefig(o / "fc_method_dispersion_si.pdf")
plt.close(fig)
print(f"[done] wrote fc_method_dispersion_si.pdf")
