"""(3,3) CNT mode-resolved phonon-phonon physics via phono3py (hiPhive FC3).

The CNT is periodic along z (vacuum in x,y), so kappa is computed on a 1D
[1,1,N] q-mesh. Reports kappa_zz (box-normalised) and the mode lifetimes
tau(omega) -- including the fate of the 0.026 THz twist mode.
"""
from pathlib import Path

import numpy as np
import h5py
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import phono3py  # noqa: E402
from phonopy.structure.atoms import PhonopyAtoms  # noqa: E402

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
cfg = yaml.safe_load(open(_REPO / "phonon/configs/cnt/cnt33_vasp.yaml"))
s = cfg["structure"]
prim = PhonopyAtoms(symbols=s["symbols"], cell=s["lattice"],
                    scaled_positions=s["scaled_positions"])
with h5py.File(_REPO / "phonon/configs/cnt/fc3_hiphive_cnt33_vasp/fc3.hdf5", "r") as f:
    fc2 = np.array(f["fc2"]); fc3 = np.array(f["fc3"])

OUTS = [_REPO / "phonon/scripts/out/phph_physics",
        _REPO / "document/fig/transport_sweeps"]
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)

ph3 = phono3py.Phono3py(prim, supercell_matrix=np.diag([1, 1, 3]),
                        primitive_matrix="auto", log_level=0)
ph3.fc2 = fc2
ph3.fc3 = fc3
ph3.mesh_numbers = [1, 1, 48]        # 1D mesh along the tube
ph3.init_phph_interaction()
ph3.run_thermal_conductivity(temperatures=[300], write_kappa=False)
tc = ph3.thermal_conductivity
kzz = float(np.array(tc.kappa).reshape(-1, 6)[0, 2])     # zz (tube axis)
freq = np.array(tc.frequencies)              # (nq, nb)
gamma = np.array(tc.gamma).reshape((-1,) + freq.shape)[0]  # -> (nq, nb)
m = (gamma > 1e-6) & (freq > 1e-3)
tau = 1.0 / (2 * 2 * np.pi * gamma[m])       # ps
fr = freq[m]
# twist branch: lowest non-acoustic near Gamma (~0.026 THz) -> find lifetime of
# the lowest finite-gamma modes below 1 THz
low = fr < 1.0
print(f"CNT kappa_zz(300K, box-normalised) = {kzz:.1f} W/mK")
print(f"  (box 14x14 A; tube cross-section ~ a few A^2 -> intrinsic kappa ~1-2 orders higher)")
print(f"median lifetime = {np.median(tau):.1f} ps; low-freq(<1THz) median = "
      f"{np.median(tau[low]) if low.any() else float('nan'):.1f} ps; "
      f"max = {tau.max():.0f} ps")

fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.scatter(fr, tau, s=6, alpha=0.4, color="#2ca02c")
ax.set_yscale("log")
ax.set_xlabel("frequency (THz)"); ax.set_ylabel(r"lifetime $\tau$ (ps)")
ax.set_title(f"(3,3) CNT phonon lifetimes, 300 K ($\\kappa_{{zz}}$={kzz:.0f} W/mK, box)")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
for o in OUTS:
    fig.savefig(o / "cnt_lifetimes.pdf")
plt.close(fig)
print("[done] wrote cnt_lifetimes.pdf")
