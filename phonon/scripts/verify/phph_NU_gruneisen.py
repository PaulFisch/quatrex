"""Normal vs Umklapp decomposition and mode Grueneisen for bulk Si (FD FC3).

Recomputes the RTA thermal conductivity with the N/U split and the FC3 mode
Grueneisen parameters -- the two standard measures of which scattering processes
limit kappa and how strongly anharmonic each mode is. Writes a figure + numbers.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, "/usr/scratch/mont-fort11/pfischill/quatrex/phonon")
from finite_analysis.plot_style import set_publication_style  # noqa: E402
set_publication_style()
import phono3py  # noqa: E402

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
work = _REPO / "phonon/reaps/si_primitive_work"
OUTS = [_REPO / "phonon/scripts/out/phph_physics",
        _REPO / "document/fig/transport_sweeps"]
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)

ph3 = phono3py.load(str(work / "phono3py.yaml"),
                    fc3_filename=str(work / "fc3.hdf5"),
                    fc2_filename=str(work / "fc2.hdf5"), log_level=0)
mesh = 13
ph3.mesh_numbers = [mesh, mesh, mesh]
ph3.init_phph_interaction()
ph3.run_thermal_conductivity(temperatures=[300], is_N_U=True,
                             boundary_mfp=1e6, write_kappa=False)
tc = ph3.thermal_conductivity
kappa = float(np.array(tc.kappa)[0, 0, 0])   # xx component, W/mK
gN = np.array(tc._gamma_N)[0, 0]   # (n_sigma, n_temp, nq, nb) -> (nq, nb)
gU = np.array(tc._gamma_U)[0, 0]
freq = np.array(tc.frequencies)    # (nq, nb)
g_tot = gN + gU
m = g_tot > 1e-6
u_frac = gU[m].sum() / g_tot[m].sum()
print(f"kappa(300K, mesh {mesh}^3) = {kappa:.1f} W/mK")
print(f"Umklapp fraction of total scattering = {u_frac:.2f} "
      f"(Normal = {1-u_frac:.2f})")

# --- mode Grueneisen from FC3 (on a q-mesh) ---
from phono3py.phonon3.gruneisen import Gruneisen
gr = Gruneisen(ph3.fc2, ph3.fc3, ph3.phonon_supercell, ph3.phonon_primitive)
gr.set_sampling_mesh([11, 11, 11])
gr.run()
graw = np.array(gr.get_gruneisen_parameters())   # (nq, nb, 3, 3) tensor
fr_g = np.ravel(np.array(gr.frequencies))
if graw.size == fr_g.size * 9:
    gam = np.trace(graw.reshape(-1, 3, 3), axis1=1, axis2=2) / 3.0   # scalar mode g
else:
    gam = np.ravel(graw)
ok = np.isfinite(gam) & np.isfinite(fr_g) & (fr_g > 1e-3)
gam, fr_g = gam[ok], fr_g[ok]
print(f"mode Grueneisen: mean|g|={np.nanmean(np.abs(gam)):.2f}, "
      f"max|g|={np.nanmax(np.abs(gam)):.2f}")

fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.0))
ax[0].scatter(freq[m], (gU/g_tot)[m], s=5, alpha=0.3, color="#d62728")
ax[0].set_xlabel("frequency (THz)")
ax[0].set_ylabel("Umklapp fraction of scattering")
ax[0].set_ylim(-0.05, 1.05); ax[0].grid(alpha=0.3)
ax[1].scatter(fr_g, gam, s=8, alpha=0.5, color="#1f77b4")
ax[1].set_xlabel("frequency (THz)")
ax[1].set_ylabel("mode Grueneisen $\\gamma$")
ax[1].grid(alpha=0.3)
fig.tight_layout()
for o in OUTS:
    fig.savefig(o / "phph_NU_gruneisen_si.pdf")
plt.close(fig)
print("[done] wrote phph_NU_gruneisen_si.pdf")
