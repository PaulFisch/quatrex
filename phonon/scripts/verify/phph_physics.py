"""Mode-resolved three-phonon physics for bulk Si from the phono3py RTA result.

Reads kappa-m191919.hdf5 (FD FC3) and extracts the genuinely interesting
phonon-phonon observables: mode lifetimes tau(omega), the thermal-conductivity
accumulation vs phonon mean free path (which length scales carry the heat), and
the mode-resolved kappa spectrum (which modes carry it). Writes figures + a
summary of the dominant carriers.
"""
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
OUTS = [_REPO / "phonon/scripts/out/phph_physics",
        _REPO / "document/fig/transport_sweeps"]
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)

f = h5py.File(_REPO / "phonon/reaps/si_primitive_work/kappa-m191919.hdf5", "r")
T = np.array(f["temperature"])
it = int(np.argmin(np.abs(T - 300)))          # 300 K index
freq = np.array(f["frequency"])               # (nq, nb) THz
gamma = np.array(f["gamma"])[it]              # (nq, nb) THz (HWHM)
vg = np.array(f["group_velocity"])           # (nq, nb, 3) THz*Angstrom
mode_kappa = np.array(f["mode_kappa"])[it]   # (nq, nb, 6) W/mK
weight = np.array(f["weight"])               # (nq,)
kappa = np.array(f["kappa"])[it]             # (6,)

# lifetime tau [ps] = 1 / (2 * gamma[THz] * 2pi); MFP = |v_g| * tau
with np.errstate(divide="ignore"):
    tau = 1.0 / (2.0 * 2.0 * np.pi * gamma)   # ps
vg_mag = np.linalg.norm(vg, axis=2) * 100.0   # THz*A -> m/s
mfp_nm = vg_mag * (tau * 1e-12) * 1e9         # m/s * s -> m -> nm
mk = mode_kappa[:, :, 0]                       # xx component, W/mK

# phono3py mode_kappa already carries the q-multiplicity: kappa = sum_IBZ(mk)/N_mesh
valid = np.isfinite(tau) & (gamma > 1e-4) & (freq > 1e-3)
fr_v, tau_v, mfp_v, mk_v = freq[valid], tau[valid], mfp_nm[valid], mk[valid]

# --- kappa accumulation vs MFP ---
order = np.argsort(mfp_v)
mfp_sorted = mfp_v[order]
kcum = np.cumsum(mk_v[order])
kcum /= kcum[-1]
mfp50 = mfp_sorted[np.searchsorted(kcum, 0.5)]
mfp90 = mfp_sorted[np.searchsorted(kcum, 0.9)]

print(f"kappa(300K) = {kappa[0]:.1f} W/mK (sum_IBZ mode_kappa / N_mesh = "
      f"{mk.sum()/weight.sum():.1f})")
print(f"median lifetime = {np.median(tau_v):.2f} ps; max = {tau_v.max():.1f} ps")
print(f"MFP carrying 50% / 90% of kappa: {mfp50:.1f} / {mfp90:.1f} nm")
print(f"fraction of kappa from acoustic (f<8 THz): "
      f"{mk[freq<8].sum()/mk.sum():.2f}")
# nanowire boundary-scattering connection: kappa fraction below typical scales
for d in (1.0, 10.0, 100.0):
    frac = kcum[np.searchsorted(mfp_sorted, d) - 1] if d > mfp_sorted[0] else 0.0
    print(f"  kappa fraction from MFP < {d:6.0f} nm: {frac*100:5.1f}%  "
          f"(=> a wire of that diameter retains ~this much bulk kappa)")

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
ax[0].scatter(fr_v, tau_v, s=4, alpha=0.3, color="#1f77b4")
ax[0].set_yscale("log")
ax[0].set_xlabel("frequency (THz)"); ax[0].set_ylabel(r"lifetime $\tau$ (ps)")
ax[0].set_title("Phonon lifetimes, bulk Si 300 K")
# guide ~ omega^-2
ff = np.linspace(2, 15, 50)
ax[0].plot(ff, 2e2 * ff**-2.0, "k--", lw=0.8, alpha=0.6, label=r"$\propto\omega^{-2}$")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

ax[1].semilogx(mfp_sorted, kcum, color="#d62728")
ax[1].axhline(0.5, color="k", lw=0.6, ls=":"); ax[1].axhline(0.9, color="k", lw=0.6, ls=":")
ax[1].set_xlabel("phonon mean free path (nm)")
ax[1].set_ylabel(r"cumulative $\kappa/\kappa_\mathrm{tot}$")
ax[1].set_title(f"$\\kappa$ accumulation (50%@{mfp50:.0f}nm, 90%@{mfp90:.0f}nm)")
ax[1].grid(alpha=0.3, which="both")

ax[2].scatter(fr_v, mk_v, s=4, alpha=0.3, color="#2ca02c")
ax[2].set_xlabel("frequency (THz)"); ax[2].set_ylabel(r"mode $\kappa$ (W m$^{-1}$K$^{-1}$)")
ax[2].set_title("Mode-resolved $\\kappa$ spectrum")
ax[2].grid(alpha=0.3)
fig.tight_layout()
for o in OUTS:
    fig.savefig(o / "phph_physics_si.pdf")
plt.close(fig)
print("[done] wrote phph_physics_si.pdf")
