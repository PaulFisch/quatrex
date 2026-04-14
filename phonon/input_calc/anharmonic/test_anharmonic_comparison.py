"""Compare anharmonic NEGF results with experimental data and literature.

This script:
1. Extracts frequency-dependent phonon scattering rates from Sigma^R
2. Computes phonon lifetimes and compares with first-principles BTE values
3. Runs a temperature sweep to verify T-dependence of scattering
4. Compares ballistic Si/Ge interface with Guo et al. (2020) Fig 5(b)

Uses the 2-atom FCC primitive cell with FC3 from phono3py + symfc.
Requires fc3_prim/fc3.hdf5 (run fc3-reap first).

References:
- Guo et al., PRB 102, 195412 (2020): NEGF + SCBA for phonon transport
- Esfarjani & Chen, PRB 84, 085204 (2011): Si phonon lifetimes from BTE
- Ward & Broido, PRB 81, 085205 (2010): Si anharmonic properties
- Experimental Si kappa ~ 150 W/(mK) at 300K
"""

import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent  # input_calc/
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.anharmonic import anharmonic_transmission
from phonon_inputs.constants import HBAR_SI, THZ_TO_RAD, HBAR_EV, KB_EV


# ---------------------------------------------------------------------------
# 1. Load Si data
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading Si primitive cell (phono3py)...")
phonon, fc3_data = load_primitive_cell(work_dir)

n_atoms = len(phonon.primitive.masses)
masses = phonon.primitive.masses
prim_cell = phonon.primitive.cell
a1_len = np.linalg.norm(prim_cell[0])  # transport lattice vector length
print(f"  Primitive cell: {n_atoms} atoms, |a1| = {a1_len:.4f} A")
print(f"  FC3 blocks: {fc3_data['n_blocks']}")


# ---------------------------------------------------------------------------
# 2. Run SCBA at 300K and extract phonon lifetimes
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Running SCBA at 300K to extract phonon lifetimes...")
print("=" * 60)

t0 = time.time()
result_300 = anharmonic_transmission(
    phonon, fc3_data,
    q_mesh_transverse=(4, 4),
    freq_range_thz=(0.0, 15.0, 101),
    transport_direction="x",
    eta_factor=0.5,
    temperature=300.0,
    max_scba_iter=20,
    scba_tol=0.005,
    mixing=0.3,
    fc3_mode="full",
    verbose=True,
)
t1 = time.time()
print(f"\n300K completed in {t1 - t0:.1f} s")

# Extract phonon scattering rates from self-energy
omega_rad = result_300["omega_rad"]
Sigma_R = result_300["self_energy_retarded"]
n_dof = Sigma_R.shape[1]
freqs_thz = result_300["freqs_thz"]

# Average scattering rate: gamma(w) = -Tr[Im[Sigma^R(w)]] / n_dof
im_sigma_avg = np.array([
    -np.trace(Sigma_R[iw].imag).real / n_dof
    for iw in range(len(omega_rad))
])

# Scattering rate (broadening HWHM in rad/s)
gamma_rad = np.zeros_like(omega_rad)
valid = omega_rad > 1e10
gamma_rad[valid] = im_sigma_avg[valid] / (2.0 * omega_rad[valid])

# Phonon lifetime in ps
tau_ps = np.zeros_like(omega_rad)
tau_ps[gamma_rad > 0] = 1.0 / gamma_rad[gamma_rad > 0] * 1e12

# Mean free path (using group velocity ~ 5000 m/s for acoustic Si)
v_group_approx = 5000.0  # m/s, rough average for Si acoustic modes
mfp_nm = tau_ps * 1e-12 * v_group_approx * 1e9

print("\nExtracted phonon scattering parameters at 300K:")
print(f"{'Freq [THz]':>12} {'|Im Sigma^R| [THz^2]':>22} "
      f"{'gamma [THz]':>13} {'tau [ps]':>10} {'MFP [nm]':>10}")
print("-" * 70)
for iw in range(0, len(freqs_thz), 10):
    if freqs_thz[iw] > 0.5:
        print(f"{freqs_thz[iw]:>12.1f} {im_sigma_avg[iw]:>22.4e} "
              f"{gamma_rad[iw]/THZ_TO_RAD:>13.4f} "
              f"{tau_ps[iw]:>10.1f} {mfp_nm[iw]:>10.1f}")


# ---------------------------------------------------------------------------
# 3. Temperature sweep
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Temperature sweep: 100K to 500K")
print("(Using 1 SCBA iteration = Born approximation for speed)")
print("=" * 60)

temperatures = [100, 200, 300, 400, 500]
temp_results = {}

for T in temperatures:
    print(f"\n--- T = {T} K ---")
    t0 = time.time()
    res = anharmonic_transmission(
        phonon, fc3_data,
        q_mesh_transverse=(4, 4),
        freq_range_thz=(0.0, 15.0, 51),
        transport_direction="x",
        eta_factor=0.5,
        temperature=float(T),
        max_scba_iter=1,
        scba_tol=1e-10,
        mixing=1.0,
        fc3_mode="full",
        verbose=False,
    )
    t1 = time.time()
    temp_results[T] = res

    G_b = res["thermal_conductance_ballistic"]
    G_a = res["thermal_conductance_anharmonic"]
    red = (1 - G_a / G_b) * 100 if G_b > 0 else 0

    Sig = res["self_energy_retarded"]
    w_arr = res["omega_rad"]
    im_sig = np.array([-np.trace(Sig[i].imag).real / Sig.shape[1]
                        for i in range(len(w_arr))])
    avg_scatt = np.mean(im_sig[im_sig > 0]) if np.any(im_sig > 0) else 0

    print(f"  Time: {t1-t0:.1f}s, G_ball={G_b/1e6:.1f}, "
          f"G_anh={G_a/1e6:.1f} MW/(m^2 K), reduction={red:.1f}%")
    print(f"  Avg |Im Sigma^R|: {avg_scatt:.4e}")


# ---------------------------------------------------------------------------
# 4. Compare with literature values
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Comparison with literature")
print("=" * 60)

print("\nExperimental reference values for Si at 300K:")
print("  Bulk thermal conductivity: 148 W/(m K)  [Glassbrenner & Slack, 1964]")
print("  Phonon MFP (dominant):     ~100-300 nm   [Esfarjani & Chen, 2011]")
print("  Acoustic phonon lifetime:")
print("    1 THz:  ~50-200 ps")
print("    5 THz:  ~2-10 ps")
print("    10 THz: ~0.5-2 ps")

print("\nOur results at 300K (4x4 q-mesh, SCBA):")
for f_target in [1.0, 3.0, 5.0, 7.0, 10.0, 13.0]:
    idx = np.argmin(np.abs(freqs_thz - f_target))
    if tau_ps[idx] > 0 and tau_ps[idx] < 1e6:
        print(f"    {freqs_thz[idx]:.1f} THz: tau = {tau_ps[idx]:.1f} ps, "
              f"MFP ~ {mfp_nm[idx]:.0f} nm")
    else:
        print(f"    {freqs_thz[idx]:.1f} THz: tau = inf (no scattering)")

# Single-slab thermal conductivity estimate
d_slab = a1_len * 1e-10  # slab thickness in m
G_300 = result_300["thermal_conductance_anharmonic"]
kappa_eff = G_300 * d_slab
print(f"\n  Single-slab effective kappa: {kappa_eff:.2f} W/(m K)")
print(f"  (Expected: ~148 W/(m K) for bulk, but our single-slab model")
print(f"   at d={a1_len:.2f} A << MFP is firmly ballistic)")

# Temperature scaling check
print("\nTemperature dependence of scattering rate:")
print("  (Classical limit: Sigma ~ T, quantum corrections at low T)")
print(f"  {'T [K]':>6} {'Avg |Im Sigma^R|':>20} {'Relative to 300K':>18} "
      f"{'T/300':>8} {'Reduction [%]':>14}")
print("-" * 70)
ref_scatt = None
for T in temperatures:
    Sig = temp_results[T]["self_energy_retarded"]
    w_arr = temp_results[T]["omega_rad"]
    im_sig = np.array([-np.trace(Sig[i].imag).real / Sig.shape[1]
                        for i in range(len(w_arr))])
    avg_s = np.mean(im_sig[im_sig > 0]) if np.any(im_sig > 0) else 0
    if T == 300:
        ref_scatt = avg_s
    ratio = avg_s / ref_scatt if ref_scatt and ref_scatt > 0 else 0

    G_b = temp_results[T]["thermal_conductance_ballistic"]
    G_a = temp_results[T]["thermal_conductance_anharmonic"]
    red = (1 - G_a / G_b) * 100 if G_b > 0 else 0

    print(f"  {T:>6d} {avg_s:>20.4e} {ratio:>18.3f} "
          f"{T/300:>8.3f} {red:>14.1f}")


# ---------------------------------------------------------------------------
# 5. Load and compare with Guo Si/Ge interface data
# ---------------------------------------------------------------------------
lit_file = work_dir / "examples" / "literature_fig5b.npz"
has_lit = lit_file.exists()
if has_lit:
    lit = np.load(lit_file)
    print(f"\nLoaded Guo et al. Fig 5(b) digitized data:")
    print(f"  Guo:    {len(lit['guo_freq'])} points")
    print(f"  Latour: {len(lit['latour_freq'])} points")
    print(f"  Tian:   {len(lit['tian_freq'])} points")


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 12))

# (a) Transmission at 300K: ballistic vs anharmonic
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(freqs_thz, result_300["spectral_heat_current_ballistic"],
         "b-", lw=1.5, label="Ballistic")
ax1.plot(freqs_thz, result_300["spectral_heat_current"],
         "r--", lw=1.5, label="Anharmonic (SCBA)")
ax1.set_xlabel("Frequency (THz)")
ax1.set_ylabel("Spectral heat current (W)")
ax1.set_title("(a) Bulk Si spectral heat current, 300K")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 16)

# (b) Phonon lifetime vs frequency
ax2 = fig.add_subplot(2, 3, 2)
valid_tau = (tau_ps > 0) & (tau_ps < 1e6) & (freqs_thz > 0.5)
ax2.semilogy(freqs_thz[valid_tau], tau_ps[valid_tau],
             "b-", lw=1.5, label="This work (SCBA)")

ref_freqs = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0]
ref_tau = [100, 30, 15, 4, 2, 0.8, 0.5]
ax2.semilogy(ref_freqs, ref_tau, "ks", ms=8, mfc="none", lw=1.5,
             label="BTE literature (approx.)")

ax2.set_xlabel("Frequency (THz)")
ax2.set_ylabel("Phonon lifetime (ps)")
ax2.set_title("(b) Phonon lifetime at 300K")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, which="both")
ax2.set_xlim(0, 16)
ax2.set_ylim(0.01, 1000)

# (c) Scattering rate (Im Sigma) vs frequency
ax3 = fig.add_subplot(2, 3, 3)
ax3.semilogy(freqs_thz[valid_tau], im_sigma_avg[valid_tau],
             "r-", lw=1.5)
ax3.set_xlabel("Frequency (THz)")
ax3.set_ylabel(r"$-\mathrm{Tr}[\mathrm{Im}\,\Sigma^R]/n_{dof}$ (THz$^2$)")
ax3.set_title("(c) Scattering rate at 300K")
ax3.grid(True, alpha=0.3, which="both")
ax3.set_xlim(0, 16)

# (d) Temperature dependence of transmission
ax4 = fig.add_subplot(2, 3, 4)
colors_T = {100: "C0", 200: "C1", 300: "C2", 400: "C3", 500: "C4"}
for T in temperatures:
    r = temp_results[T]
    ax4.plot(r["freqs_thz"], r["spectral_heat_current"],
             color=colors_T[T], lw=1.2, label=f"{T} K")
ax4.plot(temp_results[300]["freqs_thz"],
         temp_results[300]["spectral_heat_current_ballistic"],
         "k--", lw=1, alpha=0.5, label="Ballistic")
ax4.set_xlabel("Frequency (THz)")
ax4.set_ylabel("Spectral heat current (W)")
ax4.set_title("(d) Temperature dependence")
ax4.legend(fontsize=7, ncol=2)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 16)

# (e) Scattering rate temperature scaling
ax5 = fig.add_subplot(2, 3, 5)
avg_scatt_vs_T = []
for T in temperatures:
    Sig = temp_results[T]["self_energy_retarded"]
    im_sig = np.array([-np.trace(Sig[i].imag).real / Sig.shape[1]
                        for i in range(len(temp_results[T]["omega_rad"]))])
    avg_scatt_vs_T.append(np.mean(im_sig[im_sig > 0]))

avg_scatt_arr = np.array(avg_scatt_vs_T)
T_arr = np.array(temperatures)
norm_scatt = avg_scatt_arr / avg_scatt_arr[temperatures.index(300)]

ax5.plot(T_arr, norm_scatt, "bo-", lw=1.5, ms=8, label="SCBA (Born approx.)")
ax5.plot(T_arr, T_arr / 300.0, "k--", lw=1, alpha=0.7,
         label="Linear (classical)")
ax5.set_xlabel("Temperature (K)")
ax5.set_ylabel(r"$\langle|\mathrm{Im}\,\Sigma^R|\rangle$ / value at 300K")
ax5.set_title("(e) Scattering rate vs temperature")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# (f) Comparison with Guo Fig 5(b)
ax6 = fig.add_subplot(2, 3, 6)
if has_lit:
    if len(lit["guo_freq"]) > 0:
        ax6.plot(lit["guo_freq"], lit["guo_trans"], "b-", lw=1.5,
                 label="Guo et al. (2020)")
    if len(lit["tian_freq"]) > 0:
        ax6.plot(lit["tian_freq"], lit["tian_trans"], "ko", ms=5,
                 mfc="none", label="Tian et al. (2012)")
    if len(lit["latour_freq"]) > 0:
        ax6.plot(lit["latour_freq"], lit["latour_trans"], "m^", ms=5,
                 mfc="none", label="Latour et al. (2017)")
ax6.set_title("(f) Si/Ge interface (Guo Fig 5b)")
ax6.set_xlabel("Frequency (THz)")
ax6.set_ylabel("Transmission")
ax6.legend(fontsize=7)
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, 14)
ax6.set_ylim(0, 1.0)

plt.tight_layout()
fig.savefig(script_dir / "anharmonic_comparison.png", dpi=150)
plt.close("all")
print(f"\nSaved anharmonic_comparison.png")


# ---------------------------------------------------------------------------
# 7. Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

G_b_300 = result_300["thermal_conductance_ballistic"]
G_a_300 = result_300["thermal_conductance_anharmonic"]
red_300 = (1 - G_a_300 / G_b_300) * 100

print(f"\nBulk Si at 300K (4x4 q-mesh, {result_300['n_scba_iterations']} SCBA iters):")
print(f"  G_ballistic:   {G_b_300/1e6:.1f} MW/(m^2 K)")
print(f"  G_anharmonic:  {G_a_300/1e6:.1f} MW/(m^2 K)")
print(f"  Reduction:     {red_300:.1f}%")

print(f"\nPhonon lifetimes (this work vs BTE literature):")
print(f"  {'Freq':>6} {'This work':>12} {'Literature':>12} {'Ratio':>8}")
print(f"  {'[THz]':>6} {'[ps]':>12} {'[ps]':>12}")
print("  " + "-" * 42)
for f_t, t_lit in zip([1.0, 3.0, 5.0, 7.0, 10.0],
                       [100, 15, 4, 2, 0.8]):
    idx = np.argmin(np.abs(freqs_thz - f_t))
    if tau_ps[idx] > 0 and tau_ps[idx] < 1e6:
        ratio = tau_ps[idx] / t_lit
        print(f"  {freqs_thz[idx]:>6.1f} {tau_ps[idx]:>12.1f} "
              f"{t_lit:>12.1f} {ratio:>8.2f}")

print(f"\nTemperature scaling (should be ~linear for T >> Debye/3):")
print(f"  Si Debye temperature: ~645 K")
for i, T in enumerate(temperatures):
    print(f"  {T}K: scattering rate = {norm_scatt[i]:.3f} x (300K value), "
          f"T/300 = {T/300:.3f}")
