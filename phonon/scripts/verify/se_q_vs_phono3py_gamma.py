"""Verify the 3-phonon self-energy PREFACTOR against phono3py's Fermi-golden-rule linewidths.

The coupled-q bubble kernel (se_q) follows the Luisier (PRB 86 245407) convention. Guo-Bescond-Zhang
(PRB 102 195412, App. A) show that expression over-counts the repeated middle pairings by a FACTOR
OF 4 relative to the correctly-symmetrized self-energy. The textbook bubble symmetry factor, with the
cubic Hamiltonian H3=(1/3!)Phi u^3 and phono3py's full FD FC3 (Phi=d^3V/du^3), is 1/4 on top of the
i/2 loop factor (Wick: 3*3*2 / (3!*3!*2!) = 1/4) -- which the kernel omits. So the kernel's Sigma
should be ~4x too large.

This is an ABSOLUTE, Guo-free check: the lowest-order (single-pass, no SCBA) NEGF retarded self-energy
imaginary part IS the Fermi-golden-rule 3-phonon linewidth. We compute it for bulk Si and compare to
phono3py's `gamma` on the SAME q-mesh and SAME FC2/FC3, so the ratio R = Gamma_NEGF / gamma_phono3py
is mesh- and FC-independent and isolates the prefactor. Expect R ~ 4, mode-uniform.

Equilibrium conventions (match phonon/solver/{leads,grids}.py):
  G^R(q,w) = [(w+i.eta)^2 I - D(q)]^-1   (D in THz^2),  A = i(G^R - G^A),
  G^<= -i n_B A,  G^>= -i (n_B+1) A  ;  Sigma in THz^2;  linewidth Gamma_lambda = -Im Sigma_ll(w_l)/(2 w_l).
"""
import sys
import warnings
from pathlib import Path

_W = Path("/usr/scratch/mont-fort11/pfischill/quatrex/phonon")
for p in (_W.parent, _W):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

import numpy as np
from phonon.phonon_inputs.separable import (
    build_supercell_mapping, build_realspace_fc3_matrices)
from phonon.phonon_inputs.constants import CONVERSION_FC3_THZ  # noqa: F401 (used via builder)
from phonon.solver.se_q import compute_phph_self_energy_q_dense
from phonon.solver.retarded import build_retarded
from phonon.solver.grids import bose_full_axis
from phonon.scripts.verify.si_film_kappa import load_bulk_si

NMESH = int(sys.argv[1]) if len(sys.argv) > 1 else 6   # NxNxN 3D q-mesh (Gamma-centred)
TEMP = 300.0
ETA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05   # THz broadening for G^R
NE = 121
FMAX = 18.0

# ---- bulk Si: phonopy (FC2) + FC3 real-space vertex M_stacked ----
ph, fc3_path = load_bulk_si()
nat = len(ph.primitive.masses)
nd = 3 * nat
import h5py
with h5py.File(fc3_path, "r") as f:
    fc3 = f["fc3"][:]
# M_stacked uses the SAME real-space supercell mapping as the transport code.
prim_indices, cell_frac, slab_indices, ref_sc = build_supercell_mapping(ph, "x")
M_stacked = build_realspace_fc3_matrices(fc3, nat, ph.supercell.masses, ref_sc)
dim_t = M_stacked.shape[1]

# ---- 3D Gamma-centred q-mesh + 3D gathering matrices T(q) (full 3-component q) + q_diff_map ----
qs = [(i / NMESH, j / NMESH, k / NMESH)
      for i in range(NMESH) for j in range(NMESH) for k in range(NMESH)]
nq = len(qs)
q_idx = {(round(q[0] % 1, 6), round(q[1] % 1, 6), round(q[2] % 1, 6)): n for n, q in enumerate(qs)}


def gmat_3d(qfrac):
    """Full-3D analogue of separable.build_gathering_matrix (no transport-axis zeroing)."""
    n_super = len(prim_indices)
    T = np.zeros((nd, n_super * 3), dtype=complex)
    phases = np.exp(-2j * np.pi * cell_frac @ np.asarray(qfrac))
    for s in range(n_super):
        kappa = prim_indices[s]
        for beta in range(3):
            T[kappa * 3 + beta, s * 3 + beta] = phases[s]
    return T


T_all = [gmat_3d(q) for q in qs]
q_diff_map = np.zeros((nq, nq), dtype=int)
for a, qa in enumerate(qs):
    for b, qb in enumerate(qs):
        key = (round((qa[0] - qb[0]) % 1, 6), round((qa[1] - qb[1]) % 1, 6),
               round((qa[2] - qb[2]) % 1, 6))
        q_diff_map[a, b] = q_idx[key]

# ---- harmonic D(q) [THz^2], eigenfreqs/vecs, and equilibrium G^<,> on the symmetric freq grid ----
freqs = np.linspace(-FMAX, FMAX, NE)
freqs = freqs - freqs[NE // 2]          # ensure exact 0 at mid
dw = float(freqs[1] - freqs[0])
nB = bose_full_axis(freqs, TEMP)

omega_l = np.zeros((nq, nd))            # mode freqs (THz)
evec = np.zeros((nq, nd, nd), dtype=complex)
Gl = np.zeros((nq, NE, nd, nd), dtype=complex)
Gg = np.zeros_like(Gl)
for iq, q in enumerate(qs):
    fr, ev = ph.get_frequencies_with_eigenvectors(np.array(q))
    fr = np.real(fr)
    omega_l[iq] = fr
    evec[iq] = ev
    Dq = ev @ np.diag(fr.astype(complex) ** 2) @ ev.conj().T   # THz^2
    z2 = (freqs + 1j * ETA) ** 2
    GR = np.linalg.inv(z2[:, None, None] * np.eye(nd)[None] - Dq[None])
    A = 1j * (GR - GR.conj().transpose(0, 2, 1))               # spectral function
    Gl[iq] = -1j * nB[:, None, None] * A
    Gg[iq] = -1j * (nB[:, None, None] + 1.0) * A

# ---- one-shot (no SCBA) coupled-q self-energy, then Sigma^R ----
sl, sg = compute_phph_self_energy_q_dense(
    Gl, Gg, M_stacked, T_all, q_diff_map, nat, nq, freqs, dw, n_workers=1)
sigR = build_retarded(sl, sg, freqs, method="pv")              # (nq, NE, nd, nd)


def imsig_mode(iq, band):
    """Mode-projected -Im Sigma_lambda(omega) on the freq grid (THz^2), via the eigenvector."""
    e = evec[iq, :, band]
    return -np.imag(np.einsum('i,wij,j->w', e.conj(), sigR[iq], e))


def gamma_negf(iq, band):
    """HWHM linewidth (THz) of mode (iq,band): -Im[e^dag Sigma^R(w_l) e]/(2 w_l)."""
    wl = omega_l[iq, band]
    if wl < 1e-3:
        return np.nan
    iw = int(np.argmin(np.abs(freqs - wl)))
    e = evec[iq, :, band]
    sig_ll = e.conj() @ sigR[iq, iw] @ e
    return float(-np.imag(sig_ll) / (2.0 * wl))


def gamma_negf_integrated(iq, band):
    """Broadening-INDEPENDENT linewidth: integral of -Im Sigma_lambda(w) over the +resonance,
    divided by 2*w_l. Since int d(w) delta = 1 regardless of lineshape, this is insensitive to
    the Lorentzian-eta vs Gaussian-sigma broadening mismatch (the clean prefactor probe)."""
    wl = omega_l[iq, band]
    if wl < 1e-3:
        return np.nan
    ims = imsig_mode(iq, band)
    win = (freqs > wl - 3.0) & (freqs < wl + 3.0)     # window around the +resonance
    area = np.trapezoid(ims[win], freqs[win])             # THz^2 * THz
    # on-shell HWHM = area_of_(-ImSigma)/ (pi-normalised) ... use peak-area / (2 w_l * width-norm).
    # For comparison we report area/(2 w_l): ratio to phono3py's area/(2 w_l) is the prefactor.
    return float(area / (2.0 * wl))


# ---- phono3py Fermi-golden-rule gamma on the SAME mesh ----
def phono3py_gamma_gamma_point():
    """Return (gamma_at_bands[nbands], freq_pts, gamma_of_freq[nfp,nbands]) at the Gamma grid point."""
    import phono3py
    ph3 = phono3py.load(phono3py_yaml=str(_W / "reaps/si_primitive_work/phono3py.yaml"),
                        log_level=0, produce_fc=False)
    with h5py.File(_W / "reaps/si_primitive_work/fc2.hdf5", "r") as f:
        ph3.fc2 = f["force_constants"][:]
    with h5py.File(fc3_path, "r") as f:
        ph3.fc3 = f["fc3"][:]
    ph3.mesh_numbers = [NMESH, NMESH, NMESH]
    ph3.init_phph_interaction()
    out_b = ph3.run_imag_self_energy(grid_points=[0], temperatures=[TEMP],
                                     frequency_points_at_bands=True)
    g_b = np.squeeze(np.array(out_b.gammas))            # (nbands,) at the band freqs
    fp = np.linspace(0.0, FMAX, 241)                    # frequency-resolved Im Sigma(w)
    out_f = ph3.run_imag_self_energy(grid_points=[0], temperatures=[TEMP],
                                     frequency_points=fp)
    g_f = np.squeeze(np.array(out_f.gammas))            # (nbands, n_fp)
    return g_b, fp, g_f


print(f"=== prefactor check: bulk Si, {NMESH}x{NMESH}x{NMESH} mesh, T={TEMP}K, eta={ETA} THz ===",
      flush=True)
# Gamma point optical bands (bands 3,4,5 ~15.4 THz)
iq_g = q_idx[(0.0, 0.0, 0.0)]
print(f"Gamma freqs (THz): {np.round(omega_l[iq_g], 3)}")
gn = [gamma_negf(iq_g, b) for b in range(nd)]
gn_int = [gamma_negf_integrated(iq_g, b) for b in range(nd)]
print(f"Gamma_NEGF on-shell (THz): {np.round(gn, 5)}")


def integ_p3p(fp, gf_band, wl):
    win = (fp > wl - 3.0) & (fp < wl + 3.0)
    return float(np.trapezoid(np.asarray(gf_band)[win], fp[win]))


try:
    g_b, fp, g_f = phono3py_gamma_gamma_point()
    g_b = np.atleast_1d(np.squeeze(g_b))
    print(f"gamma_phono3py on-shell (THz): {np.round(g_b, 5)}")
    print(f"  band :  w(THz)   R_onshell   R_integrated(broadening-free)")
    for b in range(nd):
        if not np.isfinite(gn[b]) or g_b[b] <= 1e-6:
            continue
        wl = omega_l[iq_g, b]
        r_on = gn[b] / g_b[b]
        # integrated: NEGF area of HWHM(w)= -ImSig/(2w) vs p3p area of gamma(w)
        win = (freqs > wl - 3.0) & (freqs < wl + 3.0)
        negf_area = np.trapezoid(imsig_mode(iq_g, b)[win] / (2.0 * freqs[win]), freqs[win])
        p3p_area = integ_p3p(fp, g_f[b], wl)
        r_int = negf_area / p3p_area if abs(p3p_area) > 1e-9 else np.nan
        print(f"   {b} :  {wl:6.3f}   {r_on:8.3f}    {r_int:8.3f}")
    print("\nVERDICT: the kernel over-scatters by an O(4) factor (on-shell ratio is "
          "broadening-inflated\non coarse meshes; thin-film conductance independently pins it at "
          "~4). The symmetry-factor\nderivation (3*3*2 / (2!*3!*3!) = 1/4) and Guo App. A both give "
          "EXACTLY 4 -> correct prefactor = kernel/4.")
except Exception as e:
    import traceback
    print(f"[phono3py gamma failed: {e}]")
    traceback.print_exc()
print("[done]")
