"""Physical constants and unit conversion factors.

All constants use SI base units except where noted.

Two unit systems for dynamical-matrix eigenvalues:
  (rad/s)^2:  CONVERSION  ≈ 9.648e27  (legacy, large magnitudes)
  THz^2:      CONVERSION_THZ2 ≈ 244.5  (preferred, O(1)–O(100) magnitudes)
"""

import numpy as np

# Fundamental constants
AMU_KG = 1.66053906660e-27  # atomic mass unit [kg]
EV_TO_J = 1.602176634e-19  # electron-volt [J]
HBAR_EV = 6.582119569e-16  # reduced Planck constant [eV*s]
HBAR_SI = HBAR_EV * EV_TO_J  # reduced Planck constant [J*s]
KB_EV = 8.617333262e-5  # Boltzmann constant [eV/K]
KB_SI = KB_EV * EV_TO_J  # Boltzmann constant [J/K]

# Derived conversion factors
EV_PER_A2_TO_SI = EV_TO_J / (1e-10) ** 2  # eV/Angstrom^2 -> kg/s^2
THZ_TO_RAD = 2 * np.pi * 1e12  # THz -> rad/s

# phonopy eigenvalue -> (rad/s)^2
# D(q) eigenvalues in eV/(amu*A^2), multiply by this to get (rad/s)^2
CONVERSION = EV_PER_A2_TO_SI / AMU_KG  # ≈ 9.648e27

# phonopy eigenvalue -> THz^2
# Same as CONVERSION but divided by THZ_TO_RAD^2
CONVERSION_THZ2 = CONVERSION / THZ_TO_RAD**2  # ≈ 244.5

# FC3 conversion factors (mass-weighted FC3 to frequency^{5/2} units)
# FC3 raw: eV/A^3, mass-weighted: eV/(A^3 amu^{3/2})
# To (rad/s)^{5/2}: multiply by CONVERSION / (1e-10 * sqrt(AMU_KG))
CONVERSION_FC3 = CONVERSION / (1e-10 * np.sqrt(AMU_KG))
# To THz^{5/2}: divide by THZ_TO_RAD^{5/2}
CONVERSION_FC3_THZ = CONVERSION_FC3 / THZ_TO_RAD**2.5

# phonopy's VaspToTHz: sqrt(eigenvalue) * VaspToTHz = frequency in THz
VASP_TO_THZ = 1 / (2 * np.pi) * np.sqrt(EV_TO_J / (AMU_KG * 1e-20))  # ≈ 15.633

# Adjustable scalar prefactor on the 3-phonon bubble self-energy. DEFAULT 1.0 =
# the self-energy as derived/implemented (the standard sunset diagram of
# theory.tex, Sigma = (i hbar/2) Phi G G Phi, which reduces analytically to the
# correct Fermi-golden-rule three-phonon linewidth).
#
# VERIFIED native (F28; scripts/verify/verify_gamma_opt.py). The native prefactor
# was checked against phono3py's golden-rule linewidths for bulk Si on the
# identical FC3. The AREA-INTEGRATED ratio R = int(-Im Sigma_NEGF)/int(2 w_s
# gamma_p3p) for the Gamma-optical mode (the one with a converged 3-phonon
# joint-DOS) is eta-INVARIANT at R/(2pi)^2 ~ 1.06 (+-4 THz window; the (2pi)^2 is
# the THz^2-vs-THz units convention), i.e. native matches the ab-initio linewidth
# to ~10-15% and a factor-of-4 error is excluded by an order of magnitude
# (div4 -> 0.26, x4 -> 4.2). The "factor of 4" that earlier motivated a div4 was
# the ON-SHELL peak ratio, which is a pure broadening artifact (gamma_NEGF ∝ eta).
# So the div4 is RETRACTED; native is the correct physics. (Residual ~15% is the
# window/grid method uncertainty; the convention-free closer is bulk-Si kappa from
# the code's own SCBA vs phono3py RTA ~110 W/mK.) The thin-film over-scatter vs
# Guo is therefore NOT a prefactor error (transport setup / their approximations /
# XC). Do not rescale: set this away from 1.0 only to explore, never as a correction.
PHPH_SYMMETRY_FACTOR = 1.0
