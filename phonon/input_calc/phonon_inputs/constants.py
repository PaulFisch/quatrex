"""Physical constants and unit conversion factors.

All constants use SI base units except where noted.
The CONVERSION factor converts phonopy dynamical-matrix eigenvalues
(eV / (amu * Angstrom^2)) to angular-frequency squared ((rad/s)^2).
"""

import numpy as np

# Fundamental constants
AMU_KG = 1.66053906660e-27  # atomic mass unit [kg]
EV_TO_J = 1.602176634e-19  # electron-volt [J]
HBAR_EV = 6.582119569e-16  # reduced Planck constant [eV*s]
KB_EV = 8.617333262e-5  # Boltzmann constant [eV/K]

# Derived conversion factors
EV_PER_A2_TO_SI = EV_TO_J / (1e-10) ** 2  # eV/Angstrom^2 -> kg/s^2
THZ_TO_RAD = 2 * np.pi * 1e12  # THz -> rad/s

# phonopy eigenvalue -> (rad/s)^2
# D(q) eigenvalues in eV/(amu*A^2), multiply by this to get (rad/s)^2
CONVERSION = EV_PER_A2_TO_SI / AMU_KG  # ≈ 9.648e27

# phonopy's VaspToTHz: sqrt(eigenvalue) * VaspToTHz = frequency in THz
VASP_TO_THZ = 1 / (2 * np.pi) * np.sqrt(EV_TO_J / (AMU_KG * 1e-20))  # ≈ 15.633
