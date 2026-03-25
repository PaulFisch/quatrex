import numpy as np
from pathlib import Path
import phonon_transport_sancho_rubio as ptsr

base = Path('/home/jiacao/quatrex_project/phonon/phonon-data/mos2/harmonic')
labels = [
    'q_m0d400000_p0d000000',
    'q_m0d200000_p0d000000',
    'q_p0d000000_p0d000000',
    'q_p0d200000_p0d000000',
    'q_p0d400000_p0d000000',
]

for label in labels:
    k00 = np.load(base / f'transport_blocks_device_fc_matrix_a_7_{label}_K00.npy')
    k01 = np.load(base / f'transport_blocks_device_fc_matrix_a_7_{label}_K01.npy')
    k10 = np.load(base / f'transport_blocks_device_fc_matrix_a_7_{label}_K10.npy')
    KC = ptsr.build_device_matrix_from_blocks(k00, k01, k10, num_cells=7)
    try:
        omega, trans, skipped = ptsr.compute_transmission_spectrum(
            KC=KC,
            K00=k00,
            K01=k01,
            K10=k10,
            omega_min=0.01,
            omega_max=20.0,
            omega_num=400,
            eta=1e-3,
            sancho_tol=1e-5,
            sancho_max_iter=5000,
        )
        print(label, 'ok', 'computed', len(omega), 'skipped', len(skipped), 'maxT', float(trans.max()))
    except Exception as exc:
        print(label, 'FAILED', str(exc))
