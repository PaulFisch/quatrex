"""Micro-benchmark: time one SCBA bubble for a finite wire vs BLAS threads.

The phph self-energy bubble (`phonon/solver/bubble.py:_bubble_contract_batched_matmul`)
is a three batched-matmul kernel routed through BLAS. For an n_slabs=1 device there is
exactly one (I,J) slab pair, so the phph thread-pool offers no parallelism and the kernel
is bound by BLAS threading. This times a fixed 2-iteration SCBA (= a couple of bubbles)
so the wall time is dominated by the bubble; run it under different OPENBLAS_NUM_THREADS
to measure the scaling. Set the wire via argv[1] (d5a|d11a).
"""
import os
import sys
import time
import warnings
from pathlib import Path

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

from phonon.finite_analysis.loader import load_system  # noqa: E402
from phonon.solver.dense import transmission_finite  # noqa: E402

wire = sys.argv[1] if len(sys.argv) > 1 else "d5a"
cfg = {"d5a": "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml",
       "d11a": "phonon/configs/sinw/sinw100_d11a_vasp_sc4.yaml"}[wire]
blas = os.environ.get("OPENBLAS_NUM_THREADS", "?")

b = load_system(_REPO / cfg, validate=False, transport_axis=2)
fc3 = str(Path(b.meta["fc3_path"]).expanduser().resolve())

t0 = time.time()
transmission_finite(
    b.phonon, fc3_hdf5=fc3, freq_range_thz=(0.01, 18.0, 41),
    transport_direction="z", temperature=300.0, delta_T=10.0, n_slabs=1,
    eta_factor=0.5, vertex_scale=1.0, max_scba_iter=2, enforce_asr=True,
    anderson_mixing=True, verbose=False)
dt = time.time() - t0
print(f"BENCH wire={wire} OPENBLAS_NUM_THREADS={blas} : 2-iter wall = {dt:.1f}s",
      flush=True)
