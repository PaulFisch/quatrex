"""Peak-memory footprint of the q-resolved 3-phonon vertex: dense Phi(q1,q2) vs streamed.

The coupled-q self-energy needs Phi(q1,q2)=T(q1) M T(q2)^H for all (q1,q2). Materialising it
densely costs O(N_q^2 n_dof^3) (se_q.py:Phi_all); streaming it (stream_phi=True) keeps only
T_arr/M and rebuilds each batch, dropping the peak to O(N_q n_dof^2 dim_t + qp_batch n_dof^3) --
the GPU-memory-bound path the production run needs. A measured tracemalloc point validates the
analytic curves.
"""
import sys
import tracemalloc
import warnings
from pathlib import Path

_W = Path("/usr/scratch/mont-fort11/pfischill/quatrex/phonon")
for p in (_W.parent, _W):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTS = [_W / "scripts/out/si_film",
        Path("/usr/scratch/mont-fort11/pfischill/quatrex/document/fig/transport_sweeps")]
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)

# --- measured peak: dense vs stream Phi for a moderate case (n_dof=63 wire-like) ---
def measure(n_dof, N_q, dim_t, stream):
    rng = np.random.default_rng(0)
    T_arr = (rng.standard_normal((N_q, n_dof, dim_t))
             + 1j * rng.standard_normal((N_q, n_dof, dim_t)))
    M_blocks = rng.standard_normal((n_dof, dim_t, dim_t)).astype(complex)
    tracemalloc.start()
    TM = np.einsum('qci,aij->qacj', T_arr, M_blocks)
    T_arr_H = T_arr.conj().transpose(0, 2, 1).copy()
    if not stream:
        Phi_all = np.einsum('qacj,rjd->qracd', TM, T_arr_H)  # noqa: F841
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


N_q_meas, n_dof_meas, dim_t_meas = 8, 63, 189
try:
    peak_dense = measure(n_dof_meas, N_q_meas, dim_t_meas, stream=False)
    peak_stream = measure(n_dof_meas, N_q_meas, dim_t_meas, stream=True)
    print(f"measured (n_dof={n_dof_meas}, N_q={N_q_meas}, dim_t={dim_t_meas}): "
          f"dense peak={peak_dense/1e6:.1f} MB, stream peak={peak_stream/1e6:.1f} MB, "
          f"ratio={peak_dense/max(peak_stream,1):.1f}x")
except Exception as e:
    peak_dense = peak_stream = None
    print(f"[measure skipped: {e}]")

# --- analytic curves vs N_q ---
fig, ax = plt.subplots(figsize=(6.0, 4.3))
Nq = np.arange(2, 33)
for n_dof, dim_t, col in [(6, 16, "#1f77b4"), (63, 189, "#d62728")]:
    dense = Nq**2 * n_dof**3 * 16 / 1e9
    stream = (Nq * n_dof**2 * dim_t * 16 + n_dof**3 * 16) / 1e9
    ax.plot(Nq, dense, "-", color=col, label=f"dense $\\Phi$, $n_{{dof}}={n_dof}$")
    ax.plot(Nq, stream, "--", color=col, label=f"streamed, $n_{{dof}}={n_dof}$")
ax.axhline(80, color="gray", ls=":", lw=1.0, label="80 GB GPU")
if peak_dense:
    ax.plot([N_q_meas], [peak_dense / 1e9], "k*", ms=12, label="measured (dense)")
ax.set_yscale("log")
ax.set_xlabel("transverse q-mesh count $N_q$")
ax.set_ylabel("peak vertex memory (GB)")
ax.set_title("q-resolved 3-phonon vertex: dense $O(N_q^2 n_{dof}^3)$ vs streamed $O(N_q n_{dof}^2)$")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
for o in OUTS:
    fig.savefig(o / "phph_memory.pdf")
plt.close(fig)
print("[done] wrote phph_memory.pdf")
