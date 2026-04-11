"""Parameter and memory scaling plots for all four FC3 approximation methods.

Analytical scaling with system size (n_prim) at fixed representative ranks,
plus compression ratio and memory estimates.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

fig_dir = Path(__file__).resolve().parent / "figures"
fig_dir.mkdir(exist_ok=True)

n_cells = 8  # 2x2x2 supercell

n_prim_range = np.array([2, 4, 8, 16, 32, 64])
n_super = n_prim_range * n_cells
n_dof = 3 * n_prim_range
dim_sc = 3 * n_super

# Fixed ranks (representative for ~5% Frobenius error based on Si results)
R_svd = 8
R_pscp = 36
R_scp3 = 8
R_fscp = 16

METHODS = ["SVD", "PSCP", "SCP3", "FSCP"]
COLORS = {"SVD": "#2ca02c", "PSCP": "#ff7f0e", "SCP3": "#1f77b4", "FSCP": "#9467bd"}
MARKERS = {"SVD": "o", "PSCP": "s", "SCP3": "^", "FSCP": "D"}

# Parameter counts
params_full = n_dof * dim_sc ** 2
params_svd = R_svd * dim_sc * (n_dof + 1)
params_pscp = R_pscp * (n_dof + dim_sc)
params_scp3 = R_scp3 * (3 * dim_sc + 1)
params_fscp = R_fscp * (dim_sc + 1)

params = {"SVD": params_svd, "PSCP": params_pscp,
          "SCP3": params_scp3, "FSCP": params_fscp}
ranks = {"SVD": R_svd, "PSCP": R_pscp, "SCP3": R_scp3, "FSCP": R_fscp}

# Memory in bytes (float64 = 8 bytes)
bytes_per_param = 8
mem_full = params_full * bytes_per_param
mem = {m: params[m] * bytes_per_param for m in METHODS}


# =========================================================================
# Figure 1: Parameter count vs system size
# =========================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.loglog(n_prim_range, params_full, "k--", lw=2, label="Full tensor", alpha=0.5)
for method in METHODS:
    ax1.loglog(n_prim_range, params[method],
               f"-{MARKERS[method]}", color=COLORS[method],
               label=f"{method} ($R={ranks[method]}$)", markersize=7)

ax1.set_xlabel(r"$n_{\mathrm{prim}}$ (atoms in primitive cell)", fontsize=12)
ax1.set_ylabel("Number of parameters", fontsize=12)
ax1.set_title(r"(a) Parameter count ($2\times2\times2$ supercell)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, which="both")

# Right: compression ratio
for method in METHODS:
    ax2.loglog(n_prim_range, params_full / params[method],
               f"-{MARKERS[method]}", color=COLORS[method],
               label=f"{method} ($R={ranks[method]}$)", markersize=7)

ax2.set_xlabel(r"$n_{\mathrm{prim}}$ (atoms in primitive cell)", fontsize=12)
ax2.set_ylabel("Compression ratio (full / approx)", fontsize=12)
ax2.set_title(r"(b) Compression ratio vs system size")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(fig_dir / "scaling_params_vs_nprim.pdf", bbox_inches="tight")
fig.savefig(fig_dir / "scaling_params_vs_nprim.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: scaling_params_vs_nprim.pdf")


# =========================================================================
# Figure 2: Memory scaling
# =========================================================================

fig, ax = plt.subplots(figsize=(8, 5))

def to_readable(b):
    if b > 1e9:
        return b / 1e9, "GB"
    elif b > 1e6:
        return b / 1e6, "MB"
    elif b > 1e3:
        return b / 1e3, "KB"
    return b, "B"

ax.loglog(n_prim_range, mem_full / 1e6, "k--", lw=2, label="Full tensor", alpha=0.5)
for method in METHODS:
    ax.loglog(n_prim_range, mem[method] / 1e6,
              f"-{MARKERS[method]}", color=COLORS[method],
              label=f"{method} ($R={ranks[method]}$)", markersize=7)

# Annotate the n_prim=64 point
for method in METHODS:
    val, unit = to_readable(mem[method][-1])
    ax.annotate(f"{val:.0f} {unit}", (n_prim_range[-1], mem[method][-1] / 1e6),
                textcoords="offset points", xytext=(8, 0), fontsize=7,
                color=COLORS[method])
val_f, unit_f = to_readable(mem_full[-1])
ax.annotate(f"{val_f:.0f} {unit_f}", (n_prim_range[-1], mem_full[-1] / 1e6),
            textcoords="offset points", xytext=(8, 0), fontsize=7, color="k")

ax.set_xlabel(r"$n_{\mathrm{prim}}$ (atoms in primitive cell)", fontsize=12)
ax.set_ylabel("Memory (MB)", fontsize=12)
ax.set_title(r"Memory scaling ($2\times2\times2$ supercell, float64)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(fig_dir / "scaling_memory.pdf", bbox_inches="tight")
fig.savefig(fig_dir / "scaling_memory.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: scaling_memory.pdf")


# =========================================================================
# Figure 3: Scaling exponents
# =========================================================================

fig, ax = plt.subplots(figsize=(8, 5))

# Fit log-log slopes
for method in METHODS:
    p = np.polyfit(np.log(n_prim_range), np.log(params[method]), 1)
    ax.bar(method, p[0], color=COLORS[method], alpha=0.8, edgecolor="gray")
    ax.text(method, p[0] + 0.05, f"{p[0]:.2f}", ha="center", fontsize=10)

p_full = np.polyfit(np.log(n_prim_range), np.log(params_full), 1)
ax.axhline(p_full[0], color="k", ls="--", lw=1, alpha=0.5)
ax.text(len(METHODS) - 0.5, p_full[0] + 0.05,
        f"Full: {p_full[0]:.2f}", fontsize=9, color="k")

ax.set_ylabel(r"Scaling exponent $\alpha$ in $O(n_{\mathrm{prim}}^\alpha)$", fontsize=12)
ax.set_title("Parameter scaling exponent (fixed rank)")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(fig_dir / "scaling_exponents.pdf", bbox_inches="tight")
fig.savefig(fig_dir / "scaling_exponents.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: scaling_exponents.pdf")


# =========================================================================
# Figure 4: Variable supercell size
# =========================================================================

n_prim_fixed = 2
sc_sizes = np.array([1, 2, 3, 4, 5])
n_cells_var = sc_sizes ** 3
n_super_var = n_prim_fixed * n_cells_var
n_dof_fixed = 3 * n_prim_fixed
dim_sc_var = 3 * n_super_var

p_full_sc = n_dof_fixed * dim_sc_var ** 2
p_svd_sc = R_svd * dim_sc_var * (n_dof_fixed + 1)
p_pscp_sc = R_pscp * (n_dof_fixed + dim_sc_var)
p_scp3_sc = R_scp3 * (3 * dim_sc_var + 1)
p_fscp_sc = R_fscp * (dim_sc_var + 1)

params_sc = {"SVD": p_svd_sc, "PSCP": p_pscp_sc,
             "SCP3": p_scp3_sc, "FSCP": p_fscp_sc}

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(n_cells_var, p_full_sc, "k--", lw=2, label="Full tensor", alpha=0.5)
for method in METHODS:
    ax.loglog(n_cells_var, params_sc[method],
              f"-{MARKERS[method]}", color=COLORS[method],
              label=f"{method} ($R={ranks[method]}$)", markersize=7)

ax.set_xlabel(r"Number of cells $N_c$ ($N_c = L^3$)", fontsize=12)
ax.set_ylabel("Number of parameters", fontsize=12)
ax.set_title(r"Scaling with supercell size ($n_{\mathrm{prim}}=2$)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(fig_dir / "scaling_vs_supercell.pdf", bbox_inches="tight")
fig.savefig(fig_dir / "scaling_vs_supercell.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: scaling_vs_supercell.pdf")


# =========================================================================
# Print summary table
# =========================================================================

print(f"\n{'n_prim':>6} {'Full':>14} {'SVD':>10} {'PSCP':>10} {'SCP3':>10} {'FSCP':>10}")
print("-" * 66)
for i, np_ in enumerate(n_prim_range):
    print(f"{np_:>6} {params_full[i]:>14,} {params_svd[i]:>10,} "
          f"{params_pscp[i]:>10,} {params_scp3[i]:>10,} {params_fscp[i]:>10,}")

print(f"\nScaling exponents (fixed rank):")
for method in METHODS:
    p = np.polyfit(np.log(n_prim_range), np.log(params[method]), 1)
    print(f"  {method}: O(n_prim^{p[0]:.2f})")
p_full = np.polyfit(np.log(n_prim_range), np.log(params_full), 1)
print(f"  Full: O(n_prim^{p_full[0]:.2f})")
