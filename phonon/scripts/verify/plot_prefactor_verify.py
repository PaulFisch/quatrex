"""Figure: prefactor verification (F28). Gamma-optical NEGF/phono3py area-integrated
linewidth ratio R/(2pi)^2 vs integration window, for three eta -- demonstrating the
integrated area is eta-invariant and sits at ~1 (native correct), while the on-shell
peak ratio is an eta-proportional broadening artifact. Factor-4 hypotheses excluded."""
import matplotlib; matplotlib.use("Agg"); import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
OUT=[Path("/usr/scratch/mont-fort11/pfischill/quatrex/document/fig/transport_sweeps")]
for o in OUT: o.mkdir(parents=True,exist_ok=True)
win=[2,4,6,8,12]
R={0.02:[0.8483,1.0505,0.8437,0.5305,0.6198],
   0.04:[0.9271,1.0651,0.8348,0.5369,0.5230],
   0.08:[0.9369,1.0638,0.8309,0.5411,0.5037]}
onshell_eta=[0.02,0.04,0.08]; onshell_g=[0.129,0.256,0.436]; g_p3p=0.0324
fig,(ax,ax2)=plt.subplots(1,2,figsize=(11,4.3))
cols={0.02:"#1f77b4",0.04:"#2ca02c",0.08:"#d62728"}
for eta in (0.02,0.04,0.08):
    ax.plot(win,R[eta],"o-",color=cols[eta],lw=1.8,label=fr"$\eta={eta}$ THz")
ax.axhline(1.0,ls="--",color="k",lw=1.1,label="native exact")
ax.axhspan(0.26/1,0.26,color="0.8",alpha=0)  # noop
ax.axhline(4.0,ls=":",color="#7f7f7f",lw=1.3,label=r"$\times4$ hypothesis")
ax.axhline(0.25,ls=":",color="#9467bd",lw=1.3,label=r"$\div4$ hypothesis")
ax.set_yscale("log"); ax.set_xticks(win)
ax.set_xlabel(r"integration half-window about $\omega_s$ (THz)")
ax.set_ylabel(r"$R/(2\pi)^2=\frac{\int-\mathrm{Im}\Sigma^{\rm NEGF}_s\,d\omega}{\int 2\omega_s\gamma^{\rm p3p}_s\,d\omega}\big/(2\pi)^2$")
ax.set_title(r"Area-integrated $\Gamma$-optical linewidth ratio"+"\n(native; $\\eta$-invariant, $\\approx1$)")
ax.legend(fontsize=8); ax.grid(alpha=0.3,which="both")
# on-shell artifact panel
ax2.plot(onshell_eta,onshell_g,"s-",color="#d62728",lw=2,label=r"NEGF on-shell $\gamma$ (peak)")
ax2.axhline(g_p3p,ls="--",color="k",label=r"phono3py $\gamma$ (converged)")
ax2.plot(onshell_eta,[6.0*e for e in onshell_eta],"k:",lw=1,label=r"$\propto\eta$ guide")
ax2.set_xlabel(r"broadening $\eta$ (THz)"); ax2.set_ylabel(r"$\gamma_s$ (THz)")
ax2.set_title("On-shell peak ratio is a broadening artifact\n"+r"($\gamma^{\rm NEGF}\propto\eta$; the spurious `factor 4')")
ax2.legend(fontsize=8.5); ax2.grid(alpha=0.3)
fig.tight_layout()
for o in OUT: fig.savefig(o/"prefactor_verification.pdf")
print("wrote prefactor_verification.pdf")
