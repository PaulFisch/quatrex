"""Pin the 'last 25%': Gamma-optical NEGF vs phono3py linewidth, native prefactor,
window + eta sensitivity. If R/(2pi)^2 -> ~1 as the integration window widens, the 0.77
was window truncation and native is exact; if it plateaus < 1, there is a real residual.
Also report the ON-SHELL ratio (eta->0 sensitive) and the FULL-spectrum integral."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"phonon"); sys.path.insert(0,".")
import numpy as np, h5py, phono3py
from phonon.phonon_inputs.separable import build_supercell_mapping, build_realspace_fc3_matrices
from phonon.solver.se_q import compute_phph_self_energy_q_dense
from phonon.solver.retarded import build_retarded
from phonon.solver.grids import bose_full_axis
from phonon.scripts.verify.si_film_kappa import load_bulk_si
NM=int(sys.argv[1]) if len(sys.argv)>1 else 6
T=300.0; NE=401; FMAX=24.0; TPI2=(2*np.pi)**2
ph,fc3p=load_bulk_si(); nat=len(ph.primitive.masses); nd=3*nat
with h5py.File(fc3p) as f: fc3=f["fc3"][:]
pi,cf,si,rs=build_supercell_mapping(ph,"x"); M=build_realspace_fc3_matrices(fc3,nat,ph.supercell.masses,rs)
qs=[(i/NM,j/NM,k/NM) for i in range(NM) for j in range(NM) for k in range(NM)]
nq=len(qs); idx={(round(q[0]%1,6),round(q[1]%1,6),round(q[2]%1,6)):n for n,q in enumerate(qs)}
def gm(qf):
    ns=len(pi); Tm=np.zeros((nd,ns*3),complex); p=np.exp(-2j*np.pi*cf@np.asarray(qf))
    for s in range(ns):
        for b in range(3): Tm[pi[s]*3+b,s*3+b]=p[s]
    return Tm
Tall=[gm(q) for q in qs]
qd=np.zeros((nq,nq),int)
for a,qa in enumerate(qs):
    for b,qb in enumerate(qs):
        qd[a,b]=idx[(round((qa[0]-qb[0])%1,6),round((qa[1]-qb[1])%1,6),round((qa[2]-qb[2])%1,6))]
freqs=np.linspace(-FMAX,FMAX,NE); freqs-=freqs[NE//2]; dw=float(freqs[1]-freqs[0])
# phono3py Gamma-optical gamma(omega), fine frequency grid + a converged comparison run
ph3=phono3py.load(phono3py_yaml="phonon/reaps/si_primitive_work/phono3py.yaml",log_level=0,produce_fc=False)
with h5py.File("phonon/reaps/si_primitive_work/fc2.hdf5") as f: ph3.fc2=f["force_constants"][:]
with h5py.File(fc3p) as f: ph3.fc3=f["fc3"][:]
fp=np.linspace(0,FMAX,801)
ph3.mesh_numbers=[NM,NM,NM]; ph3.init_phph_interaction()
out=ph3.run_imag_self_energy(grid_points=[0],temperatures=[T],frequency_points=fp)
gf=np.squeeze(np.array(out.gammas)); ws=15.366  # Gamma-optical
gopt=gf[3]  # band 3 (degenerate with 4,5)
print(f"phono3py Gamma-optical gamma(w_s) freq-res {NM}^3 = {gopt[np.argmin(abs(fp-ws))]:.4f} THz")
for eta in (0.02,0.04,0.08):
    nB=bose_full_axis(freqs,T)
    Gl=np.zeros((nq,NE,nd,nd),complex); Gg=np.zeros_like(Gl); om=np.zeros((nq,nd)); ev=np.zeros((nq,nd,nd),complex)
    for iq,q in enumerate(qs):
        fr,e=ph.get_frequencies_with_eigenvectors(np.array(q)); fr=np.real(fr); om[iq]=fr; ev[iq]=e
        Dq=e@np.diag(fr.astype(complex)**2)@e.conj().T; z2=(freqs+1j*eta)**2
        GR=np.linalg.inv(z2[:,None,None]*np.eye(nd)[None]-Dq[None]); A=1j*(GR-GR.conj().transpose(0,2,1))
        Gl[iq]=-1j*nB[:,None,None]*A; Gg[iq]=-1j*(nB[:,None,None]+1.0)*A
    sl,sg=compute_phph_self_energy_q_dense(Gl,Gg,M,Tall,qd,nat,nq,freqs,dw,n_workers=8,symmetry_factor=1.0)
    sigR=build_retarded(sl,sg,freqs,method="pv")
    e3=ev[0,:,3]; imS=-np.imag(np.einsum('i,wij,j->w',e3.conj(),sigR[0],e3))
    # on-shell
    iw=np.argmin(abs(freqs-ws)); g_negf_onshell=imS[iw]/(2*ws)
    print(f"\n eta={eta}: on-shell  gamma_NEGF={g_negf_onshell:.3f} THz  / (2pi)^2={g_negf_onshell/TPI2:.4f}  "
          f"vs phono3py freq-res {gopt[np.argmin(abs(fp-ws))]:.4f}")
    for half in (2,4,6,8,12):
        wn=(freqs>ws-half)&(freqs<ws+half); In=np.trapezoid(imS[wn],freqs[wn])
        wp=(fp>ws-half)&(fp<ws+half); Ip=np.trapezoid(2*ws*gopt[wp],fp[wp])
        R=In/Ip if abs(Ip)>1e-12 else 0
        print(f"   window +/-{half:2d} THz: R={R:8.3f}  R/(2pi)^2={R/TPI2:.4f}")
