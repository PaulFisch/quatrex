"""Decisive SSE check: is the NEGF/phono3py linewidth ratio CONSTANT across the BZ?
If R(mode) = const (= a pure units factor), the native SSE reproduces phono3py's golden-rule
linewidth mode-by-mode -> the physics is correct (native), the constant is a units convention.
If R varies by mode -> the SSE is genuinely wrong. Native prefactor; broadening-free integrated ratio
R = int(-Im Sigma_NEGF) dw / int(2 w_s gamma_p3p) dw, computed at several distinct grid points."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"phonon"); sys.path.insert(0,".")
import numpy as np, h5py, phono3py
from phonon.phonon_inputs.separable import build_supercell_mapping, build_realspace_fc3_matrices
from reference_kernels import compute_phph_self_energy_q_dense
from phonon.solver.retarded import build_retarded
from phonon.solver.grids import bose_full_axis
from phonon.scripts.verify.si_film_kappa import load_bulk_si
NM=int(sys.argv[1]) if len(sys.argv)>1 else 8
ETA=float(sys.argv[2]) if len(sys.argv)>2 else 0.04
T=300.0; NE=201; FMAX=20.0
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
freqs=np.linspace(-FMAX,FMAX,NE); freqs-=freqs[NE//2]; dw=float(freqs[1]-freqs[0]); nB=bose_full_axis(freqs,T)
om=np.zeros((nq,nd)); ev=np.zeros((nq,nd,nd),complex); Gl=np.zeros((nq,NE,nd,nd),complex); Gg=np.zeros_like(Gl)
for iq,q in enumerate(qs):
    fr,e=ph.get_frequencies_with_eigenvectors(np.array(q)); fr=np.real(fr); om[iq]=fr; ev[iq]=e
    Dq=e@np.diag(fr.astype(complex)**2)@e.conj().T; z2=(freqs+1j*ETA)**2
    GR=np.linalg.inv(z2[:,None,None]*np.eye(nd)[None]-Dq[None]); A=1j*(GR-GR.conj().transpose(0,2,1))
    Gl[iq]=-1j*nB[:,None,None]*A; Gg[iq]=-1j*(nB[:,None,None]+1.0)*A
sl,sg=compute_phph_self_energy_q_dense(Gl,Gg,M,Tall,qd,nat,nq,freqs,dw,n_workers=8,symmetry_factor=1.0)
sigR=build_retarded(sl,sg,freqs,method="pv")
# phono3py: gamma at selected grid points, frequency-resolved
ph3=phono3py.load(phono3py_yaml="phonon/reaps/si_primitive_work/phono3py.yaml",log_level=0,produce_fc=False)
with h5py.File("phonon/reaps/si_primitive_work/fc2.hdf5") as f: ph3.fc2=f["force_constants"][:]
with h5py.File(fc3p) as f: ph3.fc3=f["fc3"][:]
ph3.mesh_numbers=[NM,NM,NM]; ph3.init_phph_interaction()
bz=ph3.grid; addr=bz.addresses  # (n_gp,3) integer; q = addr/NM
# pick a spread of grid points (distinct |q|, acoustic+optical content)
targets=[(0,0,0),(0,0,2),(0,2,2),(2,2,2),(0,0,3),(2,2,3)]  # in units of 1/NM (NM=6)
fp=np.linspace(0,FMAX,401)
print(f"native SSE, {NM}^3, eta={ETA}: R = int(-ImSig_NEGF)/int(2 w_s gamma_p3p)  [const => units only]")
allR=[]
for tgt in targets:
    qf=tuple((t/NM) for t in tgt)
    key=(round(qf[0]%1,6),round(qf[1]%1,6),round(qf[2]%1,6))
    if key not in idx: continue
    iqn=idx[key]
    # find phono3py gp with this address
    gp=None
    for g in range(len(addr)):
        a=tuple(int(x)%NM for x in addr[g])
        if a==tuple(t%NM for t in tgt): gp=g; break
    if gp is None: continue
    out=ph3.run_imag_self_energy(grid_points=[gp],temperatures=[T],frequency_points=fp)
    gf=np.squeeze(np.array(out.gammas))  # (nbands,nfp)
    for b in range(nd):
        ws=om[iqn,b]
        if ws<1.5: continue
        e=ev[iqn,:,b]; imS=-np.imag(np.einsum('i,wij,j->w',e.conj(),sigR[iqn],e))
        win=(freqs>ws-4)&(freqs<ws+4); I_n=np.trapezoid(imS[win],freqs[win])
        wp=(fp>ws-4)&(fp<ws+4); I_p=np.trapezoid(2*ws*gf[b][wp],fp[wp])
        if abs(I_p)>1e-10:
            R=I_n/I_p; allR.append(R)
            print(f"  q={tgt}/{NM} band{b} w={ws:6.2f} THz : R={R:8.3f}  R/(2pi)^2={R/(2*np.pi)**2:.3f}")
allR=np.array(allR)
print(f"\nR: mean={allR.mean():.2f} std={allR.std():.2f} rel-spread={allR.std()/allR.mean():.1%}  "
      f"mean/(2pi)^2={allR.mean()/(2*np.pi)**2:.3f}")
print("CONSTANT across modes => native SSE correct up to a units convention" if allR.std()/allR.mean()<0.15
      else "VARIES across modes => SSE physics is off, not a pure prefactor")
