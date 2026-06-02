"""Does the on-shell Gamma-optical gamma_NEGF stop scaling linearly with eta?
Compute Sigma for the SINGLE external Gamma mode (O(N), streaming vertex) over a fine
internal q-mesh, sweep eta down, at several meshes. Linear-eta regime = eta >> mesh
frequency spacing (artificial broadening dominates); it must turn over toward the true
linewidth (phono3py ~0.032 THz) once eta and the mesh resolve the joint-DOS."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"phonon"); sys.path.insert(0,".")
import numpy as np, h5py
from phonon.phonon_inputs.separable import build_supercell_mapping, build_realspace_fc3_matrices
from phonon.solver.se_q import _se_worker_iq
from phonon.phonon_inputs.constants import HBAR_SI
from phonon.solver.retarded import build_retarded
from phonon.solver.grids import bose_full_axis
from phonon.scripts.verify.si_film_kappa import load_bulk_si
T=300.0; FMAX=24.0; DW=0.05; ws=15.366; G_P3P=0.0324
ph,fc3p=load_bulk_si(); nat=len(ph.primitive.masses); nd=3*nat
with h5py.File(fc3p) as f: fc3=f["fc3"][:]
pi,cf,si,rs=build_supercell_mapping(ph,"x"); M=build_realspace_fc3_matrices(fc3,nat,ph.supercell.masses,rs)
NE=int(2*FMAX/DW)|1; freqs=np.linspace(-FMAX,FMAX,NE); freqs-=freqs[NE//2]; dw=float(freqs[1]-freqs[0])
n_fft=2*NE-1; mid=NE//2; nB=bose_full_axis(freqs,T)
dim_t=M.shape[1]; M_blocks=M.reshape(nd,dim_t,dim_t)
print(f"NE={NE} dw={dw:.3f} THz ; phono3py Gamma-opt gamma={G_P3P} THz")
def gm(qf):
    ns=len(pi); Tm=np.zeros((nd,ns*3),complex); p=np.exp(-2j*np.pi*cf@np.asarray(qf))
    for s in range(ns):
        for b in range(3): Tm[pi[s]*3+b,s*3+b]=p[s]
    return Tm
for NM in (6,8,10,12):
    qs=[(i/NM,j/NM,k/NM) for i in range(NM) for j in range(NM) for k in range(NM)]
    nq=len(qs); idx={(round(q[0]%1,6),round(q[1]%1,6),round(q[2]%1,6)):n for n,q in enumerate(qs)}
    qdiff_gamma=np.array([idx[(round((-q[0])%1,6),round((-q[1])%1,6),round((-q[2])%1,6))] for q in qs])
    q_diff_map={0:qdiff_gamma}
    T_arr=np.array([gm(q) for q in qs])
    TM=np.einsum('qci,aij->qacj',T_arr,M_blocks); T_arr_H=T_arr.conj().transpose(0,2,1).copy()
    om=np.zeros((nq,nd)); ev=np.zeros((nq,nd,nd),complex)
    for iq,q in enumerate(qs):
        fr,e=ph.get_frequencies_with_eigenvectors(np.array(q)); om[iq]=np.real(fr); ev[iq]=e
    e3=ev[0,:,3]  # Gamma optical
    df=(FMAX/NM)  # rough mesh frequency spacing (max accoustic slope proxy)
    print(f"\nNM={NM}^3 (nq={nq}, ~mesh freq spacing v_g/N ~ {df:.2f} THz):")
    for eta in (0.005,0.01,0.02,0.04,0.08,0.16):
        z2=(freqs+1j*eta)**2
        Gl=np.zeros((nq,NE,nd,nd),complex); Gg=np.zeros_like(Gl)
        for iq in range(nq):
            Dq=ev[iq]@np.diag(om[iq].astype(complex)**2)@ev[iq].conj().T
            GR=np.linalg.inv(z2[:,None,None]*np.eye(nd)[None]-Dq[None]); A=1j*(GR-GR.conj().transpose(0,2,1))
            Gl[iq]=-1j*nB[:,None,None]*A; Gg[iq]=-1j*(nB[:,None,None]+1.0)*A
        Gl[:,mid]=0; Gg[:,mid]=0
        Gp=np.zeros((nq,n_fft,nd,nd),complex); Gp[:,:NE]=Gl; GL_fft=np.fft.fft(Gp,axis=1)
        Gp[:]=0; Gp[:,:NE]=Gg; GG_fft=np.fft.fft(Gp,axis=1)
        pref=1.0*0.5j*HBAR_SI*dw/(2*np.pi)/nq  # native symmetry_factor=1
        qp_batch=max(1,min(nq,(16*1024*1024)//(n_fft*nd**3*16)))
        common=(GL_fft,GG_fft,("stream",TM,T_arr_H),q_diff_map,NE,nd,nq,n_fft,mid,mid+NE,pref,qp_batch)
        _,sl,sg=_se_worker_iq(([0],*common))
        sigR=build_retarded(sl,sg,freqs,method="pv")
        imS=-np.imag(np.einsum('i,wij,j->w',e3.conj(),sigR[0],e3))
        g_on=imS[np.argmin(abs(freqs-ws))]/(2*ws)
        print(f"   eta={eta:5.3f}: gamma_NEGF(w_s)={g_on:7.4f} THz   gamma/eta={g_on/eta:7.3f}   /gamma_p3p={g_on/G_P3P:6.2f}")
