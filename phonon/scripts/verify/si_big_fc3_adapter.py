"""Adapter: hiphive fc3.hdf5 (fc2+fc3, 5x5x5 supercell) -> load_bulk_si-compatible reaps dir
(phono3py.yaml + fc2.hdf5[force_constants] + fc3.hdf5[fc3]); verify dispersion + bulk kappa."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, h5py
from pathlib import Path
from phonopy.structure.atoms import PhonopyAtoms
from phono3py import Phono3py
SRC=Path("configs/si_primitive/fc3_hiphive_si_big/fc3.hdf5")
OUT=Path("reaps/si_big_hiphive"); OUT.mkdir(parents=True,exist_ok=True)
a=2.734
cell=PhonopyAtoms(symbols=["Si","Si"],
    cell=[[0,a,a],[a,0,a],[a,a,0]],scaled_positions=[[0,0,0],[0.25,0.25,0.25]])
ph3=Phono3py(cell,supercell_matrix=[5,5,5],primitive_matrix=np.eye(3),log_level=0)
with h5py.File(SRC) as f: fc2=f["fc2"][:]; fc3=f["fc3"][:]
print("fc2",fc2.shape,"fc3",fc3.shape,"n_satom",len(ph3.supercell))
ph3.fc2=fc2; ph3.fc3=fc3
# write yaml + hdf5 compatible with load_bulk_si
ph3.save(str(OUT/"phono3py.yaml"))
with h5py.File(OUT/"fc2.hdf5","w") as f: f.create_dataset("force_constants",data=fc2)
with h5py.File(OUT/"fc3.hdf5","w") as f: f.create_dataset("fc3",data=fc3)
print("wrote",OUT,"->",[p.name for p in OUT.iterdir()])
# dispersion sanity: frequencies at a few q (no imaginary modes?)
from phonopy import Phonopy
ph=Phonopy(cell,supercell_matrix=[5,5,5],primitive_matrix=np.eye(3)); ph.force_constants=fc2
fmin=1e9
for q in [(0,0,0),(.5,0,0),(.5,.5,0),(.5,.5,.5),(.25,.25,.25)]:
    fr=ph.get_frequencies(np.array(q)); fmin=min(fmin,fr.min())
    if q==(0,0,0) or q==(.5,.5,.5): print(f"  q={q}: {np.round(fr,3)}")
print(f"  global min freq over sampled q: {fmin:.4f} THz  ({'CLEAN' if fmin>-0.05 else 'IMAGINARY MODES'})")
# bulk kappa (RTA) at 11^3
for nm in (11,):
    ph3.mesh_numbers=[nm,nm,nm]; ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=[300],is_isotope=True)
    kap=ph3.thermal_conductivity.kappa[0,0,:3].mean()
    print(f"  bulk Si kappa(300K, {nm}^3 RTA, +isotope) = {kap:.1f} W/mK   (vs 2x2x2 phono3py 110; Guo 1st/2nd/3rd-NN 120/136/147)")
