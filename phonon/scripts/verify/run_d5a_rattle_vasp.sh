#!/usr/bin/env bash
# Parallel VASP over the d5a rattle-sweep disp dirs (0.015/0.05/0.08), 8 wide.
cd /usr/scratch/mont-fort11/pfischill/quatrex/phonon/configs/sinw || exit 1
VCMD='ulimit -s unlimited; export OMP_STACKSIZE=512m; export LD_LIBRARY_PATH=/usr/pack/intel_compiler-2020-af/x64/lib/intel64:/usr/pack/intel_compiler-2020-af/x64/mkl/lib/intel64:/home/jiacao/openmpi/4.1.1-ifort/lib/; /home/jiacao/openmpi/4.1.1-ifort/bin/mpirun -np 28 /home/jiacao/Software/vasp.6.3.0/bin/vasp_std'
ls -d fc3_d5a_rattle*_vasp/disp-* | xargs -P 8 -I{} bash -c '
  d="{}"; cd "$d" || exit 0
  if grep -q "Total CPU time\|reached required\|aborting loop" OUTCAR 2>/dev/null; then exit 0; fi
  '"$VCMD"' > vasp.out 2>&1
'
echo D5A_RATTLE_VASP_DONE
