import os
import numpy as np
import time

import sys
sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')
from pyscf import gto, scf

if __name__ == "__main__":
  geometries_base_path = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  output_path = '/mnt/c/users/rhjva/imperial/fulvene/casscf_calculations/geom_scan_200/'
  n_geometries = 200

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  for i in range(n_geometries):

    # make MOL
    mol = gto.M(atom=geometries_base_path + "geometry_" + str(i) + ".xyz",
                basis="sto-6g",
                spin=0)

    # run HF
    myscf = mol.RHF()
    guess = scf.hf.init_guess_by_minao(mol)

    tstart = time.time()
    imacro, mo_coeff = myscf.kernel()
    t_tot = time.time() - tstart

    S = myscf.get_ovlp(mol)
    F = myscf.get_fock()

    # save HF result
    np.savez(output_path + 'geometry_' + str(i) + '.npz',
             t_tot=t_tot,
             imacro=imacro,
             mo_coeffs=mo_coeff,
             guess=guess,
             S=S,
             F=F)

    print(i / n_geometries * 100, ' %')