import sys
# sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')
from pyscf import gto, mcscf
from pyscf.tools import molden

import os
import numpy as np

from infer_schnet import predict_guess_F
from casscf_calculations import casscf_calculation

def save_molden(file_name, scf, mol):
  with open(file_name, 'w') as f1:
    molden.header(mol, f1)
    molden.orbital_coeff(mol, f1, scf.mo_coeff, ene=scf.mo_energy, occ=scf.mo_occ)

if __name__ == "__main__":
  model_path = '../../checkpoints/geom_scan_200_casscf_sto3g_hamiltonian_mse.pt'
  n_mo = 36
  cutoff = 5.0

  base_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  split_file = '../../data/geom_scan_200.npz'

  indices = np.load(split_file)['val_idx']

  index = indices[0]

  geom_file = base_dir + 'geometry_' + str(index) + '.xyz'
  initial_guess = predict_guess_F(model_path=model_path, geometry_path=geom_file)
  np.save('mo.npy', initial_guess)

  # molecule
  fulvene = gto.M(atom=geom_file,
                basis="sto-3g",
                spin=0,
                symmetry=True)

  # ML guess orbs
  myhf = fulvene.RHF()
  myhf.kernel()
  n_states = 2
  weights = np.ones(n_states) / n_states
  mcas = myhf.CASSCF(ncas=6, nelecas=6).state_average(weights)
  mo = mcscf.project_init_guess(mcas, initial_guess)
  mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)
  mcas.mo_coeffs = mo
  save_molden('ML_orbs.molden', mcas, fulvene)

  # converged orbs
  myhf = fulvene.RHF()
  myhf.kernel()
  n_states = 2
  weights = np.ones(n_states) / n_states
  mcas = myhf.CASSCF(ncas=6, nelecas=6).state_average(weights)
  mo = mcscf.project_init_guess(mcas, initial_guess)
  mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)
  mcas.kernel(mo)
  save_molden('converged_orbs.molden', mcas, fulvene)
