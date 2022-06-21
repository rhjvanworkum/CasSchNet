import os
import numpy as np
import time

import sys
sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')
from pyscf import gto, mcscf, lib

if __name__ == "__main__":
  # make MOL
  mol = gto.M(atom="/mnt/c/users/rhjva/imperial/molcas_files/fulvene/fulvene.xyz",
              basis="6-31g",
              spin=0)
  
  # run HF
  scf = mol.RHF()
  scf.kernel()
  S = scf.get_ovlp(mol)

  # initiate CASSCF object
  n_states = 2
  weights = np.ones(n_states) / n_states
  mcas = scf.CASSCF(ncas=6, nelecas=6).state_average(weights)
  mcas.conv_tol = 1e-8

  # project initial guess
  mo = mcscf.project_init_guess(mcas, scf.mo_coeff)
  mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)

  # run CASSCF
  tstart = time.time()
  (imacro, _, _, _, mo_coeffs, _) = mcas.kernel(mo)
  t_tot = time.time() - tstart
  F = mcas.get_fock()

  # save CASSCF result
  np.savez('/mnt/c/users/rhjva/imperial/molcas_files/fulvene/fulvene.npz',
            t_tot=t_tot,
            imacro=imacro,
            mo_coeffs=mo_coeffs,
            guess=mo,
            S=S,
            F=F)
