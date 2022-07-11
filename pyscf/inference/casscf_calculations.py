import sys
sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')

from pyscf import gto, scf, mcscf

import numpy as np
import os
import time
from infer_schnet import predict_guess, predict_guess_rotating, predict_guess_F

def casscf_calculation(geom_file, initial_guess=None):

  fulvene = gto.M(atom=geom_file,
                basis="sto-6g",
                spin=0,
                symmetry=True)

  myhf = fulvene.RHF()
  S = myhf.get_ovlp(fulvene)

  # initiate CASSCF object
  n_states = 2
  weights = np.ones(n_states) / n_states
  mcas = myhf.CASSCF(ncas=6, nelecas=6).state_average(weights)
  mcas.conv_tol = 1e-8

  if initial_guess is not None:
    guess = initial_guess

    # project initial guess
    mo = mcscf.project_init_guess(mcas, initial_guess)
    mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)

    tstart = time.time()
    (imacro, _, _, fcivec, mo_coeffs, _) = mcas.kernel(guess)
    t_tot = time.time() - tstart
  else:
    myhf.kernel()
    guess = myhf.mo_coeff

    # project initial guess
    mo = mcscf.project_init_guess(mcas, myhf.mo_coeff)
    mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)

    tstart = time.time()
    (imacro, _, _, fcivec, mo_coeffs, _) = mcas.kernel(mo)
    t_tot = time.time() - tstart

  print(imacro)

  return t_tot, imacro, fcivec, mo_coeffs, guess, S

if __name__ == "__main__":
  model_path = '../../checkpoints/geom_scan_200_casscf_hamiltonian_mse.pt'
  n_mo = 36
  cutoff = 5.0

  base_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  split_file = '../../data/geom_scan_200.npz'
  output_dir = '/mnt/c/users/rhjva/imperial/fulvene/casscf_calculations/experiments/geom_scan_200_hamiltonian_mse_withoutprojection/'

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  indices = np.load(split_file)['val_idx']

  for index in indices:
    geom_file = base_dir + 'geometry_' + str(index) + '.xyz'

    """ HF GUESS """
    # prev_geom_guess = np.load('/mnt/c/users/rhjva/imperial/fulvene/casscf_calculations/geom_scan_200/geometry_' + str(index) + '.npz')['mo_coeffs']
    (t_tot, imacro, fcivec, mo_coeffs, guess, S) = casscf_calculation(geom_file=geom_file, initial_guess=guess)
    # save converged MO's
    np.savez(output_dir + 'geometry_' + str(index) + '.npz',
             t_tot=t_tot,
             imacro=imacro,
             fcivec=fcivec,
             mo_coeffs=mo_coeffs,
             guess=guess,
             S=S)

    
    """ ML GUESS """
    initial_guess = predict_guess_F(model_path=model_path, geometry_path=geom_file)
    (t_tot, imacro, fcivec, mo_coeffs, guess, S) = casscf_calculation(geom_file, initial_guess=initial_guess)
    # save converged MO's
    np.savez(output_dir + 'geometry_' + str(index) + '_ML.npz',
             t_tot=t_tot,
             imacro=imacro,
             fcivec=fcivec,
             mo_coeffs=mo_coeffs,
             guess=guess,
             S=S)    