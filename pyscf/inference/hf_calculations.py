import sys
sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')

from pyscf import gto, scf

import numpy as np
import os
import time
from infer_schnet import predict_guess, predict_guess_rotating, predict_guess_F

def hf_calculation(geom_file, initial_guess=None):

  fulvene = gto.M(atom=geom_file,
                basis="sto-6g",
                spin=0,
                symmetry=True)

  myhf = fulvene.RHF()
  S = myhf.get_ovlp(fulvene)

  if initial_guess is not None:
    occ = np.zeros(36)
    occ[:21] = 2
    dm1 = myhf.make_rdm1(mo_coeff=initial_guess, mo_occ=occ)
    guess = dm1

    tstart = time.time()
    imacro, mo_coeffs = myhf.kernel(dm0=dm1)
    t_tot = time.time() - tstart
  else:
    guess = scf.hf.init_guess_by_minao(fulvene)

    tstart = time.time()
    imacro, mo_coeffs = myhf.kernel()
    t_tot = time.time() - tstart

  print(imacro)

  dm_final = myhf.make_rdm1()
  return t_tot, imacro, dm_final, guess, S

if __name__ == "__main__":
  model_path = '../../checkpoints/geom_scan_200_hamiltonian_moe.pt'
  n_mo = 36
  cutoff = 5.0

  base_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  split_file = '../../data/geom_scan_200.npz'
  output_dir = '/mnt/c/users/rhjva/imperial/fulvene/hf_calculations/experiments/geom_scan_200_hamiltonian_moe/'

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  indices = np.load(split_file)['val_idx']

  for index in indices:
    geom_file = base_dir + 'geometry_' + str(index) + '.xyz'

    """ HF GUESS """
    (t_tot, imacro, dm_final, guess, S) = hf_calculation(geom_file=geom_file)
    # save converged MO's
    np.savez(output_dir + 'geometry_' + str(index) + '.npz',
             t_tot=t_tot,
             imacro=imacro,
             dm_final=dm_final,
             guess=guess,
             S=S)

    
    """ ML GUESS """
    initial_guess = predict_guess_F(model_path=model_path, geometry_path=geom_file)
    (t_tot, imacro, dm_final, guess, S) = hf_calculation(geom_file, initial_guess=initial_guess)
    # save converged MO's
    np.savez(output_dir + 'geometry_' + str(index) + '_ML.npz',
             t_tot=t_tot,
             imacro=imacro,
             dm_final=dm_final,
             guess=guess,
             S=S)    