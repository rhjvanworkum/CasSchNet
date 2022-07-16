import sys
# sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')

from pyscf import gto, scf, mcscf

import numpy as np
import os
import time
from infer_schnet import predict_guess, predict_guess_rotating, predict_guess_F

CUTOFF = 5.0

# cp -r /home/ubuntu/.local/lib/python3.6/site-packages/pyscf 

METHODS = [
  'ao_min',
  'hf',
  'ML_MO',
  'ML_U',
  'ML_F'
]

def casscf_calculation(geom_file, initial_guess='ao_min', basis='sto-6g'):

  fulvene = gto.M(atom=geom_file,
                basis=basis,
                spin=0,
                symmetry=True)

  myhf = fulvene.RHF()
  S = myhf.get_ovlp(fulvene)

  # initiate CASSCF object
  n_states = 2
  weights = np.ones(n_states) / n_states
  mcas = myhf.CASSCF(ncas=6, nelecas=6).state_average(weights)
  mcas.conv_tol = 1e-8
  
  myhf.kernel()

  if initial_guess == 'ao_min':
    guess = scf.hf.init_guess_by_minao(fulvene)
    # project initial guess
    mo = mcscf.project_init_guess(mcas, guess)
    mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)
  elif initial_guess == 'hf':
    guess = myhf.mo_coeff
    # project initial guess
    mo = mcscf.project_init_guess(mcas, myhf.mo_coeff)
    mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)
  else:
    guess = initial_guess
    # project initial guess
    mo = mcscf.project_init_guess(mcas, initial_guess)
    mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)

  tstart = time.time()
  (imacro, _, _, fcivec, mo_coeffs, _) = mcas.kernel(mo)
  t_tot = time.time() - tstart

  return t_tot, imacro, fcivec, mo_coeffs, guess, S


def run_pyscf_calculations(method_name, geom_file, index, basis):
  if method_name not in METHODS:
    raise ValueError("method name not found")
  
  if method_name == 'ao_min':
    (t_tot, imacro, fcivec, mo_coeffs, guess, S) = casscf_calculation(geom_file, initial_guess='ao_min', basis=basis)
  elif method_name == 'hf':
    (t_tot, imacro, fcivec, mo_coeffs, guess, S) = casscf_calculation(geom_file, initial_guess='hf', basis=basis)
  elif method_name == 'ML_MO':
    initial_guess = predict_guess(model_path=model_path, geometry_path=geom_file, n_mo=n_mo, basis=basis)
    (t_tot, imacro, fcivec, mo_coeffs, guess, S) = casscf_calculation(geom_file, initial_guess=initial_guess, basis=basis)
  elif method_name == 'ML_U':
    initial_guess = predict_guess_rotating(model_path=model_path, geometry_path=geom_file, n_mo=n_mo, basis=basis)
    (t_tot, imacro, fcivec, mo_coeffs, guess, S) = casscf_calculation(geom_file, initial_guess=initial_guess, basis=basis)
  elif method_name == 'ML_F':
    initial_guess = predict_guess_F(model_path=model_path, geometry_path=geom_file, n_mo=n_mo, basis=basis)
    (t_tot, imacro, fcivec, mo_coeffs, guess, S) = casscf_calculation(geom_file, initial_guess=initial_guess, basis=basis)
  
  np.savez(output_dir + 'geometry_' + str(index) + '.npz',
          t_tot=t_tot,
          imacro=imacro,
          fcivec=fcivec,
          mo_coeffs=mo_coeffs,
          guess=guess,
          S=S)
  
  

if __name__ == "__main__":
  base_dir = '/home/ubuntu/fulvene/geometries/geom_scan_200/'
  split_file = '../../data/geom_scan_200.npz'
  
  models = ['geom_scan_200_4-31G_ML_U', 'geom_scan_200_4-31G_ML_F'] # ['', '', 'geom_scan_200_4-31G_ML_MO', 'geom_scan_200_4-31G_ML_U' 'geom_scan_200_4-31G_ML_F']
  method_names = ['ML_U', 'ML_F'] #  ['ao_min', 'hf', 'ML_MO', 'ML_U', 'ML_F']
  outputs = ['geom_scan_200_ML_U_4-31G', 'geom_scan_200_ML_F_4-31G'] # ['geom_scan_200_ao_min_4-31G', 'geom_scan_200_hf_4-31G', 'geom_scan_200_ML_MO_4-31G', 'geom_scan_200_ML_U_4-31G', 'geom_scan_200_ML_F_4-31G']
  basis = '4-31G'
  n_mo = 66
  
  for idx, (model, method_name, output) in enumerate(zip(models, method_names, outputs)):
    model_path = '../../checkpoints/' + model + '.pt'
    output_dir = '/home/ubuntu/fulvene/casscf_calculations/experiments/' + output + '/'

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    indices = np.load(split_file)['val_idx']

    for i, index in enumerate(indices):
      geom_file = base_dir + 'geometry_' + str(index) + '.xyz'
      run_pyscf_calculations(method_name, geom_file, index, basis=basis)
      
      print(idx / len(models) * 100, '% total   ', i / len(indices) * 100, '% job')