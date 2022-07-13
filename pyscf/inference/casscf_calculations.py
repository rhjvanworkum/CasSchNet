import sys
sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')

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

  if initial_guess == 'ao_min':
    myhf.kernel()
    guess = scf.hf.init_guess_by_minao(fulvene)
    # project initial guess
    mo = mcscf.project_init_guess(mcas, guess)
    mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)
  elif initial_guess == 'hf':
    myhf.kernel()
    guess = myhf.mo_coeff
    # project initial guess
    mo = mcscf.project_init_guess(mcas, myhf.mo_coeff)
    mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)
  else:
    # project initial guess
    guess = initial_guess
    mo = mcscf.project_init_guess(mcas, initial_guess)
    mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)

  tstart = time.time()
  (imacro, _, _, fcivec, mo_coeffs, _) = mcas.kernel(mo)
  t_tot = time.time() - tstart

  print(imacro)

  return t_tot, imacro, fcivec, mo_coeffs, guess, S


def run_pyscf_calculations(method_name, geom_file, index, model_path=None):
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
  prefix = '/mnt/c/users/rhjva/imperial/'
  base_dir = prefix + 'fulvene/geometries/geom_scan_200/'
  split_file = '../../data/geom_scan_200_aws.npz'
  
  # basises = ['sto_6g', '4-31G', '6-31G*']
  # n_mos = [36, 66, 96]

  basises = ['6-31G*']
  n_mos = [96]

  for n_mo, basis in zip(n_mos, basises):
    models = ['geom_scan_200_' + basis + '_ML_MO', 'geom_scan_200_' + basis + '_ML_F']
    if basis == '6-31G*':
      models = ['geom_scan_200_6-31Gstar_ML_MO', 'geom_scan_200_6-31Gstar_ML_F']

    method_names = ['ML_MO', 'ML_F']
    outputs = ['geom_scan_200_ML_MO_' + basis, 'geom_scan_200_ML_F_' + basis]
  
    for idx, (model, method_name, output) in enumerate(zip(models, method_names, outputs)):
      model_path = '../../checkpoints/pyscf-runs/' + model + '.pt'
      output_dir = prefix + 'fulvene/casscf_calculations/pyscf-runs/' + output + '/'

      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      indices = np.load(split_file)['val_idx']

      for i, index in enumerate(indices):
        geom_file = base_dir + 'geometry_' + str(index) + '.xyz'
        run_pyscf_calculations(method_name, geom_file, index, model_path=model_path)
        
        print(idx / len(models) * 100, '% total   ', i / len(indices) * 100, '% job')