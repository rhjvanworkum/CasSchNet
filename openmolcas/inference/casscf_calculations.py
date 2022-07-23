import numpy as np
import os
import shutil
import subprocess
import h5py
from infer_schnet import predict_guess, predict_guess_F, predict_guess_rotating
from openmolcas.utils import read_in_orb_file, write_coeffs_to_orb_file, read_log_file
import tempfile
import multiprocessing
from tqdm import tqdm

import numpy as np

CUTOFF = 5.0

METHODS = [
  # 'standard',
  'ML_MO',
  'ML_U',
  'ML_F'
]

def casscf_calculation(index, geom_file, guess_file=None):
  # make temp dir
  dir_path = output_dir + 'geometry_' + str(index) + '/'
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

  # copy files there
  shutil.copy2(geom_file, dir_path + 'geom.xyz')
  shutil.copy2('openmolcas/calculation/input_files/CASSCF.input', dir_path + 'CASSCF.input')
  shutil.copy2(guess_file, dir_path + 'geom.orb')

  # execute OpenMolcas
  os.system('cd ' + dir_path + ' && sudo' + ' Project=geom' + str(index) + ' WorkDir=/tmp/geom' + str(index) + '/ /home/ubuntu/build/pymolcas CASSCF.input > calc.log')
  os.system('sudo rm -r /tmp/geom' + str(index) + '/')

  # read results back
  t_tot, _, imacro = read_log_file(dir_path + 'calc.log')
  file = h5py.File(dir_path + 'geom' + str(index) + '.rasscf.h5')
  fcivec = file.get('CI_VECTORS')[:]
  mo_coeffs = file.get('MO_VECTORS')[:].reshape(-1, 174)
  S = file.get('AO_OVERLAP_MATRIX')[:].reshape(-1, 174)

  return t_tot, imacro, fcivec, mo_coeffs, S

def run_molcas_calculations(args):
  method_name, geom_file, model_path, guess_orbs, S, index = args
  if method_name not in METHODS:
    raise ValueError("method name not found")

  elif method_name == 'ML_MO':
    initial_guess = predict_guess(model_path=model_path, geometry_path=geom_file, basis=n_mo)
    tmpfile = tempfile.NamedTemporaryFile(suffix='.orb')
    write_coeffs_to_orb_file(initial_guess.flatten(), example_guess_file, tmpfile.name, n=n_mo)
    (t_tot, imacro, fcivec, mo_coeffs, S) = casscf_calculation(index, geom_file, guess_file=tmpfile.name)
  elif method_name == 'ML_U':
    initial_guess = predict_guess_rotating(model_path=model_path, geometry_path=geom_file, guess_orbs=guess_orbs, basis=n_mo)
    tmpfile = tempfile.NamedTemporaryFile(suffix='.orb')
    write_coeffs_to_orb_file(initial_guess.flatten(), example_guess_file, tmpfile.name, n=n_mo)
    (t_tot, imacro, fcivec, mo_coeffs, S) = casscf_calculation(index, geom_file, guess_file=tmpfile.name)
  elif method_name == 'ML_F':
    initial_guess = predict_guess_F(model_path=model_path, geometry_path=geom_file, S=S, basis=n_mo)
    print(initial_guess)
    write_coeffs_to_orb_file(initial_guess.flatten(), example_guess_file, 'temp.orb', n=n_mo)
    (t_tot, imacro, fcivec, mo_coeffs, S) = casscf_calculation(index, geom_file, guess_file='temp.orb')
  
  np.savez(output_dir + 'geometry_' + str(index) + '.npz',
          t_tot=t_tot,
          imacro=imacro,
          fcivec=fcivec,
          mo_coeffs=mo_coeffs,
          guess=initial_guess,
          S=S)

  return True, index

if __name__ == "__main__":
  prefix = '/home/ubuntu/'
  base_dir = prefix + 'fulvene/geometries/geom_scan_200/'
  calc_dir = prefix + 'fulvene/openmolcas_calculations/geom_scan_200_ANO-L-VTZ/'
  split_file = 'data/geom_scan_200_molcas.npz'
  example_guess_file = 'openmolcas/calculation/input_files/geom_2.orb'

  #####
  #####
  # DONT FORGET TO CHANGE THE CASSCF.input FILE
  ####

  # experiment specific stuff  
  method_name = 'ML_MO'
  model_path = 'checkpoints/gs199_molcas_ANO-L-VTZ_ML_MO.pt'
  output_dir = prefix + 'fulvene/openmolcas_calculations/mol-runs/gs199_ANOLVDZ_' + method_name + '/'
  n_mo = 174
  N_JOBS = 4

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  indices = np.load(split_file)['val_idx']
  
  parallel_args = []
  for idx, index in enumerate(indices):
    geom_file = base_dir + 'geometry_' + str(index) + '.xyz'
    guess_orbs, _ = read_in_orb_file(calc_dir + 'geometry_' + str(index) + '/geom' + str(index) + '.GssOrb')
    S = h5py.File(calc_dir + 'geometry_' + str(index) + '/geom' + str(index) + '.rasscf.h5').get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_mo)
    parallel_args.append((method_name, geom_file, model_path, guess_orbs, S, index))
    
  pool = multiprocessing.Pool(N_JOBS)
  for result in tqdm(pool.imap(run_molcas_calculations, parallel_args), total=len(parallel_args)):
    success, idx = result
    if not success:
      print('Calculation failed at index: ', idx)