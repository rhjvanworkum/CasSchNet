import numpy as np
import os
import shutil
import subprocess
import multiprocessing
from tqdm import tqdm
import h5py
from models.inference import predict_guess, predict_guess_F, predict_guess_rotating
from openmolcas.utils import get_s1_energy, get_s2_energy, read_in_orb_file, write_coeffs_to_orb_file, read_log_file
import numpy as np

CUTOFF = 5.0

METHODS = [
  # 'standard',
  'ML_MO',
  'ML_U',
  'ML_F'
]

def CASCI_calculation(index, geom_file, guess_file):
  # make temp dir
  dir_path = output_dir + 'geometry_' + str(index) + '/'
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

  # copy files there
  shutil.copy2(geom_file, dir_path + 'geom.xyz')
  shutil.copy2('openmolcas/calculation/input_files/CASCI_ML.input', dir_path + 'CASCI_ML.input')
  shutil.copy2(guess_file, dir_path + 'geom.orb')

  # run openmolcas
  if not os.path.exists(dir_path + 'temp/'):
    os.makedirs(dir_path + 'temp/')
  os.system('cd ' + dir_path + ' && WorkDir=./temp/ /opt/molcas/bin/pymolcas CASCI_ML.input > calc.log')
  shutil.rmtree(dir_path + 'temp/')

  # read results back
  s1_energy = get_s1_energy(dir_path + 'calc.log')
  s2_energy = get_s2_energy(dir_path + 'calc.log')
  return (s1_energy, s2_energy)

def run_molcas_calculations(args):
  method_name, geom_file, model_path, guess_orbs, S, index = args

  if method_name not in METHODS:
    raise ValueError("method name not found")

  elif method_name == 'ML_MO':
    initial_guess = predict_guess(model_path=model_path, geometry_path=geom_file, basis=n_mo)
    write_coeffs_to_orb_file(initial_guess.flatten(), example_guess_file, 'temp.orb', n=n_mo)
    energies = CASCI_calculation(index, geom_file, guess_file='temp.orb')
  elif method_name == 'ML_U':
    initial_guess = predict_guess_rotating(model_path=model_path, geometry_path=geom_file, guess_orbs=guess_orbs, basis=n_mo)
    write_coeffs_to_orb_file(initial_guess.flatten(), example_guess_file, 'temp.orb', n=n_mo)
    energies = CASCI_calculation(index, geom_file, guess_file='temp.orb')
  elif method_name == 'ML_F':
    initial_guess = predict_guess_F(model_path=model_path, geometry_path=geom_file, S=S, basis=n_mo)
    write_coeffs_to_orb_file(initial_guess.flatten(), example_guess_file, 'temp.orb', n=n_mo)
    energies = CASCI_calculation(index, geom_file, guess_file='temp.orb')
  
  return index, energies[0]

if __name__ == "__main__":
  prefix = '/mnt/c/users/rhjva/imperial/'
  base_dir = prefix + 'fulvene/geometries/geom_scan_200/'
  calc_dir = prefix + 'fulvene/openmolcas_calculations/geom_scan_200/'
  split_file = 'data/geom_scan_200_molcas.npz'
  example_guess_file = 'openmolcas/calculation/input_files/geom.orb'
  
  method_name = 'ML_MO'
  model_path = 'checkpoints/gs199_molcas_ANO-S-MB_ML_MO.pt'
  output_dir = prefix + 'fulvene/openmolcas_calculations/CASCI' + method_name + '_gs199/'
  n_mo = 36
  N_JOBS = 2

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  indices = np.load(split_file)['val_idx']

  true_energies = []

  parallel_args = []
  for idx, index in enumerate(indices):
    geom_file = base_dir + 'geometry_' + str(index) + '.xyz'
    guess_orbs, _ = read_in_orb_file(calc_dir + 'geometry_' + str(index) + '/CASSCF.GssOrb')
    S = h5py.File(calc_dir + 'geometry_' + str(index) + '/CASSCF.rasscf.h5').get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_mo)
    parallel_args.append((method_name, geom_file, model_path, guess_orbs, S, index))

    s1_energy = get_s1_energy(calc_dir + 'geometry_' + str(index) + '/calc.log')
    true_energies.append(s1_energy)
  

  pool = multiprocessing.Pool(N_JOBS)
  pred_energies = {}
  for result in tqdm(pool.imap(run_molcas_calculations, parallel_args), total=len(parallel_args)):
    idx, energy = result
    pred_energies[idx] = energy

  # pred_energies = []
  # for args in parallel_args:
  #   _, energy = run_molcas_calculations(args)
  #   pred_energies.append(energy)
  #   print('done')
  # pred_energies = np.array(pred_energies)

  true_energies = np.array(true_energies)
  pred_energies = np.array([pred_energies[idx] for idx in indices])

  print(np.mean(np.abs(true_energies - pred_energies)))
