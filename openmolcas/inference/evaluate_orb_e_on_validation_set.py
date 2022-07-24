import os
import shutil
import h5py
import subprocess
import numpy as np
from openmolcas.inference.evaluate_iterations_on_validation_set import METHODS
from openmolcas.inference.infer_schnet import predict_guess, predict_guess_F, predict_guess_rotating

from openmolcas.utils import read_in_orb_file, write_coeffs_to_orb_file

def CASCI_calculation(dir_path, geom_file, initial_guess_file):
  # copy files there
  shutil.copy2('./openmolcas/calculation/input_files/CASCI_ML.input', dir_path + 'CASCI_ML.input')
  shutil.copy2(geom_file, dir_path + 'geom.xyz')
  shutil.copy2(initial_guess_file, dir_path + 'geom.orb')

  # run openmolcas
  subprocess.run('cd ' + dir_path + ' && sudo /opt/OpenMolcas/pymolcas CASCI_ML.input > calc.log', shell=True)

def CASSCF_ML_calculation(dir_path, geom_file, initial_guess_file):
  # copy files there
  shutil.copy2('./openmolcas/calculation/input_files/CASSCF.input', dir_path + 'CASSCF.input')
  shutil.copy2(geom_file, dir_path + 'geom.xyz')
  shutil.copy2(initial_guess_file, dir_path + 'geom.orb')

  # run openmolcas
  subprocess.run('cd ' + dir_path + ' && sudo /opt/OpenMolcas/pymolcas CASSCF.input > calc.log', shell=True)


def calculate_orbital_energies_mae(method_name, geom_file, guess_orbs, S, index):
  if method_name not in METHODS:
    raise ValueError("method name not found")

  elif method_name == 'ML_MO':
    initial_guess = predict_guess(model_path=model_path, geometry_path=geom_file, basis=n_mo)
    write_coeffs_to_orb_file(initial_guess.flatten(), example_guess_file, 'temp.orb', n=n_mo)
  elif method_name == 'ML_U':
    initial_guess = predict_guess_rotating(model_path=model_path, geometry_path=geom_file, guess_orbs=guess_orbs, basis=n_mo)
    write_coeffs_to_orb_file(initial_guess.flatten(), example_guess_file, 'temp.orb', n=n_mo)
  elif method_name == 'ML_F':
    initial_guess = predict_guess_F(model_path=model_path, geometry_path=geom_file, S=S, basis=n_mo)
    write_coeffs_to_orb_file(initial_guess.flatten(), example_guess_file, 'temp.orb', n=n_mo)
    
  casci_dir = './temp/CASCI/'
  if not os.path.exists(casci_dir):
    os.makedirs(casci_dir)
  CASCI_calculation(casci_dir, geom_file, initial_guess_file='temp.orb')
  _, ml_orb_energies = read_in_orb_file(casci_dir + 'CASCI_ML.RasOrb')
  shutil.rmtree(casci_dir)

  casscf_dir = './temp/CASSCF/'
  if not os.path.exists(casscf_dir):
    os.makedirs(casscf_dir)
  CASSCF_ML_calculation(casscf_dir, geom_file, initial_guess_file='temp.orb')
  _, conv_orb_energies = read_in_orb_file(casscf_dir + 'CASSCF.RasOrb')
  shutil.rmtree(casscf_dir)

  return np.abs(conv_orb_energies - ml_orb_energies)


if __name__ == "__main__":
  base_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  calc_dir = '/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200_ANO-S-MB/'
  split_file = 'data/geom_scan_200_molcas.npz'
  example_guess_file = 'openmolcas/calculation/input_files/geom.orb'
  
  method_name = 'ML_MO'
  model_path = 'checkpoints/gs199_molcas_ANO-S-MB_ML_MO.pt'
  n_mo = 36

  if not os.path.exists('./temp/'):
    os.makedirs('./temp')

  indices = np.load(split_file)['val_idx']
  
  orb_maes = []
  for idx, index in enumerate(indices):
    geom_file = base_dir + 'geometry_' + str(index) + '.xyz'
    guess_orbs, _ = read_in_orb_file(calc_dir + 'geometry_' + str(index) + '/CASSCF.GssOrb')
    S = h5py.File(calc_dir + 'geometry_' + str(index) + '/CASSCF.rasscf.h5').get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_mo)
    orb_mae = calculate_orbital_energies_mae(method_name, geom_file, guess_orbs, S, index)
    orb_maes.append(orb_mae)

    print('progress: ', idx / len(indices) * 100, ' %')

  shutil.rmtree('./temp')

  orb_maes = np.array(orb_maes)
  mean_orbs_meas = np.mean(orb_maes, axis=0)
  print(mean_orbs_meas)