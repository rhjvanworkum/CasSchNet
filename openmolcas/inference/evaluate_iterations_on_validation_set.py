import numpy as np
import os
import shutil
import subprocess
import h5py
from infer_schnet import predict_guess, predict_guess_F, predict_guess_rotating
from openmolcas.utils import read_in_orb_file, write_coeffs_to_orb_file, read_log_file
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

  # run openmolcas
  subprocess.run('cd ' + dir_path + ' && sudo /opt/OpenMolcas/pymolcas CASSCF.input > calc.log', shell=True)

  # read results back
  t_tot, _, imacro = read_log_file(dir_path + 'calc.log')
  file = h5py.File(dir_path + 'CASSCF.rasscf.h5')
  fcivec = file.get('CI_VECTORS')[:]
  mo_coeffs = file.get('MO_VECTORS')[:].reshape(-1, 36)
  S = file.get('AO_OVERLAP_MATRIX')[:].reshape(-1, 36)

  return t_tot, imacro, fcivec, mo_coeffs, S

def run_molcas_calculations(method_name, geom_file, guess_orbs, S, index):
  if method_name not in METHODS:
    raise ValueError("method name not found")

  elif method_name == 'ML_MO':
    initial_guess = predict_guess(model_path=model_path, geometry_path=geom_file, basis=n_mo)
    write_coeffs_to_orb_file(initial_guess.flatten(), example_guess_file, 'temp.orb', n=n_mo)
    (t_tot, imacro, fcivec, mo_coeffs, S) = casscf_calculation(index, geom_file, guess_file='temp.orb')
  elif method_name == 'ML_U':
    initial_guess = predict_guess_rotating(model_path=model_path, geometry_path=geom_file, guess_orbs=guess_orbs, basis=n_mo)
    write_coeffs_to_orb_file(initial_guess.flatten(), example_guess_file, 'temp.orb', n=n_mo)
    (t_tot, imacro, fcivec, mo_coeffs, S) = casscf_calculation(index, geom_file, guess_file='temp.orb')
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

if __name__ == "__main__":
  base_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/MD_trajectories_5_01/'
  calc_dir = '/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200_ANO-S-VDZ/'
  split_file = 'data/MD_trajectories_05_01_random.npz'
  example_guess_file = 'openmolcas/calculation/input_files/geom.orb'
  print_iterations = True

  #####
  #####
  # DONT FORGET TO CHANGE THE CASSCF.input FILE
  ####
  
  method_name = 'ML_MO'
  model_path = 'checkpoints/wd200_molcas_ANO-S-MB_ML_MO.pt'
  output_dir = '/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/molcas-runs/gs199_ANOSMB_' + method_name + '_MD_Traj/'
  n_mo = 36

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  indices = np.load(split_file)['val_idx']
  
  for idx, index in enumerate(indices):
    geom_file = base_dir + 'geometry_' + str(index) + '.xyz'
    # guess_orbs, _ = read_in_orb_file(calc_dir + 'geometry_' + str(index) + '/CASSCF.GssOrb')
    # S = h5py.File(calc_dir + 'geometry_' + str(index) + '/CASSCF.rasscf.h5').get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_mo)
    guess_orbs = None
    S = None
    run_molcas_calculations(method_name, geom_file, guess_orbs, S, index)

    print('progress: ', idx / len(indices) * 100, ' %')

  if print_iterations:
    iterations = [np.load(output_dir + 'geometry_' + str(index) + '.npz')['imacro'] for index in indices]
    print(method_name, np.mean(iterations), np.std(iterations))
