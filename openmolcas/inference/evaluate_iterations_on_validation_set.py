import numpy as np
import os
import shutil
import subprocess
import multiprocessing
from tqdm import tqdm
import h5py
from models.inference import predict_guess, predict_guess_F, predict_guess_rotating
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
  if not os.path.exists(dir_path + 'temp/'):
    os.makedirs(dir_path + 'temp/')
  os.system('cd ' + dir_path + ' && WorkDir=./temp/ /opt/molcas/bin/pymolcas CASSCF.input > calc.log')
  shutil.rmtree(dir_path + 'temp/')

  # read results back
  t_tot, _, imacro = read_log_file(dir_path + 'calc.log')
  return imacro

def run_molcas_calculations(args):
  method_name, geom_file, model_path, guess_orbs, S, index = args

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
  base_dir = prefix + 'fulvene/geometries/MD_trajectories_5_01/'
  calc_dir = prefix + 'fulvene/openmolcas_calculations/MD_trajectories_05_01_random/'
  split_file = 'data/MD_trajectories_05_01_random.npz'
  example_guess_file = 'openmolcas/calculation/input_files/geom.orb'
  print_iterations = True

  #####
  #####
  # DONT FORGET TO CHANGE THE CASSCF.input FILE
  ####
  
  method_name = 'ML_F'
  model_path = 'checkpoints/wd200_molcas_ANO-S-MB_ML_F.pt'
  output_dir = prefix + 'fulvene/openmolcas_calculations/gs199_ANOSMB_' + method_name + '_MD_Traj_2/'
  n_mo = 36
  N_JOBS = 4

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  indices = np.load(split_file)['val_idx']

  parallel_args = []
  for idx, index in enumerate(indices):
    geom_file = base_dir + 'geometry_' + str(index) + '.xyz'
    guess_orbs, _ = read_in_orb_file(calc_dir + 'geometry_' + str(index) + '/CASSCF.GssOrb')
    S = h5py.File(calc_dir + 'geometry_' + str(index) + '/CASSCF.rasscf.h5').get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_mo)
    parallel_args.append((method_name, geom_file, model_path, guess_orbs, S, index))
    
  pool = multiprocessing.Pool(N_JOBS)
  for result in tqdm(pool.imap(run_molcas_calculations, parallel_args), total=len(parallel_args)):
    success, idx = result
    if not success:
      print('Calculation failed at index: ', idx)

  if print_iterations:
    iterations = [np.load(output_dir + 'geometry_' + str(index) + '.npz', allow_pickle=True)['imacro'] for index in indices]
    iterations = np.array(list(filter(lambda x: x != None, iterations)))
    print(method_name, np.mean(iterations), np.std(iterations))
