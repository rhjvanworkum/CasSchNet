import os
import shutil
from typing import List
from db.generate_split_files import generate_split
from models.inference import predict_guess_F, predict_guess_F_delta
import numpy as np
import h5py
import multiprocessing
from tqdm import tqdm

from db.save_molcas_calculations import save_molcas_calculations_to_db
from openmolcas.utils import read_log_file, write_coeffs_to_orb_file, read_in_orb_file
from models.training import train_model

METHODS = [
  'standard',
  'previous_geom',
  'ML_F'
]

def casscf_calculation(output_dir: str, index: int, 
                       geom_file: str, guess_file: str = None) -> int:
  """
  :param output_dir: path to folder to store openmolcas results
  :param index: geometry index for the geometry the calculation is performed on
  :param geom_file: path to geometry file
  :param guess_file: path to file containing guess orbitals
  """
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
  _, _, imacro = read_log_file(dir_path + 'calc.log')
  return imacro

def update_ml_model(current_model_path: str, geometry_base_dir: str,
                    calculations_base_dir: str, new_geometry_idxs: List[int], update_number: int, n_basis: int = 36) -> str:
  """
  Finetunes a ML model during MD trajectory with CASSCF results on new geometries
  :param current_model_path: path to current torch model

  """
  for dir in ['db/', 'split/', 'model/']:
    if not os.path.exists(calculations_base_dir + dir):
      os.makedirs(calculations_base_dir + dir)

  # save calculations to db
  db_path = calculations_base_dir + 'db/database_' + str(update_number) + '.db'
  save_molcas_calculations_to_db(geometry_base_dir, calculations_base_dir, new_geometry_idxs, db_path=db_path, n_basis=n_basis)
  # generate split
  split_file = calculations_base_dir + 'split/split_' + str(update_number) + '.npz'  
  generate_split(train_split=1.00, val_split=0.0, test_split=0.0, n=len(new_geometry_idxs), save_path=split_file)
  # train new model
  new_model_path = calculations_base_dir + 'model/model_' + str(update_number)
  train_model(
    initial_model_path=current_model_path,
    save_path=new_model_path,
    epochs=40,
    database_path=db_path,
    split_file=split_file
  )

  return new_model_path

def run_molcas_calculation(args):
  current_model_path, geometry_base_dir, geometry_idx, calculations_base_dir, working_dir, initial_model_path, example_guess_file, n_basis = args

  geometry_path = geometry_base_dir + 'geometry_' + str(geometry_idx) + '.xyz'
  # perform calculation
  initial_guess = predict_guess_F(model_path=current_model_path, 
                                  geometry_path=geometry_path, 
                                  S=h5py.File(calculations_base_dir + 'geometry_' + str(geometry_idx) + '/CASSCF.rasscf.h5', 'r').get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_basis),
                                  basis=n_basis)
  write_coeffs_to_orb_file(initial_guess.flatten(), example_guess_file, working_dir + 'temp_' + str(geometry_idx) + '.orb', n=n_basis)
  n_iterations = casscf_calculation(output_dir=working_dir, index=geometry_idx, geom_file=geometry_path, guess_file=working_dir + 'temp_' + str(geometry_idx) + '.orb')
  
  return geometry_idx, n_iterations

def run_md_trajectory_ml(geometry_base_dir: str, geometry_idxs: List[int], calculations_base_dir: str, working_dir: str,
                         initial_model_path: str, example_guess_file: str, n_update: int = 50, n_basis: int = 36) -> List[int]:
  total_steps = len(geometry_idxs)
  total_n_iterations = {}

  update_step = 0
  current_model_path = initial_model_path

  n_episodes = total_steps // n_update
  n_left_over = total_steps % n_update

  for idx, episode in enumerate(range(n_episodes)):
    # parallel calculations
    parallel_args = [(current_model_path, geometry_base_dir, idx, 
                      calculations_base_dir, working_dir,
                      initial_model_path, example_guess_file, n_basis) for idx in np.arange(idx*n_update, (idx+1)*n_update)]

    pool = multiprocessing.Pool(N_JOBS)
    for result in tqdm(pool.imap(run_molcas_calculation, parallel_args), total=len(parallel_args)):
      idx, n_iterations = result
      total_n_iterations[idx] = n_iterations

    # retrain model
    new_geometry_idxs = geometry_idxs[np.arange(total_steps)[update_step*n_update:(update_step+1)*n_update]]
    current_model_path = update_ml_model(current_model_path=current_model_path,
                                          geometry_base_dir=geometry_base_dir,
                                          calculations_base_dir=working_dir,
                                          new_geometry_idxs=new_geometry_idxs,
                                          update_number=update_step)
    update_step += 1

  # # leftover calculations
  # parallel_args = [(idx) for idx in np.arange(total_steps - n_left_over, total_steps)]
  # pool = multiprocessing.Pool(N_JOBS)
  # for result in tqdm(pool.imap(run_molcas_calculations, parallel_args), total=len(parallel_args)):
  #   idx, n_iterations = result
  #   total_n_iterations[idx] = n_iterations

  # put into list
  total_n_iterations = [total_n_iterations[idx] for idx in range(total_steps)]
  return total_n_iterations

def run_md_trajectory(geometry_base_dir: str, geometry_idxs: List[int], working_dir: str, initial_guess_file: str) -> List[int]:
  total_steps = len(geometry_idxs)
  total_n_iterations = []

  for step, geometry_idx in enumerate(geometry_idxs):
    # geometry_path
    geometry_path = geometry_base_dir + 'geometry_' + str(geometry_idx) + '.xyz'
    # perform calculation
    if step == 0:
      guess_file = initial_guess_file
    else:
      guess_file = working_dir + 'geometry_' + str(geometry_idxs[step - 1]) + '/CASSCF.RasOrb'
    n_iterations = casscf_calculation(output_dir=working_dir, index=geometry_idx, geom_file=geometry_path, guess_file=guess_file)
    total_n_iterations.append(n_iterations)

    print('progress: ', step / len(geometry_idxs) * 100, '%')
    
  return total_n_iterations


def run_md_trajectory_ml_delta(geometry_base_dir: str, geometry_idxs: List[int], calculations_base_dir: str, working_dir: str,
                         initial_model_path: str, example_guess_file: str, n_update: int = 50, n_basis: int = 36) -> List[int]:
  total_steps = len(geometry_idxs)
  total_n_iterations = []

  model_path = initial_model_path

  for step, geometry_idx in enumerate(geometry_idxs):
    # geometry_path
    geometry_path = geometry_base_dir + 'geometry_' + str(geometry_idx) + '.xyz'
    # perform calculation
    if step == 0:
      guess_file = initial_guess_file
    else:
      prev_orbs, prev_ener = read_in_orb_file(working_dir + 'geometry_' + str(geometry_idxs[step - 1]) + '/CASSCF.RasOrb')
      prev_S = h5py.File(working_dir + 'geometry_' + str(geometry_idxs[step - 1]) + '/CASSCF.rasscf.h5', 'r').get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_basis)
      curr_S = h5py.File(calculations_base_dir + 'geometry_' + str(geometry_idx) + '/CASSCF.rasscf.h5', 'r').get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_basis)
      prev_F = np.matmul(prev_S, np.matmul(prev_orbs.T, np.matmul(np.diag(prev_ener), np.linalg.inv(prev_orbs.T))))
      initial_guess = predict_guess_F_delta(model_path=model_path, 
                                    prev_geometry_path=geometry_base_dir + 'geometry_' + str(geometry_idxs[step - 1]) + '.xyz',
                                    curr_geometry_path=geometry_path,
                                    prev_F=prev_F,
                                    curr_S=curr_S,
                                    basis=n_basis)
      write_coeffs_to_orb_file(initial_guess.flatten(), example_guess_file, working_dir + 'temp_' + str(geometry_idx) + '.orb', n=n_basis)
      guess_file = working_dir + 'temp_' + str(geometry_idx) + '.orb'

    n_iterations = casscf_calculation(output_dir=working_dir, index=geometry_idx, geom_file=geometry_path, guess_file=guess_file)
    total_n_iterations.append(n_iterations)

    print('progress: ', step / len(geometry_idxs) * 100, '%')
    
  return total_n_iterations

if __name__ == "__main__":
  """
  Execute this script from root dir
  """
  prefix = '/home/ubuntu/'
  geometry_base_dir = prefix + 'fulvene/geometries/MD_trajectories_5_01/'
  geometry_idxs = np.arange(200)
  calculations_base_dir = prefix + 'fulvene/openmolcas_calculations/MD_trajectory_1/'
  working_dir = prefix + 'fulvene/openmolcas_calculations/MD_delta_test/'
  initial_model_path = prefix + 'schnetpack/checkpoints/md01_delta_test.pt'
  example_guess_file = prefix + 'schnetpack/openmolcas/calculation/input_files/geom.orb'
  initial_guess_file = example_guess_file
  n_update = 200
  n_basis = 36
  N_JOBS = 6

  """
  DONT FORGET TO CHANGE CASSCF.INPUT TO FILEORB=GEOM.ORB
  """

  if not os.path.exists(working_dir):
    os.makedirs(working_dir)

  # run ML MD trajectory
  # total_n_iterations = run_md_trajectory_ml(geometry_base_dir, geometry_idxs, calculations_base_dir, working_dir,
  #                      initial_model_path, example_guess_file, n_update, n_basis)

  # run ML delta MD trajectory
  total_n_iterations = run_md_trajectory_ml_delta(geometry_base_dir, geometry_idxs, calculations_base_dir, working_dir,
                       initial_model_path, example_guess_file, n_update, n_basis)

  # # run previous geom MD trajectory
  # total_n_iterations = run_md_trajectory(geometry_base_dir, geometry_idxs, working_dir, initial_guess_file)

  print(total_n_iterations)