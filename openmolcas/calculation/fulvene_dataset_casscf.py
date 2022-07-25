import os
import shutil
import subprocess
import multiprocessing
from tqdm import tqdm
import numpy as np

import time

def run_molcas_calculation(args):
  geometries_base_path, geom_idx = args

  # make dir
  geometry_path = output_path + 'geometry_' + str(geom_idx) + '/'
  if not os.path.exists(geometry_path):
    os.makedirs(geometry_path)

  # copy files
  shutil.copy2('./input_files/CASSCF.input', geometry_path + 'CASSCF.input')
  shutil.copy2('./input_files/geom.orb', geometry_path + 'geom.orb')
  shutil.copy2(geometries_base_path + 'geometry_' + str(geom_idx) + '.xyz', geometry_path + 'geom.xyz')

  # execute OpenMolcas
  if not os.path.exists(geometry_path + 'temp/'):
    os.makedirs(geometry_path + 'temp/')
  os.system('cd ' + geometry_path + ' && WorkDir=./temp/ /opt/molcas/bin/pymolcas CASSCF.input > calc.log')
  shutil.rmtree(geometry_path + 'temp/')

  return True, geom_idx

if __name__ == "__main__":
  prefix = '/home/ubuntu/'
  geometries_base_path = prefix + 'fulvene/geometries/MD_trajectories_5_01/'
  output_path = prefix + 'fulvene/openmolcas_calculations/MD_trajectory_1/'
  N_JOBS = 6

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  # indices = np.load(prefix + 'schnetpack/data/MD_trajectories_05_01_random.npz')['val_idx']
  # parallel_args = [()]

  n_geometries = 200
  parallel_args = [(geometries_base_path, i) for i in range(n_geometries)]

  pool = multiprocessing.Pool(N_JOBS)
  for result in tqdm(pool.imap(run_molcas_calculation, parallel_args), total=len(parallel_args)):
    success, idx = result
    if not success:
      print('Calculation failed at index: ', idx)
