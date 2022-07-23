import os
import shutil
import subprocess
import multiprocessing
from tqdm import tqdm
import time

"""
This Calculation is performed by taking a set of converged orbital coeffs from the 1 reference orbitals as an input
"""

# if __name__ == "__main__":
#   prefix = '/home/ubuntu/'
#   geometries_base_path = prefix + 'fulvene/geometries/geom_scan_200/'
#   output_path = prefix + 'fulvene/openmolcas_calculations/geom_scan_200_ANO-L-VTZ/'
#   n_geometries = 200

#   if not os.path.exists(output_path):
#     os.makedirs(output_path)

#   for i in range(1, n_geometries):
#     tstart = time.time()

#     # make dir
#     geometry_path = output_path + 'geometry_' + str(i) + '/'
#     if not os.path.exists(geometry_path):
#       os.makedirs(geometry_path)

#     # copy files
#     shutil.copy2('./input_files/CASSCF.input', geometry_path + 'CASSCF.input')
#     shutil.copy2('./input_files/geom.orb', geometry_path + 'geom.orb')
#     shutil.copy2(geometries_base_path + 'geometry_' + str(i) + '.xyz', geometry_path + 'geom.xyz')

#     # execute OpenMolcas
#     os.system('cd ' + geometry_path + ' && sudo /home/ubuntu/build/pymolcas CASSCF.input > calc.log')

#     print(time.time() - tstart)

#     print('Progress: ', i / n_geometries * 100, '%')

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
  os.system('cd ' + geometry_path + ' && sudo' + ' Project=geom' + str(geom_idx) + ' WorkDir=/tmp/geom' + str(geom_idx) + '/ /home/ubuntu/build/pymolcas CASSCF.input > calc.log')
  os.system('sudo rm -r /tmp/geom' + str(geom_idx) + '/')

  return True, geom_idx

if __name__ == "__main__":
  prefix = '/home/ubuntu/'
  geometries_base_path = prefix + 'fulvene/geometries/geom_scan_200/'
  output_path = prefix + 'fulvene/openmolcas_calculations/geom_scan_200_ANO-L-VTZ/'
  n_geometries = 200
  N_JOBS = 4

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  parallel_args = [(geometries_base_path, i) for i in range(n_geometries)]

  pool = multiprocessing.Pool(N_JOBS)
  for result in tqdm(pool.imap(run_molcas_calculation, parallel_args), total=len(parallel_args)):
    success, idx = result
    if not success:
      print('Calculation failed at index: ', idx)
