import numpy as np
import os
import shutil
import subprocess
import h5py
from infer_schnet import predict_guess, predict_guess_F, predict_guess_rotating
# from openmolcas.utils import read_in_orb_file, write_coeffs_to_orb_file, read_log_file

import numpy as np

def read_in_orb_file(orb_file : str):
  orbitals = []
  energies = []

  append = False

  with open(orb_file, 'r') as file:
    # check for the ORB keyword in RasOrb file
    while True:
      line = file.readline()
      if line[:4] == "#ORB":
        break

    # construct orbitals
    while True:
      line = file.readline()
      # end of block
      if '#' in line:
        break
      # add new orbital
      elif '* ORBITAL' in line:
        orbitals.append([])
      # add coeffs
      else:
        for coeff in line.split(' '):
          if len(coeff) > 0:
            orbitals[-1].append(float(coeff.replace('\n', '')))

    # get energies
    while True:
      line = file.readline()
      # end of block
      if append and '#' in line:
        break
      # append energies
      if append:
        for coeff in line.split(' '):
          if len(coeff) > 0:
            energies.append(float(coeff.replace('\n', '')))
      # begin of block
      if '* ONE ELECTRON ENERGIES' in line:
        append = True

  return np.array(orbitals), np.array(energies)

def get_orbital_occupations(orb_file: str):
  occupations = []

  with open(orb_file, 'r') as file:
    # check for the ORB keyword in RasOrb file
    while True:
      line = file.readline()
      if line[:4] == "#OCC":
        break

    # construct orbitals
    while True:
      line = file.readline()
      # end of block
      if '#' in line:
        break
      elif '* OCCUPATION NUMBERS' in line:
        continue
      else:
        for occ in line.split(' '):
          print
          if len(occ) > 0:
            occupations.append(float(occ.replace('\n', '')))

  return np.array(occupations)

def numpy_to_string(array: np.ndarray) -> str:
  string = ''

  for idx, elem in enumerate(array):
    if elem < 0:
      string += ' '
    else:
      string += '  '
    
    string += '%.14E' % elem

    if (idx + 1) % 5 == 0:
      string += '\n'
    if (idx + 1) == len(array):
      string += '\n'

  return string

def write_coeffs_to_orb_file(coeffs: np.ndarray, input_file_path: str, output_file_path: str, n: int) -> None:
  lines = []

  with open(input_file_path, 'r') as f:
    # add initial lines
    while True:
      line = f.readline()
      if '* ORBITAL' not in line:
        lines.append(line)
      else:
        break

    # add orbitals
    for i in range(1, n+1):
      lines.append(f'* ORBITAL \t 1 \t {i} \n')
      lines.append(numpy_to_string(coeffs[(i-1)*n:i*n]))

    append = False
    while True:
      line = f.readline()
      if '#OCC' in line:
        append = True

      if append:
        lines.append(line)

      if line == '':
        break
      
  with open(output_file_path, 'w+') as f:
    f.writelines(lines)


def read_log_file(file, read_iterations=True):
  n_iterations = None
  rasscf_timing = None
  wall_timing = None

  with open(file, 'r') as f:
    while True:
      line = f.readline()

      if read_iterations:
        if "Convergence after" in line:
          for el in line.split():
            if el.isdigit():
              n_iterations = int(el)
              break

      if "--- Module rasscf spent" in line:
        for el in line.split():
          if el.isdigit():
            rasscf_timing = float(el)
            break

      if "Timing: Wall" in line:
        wall_timing = float(line.replace("=", " ").split()[2])
        break

  return rasscf_timing, wall_timing, n_iterations







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
  base_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  calc_dir = '/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200_ANO-S-MB/'
  split_file = 'data/geom_scan_200_molcas.npz'
  example_guess_file = 'openmolcas/calculation/input_files/geom.orb'

  # experiment specific stuff
  method_name = 'ML_F'
  model_path = 'checkpoints/gs200_molcas_ANO-S-MB_' + method_name + '.pt'
  output_dir = '/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/mol-runs/gs200_ANOSMB_' + method_name + '/'
  n_mo = 36

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  indices = np.load(split_file)['val_idx']
  
  for idx, index in enumerate(indices):
    geom_file = base_dir + 'geometry_' + str(index) + '.xyz'
    guess_orbs, _ = read_in_orb_file(calc_dir + 'geometry_' + str(index) + '/CASSCF.GssOrb')
    S = h5py.File(calc_dir + 'geometry_' + str(index) + '/CASSCF.rasscf.h5').get('AO_OVERLAP_MATRIX')[:].reshape(n_mo, n_mo)
    run_molcas_calculations(method_name, geom_file, guess_orbs, S, index)

    print('progress: ', idx / len(indices) * 100, ' %')