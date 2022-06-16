import numpy as np
import os
import shutil
import subprocess
import h5py
from infer_schnet import predict_guess, predict_guess_F
from utils import write_coeffs_to_orb_file, read_log_file

def casscf_calculation(index, geom_file, output_dir, initial_guess_file):
  # make temp dir
  dir_path = output_dir + 'geometry_' + str(index) + '/'
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

  # copy files there
  shutil.copy2('../calculation/input_files/CASSCF.input', dir_path + 'CASSCF.input')
  shutil.copy2(geom_file, dir_path + 'geom.xyz')
  shutil.copy2(initial_guess_file, dir_path + 'geom.orb')

  # run openmolcas
  subprocess.run('cd ' + dir_path + ' && sudo /opt/OpenMolcas/pymolcas CASSCF.input > calc.log', shell=True)

  # read results back
  t_tot, _, imacro = read_log_file(dir_path + 'calc.log')
  file = h5py.File(dir_path + 'CASSCF.rasscf.h5')
  fcivec = file.get('CI_VECTORS')[:]
  mo_coeffs = file.get('MO_VECTORS')[:].reshape(-1, 36)
  S = file.get('AO_OVERLAP_MATRIX')[:].reshape(-1, 36)

  return t_tot, imacro, fcivec, mo_coeffs, S

if __name__ == "__main__":
  model_path = '../../checkpoints/geom_scan_200_molcas_hamiltonian_mse.pt'
  initial_guess_file = '../calculation/input_files/geom.orb'
  n_mo = 36
  cutoff = 5.0

  base_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  split_file = '../../data/geom_scan_200.npz'
  output_dir = '/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/experiments/geom_scan_200_hamiltonian_mse/'

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  indices = np.load(split_file)['val_idx']

  for index in indices:
    geom_file = base_dir + 'geometry_' + str(index) + '.xyz'

    """ HF GUESS """
    (t_tot, imacro, fcivec, mo_coeffs, S) = casscf_calculation(index, geom_file, output_dir, initial_guess_file)
    # save converged MO's
    np.savez(output_dir + 'geometry_' + str(index) + '.npz',
             t_tot=t_tot,
             imacro=imacro,
             fcivec=fcivec,
             mo_coeffs=mo_coeffs,
             S=S)

    
    """ ML GUESS """
    # make initial guess file
    S = h5py.File('/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200/geometry_' + str(index) + '/CASSCF.rasscf.h5').get('AO_OVERLAP_MATRIX')[:].reshape(-1, 36)
    initial_guess = predict_guess_F(model_path=model_path, geometry_path=geom_file, S=S)
    write_coeffs_to_orb_file(initial_guess.flatten(), initial_guess_file, 'temp.orb', n=36)


    (t_tot, imacro, fcivec, mo_coeffs, S) = casscf_calculation(index, geom_file, output_dir, 'temp.orb')
    # save converged MO's
    np.savez(output_dir + 'geometry_' + str(index) + '_ML.npz',
             t_tot=t_tot,
             imacro=imacro,
             fcivec=fcivec,
             mo_coeffs=mo_coeffs,
             S=S)    