from openmolcas.utils import read_in_orb_file, write_coeffs_to_orb_file
from db.utils import correct_phase, order_orbitals

import os
import shutil
import h5py
import numpy as np
import subprocess
import scipy
from scipy.linalg import eigh

from infer_schnet import predict_guess_F

def CASCI_calculation(geom_file, initial_guess_file):
  # make temp dir
  dir_path = './CASCI/'
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

  # copy files there
  shutil.copy2('../calculation/input_files/CASCI_ML.input', dir_path + 'CASCI_ML.input')
  shutil.copy2(geom_file, dir_path + 'geom.xyz')
  shutil.copy2(initial_guess_file, dir_path + 'geom.orb')

  # run openmolcas
  subprocess.run('cd ' + dir_path + ' && sudo /opt/OpenMolcas/pymolcas CASCI_ML.input > calc.log', shell=True)

def CASSCF_ML_calculation(geom_file, initial_guess_file):
  # make temp dir
  dir_path = './CASSCF_ML/'
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

  # copy files there
  shutil.copy2('../calculation/input_files/CASSCF.input', dir_path + 'CASSCF.input')
  shutil.copy2(geom_file, dir_path + 'geom.xyz')
  shutil.copy2(initial_guess_file, dir_path + 'geom.orb')

  # run openmolcas
  subprocess.run('cd ' + dir_path + ' && sudo /opt/OpenMolcas/pymolcas CASSCF.input > calc.log', shell=True)

def CASSCF_calculation(geom_file, initial_guess_file):
  # make temp dir
  dir_path = './CASSCF/'
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

  # copy files there
  shutil.copy2('../calculation/input_files/CASSCF_GSS.input', dir_path + 'CASSCF.input')
  shutil.copy2(geom_file, dir_path + 'geom.xyz')
  shutil.copy2(initial_guess_file, dir_path + 'geom.orb')

  # run openmolcas
  subprocess.run('cd ' + dir_path + ' && sudo /opt/OpenMolcas/pymolcas CASSCF.input > calc.log', shell=True)

if __name__ == "__main__":
  model_path = '../../checkpoints/geom_scan_199_molcas_fock.pt'
  initial_guess_file = '../calculation/input_files/geom.orb'
  n_mo = 36
  cutoff = 5.0

  base_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  split_file = '../../data/geom_scan_200_molcas.npz'

  indices = np.load(split_file)['val_idx']
  # index = indices[0]
  index = 9
  geom_file = base_dir + 'geometry_' + str(index) + '.xyz'

  file = h5py.File('/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200_ANO-S-MB/geometry_' + str(index) + '/CASSCF.rasscf.h5')
  S = file.get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_mo)
  orbitals, energies = read_in_orb_file('/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200_ANO-S-MB/geometry_' + str(index) + '/CASSCF.RasOrb')


  assert np.allclose(np.matmul(orbitals.T, np.linalg.inv(orbitals.T)), np.eye(n_mo))

  # write_coeffs_to_orb_file(orbitals.flatten(), initial_guess_file, 'extracted.orb', n=n_mo)

  # S = S.astype(np.double)
  # orbitals = orbitals.astype(np.double)
  # energies = energies.astype(np.double)

  # # F = np.matmul(S, np.matmul(orbitals.T, np.linalg.solve(orbitals.T, np.diag(energies))))

  # eval, evec = np.linalg.eig(F)
  # a2 = np.dot(evec, eval[:,np.newaxis] * evec.T)
  # assert np.allclose(F, a2)


  matrices = [S, orbitals.T, np.diag(energies), np.linalg.inv(orbitals.T)]

  import itertools

  for set in itertools.permutations([0, 1, 2, 3], 4):
    print(set)
    F = np.matmul(matrices[set[0]], np.matmul(matrices[set[1]], np.matmul(matrices[set[2]], matrices[set[3]])))
    eigvals, eigvecs = scipy.linalg.eigh(F, S)
    print(np.mean(np.abs(eigvals - energies)))

  # F = np.matmul(S, np.matmul(orbitals.T, np.matmul(np.diag(energies), np.linalg.inv(orbitals.T))))
  # F = np.matmul(S, np.matmul(orbitals.T, np.linalg.solve(orbitals.T, np.diag(energies))))
  # F = np.matmul(np.linalg.inv(orbitals.T), np.matmul(np.diag(energies), np.matmul(S, orbitals.T)))

  eigvals, eigvecs = scipy.linalg.eigh(F, S)
  print(np.mean(np.abs(eigvals - energies)))

  # assert np.allclose(eigvecs.T, orbitals, atol=1e-2)

  # e_s, U = np.linalg.eig(S)
  # diag_s = np.diag(e_s ** -0.5)
  # X = np.dot(U, np.dot(diag_s, U.T))

  # F_prime = np.dot(X.T, np.dot(F, X))
  # evals_prime, C_prime = np.linalg.eig(F_prime)
  # indices = evals_prime.argsort()
  # C_prime = C_prime[:, indices]
  # initial_guess = np.dot(X, C_prime).T

  # initial_guess = order_orbitals(orbitals, initial_guess)

  # print(initial_guess - orbitals)

  write_coeffs_to_orb_file(eigvecs.T.flatten(), initial_guess_file, 'temp.orb', n=n_mo)

  CASCI_calculation(geom_file, 'temp.orb')
  CASSCF_ML_calculation(geom_file, 'temp.orb')
  CASSCF_calculation(geom_file, '../calculation/input_files/geom_2.orb')


  # """ METHOD 2: F = SCEC-1 """
  # F = np.matmul(S, np.matmul(orbitals, np.matmul(np.diag(energies), np.linalg.inv(orbitals))))

  # e_s, U = np.linalg.eig(S)
  # diag_s = np.diag(e_s ** -0.5)
  # X = np.dot(U, np.dot(diag_s, U.T))

  # F_prime = np.dot(X.T, np.dot(F, X))
  # evals_prime, C_prime = np.linalg.eig(F_prime)
  # indices = evals_prime.argsort()
  # C_prime = C_prime[:, indices]
  # C = np.dot(X, C_prime).T
