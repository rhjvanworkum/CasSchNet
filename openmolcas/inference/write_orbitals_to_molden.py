import sys
sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')
from pyscf import gto, mcscf
from pyscf.tools import molden

import os
import shutil
import h5py
import numpy as np
import subprocess

from infer_schnet import predict_guess_F
from utils import write_coeffs_to_orb_file

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
  shutil.copy2('../calculation/input_files/CASSCF.input', dir_path + 'CASSCF.input')
  shutil.copy2(geom_file, dir_path + 'geom.xyz')
  shutil.copy2(initial_guess_file, dir_path + 'geom.orb')

  # run openmolcas
  subprocess.run('cd ' + dir_path + ' && sudo /opt/OpenMolcas/pymolcas CASSCF.input > calc.log', shell=True)

def normalise_rows(mat):
    '''Normalise each row of mat'''
    return np.array(tuple(map(lambda v: v / np.linalg.norm(v), mat)))

def order_orbitals(ref, target):
    '''Reorder target molecular orbitals according to maximum overlap with ref.
    Orbitals phases are also adjusted to match ref.'''
    Moverlap=np.dot(normalise_rows(ref),normalise_rows(target).T)
    orb_order=np.argmax(abs(Moverlap),axis=1)
    target = target[orb_order]

    for idx in range(target.shape[0]):
        if np.dot(ref[idx], target[idx]) < 0:
            target[idx] = -1 * target[idx]

    return target

def read_in_orb_file(orb_file : str):
  orbitals = []

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

  return orbitals

if __name__ == "__main__":
  model_path = '../../checkpoints/geom_scan_200_molcas_hamiltonian_mse.pt'
  initial_guess_file = '../calculation/input_files/geom.orb'
  n_mo = 36
  cutoff = 5.0

  base_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  split_file = '../../data/geom_scan_200.npz'

  indices = np.load(split_file)['val_idx']

  index = indices[0]

  geom_file = base_dir + 'geometry_' + str(index) + '.xyz'
  file = h5py.File('/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200/geometry_' + str(index) + '/CASSCF.rasscf.h5')
  S = file.get('AO_OVERLAP_MATRIX')[:].reshape(36, 36)
  F = file.get('AO_FOCKINT_MATRIX')[:].reshape(36, 36)
  initial_guess = predict_guess_F(model_path=model_path, geometry_path=geom_file, S=S, Fock=F)

  # # # sort and order orbitals here
  # ref = read_in_orb_file('../calculation/input_files/geom.orb')
  # initial_guess = order_orbitals(ref, initial_guess)

  # initial_guess = np.load('../../pyscf/inference/mo.npy').T

  # e_s, U = np.linalg.eig(S)
  # diag_s = np.diag(e_s ** -0.5)
  # X = np.dot(U, np.dot(diag_s, U.T))

  # F_prime = np.dot(X.T, np.dot(F, X))
  # evals_prime, C_prime = np.linalg.eig(F_prime)
  # indices = evals_prime.argsort()
  # C_prime = C_prime[:, indices]
  # C = np.dot(X, C_prime).T
  # initial_guess = C


  write_coeffs_to_orb_file(initial_guess.flatten(), initial_guess_file, 'temp.orb', n=36)

  CASCI_calculation(geom_file, 'temp.orb')
  CASSCF_ML_calculation(geom_file, 'temp.orb')
  CASSCF_calculation(geom_file, '../calculation/input_files/geom.orb')
