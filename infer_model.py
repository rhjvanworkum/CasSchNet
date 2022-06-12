from math import factorial
import torch
from calculate_evaluation import get_orbital_coeffs
from rmse import normalise_rows
import src.schnetpack as spk
from src.schnetpack.data.parser import read_in_orb_file
import src.schnetpack.transform as trn
from ase import io
import numpy as np
import os
import shutil
import subprocess
import math

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

def predict_orbital_coeffs(geometry_path: str, model_path: str, cutoff: float) -> np.ndarray:
  """ Transform the example into torch input batch """
  atoms = io.read(geometry_path)
  converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=cutoff), dtype=torch.float32)
  input = converter(atoms)

  """ Load the model """
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  model = torch.load(model_path, map_location=device).to(device)
  model.eval()

  """ Predict orbital & write to .Orb file """
  output = model(input)

  return output['orbital_coeffs'].detach().numpy()[0]

import scipy.linalg
def matrix_exp(X):
  # exp = torch.eye(n=X.shape[0])
  # k = 500

  # for i in range(1, k):
  #   exp += (1 / math.factorial(i)) * torch.matrix_power(X, i)

  # return exp
  return scipy.linalg.expm(X.numpy())


def predict_orbital_coeffs_rotation(geometry_path: str, model_path: str, cutoff: float, ref_file: str) -> np.ndarray:
  """ Transform the example into torch input batch """
  atoms = io.read(geometry_path)
  converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=cutoff), dtype=torch.float32)
  input = converter(atoms)

  """ Load the model """
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  model = torch.load(model_path, map_location=device).to(device)
  model.eval()

  output = model(input)
  values = output['orbital_coeffs'].detach()[0]

  ind = torch.triu_indices(36, 36)
  x_ind = ind[0].tolist()
  y_ind = ind[1].tolist()
  for idx, (x, y) in enumerate(zip(x_ind, y_ind)):
    if x == y:
      x_ind.pop(idx)
      y_ind.pop(idx)
  x_ind = torch.tensor(x_ind)
  y_ind = torch.tensor(y_ind)

  X = torch.zeros((36, 36))
  X[(x_ind, y_ind)] = values
  X = X - X.T

  # print(np.linalg.det(torch.linalg.matrix_exp(X)))
  # # U = torch.linalg.matrix_exp(X)
  # # print(np.matmul(U, np.conj(U).T))
  # U = matrix_exp(X)
  # print(np.linalg.det(U))

  # refs = torch.from_numpy(get_orbital_coeffs(ref_file)).double()
  # orbs = torch.matmul(torch.linalg.matrix_exp(X).double(), refs).flatten().numpy()
  refs = get_orbital_coeffs(ref_file)
  orbs = np.matmul(scipy.linalg.expm(X.numpy()), refs).flatten()

  return orbs


def prepare_molcas_calculation(model_path, geometry_file, orb_file, output_dir, n_mo, is_delta=False):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # shutil.rmtree(output_dir + 'CASCI_ML')
  # shutil.rmtree(output_dir + 'CASSCF_ML')

  if not os.path.exists(output_dir + 'CASSCF/'):
    os.makedirs(output_dir + 'CASSCF/')
    shutil.copy2(geometry_file, output_dir + 'CASSCF/geom.xyz')
    shutil.copy2(orb_file, output_dir + 'CASSCF/geom.orb')
    shutil.copy2('./CASSCF.input', output_dir + 'CASSCF/CASSCF.input')
  if not os.path.exists(output_dir + 'CASCI_ML/'):
    os.makedirs(output_dir + 'CASCI_ML/')
    shutil.copy2(geometry_file, output_dir + 'CASCI_ML/geom.xyz')
    shutil.copy2('./CASCI_ML.input', output_dir + 'CASCI_ML/CASCI_ML.input')
  if not os.path.exists(output_dir + 'CASSCF_ML/'):
    os.makedirs(output_dir + 'CASSCF_ML/')
    shutil.copy2(geometry_file, output_dir + 'CASSCF_ML/geom.xyz')
    shutil.copy2('./CASSCF_ML.input', output_dir + 'CASSCF_ML/CASSCF_ML.input')

  coeffs = predict_orbital_coeffs_rotation(geometry_path=geometry_file,
                                  model_path=model_path,
                                  cutoff=cutoff,
                                  ref_file=orb_file)

  if is_delta:
    ref_coeffs = np.array(read_in_orb_file(orb_file))
    coeffs += ref_coeffs.flatten()
  
  for output_orb_file in [output_dir + 'CASCI_ML/geom.orb', output_dir + 'CASSCF_ML/geom.orb']:
    write_coeffs_to_orb_file(
      coeffs=coeffs, 
      input_file_path=orb_file,
      output_file_path=output_orb_file,
      n=n_mo)

def run_molcas_calculations(output_dir):
  subprocess.run('cd ' + output_dir + 'CASCI_ML && sudo /opt/OpenMolcas/pymolcas CASCI_ML.input > calc.log', shell=True)
  subprocess.run('cd ' + output_dir + 'CASSCF_ML && sudo /opt/OpenMolcas/pymolcas CASSCF_ML.input > calc.log', shell=True)
  subprocess.run('cd ' + output_dir + 'CASSCF && sudo /opt/OpenMolcas/pymolcas CASSCF.input > calc.log', shell=True)

if __name__ == "__main__":
  model_path = './checkpoints/fulvene_scan_molcas_nocorr.pt'
  n_mo = 36
  cutoff = 5.0

  """ Preparing one calculation """
  # geometry_path = 'C:/Users/rhjva/imperial/molcas_files/fulvene_scan_2/geometry_10.xyz'
  # example_orb_file = 'C:/Users/rhjva/imperial/molcas_files/fulvene_scan_2/geometry_10/CASSCF/geom.orb'
  # dir = 'C:/Users/rhjva/imperial/molcas_files/test_molwise_tanh/'
  # prepare_molcas_calculation(
  #     model_path=model_path,
  #     geometry_file=geometry_path,
  #     orb_file=example_orb_file,
  #     output_dir=dir,
  #     n_mo=n_mo
  #   )

  """ Preparing a list of calculations - model output """
  # # split_file = './data/fulvene_MB_140.npz'
  # # base_dir = 'C:/Users/rhjva/imperial/sharc_files/run_MB/'
  # # output_dir = 'C:/Users/rhjva/imperial/molcas_files/wigner_dist_200/run_MB_140_molwise_concatenate/'

  # split_file = './data/fulvene_MB_140.npz'
  # base_dir = 'C:/Users/rhjva/imperial/molcas_files/fulvene_scan_2/'
  # output_dir = 'C:/Users/rhjva/imperial/molcas_files/fulvene_scan_molcas_nocorr/'

  # indices = np.load(split_file)['test_idx']

  # for index in indices:
  #   prepare_molcas_calculation(
  #     model_path=model_path,
  #     geometry_file=base_dir + 'geometry_' + str(index) + '/CASSCF/geom.xyz',
  #     orb_file="C:/Users/rhjva/imperial/molcas_files/fulvene_scan/geometry_" + str(index) + '/casscf/geom.Orb',
  #     output_dir=output_dir + 'geometry_' + str(index) + '/',
  #     n_mo=n_mo,
  #   )

  """ Running the calculations - model output """
  split_file = './data/fulvene_MB_140.npz'
  output_dir = '/mnt/c/users/rhjva/imperial/molcas_files/fulvene_scan_molcas_nocorr/'
  indices = np.load(split_file)['test_idx']

  for i, index in enumerate(indices):
    run_molcas_calculations(output_dir + 'geometry_' + str(index) + '/')
    print('Progress: ', i / len(indices) * 100, '%')

  """ Preparing a list of calculations - geometry scan """
  # output_dir = 'C:/Users/rhjva/imperial/molcas_files/fulvene_scan_7/'

  # for _, _, filenames in os.walk(output_dir):
  #     for filename in filenames:
  #       prepare_molcas_calculation(
  #         model_path=model_path,
  #         geometry_file=output_dir + filename,
  #         orb_file="C:/Users/rhjva/imperial/molcas_files/wigner_dist_200/config_200_02/CASSCF/geom.orb",
  #         output_dir=output_dir + filename.split('.')[0] + '/',
  #         n_mo=n_mo,
  #         is_delta=True
  #       )
  #     break

  """ Running the calculations - geometry scan """
  # output_dir = '/mnt/c/users/rhjva/imperial/molcas_files/fulvene_scan_7/'

  # for _, _, filenames in os.walk(output_dir):
  #   for idx, filename in enumerate(filenames):
  #     run_molcas_calculations(output_dir + filename.split('.')[0] + '/')
  #     print('Progress: ', idx / len(filenames) * 100, '%')
  #   break