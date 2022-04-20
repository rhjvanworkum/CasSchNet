import torch
import schnetpack as spk
import schnetpack.transform as trn
from ase import io
import numpy as np
import os
import shutil

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

def write_coeffs_to_orb_file(coeffs: np.ndarray, input_file_path: str, output_file_path: str) -> None:
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
    for i in range(1, 37):
      lines.append(f'* ORBITAL \t 1 \t {i} \n')
      lines.append(numpy_to_string(coeffs[(i-1)*36:i*36]))

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

def predict_orbital_coeffs(geometry_path: str, model_path: str) -> np.ndarray:
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

if __name__ == "__main__":
  cutoff = 5.0

  geometry_path = 'C:/Users/rhjva/imperial/sharc_files/run02/config_200/geom.xyz'
  example_orb_file = 'C:/Users/rhjva/imperial/sharc_files/run02/config_200/fulvene.Orb'

  model_path = './checkpoints/fulvene_wigner_dist_200_02_painn.pt'
  dir = 'C:/Users/rhjva/imperial/molcas_files/wigner_dist_200/config_200_02_painn/'

  if not os.path.exists(dir):
    os.makedirs(dir)
  if not os.path.exists(dir + 'CASSCF/'):
    os.makedirs(dir + 'CASSCF/')
    shutil.copy2(geometry_path, dir + 'CASSCF/geom.xyz')
    shutil.copy2(example_orb_file, dir + 'CASSCF/geom.orb')
    shutil.copy2('./CASSCF.input', dir + 'CASSCF/CASSCF.input')
  if not os.path.exists(dir + 'CASCI_ML/'):
    os.makedirs(dir + 'CASCI_ML/')
    shutil.copy2(geometry_path, dir + 'CASCI_ML/geom.xyz')
    shutil.copy2('./CASCI_ML.input', dir + 'CASCI_ML/CASCI_ML.input')
  if not os.path.exists(dir + 'CASSCF_ML/'):
    os.makedirs(dir + 'CASSCF_ML/')
    shutil.copy2(geometry_path, dir + 'CASSCF_ML/geom.xyz')
    shutil.copy2('./CASSCF_ML.input', dir + 'CASSCF_ML/CASSCF_ML.input')

  coeffs = predict_orbital_coeffs(geometry_path=geometry_path,
                                  model_path=model_path)
  
  for output_orb_file in [dir + 'CASCI_ML/geom.orb', dir + 'CASSCF_ML/geom.orb']:
    write_coeffs_to_orb_file(
      coeffs=coeffs, 
      input_file_path=example_orb_file,
      output_file_path=output_orb_file)