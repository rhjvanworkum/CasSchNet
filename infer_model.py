import torch
import schnetpack as spk
import schnetpack.transform as trn
from ase import io
import numpy as np

cutoff = 5
n_coeffs = 36 * 36
model_path = './checkpoints/fulvene_wigner_dist_200.pt'
example_path = 'C:/Users/rhjva/imperial/sharc_files/run00/config_200/geom.xyz'

def write_coeffs_to_orb_file(coeffs: np.ndarray, file_path: str) -> None:
  print(coeffs.shape)
  return None

""" Transform the example into torch input batch """
atoms = io.read(example_path)
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
write_coeffs_to_orb_file(output['orbital_coeffs'].detach().numpy()[0], 'test.orb')