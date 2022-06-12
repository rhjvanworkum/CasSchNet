import sys
sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')

from pyscf import gto, scf

import numpy as np
import torch
import schnetpack as spk
import schnetpack.transform as trn
from ase import io
import scipy
import scipy.linalg

""" Predict guess using model infering MO coeffs """
def predict_guess(model_path, geometry_path, cutoff=5.0):
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  # transform geometry into input batch
  atoms = io.read(geometry_path)
  converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=cutoff), dtype=torch.float32, device=device)
  input = converter(atoms)
  # load model
  model = torch.load(model_path, map_location=device).to(device)
  model.eval()

  # infer the model
  output = model(input)
  values = output['mo_coeffs'].detach().cpu().numpy()[0]
  return values.reshape(-1, 36)

""" Predict guess using model infering Orbital rotations """
def predict_guess_rotating(model_path, geometry_path, cutoff=5.0):
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  # transform geometry into input batch
  atoms = io.read(geometry_path)
  converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=cutoff), dtype=torch.float32, device=device)
  input = converter(atoms)
  # load model
  model = torch.load(model_path, map_location=device).to(device)
  model.eval()

  # infer rotation matrix
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

  X = torch.zeros((36, 36), device=device)
  X[(x_ind, y_ind)] = values
  X = X - X.T

  U = scipy.linalg.expm(X.cpu().detach().numpy())

  # predict new guess
  fulvene = gto.M(atom=geometry_path,
                basis="sto-6g",
                spin=0,
                symmetry=True)
  guess = scf.hf.init_guess_by_minao(fulvene)
  return np.matmul(U, guess)

""" Predict guess using model infering the Hamiltonian matrix """
def predict_guess_F(model_path, geometry_path, cutoff=5.0):
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  # transform geometry into input batch
  atoms = io.read(geometry_path)
  converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=cutoff), dtype=torch.float32, device=device)
  input = converter(atoms)
  # load model
  model = torch.load(model_path, map_location=device).to(device)
  model.eval()


  # predicting Fock matrix
  output = model(input)
  values = output['F'].detach().cpu().numpy()[0]
  F = values.reshape(36, 36)

  # creating mol
  mol = gto.M(atom=geometry_path,
            basis="sto-6g",
            spin=0)
  myscf = mol.RHF()

  # F -> MO coeffs
  S = myscf.get_ovlp(mol)
  e_s, U = np.linalg.eig(S)
  diag_s = np.diag(e_s ** -0.5)
  X = np.dot(U, np.dot(diag_s, U.T))

  F_prime = np.dot(X.T, np.dot(F, X))
  evals_prime, C_prime = np.linalg.eig(F_prime)
  indices = evals_prime.argsort()
  C_prime = C_prime[:, indices]
  C = np.dot(X, C_prime)

  return C