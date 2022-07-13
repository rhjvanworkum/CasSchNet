import sys
# sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')

from pyscf import gto, scf, mcscf

import numpy as np
import torch
import schnetpack as spk
import schnetpack.transform as trn
from ase import io
import scipy
import scipy.linalg

""" Predict guess using model infering MO coeffs """
def predict_guess(model_path, geometry_path, n_mo, basis, cutoff=5.0):
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
  return values.reshape(-1, n_mo)

""" Predict guess using model infering Orbital rotations """
def predict_guess_rotating(model_path, geometry_path, n_mo, basis, cutoff=5.0):
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
  values = output['mo_coeffs'][0].reshape(n_mo, n_mo)
  X = 0.5 * (values - values.T)
  U = scipy.linalg.expm(X.cpu().detach().numpy())

  # predict new guess
  fulvene = gto.M(atom=geometry_path,
                basis=basis,
                spin=0,
                symmetry=True)

  scf = fulvene.RHF()
  scf.kernel()

  n_states = 2
  weights = np.ones(n_states) / n_states
  mcas = scf.CASSCF(ncas=6, nelecas=6).state_average(weights)
  mcas.conv_tol = 1e-8

  # project initial guess
  mo = mcscf.project_init_guess(mcas, scf.mo_coeff)
  mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)

  return np.matmul(U, mo)

""" Predict guess using model infering the Hamiltonian matrix """
def predict_guess_F(model_path, geometry_path, n_mo, basis, cutoff=5.0):
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
  F = values.reshape(n_mo, n_mo)
  F = 0.5 * (F + F.T)

  # creating mol
  mol = gto.M(atom=geometry_path,
            basis=basis,
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