# test pyscf matrix here

import numpy as np
import h5py
import torch
import schnetpack as spk
import schnetpack.transform as trn
from ase import io

import sys
sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')
from pyscf import gto, mcscf, lib


# test openmolcas fock matrix learning here

model_path = './checkpoints/geom_scan_200_casscf_hamiltonian_mse.pt'
n_mo = 36
cutoff = 5.0

base_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
split_file = './data/geom_scan_200.npz'
output_path = '/mnt/c/users/rhjva/imperial/fulvene/casscf_calculations/geom_scan_200/'

indices = np.load(split_file)['val_idx']
index = indices[0]

file = np.load(output_path + 'geometry_' + str(index) + '.npz')
fock = file['F']
overlap = file['S']
mo_coeffs = file['mo_coeffs']

e_s, U = np.linalg.eig(overlap)
diag_s = np.diag(e_s ** -0.5)
X = np.dot(U, np.dot(diag_s, U.T))

F_prime = np.dot(X.T, np.dot(fock, X))
evals_prime, C_prime = np.linalg.eig(F_prime)
indices = evals_prime.argsort()
C_prime = C_prime[:, indices]
C = np.dot(X, C_prime)

print(mo_coeffs, C)


# for index in indices:
#   file = np.load(output_path + 'geometry_' + str(index) + '.npz')
#   fock = file['F']
#   overlap = file['S']

#   assert np.allclose(fock, fock.T)

# # F -> MO coeffs
# e_s, U = np.linalg.eig(overlap)
# diag_s = np.diag(e_s ** -0.5)
# X = np.dot(U, np.dot(diag_s, U.T))

# F_prime = np.dot(X.T, np.dot(fock, X))
# evals_prime, C_prime = np.linalg.eig(F_prime)
# indices = evals_prime.argsort()
# C_prime = C_prime[:, indices]
# C = np.dot(X, C_prime)

# # make MOL
# mol = gto.M(atom=base_dir + "geometry_" + str(indices[0]) + ".xyz",
#             basis="sto-6g",
#             spin=0)

# # run HF
# scf = mol.RHF()
# scf.kernel()

# # initiate CASSCF object
# n_states = 2
# weights = np.ones(n_states) / n_states
# mcas = scf.CASSCF(ncas=6, nelecas=6).state_average(weights)
# mcas.conv_tol = 1e-8

# # project initial guess
# mo = mcscf.project_init_guess(mcas, C)
# mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)

# MO_coeffs = file['mo_coeffs']

# print(np.sum((MO_coeffs.flatten() - mo.flatten()) ** 2) / len(mo.flatten()))