import numpy as np
import h5py
import torch
import schnetpack as spk
import schnetpack.transform as trn
from ase import io

def ao_2_mo(fock, mo):
  return np.matmul(mo.T, np.matmul(fock, mo))

# test openmolcas fock matrix learning here
model_path = './checkpoints/geom_scan_200_molcas_hamiltonian_mse.pt'
n_mo = 36
cutoff = 5.0

base_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
split_file = './data/geom_scan_200.npz'
output_dir = '/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/experiments/geom_scan_200_hamiltonian_mse/'

indices = np.load(split_file)['val_idx']
index = indices[0]
file = h5py.File('/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200/geometry_' + str(index) + '/CASSCF.rasscf.h5')
fock = file.get('AO_FOCKINT_MATRIX')[:].reshape(36, 36)
overlap = file.get('AO_OVERLAP_MATRIX')[:].reshape(36, 36)
mo = file.get('MO_VECTORS')[:].reshape(36, 36)


e_s, U = np.linalg.eig(overlap)
diag_s = np.diag(e_s ** -0.5)
X = np.dot(U, np.dot(diag_s, U.T))

F_prime = np.dot(X.T, np.dot(fock, X))
evals_prime, C_prime = np.linalg.eig(F_prime)
indices = evals_prime.argsort()
C_prime = C_prime[:, indices]
C = np.dot(X, C_prime).T

print(mo, C)


# for index in indices:
#   file = h5py.File('/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200/geometry_' + str(index) + '/CASSCF.rasscf.h5')
#   fock = file.get('AO_FOCKINT_MATRIX')[:].reshape(36, 36)
#   overlap = file.get('AO_OVERLAP_MATRIX')[:].reshape(36, 36)
#   mo = file.get('MO_VECTORS')[:].reshape(36, 36)

#   fock = ao_2_mo(fock, mo)
#   print(fock)

#   assert np.allclose(fock, fock.T)

# # F -> MO coeffs
# e_s, U = np.linalg.eig(overlap)
# diag_s = np.diag(e_s ** -0.5)
# X = np.dot(U, np.dot(diag_s, U.T))

# F_prime = np.dot(X.T, np.dot(fock, X))
# evals_prime, C_prime = np.linalg.eig(F_prime)
# # indices = evals_prime.argsort()
# # C_prime = C_prime[:, indices]
# C = np.dot(X, C_prime)

# assert np.allclose(fock, fock.T)

# for i in range(36):
#   for j in range(36):
#     print(2 * (fock[i, j] - fock[j, i]))


# evals, C = np.linalg.eig(fock)
# indices = evals.argsort()
# C = C[indices, :]

# MO_coeffs = file.get('MO_VECTORS')[:].reshape(36, 36)
# print(np.sum((MO_coeffs.flatten() - C.flatten()) ** 2) / len(C.flatten()))