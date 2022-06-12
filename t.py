import sys
sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')

from pyscf import gto, mcscf, lib
import numpy as np
import time
import scipy.linalg

geometries_base_path = '../molcas_files/fulvene_scan_2/'


def to_orthonormal(F, X):
  return np.dot(X.T, np.dot(F, X))

# make MOL
mol = gto.M(atom=geometries_base_path + "geometry_" + str(45) + "/CASSCF/geom.xyz",
            basis="sto-6g",
            spin=0)

# run HF
scf = mol.RHF()
cycle, mo_coeff = scf.kernel()

F = scf.get_fock()
S = scf.get_ovlp()

e_s, U = np.linalg.eig(S)
diag_s = np.diag(e_s ** -0.5)
X = np.dot(U, np.dot(diag_s, U.T))

F_prime = to_orthonormal(F, X)
evals_prime, C_prime = np.linalg.eig(F_prime)
indices = evals_prime.argsort()
C_prime = C_prime[:, indices]
C = np.dot(X, C_prime)

print(mo_coeff)
print(C)





# # initiate CASSCF object
# n_states = 2
# weights = np.ones(n_states) / n_states
# mcas = scf.CASSCF(ncas=6, nelecas=6).state_average(weights)
# mcas.conv_tol = 1e-12       # 1e-8
# mcas.max_stepsize = 0.003

# # init_guess = scf.mo_coeff
# init_guess = np.load('pt_orbs.npy').reshape(-1, 36)
# # init_guess = np.load('test.npy')

# # ini = scf.mo_coeff
# # np.save('test.npy', ini)

# # project initial guess
# mo = mcscf.project_init_guess(mcas, init_guess)
# mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)
# tstart = time.time()
# (imacro, _, _, _, mo_coeffs, _) = mcas.kernel(mo)
# print(imacro, time.time() - tstart)

# # np.save('test_X.npy', scipy.linalg.logm(np.matmul(mo_coeffs, np.linalg.inv(ini))))