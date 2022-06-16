import sys
sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')

from pyscf import gto, scf
import numpy as np
import matplotlib.pyplot as plt

from evaluate import extract_results

""" ConvERT into MO basis + 0.5 *"""
def get_mo_energy(geom_file, dm):
  mol = gto.M(atom=geom_file,
            basis="sto-6g",
            spin=0)
  myscf = mol.RHF()

  S = myscf.get_ovlp(mol)
  F = myscf.get_fock(dm=dm)
  hcore = myscf.get_hcore(mol)

  e_s, U = np.linalg.eig(S)
  diag_s = np.diag(e_s ** -0.5)
  X = np.dot(U, np.dot(diag_s, U.T))

  F_prime = np.dot(X.T, np.dot(F, X))
  evals_prime, C_prime = np.linalg.eig(F_prime)
  indices = evals_prime.argsort()
  C_prime = C_prime[:, indices]
  C = np.dot(X, C_prime)

  hcore = np.matmul(C.T, np.matmul(hcore, C))

  return np.sum(evals_prime[:21]) + np.sum(np.diag(hcore)[:21])

def moe_error(geom_file, dm_guess, dm_converged):
  return np.abs(get_mo_energy(geom_file, dm_converged) - get_mo_energy(geom_file, dm_guess))

if __name__ == "__main__":
  base_dir = '/mnt/c/users/rhjva/imperial/fulvene/hf_calculations/experiments/'
  geom_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/wigner_dist_2000/'
  split_file = '../../data/wigner_dist_2000.npz'

  """ PLOT HERE the thing where you compare the MSE's of different guesses with their t_tot """
  for dir in ['wigner_dist_2000_mo_coeffs_mse/', 'wigner_dist_2000_hamiltonian_mse/', 'wigner_dist_2000_hamiltonian_moe/']:
    hf_results, ml_results = extract_results(split_file, base_dir + dir)
    mse_errors = [moe_error(geom_dir + 'geometry_' + str(result.index) + '.xyz', result.guess, result.converged) for result in ml_results]
    t_tots = [result.imacro for result in ml_results]
    plt.scatter(mse_errors, t_tots, label=dir.replace('wigner_dist_2000_', ''))

  mse_errors = [moe_error(geom_dir + 'geometry_' + str(result.index) + '.xyz', result.guess, result.converged) for result in hf_results]
  t_tots = [result.imacro for result in hf_results]
  plt.scatter(mse_errors, t_tots, label='ao-min')
  
  plt.title('N iterations vs. MO energy MAE - different guesses')
  plt.xlabel('MAE of the converged & guess MO energies')
  # plt.ylabel('Time taken to converge the SCF procedure')
  plt.ylabel('N SCF Iterations')
  plt.legend()
  plt.savefig('img1.png')  
  plt.clf()


  """ PLOT HERE the thing where you compare the MSE's of different geometries """
  results_list = []
  for dir in ['wigner_dist_2000_mo_coeffs_mse/', 'wigner_dist_2000_hamiltonian_mse/', 'wigner_dist_2000_hamiltonian_moe/']:
    hf_results, ml_results = extract_results(split_file, base_dir + dir)
    results_list.append(ml_results)
  results_list.append(hf_results)

  for geom_idx in range(4):
    mse_errors = np.array([moe_error(geom_dir + 'geometry_' + str(results_list[i][geom_idx].index) + '.xyz', results_list[i][geom_idx].guess, results_list[i][geom_idx].converged) for i in range(4)])
    ind = np.argsort(mse_errors)
    t_tots = np.array([results_list[i][geom_idx].imacro for i in range(4)])
    plt.plot(mse_errors[ind], t_tots[ind], '-*', label='geometry ' + str(geom_idx))

  plt.title('N iterations vs. MO energy MAE - different geometries')
  plt.xlabel('MAE of the converged & guess MO energies')
  # plt.ylabel('Time taken to converge the SCF procedure')
  plt.ylabel('N SCF Iterations')
  plt.legend()
  plt.savefig('img2.png')
  plt.clf()

  """ LEAVE out one specifically """
  results_list = []
  for dir in ['wigner_dist_2000_mo_coeffs_mse/', 'wigner_dist_2000_hamiltonian_mse/']:   # , 'wigner_dist_2000_hamiltonian_moe/'
    hf_results, ml_results = extract_results(split_file, base_dir + dir)
    results_list.append(ml_results)
  results_list.append(hf_results)

  for geom_idx in range(4):
    mse_errors = np.array([moe_error(geom_dir + 'geometry_' + str(results_list[i][geom_idx].index) + '.xyz', results_list[i][geom_idx].guess, results_list[i][geom_idx].converged) for i in range(3)])
    ind = np.argsort(mse_errors)
    t_tots = np.array([results_list[i][geom_idx].imacro for i in range(3)])
    plt.plot(mse_errors[ind], t_tots[ind], '-*', label='geometry ' + str(geom_idx))

  plt.title('N iterations vs. MO energy MAE - different geometries')
  plt.xlabel('MAE of the converged & guess MO energies')
  # plt.ylabel('Time taken to converge the SCF procedure')
  plt.ylabel('N SCF Iterations')
  plt.legend()
  plt.savefig('img3.png')
  plt.clf()
