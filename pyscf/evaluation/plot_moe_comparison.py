import sys
sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')

from pyscf import gto, scf
import numpy as np
import matplotlib.pyplot as plt

from evaluate import extract_results


def get_mo_energy(geom_file, dm):
  mol = gto.M(atom=geom_file,
            basis="sto-6g",
            spin=0)
  myscf = mol.RHF()

  S = myscf.get_ovlp(mol)
  F = myscf.get_fock(dm=dm)

  e_s, U = np.linalg.eig(S)
  diag_s = np.diag(e_s ** -0.5)
  X = np.dot(U, np.dot(diag_s, U.T))

  F_prime = np.dot(X.T, np.dot(F, X))
  evals_prime, _ = np.linalg.eig(F_prime)
  return np.sum(evals_prime)

if __name__ == "__main__":
  base_dir = '/mnt/c/users/rhjva/imperial/fulvene/hf_calculations/experiments/'
  geom_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  split_file = '../../data/geom_scan_200.npz'

  """ PLOT HERE the thing where you compare the MSE's of different guesses with their t_tot """
  for dir in ['geom_scan_200_mo_coeffs_mse/', 'geom_scan_200_hamiltonian_mse/', 'geom_scan_200_hamiltonian_moe/']:
    hf_results, ml_results = extract_results(split_file, base_dir + dir)
    mse_errors = [get_mo_energy(geom_dir + 'geometry_' + str(result.index) + '.xyz', result.guess) for result in ml_results]
    t_tots = [result.t_tot for result in ml_results]
    plt.scatter(mse_errors, t_tots, label=dir.replace('geom_scan_200_', ''))

  mse_errors = [get_mo_energy(geom_dir + 'geometry_' + str(result.index) + '.xyz', result.guess) for result in hf_results]
  t_tots = [result.t_tot for result in hf_results]
  plt.scatter(mse_errors, t_tots, label='ao-min')
  
  plt.title('Convergence time vs. guess Density Matrix MSE - different guesses')
  plt.xlabel('MSE of guess Density Matrix & converged Density Matrix')
  plt.ylabel('Time taken to converge the SCF procedure')
  plt.legend()
  plt.savefig('img1.png')  


  """ PLOT HERE the thing where you compare the MSE's of different geometries """
  results_list = []
  for dir in ['geom_scan_200_mo_coeffs_mse/', 'geom_scan_200_hamiltonian_mse/', 'geom_scan_200_hamiltonian_moe/']:
    hf_results, ml_results = extract_results(split_file, base_dir + dir)
    results_list.append(ml_results)
  results_list.append(hf_results)

  for geom_idx in range(4):
    mse_errors = np.array([get_mo_energy(geom_dir + 'geometry_' + str(results_list[i][geom_idx].index) + '.xyz', results_list[i][geom_idx].guess) for i in range(4)])
    ind = np.argsort(mse_errors)
    t_tots = np.array([results_list[i][geom_idx].t_tot for i in range(4)])
    plt.plot(mse_errors[ind], t_tots[ind], '-*', label='geometry ' + str(geom_idx))

  plt.title('Convergence time vs. guess Density Matrix MSE - different geometries')
  plt.xlabel('MSE of guess Density Matrix & converged Density Matrix')
  plt.ylabel('Time taken to converge the SCF procedure')
  plt.legend()
  plt.savefig('img2.png')
