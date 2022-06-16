import sys
sys.path.insert(1, '/mnt/c/users/rhjva/imperial/pyscf/')

from pyscf import gto, scf, mcscf
import numpy as np
import matplotlib.pyplot as plt

from evaluate import extract_results

""" Convert into MO basis + 0.5 *"""
def get_mo_energy(geom_file, guess, fci_vec):
  mol = gto.M(atom=geom_file,
            basis="sto-6g",
            spin=0)

  myscf = mol.RHF()
  myscf.kernel()
  S = myscf.get_ovlp(mol)

  n_states = 2
  weights = np.ones(n_states) / n_states
  mcas = myscf.CASSCF(ncas=6, nelecas=6).state_average(weights)
  mcas.conv_tol = 1e-8

  # project initial guess
  mo = mcscf.project_init_guess(mcas, guess)
  mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)

  F = mcas.get_fock(mo_coeff=mo, ci=fci_vec)
  hcore = mcas.get_hcore(mol)

  e_s, U = np.linalg.eig(S)
  diag_s = np.diag(e_s ** -0.5)
  X = np.dot(U, np.dot(diag_s, U.T))

  F_prime = np.dot(X.T, np.dot(F, X))
  evals_prime, C_prime = np.linalg.eig(F_prime)
  indices = evals_prime.argsort()
  C_prime = C_prime[:, indices]
  C = np.dot(X, C_prime)

  hcore = np.matmul(C.T, np.matmul(hcore, C))

  return np.sum(evals_prime)


def moe_error(geom_file, guess, converged, fci_vec):
  return np.abs(get_mo_energy(geom_file, converged, fci_vec) - get_mo_energy(geom_file, guess, fci_vec))


if __name__ == "__main__":
  base_dir = '/mnt/c/users/rhjva/imperial/fulvene/casscf_calculations/experiments/'
  geom_dir = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  split_file = '../../data/geom_scan_200.npz'

  """ PLOT HERE the thing where you compare the MSE's of different guesses with their t_tot """
  # for dir in ['geom_scan_200_mo_coeffs_mse_fcivec/', 'geom_scan_200_mo_coeffs_rot_mse_fcivec/', 'geom_scan_200_hamiltonian_mse_fcivec/']:
  #   hf_results, ml_results = extract_results(split_file, base_dir + dir, CASSCF=True)
  #   mse_errors = [moe_error(geom_dir + 'geometry_' + str(result.index) + '.xyz', result.guess, result.converged, result.fci_vec) for result in ml_results]
  #   t_tots = [result.t_tot for result in ml_results]
  #   plt.scatter(mse_errors, t_tots, label=dir.replace('geom_scan_200_', ''))

  # mse_errors = [moe_error(geom_dir + 'geometry_' + str(result.index) + '.xyz', result.guess, result.converged, result.fci_vec) for result in hf_results]
  # t_tots = [result.t_tot for result in hf_results]
  # plt.scatter(mse_errors, t_tots, label='hf-guess')
  
  # plt.title('Convergence time vs. MO energy MAE - different guesses')
  # plt.xlabel('MAE of the converged & guess MO energies')
  # plt.ylabel('Time taken to converge the SCF procedure')
  # plt.legend()
  # plt.savefig('img1.png')  
  # plt.clf()


  """ PLOT HERE the thing where you compare the MSE's of different geometries """
  results_list = []
  for dir in ['geom_scan_200_mo_coeffs_mse_fcivec/', 'geom_scan_200_mo_coeffs_rot_mse_fcivec/', 'geom_scan_200_hamiltonian_mse_fcivec/']:
    hf_results, ml_results = extract_results(split_file, base_dir + dir, CASSCF=True)
    results_list.append(ml_results)
  results_list.append(hf_results)

  for geom_idx in [0, 1, 2, 5]:
    mse_errors = np.array([moe_error(geom_dir + 'geometry_' + str(results_list[i][geom_idx].index) + '.xyz', results_list[i][geom_idx].guess, results_list[i][geom_idx].converged, results_list[i][geom_idx].fci_vec) for i in range(4)])
    ind = np.argsort(mse_errors)
    t_tots = np.array([results_list[i][geom_idx].t_tot for i in range(4)])
    if geom_idx == 5:
      geom_idx = 3
    plt.plot(mse_errors[ind], t_tots[ind], '-*', label='geometry ' + str(geom_idx))

  plt.title('Convergence time vs. MO energy MAE - different geometries')
  plt.xlabel('MAE of the converged & guess MO energies')
  plt.ylabel('Time taken to converge the SCF procedure')
  plt.legend()
  plt.savefig('img2.png')
  plt.clf()

  # """ LEAVE out one specifically """
  # results_list = []
  # for dir in ['geom_scan_200_mo_coeffs_mse/', 'geom_scan_200_hamiltonian_mse/']:   # , 'geom_scan_200_hamiltonian_moe/'
  #   hf_results, ml_results = extract_results(split_file, base_dir + dir)
  #   results_list.append(ml_results)
  # results_list.append(hf_results)

  # for geom_idx in range(4):
  #   mse_errors = np.array([moe_error(geom_dir + 'geometry_' + str(results_list[i][geom_idx].index) + '.xyz', results_list[i][geom_idx].guess, results_list[i][geom_idx].converged) for i in range(3)])
  #   ind = np.argsort(mse_errors)
  #   t_tots = np.array([results_list[i][geom_idx].t_tot for i in range(3)])
  #   plt.plot(mse_errors[ind], t_tots[ind], '-*', label='geometry ' + str(geom_idx))

  # plt.title('Convergence time vs. MO energy MAE - different geometries')
  # plt.xlabel('MAE of the converged & guess MO energies')
  # plt.ylabel('Time taken to converge the SCF procedure')
  # plt.legend()
  # plt.savefig('img3.png')
  # plt.clf()
