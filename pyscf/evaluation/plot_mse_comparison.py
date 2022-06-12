import numpy as np
import matplotlib.pyplot as plt

from evaluate import extract_results

def mse(mat1, mat2):
  return np.sum((mat2.flatten() - mat1.flatten()) ** 2) / len(mat2.flatten())

if __name__ == "__main__":
  base_dir = 'C:/Users/rhjva/imperial/fulvene/hf_calculations/experiments/'
  split_file = '../../data/geom_scan_200.npz'

  """ PLOT HERE the thing where you compare the MSE's of different guesses with their t_tot """
  for dir in ['geom_scan_200_mo_coeffs_mse/', 'geom_scan_200_hamiltonian_mse/', 'geom_scan_200_hamiltonian_moe/']:
    hf_results, ml_results = extract_results(split_file, base_dir + dir)
    mse_errors = [mse(result.guess, result.converged) for result in ml_results]
    t_tots = [result.t_tot for result in ml_results]
    plt.scatter(mse_errors, t_tots, label=dir.replace('geom_scan_200_', ''))

  mse_errors = [mse(result.guess, result.converged) for result in hf_results]
  t_tots = [result.t_tot for result in hf_results]
  plt.scatter(mse_errors, t_tots, label='ao-min')
  
  plt.title('Convergence time vs. guess Density Matrix MSE - different guesses')
  plt.xlabel('MSE of guess Density Matrix & converged Density Matrix')
  plt.ylabel('Time taken to converge the SCF procedure')
  plt.legend()
  plt.show()  


  """ PLOT HERE the thing where you compare the MSE's of different geometries """
  results_list = []
  for dir in ['geom_scan_200_mo_coeffs_mse/', 'geom_scan_200_hamiltonian_mse/', 'geom_scan_200_hamiltonian_moe/']:
    hf_results, ml_results = extract_results(split_file, base_dir + dir)
    results_list.append(ml_results)
  results_list.append(hf_results)

  for geom_idx in range(4):
    mse_errors = np.array([mse(results_list[i][geom_idx].guess, results_list[i][geom_idx].converged) for i in range(4)])
    ind = np.argsort(mse_errors)
    t_tots = np.array([results_list[i][geom_idx].t_tot for i in range(4)])
    plt.plot(mse_errors[ind], t_tots[ind], '-*', label='geometry ' + str(geom_idx))

  plt.title('Convergence time vs. guess Density Matrix MSE - different geometries')
  plt.xlabel('MSE of guess Density Matrix & converged Density Matrix')
  plt.ylabel('Time taken to converge the SCF procedure')
  plt.legend()
  plt.show()