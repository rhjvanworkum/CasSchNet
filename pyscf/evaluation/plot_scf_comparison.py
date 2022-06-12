import numpy as np
import matplotlib.pyplot as plt

from evaluate import extract_results
  

if __name__ == "__main__":
  base_dir = 'C:/Users/rhjva/imperial/fulvene/hf_calculations/experiments/'
  split_file = '../../data/geom_scan_200.npz'


  """ PLOT N ITERATIONS PLOT """
  max_value = 0
  min_value = 1e10

  for dir in ['geom_scan_200_mo_coeffs_mse/', 'geom_scan_200_hamiltonian_mse/', 'geom_scan_200_hamiltonian_moe/']:
    hf_results, ml_results = extract_results(split_file, base_dir + dir)
    hf_iterations = [result.imacro for result in hf_results]
    ml_iterations = [result.imacro for result in ml_results]
    plt.scatter(x=hf_iterations, y=ml_iterations, label=dir.replace('geom_scan_200_', ''))

    if max(max(hf_iterations), max(ml_iterations)) > max_value:
      max_value = max(max(hf_iterations), max(ml_iterations))
    if min(min(hf_iterations), min(ml_iterations)) < min_value:
      min_value = min(min(hf_iterations), min(ml_iterations))
    

  max_value += 2
  min_value -= 1

  plt.title('Comparison of SCF iterations between standard & ML guess')
  plt.plot(np.arange(min_value, max_value), np.arange(min_value, max_value), '--')
  plt.xlabel('N iterations using minao-guess')
  plt.ylabel('N iterations using ML-guess')
  plt.xlim(min_value, max_value)
  plt.ylim(min_value, max_value)
  plt.legend()
  plt.show()



  """ PLOT T_TOT PLOT """
  max_value = 0
  min_value = 1e10

  for dir in ['geom_scan_200_mo_coeffs_mse/', 'geom_scan_200_hamiltonian_mse/', 'geom_scan_200_hamiltonian_moe/']:
    hf_results, ml_results = extract_results(split_file, base_dir + dir)
    hf_times = [result.t_tot for result in hf_results]
    ml_times = [result.t_tot for result in ml_results]
    plt.scatter(x=hf_times, y=ml_times, label=dir.replace('geom_scan_200_', ''))

    if max(max(hf_times), max(ml_times)) > max_value:
      max_value = max(max(hf_times), max(ml_times))
    if min(min(hf_times), min(ml_times)) < min_value:
      min_value = min(min(hf_times), min(ml_times))

  max_value += 0.5
  min_value -= 0.5

  plt.title('Comparison of SCF total_time between standard & ML guess')
  plt.plot(np.arange(min_value, max_value), np.arange(min_value, max_value), '--')
  plt.xlabel('total_time using minao-guess')
  plt.ylabel('total_time using ML-guess')
  plt.xlim(min_value, max_value)
  plt.ylim(min_value, max_value)
  plt.legend()
  plt.show()