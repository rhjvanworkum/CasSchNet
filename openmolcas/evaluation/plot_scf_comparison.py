# import numpy as np
# import matplotlib.pyplot as plt

# from evaluate import extract_results
  

# if __name__ == "__main__":
#   base_dir = 'C:/Users/rhjva/imperial/fulvene/casscf_calculations/experiments/'
#   split_file = '../../data/geom_scan_200.npz'

#   # # GS - geom_scan_200
#   # dir_list = ['geom_scan_200_mo_coeffs_mse/', 'geom_scan_200_hamiltonian_mse/', 'geom_scan_200_hamiltonian_moe/']
#   # GS - wigner_dist_2000
#   # dir_list = ['wigner_dist_2000_mo_coeffs_mse/', 'wigner_dist_2000_hamiltonian_mse/', 'wigner_dist_2000_hamiltonian_moe/']
#   # # ES - geom_scan_200
#   dir_list = ['geom_scan_200_mo_coeffs_mse/', 'geom_scan_200_mo_coeffs_rot_mse/', 'geom_scan_200_hamiltonian_mse/']

#   # dir_list = ['wigner_dist_2000_geom_scan_mo_coeffs_mse/', 'wigner_dist_2000_geom_scan_hamiltonian_mse/', 'wigner_dist_2000_geom_scan_hamiltonian_moe/']
#   prefix = 'geom_scan_200'

#   """ PLOT N ITERATIONS PLOT """
#   max_value = 0
#   min_value = 1e10

#   for dir in dir_list:
#     hf_results, ml_results = extract_results(split_file, base_dir + dir, CASSCF=True)
#     hf_iterations = [result.imacro for result in hf_results]
#     ml_iterations = [result.imacro for result in ml_results]
#     plt.scatter(x=hf_iterations, y=ml_iterations, label=dir.replace(prefix, ''))

#     if max(max(hf_iterations), max(ml_iterations)) > max_value:
#       max_value = max(max(hf_iterations), max(ml_iterations))
#     if min(min(hf_iterations), min(ml_iterations)) < min_value:
#       min_value = min(min(hf_iterations), min(ml_iterations))

#     print(dir, ' ', 'N iterations: ', np.mean(ml_iterations), ' +/- ', np.std(ml_iterations))
#   print('AO-min guess ', ' ', 'N iterations: ', np.mean(hf_iterations), ' +/- ', np.std(hf_iterations))
    

#   max_value += 2
#   min_value -= 1

#   plt.title('Comparison of SCF iterations between standard & ML guess')
#   plt.plot(np.arange(min_value, max_value), np.arange(min_value, max_value), '--')
#   plt.xlabel('N iterations using minao-guess')
#   plt.ylabel('N iterations using ML-guess')
#   plt.xlim(min_value, max_value)
#   plt.ylim(min_value, max_value)
#   plt.legend()
#   plt.show()



#   """ PLOT T_TOT PLOT """
#   max_value = 0
#   min_value = 1e10

#   for dir in dir_list:
#     hf_results, ml_results = extract_results(split_file, base_dir + dir, CASSCF=True)
#     hf_times = [result.t_tot for result in hf_results]
#     ml_times = [result.t_tot for result in ml_results]
#     plt.scatter(x=hf_times, y=ml_times, label=dir.replace(prefix, ''))

#     if max(max(hf_times), max(ml_times)) > max_value:
#       max_value = max(max(hf_times), max(ml_times))
#     if min(min(hf_times), min(ml_times)) < min_value:
#       min_value = min(min(hf_times), min(ml_times))

#     print(dir, ' ', 'N iterations: ', np.mean(ml_times), ' +/- ', np.std(ml_times))
#   print('AO-min guess ', ' ', 'N iterations: ', np.mean(hf_times), ' +/- ', np.std(hf_times))

#   max_value += 0.5
#   min_value -= 0.5

#   plt.title('Comparison of SCF total_time between standard & ML guess')
#   plt.plot(np.arange(min_value, max_value), np.arange(min_value, max_value), '--')
#   plt.xlabel('total_time using minao-guess')
#   plt.ylabel('total_time using ML-guess')
#   plt.xlim(min_value, max_value)
#   plt.ylim(min_value, max_value)
#   plt.legend()
#   plt.show()