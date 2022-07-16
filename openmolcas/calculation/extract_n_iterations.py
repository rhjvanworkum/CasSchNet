from openmolcas.utils import read_log_file
import numpy as np

if __name__ == "__main__":
  geometries_base_path = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  output_path = '/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200_ANO-L-VTZ/'
  n_geometries = 200

  n_iterations = []

  for i in range(n_geometries):
    geometry_path = output_path + 'geometry_' + str(i) + '/'
    _, _, n = read_log_file(geometry_path + 'CASSCF.log')
    n_iterations.append(n)

  np.save('n_iterations.npy', np.array(n_iterations))