import numpy as np

from openmolcas.utils import get_s1_energy

if __name__ == "__main__":
  prefix = 'C:/Users/rhjva/imperial'
  output_path = prefix + '/fulvene/openmolcas_calculations/geom_scan_200/'
  split_file = './data/geom_scan_200_molcas.npz'

  s1_energies = []
  for i in np.load(split_file)['val_idx']:
    geometry_path = output_path + 'geometry_' + str(i) + '/'
    s1_energies.append(get_s1_energy(geometry_path + 'calc.log'))
  
  print(np.mean(s1_energies))