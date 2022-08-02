import numpy as np
import matplotlib.pyplot as plt
from openmolcas.utils import get_mo_energies, get_s1_energy

if __name__ == "__main__":
  prefix = '/mnt/c/users/rhjva/imperial'
  geometries_base_path = prefix + '/fulvene/geometries/wigner_dist_200/'
  output_path = prefix + '/fulvene/openmolcas_calculations/wigner_dist_200_canonical/'
  n_geometries = 200

  mo_energies = []
  s1_energies = []
  for i in range(n_geometries):
    geometry_path = output_path + 'geometry_' + str(i) + '/'
    mo_energies.append(get_mo_energies(geometry_path + 'CASSCF.RasOrb'))
    s1_energies.append(get_s1_energy(geometry_path + 'calc.log'))

  # plot MO ENERGIES
  x = []
  y = []
  for _mo_energies in mo_energies:
    for idx, mo_energy in enumerate(_mo_energies):
      x.append(idx)
      y.append(mo_energy)

  plt.scatter(x, y)
  plt.xlabel('MO idx')
  plt.ylabel('MO energy')
  plt.title('MO energies of 200 RASSCF calculations')
  plt.savefig('plots/mo_energies.png')
  plt.clf()


  # plot S1 ENERGIES
  plt.scatter(np.arange(len(s1_energies)), s1_energies)
  plt.xlabel('Calculation n')
  plt.ylabel('S1 energy')
  plt.title('S1 energies of 200 RASSCF calculations')
  plt.savefig('plots/s1_energies.png')
