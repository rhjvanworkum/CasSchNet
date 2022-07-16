import numpy as np
import matplotlib.pyplot as plt

def get_s1_energy(calc_log_file):
  with open(calc_log_file, 'r') as f:
    while True:
      line = f.readline()

      if 'RASSCF root number  1 Total energy' in line:
        return float(line.replace('\n', '').split(' ')[-1])

def get_mo_energies(orb_file):
  with open(orb_file, 'r') as f:

    append = False
    mo_energies = []

    while True:
      line = f.readline()

      if "#INDEX" in line:
        append = False

      if append:
        energies = line.replace('\n', '').split(' ')
        for energy in energies:
          if energy != '':
            mo_energies.append(float(energy))

      if "* ONE ELECTRON ENERGIES" in line:
        append = True

      if line == '':
        break
  
  return mo_energies

if __name__ == "__main__":
  geometries_base_path = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_2000/'
  output_path = '/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200_ANO-S-VDZ/'
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
