import matplotlib.pyplot as plt
import numpy as np

calcs = ["CASCI_ML", "CASSCF_ML", "CASSCF"]

def extract_mo_energies(file_path: str) -> np.ndarray:
  mo_energies = []

  with open(file_path, 'r') as f:
    append = False

    while True:
      line = f.readline()

      if "#INDEX" in line:
        append = False

      if append:
        energies = line.replace('\n', '').split(' ')
        for energy in energies:
          if energy != '':
            mo_energies.append(float(energy))

      if '* ONE ELECTRON ENERGIES' in line:
        append = True

      if line == '':
        break

  return np.array(mo_energies)

if __name__ == "__main__":
  dir = 'C:/Users/rhjva/imperial/molcas_files/wigner_dist_200/config_200_02/'

  mo_energy_dict = {}
  for calc in calcs:
    mo_energy_dict[calc] = extract_mo_energies(dir + calc + '/' + calc + '.RasOrb')

  """ Plotting the MO energies """
  plt.title('MO energies')
  for key, value in mo_energy_dict.items():
    plt.scatter(np.arange(1, len(value) + 1), value, label=key)
  plt.xlabel('MO idx')
  plt.ylabel('energy')
  plt.legend()
  plt.show()

  """ Printing the MAE """
  active_space = [1, 36]
  print('Average amount CASCI_ML is higher in energy than CASSCF: ', 
    '%.14E' % np.sum(mo_energy_dict["CASCI_ML"][active_space[0] - 1 : active_space[1]] - mo_energy_dict["CASSCF"][active_space[0] - 1 : active_space[1]]))
  print('Average amount CASSCF_ML is higher in energy than CASSCF: ', 
    '%.14E' % np.sum(np.abs(mo_energy_dict["CASSCF_ML"][active_space[0] - 1 : active_space[1]] - mo_energy_dict["CASSCF"][active_space[0] - 1 : active_space[1]])))