import matplotlib.pyplot as plt
import numpy as np

""" Plotting the MO energies """
def plot_mo_energies():
  mo_energies = []
  for idx in range(2001):
    file_path = path + "config_" + str(idx) + "/fulvene.RasOrb"
    with open(file_path, 'r') as f:

      append = False
      mo_energies.append([])

      while True:
        line = f.readline()

        if "#INDEX" in line:
          append = False

        if append:
          energies = line.replace('\n', '').split(' ')
          for energy in energies:
            if energy != '':
              mo_energies[-1].append(float(energy))

        if "* ONE ELECTRON ENERGIES" in line:
          append = True

        if line == '':
          break

  x = []
  y = []
  for _mo_energies in mo_energies:
    for idx, mo_energy in enumerate(_mo_energies):
      x.append(idx)
      y.append(mo_energy)

  plt.scatter(x, y)
  plt.xlabel('MO idx')
  plt.ylabel('MO energy')
  plt.title('MO energies of 2000 RASSCF calculations')
  plt.savefig('mo_energies.png')

""" Plotting the S1-state energies """
def plot_s1_energies():
  s1_energies = []
  for idx in range(2001):
    file_path = path + "config_" + str(idx) + "/calc.log"
    with open(file_path, 'r') as f:
      while True:
        line = f.readline()

        if 'RASSCF root number  1 Total energy' in line:
          s1_energies.append(line.replace('\n', '').split(' ')[-1])
          break

  plt.scatter(np.arange(len(s1_energies)), s1_energies)
  plt.xlabel('Calculation n')
  plt.yticks(color='w')
  plt.ylabel('S1 energy')
  plt.title('S1 energies of 2000 RASSCF calculations')
  plt.savefig('s1_energies.png')

if __name__ == "__main__":
  path = "./work/run_MB/"
  plot_s1_energies()