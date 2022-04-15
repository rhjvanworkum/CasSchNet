import matplotlib.pyplot as plt

""" check the MO energies of the submitted calculations """
path = "/mnt/c/users/rhjva/imperial/sharc_files/run02/"

mo_energies = []

for idx in range(201):
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
plt.savefig('mo_energies_2.png')