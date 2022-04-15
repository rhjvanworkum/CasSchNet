from typing import List
import os
import shutil
import time
import subprocess

class Atom:

  def __init__(self, type, x, y, z) -> None:
    self.type = type
    self.x = x
    self.y = y
    self.z = z

def remove_empty_strings(str_list):
  while '' in str_list:
    str_list.remove('')

def read_initconds(file_path):
  with open(file_path, 'r') as f:
    configs = []

    while True:
      line = f.readline()

      if 'Index' in line or 'Equilibrium' in line:
        configs.append([])

      str_list = list(filter(None, line.replace('\n', '').split(' ')))
      if len(str_list) == 9:
        configs[-1].append(Atom(str_list[0], str_list[2], str_list[3], str_list[4]))

      if 'EOF' in line:
        break
  
  return configs

def write_xyz(filename: str, atoms: List[Atom]):
  with open(filename, 'w') as f:
    f.write(str(len(atoms)) + ' \n')
    f.write('bohr \n')
    for atom in atoms:
      f.write(atom.type + '\t\t\t' + atom.x + '\t\t\t' + atom.y + '\t\t\t' + atom.z + ' \n')

if __name__ == "__main__":
  configs = read_initconds('./opt_freq/initconds')
  
  if not os.path.exists('./run02/'):
    os.makedirs('./run02/')

  for idx, config in enumerate(configs):
    path = './run02/config_' + str(idx) + '/'
    
    if not os.path.exists(path):
      os.makedirs(path)

      write_xyz('./run02/config_' + str(idx) + '/geom.xyz', config)
      shutil.copy2('./fulvene.input', './run02/config_' + str(idx) + '/fulvene.input')
      shutil.copy2('fulvene.Orb', './run02/config_' + str(idx) + '/fulvene.Orb')


  for idx, _ in enumerate(configs):
    subprocess.run('cd /mnt/c/users/rhjva/imperial/sharc_files/run02/config_' + str(idx) + '/ && pymolcas fulvene.input', shell=True)
