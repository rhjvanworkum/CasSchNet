import numpy as np

def get_s1_energy(calc_log_file):
  with open(calc_log_file, 'r') as f:
    while True:
      line = f.readline()

      if 'RASSCF root number  1 Total energy' in line:
        return float(line.replace('\n', '').split(' ')[-1])

def get_s2_energy(calc_log_file):
  with open(calc_log_file, 'r') as f:
    while True:
      line = f.readline()

      if 'RASSCF root number  2 Total energy' in line:
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

def read_in_orb_file(orb_file : str):
  orbitals = []
  energies = []

  append = False

  with open(orb_file, 'r') as file:
    # check for the ORB keyword in RasOrb file
    while True:
      line = file.readline()
      if line[:4] == "#ORB":
        break

    # construct orbitals
    while True:
      line = file.readline()
      # end of block
      if '#' in line:
        break
      # add new orbital
      elif '* ORBITAL' in line:
        orbitals.append([])
      # add coeffs
      else:
        for coeff in line.split(' '):
          if len(coeff) > 0:
            orbitals[-1].append(float(coeff.replace('\n', '')))

    # get energies
    while True:
      line = file.readline()
      # end of block
      if append and '#' in line:
        break
      # append energies
      if append:
        for coeff in line.split(' '):
          if len(coeff) > 0:
            energies.append(float(coeff.replace('\n', '')))
      # begin of block
      if '* ONE ELECTRON ENERGIES' in line:
        append = True

  return np.array(orbitals), np.array(energies)

def get_orbital_occupations(orb_file: str):
  occupations = []

  with open(orb_file, 'r') as file:
    # check for the ORB keyword in RasOrb file
    while True:
      line = file.readline()
      if line[:4] == "#OCC":
        break

    # construct orbitals
    while True:
      line = file.readline()
      # end of block
      if '#' in line:
        break
      elif '* OCCUPATION NUMBERS' in line:
        continue
      else:
        for occ in line.split(' '):
          print
          if len(occ) > 0:
            occupations.append(float(occ.replace('\n', '')))

  return np.array(occupations)

def numpy_to_string(array: np.ndarray) -> str:
  string = ''

  for idx, elem in enumerate(array):
    if elem < 0:
      string += ' '
    else:
      string += '  '
    
    string += '%.14E' % elem

    if (idx + 1) % 5 == 0:
      string += '\n'
    if (idx + 1) == len(array):
      string += '\n'

  return string

def write_coeffs_to_orb_file(coeffs: np.ndarray, input_file_path: str, output_file_path: str, n: int) -> None:
  lines = []

  with open(input_file_path, 'r') as f:
    # add initial lines
    while True:
      line = f.readline()
      if '* ORBITAL' not in line:
        lines.append(line)
      else:
        break

    # add orbitals
    for i in range(1, n+1):
      lines.append(f'* ORBITAL \t 1 \t {i} \n')
      lines.append(numpy_to_string(coeffs[(i-1)*n:i*n]))

    append = False
    while True:
      line = f.readline()
      if '#OCC' in line:
        append = True

      if append:
        lines.append(line)

      if line == '':
        break
      
  with open(output_file_path, 'w+') as f:
    f.writelines(lines)


def read_log_file(file, read_iterations=True):
  n_iterations = None
  rasscf_timing = None
  wall_timing = None

  with open(file, 'r') as f:
    while True:
      line = f.readline()

      if read_iterations:
        if "Convergence after" in line:
          for el in line.split():
            if el.isdigit():
              n_iterations = int(el)
              break

      if "--- Module rasscf spent" in line:
        for el in line.split():
          if el.isdigit():
            rasscf_timing = float(el)
            break

      if "Timing: Wall" in line:
        wall_timing = float(line.replace("=", " ").split()[2])
        break

  return rasscf_timing, wall_timing, n_iterations
