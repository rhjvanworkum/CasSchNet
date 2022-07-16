import numpy as np


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

def normalise_rows(mat):
    '''Normalise each row of mat'''
    return np.array(tuple(map(lambda v: v / np.linalg.norm(v), mat)))

def flip(v):
    '''Returns 1 if max(abs(v))) is positive, and -1 if negative'''
    maxpos=np.argmax(abs(v))
    return v[maxpos]/abs(v[maxpos])

def order_orbitals(ref, target):
    '''Reorder target molecular orbitals according to maximum overlap with ref.
    Orbitals phases are also adjusted to match ref.'''
    # Moverlap=np.dot(normalise_rows(ref), normalise_rows(target).T)
    # orb_order=np.argmax(abs(Moverlap),axis=1)
    # target = target[orb_order]

    for idx in range(target.shape[0]):
        if np.dot(ref[:, idx], target[:, idx]) < 0:
            target[:, idx] = -1 * target[:, idx]

    return target # , orb_order

def correct_phase(mo_array: np.ndarray) -> None:
  """
  mo_array -> List of coeffs for 1 MO among each calculation
  """
  ref = mo_array[0]

  for idx in range(1, len(mo_array)):
    if np.dot(mo_array[idx], ref) < 0:
      mo_array[idx] = np.negative(mo_array[idx])