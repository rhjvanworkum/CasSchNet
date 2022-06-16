import os
from typing import List
from ase.db import connect
from ase.io.extxyz import read_xyz
from tqdm import tqdm
import tempfile
import numpy as np
import re
import h5py

def parse_property_string(prop_str):
    """
    Generate valid property string for extended xyz files.
    (ref. https://libatoms.github.io/QUIP/io.html#extendedxyz)
    Args:
        prop_str (str): Valid property string, or appendix of property string
    Returns:
        valid property string
    """
    if prop_str.startswith("Properties="):
        return prop_str
    return "Properties=species:S:1:pos:R:3:" + prop_str


def xyz_to_extxyz(
    xyz_path,
    extxyz_path,
    atomic_properties="Properties=species:S:1:pos:R:3",
):
    """
    Convert a xyz-file to extxyz.
    Args:
        xyz_path (str): path to the xyz file
        extxyz_path (str): path to extxyz-file
        atomic_properties (str): property-string
        molecular_properties (list): molecular properties contained in the
            comment line
    """
    # ensure valid property string
    atomic_properties = parse_property_string(atomic_properties)
    new_file = open(extxyz_path, "w")
    with open(xyz_path, "r") as xyz_file:
        while True:
            first_line = xyz_file.readline()
            if first_line == "":
                break
            n_atoms = int(re.findall(r'\d+', first_line)[0])
            _ = xyz_file.readline().strip("/n").split()

            comment = ''
            new_file.writelines(str(n_atoms) + "\n")
            new_file.writelines(" ".join([atomic_properties, comment]) + "\n")
            for i in range(n_atoms):
                line = xyz_file.readline()
                new_file.writelines(line)
            break
    new_file.close()


def extxyz_to_db(extxyz_path, db_path, idx, molecular_properties=[]):
    r"""
    Convertes en extxyz-file to an ase database
    Args:
        extxyz_path (str): path to extxyz-file
        db_path(str): path to sqlite database
    """
    with connect(db_path, use_lock_file=False) as conn:
        with open(extxyz_path) as f:
            for at in tqdm(read_xyz(f, index=slice(None)), "creating ase db"):
                data = {}
                for property in molecular_properties:
                  data.update(property)
                conn.write(at, data=data, idx=idx)

def xyz_to_db(
    xyz_path,
    db_path,
    idx,
    atomic_properties="Properties=species:S:1:pos:R:3",
    molecular_properties=[],
):
    """
    Convertes a xyz-file to an ase database.
    Args:
        xyz_path (str): path to the xyz file
        db_path(str): path to sqlite database
        atomic_properties (str): property-string
        molecular_properties (list): molecular properties contained in the
            comment line
    """
    # build temp file in extended xyz format
    extxyz_path = os.path.join(tempfile.mkdtemp(), "temp.extxyz")
    xyz_to_extxyz(xyz_path, extxyz_path, atomic_properties)
    # build database from extended xyz
    extxyz_to_db(extxyz_path, db_path, idx, molecular_properties)


def read_in_orb_file(orb_file : str) -> List[List[float]]:
  orbitals = []

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

  return orbitals

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
    # Moverlap=np.dot(normalise_rows(ref),normalise_rows(target).T)
    # orb_order=np.argmax(abs(Moverlap),axis=1)
    # target = target[orb_order]

    for idx in range(target.shape[0]):
        if np.dot(ref[idx], target[idx]) < 0:
            target[idx] = -1 * target[idx]

    return target

def correct_phase(mo_array: np.ndarray) -> None:
  """
  mo_array -> List of coeffs for 1 MO among each calculation
  """
  ref = mo_array[0]

  for idx in range(1, len(mo_array)):
    if np.dot(mo_array[idx], ref) < 0:
      mo_array[idx] = np.negative(mo_array[idx])

def parse_molcas_calculations(geom_files, hdf5_file_path, db_path, save_property='MO_VECTORS', apply_phase_correction=False, n_basis=36):
  hdf5_files = [h5py.File(hdf5_file) for hdf5_file in hdf5_file_path]
  main_property = [hdf5.get(save_property)[:].reshape(-1, n_basis) for hdf5 in hdf5_files]
  all_guess = [hdf5.get('MO_VECTORS')[:].reshape(-1, n_basis) for hdf5 in hdf5_files]
  all_overlap = [hdf5.get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_basis) for hdf5 in hdf5_files]
  
  if apply_phase_correction:
    ref = main_property[0]
    for idx, orbitals in enumerate(main_property):
      main_property[idx] = order_orbitals(ref, orbitals)

  for idx, geom_file in enumerate(geom_files):
    xyz_to_db(geom_file,
              db_path,
              idx=idx,
              atomic_properties="",
              molecular_properties=[{save_property: main_property[idx].flatten(), 'hf_guess': all_guess[idx], 'overlap': all_overlap[idx]}])

def parse_pyscf_calculations(geom_files, mo_files, db_path, save_property='mo_coeffs', apply_phase_correction=False):
  main_property = [np.load(mo_file)[save_property] for mo_file in mo_files]
  all_guess = [np.load(mo_file)['guess'] for mo_file in mo_files]
  all_overlap = [np.load(mo_file)['S'] for mo_file in mo_files]
  
  if apply_phase_correction:
    ref = main_property[0]
    for idx, orbitals in enumerate(main_property):
      main_property[idx] = order_orbitals(ref, orbitals)

  for idx, geom_file in enumerate(geom_files):
    xyz_to_db(geom_file,
              db_path,
              idx=idx,
              atomic_properties="",
              molecular_properties=[{save_property: main_property[idx].flatten(), 'hf_guess': all_guess[idx], 'overlap': all_overlap[idx]}])