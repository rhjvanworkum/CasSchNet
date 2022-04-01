import os
from ase.db import connect
from ase.io.extxyz import read_xyz
from tqdm import tqdm
import tempfile

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
    molecular_properties=[],
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
            n_atoms = int(first_line.strip("\n"))
            molecular_data = xyz_file.readline().strip("/n").split()
            assert len(molecular_data) == len(molecular_properties), (
                "The number of datapoints and " "properties do not match!"
            )
            comment = " ".join(
                [
                    "{}={}".format(prop, val)
                    for prop, val in zip(molecular_properties, molecular_data)
                ]
            )
            new_file.writelines(str(n_atoms) + "\n")
            new_file.writelines(" ".join([atomic_properties, comment]) + "\n")
            for i in range(n_atoms):
                line = xyz_file.readline()
                new_file.writelines(line)
    new_file.close()


def extxyz_to_db(extxyz_path, db_path):
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
                if at.has("forces"):
                    data["forces"] = at.get_forces()
                data.update(at.info)
                conn.write(at, data=data)

def xyz_to_db(
    xyz_path,
    db_path,
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
    xyz_to_extxyz(xyz_path, extxyz_path, atomic_properties, molecular_properties)
    # build database from extended xyz
    extxyz_to_db(extxyz_path, db_path)


def read_in_orb_file(orb_file):
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
        
def parse_molcas_rasscf_calculation(workdir, db_path):
  xyz_file = None
  orb_file = None

  for _, _, files in os.walk(workdir):
    for file in files:
      if file.split('.')[-1] == 'xyz':
        xyz_file = file
      elif file.split('.')[-1] == 'RasOrb':
        orb_file = file

  orbitals = read_in_orb_file(workdir + orb_file)
  xyz_to_db(workdir + xyz_file, db_path, "", [{'orbital_coeffs': orbitals}])