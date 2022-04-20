from src.schnetpack.data.parser import parse_molcas_rasscf_calculations
from ase.db import connect

""" generate ASE database """
base_dir = 'C:/users/rhjva/imperial/sharc_files/run02/'
db_path = './data/fulvene_wigner_dist_200_02.db'

calc_folders = [base_dir + 'config_' + str(idx) + '/' for idx in range(200)]

parse_molcas_rasscf_calculations(calc_folders, db_path, correct_phases=True)

with connect(db_path) as conn:
  conn.metadata = {"_distance_unit": 'angstrom',
                   "_property_unit_dict": {"orbital_coeffs": 1.0},
                   "atomrefs": {
                     'orbital_coeffs': [0.0 for _ in range(32)]
                    }
                  }