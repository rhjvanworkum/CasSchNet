from schnetpack.data.atoms import AtomsDataFormat, load_dataset
from src.schnetpack.data.parser import parse_molcas_rasscf_calculation
from ase.db import connect

import numpy as np



""" generate ASE database """
calc_dir = 'C:/users/rhjva/imperial/molcas_files/fulvene/'
db_path = './data/test.db'

for _ in range(32):
  parse_molcas_rasscf_calculation(calc_dir, db_path)

# with connect('./testje.db') as conn:
#   for row in conn.select():
#     print(row.data.orbital_coeffs)

#   # print(conn.count(), conn.metadata)
#   # conn.metadata = {"_distance_unit": 'nm',
#   #                  "_property_unit_dict": {"orbital_coeffs": 1.0}}

#     for row in conn.select():
#       print(row['orbital_coeffs'])


with connect('./test.db') as conn:
  conn.metadata = {"_distance_unit": 'nm',
                   "_property_unit_dict": {"orbital_coeffs": 1.0},
                   "atomrefs": {
                     'orbital_coeffs': [0.0 for _ in range(32)]
                    }
                  }
  
  # print(len(conn.metadata['atomrefs']['zpve']))
  # print(conn.count())

#   print(conn.count())

# dataset = load_dataset(db_path,
#                       AtomsDataFormat.ASE)

# print(len(dataset))

# for item in dataset:
#   print(item)