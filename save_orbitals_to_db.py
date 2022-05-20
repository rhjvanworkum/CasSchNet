from src.schnetpack.data.parser import parse_molcas_rasscf_calculations
from ase.db import connect
import numpy as np

""" generate ASE database """
base_dir = 'C:/users/rhjva/imperial/molcas_files/fulvene_scan_2/'
db_path = './data/fulvene_scan_2_delta.db'

name = 'fulvene_scan_140'
n = 200
train_split = 0.7
val_split = 0.2
test_split = 0.1

calc_folders = [base_dir + 'geometry_' + str(idx) + '/CASSCF/' for idx in range(200)]

parse_molcas_rasscf_calculations(calc_folders, 
                                 db_path, 
                                 reference_file="C:/Users/rhjva/imperial/molcas_files/fulvene_scan_2/geometry_0/CASSCF/CASSCF.RasOrb",
                                 use_delta=True)

with connect(db_path) as conn:
  conn.metadata = {"_distance_unit": 'angstrom',
                   "_property_unit_dict": {"orbital_coeffs": 1.0},
                   "atomrefs": {
                     'orbital_coeffs': [0.0 for _ in range(32)]
                    }
                  }

# data_idx = np.arange(n)
# np.random.shuffle(data_idx)
# train_idxs = data_idx[:int(train_split * n)]
# print(len(train_idxs))
# val_idxs = data_idx[int(train_split * n):int((train_split + val_split) * n)]
# test_idxs = data_idx[int((train_split + val_split) * n):]

# np.savez('./data/' + name + '.npz', 
#   train_idx=train_idxs, 
#   val_idx=val_idxs,
#   test_idx=test_idxs)