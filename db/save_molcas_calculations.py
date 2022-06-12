from src.schnetpack.data.parser import parse_molcas_calculations, parse_molcas_rasscf_calculations, parse_pyscf_calculations, parse_pyscf_hf_calculations, parse_pyscf_rasscf_calculations
from ase.db import connect
import numpy as np

""" generate ASE database """
base_dir = 'C:/users/rhjva/imperial/molcas_files/fulvene_dataset_2200/'
db_path = './data/fulvene_scan_pyscf_hf_F_big.db'


name = 'fulvene_scan_2201'
n = 2201
train_split = 0.9
val_split = 0.1
test_split = 0.0

""" SAVE MOLCAS calcualtions in DB - old """
# calc_folders = [base_dir + 'geometry_' + str(idx) + '/CASSCF/' for idx in range(200)]
# parse_molcas_rasscf_calculations(calc_folders, 
#                                  db_path, 
#                                  reference_file="C:/Users/rhjva/imperial/molcas_files/fulvene_scan_2/geometry_0/CASSCF/CASSCF.RasOrb",
#                                  use_delta=False)

""" SAVE MOLCAS calculations in DB - new """
# base_dir = 'C:/users/rhjva/imperial/molcas_files/fulvene_scan/'
# geom_files = [base_dir + 'geometry_' + str(idx) + '/scf/geom.xyz' for idx in range(200)]
# mo_files = [base_dir + 'geometry_' + str(idx) + '/casscf/CASSCF.RasOrb' for idx in range(200)]
# hf_guesses = [base_dir + 'geometry_' + str(idx) + '/scf/scf.ScfOrb' for idx in range(200)]
# ref_mo_file = base_dir + 'geometry_0/casscf/CASSCF.RasOrb'
# parse_molcas_calculations(geom_files=geom_files,
#                           mo_files=mo_files,
#                           hf_guesses=hf_guesses,
#                           ref_mo_file=ref_mo_file,
#                           db_path=db_path)


""" SAVE PYSCF calculations in DB """
# geom_files = [base_dir + 'geometry_' + str(idx) + '/CASSCF/geom.xyz' for idx in range(200)]
# mo_files = ['C:/Users/rhjva/imperial/pyscf_files/fulvene_scan/geometry_' + str(idx) + '.npz' for idx in range(200)]
# hf_guesses = ['C:/Users/rhjva/imperial/pyscf_files/fulvene_scan/geometry_' + str(idx) + '_hf_guess.npy' for idx in range(200)]
# # ref_mo_file = "C:/Users/rhjva/imperial/pyscf/geom_0_hf_guess.npy"

# parse_pyscf_rasscf_calculations(geom_files=geom_files,
#                                 mo_files=mo_files,
#                                 hf_guesses=hf_guesses,
#                                 ref_mo_file=None,
#                                 db_path=db_path)

# with connect(db_path) as conn:
#   conn.metadata = {"_distance_unit": 'angstrom',
#                    "_property_unit_dict": {"orbital_coeffs": 1.0, "hf_guess": 1.0},
#                    "atomrefs": {
#                      'orbital_coeffs': [0.0 for _ in range(32)],
#                      'hf_guess': [0.0 for _ in range(32)]
#                     }
#                   }

""" SAVE PYSCF calculations in DB - new """
# geom_files = [base_dir + 'geometry_' + str(idx) + '/CASSCF/geom.xyz' for idx in range(200)]
# mo_files = ['C:/Users/rhjva/imperial/pyscf_files/fulvene_scan_projection/geometry_' + str(idx) + '.npz' for idx in range(200)]
# hf_guesses = ['C:/Users/rhjva/imperial/pyscf_files/fulvene_scan_projection/geometry_' + str(idx) + '_hf_guess.npy' for idx in range(200)]
# overlap_files = ['C:/Users/rhjva/imperial/pyscf_files/fulvene_scan_projection/geometry_' + str(idx) + '_overlap.npy' for idx in range(200)]
# guess_occ_files = ['C:/Users/rhjva/imperial/pyscf_files/fulvene_scan_projection/geometry_' + str(idx) + '_guess_occ.npy' for idx in range(200)]
# conv_occ_files = ['C:/Users/rhjva/imperial/pyscf_files/fulvene_scan_projection/geometry_' + str(idx) + '_conv_occ.npy' for idx in range(200)]

# parse_pyscf_calculations(geom_files=geom_files,
#                           mo_files=mo_files,
#                           hf_guesses=hf_guesses,
#                           overlap_files=overlap_files,
#                           guess_occ_files=guess_occ_files,
#                           conv_occ_files=conv_occ_files,
#                           db_path=db_path)

# with connect(db_path) as conn:
#   conn.metadata = {"_distance_unit": 'angstrom',
#                    "_property_unit_dict": {
#                      "orbital_coeffs": 1.0, 
#                      "hf_guess": 1.0,
#                      "overlap": 1.0,
#                      "guess_occ": 1.0,
#                      "conv_occ": 1.0
#                     },
#                    "atomrefs": {
#                      'orbital_coeffs': [0.0 for _ in range(32)],
#                      'hf_guess': [0.0 for _ in range(32)],
#                      'overlap': [0.0 for _ in range(32)],
#                      'guess_occ': [0.0 for _ in range(32)],
#                      'conv_occ': [0.0 for _ in range(32)],
#                     }
#                   }

""" SAVE PYSCF calculations in DB - ground-state calculations """
# geom_files = [base_dir + 'geometry_' + str(idx) + '/CASSCF/geom.xyz' for idx in range(200)]
# geom_files = [base_dir + 'geometry_' + str(idx) + '.xyz' for idx in range(2201)]
# mo_files = ['C:/Users/rhjva/imperial/pyscf_files/fulvene_scan_hf_F_big/geometry_' + str(idx) + '.npz' for idx in range(2201)]

# parse_pyscf_hf_calculations(geom_files=geom_files,
#                             mo_files=mo_files,
#                             db_path=db_path)

# with connect(db_path) as conn:
#   conn.metadata = {"_distance_unit": 'angstrom',
#                    "_property_unit_dict": {"orbital_coeffs": 1.0, "hf_guess": 1.0, "overlap": 1.0},
#                    "atomrefs": {
#                      'orbital_coeffs': [0.0 for _ in range(32)],
#                      'hf_guess': [0.0 for _ in range(32)],
#                      'overlap': [0.0 for _ in range(32)]
#                     }
#                   }




""" GENERATE SPLIT FILES """
data_idx = np.arange(n)
np.random.shuffle(data_idx)
train_idxs = data_idx[:int(train_split * n)]
print(len(train_idxs))
val_idxs = data_idx[int(train_split * n):int((train_split + val_split) * n)]
test_idxs = data_idx[int((train_split + val_split) * n):]

np.savez('./data/' + name + '.npz', 
  train_idx=train_idxs, 
  val_idx=val_idxs,
  test_idx=test_idxs)