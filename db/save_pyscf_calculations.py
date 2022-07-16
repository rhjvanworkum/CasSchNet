
from ase.db import connect
import numpy as np
from db.utils import order_orbitals

from src.schnetpack.data.parser import xyz_to_db

def parse_pyscf_calculations(geom_files, mo_files, db_path, apply_phase_correction=True):
  all_mo_coeffs = [np.load(mo_file)['mo_coeffs'] for mo_file in mo_files]
  all_fock = [np.load(mo_file)['F'] for mo_file in mo_files]
  all_guess = [np.load(mo_file)['guess'] for mo_file in mo_files]
  all_overlap = [np.load(mo_file)['S'] for mo_file in mo_files]
  
  # apply phase correction
  if apply_phase_correction:
    for idx, orbitals in enumerate(all_mo_coeffs):
      if idx > 0:
        all_mo_coeffs[idx] = order_orbitals(all_mo_coeffs[idx-1], orbitals)

  for idx, geom_file in enumerate(geom_files):
    xyz_to_db(geom_file,
              db_path,
              idx=idx,
              atomic_properties="",
              molecular_properties=[{'mo_coeffs': all_mo_coeffs[idx].flatten(), 
                                     'F': all_fock[idx].flatten(),
                                     'hf_guess': all_guess[idx], 
                                     'overlap': all_overlap[idx]}])

def save_pyscf_calculations_to_db(geometry_base_dir, calculations_base_dir, n_geometries, db_path):
  geom_files = [geometry_base_dir + 'geometry_' + str(idx) + '.xyz' for idx in range(n_geometries)]
  mo_files = [calculations_base_dir + 'geometry_' + str(idx) + '.npz' for idx in range(n_geometries)]

  parse_pyscf_calculations(geom_files=geom_files,
                              mo_files=mo_files,
                              db_path=db_path,
                              apply_phase_correction=True)

  with connect(db_path) as conn:
    conn.metadata = {"_distance_unit": 'angstrom',
                    "_property_unit_dict": {"mo_coeffs": 1.0, "F": 1.0, "hf_guess": 1.0, "overlap": 1.0},
                    "atomrefs": {
                      'mo_coeffs': [0.0 for _ in range(32)],
                      'F': [0.0 for _ in range(32)],
                      'hf_guess': [0.0 for _ in range(32)],
                      'overlap': [0.0 for _ in range(32)]
                      }
                    }

if __name__ == "__main__":
  geometry_base_dir = '/home/ubuntu/fulvene/geometries/geom_scan_200/'
  calculations_base_dir = '/home/ubuntu/fulvene/casscf_calculations/geom_scan_200_6-31G*/'
  n_geometries = 200
  db_path = './data/geom_scan_200_6-31G*.db'
  
  save_pyscf_calculations_to_db(geometry_base_dir, calculations_base_dir, n_geometries, db_path)