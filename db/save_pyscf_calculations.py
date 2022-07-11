from src.schnetpack.data.parser import parse_pyscf_calculations
from ase.db import connect

def save_mo_coeffs_to_db(geometry_base_dir, calculations_base_dir, n_geometries, db_path):
  geom_files = [geometry_base_dir + 'geometry_' + str(idx) + '.xyz' for idx in range(n_geometries)]
  mo_files = [calculations_base_dir + 'geometry_' + str(idx) + '.npz' for idx in range(n_geometries)]

  parse_pyscf_calculations(geom_files=geom_files,
                              mo_files=mo_files,
                              db_path=db_path,
                              save_property='mo_coeffs',
                              apply_phase_correction=True)

  with connect(db_path) as conn:
    conn.metadata = {"_distance_unit": 'angstrom',
                    "_property_unit_dict": {"mo_coeffs": 1.0, "hf_guess": 1.0, "overlap": 1.0},
                    "atomrefs": {
                      'mo_coeffs': [0.0 for _ in range(32)],
                      'hf_guess': [0.0 for _ in range(32)],
                      'overlap': [0.0 for _ in range(32)]
                      }
                    }

def save_hamiltonian_to_db(geometry_base_dir, calculations_base_dir, n_geometries, db_path):
  geom_files = [geometry_base_dir + 'geometry_' + str(idx) + '.xyz' for idx in range(n_geometries)]
  mo_files = [calculations_base_dir + 'geometry_' + str(idx) + '.npz' for idx in range(n_geometries)]

  parse_pyscf_calculations(geom_files=geom_files,
                              mo_files=mo_files,
                              db_path=db_path,
                              save_property='F')

  with connect(db_path) as conn:
    conn.metadata = {"_distance_unit": 'angstrom',
                    "_property_unit_dict": {"F": 1.0, "hf_guess": 1.0, "overlap": 1.0},
                    "atomrefs": {
                      'F': [0.0 for _ in range(32)],
                      'hf_guess': [0.0 for _ in range(32)],
                      'overlap': [0.0 for _ in range(32)]
                      }
                    }

if __name__ == "__main__":
  geometry_base_dir = 'C:/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  calculations_base_dir = 'C:/Users/rhjva/imperial/fulvene/casscf_calculations/geom_scan_200_sto_3g/'
  n_geometries = 200
  db_path = './data/geom_scan_200_casscf_sto3g_hamiltonian.db'

  # save_mo_coeffs_to_db(geometry_base_dir, calculations_base_dir, n_geometries, db_path)
  save_hamiltonian_to_db(geometry_base_dir, calculations_base_dir, n_geometries, db_path)