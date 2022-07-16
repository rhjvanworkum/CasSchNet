from src.schnetpack.data.parser import parse_pyscf_calculations
from ase.db import connect

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