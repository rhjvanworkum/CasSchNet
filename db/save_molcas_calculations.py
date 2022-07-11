from src.schnetpack.data.parser import parse_molcas_calculations, parse_molcas_calculations_canonical
from ase.db import connect

def save_mo_coeffs_to_db(geometry_base_dir, calculations_base_dir, n_geometries, db_path):
  geom_files = [geometry_base_dir + 'geometry_' + str(idx) + '.xyz' for idx in range(n_geometries)]
  hdf5_file_path = [calculations_base_dir + 'geometry_' + str(idx) + '/CASSCF.rasscf.h5' for idx in range(n_geometries)]

  parse_molcas_calculations(geom_files=geom_files,
                              hdf5_file_path=hdf5_file_path,
                              db_path=db_path,
                              save_property='MO_VECTORS',
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
  hdf5_file_path = [calculations_base_dir + 'geometry_' + str(idx) + '/CASSCF.rasscf.h5' for idx in range(n_geometries)]

  parse_molcas_calculations(geom_files=geom_files,
                              hdf5_file_path=hdf5_file_path,
                              db_path=db_path,
                              save_property='AO_FOCKINT_MATRIX')

  with connect(db_path) as conn:
    conn.metadata = {"_distance_unit": 'angstrom',
                    "_property_unit_dict": {"AO_FOCKINT_MATRIX": 1.0, "hf_guess": 1.0, "overlap": 1.0},
                    "atomrefs": {
                      'AO_FOCKINT_MATRIX': [0.0 for _ in range(32)],
                      'hf_guess': [0.0 for _ in range(32)],
                      'overlap': [0.0 for _ in range(32)]
                      }
                    }

def save_fock_to_db(geometry_base_dir, calculations_base_dir, n_geometries, db_path, use_overlap=True):
  geom_files = [geometry_base_dir + 'geometry_' + str(idx) + '.xyz' for idx in range(n_geometries)]
  rasorb_files = [calculations_base_dir + 'geometry_' + str(idx) + '/CASSCF.RasOrb' for idx in range(n_geometries)]
  hdf5_file_path = [calculations_base_dir + 'geometry_' + str(idx) + '/CASSCF.rasscf.h5' for idx in range(n_geometries)]

  parse_molcas_calculations_canonical(geom_files=geom_files,
                            rasorb_files=rasorb_files,
                            hdf5_file_path=hdf5_file_path,
                            db_path=db_path,
                            save_property='F',
                            use_overlap=use_overlap)

  with connect(db_path) as conn:
    conn.metadata = {"_distance_unit": 'angstrom',
                    "_property_unit_dict": {"F": 1.0, "overlap": 1.0, "energies": 1.0},
                    "atomrefs": {
                      'F': [0.0 for _ in range(32)],
                      'overlap': [0.0 for _ in range(32)],
                      'energies': [0.0 for _ in range(32)]
                      }
                    }

if __name__ == "__main__":
  geometry_base_dir = 'C:/users/rhjva/imperial/fulvene/geometries/geom_scan_2000/'
  calculations_base_dir = 'C:/Users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_2000/'
  n_geometries = 199
  db_path = './data/geom_scan_199_molcas_fock_noncan.db'

  # save_mo_coeffs_to_db(geometry_base_dir, calculations_base_dir, n_geometries, db_path)
  # save_hamiltonian_to_db(geometry_base_dir, calculations_base_dir, n_geometries, db_path)
  save_fock_to_db(geometry_base_dir, calculations_base_dir, n_geometries, db_path, use_overlap=True)