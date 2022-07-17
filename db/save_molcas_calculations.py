from db.utils import order_orbitals
from openmolcas.utils import read_in_orb_file
from ase.db import connect
import h5py
import numpy as np

from src.schnetpack.data.parser import xyz_to_db

def parse_molcas_calculations(geom_files, rasorb_files, gssorb_files, hdf5_file_path, db_path, n_basis, apply_phase_correction=True):
  # read in HDF5
  hdf5_files = [h5py.File(hdf5_file) for hdf5_file in hdf5_file_path]
  all_overlap = [hdf5.get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_basis) for hdf5 in hdf5_files]

  # read in GUESS
  all_guess = []
  for gssorb_file in gssorb_files:
    orbitals, _ = read_in_orb_file(gssorb_file)
    all_guess.append(orbitals)

  # read in MO COEFFS
  all_mo_coeffs = []
  all_energies = []
  for rasorb_file in rasorb_files:
    orbitals, energies = read_in_orb_file(rasorb_file)
    all_mo_coeffs.append(orbitals)
    all_energies.append(energies)
  
  # apply phase correction
  all_mo_coeffs_adjusted = []
  if apply_phase_correction:
    for idx, orbitals in enumerate(all_mo_coeffs):
      if idx > 0:
        all_mo_coeffs_adjusted.append(order_orbitals(all_mo_coeffs[idx-1], orbitals))
      else:
        all_mo_coeffs_adjusted.append(orbitals)

  # get fock matrices
  all_fock = []
  for idx in range(len(geom_files)):
    S = all_overlap[idx]
    energies = all_energies[idx]
    orbitals = all_mo_coeffs[idx]
    all_fock.append(np.matmul(S, np.matmul(orbitals.T, np.matmul(np.diag(energies), np.linalg.inv(orbitals.T)))))

  for idx, geom_file in enumerate(geom_files):
    xyz_to_db(geom_file,
              db_path,
              idx=idx,
              atomic_properties="",
              molecular_properties=[{'mo_coeffs': all_mo_coeffs[idx].flatten(),
                                     'mo_coeffs_adjusted': all_mo_coeffs_adjusted[idx].flatten(), 
                                     'F': all_fock[idx].flatten(),
                                     'guess': all_guess[idx], 
                                     'overlap': all_overlap[idx]}])

def save_molcas_calculations_to_db(geometry_base_dir, calculations_base_dir, n_geometries, db_path, n_basis):
  geom_files = [geometry_base_dir + 'geometry_' + str(idx) + '.xyz' for idx in range(n_geometries)]
  rasorb_files = [calculations_base_dir + 'geometry_' + str(idx) + '/CASSCF.RasOrb' for idx in range(n_geometries)]
  gssorb_files = [calculations_base_dir + 'geometry_' + str(idx) + '/CASSCF.GssOrb' for idx in range(n_geometries)]
  hdf5_file_path = [calculations_base_dir + 'geometry_' + str(idx) + '/CASSCF.rasscf.h5' for idx in range(n_geometries)]

  parse_molcas_calculations(geom_files=geom_files,
                              rasorb_files=rasorb_files,
                              gssorb_files=gssorb_files,
                              hdf5_file_path=hdf5_file_path,
                              db_path=db_path,
                              n_basis=n_basis,
                              apply_phase_correction=True)

  with connect(db_path) as conn:
    conn.metadata = {"_distance_unit": 'angstrom',
                    "_property_unit_dict": {"mo_coeffs": 1.0, "mo_coeffs_adjusted": 1.0, "F": 1.0, "guess": 1.0, "overlap": 1.0},
                    "atomrefs": {
                      'mo_coeffs': [0.0 for _ in range(32)],
                      'mo_coeffs_adjusted': [0.0 for _ in range(32)],
                      'F': [0.0 for _ in range(32)],
                      'guess': [0.0 for _ in range(32)],
                      'overlap': [0.0 for _ in range(32)]
                      }
                    }

if __name__ == "__main__":
  geometry_base_dir = 'C:/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  calculations_base_dir = 'C:/Users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200_ANO-S-MB_canonical/'
  n_geometries = 200
  n_basis = 36
  db_path = './data/geom_scan_200_molcas_ANO-S-MB_canonical.db'

  save_molcas_calculations_to_db(geometry_base_dir, calculations_base_dir, n_geometries, db_path, 36)