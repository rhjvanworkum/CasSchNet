from db.utils import correct_phase, order_orbitals
from openmolcas.utils import read_in_orb_file
from ase.db import connect
import h5py
import numpy as np

from src.schnetpack.data.parser import xyz_to_db

def parse_molcas_calculations(geom_files, rasorb_files, gssorb_files, hdf5_file_path, db_path, n_basis, apply_phase_correction=True):
  # read in HDF5
  hdf5_files = [h5py.File(hdf5_file, 'r') for hdf5_file in hdf5_file_path]
  all_overlap = [hdf5.get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_basis) for hdf5 in hdf5_files]
  all_natural_orbs = [hdf5.get('MO_VECTORS')[:].reshape(-1, n_basis) for hdf5 in hdf5_files]

  # read in GUESS
  all_guess = []
  for gssorb_file in gssorb_files:
    orbitals, _ = read_in_orb_file(gssorb_file)
    all_guess.append(orbitals)

  # read in CANONICAL MO COEFFS
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
              molecular_properties=[{'mo_coeffs': all_natural_orbs[idx].flatten(),
                                     'mo_coeffs_adjusted': all_mo_coeffs_adjusted[idx].flatten(), 
                                     'F': all_fock[idx].flatten(),
                                     'guess': all_guess[idx], 
                                     'overlap': all_overlap[idx]}])

def save_molcas_calculations_to_db(geometry_base_dir, calculations_base_dir, geometry_idxs, db_path, n_basis):
  geom_files = [geometry_base_dir + 'geometry_' + str(idx) + '.xyz' for idx in geometry_idxs]
  rasorb_files = [calculations_base_dir + 'geometry_' + str(idx) + '/CASSCF.RasOrb' for idx in geometry_idxs]
  gssorb_files = [calculations_base_dir + 'geometry_' + str(idx) + '/CASSCF.GssOrb' for idx in geometry_idxs]
  hdf5_file_path = [calculations_base_dir + 'geometry_' + str(idx) + '/CASSCF.rasscf.h5' for idx in geometry_idxs]

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
  prefix = '/home/ubuntu/'
  geometry_base_dir = prefix + 'fulvene/geometries/MD_trajectories_5_01/'
  calculations_base_dir = prefix + 'fulvene/openmolcas_calculations/MD_trajectory_1/'
  geometry_idxs = np.arange(200)
  n_basis = 36
  db_path = './data/MD01_molcas_ANO-S-MB.db'

  save_molcas_calculations_to_db(geometry_base_dir, calculations_base_dir, geometry_idxs, db_path, n_basis=n_basis)