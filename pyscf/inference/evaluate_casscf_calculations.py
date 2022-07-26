from pyscf import gto, scf, mcscf
import numpy as np
import os
import time
import scipy
import multiprocessing
from tqdm import tqdm
from models.inference import predict_guess, predict_guess_F

CUTOFF = 5.0

METHODS = [
  'huckel',
  'hf',
  'ML_MO',
  'ML_F'
]

def calculate_overlap_matrix(geometry_path: str, basis: str) -> np.ndarray:
  mol = gto.M(atom=geometry_path,
              basis=basis,
              spin=0)
  myscf = mol.RHF()
  return myscf.get_ovlp(mol)

def casscf_calculation(geom_file, initial_guess='ao_min', basis='sto-6g'):

  fulvene = gto.M(atom=geom_file,
                basis=basis,
                spin=0,
                symmetry=True)

  myhf = fulvene.RHF()
  S = myhf.get_ovlp(fulvene)

  # initiate CASSCF object
  n_states = 2
  weights = np.ones(n_states) / n_states
  mcas = myhf.CASSCF(ncas=6, nelecas=6).state_average(weights)
  mcas.conv_tol = 1e-8
  
  myhf.kernel()

  if initial_guess == 'huckel':
    guess = scf.hf.init_guess_by_huckel(fulvene)
    mo = mcscf.project_init_guess(mcas, guess)
    mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)
  elif initial_guess == 'hf':
    guess = myhf.mo_coeff
    mo = mcscf.project_init_guess(mcas, myhf.mo_coeff)
    mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)
  else:
    guess = initial_guess
    mo = mcscf.project_init_guess(mcas, guess)
    mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)

  tstart = time.time()
  (imacro, _, _, fcivec, mo_coeffs, mo_energy) = mcas.kernel(mo)
  t_tot = time.time() - tstart

  F = mcas.get_fock(mo_coeff=mo, ci=fcivec)
  mo_e, _ = scipy.linalg.eigh(F, S)
  mo_e = np.abs(mo_energy - mo_e)

  return t_tot, imacro, mo_e, mo_coeffs, guess, S


def run_pyscf_calculations(args):
  method_name, geom_file, index, basis = args

  if method_name not in METHODS:
    raise ValueError("method name not found")
  
  if method_name == 'huckel':
    (t_tot, imacro, mo_e, mo_coeffs, guess, S) = casscf_calculation(geom_file, initial_guess='huckel', basis=basis)
  elif method_name == 'hf':
    (t_tot, imacro, mo_e, mo_coeffs, guess, S) = casscf_calculation(geom_file, initial_guess='hf', basis=basis)
  elif method_name == 'ML_MO':
    initial_guess = predict_guess(model_path=model_path, geometry_path=geom_file, basis=n_mo)
    (t_tot, imacro, mo_e, mo_coeffs, guess, S) = casscf_calculation(geom_file, initial_guess=initial_guess, basis=basis)
  elif method_name == 'ML_F':
    overlap_matrix = calculate_overlap_matrix(geometry_path=geometry_path, basis=basis)
    initial_guess = predict_guess_F(model_path=model_path, geometry_path=geom_file, S=overlap_matrix, basis=n_mo)
    (t_tot, imacro, mo_e, mo_coeffs, guess, S) = casscf_calculation(geom_file, initial_guess=initial_guess, basis=basis)
  
  np.savez(output_dir + 'geometry_' + str(index) + '.npz',
          t_tot=t_tot,
          imacro=imacro,
          mo_e=mo_e,
          mo_coeffs=mo_coeffs,
          guess=guess,
          S=S)

  return 'succes'
  
  

if __name__ == "__main__":
  base_dir = '/home/ubuntu/fulvene/geometries/geom_scan_200/'
  split_file = '../../data/geom_scan_200_molcas.npz'
  print_iterations = True
  
  # models = ['geom_scan_200_4-31G_ML_U', 'geom_scan_200_4-31G_ML_F'] # ['', '', 'geom_scan_200_4-31G_ML_MO', 'geom_scan_200_4-31G_ML_U' 'geom_scan_200_4-31G_ML_F']
  # method_names = ['ML_U', 'ML_F'] #  ['ao_min', 'hf', 'ML_MO', 'ML_U', 'ML_F']
  # outputs = ['geom_scan_200_ML_U_4-31G', 'geom_scan_200_ML_F_4-31G'] # ['geom_scan_200_ao_min_4-31G', 'geom_scan_200_hf_4-31G', 'geom_scan_200_ML_MO_4-31G', 'geom_scan_200_ML_U_4-31G', 'geom_scan_200_ML_F_4-31G']
  
  models = ['gs200_pyscf_4-31G_ML_MO']
  method_names = ['ML_MO']
  outputs = ['gs200_pyscf_4-31G_ML_MO']

  basis = '4-31G'
  n_mo = 66
  N_JOBS = 8
  
  for idx, (model, method_name, output) in enumerate(zip(models, method_names, outputs)):
    model_path = '../../checkpoints/pyscf-runs/' + model + '.pt'
    output_dir = '/home/ubuntu/fulvene/casscf_calculations/experiments/' + output + '/'

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    indices = np.load(split_file)['val_idx']

    # parallel
    parallel_args = [(method_name, base_dir + 'geometry_' + str(index) + '.xyz', index, basis) for index in indices]
    pool = multiprocessing.Pool(N_JOBS)
    for result in tqdm(pool.imap(run_pyscf_calculations, parallel_args), total=len(parallel_args)):
      print(result)

    # serial
    # for i, index in enumerate(indices):
    #   geom_file = base_dir + 'geometry_' + str(index) + '.xyz'
    #   run_pyscf_calculations((method_name, geom_file, index, basis))
    #   print(idx / len(models) * 100, '% total   ', i / len(indices) * 100, '% job')

    if print_iterations:
      iterations = [np.load(output_dir + 'geometry_' + str(index) + '.npz')['imacro'] for index in indices]
      print(iterations)
      print(model, method_name, np.mean(iterations), np.std(iterations))

      iterations = [np.mean(np.load(output_dir + 'geometry_' + str(index) + '.npz')['mo_e']) for index in indices]
      print(model, method_name, np.mean(iterations), np.std(iterations))
