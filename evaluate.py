import numpy as np
from ase.db import connect

from infer_model import predict_orbital_coeffs
from calculate_evaluation import get_orbital_coeffs

def rmse(orbs1, orbs2):
  return np.linalg.norm(orbs1 - orbs2)
  # error = 0
  # for idx in range(orbs1.shape[0]):
  #   error += (orbs2[idx] - orbs1) ** 2

  # return error / len(orbs1)

def normalise_rows(mat):
  "Normalise each row of mat"
  return np.array(tuple(map(lambda v:v/np.linalg.norm(v),mat)))

def get_rmse(dir, model_path):
  ml_coeffs = predict_orbital_coeffs(geometry_path=dir + 'CASSCF/geom.xyz',
                                  model_path=model_path,
                                  cutoff=5.0).reshape(36, -1)

  # ref_coeffs = get_orbital_coeffs(dir + 'CASSCF/CASSCF.RasOrb')

  # ref_coeffs_norm = normalise_rows(ref_coeffs)
  # ml_coeffs_norm = normalise_rows(ml_coeffs)

  # overlap = ref_coeffs_norm.dot(ml_coeffs_norm.T)
  # orb_order = abs(overlap).argmax(axis=1)

  # ml_coeffs = ml_coeffs[orb_order]



  # db_path = './data/fulvene_scan_2.db'
  # with connect(db_path) as conn:
  #   workdir = conn.get(idx=int(idx)).data['workdir']
  #   ml_coeffs = np.array(conn.get(idx=int(idx)).data['orbital_coeffs']) # .reshape(-1, 36)

  converged_coeffs = get_orbital_coeffs(dir + 'CASSCF/CASSCF.RasOrb')

  return rmse(ml_coeffs.flatten(), converged_coeffs.flatten())

if __name__ == "__main__":
  # compare to models here on train/val/test set on fulvene_scan_2
  n_mo = 36
  cutoff = 5.0

  model = 'fulvene_scan_2_deep_uncorrected.pt'
  base_dir = 'C:/users/rhjva/imperial/molcas_files/fulvene_scan_2/'
  split_file = './data/fulvene_scan_140.npz'
  
  indices = np.load(split_file)
  train_idx = indices['train_idx']
  val_idx = indices['val_idx']
  test_idx = indices['test_idx']


  train_rmse = np.mean([get_rmse(base_dir + 'geometry_' + str(idx) + '/', './checkpoints/' + model) for idx in train_idx])
  val_rmse = np.mean([get_rmse(base_dir + 'geometry_' + str(idx) + '/', './checkpoints/' + model) for idx in val_idx])
  test_rmse = np.mean([get_rmse(base_dir + 'geometry_' + str(idx) + '/', './checkpoints/' + model) for idx in test_idx])

  # print(val_rmse)
  print(train_rmse, val_rmse, test_rmse)