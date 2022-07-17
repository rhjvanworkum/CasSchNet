import numpy as np

import numpy as np
import matplotlib.pyplot as plt

class PySCFResult:
  def __init__(self, ml, index, t_tot, imacro, mo_coeffs, S, fci_vec=None) -> None:
    self.ml = ml
    self.index = index
    self.t_tot = t_tot
    self.imacro = imacro
    self.mo_coeffs = mo_coeffs
    self.S = S
    self.fci_vec = fci_vec

def extract_results(split_file, base_dir):
  indices = np.load(split_file)['val_idx']

  results = []
  for index in indices:
    result = np.load(base_dir + 'geometry_' + str(index) + '.npz', allow_pickle=True)
    results.append(PySCFResult(ml=False,
                              index=index,
                              t_tot=result['t_tot'],
                              imacro=result['imacro'],
                              mo_coeffs=result['mo_coeffs'],
                              S=result['S'],
                              fci_vec=result['fcivec']))
  return results


if __name__ == "__main__":
  split_file = 'data/geom_scan_200_molcas.npz'
  dirs = ['/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/mol-runs/gs200_ANOSMB_ML_MO/',
          '/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/mol-runs/gs200_ANOSMB_ML_U/',
          '/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/mol-runs/gs200_ANOSMB_ML_F/']
  names = ['temp/gs200_ANOSMB_ML_MO.npy',
           'temp/gs200_ANOSMB_ML_U.npy',
           'temp/gs200_ANOSMB_ML_F.npy']

  for dir, name in zip(dirs, names):
    results = extract_results(split_file, dir)
    iterations = [result.imacro for result in results]
    np.save(name, np.array(iterations))