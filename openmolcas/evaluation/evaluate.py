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
    # add HF Results
    result = np.load(base_dir + 'geometry_' + str(index) + '.npz', allow_pickle=True)
    results.append(PySCFResult(ml=False,
                               index=index,
                               t_tot=result['t_tot'],
                               imacro=result['imacro'],
                               mo_coeffs=result['mo_coeffs'],
                               S=result['S'],
                               fci_vec=result['fcivec']))

    # add HF_ML Results
    result = np.load(base_dir + 'geometry_' + str(index) + '_ML.npz', allow_pickle=True)
    if result['imacro'] != None: 
      results.append(PySCFResult(ml=True,
                                index=index,
                                t_tot=result['t_tot'],
                                imacro=result['imacro'],
                                mo_coeffs=result['mo_coeffs'],
                                S=result['S'],
                                fci_vec=result['fcivec']))

  hf_results = list(filter(lambda x: not x.ml, results))
  ml_results = list(filter(lambda x: x.ml, results))
  return hf_results, ml_results