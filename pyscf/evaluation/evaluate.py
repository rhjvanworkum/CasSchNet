import numpy as np
import matplotlib.pyplot as plt

class PySCFResult:
  def __init__(self, index, t_tot, imacro, guess, mo_coeffs, S, fci_vec=None, P_guess=None, P_conv=None, occ_guess=None, occ_conv=None) -> None:
    self.index = index
    self.t_tot = t_tot
    self.imacro = imacro
    self.guess = guess
    self.mo_coeffs = mo_coeffs
    self.S = S
    self.fci_vec = fci_vec
    self.P_guess = P_guess
    self.P_conv = P_conv
    self.occ_guess = occ_guess
    self.occ_conv = occ_conv

def extract_results(split_file, base_dir):
  indices = np.load(split_file)['val_idx']

  results = []
  for index in indices:
    # add HF Results
    result = np.load(base_dir + 'geometry_' + str(index) + '.npz')
    results.append(PySCFResult(index=index,
                               t_tot=result['t_tot'],
                               imacro=result['imacro'],
                               guess=result['guess'],
                               mo_coeffs=result['mo_coeffs'],
                               S=result['S'],))
                              #  fci_vec=result['fcivec']))

  return results