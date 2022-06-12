import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

class PySCFResult:

  def __init__(self, ml, index, t_tot, imacro, guess, converged, S, P_guess, P_conv, occ_guess, occ_conv) -> None:
    self.ml = ml
    self.index = index
    self.t_tot = t_tot
    self.imacro = imacro
    self.guess = guess
    self.converged = converged
    self.S = S
    self.P_guess = P_guess
    self.P_conv = P_conv
    self.occ_guess = occ_guess
    self.occ_conv = occ_conv

# def dot(vec1, vec2):
#   return np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def density_matrix(orbitals, occ):
  P = np.zeros((orbitals.shape[0], orbitals.shape[1]))
  for i in range(orbitals.shape[0]):
    for j in range(orbitals.shape[1]):
      for idx, k in enumerate(occ):
        P[i, j] += k * orbitals[idx, i] * orbitals[idx, j] 
  return P

def projection(P_guess, P_conv, S):
  return np.trace(np.matmul(P_guess, np.matmul(S, np.matmul(P_conv, S))))



def error(result):
  P_guess = density_matrix(result.guess, result.occ_guess)
  P_conv = density_matrix(result.converged, result.occ_conv)
  return projection(P_guess, P_conv, result.S)
  # return projection(result.P_guess, result.P_conv, result.S)

  # return (np.sum([result.occ_conv[i] * result.S[i, i] for i in range(len(result.occ_conv))]) - np.sum([result.occ_guess[i] * result.S[i, i] for i in range(len(result.occ_guess))])) ** 2

  # return np.sum([np.dot(orbs1[idx], orbs2[idx]) for idx in range(36)]) / 36

def print_out_results(results):
  ML_results = list(filter(lambda x: x.ml, results))
  print(f'CASSCF_ML ===> T_tot: {round(np.mean([result.t_tot for result in ML_results]), 2)} +/- {round(np.std([result.t_tot for result in ML_results]), 2)}' +
                      f' Iterations: {round(np.mean([result.imacro for result in ML_results]), 2)} +/- {round(np.std([result.imacro for result in ML_results]), 2)} \n')
  
  CASSCF_results = list(filter(lambda x: not x.ml, results))
  print(f'CASSCF    ===> WallTime: {round(np.mean([result.t_tot for result in CASSCF_results]), 2)} +/- {round(np.std([result.t_tot for result in CASSCF_results]), 2)}' +
                      f' Iterations: {round(np.mean([result.imacro for result in CASSCF_results]), 2)} +/- {round(np.std([result.imacro for result in CASSCF_results]), 2)} \n')

  print(f'MSE of    ML_coeffs & converged_coeffs: {round(np.mean([error(result) for result in ML_results]), 4)} +/- {round(np.std([error(result) for result in ML_results]), 4)}')
  print(f'MSE of guess_coeffs & converged_coeffs: {round(np.mean([error(result) for result in CASSCF_results]), 4)} +/- {round(np.std([error(result) for result in CASSCF_results]), 4)}')

def log_all_results(results):
  string = ""

  for idx, result in enumerate(results):
    string += f"calculation {result.index}" + " ML: " + str(result.ml) + "\n"
    string += f"RMSE of ML_coeffs: {error(result)} \n" 
    string += f"iterations: {result.imacro} \n" 
    string += f"timing: {result.t_tot} \n" 
    string += "\n"

  return string

def plot_rmse_timing(results):
  CASSCF_results = list(filter(lambda x: not x.ml, results))
  ML_results = list(filter(lambda x: x.ml, results))
  ML_rmse = [error(result) for result in ML_results]
  wall_timings = [result.t_tot for result in ML_results]
  # wall_timings = [result.t_tot - mlresult.t_tot for (result, mlresult) in zip(CASSCF_results, ML_results)]

  p = sn.regplot(x=ML_rmse, y=wall_timings)
  plt.xlabel("Projection of guess & CASSCF converged coeffs")
  plt.ylabel("total Time of CASSCF_ML method")
  plt.title('total Time vs. Projection')
  plt.show()

  CASSCF_results = list(filter(lambda x: not x.ml, results))
  ML_rmse = [error(result) for result in CASSCF_results]
  wall_timings = [result.t_tot for result in CASSCF_results]

  p = sn.regplot(x=ML_rmse, y=wall_timings)
  plt.xlabel("Projection of guess & CASSCF converged coeffs")
  plt.ylabel("total Time of CASSCF_ML method")
  plt.title('total Time vs. Projection')
  plt.show()

  ML_rmse = [error(result) for result in results]
  wall_timings = [result.t_tot for result in results]

  p = sn.regplot(x=ML_rmse, y=wall_timings)
  plt.xlabel("Projection of guess & CASSCF converged coeffs")
  plt.ylabel("total Time of CASSCF_ML method")
  plt.title('total Time vs. Projection')
  plt.show()

if __name__ == "__main__":
  base_dir = 'C:/Users/rhjva/imperial/pyscf_files/fulvene_scan_pyscf_projection/'
  split_file = './data/fulvene_MB_140.npz'

  indices = np.load(split_file)['test_idx']

  results = []
  for index in indices:

    # add CASSCF Results
    result = np.load(base_dir + 'geometry_' + str(index) + '.npz')
    results.append(PySCFResult(ml=False,
                               index=index,
                               t_tot=result['t_tot'],
                               imacro=result['imacro'],
                               guess=np.load(base_dir + 'geometry_' + str(index) + '_hf_guess.npy'),
                               converged=result['mo_coeffs'],
                               S=np.load(base_dir + 'geometry_' + str(index) + '_overlap.npy'),
                               P_guess=np.load(base_dir + 'geometry_' + str(index) + '_hf_guess_P.npy'),
                               P_conv=np.load(base_dir + 'geometry_' + str(index) + '_hf_conv_P.npy'),
                               occ_guess=np.load(base_dir + 'geometry_' + str(index) + '_guess_natocc.npy'),
                               occ_conv=np.load(base_dir + 'geometry_' + str(index) + '_hf_conv_natocc.npy')))

    # add CASSCF_ML Results
    result = np.load(base_dir + 'geometry_' + str(index) + '_ML.npz')
    results.append(PySCFResult(ml=True,
                               index=index,
                               t_tot=result['t_tot'],
                               imacro=result['imacro'],
                               guess=np.load(base_dir + 'geometry_' + str(index) + '_ml_guess.npy'),
                               converged=result['mo_coeffs'],
                               S=np.load(base_dir + 'geometry_' + str(index) + '_overlap.npy'),
                               P_guess=np.load(base_dir + 'geometry_' + str(index) + '_ml_guess_P.npy'),
                               P_conv=np.load(base_dir + 'geometry_' + str(index) + '_ml_conv_P.npy'),
                               occ_guess=np.load(base_dir + 'geometry_' + str(index) + '_guess_natocc.npy'),
                               occ_conv=np.load(base_dir + 'geometry_' + str(index) + '_ml_conv_natocc.npy')))

  # correct_orbitals(results, guess)

  with open('pyscf_output.txt', 'w') as f:
    f.writelines(log_all_results(results))

  # for result in results:
  #   print(result.index, result.ml, result.imacro, result.t_tot)
  #   print(error(result), '\n')

  print_out_results(results)
  plot_rmse_timing(results)