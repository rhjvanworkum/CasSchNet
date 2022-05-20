from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import math
import os
from ase.db import connect

from src.schnetpack.data.parser import order_orbitals, read_in_orb_file, correct_phase


class MolcasResults:
  def __init__(self, rasscf_timing, wall_timing, n_iterations=None):
    self.rasscf_timing = rasscf_timing
    self.wall_timing = wall_timing
    self.n_iterations = n_iterations

class ExperimentResults:
  def __init__(self, 
    index, 
    casci_result: MolcasResults, 
    casscf_ml_result: MolcasResults, 
    casscf_result: MolcasResults,
    ml_orb_coeffs: np.ndarray,
    ml_converged_coeffs: np.ndarray,
    guess_orb_coeffs: np.ndarray,
    converged_coeffs: np.ndarray,
    converged_s1_energy: float = None):
      self.index = index
      self.casci_result = casci_result
      self.casscf_ml_result = casscf_ml_result
      self.casscf_result = casscf_result
      self.ml_orb_coeffs = ml_orb_coeffs
      self.ml_converged_coeffs = ml_converged_coeffs
      self.guess_orb_coeffs = guess_orb_coeffs
      self.converged_coeffs = converged_coeffs
      self.converged_s1_energy = converged_s1_energy

def read_log_file(file, read_iterations=True):
  n_iterations = None
  rasscf_timing = None
  wall_timing = None

  with open(file, 'r') as f:
    while True:
      line = f.readline()

      if read_iterations:
        if "Convergence after" in line:
          for el in line.split():
            if el.isdigit():
              n_iterations = int(el)
              break

      if "--- Module rasscf spent" in line:
        for el in line.split():
          if el.isdigit():
            rasscf_timing = float(el)
            break

      if "Timing: Wall" in line:
        wall_timing = float(line.replace("=", " ").split()[2])
        break

  return MolcasResults(rasscf_timing, wall_timing, n_iterations)

def get_orbital_coeffs(file):
  return np.array(read_in_orb_file(file))

def get_s1_energy(file):
  with open(file, 'r') as f:
      while True:
        line = f.readline()

        if 'RASSCF root number  1 Total energy' in line:
          s1_energy = float(line.replace('\n', '').split(' ')[-1])
          break

  return s1_energy

def correct_orbitals(results : List[ExperimentResults], ref_file):
  ref = read_in_orb_file(ref_file)
  for idx, result in enumerate(results):
    results[idx].converged_coeffs = order_orbitals(ref, result.converged_coeffs)

def rmse(orbs1, orbs2):
  # return np.linalg.norm(orbs1 - orbs2)

  """ Magnitude weighted MSE """
  se = (orbs2.flatten() - orbs1.flatten()) ** 2
  wse = se / np.abs(orbs2.flatten())
  return np.sum(wse) / (36 ** 2)

  """ MSE """
  # return np.sum((orbs2.flatten() - orbs1.flatten()) ** 2) / len(orbs2.flatten())

  """ GAuSSIAN WEIGHTED AROUND ACTIVE SPACE"""
  # def gauss(x, x0, sigma):
  #   return 1000 * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
  # x = np.linspace(0, 1296, 1296)

  # return np.sum(np.multiply((orbs2.flatten() - orbs1.flatten()) ** 2, gauss(x, 774, 100))) / len(orbs2.flatten())
  
  # errors = (orbs2.flatten() - orbs1.flatten()) ** 2
  # errors = errors[:(20 * 36)]
  # top_n = 3
  # top_ind = np.argpartition(errors, -top_n)[-top_n:]
  # return np.sum(errors[top_ind])

  # error = 0
  # weights = np.cos(np.linspace(0, 0.5 * np.pi, 36))
  # weights[20:] = 0
  # for i in range(36):
  #   for j in range(36):
  #     error += (orbs2[i][j] - orbs1[i][j]) ** 2 * weights[i]
  # return error / (36 ** 2)

  # error = 0
  # weights = np.load('weights.npy')
  # for i in range(36 ** 2):
  #   error += (orbs2.flatten()[i] - orbs1.flatten()[i]) ** 2 * weights[i]
  # return error / len(np.where(weights > 0)[0])

  # error = 0
  # weights = np.load('weights_abs.npy')
  # for i in range(36 ** 2):
  #   error += np.abs(orbs2.flatten()[i] - orbs1.flatten()[i]) * weights[i]
  # return error / len(np.where(weights > 0)[0])


def print_out_results(results: List[ExperimentResults]):
  print(f'CASCI_ML  ===> WallTime: {round(np.mean([result.casci_result.wall_timing for result in results]), 2)} +/- {round(np.std([result.casci_result.wall_timing for result in results]), 2)}' +
                      f' RASSCF-Time: {round(np.mean([result.casci_result.rasscf_timing for result in results]), 2)} +/- {round(np.std([result.casci_result.rasscf_timing for result in results]), 2)} \n')
  print(f'CASSCF_ML ===> WallTime: {round(np.mean([result.casscf_ml_result.wall_timing for result in results]), 2)} +/- {round(np.std([result.casscf_ml_result.wall_timing for result in results]), 2)}' +
                      f' RASSCF-Time: {round(np.mean([result.casscf_ml_result.rasscf_timing for result in results]), 2)} +/- {round(np.std([result.casscf_ml_result.rasscf_timing for result in results]), 2)}' +
                      f' Iterations: {round(np.mean([result.casscf_ml_result.n_iterations for result in results]), 2)} +/- {round(np.std([result.casscf_ml_result.n_iterations for result in results]), 2)} \n')
  print(f'CASSCF    ===> WallTime: {round(np.mean([result.casscf_result.wall_timing for result in results]), 2)} +/- {round(np.std([result.casscf_result.wall_timing for result in results]), 2)}' +
                      f' RASSCF-Time: {round(np.mean([result.casscf_result.rasscf_timing for result in results]), 2)} +/- {round(np.std([result.casscf_result.rasscf_timing for result in results]), 2)}' +
                      f' Iterations: {round(np.mean([result.casscf_result.n_iterations for result in results]), 2)} +/- {round(np.std([result.casscf_result.n_iterations for result in results]), 2)} \n')

  print(f'MSE of    ML_coeffs & converged_coeffs: {round(np.mean([rmse(result.ml_orb_coeffs, result.converged_coeffs) for result in results]), 4)} +/- {round(np.std([rmse(result.ml_orb_coeffs, result.converged_coeffs) for result in results]), 4)}')
  print(f'MSE of guess_coeffs & converged_coeffs: {round(np.mean([rmse(result.guess_orb_coeffs, result.converged_coeffs) for result in results]), 4)} +/- {round(np.std([rmse(result.guess_orb_coeffs, result.converged_coeffs) for result in results]), 4)}')


def log_all_results(results: List[ExperimentResults]):
  string = ""

  for idx, result in enumerate(results):
    string += f"calculation {result.index} \n"
    string += f"RMSE of ML_coeffs: {rmse(result.ml_orb_coeffs, result.converged_coeffs)} \n" 
    string += f"CASSCF_ML iterations: {result.casscf_ml_result.n_iterations} \n" 
    string += f"CASSCF_ML WallTime: {result.casscf_ml_result.wall_timing} \n" 
    string += f"RMSE of guess_coeffs: {rmse(result.guess_orb_coeffs, result.converged_coeffs)} \n" 
    string += f"CASSCF iterations: {result.casscf_result.n_iterations} \n" 
    string += f"CASSCF WallTime: {result.casscf_result.wall_timing} \n" 
    string += "\n"

  return string


def plot_rsme_calculations(results: List[ExperimentResults]):
  ML_rmse = [rmse(result.ml_orb_coeffs, result.converged_coeffs) for result in results]
  guess_rmse = [rmse(result.guess_orb_coeffs, result.converged_coeffs) for result in results]

  plt.scatter(np.arange(len(ML_rmse)), ML_rmse, label="ML guess")
  plt.scatter(np.arange(len(guess_rmse)), guess_rmse, label="Guess")
  plt.xlabel('Experiment n')
  plt.ylabel('MSE')
  plt.title('MSE of orbs coeffs between guess & converged coeffs')
  plt.legend()
  plt.show()

def plot_rmse_timing(results: List[ExperimentResults]):
  ML_rmse = [rmse(result.ml_orb_coeffs, result.converged_coeffs) for result in results]
  wall_timings = [result.casscf_ml_result.n_iterations for result in results]

  p = sn.regplot(x=ML_rmse, y=wall_timings)
  plt.xlabel("MSE of ML guess & CASSCF converged coeffs")
  plt.ylabel("N iterations of CASSCF_ML method")
  plt.title('N iterations vs. MSE')
  plt.show()

if __name__ == "__main__":
  """ Evaluate Model output """
  split_file = './data/fulvene_MB_140.npz'
  ref_file = 'C:/users/rhjva/imperial/molcas_files/fulvene_scan_2/geometry_0/CASSCF/CASSCF.RasOrb'
  output_dir = 'C:/Users/rhjva/imperial/molcas_files/fulvene_scan_2_experiment0_mse/'
  indices = np.load(split_file)['test_idx']

  results = []
  for i, index in enumerate(indices):
    results.append(
      ExperimentResults(
        index=index,
        casci_result=read_log_file(output_dir + 'geometry_' + str(index) + '/CASCI_ML/calc.log', read_iterations=False),
        casscf_ml_result=read_log_file(output_dir + 'geometry_' + str(index) + '/CASSCF_ML/calc.log'),
        casscf_result=read_log_file(output_dir + 'geometry_' + str(index) + '/CASSCF/calc.log'),
        ml_orb_coeffs=get_orbital_coeffs(output_dir + 'geometry_' + str(index) + '/CASSCF_ML/geom.orb'),
        ml_converged_coeffs=get_orbital_coeffs(output_dir + 'geometry_' + str(index) + '/CASSCF_ML/CASSCF_ML.RasOrb'),
        guess_orb_coeffs=get_orbital_coeffs(output_dir + 'geometry_' + str(index) + '/CASSCF/geom.orb'),
        converged_coeffs=get_orbital_coeffs(output_dir + 'geometry_' + str(index) + '/CASSCF/CASSCF.RasOrb')
      )
    )
    print('Progress: ', i / len(indices) * 100, '%')

  # sort the results
  results = sorted(results, key=lambda x: x.index)
  correct_orbitals(results, ref_file)

  with open('output.txt', 'w') as f:
    f.writelines(log_all_results(results))

  print_out_results(results)
  plot_rsme_calculations(results)
  plot_rmse_timing(results)

  """ Evaluate geometry scan """
  # output_dir = 'C:/Users/rhjva/imperial/molcas_files/fulvene_scan_4/'

  # results = []
  # for _, _, filenames in os.walk(output_dir):
  #   for index, filename in enumerate(filenames):
  #     if read_log_file(output_dir + filename.split('.')[0] + '/CASSCF_ML/calc.log').n_iterations is not None:
  #       results.append(
  #         ExperimentResults(
  #           index=int(filename.split('.')[0].split('_')[-1]),
  #           casci_result=read_log_file(output_dir + filename.split('.')[0] + '/CASCI_ML/calc.log', read_iterations=False),
  #           casscf_ml_result=read_log_file(output_dir + filename.split('.')[0] + '/CASSCF_ML/calc.log'),
  #           casscf_result=read_log_file(output_dir + filename.split('.')[0] + '/CASSCF/calc.log'),
  #           ml_orb_coeffs=get_orbital_coeffs(output_dir + filename.split('.')[0] + '/CASSCF_ML/geom.orb'),
  #           ml_converged_coeffs=get_orbital_coeffs(output_dir + filename.split('.')[0] + '/CASSCF_ML/CASSCF_ML.RasOrb'),
  #           guess_orb_coeffs=get_orbital_coeffs(output_dir + filename.split('.')[0] + '/CASSCF/geom.orb'),
  #           converged_coeffs=get_orbital_coeffs(output_dir + filename.split('.')[0] + '/CASSCF/CASSCF.RasOrb'),
  #           converged_s1_energy=get_s1_energy(output_dir + filename.split('.')[0] + '/CASSCF/calc.log')
  #         )
  #       )
  #   break

  # # sort the results
  # results = sorted(results, key=lambda x: x.index)
  # correct_orbitals(results)

  # print_out_results(results)
  # plot_rsme_calculations(results)
  # plot_rmse_timing(results) 




  # with open('output_lr_mse.txt', 'w') as f:
  #   f.writelines(log_all_results(results))

  # n_biggest = 10
  # mo = 21

  # ind = np.argpartition(results[0].converged_coeffs[mo], -n_biggest)[-n_biggest:]
  # ind = [19, 21, 26, 22, 9, 8, 32, 33, 2, 12]
  # for orb_coeff in ind:
  #   data = [result.converged_coeffs[mo][orb_coeff] for result in results]
  #   plt.plot(np.arange(len(data)), data, label="orb coeff " + str(orb_coeff))
  
  # plt.xlabel('geometry i')
  # plt.ylabel('orb coeff')
  # plt.legend()
  # plt.title('Orb coeff Scan of MO 20 - fulvene - converged CASSCF calc')
  # plt.show()

  # for orb_coeff in ind:
  #   data = [result.ml_orb_coeffs[mo][orb_coeff] for result in results]
  #   plt.plot(np.arange(len(data)), data, label="orb coeff " + str(orb_coeff))
  
  # plt.xlabel('geometry i')
  # plt.ylabel('orb coeff')
  # plt.legend()
  # plt.title('Orb coeff Scan of MO 20 - fulvene - ML output')
  # plt.show()

  # plt.plot(np.arange(len(results)), [result.converged_s1_energy for result in results], label='S1 energy')
  # plt.xlabel('geometry i')
  # plt.ylabel('Energy (Hartree)')
  # plt.legend()
  # plt.title('S1 energy Scan - fulvene - converged CASSCF calc')
  # plt.show()

  # for index, energy in enumerate([result.converged_s1_energy for result in results]): # [result.converged_coeffs[21][5] for result in results]):
  #   print(index, energy)