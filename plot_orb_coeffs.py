from matplotlib.pyplot import get
from src.schnetpack.data.parser import read_in_orb_file

import numpy as np
import matplotlib.pyplot as plt

def get_orbital_coeffs(file):
  return np.array(read_in_orb_file(file))

def euclidean_distance(vec1, vec2):
  return np.linalg.norm(vec1 - vec2)


if __name__ == "__main__":
  dir = 'C:/Users/rhjva/imperial/molcas_files/wigner_dist_200/config_200_VDZP/'

  ML_coeffs = get_orbital_coeffs(dir + 'CASCI_ML/geom.orb')
  guess_coeffs = get_orbital_coeffs(dir + 'CASSCF/geom.orb')
  ML_converged_coeffs = get_orbital_coeffs(dir + 'CASSCF_ML/CASSCF_ML.RasOrb')
  converged_coeffs = get_orbital_coeffs(dir + 'CASSCF/CASSCF.RasOrb')

  ML_distances = [euclidean_distance(ML_coeffs[i], converged_coeffs[i]) for i in range(converged_coeffs.shape[0])]
  guess_distances = [euclidean_distance(guess_coeffs[i], converged_coeffs[i]) for i in range(converged_coeffs.shape[0])]

  plt.title('Euclidean distance of orbital coeffs')
  plt.plot(np.arange(len(ML_distances)), ML_distances, label="ML Guess")
  plt.plot(np.arange(len(guess_distances)), guess_distances, label="Guess Orbitals (HF)")
  plt.xlabel('MO i')
  plt.ylabel('Eucleudian distance')
  plt.legend()
  plt.show()

  plt.title('orbital coeffs')
  plt.scatter(converged_coeffs, converged_coeffs, label="ML Guess")
  plt.scatter(converged_coeffs, guess_coeffs, label="Guess Orbitals (HF)")
  plt.xlabel('converged coeffs')
  plt.ylabel('ML/guess coeffs')
  plt.legend()
  plt.show()

  print('ML: ', np.sum(ML_distances))
  print('guess: ', np.sum(guess_distances))