# lets try to fit a linear regression model to the mse errors and the predicted n_iterations

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

from calculate_evaluation import get_orbital_coeffs, order_orbitals, read_log_file
from src.schnetpack.data.parser import read_in_orb_file

def mse(orbs1, orbs2):
  return np.abs(orbs2.flatten() - orbs1.flatten())

# gather all data here
base_path = 'C:/Users/rhjva/imperial/molcas_files/fulvene_scan_4/'

X = []
y = []

for i in range(200):
  casscf_ml_result = read_log_file(base_path + 'geometry_' + str(i) + '/CASSCF_ML/calc.log')
  y.append(casscf_ml_result.n_iterations)

  ml_orb_coeffs = get_orbital_coeffs(base_path + 'geometry_' + str(i) + '/CASSCF_ML/geom.orb')
  converged_coeffs = get_orbital_coeffs(base_path + 'geometry_' + str(i) + '/CASSCF/CASSCF.RasOrb')
  ref = read_in_orb_file('C:/users/rhjva/imperial/molcas_files/fulvene_scan_2/geometry_0/CASSCF/CASSCF.RasOrb')
  converged_coeffs = order_orbitals(ref, converged_coeffs)

  X.append(mse(ml_orb_coeffs, converged_coeffs))

for i in range(200):
  casscf_result = read_log_file(base_path + 'geometry_' + str(i) + '/CASSCF/calc.log')
  y.append(casscf_result.n_iterations)

  guess_orbs = get_orbital_coeffs(base_path + 'geometry_' + str(i) + '/CASSCF/geom.orb')
  converged_coeffs = get_orbital_coeffs(base_path + 'geometry_' + str(i) + '/CASSCF/CASSCF.RasOrb')
  ref = read_in_orb_file('C:/users/rhjva/imperial/molcas_files/fulvene_scan_2/geometry_0/CASSCF/CASSCF.RasOrb')
  converged_coeffs = order_orbitals(ref, converged_coeffs)

  X.append(mse(guess_orbs, converged_coeffs))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=420)

reg = LinearRegression(positive=True).fit(X, y)
print(reg.score(X, y))

coeffs = reg.coef_
coeffs /= np.sum(reg.coef_) 
coeffs *= 1296

np.save('weights_abs.npy', coeffs)

import matplotlib.pyplot as plt
plt.plot(np.arange(len(coeffs)), coeffs)
plt.show()

# import numpy as np

# weights = np.load('weights.npy')
# print(weights)
# print(len(np.where(weights > 0)[0]))