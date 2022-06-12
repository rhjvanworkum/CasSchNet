# import numpy as np

# final = np.array([
#   [0.1, 0.2, 0.4],
#   [0.1, 0.4, 0.2],
#   [0.1, 0.3, 0.7]
# ])

# guess = np.array([
#   [0.1, 0.1, 0.4],
#   [0.2, 0.1, 0.2],
#   [0.4, 0.1, 0.7]
# ])

# U = np.matmul(np.linalg.inv(guess), final)
# # U = np.log(U)

# assert np.allclose(np.matmul(guess, U), final, atol=1e-08)


""" Skew-Symmetry """
# import numpy as np
import scipy.linalg

# A = np.array([
#   [0, 2, -45],
#   [-2, 0, -4],
#   [45, 4, 0]
# ])

# U = scipy.linalg.expm(A)

# print(np.matmul(U, np.conj(U).T))

import numpy as np
from calculate_evaluation import correct_orbitals, get_orbital_coeffs
from src.schnetpack.data.parser import order_orbitals

# ref = get_orbital_coeffs('C:/users/rhjva/imperial/molcas_files/fulvene_scan_2/geometry_0/CASSCF/CASSCF.RasOrb')
# target = get_orbital_coeffs('C:/users/rhjva/imperial/molcas_files/fulvene_scan_2/geometry_30/CASSCF/CASSCF.RasOrb')

# print(np.sum(ref), np.sum(target))

# print(np.sum(ref, axis=0))
# print(np.sum(target, axis=0))

# U = np.dot(target, np.linalg.inv(ref))
# X = scipy.linalg.logm(U)
# print(X)

""" Investigating how unitary transformation are on ML guesses and standard guesses """
# def normalise_rows(mat):
#   '''Normalise each row of mat'''
#   return np.array(tuple(map(lambda v: v / np.linalg.norm(v), mat)))

def dot(vec1, vec2):
  return np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def mse(vec1, vec2):
  return np.sum((vec1 - vec2) ** 2)

# ref = get_orbital_coeffs('C:/Users/rhjva/imperial/molcas_files/fulvene_scan_2/geometry_0/CASSCF/geom.orb')
# target = get_orbital_coeffs('C:/users/rhjva/imperial/molcas_files/fulvene_scan2_rotation/geometry_25/CASSCF/CASSCF.RasOrb')

# for idx in range(36):
#   print(np.linalg.norm(ref[idx]), np.linalg.norm(target[idx]))

# print(np.sum([dot(ref[idx], target[idx]) for idx in range(36)]))

# ref = normalise_rows(ref)
# target = normalise_rows(target)

# U = np.dot(target, np.linalg.inv(ref))
# print(np.matmul(U, np.conj(U).T))
# print(np.linalg.det(U))


# ref = get_orbital_coeffs('C:/users/rhjva/imperial/molcas_files/fulvene_scan2_rotation_2/geometry_25/CASSCF_ML/geom.orb')
# target = get_orbital_coeffs('C:/users/rhjva/imperial/molcas_files/fulvene_scan2_rotation/geometry_25/CASSCF_ML/CASSCF_ML.RasOrb')

# for idx in range(36):
#   print(np.linalg.norm(ref[idx]), np.linalg.norm(target[idx]))

# print(np.sum([dot(ref[idx], target[idx]) for idx in range(36)]))

# reff = get_orbital_coeffs('C:/Users/rhjva/imperial/molcas_files/fulvene_scan/geometry_' + str(0) + '/casscf/CASSCF.RasOrb')

# for i in range(200):
#   ref = get_orbital_coeffs('C:/Users/rhjva/imperial/molcas_files/fulvene_scan/geometry_' + str(i) + '/casscf/geom.orb')
#   target = get_orbital_coeffs('C:/users/rhjva/imperial/molcas_files/fulvene_scan/geometry_' + str(i) + '/casscf/CASSCF.RasOrb')

#   # ref = order_orbitals(ref=reff, target=ref)
#   # target = order_orbitals(ref=reff, target=target)

#   U = np.matmul(target, np.linalg.inv(ref))
#   # print(np.matmul(U, np.conj(U).T))
#   print(np.linalg.det(U))

""" ANALYZING the PYSCF guesses"""
base_path = 'C:/Users/rhjva/imperial/pyscf_files/fulvene_scan_pyscf_new/'
# ref = np.load(base_path + 'geometry_45_hf_guess.npy')
# target = np.load(base_path + 'geometry_45.npz')['mo_coeffs']

# # target = order_orbitals(ref, target)
# U = np.dot(np.linalg.inv(ref), target)
# L21 = np.sum(np.sqrt(np.sum(U**2, axis=1)))
# print(L21)

# ref = np.load(base_path + 'geometry_45_ml_guess.npy')
# target = np.load(base_path + 'geometry_45_ML.npz')['mo_coeffs']

# for idx in range(36):
#   print(np.abs(np.linalg.norm(ref[idx]) - np.linalg.norm(target[idx])))

# # target = order_orbitals(ref, target)
# U = np.dot(np.linalg.inv(ref), target)
# L21 = np.sum(np.sqrt(np.sum(U**2, axis=1)))
# print(L21)

""" Print transformation matrix """


target = get_orbital_coeffs('C:/Users/rhjva/imperial/molcas_files/fulvene_scan_molcas_nocorr/geometry_45/CASSCF/CASSCF.RasOrb')
ref = get_orbital_coeffs('C:/Users/rhjva/imperial/molcas_files/fulvene_scan_molcas_nocorr/geometry_45/CASSCF/geom.orb')
new_guess = get_orbital_coeffs('C:/users/rhjva/imperial/molcas_files/fulvene_scan_molcas_nocorr/geometry_45/CASSCF_ML/geom.orb')

print('\nHF -> ML GUess')
U = np.matmul(new_guess, np.linalg.inv(ref))
print(np.matmul(U, np.conj(U).T))
print(np.linalg.det(U))
print(np.sum([dot(ref[idx], new_guess[idx]) for idx in range(36)]) / 36)

print('\nHF -> Converged')
U = np.matmul(target, np.linalg.inv(ref))
print(np.matmul(U, np.conj(U).T))
print(np.linalg.det(U))
print(np.sum([dot(ref[idx], target[idx]) for idx in range(36)]) / 36)

print('\n ML Guess -> Converged')
U = np.matmul(target, np.linalg.inv(new_guess))
print(np.matmul(U, np.conj(U).T))
print(np.linalg.det(U))
print(np.sum([dot(new_guess[idx], target[idx]) for idx in range(36)]) / 36)
# for idx in range(36):
#   print(np.abs(np.linalg.norm(ref[idx]) - np.linalg.norm(new_guess[idx])))



# import matplotlib.pyplot as plt

# test_idx = np.load('data/fulvene_MB_140.npz')['test_idx']
# for idx in test_idx:
#   target = get_orbital_coeffs('C:/users/rhjva/imperial/molcas_files/fulvene_scan2_rotation/geometry_' + str(idx) + '/CASSCF/CASSCF.RasOrb')
#   plt.plot(np.arange(36), [np.linalg.norm(target[idx]) for idx in np.arange(36)], label='geometry ' + str(idx))

# for idx in test_idx:
#   target = get_orbital_coeffs('C:/users/rhjva/imperial/molcas_files/fulvene_scan2_rotation_2/geometry_' + str(idx) + '/CASSCF_ML/geom.orb')
#   plt.plot(np.arange(36), [np.linalg.norm(target[idx]) for idx in np.arange(36)], label='geometry rotated ' + str(idx))

# ref = get_orbital_coeffs('C:/users/rhjva/imperial/molcas_files/fulvene_scan2_rotation/geometry_25/CASSCF/geom.orb')
# plt.plot(np.arange(36), [np.linalg.norm(ref[idx]) for idx in np.arange(36)], label='ref')
  
# plt.legend()
# plt.show()