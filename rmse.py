import numpy as np
import scipy

def normalise_rows(mat):
    '''Normalise each row of mat'''
    return np.array(tuple(map(lambda v: v / np.linalg.norm(v), mat)))


def dot(vec1, vec2):
  return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def density_matrix(orbitals, occ):
  P = np.zeros((orbitals.shape[0], orbitals.shape[1]))
  for i in range(orbitals.shape[0]):
    for j in range(orbitals.shape[1]):
      for idx, k in enumerate(occ):
        P[i, j] += k * orbitals[idx, i] * orbitals[idx, j] 
  return P


def projection(guess, conv, S, occupations):
  P_guess = density_matrix(guess, occupations)
  P_conv = density_matrix(conv, occupations)

  # return np.trace(np.matmul(P_conv, S)) - np.trace(np.matmul(P_guess, S))
  # guess = np.matmul(P_guess, S)
  # conv = np.matmul(P_conv, S)

  # return np.sum((conv.flatten() - guess.flatten()) ** 2) / len(conv.flatten())
  return np.trace(np.matmul(P_conv, np.matmul(S, np.matmul(P_guess, S))))

def mo_overlap(C_1, C_2, overlap_matrix):
  return 0.5 * np.einsum('i,j,ij', C_1, C_2, overlap_matrix)

def overlap(mo_coeffs_1, mo_coeffs_2, overlap_matrix):
  overlap_m = np.zeros((6, 6))
  for idx_i, i in enumerate(range(18, 24)):
    for idx_j, j in enumerate(range(18, 24)):
      overlap_m[idx_i, idx_j] = mo_overlap(mo_coeffs_1[i, :], mo_coeffs_2[j, :], overlap_matrix)

  return np.abs(np.linalg.det(overlap_m))

def rmse(result, ML=False):

  if ML:
    orbs1 = result.ml_orb_coeffs
  else:
    orbs1 = result.guess_orb_coeffs

  orbs2 = result.converged_coeffs

  S = np.array(result.h5.get('AO_OVERLAP_MATRIX')[:]).reshape(-1, 36)
  
  # over = 0.5 * np.einsum('ki,lj,kl->ij', orbs1, orbs2, S)
  # return np.abs(np.linalg.det(over))

  return overlap(orbs1, orbs2, S)
  
  # occ = result.occupations
  # return projection(orbs1, orbs2, S, occ)

  # return np.sum(np.multiply(result.occupations, np.array([dot(orbs1[idx], orbs2[idx]) for idx in range(36)]))) / 36
  # return projection(orbs1, orbs2)
  return np.mean([dot(orbs1[idx], orbs2[idx]) for idx in range(36)]) / 36

  """ Matrix Norm """
  # U = np.dot(np.linalg.inv(orbs1), orbs2)
  # # assert np.allclose(np.matmul(orbs1, U), orbs2, atol=1e-08)
  # # L21 = np.sum(np.sqrt(np.sum(U**2, axis=1)))
  # L21 = np.linalg.norm(U)
  # return L21

  """ RMSE """
  # return np.linalg.norm(orbs1 - orbs2)

  """ Magnitude weighted MSE """
  se = (orbs2.flatten() - orbs1.flatten()) ** 2
  wse = se / np.abs(orbs2.flatten())
  return np.sum(wse) / (36 ** 2)

  """ MSE """
  # return np.sum((orbs2.flatten() - orbs1.flatten()) ** 2) / len(orbs2.flatten())

  """ GAuSSIAN WEIGHTED AROUND ACTIVE SPACE"""
  # def active_gauss(sigma, n_mo=36, prefactor=1000):
  #   x = np.linspace(0, n_mo**2, n_mo**2)
  #   mid = ((24 - 18) / 2 + 18) * n_mo
  #   gauss = prefactor * np.exp(-(x - mid) ** 2 / (2 * sigma ** 2))
  #   gauss[(24*n_mo):] = 0
  #   return gauss

  # weights = active_gauss(sigma=50)
  # return np.sum(np.multiply((orbs2.flatten() - orbs1.flatten()) ** 2, weights)) / len(orbs2.flatten())
  
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