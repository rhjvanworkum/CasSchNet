import h5py
from openmolcas.utils import read_in_orb_file

folder = 'C:/Users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200_ANO-S-MB/geometry_40/'

mo_vecs = h5py.File(folder + 'CASSCF.rasscf.h5').get('MO_VECTORS')[:].reshape(-1, 36)

orbs, _ = read_in_orb_file(folder + 'CASSCF.RasOrb')

import numpy as np

print(mo_vecs - orbs)

assert np.allclose(mo_vecs, orbs)




# import numpy as np
# import h5py
# import scipy
# import scipy.linalg
# from matplotlib import pyplot as plt

# def read_in_orb_file(orb_file : str):
#   orbitals = []
#   energies = []

#   append = False

#   with open(orb_file, 'r') as file:
#     # check for the ORB keyword in RasOrb file
#     while True:
#       line = file.readline()
#       if line[:4] == "#ORB":
#         break

#     # construct orbitals
#     while True:
#       line = file.readline()
#       # end of block
#       if '#' in line:
#         break
#       # add new orbital
#       elif '* ORBITAL' in line:
#         orbitals.append([])
#       # add coeffs
#       else:
#         for coeff in line.split(' '):
#           if len(coeff) > 0:
#             orbitals[-1].append(np.double(coeff.replace('\n', '')))

#     # get energies
#     while True:
#       line = file.readline()
#       # end of block
#       if append and '#' in line:
#         break
#       # append energies
#       if append:
#         for coeff in line.split(' '):
#           if len(coeff) > 0:
#             energies.append(np.double(coeff.replace('\n', '')))
#       # begin of block
#       if '* ONE ELECTRON ENERGIES' in line:
#         append = True

# #   print(energies)

#   return np.array(orbitals), np.array(energies)

# def normalise_rows(mat):
#     '''Normalise each row of mat'''
#     return np.array(tuple(map(lambda v: v / np.linalg.norm(v), mat)))

# def order_orbitals(ref, target):
#     '''Reorder target molecular orbitals according to maximum overlap with ref.
#     Orbitals phases are also adjusted to match ref.'''
#     Moverlap=np.dot(normalise_rows(ref),normalise_rows(target).T)
#     orb_order=np.argmax(abs(Moverlap),axis=1)
#     target = target[orb_order]

#     for idx in range(target.shape[0]):
#         if np.dot(ref[idx], target[idx]) < 0:
#             target[idx] = -1 * target[idx]

#     return target, orb_order


# def plot_psycf_line(idxs):
#     for idx in idxs:
#         data = []
#         for geom in range(200):
#             path = 'C:/Users/rhjva/imperial/fulvene/casscf_calculations/geom_scan_200/geometry_' + str(geom) +'.npz'
#             dat = np.load(path)
#             F_pyscf = dat['F']
#             data.append(F_pyscf.T.flatten()[idx])
#         plt.plot(np.arange(200), data)
#     plt.show()


# """ USE THIS ORDERING to also sort overlap matrix and energies right """
# all_orbitals = []
# all_energies = []
# all_overlap = []
# for idx in range(199):
#     path = 'C:/Users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200_canonical/geometry_' + str(idx) + '/CASSCF.RasOrb'
#     orbitals, energies = read_in_orb_file(path)
#     all_orbitals.append(orbitals)
#     all_energies.append(energies)
#     S = h5py.File('C:/Users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200_canonical/geometry_' + str(idx) + '/CASSCF.rasscf.h5').get('AO_OVERLAP_MATRIX')[:].reshape(36, 36)
#     all_overlap.append(S)

# for idx in range(1, 199):
#     all_orbitals[idx], orb_order = order_orbitals(all_orbitals[idx-1], all_orbitals[idx])
#     # reshuffle energies
#     all_energies[idx] = all_energies[idx][orb_order]
#     # print(all_energies[idx])
#     overlap = np.zeros((36, 36))
#     S = all_overlap[idx]
#     for i, idxx in enumerate(orb_order):
#         overlap[i, :] = S[idxx, :]
#         overlap[:, i] = S[:, idxx]
#         all_overlap[idx] = overlap


# # 356 - 1090 - 141

# def plot_molcas_line(idxs):
#     for idx in idxs:
#         data = []
#         for geom in range(199):
#             # path = 'C:/Users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200/geometry_' + str(geom) + '/CASSCF.RasOrb'
#             # orbitals, energies = read_in_orb_file(path)
#             orbitals = all_orbitals[geom]
#             energies = all_energies[geom]
#             S = all_overlap[geom]

#             # S = h5py.File('C:/Users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200_canonical/geometry_' + str(geom) + '/CASSCF.rasscf.h5').get('AO_OVERLAP_MATRIX')[:].reshape(36, 36)
#             F_molcas = np.matmul(S, np.matmul(orbitals.T, np.matmul(np.diag(energies), np.linalg.inv(orbitals.T))))
#             # F_molcas = np.linalg.inv(orbitals.T)
#             data.append(F_molcas.flatten()[idx])
#             # data.append(orbitals.flatten()[idx])
#         plt.plot(np.arange(199), data)
#     plt.show()

# for i in range(100):
#     idxs = np.random.choice(np.arange(1296), 1)
#     print(idxs, int(idxs / 36))
#     plot_psycf_line(idxs)
#     plot_molcas_line(idxs)

# geom = 10
# orbitals = all_orbitals[geom]
# energies = all_energies[geom]
# overlap = all_overlap[geom]
# # S = h5py.File('C:/Users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200/geometry_' + str(geom) + '/CASSCF.rasscf.h5').get('AO_OVERLAP_MATRIX')[:].reshape(36, 36)
# # F_molcas_10 = np.r(S, np.matmul(orbitals.T, np.matmul(np.diag(energies), np.linalg.inv(orbitals.T))))

# path = 'C:/Users/rhjva/imperial/fulvene/casscf_calculations/geom_scan_200/geometry_' + str(geom) +'.npz'
# F_pyscf_10 = np.load(path)['F']

# plt.imshow(F_pyscf_10)
# plt.show()

# for i in np.random.choice(np.arange(200), 30):
#     # geom = i
#     # orbitals = all_orbitals[geom]
#     # energies = all_energies[geom]
#     # overlap = all_overlap[geom]

#     # S = h5py.File('C:/Users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200/geometry_' + str(geom) + '/CASSCF.rasscf.h5').get('AO_OVERLAP_MATRIX')[:].reshape(36, 36)
#     # F_molcas = np.matmul(S, np.matmul(orbitals.T, np.matmul(np.diag(energies), np.linalg.inv(orbitals.T))))

#     # Moverlap=np.dot(F_molcas_10, F_molcas.T)
#     # orb_order=np.argmax(abs(Moverlap), axis=1)
#     # print(orb_order)

#     # F_molcas_150 = F_molcas
#     # i = 15
#     # j = 17
#     # F_molcas_150 = F_molcas
#     # F_molcas_150[i, :] = F_molcas[j, :]
#     # F_molcas_150[:, i] = F_molcas[:, j]
#     # F_molcas_150[j, :] = F_molcas[i, :]
#     # F_molcas_150[:, j] = F_molcas[:, i]

#     path = 'C:/Users/rhjva/imperial/fulvene/casscf_calculations/geom_scan_200/geometry_' + str(i) +'.npz'
#     F_psycf_150 = np.load(path)['F']

#     plt.imshow(F_psycf_150)
#     plt.show()

#     plt.imshow(F_psycf_150 - F_pyscf_10)
#     plt.show()


# # Moverlap=np.dot(F_molcas_10, F_molcas_150)
# # print()
# # orb_order=np.argsort(np.diagonal(Moverlap))

# # F_molcas = np.zeros((36, 36))
# # for i, idx in enumerate(orb_order):
# #     F_molcas[i, :] = F_molcas_150[idx, :]
# #     F_molcas[:, i] = F_molcas_150[:, idx]

# # plt.imshow(F_molcas)
# # plt.show()

# # plt.imshow(F_molcas)
# # plt.show()

# # orbitals = order_orbitals(refs, orbitals)

# # F_molcas_2 = np.matmul(S, np.matmul(orbitals.T, np.matmul(np.diag(energies), np.linalg.inv(orbitals.T))))
# # # print(F)
# # plt.imshow(F_molcas)
# # plt.show()

# # plt.imshow(np.abs(F_molcas - F_molcas_2))
# # plt.show()