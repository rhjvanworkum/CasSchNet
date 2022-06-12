# import h5py
# import numpy as np

# f = h5py.File('C:/Users/rhjva/imperial/molcas_files/fulvene/CAS-CI.rasscf.h5', 'r')

# S = np.array(f.get('AO_OVERLAP_MATRIX')[:]).reshape(-1, 36)
# P = np.array(f.get('DENSITY_MATRIX')[:]).reshape(-1, 36)
# # print(P.shape)
# # print(np.trace(np.matmul(S, P)))

from src.schnetpack.data.parser import get_orbital_occupations


print(get_orbital_occupations('C:/Users/rhjva/imperial/molcas_files/fulvene_scan/geometry_1/casscf/CASSCF.RasOrb'))