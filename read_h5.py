import h5py
import numpy as np

f = h5py.File('C:/Users/rhjva/imperial/molcas_files/fulvene/CAS-CI.rasscf.h5', 'r')

print(f.keys())
print(f.get('AO_FOCKINT_MATRIX'))

# S = np.array(f.get('AO_OVERLAP_MATRIX')[:]).reshape(-1, 36)
# P = np.array(f.get('DENSITY_MATRIX')[:]).reshape(-1, 36)