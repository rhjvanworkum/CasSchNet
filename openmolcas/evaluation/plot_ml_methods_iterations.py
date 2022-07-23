# import numpy as np
# import matplotlib.pyplot as plt

# BASIS_SETS = {
#   'ANO-S-MB': 36,
#   'ANO-S-VDZ': 66,
#   'ANO-L-VTZ': 176
# }

# if __name__ == "__main__":
#   base_line_paths = ['', '', '']
#   split_file = '../../data/geom_scan_200_molcas.npz'

#   methods = ['ML_MO', 'ML_U', 'ML_F']

#   base_dir = 'C:/Users/rhjva/imperial/fulvene/casscf_calculations/pyscf-runs/'
#   dir_list = [
#     [],
#     [],
#     []
#   ]

#   # baselines
#   means = []
#   stds = []
#   for path in base_line_paths:
#     data = np.load(path)
#     idxs = np.load(split_file)['val_idx']
#     points = data[idxs]
#     means.append(np.mean(points))
#     stds.append(np.std(points))
#   plt.errorbar(BASIS_SETS.values(), means, yerr=stds, label='Standard_guess')

#   # other methods
#   for idx, method in enumerate(methods):
#     means = []
#     stds = []
#     for basis_idx in range(len(dir_list[idx])):
#       data = np.load(dir_list[idx][basis_idx])
#       points = data
#       means.append(np.mean(points))
#       stds.append(np.std(points))
#     plt.errorbar(BASIS_SETS.values(), means, yerr=stds, label=method)

#   # plot 
#   plt.xlabel('basis set (N basis functions)')
#   plt.ylabel('N RASSCF iterations')
#   plt.xticks(BASIS_SETS.values(), [key + '(N = ' + str(val) + ')' for key, val in BASIS_SETS.items()])
#   plt.legend()
#   plt.show()

import numpy as np

for file in ['gs199_ANOSMB_standard.npy', 'gs199_ANOSMB_ML_MO.npy', 'gs199_ANOSMB_ML_U.npy', 'gs199_ANOSMB_ML_F.npy',
             'gs200_ANOSVDZ_standard.npy', 'gs199_ANOSVDZ_ML_MO.npy', 'gs199_ANOSVDZ_ML_F.npy', 'tests/gs200_ANOSMB_ML_U.npy']:
  data = np.load('temp/' + file)
  print(np.mean(data), np.std(data))