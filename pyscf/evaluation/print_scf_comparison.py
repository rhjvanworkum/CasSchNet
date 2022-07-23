import numpy as np
import matplotlib.pyplot as plt

from evaluate import extract_results
  

# if __name__ == "__main__":
#   base_dir = 'C:/Users/rhjva/imperial/fulvene/casscf_calculations/pyscf-runs/'
#   split_file = '../../data/geom_scan_200_aws.npz'

#   dir_lists = [
#     ['geom_scan_200_hf_sto_6g/', 'geom_scan_200_ML_MO_sto_6g/', 'geom_scan_200_ML_F_sto_6g/'],
#     # ['geom_scan_200_hf_4-31G/', 'geom_scan_200_ML_MO_4-31G/', 'geom_scan_200_ML_F_4-31G/'],
#     ['geom_scan_200_ML_MO_6-31Gstar/', 'geom_scan_200_ML_F_6-31Gstar/'],
#   ]

#   for dir_list in dir_lists:
#     for dir in dir_list:
#       results = extract_results(split_file, base_dir + dir)
#       iterations = [result.imacro for result in results]
#       print(dir, np.mean(iterations))

data = [
  [4.85, 6.3, 5.85],
  [24.1, 12.0, 13.6],
  [3.7, 5.45, 4.75]
]

labels = ['Hartee-Fock', 'ML_MO', 'ML_F']

for dat, label in zip(data, labels):
  plt.plot([36, 66, 96], dat, label=label)

plt.xlabel('basis set')
plt.ylabel('avg_iterations')
plt.xticks([36, 66, 96], ['STO-6G (N = 36)', '4-31G (N = 66)', '6-31G* (N = 96)'])
plt.legend()
plt.show()
