import numpy as np
import matplotlib.pyplot as plt

labels = ['Huckel', 'ML_MO', 'ML_U', 'ML_F']

data = [
  (14.45, 1.20),
  (13.95, 2.09),
  (12.85, 1.82),
  (11.0, 0.32)
]


plt.errorbar(np.arange(len(data)),
                           [data[i][0] for i in range(len(data))], 
                           yerr=[data[i][1] for i in range(len(data))],
                           fmt='o',
                           capsize=5)

plt.title('N CASSCF iterations on geom_scan_200 validation set (OpenMolcas)')
plt.xlabel('Method')
plt.ylabel('N iterations')
plt.xticks(np.arange(len(data)), labels)
plt.show()