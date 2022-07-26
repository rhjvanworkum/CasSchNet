import numpy as np
import matplotlib.pyplot as plt

labels = ['Huckel', 'ML_MO', 'ML_U', 'ML_F']

data = [
  (21.46, 6.49),
  (42.3, 13.94),
  (19.34, 6.29),
  (19.0, 5.60)
]


plt.errorbar(np.arange(len(data)),
                           [data[i][0] for i in range(len(data))], 
                           yerr=[data[i][1] for i in range(len(data))],
                           fmt='o',
                           capsize=5)

plt.title('N CASSCF iterations on MD trajectory validation set (OpenMolcas)')
plt.xlabel('Method')
plt.ylabel('N iterations')
plt.xticks(np.arange(len(data)), labels)
plt.show()