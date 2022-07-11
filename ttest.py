from ase.db import connect

import numpy as np
import matplotlib.pyplot as plt

db_path = './data/geom_scan_200_molcas_fock.db'

all_F = []

with connect(db_path) as conn:
  for i in range(1, 200):
    all_F.append(conn.get(i).data['F'])
  # print(conn.get(1).data['F'])

for i in np.random.choice(np.arange(199), 50):
  plt.plot(np.arange(199), [F[i] for F in all_F])
  plt.show()