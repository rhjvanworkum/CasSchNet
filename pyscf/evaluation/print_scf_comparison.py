import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

data = [
  # [(45.25, 5.22), (49.5, 2.18), (50.0, 0.0)],
  [(4.85, 1.49), (6.30, 2.742), (6.00, 2.74)],
  [(3.60, 0.49), (5.50, 3.20), (5.20, 2.45)],
  [(3.75, 0.43), (5.15, 1.15), (4.85, 0.85)],
]

basis_sets = {
  'STO-6G': 36,
  '4-31G': 66,
  '6-31G*': 96
}

methods = {
  'HF': 'red',
  'ML_MO': 'blue',
  'ML_F': 'green'
}

fig = plt.figure(dpi=100)
ax = fig.add_subplot(111, projection='3d')

for idx, (method, color) in enumerate(methods.items()):
  x = list(basis_sets.values())
  y = [idx for _ in range(3)]
  z = [data[idx][i][0] for i in range(len(data[idx]))]
  err = [data[idx][i][1] for i in range(len(data[idx]))]
  ax.plot(x, y, z, linestyle="None", marker="o", color=color, label=method)

  for i in np.arange(0, 3):
      ax.plot([x[i], x[i]], [y[i], y[i]], [z[i] + 0.5 * err[i], z[i] - 0.5 * err[i]], color=color)

#configure axes
ax.set_xlabel('')
ax.set_xticks(list(basis_sets.values()), list(basis_sets.keys()))
ax.set_ylabel('')
ax.set_yticks(np.arange(len(methods.keys)), list(methods.keys()))
ax.set_zlabel('N Iterations')

plt.title('N CASSCF iterations on geom_scan_200 validation')
plt.legend()
plt.show()