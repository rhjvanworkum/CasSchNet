import os
import numpy as np

from scripts.extract_geometries_from_initconds import BOHR_2_ANGSTROM
from scripts.generate_scan import Atom, write_xyz_file


def extract_geometries_from_traj_file(traj_file, n_atoms=12, n_geometries=200):

  geometries = [[] for _ in range(n_geometries)]

  with open(traj_file, 'r') as f:
    lines = f.readlines()

    for geom_idx in range(n_geometries):
      for atom_idx in range(n_atoms):
        data = list(filter(lambda a: len(a) > 0, lines[geom_idx * (n_atoms + 2) + 2 + atom_idx].replace('\n', '').split(' ')))
        geometries[geom_idx].append(Atom(
            type=data[0],
            x=float(data[1]),
            y=float(data[2]),
            z=float(data[3]),
          ))

  return geometries


if __name__ == "__main__":
  traj_paths = [
    'C:/users/rhjva/imperial/fulvene_sharc/results/traj0' + str(idx + 1) + '.xyz' for idx in range(5)  
  ]
  output_path = 'C:/Users/rhjva/imperial/fulvene/geometries/MD_trajectories_5_01/'
  n_samples = 50
  mode = 'random'

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  geometries = []
  for traj_path in traj_paths:
    geometries += extract_geometries_from_traj_file(traj_path)

  for idx, geometry in enumerate(geometries):
    write_xyz_file(geometry, output_path + 'geometry_' + str(idx) + '.xyz')

  if mode == 'random':
    idxs = np.random.choice(np.arange(len(geometries)), n_samples)
    np.savez('./data/MD_trajectories_05_01_random.npz', val_idx=idxs)
  