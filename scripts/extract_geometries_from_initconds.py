

from scripts.generate_scan import Atom, write_xyz_file

BOHR_2_ANGSTROM = 0.529177249

def read_initconds(file_name, output_path, n_atoms=12, n_geometries=200):
  geometries = [[] for _ in range(n_geometries)]

  with open(file_name, 'r') as f:
    geom_idx = 0
    lines = f.readlines()
    for idx, line in enumerate(lines):
      if 'Index' in line:
        for atom_i in range(n_atoms):
          data = list(filter(lambda a: len(a) > 0, lines[idx + 2 + atom_i].replace('\n', '').split(' ')))
          geometries[geom_idx].append(Atom(
            type=data[0],
            x=float(data[2]) * BOHR_2_ANGSTROM,
            y=float(data[3]) * BOHR_2_ANGSTROM,
            z=float(data[4]) * BOHR_2_ANGSTROM,
          ))
        geom_idx += 1

  for idx, geometry in enumerate(geometries):
    write_xyz_file(geometry, output_path + 'geometry_' + str(idx) + '.xyz')

if __name__ == "__main__":
  initconds_path = 'C:/Users/rhjva/imperial/fulvene_sharc/initconds'
  output_path = 'C:/Users/rhjva/imperial/fulvene/geometries/wigner_dist_200/'

  read_initconds(initconds_path, output_path=output_path)
