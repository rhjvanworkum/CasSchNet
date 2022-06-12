# import torch

# def mo_overlap(C_1, C_2, overlap_matrix):
#   return 0.5 * torch.einsum('i,j,ij', C_1, C_2, overlap_matrix)

# def overlap(mo_coeffs_1, mo_coeffs_2, overlap_matrix):
#   overlap_m = torch.zeros((36, 36))
#   for i in range(36):
#     for j in range(36):
#       overlap_m[i, j] = mo_overlap(mo_coeffs_1[:, i], mo_coeffs_2[:, j], overlap_matrix)

#   return torch.abs(torch.linalg.det(overlap_m))


# mo_coeffs_1 = torch.rand((36, 36))
# mo_coeffs_2 = torch.rand((36, 36))
# overlap_matrix = torch.rand((36, 36))

# print(overlap(mo_coeffs_1, mo_coeffs_2, overlap_matrix))

# over = 0.5 * torch.einsum('ki,kj,ij->ij', mo_coeffs_1, mo_coeffs_2, overlap_matrix)
# print(torch.abs(torch.linalg.det(over)))

""" COPY DATAFILES """
import shutil
for i in range(200):
  shutil.copy2('C:/Users/rhjva/imperial/molcas_files/fulvene_scan_2/geometry_' + str(i) + '.xyz', 'C:/Users/rhjva/imperial/molcas_files/fulvene_dataset_2200/geometry_' + str(i + 2001) + '.xyz')

for i in range(2001):
  shutil.copy2('C:/Users/rhjva/imperial/sharc_files/run_MB/config_' + str(i) + '/geom.xyz', 'C:/Users/rhjva/imperial/molcas_files/fulvene_dataset_2200/geometry_' + str(i) + '.xyz')

""" CONVERT BOHR TO ANGSTROM """
# from generate_scan import Atom, read_xyz_file, write_xyz_file
# b2a = 0.529177249

# for i in range(2001):
#   file = 'C:/Users/rhjva/imperial/sharc_files/run_MB/config_' + str(i) + '/geom.xyz'
#   atoms = read_xyz_file(file)
#   for atom in atoms:
#     atom.x *= b2a
#     atom.y *= b2a
#     atom.z *= b2a
#   write_xyz_file(atoms, file)
