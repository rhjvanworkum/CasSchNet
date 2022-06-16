import os
import shutil
import subprocess

"""
This Calculation is performed by taking a set of converged orbital coeffs from the 1 reference orbitals as an input
"""

if __name__ == "__main__":
  geometries_base_path = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/'
  output_path = '/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200/'
  n_geometries = 200

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  for i in range(n_geometries):
    # make dir
    geometry_path = output_path + 'geometry_' + str(i) + '/'
    if not os.path.exists(geometry_path):
      os.makedirs(geometry_path)

    # copy files
    shutil.copy2('./input_files/CASSCF.input', geometry_path + 'CASSCF.input')
    shutil.copy2('./input_files/geom.orb', geometry_path + 'geom.orb')
    shutil.copy2(geometries_base_path + 'geometry_' + str(i) + '.xyz', geometry_path + 'geom.xyz')

    # execute OpenMolcas
    subprocess.run('cd ' + geometry_path + ' && sudo /opt/OpenMolcas/pymolcas CASSCF.input > calc.log', shell=True)

    print('Progress: ', i / n_geometries * 100, '%')