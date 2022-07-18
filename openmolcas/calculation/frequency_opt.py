import os
import shutil
import subprocess

"""
This Calculation is performed by taking a set of converged orbital coeffs from the 1 reference orbitals as an input
"""

if __name__ == "__main__":
  geometry_path = '/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/geometry_0.xyz'
  output_path = '/mnt/c/users/rhjva/imperial/fulvene_sharc/opt_freq/'

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  # copy files
  shutil.copy2(geometry_path, output_path + 'geom.xyz')
  shutil.copy2('./input_files/CASSCF_opt.input', output_path + 'CASSCF_opt.input')

  # execute OpenMolcas
  subprocess.run('cd ' + output_path + ' && sudo /opt/OpenMolcas/pymolcas CASSCF_opt.input > calc.log', shell=True)