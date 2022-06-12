import os 
import shutil
import subprocess

base_path = '/mnt/c/users/rhjva/imperial/molcas_files/fulvene_scan/'
os.makedirs(base_path)

# setup SCF
for i in range(200):
  path = base_path + '/geometry_' + str(i) + '/scf/'
  os.makedirs(path)
  shutil.copy2('/mnt/c/users/rhjva/imperial/molcas_files/fulvene_scan_2/geometry_' + str(i) + '.xyz',
               path + 'geom.xyz')
  shutil.copy2('/mnt/c/users/rhjva/imperial/molcas_files/unitary_test/scf/scf.input',
              path + 'scf.input')

  subprocess.run('cd ' + path + '&& sudo pymolcas scf.input > calc.log', shell=True)

  path = base_path + '/geometry_' + str(i) + '/casscf/'
  os.makedirs(path)
  shutil.copy2('/mnt/c/users/rhjva/imperial/molcas_files/fulvene_scan_2/geometry_' + str(i) + '.xyz',
               path + 'geom.xyz')
  shutil.copy2('/mnt/c/users/rhjva/imperial/molcas_files/unitary_test/casscf/CASSCF.input',
              path + 'CASSCF.input')
  shutil.copy2(base_path + '/geometry_' + str(i) + '/scf/scf.ScfOrb',
               path + 'geom.Orb')

  subprocess.run('cd ' + path + '&& sudo pymolcas CASSCF.input > calc.log', shell=True)

  print(i / 200 * 100, ' %')