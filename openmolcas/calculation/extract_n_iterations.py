
import numpy as np

def read_log_file(file, read_iterations=True):
  n_iterations = None
  rasscf_timing = None
  wall_timing = None

  with open(file, 'r') as f:
    while True:
      line = f.readline()

      if read_iterations:
        if "Convergence after" in line:
          for el in line.split():
            if el.isdigit():
              n_iterations = int(el)
              break

      if "--- Module rasscf spent" in line:
        for el in line.split():
          if el.isdigit():
            rasscf_timing = float(el)
            break

      if "Timing: Wall" in line:
        wall_timing = float(line.replace("=", " ").split()[2])
        break

  return rasscf_timing, wall_timing, n_iterations

if __name__ == "__main__":
  output_path = '/mnt/c/users/rhjva/imperial/fulvene/openmolcas_calculations/geom_scan_200_ANO-S-MB/'
  split_file = 'data/geom_scan_200_molcas.npz'

  n_iterations = []

  for i in np.load(split_file)['val_idx']:
    geometry_path = output_path + 'geometry_' + str(i) + '/'
    _, _, n = read_log_file(geometry_path + 'calc.log')
    n_iterations.append(n)

  np.save('temp/gs200_ANOSMB_standard.npy', np.array(n_iterations))