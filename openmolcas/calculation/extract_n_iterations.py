
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
  output_path = '/home/ubuntu/fulvene/openmolcas_calculations/MD_trajectory_1/'
  # split_file = 'data/MD_trajectories_05_01_random.npz'

  n_iterations = []

  # for i in np.load(split_file)['val_idx']:  
  for i in range(200):
    geometry_path = output_path + 'geometry_' + str(i) + '/'
    _, _, n = read_log_file(geometry_path + 'calc.log')
    n_iterations.append(n)

  n_iterations = np.array(n_iterations)
  print(n_iterations)
  # np.save('temp/gs200_ANOSVDZ_standard.npy', np.array(n_iterations))