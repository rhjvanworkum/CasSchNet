import numpy as np

from pyscf.evaluation.evaluate import extract_results


if __name__ == "__main__":
  split_file = ''
  dir = ''
  name = ''

  results = extract_results(split_file, dir)
  iterations = [result.imacro for result in results]
  np.save(name, np.array(iterations))