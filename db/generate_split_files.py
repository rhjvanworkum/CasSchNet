import numpy as np

if __name__ == "__main__":
  name = 'wigner_dist_2000'
  n = 2000
  train_split = 0.9
  val_split = 0.1
  test_split = 0.0

  """ GENERATE SPLIT FILES """
  data_idx = np.arange(n)
  np.random.shuffle(data_idx)
  train_idxs = data_idx[:int(train_split * n)]
  val_idxs = data_idx[int(train_split * n):int((train_split + val_split) * n)]
  test_idxs = data_idx[int((train_split + val_split) * n):]

  np.savez('./data/' + name + '.npz', 
    train_idx=train_idxs, 
    val_idx=val_idxs,
    test_idx=test_idxs)