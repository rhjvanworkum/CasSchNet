import numpy as np

def generate_split(train_split: float, val_split: float, test_split: float, n: int, save_path: str) -> None:
  data_idx = np.arange(n)
  np.random.shuffle(data_idx)
  train_idxs = data_idx[:int(train_split * n)]
  val_idxs = data_idx[int(train_split * n):int((train_split + val_split) * n)]
  test_idxs = data_idx[int((train_split + val_split) * n):]

  np.savez(save_path, 
    train_idx=train_idxs, 
    val_idx=val_idxs,
    test_idx=test_idxs)


if __name__ == "__main__":
  save_path = './data/wigner_dist_200.npz'
  n = 200
  train_split = 0.9
  val_split = 0.1
  test_split = 0.0
  generate_split(train_split, val_split, test_split, n, save_path)