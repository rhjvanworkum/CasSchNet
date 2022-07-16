import numpy as np

def normalise_rows(mat):
    '''Normalise each row of mat'''
    return np.array(tuple(map(lambda v: v / np.linalg.norm(v), mat)))

def flip(v):
    '''Returns 1 if max(abs(v))) is positive, and -1 if negative'''
    maxpos=np.argmax(abs(v))
    return v[maxpos]/abs(v[maxpos])

def order_orbitals(ref, target):
    '''Reorder target molecular orbitals according to maximum overlap with ref.
    Orbitals phases are also adjusted to match ref.'''
    # Moverlap=np.dot(normalise_rows(ref), normalise_rows(target).T)
    # orb_order=np.argmax(abs(Moverlap),axis=1)
    # target = target[orb_order]

    for idx in range(target.shape[0]):
        if np.dot(ref[:, idx], target[:, idx]) < 0:
            target[:, idx] = -1 * target[:, idx]

    return target # , orb_order

def correct_phase(mo_array: np.ndarray) -> None:
  """
  mo_array -> List of coeffs for 1 MO among each calculation
  """
  ref = mo_array[0]

  for idx in range(1, len(mo_array)):
    if np.dot(mo_array[idx], ref) < 0:
      mo_array[idx] = np.negative(mo_array[idx])