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
    Moverlap=np.dot(normalise_rows(ref), normalise_rows(target).T)
    orb_order=np.argmax(abs(Moverlap),axis=1)
    target = target[orb_order]

    for idx in range(target.shape[0]):
        if np.dot(ref[:, idx], target[:, idx]) < 0:
            target[:, idx] = -1 * target[:, idx]

    return target

def correct_phase(ref, target) -> None:
  """
  mo_array -> List of coeffs for 1 MO among each calculation
  """
  new_target = target.copy()
  
  for idx in range(target.shape[0]):
        if np.dot(ref[:, idx], target[:, idx]) < 0:
            new_target[:, idx] = -1 * new_target[:, idx]

  return new_target