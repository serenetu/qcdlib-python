import numpy as np
from numpy import linalg as nplin

def mtx_minus_mtx(mtx1, mtx2):
    return np.subtract(np.array(mtx1), np.array(mtx2))

def if_parallel(vec1, vec2):
    # Parallel     : Return  1
    # Anti-Parallel: Return -1
    # Not Parallel : Return  0

    assert len(vec1)>0, 'The Lenth of Vector 1 is 0'
    assert len(vec2)>0, 'The Lenth of Vector 2 is 0'
    assert len(vec1)==len(vec2), 'The Lenth of Vector 1 and Vector 2 Do Not Match'
    #cross_norm = nplin.norm(np.cross(vec1, vec2))
    vec1_norm = nplin.norm(vec1)
    vec2_norm = nplin.norm(vec2)
    norm_multi = vec1_norm*vec2_norm
    dot = np.dot(vec1, vec2)
    if np.absolute(np.absolute(dot)-norm_multi) < 10e-9:
      if dot > 0:
        return 1
      elif dot < 0:
        return -1
    else:
      return 0

if __name__ == '__main__':
    vec1 = [ 3,  4,  6, 8]
    vec2 = [6, 8, 12, 16]
    print mtx_minus_mtx(vec1, vec2)
    print if_parallel(vec1, vec2)
