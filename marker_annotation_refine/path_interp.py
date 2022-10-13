
from tracemalloc import start
import numpy as np

Point = tuple[int, int]

class PathInterp:

  def __init__(self, ex, ey):

    self.edges = np.transpose((ex, ey))
    
    self.lengths = np.zeros(len(self.edges) - 1)

    self.lengths[0] = 0

    for i in range(1, len(self.lengths)):

      self.lengths[i] = self.lengths[i -1]  + np.linalg.norm(self.edges[i] - self.edges[i + 1])

    self.lengths /= self.lengths[-1]

  def __call__(self, base : np.ndarray):

    '''
    Returns the path interpolated at the distances in base ranging from 0-1.
    base MUST be range from 0 - 1.0 !
    '''

    res = np.zeros((len(base), 2))

    res[0] = self.edges[0]
    res[-1] = self.edges[-1]

    for i in range(1, len(base) - 1):

      d = base[i]

      # find the two anchor points the point lies inbetween of
      idx_floor = np.argmax(self.lengths >= d) - 1
      idx_ceil = idx_floor + 1

      # distance between the anchor points
      l = self.lengths[idx_ceil] - self.lengths[idx_floor]

      # relative position between the two anchor points (0 - 1.0)
      b = (d - self.lengths[idx_floor]) / l

      res[i] = (1.0 - b) * self.edges[idx_floor] + b * self.edges[idx_ceil]

        

    return res