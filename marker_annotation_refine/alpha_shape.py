# https://stackoverflow.com/questions/23073170/calculate-bounding-polygon-of-alpha-shape-from-the-delaunay-triangulation

from scipy.spatial import Delaunay
import numpy as np

def alpha_shape(points, alpha, only_outer=True):
  """
  Compute the alpha shape (concave hull) of a set of points.
  :param points: np.array of shape (n,2) points.
  :param alpha: alpha value.
  :param only_outer: boolean value to specify if we keep only the outer border
  or also inner edges.
  :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
  the indices in the points array.
  """
  assert points.shape[0] > 3, "Need at least four points"

  def add_edge(edges, i, j):
    """
    Add a line between the i-th and j-th points,
    if not in the list already
    """
    if (i, j) in edges or (j, i) in edges:
      # already added
      assert (j, i) in edges, "Can't go twice over same directed edge right?"
      if only_outer:
        # if both neighboring triangles are in shape, it is not a boundary edge
        edges.remove((j, i))
      return
    edges.add((i, j))

  tri = Delaunay(points)
  edges = set()
  # Loop over triangles:
  # ia, ib, ic = indices of corner points of the triangle
  for ia, ib, ic in tri.simplices:
    pa = points[ia]
    pb = points[ib]
    pc = points[ic]
    # Computing radius of triangle circumcircle
    # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
    a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
    b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
    c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    circum_r = a * b * c / (4.0 * area)
    if circum_r < alpha:
      add_edge(edges, ia, ib)
      add_edge(edges, ib, ic)
      add_edge(edges, ic, ia)

  return edges


def edges_to_indices(edges):

  '''
  Turns edges returned by alpha_shape into a list of indices of points that are on the alpha shape.
  '''

  edges = list(edges)

  indices = [edges[0][0], edges[0][1]]

  while(len(indices) < len(edges) * 2 - 1):

    for e in edges:

      if e[0] == indices[-1]:

        indices.append(e[0])
        indices.append(e[1])
        break

  return indices