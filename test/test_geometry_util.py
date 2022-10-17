
import numpy as np
from marker_annotation_refine.geometry_util import calc_polygon_order

def test_order_polygon():

  order = [
    0, 3, 4, 1, 2
  ]

  p1 = np.array([
    (1,0), (2,0), (2, 1), (3,0), (10, 10)
  ])

  p2 = p1[order]

  p2_order = calc_polygon_order(p1, p2)

  assert [i == j for i,j in zip(order, p2_order)]

