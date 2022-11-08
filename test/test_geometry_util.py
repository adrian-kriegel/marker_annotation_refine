
import numpy as np
from marker_annotation_refine.geometry_util import calc_polygon_order, mask_to_distances

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

def test_mask_to_distances():

  '''
  TODO: more tests/better setup
  '''

  img = np.array(
    [
      [1,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
    ]
  )

  mask = np.array(
    [
      [0,0,1,1],
      [0,0,0,1],
      [0,0,0,0],
    ]
  )

  dist, points = mask_to_distances(img, mask)

  assert len(dist) == 1
  assert len(points) == 1
  assert points[0][0] == 0
  assert points[0][0] == 0

  assert dist[0] == 2

