
from PIL import ImageDraw, Image

from rasterio import features, transform
import numpy as np
from scipy.spatial import distance_matrix

from marker_annotation_refine.marker_annotation import draw_single_line

# list of np tuples
Polygon = np.ndarray

def mask_to_polygons(mask : np.ndarray):

  '''
  Generates polygons from a mask image.
  Returns list of polygons.
  '''

  shapes = features.shapes(
    mask.astype(np.int16), 
    mask=(mask >0),
  )

  polygons = []

  for shape,_ in shapes:

    coords = shape['coordinates']

    # shape will sometimes have multiple polygons and sometimes not
    if type(coords[0]) == tuple:

     coords = np.array([coords])

    polygons = [
      *polygons,
      *[np.array(c) for c in coords]
    ]
    

  return polygons

def polygon_to_mask(polygon, shape):

  return features.geometry_mask(
    [
      {
        'type': 'Polygon',
        'coordinates': [polygon.tolist()]
      }
    ],
    shape,
    transform=transform.Affine(1.0, 0, 0, 0, 1.0, 0),
    invert=True
  )

def draw_polygon(polygon, shape):

  img = Image.new('F', (shape[1], shape[0]))

  draw = ImageDraw.Draw(img)

  draw_single_line(
    draw,
    0, 0,
    1,
    polygon
  )
  
  return img


def calc_polygon_order(
  p1 : np.ndarray,
  p2 : np.ndarray,
):
  '''
  Re-orders points in p1 such that min|p1[i] - p2[i]| for all i
  '''

  new_order = [0]*len(p1)

  dist = distance_matrix(p1, p2)

  for i,p in enumerate(p1):

    new_order[i] = np.argmin(dist[i]).item()

  return new_order

def order_polygon(
  p1 : np.ndarray,
  p2 : np.ndarray,
):

  return p1[calc_polygon_order(p1, p2)]