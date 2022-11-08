
import typing
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

def draw_polygon(polygon, shape, width=1):

  img = Image.new('F', (shape[1], shape[0]))

  draw = ImageDraw.Draw(img)

  draw_single_line(
    draw,
    0, 0,
    width,
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

def rasterize_line(
  points,
  shape
):

  '''
  Rasterize a line and return all points in the raster that are on the line.
  '''

  line = Image.fromarray(np.zeros(shape, dtype=bool))

  draw_single_line(
    ImageDraw.Draw(line),
    0, 0,
    1,
    points
  )

  return np.transpose(np.array(line).nonzero())

def mask_to_distances(
  img : np.ndarray,
  mask : np.ndarray,
):

  '''
  Given two binary images, the shortest distance between nonzero pixels is calculated.
  Returns distances, coords where coords are the nonzero pixels of img
  '''

  coords_img = np.array(img.nonzero()).transpose()
  coords_mask = np.array(mask.nonzero()).transpose()

  dist = distance_matrix(
    coords_img,
    coords_mask
  )

  dist = np.min(dist, axis=1)

  return dist, coords_img