
from rasterio import features, transform
import numpy as np

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

    coords = np.array(shape['coordinates'])

    # shape will sometimes have multiple polygons and sometimes not
    if type(coords[0]) == tuple:

     coords = np.array([coords])

    polygons = [
      *polygons, 
      *coords
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