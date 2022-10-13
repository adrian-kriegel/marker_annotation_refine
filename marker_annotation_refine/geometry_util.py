
from rasterio import features
import numpy as np

# list of np tuples
Polygon = list[np.ndarray]

def mask_to_polygons(mask : np.ndarray) -> list[Polygon]:

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

    if type(coords[0]) == tuple:

      polygons.append(coords)

    else:

      polygons = [*polygons, *coords]

  return polygons
