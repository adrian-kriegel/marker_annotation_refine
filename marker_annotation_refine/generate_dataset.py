
import math
import os
from PIL import Image

import numpy as np
from cityscapes_helpers import CSImage
import matplotlib.pyplot as plt
import matplotlib

from marker_annotation import MarkerLine, draw_marker
from geometry_util import mask_to_polygons, polygon_to_mask
from path_interp import PathInterp

matplotlib.use('TkAgg')

MIN_POLYGON_AREA_PX = 10*10
MAX_POLYGON_AREA_RATIO = 0.9

def normalize(v):

  return v / np.linalg.norm(v)

def low_pass(data, f):

  data = np.array(data)

  res = np.zeros(data.shape)

  res[0] = data[0]

  for i in range(1, len(data)):

    res[i] = (1 - f) * data[i] + f*res[i - 1]

  return res


def simulate_marker(
  ex, ey, # edges
  interp_num : int,
  brush_size : int,
  jitter_freq : float,
  jitter_amp : int,
  jitter_phase : float,
  low_pass_factor : float
) -> MarkerLine:

  '''
  Turns a fine segmentation label into a simulation of a human-annotated marker label.
  '''

  # interpolate the polygon
  points = low_pass(PathInterp(ex, ey)(np.linspace(0, 1, interp_num)), f=1 - 1/interp_num * low_pass_factor)

  center = np.mean(points, axis=0)

  r = brush_size/2

  jitter = jitter_amp * np.sin(jitter_phase + jitter_freq * np.linspace(0, 2*np.pi, len(points)))

  points = [
    p + normalize(center - p) * (r + jitter[i]) for i,p in enumerate(points)
  ]
  
  return {
    'points': np.array(points, dtype=int).tolist(),
    'brushSize': brush_size,
    't': True,
  }

def draw_marker_from_polygon(
  x,y,
):
  x0 = np.min(x)
  x1 = np.max(x)
  y0 = np.min(y)
  y1 = np.max(y)

  lines = []

  # scale in order to normalize things to the size of the polygon
  scale = min(x1 - x0, y1 - y0)

  for i in range(6):

    brush_size = math.floor(scale / np.random.uniform(1, 6))

    line = simulate_marker(
      x,y,
      100,
      brush_size,
      np.random.uniform(0.001, 0.005)*scale,
      np.random.uniform(0.05, 0.2)*scale,
      np.random.uniform(0, 2*np.pi),
      low_pass_factor=np.random.uniform(10, 100)
    )

    lines.append(line)

  return draw_marker(lines)

if __name__ == '__main__':

  from dotenv import load_dotenv

  load_dotenv()

  img_name = 'train/erfurt/erfurt_000041_000019'

  csimg = CSImage(os.environ['CITYSCAPES_LOCATION'], img_name)

  img_area = csimg.instance_id_map().width * csimg.instance_id_map().height

  for instance_id in csimg.instance_ids():

    p0,p1,label_mask = csimg.instance_mask(instance_id)

    polygons = mask_to_polygons(label_mask)
    
    print(instance_id)

    for polygon in polygons:

      x,y = np.transpose(polygon)
      x = np.array(x).flatten()
      y = np.array(y).flatten()

      # width, height of the polygons bounding box
      pw, ph = np.max(x) - np.min(x), np.max(y) - np.min(y)

      area = abs(pw) * abs(ph)
      
      if (
        area < MIN_POLYGON_AREA_PX or 
        area / img_area > MAX_POLYGON_AREA_RATIO
      ):
        continue

      plt.subplot(1,3,1)

      (mx, my), marker = draw_marker_from_polygon(x,y)

      box = (mx, my, mx + marker.shape[1], my + marker.shape[0])

      plt.imshow(csimg.img().crop(box))

      plt.subplot(1,3,2)

      plt.imshow(marker)

      plt.subplot(1,3,3)

      instance_map = np.array(csimg.instance_id_map()) == instance_id

      mask = polygon_to_mask(polygon, instance_map.shape)
      
      plt.imshow(Image.fromarray(mask).crop(box))

      plt.show()

