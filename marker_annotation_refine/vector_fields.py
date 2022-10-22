
from cmath import cos, sin
import math
import os
import typing
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw
from marker_annotation_refine.geometry_util import rasterize_line

from marker_annotation_refine.marker_annotation import MarkerLine, draw_single_line

from marker_annotation_refine.marker_refine_dataset import \
  PolygonDataset


load_dotenv()

ds = PolygonDataset(
  os.environ['CITYSCAPES_LOCATION'],
  'train',
)

def marker_to_vector_field(
  marker : MarkerLine,
  shape : typing.Tuple[int, int],
  intensity_map : typing.Callable[[float], float]
):
  '''
  Generate a vector field from a marker annotation.
  TODO: intensity
  TODO: cudafy!

  Keyword arguments:
  marker -- MarkerLine (sufficiently high spacial sample rate)
  shape -- (h,w) of the resulting vector field
  intensity_map -- Callable to map from distance to intensity
  '''

  r = marker['brushSize']

  # rasterize the line to obtain not just edges but all points on the canvas crossed by the line
  line = rasterize_line(
    marker['points'],
    shape
  )

  dx = np.zeros(shape)
  dy = np.zeros(shape)

  for i in range(shape[0]):
    for j in range(shape[1]):
    
      p = np.array((i,j))
      
      diff = line - p

      dist = np.linalg.norm(diff, axis=1)
      
      k = np.argmin(dist).item()
      
      v = diff[k] / dist[k] * intensity_map(dist[k])

      phi = math.atan2(v[1], v[0]) + np.pi/2
      
      dx[i,j] = np.sin(phi)
      dy[i,j] = np.cos(phi)

  # override all points directly on the line

  return (dx,dy)

def channel_grad(img):

  grads = []*img.shape[2]

  for c in range(img.shape[2]):

    grads.append(np.gradient(img[:,:,c]))

  return grads


if __name__ == '__main__':

  for i,polygon in enumerate(ds):

    full_img = polygon.csinstance.csimg.img()

    pw,ph = polygon.dims()

    if pw*ph > 0.08 * full_img.width*full_img.height:
      continue

    img = np.array(polygon.cropped_img())

    x0,y0,x1,y1 = polygon.bounding_box()

    bw = np.sum(img, axis=2)
  
    
    marker = polygon.random_marker()

    mx, my = np.array(marker['points']).transpose()
    
    mx, my = (mx - x0, my - y0)

    marker['points'] = np.transpose(
      (mx, my)
    ).tolist()

    dx,dy = marker_to_vector_field(
      marker,
      img.shape[0:2],
      lambda d: 1.0,
    )

    grads = channel_grad(img)

    # plt.imshow(img)
    plt.quiver(dx, dy)

    plt.plot(polygon.x - x0, polygon.y - y0)
    
    plt.plot(mx, my)
  
    plt.show()
  