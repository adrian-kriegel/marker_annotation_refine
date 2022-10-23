
from cmath import isnan
import math
import os
import typing
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters

from skimage.color import rgb2hsv
import torch

from marker_annotation_refine.geometry_util import Polygon, rasterize_line

from marker_annotation_refine.marker_annotation import MarkerLine, draw_single_line

from marker_annotation_refine.marker_refine_dataset import \
  PolygonDataset

# quick way to look up 4 immediate neighbours of pixels
neighbours = np.array([
  (-1, 0),
  (1, 0),
  (0, 1),
  (0, -1),
])

def marker_to_vector_field(
  line, # TODO: type,
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

  # rasterize the line to obtain not just edges but all
  # points on the canvas crossed by the line
  line = rasterize_line(line, shape)

  phi = np.zeros(shape)
  amp = np.zeros(shape)
  
  for i in range(shape[0]):
    for j in range(shape[1]):
    
      p = np.array((i,j))
      
      diff = line - p

      dist = np.linalg.norm(diff, axis=1)
      
      k = np.argmin(dist).item()
      
      v = diff[k]

      amp[i,j] = intensity_map(dist[k])

      # % all angles because angles offset by 180° should be treated the same
      phi[i,j] = (math.atan2(v[1], v[0]) + np.pi/2) % np.pi - np.pi/2


  for (i,j) in line:
    # TODO: handle overflow
    idx = neighbours + (i,j)

    idx = [
      (h,k) for (h,k) in idx 
      if h > 0 and
      k > 0 and
      h < shape[1] and 
      k < shape[0]
    ]

    if len(idx) > 0:
      
      phi[i,j] = np.mean(phi[idx])
      amp[i,j] = np.mean(amp[idx])

  return to_uv(phi, amp)

def channel_grad(img):

  '''
  Gradients for each channel in an image.
  '''

  grads = []

  for c in range(img.shape[2]):

    grads.append(np.gradient(img[:,:,c]))

  return grads

def to_polar(x, y):

  ''' Returns phi, amp '''

  return np.arctan2(y, x), np.linalg.norm((x,y), axis=0)

def to_uv(phi, amp):

  return np.cos(phi)*amp, np.sin(phi)*amp

def combine_vector_fields(
  field_img,
  field_maker
):
  
  '''
  Combined vector fields
  Returns vector field in polar form
  '''

  combined = field_maker[0]*field_img[0], field_maker[1]*field_img[1]

  phi,amp = to_polar(combined[0], combined[1])

  phi = (phi + np.pi/2) % np.pi - np.pi/2

  return phi, amp

# TODO: implement
def walk_fields(
  fields : typing.List[typing.Tuple[np.ndarray, np.ndarray]],
  start_pos : typing.Tuple[int, int],
  start_angle : float,
  thrs : float
):

  points = []

  last_angle = start_angle

  phis = [field[0] for field in fields]
  amps = [field[1] for field in fields]

  while True:

    k = np.argmax(amps).item()

  return points

def gaussian(x, sigma):

  return np.exp(-np.power(x/sigma, 2.)/2.)


class VectorFieldDataset(PolygonDataset):

  def __next__(self):
    
    polygon = super().__next__()

    full = polygon.csinstance.csimg.img()
    img = np.array(polygon.cropped_img())

    if img.shape[0]*img.shape[1] > 0.1 * (full.width*full.height):

      return self.__next__()

    x0,y0,_,_ = polygon.bounding_box()
    
    marker = polygon.random_marker()

    mx, my = np.array(marker['points']).transpose()
    
    mx, my = (mx - x0, my - y0)

    marker['points'] = np.transpose(
      (mx, my)
    ).tolist()

    r = marker['brushSize'] / 2

    scale = min(img.shape[0], img.shape[1])

    dx,dy = marker_to_vector_field(
      marker['points'],
      img.shape[0:2],
      lambda d: (1.0  + gaussian((d-r)/scale,0.04)) * 0.5,
    )

    #
    # load data into a torch tensor
    #

    img = img / np.max(img)

    marked_img = np.zeros((2+3, *img.shape[0:2]))
    
    # colour channels
    marked_img[0,:,:] = img[:,:,0]
    marked_img[1,:,:] = img[:,:,1]
    marked_img[2,:,:] = img[:,:,2]

    # marker vector field channels
    marked_img[3,:,:] = dx
    marked_img[4,:,:,] = dy

    return marked_img, polygon.draw_outline()

if __name__ == '__main__':

  load_dotenv()

  ds = VectorFieldDataset(
    os.environ['CITYSCAPES_LOCATION'],
    'train',
  )


  for marked_img,mask in ds:

    img = np.moveaxis(marked_img, 0, 2)


    dx = marked_img[3,:,:]
    dy = marked_img[4,:,:]

    plt.subplot(1,2,1)

    plt.imshow(img[:,:,0:3]/np.max(img[:,:,0:3]))

    plt.quiver(dx, dy)

    plt.subplot(1,2,2)

    plt.imshow(mask)

    plt.show()
