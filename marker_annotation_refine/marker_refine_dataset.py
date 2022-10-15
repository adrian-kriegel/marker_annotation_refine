
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

from cityscapesscripts.helpers.labels import id2label

from skimage import filters

import torch.utils.data
from glob import glob

IGNORE_LABELS = ['unlabeled', 'rectification border', 'out of roi', 'static', 'dynamic']

# crop regions 20% larger than required by the marker
CROP_PADDING = 0.2

MIN_INSTANCE_AREA_PX = 12*12
MAX_INSTANCE_AREA_RATIO = 0.9

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
  padding : int
):
  x0 = np.min(x)
  x1 = np.max(x)
  y0 = np.min(y)
  y1 = np.max(y)

  lines = []

  # scale in order to normalize things to the size of the polygon
  scale = min((x1 - x0), (y1 - y0))

  for i in range(6):

    brush_size = math.floor(scale / np.random.uniform(1, 4))

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

  return draw_marker(lines, padding)


def get_image_names(dataset_path : str, mode : str):

  base_path = os.path.join(dataset_path, 'leftImg8bit')

  if base_path[-1] != '/':
    base_path = base_path + '/'

  return [
    name.replace('_leftImg8bit.png', '').replace(base_path, '') for name in glob(
      # cannot use root_dir on python 3.7
      os.path.join(base_path, mode, '*/*.png'),
    )
  ]

class InvalidInstanceError(Exception):

  pass

class MarkerRefineDataset(torch.utils.data.Dataset):

  def __init__(
    self,
    dataset_path : str,
    mode : str = 'train',
    max_blur = 0.0
  ):

    self.dataset_path = dataset_path

    self.image_names = get_image_names(dataset_path, mode)

    self.max_blur = max_blur
  
  def __len__(self):

    # I think Dataset and DataLoader do not allow datasets of unknown sizes...
    # some super huge number will do as % is used in __getitem__
    return len(self.image_names) * 20

  def __getitem__(self, _ : int):

    '''
    Returns image, marker, mask
    '''

    idx = int(np.random.rand() * len(self))

    csimg = CSImage(self.dataset_path, self.image_names[idx % len(self.image_names)])

    #
    # find a valid instance 
    #

    instance_ids = csimg.instance_ids()
      
    instance_id = instance_ids[idx % len(instance_ids)]

    label_id = csimg.instance_id_to_label_id(instance_id)  # type: ignore
    label = id2label[label_id]

    if label.category in IGNORE_LABELS:

      return None
      
    p0,p1,label_mask = csimg.instance_mask(instance_id)
 
    #
    # from the instance, pick a random polygon 
    # as instances may be comprised of multiple polygons (no idea what the authors idea of an "instance" is)
    #

    polygons = mask_to_polygons(label_mask)

    polygon = polygons[idx % len(polygons)]
 
    x,y = np.transpose(polygon)

    # no idea why that is required sometimes
    x = np.array(x)
    y = np.array(y)

    x = x.flatten()
    y = y.flatten()

    # width, height of the polygons bounding box
    pw, ph = np.max(x) - np.min(x), np.max(y) - np.min(y)

    area = pw*ph

    img_area = csimg.img().width * csimg.img().height

    if (
      area < MIN_INSTANCE_AREA_PX or
      area/img_area > MAX_INSTANCE_AREA_RATIO 
    ):
      return None

    (mx, my), marker = draw_marker_from_polygon(x, y, CROP_PADDING*max(pw, ph))

    box = (mx, my, mx + marker.shape[1], my + marker.shape[0])

    instance_map = np.array(csimg.instance_id_map()) == instance_id

    mask = polygon_to_mask(polygon, instance_map.shape)
    
    mask_cropped = np.array(Image.fromarray(mask).crop(box), dtype=float)

    marked_img = np.zeros(
      (4, marker.shape[0], marker.shape[1])
    )

    img_cropped = np.array(csimg.img().crop(box))

    marked_img[0,:,:] = img_cropped[:,:,0]
    marked_img[1,:,:] = img_cropped[:,:,1]
    marked_img[2,:,:] = img_cropped[:,:,2]

    marked_img[3,:,:] = filters.gaussian(marker, np.min(marker.shape)*np.random.uniform(0, self.max_blur)) 

    return marked_img, mask_cropped

def split_marked_image(inp):

  img = np.zeros((*inp.shape[2:4], 3))

  img[:,:,0] = inp[0, 0, :, :]
  img[:,:,1] = inp[0, 1, :, :]
  img[:,:,2] = inp[0, 2, :, :]
  img /= np.max(img)

  marker = inp[0,3,:,:]
  marker /= np.max(marker)

  return img, marker


if __name__ == '__main__':

  from dotenv import load_dotenv
  from model import prep_input

  matplotlib.use('TkAgg')

  load_dotenv()

  dataset = MarkerRefineDataset(os.environ['CITYSCAPES_LOCATION'], max_blur=0.05)
  
  for v in dataset:

    if not v == None:

      marked_img, gt = v

      inp = prep_input(marked_img, 'cpu')

      img, marker = split_marked_image(inp.detach().numpy())

      plt.subplot(1,3,1)
      plt.imshow(img)

      plt.subplot(1,3,2)
      plt.imshow(marker)

      plt.subplot(1,3,3)
      plt.imshow(gt)

      plt.show()