
import math
import os
from re import A
import typing
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from cityscapesscripts.helpers.labels import id2label

from skimage import transform
from PIL import Image, ImageDraw

from marker_annotation_refine.cityscapes_helpers import CSImage
from marker_annotation_refine.path_interp import PathInterp
from marker_annotation_refine.geometry_util import mask_to_polygons, draw_polygon, polygon_to_mask
from marker_annotation_refine.marker_annotation import MarkerLine, Point, draw_marker, draw_single_line

IGNORE_LABELS = ['unlabeled', 'rectification border', 'out of roi', 'static', 'dynamic']

# crop regions 30% larger than required by the marker
CROP_PADDING = 0.3

MIN_INSTANCE_AREA_PX = 12*12
MIN_INSTANCE_WIDTH_PX = 14
MIN_INSTANCE_HEIGHT_PX = 14
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
  interp = PathInterp(ex, ey)

  interp_num = math.ceil(interp.path_len / brush_size * 6)
  
  points = low_pass(interp(np.linspace(0, 1, interp_num)), f=1 - 1/interp_num * low_pass_factor)

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

def resize_safe_single_channel(img : np.ndarray, shape : typing.Tuple[int, int]):

  # TODO: pad images to retain their aspect ratio
  return transform.resize(img, shape)

def resize_safe(
  img : np.ndarray,
  shape : typing.Tuple[int, int],
):

  if len(img.shape) == 2:

    return resize_safe_single_channel(img, shape)

  else:

    res = np.zeros((img.shape[0], *shape))
    
    for c in range(img.shape[0]):

      res[c,:,:] = resize_safe_single_channel(img[c,:,:], shape)

    return res

class MarkerRefineDataPoint(CSImage):

  def __init__(self, dataset, idx):

    CSImage.__init__(
      self,
      dataset.dataset_path, 
      dataset.image_names[idx % len(dataset.image_names)]
    )

  def get_instance(self, instance_id):

    return CSInstance(self, instance_id)

class CSInstance:

  __polygons = None
  __mask = None

  def __init__(self, csimg : CSImage, instance_id : int):

    self.instance_id = instance_id
    self.csimg = csimg

  def mask(self):

    if self.__mask == None:

      self.__mask = self.csimg.instance_mask(self.instance_id)

    return self.__mask[2]

  def bounds(self):

    self.mask()

    p0,p1 = self.__mask[0:2] # type: ignore
    
    return np.index_exp[p0[0]:p1[0],p0[1]:p1[1]]

  def polygons(self):

    if self.__polygons == None:

      self.__polygons = [
        CSPolygon(self, points) for 
        points in mask_to_polygons(self.mask())
      ]

    return self.__polygons
  
  def get_label(self):

    '''
    Returns label
    '''

    label_id = self.csimg.instance_id_to_label_id(self.instance_id)  # type: ignore
    label = id2label[label_id]

    return label

  def is_valid(self):

    return (not self.get_label().category in IGNORE_LABELS) and (
      np.any([p.is_valid() for p in self.polygons()])
    )

class CSPolygon:

  __bounds = None

  def __init__(
    self,
    csinstance : CSInstance,
    points
  ):

    self.points = points
    self.csinstance = csinstance

    x,y = np.transpose(self.points)
    # no idea why that is required sometimes
    x = np.array(x)
    y = np.array(y)

    self.x = x.flatten()
    self.y = y.flatten()


  def dims(self):

    '''
    Returns w,h of bounding box
    '''
    x0, x1, y0, y1 = self.bounds()


    return abs(x1 - x0), abs(y1 - y0) 

  def bounds(self, alignment=4):

    if self.__bounds == None:
      x,y = self.x,self.y

      a = alignment

      x0 = (np.min(x)//a)*a
      x1 = (np.max(x)//a)*a
      y0 = (np.min(y)//a)*a
      y1 = (np.max(y)//a)*a

      if (x1 - x0) % 2 == 1:
        x1 += 1

      if (y1 - y0) % 2 == 1:
        y1 += 1

      self.__bounds = x0,x1,y0,y1

    return self.__bounds


  def is_valid(self):

    if len(self.points) < 3:
      return False

    pw,ph = self.dims()

    area = pw*ph

    img_area = self.csinstance.csimg.img().width * self.csinstance.csimg.img().height

    return not (
      area < MIN_INSTANCE_AREA_PX or
      area/img_area > MAX_INSTANCE_AREA_RATIO or
      # TODO: if this happens, just pad the image!
      pw < MIN_INSTANCE_WIDTH_PX or
      ph < MIN_INSTANCE_HEIGHT_PX
    )

  def random_marker(self):

    x0,x1,y0,y1 = self.bounds()

    # scale in order to normalize things to the size of the polygon
    scale = min((x1 - x0), (y1 - y0))

    brush_size = math.floor(scale / np.random.uniform(2, 4))

    return simulate_marker(
      self.x, 
      self.y,
      brush_size,
      np.random.uniform(0.001, 0.002)*scale,
      np.random.uniform(0.05, 0.1)*scale,
      np.random.uniform(0, 2*np.pi),
      low_pass_factor=np.random.uniform(10, 100)
    )

  def draw_random_marker(self, w, h):

    x0,y0,_,_ = self.bounding_box()
    marker = self.random_marker()
    img_marker = Image.new('F', (w, h), 0)
    
    draw_single_line(
      ImageDraw.Draw(img_marker),
      x0, y0,
      marker['brushSize'],
      marker['points']
    )

    return img_marker

  def bounding_box(self, padx=None, pady=None):

    '''
    Returns x0,y0,x1,y1 of padded bounding box.
    '''

    w,h = self.dims()

    if padx == None:
      padx = int(CROP_PADDING*w)
      if padx % 2 == 1: padx +=1

    if pady == None:
      pady = int(CROP_PADDING*h)
      if pady % 2 == 1: pady +=1

    x0,x1,y0,y1 = self.bounds()

    return (
      int(x0 - padx/2),
      int(y0 - pady/2),
      int(x1 + padx/2),
      int(y1 + pady/2),
    )

  def cropped_img(self, padx=None, pady=None):

    return self.csinstance.csimg.img().crop(self.bounding_box(padx, pady))

  def cropped_img_extent(self, padx=None, pady=None):

    l,t,r,b = self.bounding_box(padx, pady)

    return (l,r,b,t)

  def draw_outline(self, padx=None, pady=None):

    x0,y0,x1,y1 = self.bounding_box(padx, pady)

    img = Image.fromarray(np.zeros((y1 - y0, x1 - x0), dtype=float))

    draw_single_line(
      ImageDraw.Draw(img),
      x0, y0,
      1,
      self.points,
      1
    )

    return np.array(img)



class CSDataset:

  def __init__(
    self,
    dataset_path : str,
    mode : str = 'train'
  ):

    self.dataset_path = dataset_path

    self.image_names = get_image_names(dataset_path, mode)
  
  def __iter__(self):

    self.idx_img = 0

    return self


  def __next__(self):

    if self.idx_img >= len(self.image_names):

      raise StopIteration()

    dp = MarkerRefineDataPoint(self, self.idx_img)

    self.idx_img += 1

    return dp

class InstanceDataset(CSDataset):
  
  def __iter__(self):

    CSDataset.__iter__(self)

    self.idx_inst = 0

    self.curr_img = None
    self.curr_instances = []

    return self

  def __next__(self) -> CSInstance:

    while self.curr_img == None:
      
      dp = super().__next__()

      instances = [CSInstance(dp, iid) for iid in dp.instance_ids()]
      instances = [inst for inst in instances if inst.is_valid()]

      if len(instances) > 0:

        self.idx_inst = 0
        self.curr_img = dp
        self.curr_instances = instances


    if self.idx_inst < len(self.curr_instances):
      inst = self.curr_instances[self.idx_inst]

      self.idx_inst += 1

      return inst

    else:
      self.idx_inst = 0
      self.curr_img = None

      return InstanceDataset.__next__(self)

class PolygonDataset(InstanceDataset):

  def __iter__(self):

    InstanceDataset.__iter__(self)
    
    self.curr_inst = None
    self.curr_polygons = []

    self.idx_polygon = 0

    return self

  def __next__(self) -> CSPolygon:

    if self.curr_inst == None:

      self.curr_inst = InstanceDataset.__next__(self)
    
      self.curr_polygons = [p for p in self.curr_inst.polygons() if p.is_valid()]
      self.idx_polygon = 0


    if self.idx_polygon < len(self.curr_polygons):

      polygon = self.curr_polygons[self.idx_polygon]

      self.idx_polygon += 1

      return polygon

    else:

      self.curr_inst = None

      return PolygonDataset.__next__(self)


if __name__ == '__main__':

  from dotenv import load_dotenv

  matplotlib.use('TkAgg')

  load_dotenv()

  dataset = PolygonDataset(
    os.environ['CITYSCAPES_LOCATION'],
  )

  for polygon in dataset:

    plt.subplot(1,3,1)

    bounds = polygon.csinstance.bounds()
    
    cropped_img = np.array(polygon.csinstance.csimg.img())[bounds]

    plt.imshow(cropped_img)

    plt.subplot(1,3,2)

    w,h = polygon.dims()

    plt.imshow(
      polygon.cropped_img(CROP_PADDING*w, CROP_PADDING*h),
      extent=polygon.cropped_img_extent(CROP_PADDING*w, CROP_PADDING*h)
    )

    plt.plot(polygon.x, polygon.y)

    plt.show()