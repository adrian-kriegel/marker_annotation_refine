

import json
import os

# required for running on colab
from typing_extensions import TypedDict

from typing import Union

from PIL import Image
import numpy as np

LabelledPolygon = TypedDict(
  'LabelledPolygon',
  {
    'label': str,
    'polygon': list
  }
)

Polygons = TypedDict(
  'Polygons',
  {
    'imgHeight': int,
    'imgWidth': int,
    'objects': list
  }
)

class CSImage:

  __polygons : Union[Polygons, None] = None

  __instance_id_map : Union[Image.Image, None] = None

  __label_id_map : Union[Image.Image, None] = None

  __img : Union[Image.Image, None] = None

  def __init__(self, dataset_path : str, img_name : str):

    self.img_name = img_name
    self.dataset_path = dataset_path

    self.instances_by_id = {}

  def gt_fine_path(self, name : str):

    return os.path.join(
        self.dataset_path,
        'gtFine',
        self.img_name
      ) + '_gtFine_' + name

  def polygons(self) -> Polygons:

    if self.__polygons == None:

      self.__polygons = json.loads(
        open(self.gt_fine_path('polygons.json'), 'r').read()
      )

    return self.__polygons

  def instance_id_map(self):

    if self.__instance_id_map == None:

      self.__instance_id_map = Image.open(
        self.gt_fine_path('instanceIds.png')
      )

    return self.__instance_id_map

  def label_id_map(self):

    if self.__label_id_map == None:

      self.__label_id_map = Image.open(
        self.gt_fine_path('labelIds.png')
      )

    return self.__label_id_map

  def instance_ids(self):

    ids = np.unique(np.array(self.instance_id_map()))

    return ids[ids.nonzero()]

  
  def label_ids(self):

    ids = np.unique(np.array(self.label_id_map()))

    return ids[ids.nonzero()]

  def label_mask(self, id : int):

    '''
    Get an image of just one label by its id.
    Returns top_left, bottom_right, mask_img
    '''

    return self.id_mask(self.label_id_map(), id)

  def instance_mask(self, id : int):

    '''
    Get an image of just one instance by its id.
    Returns top_left, bottom_right, mask_img
    '''

    return self.id_mask(self.instance_id_map(), id)

  def id_mask(self, img_map, id):

    img = np.array(img_map) == id
    
    x_vals = np.sum(img, axis=0) > 0
    y_vals = np.sum(img, axis=1) > 0

    l,r = np.argmax(x_vals), len(x_vals) - np.argmax(np.flip(x_vals))
    t,b = np.argmax(y_vals), len(y_vals) - np.argmax(np.flip(y_vals))

    return (t,l),(b,r),img

  def img(self):

    if self.__img == None:

      self.__img = Image.open(
        os.path.join(
          self.dataset_path,
          'leftImg8bit',
          self.img_name
        ) + '_leftImg8bit.png'
      )

    return self.__img

  def instance_id_to_label_id(self, instance_id : int):

    label_ids = np.unique(np.array(self.label_id_map())[
      (np.array(self.instance_id_map()) == instance_id).nonzero()
    ])

    if not len(label_ids) == 1:

      raise Exception('Too many label ids for instance.')
    
    return label_ids[0]