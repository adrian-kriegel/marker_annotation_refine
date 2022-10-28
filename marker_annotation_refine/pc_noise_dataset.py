


import typing
import numpy as np

from skimage.color import rgb2gray
from marker_annotation_refine.geometry_util import rasterize_line

from marker_annotation_refine.marker_refine_dataset import \
  CSPolygon, \
  PolygonDataset

def draw_polygon_noisy(
  polygon : CSPolygon,
  shape : typing.Tuple[int,int]
):

  img = np.zeros(shape, dtype=np.float32)

  coords = np.transpose(np.nonzero(polygon.draw_outline()))

  for i,j in coords:

    img[i,j] = 1.0

  return img



class PCNoise:

  def __init__(
    self,
    polygon : CSPolygon,
    n = 50
  ):

    self.n = n

    self.polygon = polygon

    img = np.array(polygon.cropped_img())

    dx,dy = np.gradient(rgb2gray(img))

    amp = np.linalg.norm((dx, dy), axis=0)

    amp /= np.max(amp)

    self.amp = amp

    self.candidates = np.where(amp > 0.2)

    self.imgpoly = draw_polygon_noisy(polygon, amp.shape)

  def __iter__(self):

    self.mask = np.zeros(self.amp.shape[0:2], dtype=bool)

    return self

  def __next__(self):

    idx = np.random.choice(len(self.candidates[0]), self.n)

    idx_x = self.candidates[0][idx]
    idx_y = self.candidates[1][idx]

    self.mask[idx_x, idx_y] = True

    return (self.amp * self.mask)


class PCNoiseDataset(PolygonDataset):

  def __next__(self):
    
    polygon = super().__next__()

    return PCNoise(polygon)



if __name__ == '__main__':

  import os

  from dotenv import load_dotenv
  from matplotlib import pyplot as plt

  load_dotenv()

  ds = PCNoiseDataset(
    os.environ['CITYSCAPES_LOCATION'],
    'train',
  )

  nlevels = 5

  for noiser in ds:

    for i,noise in enumerate(noiser):

      if i >= nlevels:
        break

      plt.subplot(1, nlevels, i + 1)

      plt.imshow(noiser.imgpoly + noise)

    plt.show()