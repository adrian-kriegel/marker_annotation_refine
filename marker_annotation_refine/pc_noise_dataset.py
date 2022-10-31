
import numpy as np

from skimage.color import rgb2gray
from skimage.filters import gaussian

from marker_annotation_refine.marker_refine_dataset import \
  CSPolygon, \
  PolygonDataset



class PCNoise:

  def __init__(
    self,
    polygon : CSPolygon,
  ):

    self.polygon = polygon

    img = np.array(polygon.cropped_img())

    self.n = (img.shape[0] * img.shape[1]) // 10

    dx,dy = np.gradient(rgb2gray(img))

    self.amp = np.linalg.norm((dx, dy), axis=0)
    self.amp /= np.max(self.amp)
    self.candidates = np.where(self.amp > 0.02 * np.max(self.amp))

    self.imgpoly = polygon.draw_outline()
    self.imgpoly /= np.max(self.imgpoly)

  def __iter__(self):

    self.last_noise = np.zeros(self.imgpoly.shape[0:2], dtype=np.float32)

    self.level = 1

    return self

  def __next__(self):

    idx = np.random.choice(len(self.candidates[0]), self.n)

    idx_x = self.candidates[0][idx]
    idx_y = self.candidates[1][idx]

    mask = np.zeros(self.amp.shape[0:2], dtype=bool)

    mask[idx_x, idx_y] = True

    noise = mask * self.amp

    return noise
    
  def mix(self, img, noise):

    self.level += 1

    f = 1.0 / self.level
    
    noise_img = (1.0 - f)*self.noise + f * self.imgpoly

    return noise_img / np.max(noise_img)


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

    for i,noisyimg in enumerate(noiser):

      if i >= nlevels:
        break

      plt.subplot(1, nlevels, i + 1)

      plt.imshow(noisyimg)

    plt.show()