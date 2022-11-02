
import math
import os

from dotenv import load_dotenv

from matplotlib import pyplot as plt

import numpy as np

import torch
from torch import nn

from torchvision import transforms

from skimage.util import random_noise

from PIL import Image, ImageDraw

from marker_annotation_refine.iterator_utils import \
  IteratorWrap

from marker_annotation_refine.marker_annotation import draw_single_line \

from marker_annotation_refine.marker_refine_dataset import \
  CSPolygon, \
  PolygonDataset
from marker_annotation_refine.unet import UNet



visualize = False

model_path = 'models/unet_denoise.pt'

max_noise_level = 10
noise_mix = 0.5
display_level = 9
report_interval = 20

img_to_tensor = transforms.PILToTensor()
tensor_to_img = transforms.ToPILImage()

def create_input_tensor(
  img_cam : Image.Image,
  img_marker : Image.Image,
  batch_size : int = 1
):

  '''

  Returns 5 channel image (tensor) with identical data for each batch.
  Channels:
    0:3 rgb camera image
    3   marker intensity
    4   gt object outline + noise (set to zero by default)

  '''

  w, h = img_cam.width, img_cam.height

  n = batch_size

  # 
  res = torch.zeros((n, 3 + 1 + 1, h, w))

  # index of the channel containing the marker data
  ch_marker = 3

  tensor_marker = torch.from_numpy(np.array(img_marker))

  # populate the inputs
  for i in range(n):

    # camera image
    res[i, 0:3, :, :] = img_to_tensor(img_cam)

    # marker 
    res[i, ch_marker, :, :] = tensor_marker

    # noise is left empty and needs to be filled in later

  return res


def load_polygon_as_batch(
  p : CSPolygon,
  noise_levels : int,
  mix = 0.2
):

  '''
  Loads polygon as train batches of (inputs, outputs) where

  inputs as in create_input_tensor

  outputs: 1 channel image of respective noise
  '''

  # index of the channel containing the marker data
  ch_marker = 3

  # index of the channel containing the noisy gt image
  ch_noisy_gt = 4

  n = noise_levels

  img_cam = p.cropped_img()
  img_gt = torch.from_numpy(p.draw_outline())


  h, w = img_gt.shape[0:2]

  # generate the marker intensity image 

  inputs = create_input_tensor(
    img_cam,
    p.draw_random_marker(w, h),
    n,
  )

  noise = torch.zeros((n, h, w))

  noise[0] = torch.rand((h, w)) * mix

  # generate progessive additive noise
  for i in range(1, n):

    noise[i] = noise[i - 1] + torch.rand((h, w)) * mix

  # populate the part of the inputs containing the noisy gt images
  for i in range(n):

    # noisy gt
    inputs[i, ch_noisy_gt, :, :] = (1.0 - mix) * img_gt + noise[i]

  return inputs, noise.reshape((n, 1, h, w))

# end load_polygon_as_batch

def display_batch(inputs, outputs):

  ''' Display part of noise batch. '''

  img = tensor_to_img(inputs[0, 0:3])

  inputs, outputs = inputs.detach().cpu().numpy(), outputs.detach().cpu().numpy()

  plt.subplot(2, 2, 1)

  marker = inputs[0, 3]

  noisy_gt = inputs[display_level, 4]

  noise = outputs[display_level, 0]

  gt = noisy_gt - noise
  gt /= np.max(gt)

  plt.imshow(img)

  plt.subplot(2,2,2)

  plt.imshow(marker)

  plt.subplot(2,2,3)

  plt.imshow(gt)

  plt.subplot(2,2,4)

  plt.imshow(noisy_gt)

  plt.show()

# end def display_batch

def load_model(
  model_path='models/unet_denoise.pt',
  accept_empty=False,
):

  model = UNet(
    enc_chs=(5, 64, 128),
    dec_chs=(128, 64),
    num_class=1,
  )

  try:

    model.load_state_dict(torch.load(model_path))

  except OSError as e:
    if accept_empty:
      print("Warning, no models found!")
    else:
      raise e

  return model

# end def load_model

if __name__ == '__main__':

  model = load_model()

  loss_fn =  nn.MSELoss()

  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
  )

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  model.to(device)
  model.train()

  load_dotenv()

  ds = IteratorWrap(
    PolygonDataset(
      os.environ['CITYSCAPES_LOCATION'],
      'train',
    ),
    lambda p: load_polygon_as_batch(p, max_noise_level, noise_mix)
  )

  loss_sum = 0

  for i, (inputs, gt) in enumerate(ds):

    try:

      inputs = inputs.float().to(device)
      gt = gt.float().to(device)

      optimizer.zero_grad()

      output = model.forward(inputs)
      
      output = nn.functional.interpolate(output, (gt.shape[2:4]))

      loss = loss_fn(output, gt)

      loss.backward()
      optimizer.step()

      loss_sum += loss.item()

      i += 1

      if i % report_interval == 0:

        print(f'Avg loss: {loss_sum/report_interval}')

        loss_sum = 0

        torch.save(model.state_dict(), model_path)


      if visualize:
        display_batch(inputs, gt)

    except:

      # probably CUDA out of memory (just skipping those batches for now due to laziness)
      # TODO: make sure batches don't exceed available memory
      pass    

    