
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from geometry_util import mask_to_polygons

from marker_refine_dataset import MarkerRefineDataset, split_marked_image
from model import Encoder, Decoder, prep_input
from skimage.transform import resize
from skimage.filters import gaussian
import dotenv

dotenv.load_dotenv()

dataset = MarkerRefineDataset(
  os.environ['CITYSCAPES_LOCATION'],
  'val'
)

encoder = Encoder(60)
encoder.load_state_dict(
  torch.load('models/marker_refine_encoder.pt', map_location=torch.device('cpu'))
)
decoder = Decoder(60)
decoder.load_state_dict(
  torch.load('models/marker_refine_decoder.pt', map_location=torch.device('cpu'))
)

encoder.to('cpu')
decoder.to('cpu')

matplotlib.use('TkAgg')

thrs = 0.4

conv_rate = 0.01
num_iterations = 10

masked_blur_sigma = 2
masked_blur_rate = 0.3

def masked_blur(img, mask, sigma):

  return (1-mask) * img + mask * gaussian(img, sigma)

for v in dataset:

  if v == None:
    continue

  marked_img, gt = v

  inp = prep_input(marked_img, 'cpu')
 
  out_img = np.zeros(1)
  _inp = inp.clone()

  for i in range(num_iterations):

    output = decoder.forward(encoder.forward(_inp))
  
    out_img = output[0].cpu().detach().numpy().reshape(output.shape[2:4])
    
    out_img = resize(
      out_img, 
      gt.shape
    )

    out_fb = torch.from_numpy(out_img)

    blur_mask = (1 - out_fb)*masked_blur_rate

    _inp[0, 0, :, :] = masked_blur(_inp[0, 0, :, :], blur_mask, masked_blur_sigma)
    _inp[0, 1, :, :] = masked_blur(_inp[0, 1, :, :], blur_mask, masked_blur_sigma)
    _inp[0, 2, :, :] = masked_blur(_inp[0, 2, :, :], blur_mask, masked_blur_sigma)

    _inp[0, 3, :, :] = (1-conv_rate)*_inp[0, 3, :, :] + out_fb*conv_rate

  
  out_img = gaussian(out_img, np.min(out_img.shape)*0.02)

  inp = inp.cpu().detach().numpy()

  img,marker = split_marked_image(inp)  

  polygons = mask_to_polygons(out_img > thrs)

  polygons_gt = mask_to_polygons(gt)

  plt.subplot(1,3,1)
  plt.imshow(img)

  for polygon, polygon_gt in zip(polygons, polygons_gt):
    x,y = np.transpose(polygon)
    plt.plot(x, y, color='red')

    x,y = np.transpose(polygon_gt)
    plt.plot(x, y, color='green', alpha=0.7)

  plt.subplot(1,3,2)
  plt.imshow(marker)

  plt.subplot(1,3,3)

  blurred = np.zeros(img.shape)
  
  blurred[:,:,0] = masked_blur(img[:,:,0], 1 - out_img, 10)
  blurred[:,:,1] = masked_blur(img[:,:,1], 1 - out_img, 10)
  blurred[:,:,2] = masked_blur(img[:,:,2], 1 - out_img, 10)

  #plt.imshow(blurred)

  plt.imshow(out_img)

  plt.show()