
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from geometry_util import mask_to_polygons

from marker_refine_dataset import MarkerRefineDataset, split_marked_image
from model import Encoder, Decoder, PolygonDecoder
from skimage.transform import resize
from skimage.filters import gaussian
import dotenv

dotenv.load_dotenv()

polygon_length = 100

dataset = MarkerRefineDataset(
  os.environ['CITYSCAPES_LOCATION'],
  'val',
  return_polygon=True,
  polygon_length=100,
)

encoder = Encoder(60)
decoder = PolygonDecoder(num_points=polygon_length)

try:

  encoder.load_state_dict(
    torch.load('models_polygon/marker_refine_encoder.pt', map_location=torch.device('cpu'))
  )

  decoder.load_state_dict(
    torch.load('models_polygon/marker_refine_decoder.pt', map_location=torch.device('cpu'))
  )

except OSError:
  pass

encoder.eval()
decoder.eval()

encoder.to('cpu')
decoder.to('cpu')

matplotlib.use('TkAgg')

thrs = 0.4

conv_rate = 0.01
num_iterations = 1

masked_blur_sigma = 2
masked_blur_rate = 0.3

def masked_blur(img, mask, sigma):

  return (1-mask) * img + mask * gaussian(img, sigma)

for v in dataset:

  if v == None:
    continue

  marked_img, gt = v

  inp = torch.from_numpy(marked_img).float().reshape((1, *marked_img.shape))

  out_img = np.zeros(1)
  _inp = inp.clone()

  output = None

  output = decoder.forward(encoder.forward(_inp))
  
  #out_img = gaussian(out_img, np.min(out_img.shape)*0.02)

  inp = inp.cpu().detach().numpy()

  img,marker = split_marked_image(inp)  

  polygons = mask_to_polygons(out_img > thrs) if not dataset.return_polygon else \
              output.cpu().detach().numpy()

  polygons_gt = [gt]

  plt.subplot(2,2,1)
  plt.imshow(img)

  for polygon, polygon_gt in zip(polygons, polygons_gt):
    x,y = np.transpose(polygon)
    plt.plot(x*marker.shape[1], y*marker.shape[0], color='red')

    x,y = np.transpose(polygon_gt)
    plt.plot(x*marker.shape[1], y*marker.shape[0], color='green', alpha=0.7)

  plt.subplot(2,2,2)
  plt.imshow(marker)

  
  plt.show()