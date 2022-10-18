
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.filters import gaussian
import dotenv


from marker_annotation_refine.marker_refine_dataset import MarkerRefineDataset, split_marked_image
from marker_annotation_refine.geometry_util import mask_to_polygons
from marker_annotation_refine.model import Encoder, Decoder

dotenv.load_dotenv()

polygon_length = 100

dataset = MarkerRefineDataset(
  os.environ['CITYSCAPES_LOCATION'],
  'val',
  return_polygon=True,
  polygon_length=100,
)

encoder = Encoder()
decoder = Decoder()

try:

  encoder.load_state_dict(
    torch.load('models/marker_refine_encoder.pt', map_location=torch.device('cpu'))
  )

  decoder.load_state_dict(
    torch.load('models/marker_refine_decoder.pt', map_location=torch.device('cpu'))
  )

except OSError:
  print("No models found")
  exit()

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

  if np.sum(gt) == 0:
    continue

  inp = torch.from_numpy(marked_img).float().reshape((1, *marked_img.shape))

  _inp = inp.clone()

  output = None

  output = decoder.forward(encoder.forward(_inp))
  
  out_img = output[0,0].cpu().detach().numpy()
  out_img = gaussian(out_img, np.min(out_img.shape)*0.02)

  inp = inp.cpu().detach().numpy()

  img,marker = split_marked_image(inp)  

  polygons = mask_to_polygons(out_img > thrs)

  plt.subplot(2,2,1)
  plt.imshow(img)

  for polygon in polygons:
    x,y = np.transpose(polygon)
    plt.plot(x, y, color='red')


  plt.subplot(2,2,2)
  plt.imshow(marker)

  plt.subplot(2,2,3)
  plt.imshow(out_img)

  plt.show()