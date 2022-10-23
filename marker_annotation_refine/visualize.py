
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.filters import gaussian
import dotenv

from marker_annotation_refine.vector_fields import VectorFieldDataset
from marker_annotation_refine.geometry_util import mask_to_polygons
from marker_annotation_refine.model import Encoder, Decoder

dotenv.load_dotenv()

polygon_length = 100

dataset = VectorFieldDataset(
  os.environ['CITYSCAPES_LOCATION'],
  'val',
)

encoder = Encoder()
decoder = Decoder()

try:

  encoder.load_state_dict(
    torch.load('models_vec/encoder.pt', map_location=torch.device('cpu'))
  )

  decoder.load_state_dict(
    torch.load('models_vec/decoder.pt', map_location=torch.device('cpu'))
  )

except OSError:
  print("No models found")
  exit()

encoder.eval()
decoder.eval()

encoder.to('cpu')
decoder.to('cpu')

matplotlib.use('TkAgg')

for marked_img, gt in dataset:

  if np.sum(gt) == 0:
    continue

  inp = torch.from_numpy(marked_img).float().reshape((1, *marked_img.shape))

  _inp = inp.clone()

  output = None

  output = decoder.forward(encoder.forward(_inp))
  
  out_img = output[0,0].cpu().detach().numpy()
  out_img = gaussian(out_img, np.min(out_img.shape)*0.02)

  inp = inp.cpu().detach().numpy()

  img = np.moveaxis(marked_img, 0, 2)

  plt.subplot(2,2,1)
  plt.imshow(img[:,:,0:3])

  plt.subplot(2,2,2)
  plt.quiver(img[:,:,3], img[:,:,4])

  plt.subplot(2,2,3)
  plt.imshow(out_img)

  plt.show()