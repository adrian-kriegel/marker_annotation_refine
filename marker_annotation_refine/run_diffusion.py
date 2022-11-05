
import os
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import numpy as np

import torch

from marker_annotation_refine.marker_refine_dataset import \
  PolygonDataset

from marker_annotation_refine.train_diffusion import \
  create_input_tensor, \
  load_model

iterations = 300
step_size = 0.4

with torch.no_grad():

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  model = load_model()
  model.eval()
  model.to(device)

  load_dotenv()

  ds = PolygonDataset(
    os.environ['CITYSCAPES_LOCATION'],
    'val',
  )

  for polygon in ds:

    img_cam = polygon.cropped_img()

    if img_cam.width * img_cam.height > 50000:
      continue

    inp = create_input_tensor(
      img_cam,
      polygon.draw_random_marker(img_cam.width, img_cam.height),
    ).to(device)

    # set the initial noise input
    inp[0,4,:,:] = torch.rand_like(inp[0,4])

    for i in range(iterations):

      est_noise = model.forward(inp)

      inp[0,4,:,:] -= est_noise.reshape(inp.shape[2:4]) * step_size

    # resulting image will be contained in the input (see for-loop above)
    outimg = inp[0,4,:,:].cpu().numpy()

    outimg /= np.max(outimg)

    plt.subplot(1,3,1)

    plt.imshow(img_cam)

    plt.subplot(1,3,2)

    plt.imshow(polygon.draw_outline())

    plt.subplot(1,3,3)

    plt.imshow(outimg)

    plt.show()


