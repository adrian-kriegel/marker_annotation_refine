
import os
import time
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import numpy as np

import torch
from marker_annotation_refine.edge_detection import edge_detect

from marker_annotation_refine.marker_refine_dataset import \
  PolygonDataset

from marker_annotation_refine.train_diffusion import \
  create_input_tensor, \
  load_model

# max. number of iterations to perform
iterations = 100
# factor for each step
step_size = 0.7
# decay factor for step size (bigger -> faster descent)
step_decay = 0.2
# break once mean of predicted noise is smaller than this value
noise_threshold = 0.11

step_sizes = np.exp(- step_decay * np.arange(iterations)) * step_size

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

    start = time.time()
    edges = torch.from_numpy(edge_detect(np.array(img_cam), original_shape=True))

    print(f'Edge detection: {time.time() - start}')

    # set the initial noise input
    # prime with edges from an edge detector
    inp[0,4,:,:] = edges # torch.rand_like(inp[0,4])

    # tracks evolution of mean of noise predictions
    noise_per_iteration = np.zeros((iterations))

    start = time.time()

    for i in range(iterations):

      est_noise = model.forward(inp)

      inp[0,4,:,:] -= est_noise.reshape(inp.shape[2:4]) * step_sizes[i]

      noise_mean = np.mean(np.abs(est_noise.cpu().numpy()))

      noise_per_iteration[i] = noise_mean

      if noise_mean < noise_threshold:
        break

    print(f'Diffusion: {time.time() - start}')

    # resulting image will be contained in the input (see for-loop above)
    outimg = inp[0,4,:,:].cpu().numpy()

    outimg /= np.max(outimg)

    plt.subplot(1,4,1)

    plt.imshow(img_cam)

    plt.subplot(1,4,2)

    plt.imshow(polygon.draw_outline())

    plt.subplot(1,4,3)

    plt.imshow(edges)

    plt.subplot(1,4,4)

    plt.imshow(outimg)

    # plt.figure()

    # plt.plot(range(iterations), noise_per_iteration)
    # plt.plot(range(iterations), step_sizes)

    plt.show()


