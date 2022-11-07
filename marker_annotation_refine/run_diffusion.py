
import os
import time
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import numpy as np

from skimage import feature

import torch
from marker_annotation_refine.edge_detection import edge_detect

from marker_annotation_refine.marker_refine_dataset import \
  PolygonDataset

from marker_annotation_refine.train_diffusion import \
  create_input_tensor, \
  load_model

# max. number of iterations to perform
iterations = 20
# factor for each step
step_size = 0.5#10.0 / iterations
# decay factor for step size (bigger -> faster descent)
step_decay = 1.0 / iterations
# break once mean of predicted noise is smaller than this value
noise_threshold = -1
# mixes the initial value back into the image each iteration (relative to current step size)
mix_initial = 0.1

normalize_input = False
clip_input = True

step_sizes = np.exp(- step_decay * np.arange(iterations)) * step_size

with torch.no_grad():

  # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  device = torch.device('cpu')
  model = load_model(device=device)
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

    img_marker = polygon.draw_random_marker(img_cam.width, img_cam.height)

    inp = create_input_tensor(
      img_cam,
      img_marker,
    ).to(device)

    start = time.time()

    edges = edge_detect(np.array(img_cam), original_shape=True)
    edges = (feature.canny(np.array(img_cam)[:,:,0]) + feature.canny(np.array(img_cam)[:,:,1]) + feature.canny(np.array(img_cam)[:,:,2])) / 3.0
    edges = torch.from_numpy(edges)

    print(f'Edge detection: {time.time() - start}')

    gt = np.array(polygon.draw_outline())

    initial = torch.from_numpy(np.array(img_marker))

    # prime with edges from an edge detector
    # initial = torch.maximum(torch.rand_like(inp[0,4]), edges)

    # initial = torch.from_numpy(gt) + torch.rand_like(inp[0,4])
    inp[0,4,:,:] = torch.clone(initial)

    # tracks evolution of mean of noise predictions
    noise_per_iteration = np.zeros((iterations))
    error_per_iteration = np.zeros((iterations))
    start = time.time()

    outimg = np.zeros(inp[0,4].shape)

    for i in range(iterations):

      if clip_input:

        inp[0,4,:,:] = torch.clip(inp[0,4,:,:], -1.0, 1.0)

      # normalize the input
      if normalize_input:
        inp[0,4,:,:] -= torch.min(inp[0,4,:,:])
        inp[0,4,:,:] /= torch.max(inp[0,4,:,:])
      

      est_noise = model.forward(inp)

      # subtract some of the noise from the current input
      inp[0,4,:,:] -= est_noise.reshape(inp.shape[2:4]) * step_sizes[i]

      # re-introduce some of the initial priming features
      inp[0,4,:,:] += step_sizes[i]*mix_initial*edges

      

      noise_mean = np.mean(np.abs(est_noise.cpu().numpy()))

      noise_per_iteration[i] = noise_mean

      if noise_mean < noise_threshold:
        break

      outimg = inp[0,4,:,:].cpu().numpy()
      outimg -= np.min(outimg)
      outimg /= np.max(outimg)

      error_per_iteration[i] = np.mean(np.abs(outimg - gt))

    print(f'Diffusion: {time.time() - start}')

    outimg /= np.max(outimg)

    plt.subplot(1,4,1)

    plt.imshow(img_cam)

    plt.subplot(1,4,2)

    plt.imshow(polygon.draw_outline())

    plt.subplot(1,4,3)

    plt.imshow(initial.cpu().numpy())

    plt.subplot(1,4,4)

    plt.imshow(outimg)

    plt.figure()

    plt.plot(range(iterations), noise_per_iteration, label='noise int.')
    plt.plot(range(iterations), error_per_iteration, label='error')

    plt.legend()

    plt.show()


