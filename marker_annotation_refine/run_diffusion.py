
import os
import time
from PIL import Image
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import numpy as np

import torch
from marker_annotation_refine.edge_detection import canny, edge_detect

from marker_annotation_refine.marker_refine_dataset import \
  PolygonDataset

from marker_annotation_refine.noise import \
  perlin_noise

from marker_annotation_refine.train_diffusion import \
  create_input_tensor, \
  load_model, \
  transform_input_batch, \
  transform_output_batch

# max. number of iterations to perform
iterations = 12
# factor for each step
step_size = 0.5
# decay factor for step size (bigger -> faster descent)
step_decay = 2.0 / iterations
# break once mean of predicted noise is smaller than this value
noise_threshold = -1
# mixes the initial value back into the image each iteration (relative to current step size)
mix_initial = 0.0
# mixes in random perlin noise at each iteration
mix_noise = 0.0

normalize_input = False
clip_input = False

# cuts off less activated pixels
reduce_input = 0.1

target_size = (148, 148)

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
    )

    inp = transform_input_batch(inp).to(device)

    start = time.time()

    edges = edge_detect(np.array(img_cam), original_shape=True)
    edges += canny(img_cam)
    edges = torch.from_numpy(edges)

    print(f'Edge detection: {time.time() - start}')

    gt = np.array(Image.fromarray(polygon.draw_outline()).resize(target_size))

    # prime with edges from an edge detector
    tensor_marker = torch.from_numpy(np.array(img_marker))
    initial = (1.0 + tensor_marker) * edges
    initial = transform_output_batch(initial.reshape((1, *edges.shape)))

    # initial = torch.from_numpy(gt) + torch.rand_like(inp[0,4])
    inp[0,4,:,:] = initial.clone()

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

      if reduce_input > 0:

        inp[0,4,:,:] *= inp[0,4,:,:] > (reduce_input * torch.max(inp[0,4,:,:]))

      if mix_initial > 0:

        # re-introduce some of the initial priming features
        inp[0,4,:,:] += step_sizes[i]*mix_initial*initial

      if mix_noise > 0:

        h,w = inp[0,4,:,:].shape

        perlin = perlin_noise(
          (h, w),
          (h // 2, w // 2),
          seed=i
        )

        mix = step_sizes[i] * mix_noise

        inp[0,4,:,:] = inp[0,4,:,:]*(1.0-mix) + mix*perlin

      output = model.forward(inp)

      est_noise = output[:,0]
      est_structure = output[:,1]

      img_noise = (est_noise[0] / torch.max(est_noise)).numpy()
      img_structure = (est_structure[0] / torch.max(est_structure)).numpy()
      
      plt.subplot(1,2,1)
      plt.imshow(img_noise)

      plt.subplot(1,2,2)
      plt.imshow(img_noise)

      plt.show()

      # subtract some of the noise from the current input
      inp[0,4,:,:] -= est_noise.reshape(inp.shape[2:4]) * step_sizes[i]
      inp[0,4,:,:] += est_structure.reshape(inp.shape[2:4]) * step_sizes[i]

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


