
import os
import typing
from dotenv import load_dotenv


from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np

import torch
from torch import nn
from torchvision import transforms

from marker_annotation_refine.iterator_utils import \
  IteratorBatcher, \
  IteratorWrap

from marker_annotation_refine.marker_annotation import \
  draw_single_line

from marker_annotation_refine.marker_refine_dataset import \
  CSPolygon, \
  PolygonDataset

from marker_annotation_refine.unet import UNet

load_dotenv()

nepochs = 100000
batch_size = 1
report_interval = 1
model_path = 'models/unet.pt'
img_min_size = 32

polygons = PolygonDataset(
  os.environ['CITYSCAPES_LOCATION'],
  'train',
)

img_to_tensor = transforms.PILToTensor()
tensor_to_img = transforms.ToPILImage()

def load_polygon(polygon : CSPolygon):

  ''' Turns polygon into (input, output) pair of tensors. '''

  x0,y0,_,_ = polygon.bounding_box()

  img_cam = polygon.cropped_img()

  marker = polygon.random_marker()

  img_marker = Image.new('F', (img_cam.width, img_cam.height), 0)

  draw_single_line(
    ImageDraw.Draw(img_marker),
    x0, y0,
    marker['brushSize'],
    marker['points'],
  )

  inp = torch.zeros((3 + 1, img_cam.height, img_cam.width))

  inp[0:3,:,:] = img_to_tensor(img_cam)

  inp[3,:,:] = img_to_tensor(img_marker)

  gt = torch.from_numpy(polygon.draw_outline())

  w,h = img_cam.width, img_cam.height

  if w < img_min_size or h < img_min_size:

    crop = transforms.CenterCrop([
      max(img_min_size, h), 
      max(img_min_size, w)
    ])

    inp = crop(inp)
    gt = crop(gt)

  return inp, gt

def display_pairs(
  inp : torch.Tensor, 
  gt : torch.Tensor, 
  out : typing.Optional[torch.Tensor] = None
):

  import matplotlib.pyplot as plt

  for i in range(batch_size):

    marked_img = np.array(tensor_to_img(inp[i]))

    n = 3 if out == None else 4

    plt.subplot(1,n,1)
    plt.imshow(marked_img[:,:,0:3])
  
    plt.subplot(1,n,2)
    plt.imshow(marked_img[:,:,3])
  
    plt.subplot(1,n,3)
  
    plt.imshow(tensor_to_img(gt[i]))

    if not out == None:

      plt.subplot(1,n,4)
      plt.imshow(tensor_to_img(out[i]))

    plt.show()


train_dataset = IteratorWrap(polygons, load_polygon)

model = UNet()

try:

  model.load_state_dict(torch.load(model_path))

except OSError:
  print("Warning, no models found!")

loss_fn =  nn.MSELoss()

optimizer = torch.optim.Adam(
  model.parameters(),
  lr=0.001
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)
model.train()

loader = IteratorBatcher(train_dataset, batch_size)

for epoch in range(1, nepochs+1):
  i = 0
  # loss sum for reporting interval
  loss_sum = 0
  for marked_img, gt in loader:
    
    optimizer.zero_grad()

    output = model.forward(marked_img.float().to(device))
    
    gt = gt.float().to(device)
    
    gt = transforms.Resize(output.shape[2:4])(gt)

    loss = loss_fn(output, gt.reshape((batch_size, 1, *gt.shape[1:3])))

    loss.backward()
    optimizer.step()
    loss_sum +=loss.item()

    # display_pairs(marked_img, gt, output)

    i += 1

    if i % report_interval == 0:

      print(f'{epoch}: {loss_sum/report_interval}')
      loss_sum = 0

      torch.save(model.state_dict(), model_path)
