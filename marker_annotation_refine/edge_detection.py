#!/usr/bin/env python


import os
from PIL import Image
import typing
from dotenv import load_dotenv
import numpy
import torch
import torch.backends.cudnn
import numpy as np
from skimage.transform import resize
import numpy as np
from skimage import feature
from skimage.color import rgb2hsv

from marker_annotation_refine.marker_refine_dataset import \
  PolygonDataset

torch.set_grad_enabled(False)

torch.backends.cudnn.enabled = torch.backends.cudnn.is_available()

##########################################################

arguments_strModel = 'bsds500' 

class Network(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.netVggOne = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    self.netVggTwo = torch.nn.Sequential(
      torch.nn.MaxPool2d(kernel_size=2, stride=2),
      torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    self.netVggThr = torch.nn.Sequential(
      torch.nn.MaxPool2d(kernel_size=2, stride=2),
      torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    self.netVggFou = torch.nn.Sequential(
      torch.nn.MaxPool2d(kernel_size=2, stride=2),
      torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    self.netVggFiv = torch.nn.Sequential(
      torch.nn.MaxPool2d(kernel_size=2, stride=2),
      torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
    self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
    self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
    self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
    self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

    self.netCombine = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
      torch.nn.Sigmoid()
    )

    self.load_state_dict(
      { 
        strKey.replace('module', 'net'): tenWeight for strKey, 
        tenWeight in torch.hub.load_state_dict_from_url(
          url='http://content.sniklaus.com/github/pytorch-hed/network-' + arguments_strModel + '.pytorch', 
          file_name='hed-' + arguments_strModel).items() 
      }
    )
  # end

  def forward(self, tenInput):

    tenInput = tenInput - torch.tensor(data=[104.00698793, 116.66876762, 122.67891434], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

    tenVggOne = self.netVggOne(tenInput)
    tenVggTwo = self.netVggTwo(tenVggOne)
    tenVggThr = self.netVggThr(tenVggTwo)
    tenVggFou = self.netVggFou(tenVggThr)
    tenVggFiv = self.netVggFiv(tenVggFou)

    tenScoreOne = self.netScoreOne(tenVggOne)
    tenScoreTwo = self.netScoreTwo(tenVggTwo)
    tenScoreThr = self.netScoreThr(tenVggThr)
    tenScoreFou = self.netScoreFou(tenVggFou)
    tenScoreFiv = self.netScoreFiv(tenVggFiv)

    tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
    tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
    tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
    tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
    tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

    return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))
  # end
# end

netNetwork = None

##########################################################

def hed(img : np.ndarray):


  tenInput = torch.FloatTensor(
    numpy.ascontiguousarray(
      numpy.array(
        img[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
      )
    )
  )

  global netNetwork

  if netNetwork is None:

    netNetwork = Network().eval()
    
  # end

  intWidth = tenInput.shape[2]
  intHeight = tenInput.shape[1]

  output = netNetwork(tenInput.view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()

  return output.numpy().transpose(1, 2, 0)[:, :, 0]
# end

##########################################################

def alg_sum(imgs : list[np.ndarray]):

  '''
  Algebraic sum (fuzzy-or).
  '''

  return np.sum(imgs, axis=0) - np.product(imgs, axis=0)

def edge_detect(img, original_shape=False):


  # upscale the image if it is too small
  if img.shape[0] < 100 and img.shape[1] < 100:

    inp = resize(np.array(img, dtype=np.float32), (320, 480), anti_aliasing=False)
  
  else:

    inp = img
    

  outimg = hed(inp)
  outimg_t = hed(inp.transpose((1,0,2))).transpose((1,0))

  res = alg_sum([outimg, outimg_t])

  return resize(res, img.shape[0:2]) if original_shape else res

def canny(img : typing.Union[np.ndarray, Image.Image]):

  hsv = rgb2hsv(np.array(img))

  return (\
    0.5 * feature.canny(hsv[:,:,0]) + \
    0.25 * feature.canny(hsv[:,:,1]) + \
    0.25 * feature.canny(hsv[:,:,2])
  )

if __name__ == '__main__':

  import matplotlib.pyplot as plt

  load_dotenv()

  dataset = PolygonDataset(
    os.environ['CITYSCAPES_LOCATION'],
  )

  for polygon in dataset:

    img = np.array(polygon.cropped_img())

    outimg = edge_detect(img, True)

    plt.subplot(3,1,1)
    plt.title('Camera')
    plt.imshow(img)

    plt.subplot(3,1,2)

    plt.title('HED')
    plt.imshow(outimg)

    plt.subplot(3,1,3)

    plt.show()
# end