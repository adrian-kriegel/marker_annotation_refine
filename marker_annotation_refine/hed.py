#!/usr/bin/env python


import os
from dotenv import load_dotenv
import numpy
import torch
import torch.backends.cudnn
import numpy as np

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
    tenInput = tenInput * 255.0
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

  return output.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0]
# end

##########################################################

if __name__ == '__main__':

  import matplotlib.pyplot as plt
  import numpy as np
  from PIL import Image

  load_dotenv()

  dataset = PolygonDataset(
    os.environ['CITYSCAPES_LOCATION'],
  )

  for polygon in dataset:

    img = np.array(polygon.cropped_img())


    outimg = hed(img)

    plt.subplot(2,1,1)
    plt.imshow(img)

    plt.subplot(2,1,2)

    dx,dy = np.gradient(np.array(outimg))

    plt.imshow(outimg)

    plt.show()
# end