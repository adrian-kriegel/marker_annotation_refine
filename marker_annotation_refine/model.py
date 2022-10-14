
import os
from matplotlib import image

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataloader
from torchvision import transforms

from PIL import Image

from marker_refine_dataset import \
  MarkerRefineDataset

class Encoder(nn.Module):
    
  def __init__(self, encoded_space_dim):
    super().__init__()
    
    ### Convolutional section
    self.encoder_cnn = nn.Sequential(
      nn.Conv2d(4, 8, 3, stride=2, padding=1),
      nn.ReLU(True),
      nn.Conv2d(8, 16, 3, stride=2, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(True),
      nn.Conv2d(16, 32, 3, stride=2, padding=0),
      nn.ReLU(True)
    )
    
        
  def forward(self, x):
    x = self.encoder_cnn(x)

    return x

class Decoder(nn.Module):
    
  def __init__(self, encoded_space_dim):

   super().__init__()

   self.decoder_conv = nn.Sequential(
     nn.ConvTranspose2d(32, 16, 3, 
     stride=2, output_padding=0),
     nn.BatchNorm2d(16),
     nn.ReLU(True),
     nn.ConvTranspose2d(16, 8, 3, stride=2, 
     padding=1, output_padding=1),
     nn.BatchNorm2d(8),
     nn.ReLU(True),
     nn.ConvTranspose2d(8, 1, 3, stride=2, 
     padding=1, output_padding=1)
   )
       
  def forward(self, x):

   x = self.decoder_conv(x)
   x = torch.sigmoid(x)

   return x

def train(
  nepochs = 400,
  batch_size=1
):

  encoder = Encoder(60)
  decoder = Decoder(60)

  from dotenv import load_dotenv

  load_dotenv()

  train_dataset = MarkerRefineDataset(
    os.environ['CITYSCAPES_LOCATION'],
    'train'
  )

  test_dataset = MarkerRefineDataset(
    os.environ['CITYSCAPES_LOCATION'],
    'test'
  )

  loss_fn = nn.BCELoss()
  
  optimizer = torch.optim.Adam(
    [
      {'params': encoder.parameters()},
      {'params': decoder.parameters()}
    ],
    lr=0.001
  )

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  encoder.to(device)
  decoder.to(device)

  encoder.train()
  decoder.train()

  for epoch in range(1, nepochs+1):
    i = 0
    for v in train_dataset:
      if v == None:
        continue
    
      marked_img, gt = v

      inp = torch.from_numpy(marked_img).float().reshape((1, *marked_img.shape)).to(device)
      optimizer.zero_grad()
      
      output = decoder.forward(encoder.forward(inp))
      
      gt = torch.from_numpy(gt).float().reshape((1, *gt.shape)).to(device)

      gt = transforms.Resize(output.shape[2:4])(gt)

      loss = loss_fn(output[0], gt)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      i += 1

      if i % 100 == 0:

        print(f'{epoch}: {loss.item()}')

        torch.save(encoder.state_dict(), './models/marker_refine_encoder.pt')
        torch.save(decoder.state_dict(), './models/marker_refine_decoder.pt')
        

if __name__ == '__main__':

  train()