
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


from marker_annotation_refine.geometry_util import order_polygon
from marker_annotation_refine.iterator_batcher import IteratorBatcher
from marker_annotation_refine.pyramid_pooling import PyramidPooling
from marker_annotation_refine.vector_fields import VectorFieldDataset

class Encoder(nn.Module):
    
  def __init__(self):
    super().__init__()
    
    self.encoder_cnn = nn.Sequential(
      nn.Conv2d(5, 10, 4, stride=2, padding=1),
      nn.ReLU(True),
      nn.Conv2d(10, 16, 3, stride=2, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(True),
      nn.Conv2d(16, 32, 3, stride=2, padding=0),
      nn.ReLU(True), 
    )
    
        
  def forward(self, x):
    x = self.encoder_cnn(x)
    
    return x

class Decoder(nn.Module):
    
  def __init__(self):

    super().__init__()
  
    self.decoder_conv = nn.Sequential(
      nn.ConvTranspose2d(
        32, 16, 3, 
        stride=2,
        output_padding=0
      ),
      nn.BatchNorm2d(16),
      nn.ReLU(True),

      nn.ConvTranspose2d(
        16, 8, 8, 
        stride=2,
        output_padding=0
      ),
      nn.BatchNorm2d(8),
      nn.ReLU(True),

      nn.ConvTranspose2d(
        8, 1, 3, 
        stride=2,
        output_padding=0
      ),
    )
       
  def forward(self, x):

    x = self.decoder_conv(x)
    x = torch.sigmoid(x)
 
    return x

def train(
  encoder, decoder,
  train_dataset : VectorFieldDataset,
  out_dir : str,
  nepochs = 400,
  batch_size = 64,
  report_interval = 100
):

  loss_fn =  nn.BCELoss()
  
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

  loader = IteratorBatcher(train_dataset, batch_size)

  for epoch in range(1, nepochs+1):
    i = 0
    # loss sum for reporting interval
    loss_sum = 0
    for marked_img, gt in train_dataset:

      marked_img = marked_img.reshape((1, *marked_img.shape))
      gt = gt.reshape((1, *marked_img.shape))

      optimizer.zero_grad()
      
      output = decoder.forward(encoder.forward(marked_img.float().to(device)))

      print(np.any(np.isnan(marked_img.detach().numpy())))
      print(np.max(output.detach().cpu().numpy()))
      
      
      gt = gt.float().to(device)
      
      gt = transforms.Resize(output.shape[2:4])(gt)

      loss = loss_fn(output, gt.reshape((batch_size, 1, *gt.shape[1:3])))

      loss.backward()
      optimizer.step()
      loss_sum +=loss.item();

      i += 1

      if i % report_interval == 0:

        print(f'{epoch}: {loss_sum/report_interval}')
        loss_sum = 0

        torch.save(encoder.state_dict(), os.path.join(out_dir, 'encoder.pt'))
        torch.save(decoder.state_dict(), os.path.join(out_dir, 'decoder.pt'))

