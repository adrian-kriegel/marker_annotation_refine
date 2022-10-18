
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


from marker_annotation_refine.geometry_util import order_polygon

from marker_annotation_refine.pyramid_pooling import PyramidPooling

from marker_annotation_refine.marker_refine_dataset import \
  MarkerRefineDataset

class Encoder(nn.Module):
    
  def __init__(self):
    super().__init__()
    
    self.encoder_cnn = nn.Sequential(
      nn.Conv2d(4, 8, 4, stride=2, padding=1),
      nn.ReLU(True),
      nn.Conv2d(8, 16, 3, stride=2, padding=1),
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

class PolygonDecoder(nn.Module):

  def __init__(
    self,
    num_points = 100,
    # last conv layer size
    filter_size = 32
  ):

    super().__init__()

    pyramid = PyramidPooling([5])

    pyramid_output_size = pyramid.get_output_size(filter_size)

    self.num_points = num_points

    n = num_points*2
    
    self.decoder_pyramid = nn.Sequential(
      pyramid,
      nn.ReLU(True),
      
      nn.Linear(pyramid_output_size,  int(0.5 * pyramid_output_size + 0.5*n)),
      
      nn.ReLU(True),
      nn.Linear(int(0.5 * pyramid_output_size + 0.5*n), n)
    )

    self.filter_size = filter_size

  def forward(self, x):

    x = self.decoder_pyramid(x)

    x = torch.sigmoid(x)

    # shape = (batch, 2*num_points) -> (batch, num_points, 2)
    x = x.reshape((x.shape[0], self.num_points, 2))
    
    return x

def train(
  encoder, decoder,
  train_dataset : MarkerRefineDataset,
  out_dir : str,
  nepochs = 400,
  batch_size = 64,
  report_interval = 100
):


  loss_fn =  nn.MSELoss() if train_dataset.return_polygon else nn.BCELoss()
  
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

  train_loader = DataLoader(train_dataset, batch_size=batch_size)

  for epoch in range(1, nepochs+1):
    i = 0
    # loss sum for reporting interval
    loss_sum = 0
    for marked_img, gt in train_loader:

      # TODO: use collate_fn to filter out bad data points
      if train_dataset.return_polygon and (np.sum(gt.numpy()) == 0):

        continue

      optimizer.zero_grad()
      
      output = decoder.forward(encoder.forward(marked_img.float().to(device)))

      gt = gt.float().to(device)
      
      if train_dataset.fixed_shape == None and not train_dataset.return_polygon:
        gt = transforms.Resize(output.shape[2:4])(gt)


      if train_dataset.return_polygon:
        output_cpu = output.cpu().detach().numpy()
        gt_cpu = gt.cpu().detach().numpy()
        # for each batch, adjust the polygon
        for j in range(len(gt_cpu)):

          gt[j] = order_polygon(gt_cpu[j], output_cpu[j])

        gt = torch.from_numpy(gt).to(device)
      
      loss = loss_fn(output, gt)

      loss.backward()
      optimizer.step()
      loss_sum +=loss.item();

      i += 1

      if i % report_interval == 0:

        print(f'{epoch}: {loss_sum/report_interval}')
        loss_sum = 0

        torch.save(encoder.state_dict(), os.path.join(out_dir, 'marker_refine_encoder.pt'))
        torch.save(decoder.state_dict(), os.path.join(out_dir, 'marker_refine_decoder.pt'))


if __name__ == '__main__':

  from dotenv import load_dotenv

  load_dotenv()

  train_dataset = MarkerRefineDataset(
    os.environ['CITYSCAPES_LOCATION'],
    'train',
    fixed_shape=(508,508)
  )

  encoder = Encoder()
  decoder = Decoder()

  train(
    encoder,
    decoder,
    train_dataset,
    './models/',
    batch_size=12
  )
