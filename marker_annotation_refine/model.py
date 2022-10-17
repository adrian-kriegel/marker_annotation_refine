
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


from marker_annotation_refine.geometry_util import order_polygon

from marker_annotation_refine.pyramid_pooling import PyramidPooling

from marker_annotation_refine.marker_refine_dataset import \
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

class PolygonDecoder(nn.Module):

  def __init__(
    self,
    num_points = 100,
    # last conv layer size
    filter_size = 32
  ):

    super().__init__()

    pyramid = PyramidPooling([5, 4])

    pyramid_output_size = pyramid.get_output_size(filter_size)

    self.num_points = num_points

    n = num_points*2
    
    self.decoder_pyramid = nn.Sequential(
      pyramid,
      nn.ReLU(True),
      nn.Linear(pyramid_output_size, pyramid_output_size),
      
      nn.ReLU(True),
      nn.Linear(pyramid_output_size, int(0.75 * pyramid_output_size + 0.25*n)),
      
      
      nn.ReLU(True),
      nn.Linear(int(0.75 * pyramid_output_size + 0.25*n),  int(0.5 * pyramid_output_size + 0.5*n)),
      
      
      nn.ReLU(True),
      nn.Linear(int(0.5 * pyramid_output_size + 0.5*n),  int(0.25 * pyramid_output_size + 0.75*n)),
      
      
      nn.ReLU(True),
      nn.Linear(int(0.25 * pyramid_output_size + 0.75*n), n)
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
  batch_size = 64
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
    for marked_img, gt in train_loader:

      optimizer.zero_grad()
      
      output = decoder.forward(encoder.forward(marked_img.float().to(device)))

      gt = gt.float().to(device)
      
      if train_dataset.fixed_shape == None and not train_dataset.return_polygon:
        gt = transforms.Resize(output.shape[2:4])(gt)


      if train_dataset.return_polygon:
        output_cpu = output.cpu().detach().numpy()
        # for each batch, adjust the polygon
        for i in range(len(gt)):

          gt[i] = order_polygon(gt[i], output_cpu[i])
      
      loss = loss_fn(output, gt)

      loss.backward()
      optimizer.step()

      i += 1

      if i % 100 == 0:

        print(f'{epoch}: {loss.item()}')

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

  encoder = Encoder(60)
  decoder = Decoder(60)

  train(
    encoder,
    decoder,
    train_dataset,
    './models/',
    batch_size=12
  )
