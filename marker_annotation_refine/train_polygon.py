
import os 
from dotenv import load_dotenv
import torch
from marker_refine_dataset import MarkerRefineDataset

from model import Encoder, PolygonDecoder, train

load_dotenv()

polygon_length = 100

train_dataset = MarkerRefineDataset(
  os.environ['CITYSCAPES_LOCATION'],
  'train',
  return_polygon=True,
  polygon_length=polygon_length
)

encoder = Encoder(60)
decoder = PolygonDecoder(num_points=polygon_length)

try:

  encoder.load_state_dict(torch.load('models_polygon/marker_refine_encoder.pth'))
  decoder.load_state_dict(torch.load('models_polygon/marker_refine_encoder.pth'))

except OSError:
  print("Warning, no models found!")

train(
  encoder,
  decoder,
  train_dataset,
  './models_polygon/',
  batch_size=1
)
