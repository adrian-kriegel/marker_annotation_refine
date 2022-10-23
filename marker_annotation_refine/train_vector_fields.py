


import os 
from dotenv import load_dotenv
import torch

from marker_annotation_refine.vector_fields import \
  VectorFieldDataset

from marker_annotation_refine.model import \
    Decoder, \
    Encoder, \
    train

load_dotenv()

train_dataset = VectorFieldDataset(
  os.environ['CITYSCAPES_LOCATION'],
  'train',
)

encoder = Encoder()
decoder = Decoder()

try:

  encoder.load_state_dict(torch.load('models_vec/encoder.pt'))
  decoder.load_state_dict(torch.load('models_vec/decoder.pt'))

except OSError:
  print("Warning, no models found!")

train(
  encoder,
  decoder,
  train_dataset,
  './models_vec/',
  batch_size=1,
  report_interval=1
)

