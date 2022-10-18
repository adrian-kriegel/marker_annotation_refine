
import os 
from dotenv import load_dotenv
import torch

from marker_annotation_refine.marker_refine_dataset import \
    MarkerRefineDataset

from marker_annotation_refine.model import \
    Decoder, \
    Encoder, \
    train

load_dotenv()

train_dataset = MarkerRefineDataset(
  os.environ['CITYSCAPES_LOCATION'],
  'train',
  max_blur=0.05,
  gt_blur=0.03,
  gt_blur_mix=0.8,
  gt_fill_amount=0.2
)

encoder = Encoder()
decoder = Decoder()

try:

  encoder.load_state_dict(torch.load('models/marker_refine_encoder.pt'))
  decoder.load_state_dict(torch.load('models/marker_refine_decoder.pt'))

except OSError:
  print("Warning, no models found!")

train(
  encoder,
  decoder,
  train_dataset,
  './models/',
  batch_size=1,
  report_interval=1
)
