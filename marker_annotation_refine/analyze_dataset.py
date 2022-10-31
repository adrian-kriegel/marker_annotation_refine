
import os
from dotenv import load_dotenv
import numpy as np

from marker_annotation_refine.marker_refine_dataset import \
  CSDataset, \
  CSInstance \

limit = 100

load_dotenv()

dataset_path = os.environ['CITYSCAPES_LOCATION']

print(f'Analyzing max. {limit} images from {dataset_path}...')

dataset = CSDataset(
  dataset_path,
  mode='train'
)

num_imgs = 0
num_instances = 0
num_polygons = 0

for img in dataset:

  num_instances += len(img.instance_ids())

  instances = [CSInstance(img, instance_id) for instance_id in img.instance_ids()]
  instances = [inst for inst in instances if inst.is_valid()]

  num_polygons += np.sum([len(inst.polygons()) for inst in instances])

  num_imgs += 1

  if num_imgs > limit:

    break

print(f'Analyzed {num_imgs} Images from dataset.')
print(f'Avg. instances per image: {num_instances / num_imgs} (tot. {num_instances})')
print(f'Avg. polygons per image: {num_polygons / num_imgs} (tot. {num_polygons})')

