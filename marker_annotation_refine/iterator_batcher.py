
import typing
from typing_extensions import Self
import numpy as np

import torch


class IteratorBatcher:

  def __init__(
    self,
    dataset, # TODO typing : typing.Iterable,
    batch_size = 10
  ):

    self.dataset = dataset
    self.batch_size = batch_size

  def __iter__(self):

    self.dataset.__iter__()

    return self

  def __next__(self):

    pairs = [
      self.dataset.__next__() for i in range(self.batch_size)
    ]

    return (
      torch.tensor(np.array([pair[0] for pair in pairs])),
      torch.tensor(np.array([pair[1] for pair in pairs]))
    )
