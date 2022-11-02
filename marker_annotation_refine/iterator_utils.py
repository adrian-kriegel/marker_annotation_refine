
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
      # TODO: check what happens on stop iteration and handle accordingly
      self.dataset.__next__() for i in range(self.batch_size)
    ]

    return (
      torch.stack([pair[0] for pair in pairs]),
      torch.stack([pair[1] for pair in pairs])
    )

class IteratorWrap:

  def __init__(self, it, wrap : typing.Callable):

    self.it = it
    self.wrap = wrap

  def __iter__(self):

    self.it.__iter__()
    return self

  def __next__(self):

    return self.wrap(self.it.__next__())
