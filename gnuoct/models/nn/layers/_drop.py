# https://github.com/rwightman/pytorch-image-models/blob/02b806e00a24fa9c6e34187980700e1af927e817/timm/models/layers/drop.py

from __future__ import absolute_import

import torch
import torch.nn as nn


def drop_path(x: torch.Tensor,
              drop_prob: float,
              training: bool = False) -> torch.Tensor:
    if not training or drop_prob == 0:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.size(0),) + (1,) * (x.ndim - 1)
    w = x.new_empty(shape).bernoulli_(keep_prob)
    if 0 < keep_prob:
        w.div_(keep_prob)
    return w


class DropPath(nn.Module):

    def __init__(self, p: float):
        super().__init__()
        self.drop_prob = p

    def __repr__(self) -> str:
        return f'DropPath(p={self.drop_prob})'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)
