from __future__ import absolute_import

import torch
import torch.nn as nn


class Permute(nn.Module):

    def __init__(self, *dims):
        super().__init__()
        self.dims = tuple(dims)

    def extra_repr(self) -> str:
        return f'Permute(dims={", ".join(self.dims)})'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)
