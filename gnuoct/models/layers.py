import torch
import torch.nn as nn


class Transpose(nn.Module):

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(*self.dims)
