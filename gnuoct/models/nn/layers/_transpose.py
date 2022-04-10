import torch
import torch.nn as nn


class Transpose(nn.Module):

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def extra_repr(self) -> str:
        return 'Transpose(dim0={}, dim1={})'.format(*self.dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(*self.dims)
