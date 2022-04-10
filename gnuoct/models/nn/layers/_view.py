import torch
import torch.nn as nn


class View(nn.Module):

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def extra_repr(self) -> str:
        dims = ', '.join(map(str, self.dims))
        return f'View({dims})'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*self.dims)
