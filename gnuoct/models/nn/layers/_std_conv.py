from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from big_transfer.bit_pytorch.models import StdConv2d

__all__ = ['StdConv1d', 'StdConv2d']


class StdConv1d(nn.Conv1d):

    # pylint: disable=redefined-builtin
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv1d(input, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
