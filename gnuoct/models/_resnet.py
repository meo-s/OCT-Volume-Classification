from __future__ import absolute_import

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from big_transfer.bit_pytorch.models import KNOWN_MODELS

__all__ = 'ResNet50'.split()

_BiTResNet50 = KNOWN_MODELS['BiT-M-R50x1']


class ResNet50(nn.Module):

    def __init__(self, prt_model_path: Optional[str] = None):
        super().__init__()

        resnet50 = _BiTResNet50(head_size=1, zero_head=True)
        if prt_model_path is not None:
            if not os.path.exists(prt_model_path):
                raise ValueError(
                    'Given path to the pretrained model is invalid: {}'.format(
                        prt_model_path))
            resnet50.load_from(np.load(prt_model_path))

        self.root: nn.Module
        self.add_module('root', resnet50.root)
        self.body: nn.Module
        self.add_module('body', resnet50.body)

        del resnet50

    def forward(self, x: torch.Tensor):
        x = self.body(self.root(x))
        return x
