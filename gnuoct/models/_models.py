import inspect
import os
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import big_transfer.bit_pytorch.models

_BiTResNet50 = big_transfer.bit_pytorch.models.KNOWN_MODELS['BiT-M-R50x1']


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


class StdConv1d(nn.Conv1d):

    # pylint: disable=redefined-builtin
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv1d(input, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class LKDepthwiseConv1dBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.conv_dw: StdConv1d
        self.add_module(
            'conv_dw',
            StdConv1d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2,
                      groups=in_channels,
                      bias=False))
        self.conv_1x1: StdConv1d
        self.add_module(
            'conv_1x1',
            StdConv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_dw(x)
        x = self.conv_1x1(x)
        return x


class OCTVolumeConv1dNet(nn.Module):

    def __init__(
        self,
        feature_extractor: Union[Type[nn.Module], nn.Module] = ResNet50,
        feature_extractor_args: Optional[Iterable[Any]] = None,
        feature_extractor_kargs: Optional[Dict[str, Any]] = None,
        n_classes: int = 4,
    ):
        super().__init__()

        if inspect.isclass(feature_extractor):
            if feature_extractor_args is None:
                feature_extractor_args = []
            if feature_extractor_kargs is None:
                feature_extractor_kargs = {}
            feature_extractor = feature_extractor(*feature_extractor_args,
                                                  **feature_extractor_kargs)
        self.feature_extractor: nn.Module
        self.add_module('feature_extractor', feature_extractor)

        self.norm: nn.LayerNorm
        self.norm = nn.LayerNorm(2048)

        self.body: nn.Sequential
        # yapf: disable
        # pylint: disable=line-too-long
        self.add_module('body', nn.Sequential(OrderedDict([
            ('lkconv_block1', LKDepthwiseConv1dBlock(2048, 2048, 19)),
            ('activation1', nn.GELU()),
            ('lkconv_block2', LKDepthwiseConv1dBlock(2048, 2048, 19)),
            ('activation2', nn.GELU()),
        ])))

        self.head: nn.Sequential
        self.add_module('head', nn.Sequential(OrderedDict([
            ('avg_pool', nn.AdaptiveAvgPool1d(1)),
            ('flatten', nn.Flatten(start_dim=1)),
            ('fc', nn.Linear(2048, n_classes)),
        ])))
        # yapf: enable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = F.adaptive_avg_pool2d(x, 1)[..., 0, 0]
        x = x.view(x.size(0) // 32, 32, -1)
        x = self.norm(x)
        x = self.body(x.transpose(1, 2))
        x = F.adaptive_avg_pool1d(x, 1)[..., 0]
        return x
