# PyTorch implementation of "CBAM: Convolutional Block Attention Module".
# paper link: https://arxiv.org/abs/1807.06521

from __future__ import absolute_import

from typing import Any, Callable, Iterable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import to_2tuple


class ChannelAttentionModule(nn.Module):

    # pylint: disable=line-too-long
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: float = 16,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        dropout_rate: Union[float, Iterable[float]] = 0.,
    ):
        super().__init__()

        dropout_rates = to_2tuple(dropout_rate)

        # yapf: disable
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rates[0])
        self.act = act_layer()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rates[1])
        self.sigmoid = nn.Sigmoid()
        # yapf: enable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.new_zeros((*x.shape[:2], 1, 1))
        for v in [F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)]:
            v = self.dropout1(self.act(self.fc1(v)))
            v = self.dropout2(self.fc2(v))
            w = w + v

        return x * self.sigmoid(w)


class SpatialAttentionModule(nn.Module):

    # pylint: disable=line-too-long
    def __init__(
        self,
        kernel_size: Union[int, Iterable[int]] = 3,
        conv_layer=nn.Conv2d,
    ):
        super().__init__()

        kernel_size = to_2tuple(kernel_size)
        padding = [(k - 1) // 2 for k in kernel_size]

        # yapf: disable
        self.conv = conv_layer(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
        # yapf: enable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        w = self.conv(w)
        return x * self.sigmoid(w)


class CBAM(nn.Module):

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        dropout_rate: Union[float, Iterable[float]] = 0.,
        cattn_module: Callable[[Any], nn.Module] = ChannelAttentionModule,
        sattn_module: Callable[[Any], nn.Module] = SpatialAttentionModule,
    ):
        super().__init__()

        self.cattn = cattn_module(in_channels=in_channels,
                                  reduction_ratio=reduction_ratio,
                                  dropout_rate=dropout_rate)
        self.sattn = sattn_module()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cattn(x)
        x = self.sattn(x)
        return x
