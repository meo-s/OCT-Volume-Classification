from __future__ import absolute_import

from collections import OrderedDict
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from . import ResNet50
from .nn import cbam
from .nn import Mlp
from .nn.layers import DropPath
from .nn.layers import StdConv2d
from .nn.layers import Transpose
from .nn.layers import View

__all__ = [
    'TokenwiseChannelAttnBlock', 'TokenwiseSpatialAttnBlock',
    'TokenwiseAttnBlock', 'OCTVolumeTokenAttnNet'
]


class TokenwiseSpatialAttnBlock(nn.Module):

    # yapf: disable
    # pylint: disable=line-too-long
    def __init__(
        self,
        n_tokens: int,
        in_channels: int,
        chid: Optional[int] = None,
        attn_dropout_rate: float = 0.,
    ):
        super().__init__()

        chid = chid or (n_tokens * 4)

        self.n_tokens: torch.LongTensor
        self.register_buffer('n_tokens', torch.LongTensor([n_tokens]))

        self.gn1 = nn.GroupNorm(32, in_channels)
        self.conv1 = StdConv2d(n_tokens * 2, n_tokens * 2, kernel_size=7, padding=3, groups=(n_tokens * 2))
        self.gn2 = nn.GroupNorm(32, n_tokens * 2)
        self.conv2 = StdConv2d(n_tokens * 2, chid, kernel_size=1)
        self.gn3 = nn.GroupNorm(32, chid)
        self.conv3 = StdConv2d(chid, n_tokens, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.attn_dropout = nn.Dropout(p=attn_dropout_rate)
    # yapf: enable

    # pylint: disable=invalid-name
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_tokens = self.n_tokens.item()
        _, C, H, W = x.size()

        out = self.relu(self.gn1(x))
        v_max = out.amax(1, keepdim=True).view(-1, n_tokens, H, W)
        v_avg = out.mean(1, keepdim=True).view(-1, n_tokens, H, W)

        w = torch.cat([v_max, v_avg], 1)
        w = self.conv1(w)
        w = self.conv2(self.relu(self.gn2(w)))
        w = self.conv3(self.relu(self.gn3(w)))
        w = self.sigmoid(w)

        w = w.view(-1, n_tokens, 1, H, W)
        out = out.view(-1, n_tokens, C, H, W)
        return self.attn_dropout((out * w).view(x.size()))


class TokenwiseChannelAttnBlock(nn.Module):

    # yapf: disable
    def __init__(
        self,
        n_tokens: int,
        in_channels: int,
        mlp_ratio: Union[float, Iterable[float]] = (1., 2.),
        dropout_rate: Union[float, Iterable[Union[float, Iterable[float]]]] = 0.,
        attn_dropout_rate: float = 0.,
    ):
        super().__init__()

        mlp_ratios = utils.to_2tuple(mlp_ratio)
        dropout_rates = utils.to_2tuple(dropout_rate)

        self.n_tokens: torch.LongTensor
        self.register_buffer('n_tokens', torch.LongTensor([n_tokens]))

        self.gn = nn.GroupNorm(32, in_channels)  # pylint: disable=invalid-name
        self.relu = nn.ReLU(inplace=True)

        thid = int(n_tokens * mlp_ratios[0])
        self.tokenwise = nn.Sequential(OrderedDict([
            ('T2BCL', Transpose(1, 2)),
            ('norm', nn.LayerNorm(n_tokens)),
            ('mlp', Mlp(n_tokens, thid, dropout_rate=dropout_rates[0])),
            ('T2BLC', Transpose(1, 2)),
        ]))

        chid = int(in_channels * mlp_ratios[1])
        self.channelwise = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(in_channels)),
            ('mlp', Mlp(in_channels, chid, dropout_rate=dropout_rates[1])),
        ]))

        self.attn_dropout = nn.Dropout(p=attn_dropout_rate)
    # yapf: enable

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.tokenwise(x)
        x = x + self.channelwise(x)
        return x

    # pylint: disable=invalid-name
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.n_tokens.item()
        C = x.size(1)

        out = self.relu(self.gn(x))

        v_avg = F.adaptive_avg_pool2d(x, 1).view(-1, L, C)
        w_avg = self._attn(v_avg)

        v_max = F.adaptive_max_pool2d(x, 1).view(-1, L, C)
        w_max = self._attn(v_max)

        w = (w_avg + w_max).sigmoid()
        w = w.view(*x.shape[:2], 1, 1)
        return self.attn_dropout(out * w)


class TokenwiseAttnBlock(nn.Module):

    # yapf: disable
    def __init__(
        self,
        n_tokens: int,
        in_channels: int,
        mlp_ratio: Union[float, Iterable[float]] = (1., 2.),
        dropout_rate: Union[float, Iterable[Union[float, Iterable[float]]]] = 0.,
        attn_dropout_rate: Union[float, Iterable[float]] = 0.,
        drop_prob: Union[float, Iterable[float]] = 0.,
    ):
        super().__init__()

        attn_dropout_rates = utils.to_2tuple(attn_dropout_rate)
        drop_probs = utils.to_2tuple(drop_prob)

        self.cattn = TokenwiseChannelAttnBlock(
            n_tokens=n_tokens,
            in_channels=in_channels,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rates[0],
        )

        self.drop_cattn = DropPath(p=drop_probs[0])

        self.sattn = nn.Sequential(OrderedDict([
            ('gn', nn.GroupNorm(32, in_channels)),
            ('relu', nn.ReLU(inplace=True)),
            ('sam', cbam.SpatialAttentionModule(conv_layer=StdConv2d)),
        ]))

        self.drop_sattn = DropPath(p=drop_probs[1])
    # yapf: enable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_cattn(self.cattn(x))
        x = x + self.drop_sattn(self.sattn(x))
        return x


class OCTVolumeTokenAttnNet(nn.Module):

    # yapf: disable
    # pylint: disable=line-too-long
    def __init__(
        self,
        prt_model_path: Optional[str] = None,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        resnet50 = ResNet50(prt_model_path=prt_model_path)
        self.root = resnet50.root
        self.body = resnet50.body
        self.head = nn.Sequential(OrderedDict([
            ('norm', nn.GroupNorm(32, 1024)),
            ('relu', nn.ReLU(inplace=True)),
            ('avg_pool', nn.AdaptiveAvgPool2d(1)),
            ('view', View(-1, 32, 1024)),
            ('transpose', Transpose(1, 2)),
            ('mean', nn.AdaptiveAvgPool1d(1)),
            ('flatten', nn.Flatten(1)),
            ('dropout_classifier', nn.Dropout(p=dropout_rate)),
            ('classifier', nn.Linear(1024, 4)),
        ]))

        self.attn1 = TokenwiseAttnBlock(32, 256, mlp_ratio=(1., 1.), dropout_rate=0.1, drop_prob=0.1)
        self.attn2 = TokenwiseAttnBlock(32, 512, mlp_ratio=(1., 1.), dropout_rate=0.2, drop_prob=0.3)
        self.attn3 = TokenwiseAttnBlock(32, 1024, mlp_ratio=(1., 1.), dropout_rate=0.2, drop_prob=0.5)

        del resnet50.body.block4
        del resnet50
    # yapf: enable

    def init_weights(self):
        for n, m in self.named_modules():
            if n.startswith(('root', 'body')):
                continue
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear):
                if 'classifier' in n:
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.trunc_normal_(m.weight, std=0.02)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.root(x)
        x = self.attn1(self.body.block1(x))
        x = self.attn2(self.body.block2(x))
        x = self.attn3(self.body.block3(x))
        x = self.head(x)
        return x
