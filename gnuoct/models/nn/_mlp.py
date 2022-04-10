from __future__ import absolute_import

from typing import Optional, Union, Iterable

import torch
import torch.nn as nn

from utils import to_2tuple


class Mlp(nn.Module):
    """https://github.com/rwightman/pytorch-image-models/blob/01a0e25a67305b94ea767083f4113ff002e4435c/timm/models/layers/mlp.py"""  # pylint: disable=line-too-long

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        dropout_rate: Union[float, Iterable[float]] = 0.,
    ):
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        dropout_rates = to_2tuple(dropout_rate)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dropout1 = nn.Dropout(p=dropout_rates[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(p=dropout_rates[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x
