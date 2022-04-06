# https://github.com/rwightman/pytorch-image-models/blob/01a0e25a67305b94ea767083f4113ff002e4435c/timm/models/layers/helpers.py

import collections.abc
import itertools
from typing import Any, Callable, Tuple


def _ntuple(n: Any) -> Callable[[Any], Tuple[Any]]:

    def parse(x: Any):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(itertools.repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
