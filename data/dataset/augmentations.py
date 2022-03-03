from __future__ import absolute_import

import random
from abc import ABCMeta
from abc import abstractmethod
from typing import Optional, Tuple

import PIL
import PIL.Image
from PIL.Image import Image as PILImage


class _BaseAugmentationOp(metaclass=ABCMeta):

    def __init__(self,
                 magnitude_limit: Optional[Tuple[float, float]] = None,
                 random_mirror: Optional[bool] = None):
        super().__init__()

        self.magnitude_limit = magnitude_limit
        self.random_mirror = random_mirror

    def __call__(self, img: PILImage, magnitude: float) -> PILImage:
        if self.magnitude_limit is not None:
            min_magnitude, max_magnitude = self.magnitude_limit
            if not min_magnitude <= magnitude < max_magnitude:
                raise ValueError('Value of m exceeded the limitation: ' +
                                 '{} <= {} <= {}'.format(
                                     min_magnitude, magnitude, max_magnitude))

        if self.random_mirror and 0.5 < random.random():
            magnitude = -magnitude

        return self._apply_transformation(img, magnitude)

    @abstractmethod
    def _apply_transformation(self, img: PILImage, m: float) -> PILImage:
        pass

