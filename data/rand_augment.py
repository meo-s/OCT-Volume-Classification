from __future__ import absolute_import

import random
from typing import Iterable, Optional, Tuple

from PIL.Image import Image as PILImage

from data import augmentations as augs

# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py


def _default_augmentation_list(
) -> Tuple[Tuple[augs.AugmentationOp, float, float]]:
    # not contain CutOut.
    return (
        (augs.AutoContrast(), 0, 1),
        (augs.Equalize(), 0, 1),
        (augs.Invert(), 0, 1),
        (augs.Rotate(), 0, 30),
        (augs.Posterize(), 0, 4),
        (augs.Solarize(), 0, 256),
        (augs.SolarizeAdd(), 0, 110),
        (augs.Color(), 0.1, 1.9),
        (augs.Contrast(), 0.1, 1.9),
        (augs.Brightness(), 0.1, 1.9),
        (augs.Sharpness(), 0.1, 1.9),
        (augs.ShearX(), 0., 0.3),
        (augs.ShearY(), 0., 0.3),
        (augs.TranslateXAbs(), 0., 10),
        (augs.TranslateYAbs(), 0., 10),
    )


class RandAugment:

    def __init__(self,
                 n: int,
                 m: int,
                 max_magnitude_level: int = 30,
                 augmentation_list: Optional[Iterable[Tuple[augs.AugmentationOp,
                                                            float,
                                                            float]]] = None):
        if augmentation_list is None:
            augmentation_list = _default_augmentation_list()

        self.n = n
        self.m = m
        self.max_magnitude_level = max_magnitude_level
        self.augmentation_list = (*augmentation_list,)

    def __call__(self, img: PILImage) -> PILImage:
        ops = random.choices(self.augmentation_list, k=self.n)
        print(ops)
        for op, min_magnitude, max_magnitude in ops:  # pylint: disable=invalid-name
            m = self.m / self.max_magnitude_level
            m = (max_magnitude - min_magnitude) * m + min_magnitude
            img = op(img, m)

        return img
