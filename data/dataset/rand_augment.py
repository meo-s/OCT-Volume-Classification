from __future__ import absolute_import

import random
from typing import Callable, Tuple

from PIL.Image import Image as PILImage

from data.dataset import augmentations as augs

AugmentationOp = Callable[[PILImage, float], PILImage]

# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py


def _default_augmentation_list() -> Tuple[AugmentationOp]:
    # not contain CutOut.
    return (
        (augs.AutoContrast(), 0, 1),
        (augs.Equalize(), 0, 1),
        (augs.Invert(), 0, 1),
        (augs.Rotate(), 0, 30),
        # (augs.Posterize(), 0, 4),
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
                 gen_augmentation_list: Callable[
                     [], Tuple[Tuple[AugmentationOp, float,
                                     float]]] = _default_augmentation_list):
        self.n = n
        self.m = m
        self.max_magnitude_level = max_magnitude_level
        self.augmentation_list = gen_augmentation_list()

    def __call__(self, img: PILImage) -> PILImage:
        ops = random.choices(self.augmentation_list, k=self.n)
        print(ops)
        for op, min_magnitude, max_magnitude in ops:  # pylint: disable=invalid-name
            m = float(self.m) / self.max_magnitude_level
            m = float(max_magnitude - min_magnitude) * m + min_magnitude
            img = op(img, m)

        return img
