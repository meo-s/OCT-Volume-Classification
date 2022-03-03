from __future__ import absolute_import

import random
from abc import ABCMeta
from abc import abstractmethod
from typing import Optional, Tuple

import PIL
import PIL.Image
import PIL.ImageOps
from PIL.Image import Image as PILImage

# https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/augmentations.py
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py


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


class ShearX(_BaseAugmentationOp):

    def __init__(self,
                 magnitude_limit: Tuple[float, float] = (-0.3, 0.3),
                 random_mirror: bool = True):
        super().__init__(magnitude_limit=magnitude_limit,
                         random_mirror=random_mirror)

    def _apply_transformation(self, img: PILImage, m: float) -> PILImage:
        return img.transform(img.size, PIL.Image.AFFINE, (1, m, 0, 0, 1, 0))


class ShearY(_BaseAugmentationOp):

    def __init__(self,
                 magnitude_limit: Tuple[float, float] = (-0.3, 0.3),
                 random_mirror: bool = True):
        super().__init__(magnitude_limit=magnitude_limit,
                         random_mirror=random_mirror)

    def _apply_transformation(self, img: PILImage, m: float) -> PILImage:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, m, 1, 0))


class TranslateX(_BaseAugmentationOp):

    def __init__(self,
                 magnitude_limit: Tuple[float, float] = (-0.45, 0.45),
                 random_mirror: bool = True):
        super().__init__(magnitude_limit=magnitude_limit,
                         random_mirror=random_mirror)

    def _apply_transformation(self, img: PILImage, m: float) -> PILImage:
        m *= img.size[0]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, m, 0, 1, 0))


class TranslateY(_BaseAugmentationOp):

    def __init__(self,
                 magnitude_limit: Tuple[float, float] = (-0.45, 0.45),
                 random_mirror: bool = True):
        super().__init__(magnitude_limit=magnitude_limit,
                         random_mirror=random_mirror)

    def _apply_transformation(self, img: PILImage, m: float) -> PILImage:
        m *= img.size[1]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, m))


class TranslateXAbs(_BaseAugmentationOp):

    def __init__(self,
                 magnitude_limit: Tuple[float, float] = (0, 10),
                 random_mirror: bool = True):
        super().__init__(magnitude_limit=magnitude_limit,
                         random_mirror=random_mirror)

    def _apply_transformation(self, img: PILImage, m: float) -> PILImage:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, m, 0, 1, 0))


class TranslateYAbs(_BaseAugmentationOp):

    def __init__(self,
                 magnitude_limit: Tuple[float, float] = (0, 10),
                 random_mirror: bool = True):
        super().__init__(magnitude_limit=magnitude_limit,
                         random_mirror=random_mirror)

    def _apply_transformation(self, img: PILImage, m: float) -> PILImage:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, m))


class Rotate(_BaseAugmentationOp):

    def __init__(self,
                 magnitude_limit: Tuple[float, float] = (-30, 30),
                 random_mirror: bool = True):
        super().__init__(magnitude_limit=magnitude_limit,
                         random_mirror=random_mirror)

    def _apply_transformation(self, img: PILImage, m: float) -> PILImage:
        return img.rotate(m)


class AutoContrast(_BaseAugmentationOp):

    def _apply_transformation(self, img: PILImage, m: float) -> PILImage:
        return PIL.ImageOps.autocontrast(img)


class Invert(_BaseAugmentationOp):

    def _apply_transformation(self, img: PILImage, m: float) -> PILImage:
        return PIL.ImageOps.invert(img)


class Equalize(_BaseAugmentationOp):

    def _apply_transformation(self, img: PILImage, m: float) -> PILImage:
        return PIL.ImageOps.equalize(img)


class Flip(_BaseAugmentationOp):

    def _apply_transformation(self, img: PILImage, m: float) -> PILImage:
        return PIL.ImageOps.flip(img)


class Solarize(_BaseAugmentationOp):

    def __init__(self, magnitude_limit: Tuple[float, float] = (0, 256)):
        super().__init__(magnitude_limit, random_mirror=None)

    def _apply_transformation(self, img: PILImage, m: float) -> PILImage:
        return PIL.ImageOps.solarize(img, m)

