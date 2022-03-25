from __future__ import absolute_import

from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple, Type, Union

import PIL
import torch
import torch.utils.data
import torchvision as tv
from PIL.Image import Image as PILImage

import hyper
from data import augmentations as augs
from data import RandAugment


class GNUOCTVolume(torch.utils.data.Dataset):

    CLASSES = ('AMD', 'DME', 'DRUSEN', 'NORMAL')

    def __init__(
        self,
        data_samples: Iterable[Tuple[str, Type['GNUOCTVolume.CLASSES']]],
        transform: Optional[Callable[[PILImage], Any]] = None,
    ):
        super().__init__()

        self.data_samples = [*data_samples]
        self.transform = transform

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(
        self,
        index: Union[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(index):
            index = index.tolist()

        path, label = self.data_samples[index]
        chunk = GNUOCTVolume.load(path)
        if self.transform is not None:
            for i, bscan in enumerate(chunk):
                chunk[i] = self.transform(bscan)

        return chunk, label

    @staticmethod
    def load(file_path: str) -> List[PILImage]:
        chunk = PIL.Image.open(file_path)
        w, h = chunk.size
        return [chunk.crop((h * i, 0, h * (i + 1), h)) for i in range(w // h)]


def _gnuoct_augmentation_list(
) -> Tuple[Tuple[augs.AugmentationOp, float, float]]:
    return (
        (augs.AutoContrast(), 0, 1),
        (augs.Equalize(), 0, 1),
        (augs.Invert(), 0, 1),
        (augs.Rotate(), 0, 30),
        # (augs.Posterize(), 0, 4),
        (augs.Solarize(), 0, 256),
        (augs.SolarizeAdd(), 0, 110),
        (augs.Color(), 0.1, 1.9),
        (augs.Contrast(), 0.4, 1.6),
        (augs.Brightness(), 0.4, 1.6),
        (augs.Sharpness(), 0.1, 1.9),
        (augs.ShearX(), 0., 0.3),
        (augs.ShearY(), 0., 0.3),
        (augs.TranslateXAbs(), 0., 10),
        (augs.TranslateYAbs(), 0., 10),
        (augs.Identity(), 0, 1),
    )


def get_transform(
    ds_type: Union[Literal['train'], Literal['val'], Literal['test']],
    hp: hyper.HyperParameters,
) -> torch.utils.data.DataLoader:
    if ds_type not in ('train', 'val', 'test'):
        raise ValueError('ds_type must be one of "train", "val" and "test".')

    BICUBIC = tv.transforms.InterpolationMode.BICUBIC  # pylint: disable=invalid-name
    if ds_type == 'train':
        return tv.transforms.Compose([
            RandAugment(n=hp['aug.rand_augment.N'],
                        m=hp['aug.rand_augment.M'],
                        augmentation_list=_gnuoct_augmentation_list()),
            tv.transforms.ColorJitter(saturation=0.4, hue=0.4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomResizedCrop(224,
                                            scale=(0.9, 1.1),
                                            interpolation=BICUBIC),
            tv.transforms.ToTensor(),
        ])
    else:
        return tv.transforms.Compose([
            tv.transforms.Resize(224, interpolation=BICUBIC),
            tv.transforms.ToTensor(),
        ])
