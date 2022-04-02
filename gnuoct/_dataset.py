from __future__ import absolute_import

from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple, Union

import PIL
import torch
import torch.utils.data
from PIL.Image import Image as PILImage


class GNUOCTVolume(torch.utils.data.Dataset):

    CLASS = Union[Literal['AMD'], Literal['DME'], Literal['DRUSEN'],
                  Literal['NORMAL']]
    CLASSES: Tuple[CLASS] = ('AMD', 'DME', 'DRUSEN', 'NORMAL')

    def __init__(
        self,
        data_samples: Iterable[Tuple[str, CLASS]],
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
        bscans = GNUOCTVolume.load(path)
        if self.transform is not None:
            for i, bscan in enumerate(bscans):
                bscans[i] = self.transform(bscan)

        return bscans, GNUOCTVolume.CLASSES.index(label)

    @staticmethod
    def load(file_path: str) -> List[PILImage]:
        chunk = PIL.Image.open(file_path)
        w, h = chunk.size
        return [chunk.crop((h * i, 0, h * (i + 1), h)) for i in range(w // h)]
