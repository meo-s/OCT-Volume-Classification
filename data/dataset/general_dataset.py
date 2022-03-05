from __future__ import absolute_import

from typing import Any, Callable, Iterable, List, Tuple, Union

import torch
import torch.utils.data.dataset


class GeneralDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, data_samples: Iterable[Any],
                 pipeline: Iterable[Callable[[Any], Any]]):
        super().__init__()

        self.data_samples: Tuple[Any] = (*data_samples, )
        self.pipeline: List[Callable[[Any], Any]] = [*pipeline]

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Any:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_sample = self.data_samples[idx]
        for processor in self.pipeline:
            data_sample = processor(data_sample)

        return data_sample

    def __len__(self):
        return len(self.data_samples)
