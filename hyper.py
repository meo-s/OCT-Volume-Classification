from typing import Dict, Literal, Union

import yaml

HyperParameterNames = Union[Literal['model'],
                            Literal['n_epochs'],
                            Literal['sz_batch'],
                            Literal['optimizer'],
                            Literal['SGD.base_lr'],
                            Literal['SGD.momentum'],
                            Literal['aug.rand_augment.use'],
                            Literal['aug.rand_augment.N'],
                            Literal['aug.rand_augment.M']]
HyperParameters = Dict[HyperParameterNames, Union[float, int, str]]


def load(file_path: str) -> HyperParameters:
    with open(file_path, mode='r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.CLoader)
