from typing import Dict, Literal, Union

import yaml

HyperParameters = Dict[Union[Literal['aug.rand_augment.use'],
                             Literal['aug.rand_augment.N'],
                             Literal['aug.rand_augment.M']], Union[float, int,
                                                                  str]]


def load(file_path: str) -> HyperParameters:
    with open(file_path, mode='r', encoding='utf-8') as f:  # pylint: disable=invalid-name
        return yaml.load(f, Loader=yaml.CLoader)
