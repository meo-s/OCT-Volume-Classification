from typing import Dict, Literal, Union

import yaml

HyperParameterNames = Literal['model', 'n_epochs', 'sz_batch', 'optimizer',
                              'optimizer.base_lr', 'optimizer.weight_decay',
                              'SGD.momentum', 'aug.rand_augment.use',
                              'aug.rand_augment.N', 'aug.rand_augment.M',
                              'aug.mixup.use', 'aug.mixup.alpha',
                              'label_smoothing']
HyperParameters = Dict[HyperParameterNames, Union[float, int, str]]


def load(file_path: str) -> HyperParameters:
    with open(file_path, mode='r', encoding='utf-8') as f:
        hp = yaml.load(f, Loader=yaml.CLoader)

    for hp_name in hp.keys():
        if hp_name not in HyperParameterNames.__args__:
            raise ValueError(f'Invalid hyperparameter name: "{hp_name}".')

    return hp
