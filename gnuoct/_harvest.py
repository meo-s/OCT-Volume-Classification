from __future__ import absolute_import

import os
from collections import defaultdict
from glob import glob
from typing import Tuple

import numpy as np


def harvest(
    dataset_dir: str,
    return_relative_path: bool = True,
) -> Tuple[np.ndarray, ...]:
    """Returns train and test data array which is two seperate array of
    relative path and label of data respectively.

    Args:
        dataset_dir:
            A path to dataset directory.

        return_relative_path:
            A flag to whether use relative path or absolute path. If it is True,
            result array will only save relative paths of data sample about
            dataset_dir.

    Returns:
        Tuple of train and test data arrays. Order is x_train, x_test, y_train,
        y_test.
    """
    if dataset_dir.endswith('/'):
        dataset_dir = dataset_dir[:-1]
    if not os.path.exists(dataset_dir):
        raise OSError('Given dataset directory does not exist: %s' %
                      dataset_dir)

    sets = defaultdict(list)
    for data_sample_path in glob(os.path.join(dataset_dir, '**', '*.*'),
                                 recursive=True):
        data_sample_path = data_sample_path[len(dataset_dir) + 1:]
        root_dir, file_name = os.path.split(data_sample_path)
        if not file_name.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
            continue

        *_, stype, label = root_dir.split('/')
        stype, label = stype.lower(), label.upper()
        data_sample_path = data_sample_path if return_relative_path else \
            os.path.join(dataset_dir, data_sample_path)
        sets[f'x_{stype}'].append(data_sample_path)
        sets[f'y_{stype}'].append(label)

    stypes = ['x_train', 'x_test', 'y_train', 'y_test']
    if 'x_val' in sets and 'y_val' in sets:
        stypes += ['x_val', 'y_val']

    return [np.array(sets[i]) for i in stypes]
