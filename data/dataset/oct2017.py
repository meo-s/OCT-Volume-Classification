from __future__ import absolute_import

import os
from collections import defaultdict
from glob import glob
from typing import Tuple

import numpy as np

CLASSES: Tuple[str, ...] = ('CNV', 'DME', 'DRUSEN', 'NORMAL')


def harvest(
    dataset_dir: str,
    label_as_int: bool = True,
    use_relative_path: bool = True,
) -> Tuple[np.ndarray, ...]:
    """Returns train and test data array which is two seperate array of
    relative path and label of data respectively.

    Args:
        dataset_dir:
            A path to dataset directory.

        label_as_int:
            A flag to whether change label to integer value. If it is True,
            label is changed to integer value.

        use_relative_path:
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

    dataset = defaultdict(list)
    for data_file_path in glob(os.path.join(dataset_dir, '**', '*.*'),
                               recursive=True):
        data_file_path = data_file_path[len(dataset_dir) + 1:]
        root_dir, file_name = os.path.split(data_file_path)
        if not file_name.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
            continue

        stype, label = root_dir.split('/')
        stype, label = stype.lower(), label.upper()
        label = label if not label_as_int else CLASSES.index(label)
        data_file_path = data_file_path if use_relative_path else \
            os.path.join(dataset_dir, data_file_path)
        dataset[f'x_{stype}'].append(data_file_path)
        dataset[f'y_{stype}'].append(label)

    return [
        np.array(dataset[i])
        for i in ['x_train', 'x_test', 'y_train', 'y_test']
    ]
