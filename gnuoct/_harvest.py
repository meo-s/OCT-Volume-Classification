from __future__ import absolute_import

import os
from collections import defaultdict
from glob import glob
from typing import Dict

import numpy as np


def harvest(
    dataset_dir: str,
    return_relative_path: bool = True,
) -> Dict[str, np.ndarray]:
    """Collect data samples from file system and return a dictionary which
    saves arrays of collected data.

    Args:
        dataset_dir:
            A path to dataset directory.

        return_relative_path:
            A flag to whether use relative path or absolute path. If it is True,
            result array will only save relative paths of data sample about
            dataset_dir.

    Returns:
        A dictionary of data arrays. A data array could be accessed by using
        x_train, y_train, x_test, y_test, x_val, y_val keys. Accroding to
        directory structure of given dataset_dir argument, the dictionary key
        may be different.
    """
    if dataset_dir.endswith('/'):
        dataset_dir = dataset_dir[:-1]
    if not os.path.exists(dataset_dir):
        raise OSError('Given dataset directory does not exist: %s' %
                      dataset_dir)

    data = defaultdict(list)
    for data_sample_path in glob(os.path.join(dataset_dir, '**', '*.*'),
                                 recursive=True):
        data_sample_path = data_sample_path[len(dataset_dir) + 1:]
        root_dir, file_name = os.path.split(data_sample_path)
        if not file_name.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
            continue

        *_, dtype, label = root_dir.split('/')
        dtype, label = dtype.lower(), label.upper()
        data_sample_path = data_sample_path if return_relative_path else \
            os.path.join(dataset_dir, data_sample_path)
        data[f'x_{dtype}'].append(data_sample_path)
        data[f'y_{dtype}'].append(label)

    return data
