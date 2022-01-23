import math
from collections import Counter
from typing import Any, Optional, Tuple, Union

import numpy as np


def _validate_uniform_test_split_size(
    label: Tuple[Any, ...],
    test_size: Optional[Union[int, float]] = None,
    train_size: Optional[Union[int, float]] = None,
) -> Tuple[int, int]:
    """Validates given split size and returns verified train/test split size.

    Args:
        label:
            An array of sample's label.

        test_size:
            A desired testset size. The calculated testset size is not always
            same as it, but very close.
        train_size:
            A desired trainset size. The calculated trainset size is not always
            same as it, but very close.

    Returns:
        A pair of calculated trainset and testset size.
    """

    n_samples = len(label)
    n_classes = len(np.unique(label))
    n_test = test_size
    n_train = train_size

    if test_size is None and train_size is None:
        raise ValueError('At least one of test_size and train_size argument '
                         'must be real value.')

    if isinstance(test_size, float):
        if test_size < 0 or 1 < test_size:
            raise ValueError('The value of test_size argument must be between '
                             '0 and 1 when its type is float.')
        n_test = min(math.ceil(n_samples * test_size), n_samples)

    if isinstance(train_size, float):
        if train_size < 0 or 1 < train_size:
            raise ValueError('The value of train_size argument must be between '
                             '0 and 1 when its type is float.')
        n_train = min(math.ceil(n_samples * train_size), n_samples)

    if test_size is None:
        n_test = n_samples - n_train
        n_test -= n_test % n_classes

    n_test -= n_test % n_classes  # Make n_test always multiples of n_classes.

    if train_size is None:
        n_train = n_samples - n_test

    spc = [*dict(Counter(label)).values()]
    if min(spc) < (n_test // n_classes):
        raise ValueError('Number of samples per class is not enough to split.')

    if n_samples < (n_train + n_test):
        raise ValueError('Samples are not enough to split.\n'
                         'test_size={}, train_size={} were given.'.format(
                             test_size, train_size))

    return n_train, n_test
