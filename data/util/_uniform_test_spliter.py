import math
from collections import Counter
from typing import Any, Optional, Tuple, Union

import numpy as np


def _uniformly_distribute(n_samples: int, n_sections: int) -> np.ndarray:
    if n_sections <= 0 or n_samples < n_sections:
        raise ValueError('n_sections does not fit to n_samples.\n'
                         'n_samples=%d, n_sections=%d were given.' %
                         (n_samples, n_sections))

    n_remains = n_samples % n_sections
    distribution = np.full(n_sections, fill_value=n_samples // n_sections)
    distribution[:n_remains] += 1
    return distribution


def _validate_uniform_test_split_size(
    test_size: Optional[Union[int, float]],
    train_size: Optional[Union[int, float]],
    label: Tuple[Any, ...],
) -> Tuple[int, int]:
    """Validates given split size and returns verified train/test split size.

    Returns:
        A pair of calculated trainset and testset size.
    """

    n_samples = len(label)
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
    if train_size is None:
        n_train = n_samples - n_test

    if isinstance(train_size, float):
        if train_size < 0 or 1 < train_size:
            raise ValueError('The value of train_size argument must be between '
                             '0 and 1 when its type is float.')
        n_train = min(math.ceil(n_samples * train_size), n_samples)
    if test_size is None:
        n_test = n_samples - n_train

    if n_samples < (n_train + n_test):
        raise ValueError('Samples are not enough to split.\n'
                         'test_size={}, train_size={} were given.'
                         .format(test_size, train_size))

    spc = [*dict(Counter(label)).values()]
    if min(spc) < max(_uniformly_distribute(n_test, len(np.unique(label)))):
        raise ValueError('Number of samples per class is not enough to split.')

    return n_train, n_test
