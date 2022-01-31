import math
from collections import Counter
from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np
import sklearn.model_selection
import sklearn.utils


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


class UniformTestSpliter(sklearn.model_selection.BaseCrossValidator):
    """Uniform test samples splitter over classes."""

    def __init__(self,
                 n_splits: int,
                 *,
                 test_size: Optional[Union[float, int]] = None,
                 shuffle: bool = True,
                 random_state: Optional[Union[int,
                                              'np.random.RandomState']] = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(
        self,
        X: Optional[Tuple[Any, ...]] = None,
        y: Optional[Tuple[Any, ...]] = None,
        groups: Any = None,
    ) -> Iterator:
        y = y if y is not None else X

        rng = sklearn.utils.check_random_state(self.random_state)

        classes = np.unique(y)
        class_indices = {}
        for class_ in classes:
            indices, = np.nonzero(y == class_)
            if self.shuffle:
                indices = indices[rng.permutation(len(indices))]
            class_indices[class_] = indices

        # Check whether dataset is fitted for n-cross split.
        total_test_size = self.test_size * self.n_splits
        _validate_uniform_test_split_size(test_size=total_test_size, label=y)

        _, n_test = _validate_uniform_test_split_size(test_size=self.test_size,
                                                      label=y)
        spc = n_test // len(classes)
        for i in range(self.n_splits):
            test_indices = []
            for class_ in classes:
                indices = class_indices[class_]
                indices = indices[spc * i:spc * (i + 1)]
                test_indices.append(indices)

            yield np.concatenate(test_indices)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns self.n_splits."""

        return self.n_splits


def train_test_split(
    *arrays: Tuple[List[Any], ...],
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    random_state: Optional[Union[int, 'np.random.RandomState']] = None,
    shuffle: bool = True,
    stratify: Optional[List[Any]] = None,
    uniform: Optional[List[Any]] = None,
):
    """
    A simple extension of scikit-learn's train_test_split(). If uniform
    argument is not None, it splits dataset to train and test dataset, of which
    test dataset has uniform distribution over data sample's class.

    If uniform argument is None, it calls scikit-learn's train_test_split().
    Please see below document:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    if stratify is not None and uniform is not None:
        raise ValueError('')

    if uniform is None:
        return sklearn.model_selection.train_test_split(
            *arrays,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

    arrays = [np.array(arr) if not isinstance(arr, np.ndarray) else arr
              for arr in arrays]

    n_train, n_test = _validate_uniform_test_split_size(uniform,
                                                        test_size=test_size,
                                                        train_size=train_size)

    uniform_test_spliter = UniformTestSpliter(n_splits=1,
                                              test_size=n_test,
                                              shuffle=shuffle,
                                              random_state=random_state)
    # pylint: disable=no-value-for-parameter
    train_indices, test_indices = uniform_test_spliter.split(*arrays).__next__()
    if len(arrays) == 1:
        x, = arrays
        x_train, x_test = x[train_indices], x[test_indices]
        return x_train[:n_train], x_test
    else:
        x, y = arrays
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return x_train[:n_train], x_test, y_train[:n_train], y_test
