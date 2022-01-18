import numpy as np


def _uniformly_distribute(n_samples: int, n_sections: int) -> np.ndarray:
    if n_sections <= 0 or n_samples < n_sections:
        raise ValueError('n_sections does not fit to n_samples.'
                         'n_samples=%d, n_sections=%d were given.' %
                         (n_samples, n_sections))

    n_remains = n_samples % n_sections
    distribution = np.full(n_sections, fill_value=n_samples // n_sections)
    distribution[:n_remains] += 1
    return distribution
