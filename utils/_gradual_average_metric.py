import warnings
from typing import Optional


class GradualAverageMetric:

    def __init__(
        self,
        name: str,
        verbose: bool = True,
        total_samples: Optional[int] = None,
    ):
        self.name: str = name
        self.total_samples: Optional[int] = total_samples
        self.verbose = verbose
        self._value: float = 0
        self._cumulated_samples: int = 0

    def update(self,
               step_value: float,
               step_samples: Optional[int] = None) -> float:
        step_samples = step_samples if step_samples is not None else 1

        if self.total_samples is not None:
            if self.total_samples < self._cumulated_samples + step_samples:
                raise ValueError(
                    'Count of samples participated in calculating a metric '
                    'exceeded user-defined total sample count.')

        step_value *= step_samples
        self._value = ((self._value * self._cumulated_samples + step_value) /
                       (self._cumulated_samples + step_samples))
        self._cumulated_samples += step_samples

        return self._value

    def reset(self, total_samples: Optional[int] = None):
        total_samples = (total_samples
                         if total_samples is not None else self.total_samples)
        if self.verbose and self.total_samples is not None:
            if self._cumulated_samples < self.total_samples:
                warnings.warn(
                    'All of samples are not participated in calculating a '
                    f'metric value "{self.name}", but the object was reset.'
                    'This means the metric value before reset was incomplete.',
                    UserWarning)

        self._value = 0
        self._cumulated_samples = 0

    @property
    def value(self) -> float:
        return self._value
