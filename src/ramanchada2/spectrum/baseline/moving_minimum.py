import numpy as np
from pydantic import validate_call, PositiveInt

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@validate_call(config=dict(arbitrary_types_allowed=True))
def _moving_minimum(arr, window_size: PositiveInt):
    mov_min_left = [min(arr[max(0, i):min(i+window_size, len(arr))])
                    for i in range(len(arr))
                    ]
    mov_min_right = [min(arr[max(0, i-window_size):min(i, len(arr))])
                     for i in range(1, len(arr)+1)
                     ]
    return np.maximum.reduce([mov_min_left, mov_min_right])


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def moving_minimum(
        old_spe: Spectrum,
        new_spe: Spectrum,
        window_size: int):
    """
    Moving minimum baseline estimator.
    Successive values are calculated as minima of rolling rectangular window.
    """
    new_spe.y = _moving_minimum(old_spe.y, window_size)


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def subtract_moving_minimum(
        old_spe: Spectrum,
        new_spe: Spectrum,
        window_size: int):
    new_spe.y = old_spe.y - _moving_minimum(old_spe.y, window_size)
