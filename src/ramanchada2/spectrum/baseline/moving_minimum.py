#!/usr/bin/env python3

import numpy as np

from ramanchada2.misc.spectrum_deco import spectrum_algorithm_deco
from ..spectrum import Spectrum


@spectrum_algorithm_deco
def moving_minimum(
        old_spe: Spectrum,
        new_spe: Spectrum,
        window_size: int):
    """
    Moving minimum baseline estimator.
    Successive values are calculated as minima of rolling rectangular window.

    Parameters:
    -----------
        window_size : int
    """
    arr = old_spe.y
    mov_min_left = [min(arr[max(0, i):min(i+window_size, len(arr))])
                    for i in range(len(arr))
                    ]
    mov_min_right = [min(arr[max(0, i-window_size):min(i, len(arr))])
                     for i in range(1, len(arr)+1)
                     ]
    new_spe.y = np.maximum.reduce([mov_min_left, mov_min_right])
