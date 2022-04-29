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
    mov_min = [min(old_spe.y[i:min(i+window_size, len(old_spe.y))])
               for i in range(len(old_spe.y))
               ]
    new_spe.y = np.array(mov_min)
