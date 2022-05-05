#!/usr/bin/env python3

import numpy as np

from ramanchada2.misc.spectrum_deco import spectrum_algorithm_deco


@spectrum_algorithm_deco
def moving_average(old_spe, new_spe, /, window_size=10):
    """
    Moving average filter.

    Parameters
    ----------
    window_size : int, optional
        by default 10
    """
    y = [np.average(old_spe.y[i:min(i + window_size, len(old_spe.y))])
         for i in range(len(old_spe.y))]
    new_spe.y = np.array(y)
