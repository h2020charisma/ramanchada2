#!/usr/bin/env python3

from __future__ import annotations

from ramanchada.misc.spectrum_algorithm import spectrum_algorithm_deco
from ..spectrum import Spectrum


@spectrum_algorithm_deco
def baseline_moving_min(old_spe: Spectrum, new_spe: Spectrum, window_size: int):
    """
    Moving minimum baseline estimator.
    Successive values are calculated as minima of rolling rectangular window.

    Parameters:
    -----------
        window_size : int
    """
    # baseline = [min(old_spe.y[i:min(i+window_size, len(old_spe.y))])
    #             for i in range(len(old_spe.y))
    #             ]
    # A NumericalBaseline to be created with `baseline`
    # and to be added to the SpectralComponentCollection of new_spe
