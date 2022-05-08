#!/usr/bin/env python3

import numpy as np
from pydantic import validate_arguments, PositiveInt
from scipy import signal

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def moving_average(old_spe: Spectrum,
                   new_spe: Spectrum, /,
                   window_size: PositiveInt = 10):
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


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def moving_average_convolve(old_spe: Spectrum,
                            new_spe: Spectrum, /,
                            window_size: PositiveInt = 10):
    new_spe.y = signal.convolve(old_spe.y, np.ones(window_size)/window_size, mode='same')
