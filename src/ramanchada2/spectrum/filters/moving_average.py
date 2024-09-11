import numpy as np
from pydantic import PositiveInt, validate_call
from scipy import signal

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def moving_average(old_spe: Spectrum,
                   new_spe: Spectrum, /,
                   window_size: PositiveInt = 10):
    """
    Moving average filter.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        window_size:
            `int`, optional, default is `10`.

    Returns: modified Spectrum
    """
    y = [np.average(old_spe.y[i:min(i + window_size, len(old_spe.y))])
         for i in range(len(old_spe.y))]
    new_spe.y = np.array(y)


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def moving_average_convolve(old_spe: Spectrum,
                            new_spe: Spectrum, /,
                            window_size: PositiveInt = 10):
    """
    Moving average filter.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        window_size:
            `int`, optional, default is `10`.

    Returns: modified Spectrum
    """
    new_spe.y = signal.convolve(old_spe.y, np.ones(window_size)/window_size, mode='same')
