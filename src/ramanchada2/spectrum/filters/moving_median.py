import numpy as np
from pydantic import PositiveInt, validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@validate_call(config=dict(arbitrary_types_allowed=True))
def _moving_median(s,
                   window_size: PositiveInt = 10):
    y = ([np.median(s[:window_size]) for i in range(window_size)] +
         [np.median(s[i-window_size: i+window_size]) for i in range(window_size, len(s)-window_size)] +
         [np.median(s[-window_size:]) for i in range(window_size)]
         )
    return np.array(y)


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def moving_median(old_spe: Spectrum,
                  new_spe: Spectrum, /,
                  window_size: PositiveInt = 10):
    """
    Moving median filter.

    The resultant spectrum is moving minimum of the input.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        window_size:
            `int`, optional, default is `10`.

    Returns: modified Spectrum
    """

    new_spe.y = _moving_median(old_spe.y, window_size)


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def subtract_moving_median(
        old_spe: Spectrum,
        new_spe: Spectrum,
        window_size: int):
    """
    Subtract moving median filter.

    The resultant spectrum is moving minimum of the input subtracted from the input.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        window_size:
            `int`, optional, default is `10`.

    Returns: modified Spectrum
    """
    new_spe.y = old_spe.y - _moving_median(old_spe.y, window_size)
