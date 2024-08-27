from typing import Any, Callable, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import PositiveInt, validate_call
from scipy import fft, signal

from ramanchada2.misc.spectrum_deco import (add_spectrum_filter,
                                            add_spectrum_method)

from ..spectrum import Spectrum


@add_spectrum_method
@validate_call(config=dict(arbitrary_types_allowed=True))
def resample_NUDFT(spe: Spectrum, /,
                   x_range: Tuple[float, float] = (0, 4000),
                   xnew_bins: PositiveInt = 100,
                   window: Optional[Union[Callable,  # type: ignore[valid-type]
                                          Tuple[Any, ...],
                                          Literal[*dir(signal.windows.windows)]
                                          ]] = None,
                   cumulative: bool = False):

    x_new = np.linspace(x_range[0], x_range[1], xnew_bins, endpoint=False)
    x = spe.x
    y = spe.y
    x = np.array(x)
    x_range = (np.min(x_range), np.max(x_range))
    y = y[(x >= x_range[0]) & (x < x_range[1])]
    x = x[(x >= x_range[0]) & (x < x_range[1])]

    w = (x-x_range[0])/(x_range[1]-x_range[0])*np.pi*2
    x -= x_range[0]

    k = np.arange(xnew_bins)

    Y_new = np.sum([yi*np.exp(-1j*wi*k) for yi, wi in zip(y, w)], axis=0)

    if window is None:
        window = 'blackmanharris'

    if hasattr(window, '__call__'):
        h = (window(len(Y_new)*2))[len(Y_new):]
    else:
        h = signal.windows.get_window(window, len(Y_new)*2)[len(Y_new):]
    Y_new *= h

    y_new = fft.irfft(Y_new, n=xnew_bins)
    if cumulative:
        y_new = np.cumsum(y_new)
        y_new /= y_new[-1]
    return x_new, y_new


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def resample_NUDFT_filter(old_spe: Spectrum,
                          new_spe: Spectrum, /,
                          x_range: Tuple[float, float] = (0, 4000),
                          xnew_bins: PositiveInt = 100,
                          window=None,
                          cumulative: bool = False):
    new_spe.x, new_spe.y = resample_NUDFT(old_spe,
                                          x_range=x_range,
                                          xnew_bins=xnew_bins,
                                          window=window,
                                          cumulative=cumulative)
