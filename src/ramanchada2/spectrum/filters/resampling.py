#!/usr/bin/env python3

from typing import Tuple
from scipy import signal, fft
import numpy as np

from pydantic import validate_arguments, PositiveInt

from ramanchada2.misc.spectrum_deco import add_spectrum_filter, add_spectrum_method
from ..spectrum import Spectrum


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def resample_NUDFT(spe: Spectrum, /,
                   x_range: Tuple[float, float] = (0, 4000),
                   xnew_bins: PositiveInt = 100,
                   window=signal.windows.hann,
                   cumulative: bool = False):
    x = spe.x
    y = spe.y
    x = np.array(x)
    x = x[(x >= x_range[0]) & (x < x_range[1])]
    x = (x-x_range[0])/(x_range[1]-x_range[0])*xnew_bins
    w = np.linspace(0, np.pi, (xnew_bins)//2+1)
    Y_new = np.sum([yi*np.exp(-1j*w*xi) for xi, yi in zip(x, y)], axis=0)
    Y_new *= (window(len(Y_new)*2))[len(Y_new):]
    y_new = fft.irfft(Y_new, n=xnew_bins)
    x_new = np.linspace(x_range[0], x_range[1], xnew_bins)
    if cumulative:
        y_new = np.cumsum(y_new)
        y_new /= y_new[-1]
    return x_new, y_new


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def resample_NUDFT_filter(old_spe: Spectrum,
                          new_spe: Spectrum, /,
                          x_range: Tuple[float, float] = (0, 4000),
                          xnew_bins: PositiveInt = 100,
                          window=signal.windows.blackmanharris,
                          cumulative: bool = False):
    new_spe.x, new_spe.y = resample_NUDFT(old_spe,
                                          x_range=x_range,
                                          xnew_bins=xnew_bins,
                                          window=window,
                                          cumulative=cumulative)
