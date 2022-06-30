#!/usr/bin/env python3

import numpy as np
from pydantic import validate_arguments, confloat
from scipy import signal, fft

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def derivative_sharpening(old_spe: Spectrum,
                          new_spe: Spectrum, /,
                          filter_fraction: confloat(gt=0, le=1) = .6,  # type: ignore
                          sig_width: confloat(ge=0) = .25,  # type: ignore
                          der2_factor: float = 1,
                          der4_factor: float = .1
                          ):
    Y = fft.rfft(old_spe.y)
    h = signal.windows.hann(int(len(Y)*filter_fraction))
    h = h[(len(h))//2-1:]
    h = np.concatenate((h, np.zeros(len(Y)-len(h))))
    der = np.arange(len(Y))
    der = 1j*np.pi*der/der[-1]
    Y *= h
    Y2 = Y*der**2
    Y4 = Y2*der**2
    y0 = fft.irfft(Y)
    y2 = fft.irfft(Y2)
    y4 = fft.irfft(Y4)
    new_spe.y = y0 - y2/sig_width**2*der2_factor + y4/sig_width**4*der4_factor
