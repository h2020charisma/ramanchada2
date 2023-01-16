#!/usr/bin/env python3

from typing import Literal
import numpy as np
from pydantic import validate_arguments
import statsmodels.api as sm
from scipy.signal import wiener, savgol_filter, medfilt
from scipy.signal.windows import boxcar
from scipy.ndimage import gaussian_filter1d

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def smoothing_RC1(old_spe: Spectrum,
                  new_spe: Spectrum, /, *args,
                  method: Literal['savgol', 'sg',
                                  'wiener',
                                  'median',
                                  'gauss',
                                  'lowess',
                                  'boxcar'],
                  **kwargs):
    if method == 'savgol' or method == 'sg':
        new_spe.y = savgol_filter(old_spe.y, **kwargs)  # window_length, polyorder
    elif method == 'wiener':
        new_spe.y = wiener(old_spe.y, **kwargs)
    elif method == 'gaussian':
        new_spe.y = gaussian_filter1d(old_spe.y, **kwargs)  # sigma
    elif method == 'median':
        new_spe.y = medfilt(old_spe.y, **kwargs)
    elif method == 'lowess':
        kw = dict(span=11)
        kw.update(kwargs)
        x = np.linspace(0, 1, len(old_spe.y))
        new_spe.y = sm.nonparametric.lowess(old_spe.y, x, frac=(5*kw['span'] / len(old_spe.y)), return_sorted=False)
    elif method == 'boxcar':
        kw = dict(box_pts=11)
        kw.update(kwargs)
        box = boxcar(**kwargs, sym=True)
        new_spe.y = np.convolve(old_spe.y, box, mode='same')
