from typing import Literal

import numpy as np
import statsmodels.api as sm
from pydantic import validate_call
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt, savgol_filter, wiener
from scipy.signal.windows import boxcar

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def smoothing_RC1(old_spe: Spectrum,
                  new_spe: Spectrum, /, *args,
                  method: Literal['savgol', 'sg',
                                  'wiener',
                                  'median',
                                  'gauss', 'gaussian',
                                  'lowess',
                                  'boxcar'],
                  **kwargs):
    """
    Smooth the spectrum.

    The spectrum will be smoothed using the specified filter.
    This method is inherited from ramanchada1 for compatibility reasons.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        method: method to be used
        **kwargs: keyword arguments to be passed to the selected method

    Returns: modified Spectrum
    """
    if method == 'savgol' or method == 'sg':
        new_spe.y = savgol_filter(old_spe.y, **kwargs)  # window_length, polyorder
    elif method == 'wiener':
        new_spe.y = wiener(old_spe.y, **kwargs)
    elif method == 'gaussian' or method == 'gauss':
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
