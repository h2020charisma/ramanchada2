from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import PositiveInt, validate_call
from scipy import fft, signal
from scipy.interpolate import (Akima1DInterpolator, CubicSpline,
                               PchipInterpolator)

from ramanchada2.misc.spectrum_deco import (add_spectrum_filter,
                                            add_spectrum_method)

from ..spectrum import Spectrum


@add_spectrum_method
@validate_call(config=dict(arbitrary_types_allowed=True))
def resample_NUDFT(spe: Spectrum, /,
                   x_range: Tuple[float, float] = (0, 4000),
                   xnew_bins: PositiveInt = 100,
                   window: Optional[Union[Callable,
                                          Tuple[Any, ...],  # e.g. ('gaussian', sigma)
                                          Literal['barthann', 'bartlett', 'blackman', 'blackmanharris',
                                                  'bohman', 'boxcar', 'chebwin', 'cosine', 'dpss',
                                                  'exponential', 'flattop', 'gaussian', 'general_cosine',
                                                  'general_gaussian', 'general_hamming', 'hamming', 'hann',
                                                  'kaiser', 'nuttall', 'parzen', 'taylor', 'triang', 'tukey']
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
        h = (window(len(Y_new)*2))[len(Y_new):]  # type: ignore
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


@add_spectrum_method
@validate_call(config=dict(arbitrary_types_allowed=True))
def resample_spline(spe: Spectrum, /,
                    x_range: Tuple[float, float] = (0, 4000),
                    xnew_bins: PositiveInt = 100,
                    spline: Literal['pchip', 'akima', 'makima', 'cubic_spline'] = 'pchip',
                    interp_kw_args: Optional[Dict] = None,
                    cumulative: bool = False):

    kw_args: Dict[str, Any] = {}
    if spline == 'pchip':
        spline_fn = PchipInterpolator
    elif spline == 'akima':
        spline_fn = Akima1DInterpolator
    elif spline == 'makima':
        spline_fn = Akima1DInterpolator
        kw_args['method'] = 'makima'
    elif spline == 'cubic_spline':
        spline_fn = CubicSpline
        kw_args['bc_type'] = 'natural'

    if interp_kw_args is not None:
        kw_args.update(interp_kw_args)

    x_new = np.linspace(x_range[0], x_range[1], xnew_bins, endpoint=False)
    y_new = spline_fn(spe.x, spe.y, **kw_args)(x_new)

    y_new[np.isnan(y_new)] = 0
    if cumulative:
        y_new = np.cumsum(y_new)
        y_new /= y_new[-1]

    return x_new, y_new


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def resample_spline_filter(old_spe: Spectrum,
                           new_spe: Spectrum, /,
                           x_range: Tuple[float, float] = (0, 4000),
                           xnew_bins: PositiveInt = 100,
                           spline: Literal['pchip', 'akima', 'makima', 'cubic_spline'] = 'pchip',
                           interp_kw_args: Optional[Dict] = None,
                           cumulative: bool = False):
    new_spe.x, new_spe.y = resample_spline(old_spe,
                                           x_range=x_range,
                                           xnew_bins=xnew_bins,
                                           spline=spline,
                                           interp_kw_args=interp_kw_args,
                                           cumulative=cumulative)
