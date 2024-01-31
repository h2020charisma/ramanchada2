from typing import Literal

import numpy as np
from pydantic import validate_arguments

from ramanchada2.misc.spectrum_deco import (add_spectrum_filter,
                                            add_spectrum_method)

from ..spectrum import Spectrum
from .algos import first_derivative, gg_1spike, gg_2spike

from scipy import interpolate

METHODS = {
    'gg_1spike': gg_1spike,
    'gg_2spike': gg_2spike,
    'first_derivative': first_derivative
    'laplacian': laplacian
    'ncl_promwidth_Nspike': ncl_promwidth_Nspike
}


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_metric(spe: Spectrum, /,
                  method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                  ):
    return METHODS[method].metric(spe.y)


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_indices(spe: Spectrum, /,
                   method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                   threshold=None):
    return METHODS[method].indices(spe.y, threshold=threshold)


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_drop(old_spe: Spectrum,
                new_spe: Spectrum, /,
                method,
                threshold=None):
    idx = METHODS[method].indices(old_spe.y, threshold=threshold)
    new_spe.x = np.delete(old_spe.x, idx)
    new_spe.y = np.delete(old_spe.y, idx)


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_linfix(old_spe: Spectrum,
                  new_spe: Spectrum, /,
                  method,
                  threshold=None):
    idx = METHODS[method].indices(old_spe.y, threshold=threshold)
    x = np.delete(old_spe.x, idx)
    y = np.delete(old_spe.y, idx)
    new_spe.y = np.interp(old_spe.x, x, y)

@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_interpol_fix(old_spe: Spectrum,
                  new_spe: Spectrum, /,
                  method,
                  threshold=None,
                  moving_window=10,
                  interp_type='linear'):
    y_out = old_spe.y.copy()
    idx = METHODS[method].indices(old_spe.y, threshold=threshold)
    for i, spike in enumerate(idx):
        if spike != 0: # If we have an spike in position i
            window = np.arange(i - moving_window, i + moving_window + 1) # we select 2 * moving_window + 1 points around our spike
            window_exclude_spikes = window[idx[window] == 0] # From such interval, we choose the ones which are not spikes
            if interp_type=='linear':
                interpolator = interpolate.interp1d(window_exclude_spikes, y[window_exclude_spikes], kind='linear')
            if interp_type=='quadratic':
                interpolator = interpolate.interp1d(window_exclude_spikes, y[window_exclude_spikes], kind='quadratic')
            if interp_type=='cubic':
                interpolator = interpolate.interp1d(window_exclude_spikes, y[window_exclude_spikes], kind='cubic')
            y_out[i] = interpolator(i) # The corrupted point is exchanged by the interpolated value.
    new_spe.y = y_out



@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_only(old_spe: Spectrum,
                new_spe: Spectrum, /,
                method,
                threshold=None):
    idx = METHODS[method].indices(old_spe.y, threshold=threshold)
    x = np.delete(old_spe.x, idx)
    y = np.delete(old_spe.y, idx)
    new_spe.y = old_spe.y - np.interp(old_spe.x, x, y)
