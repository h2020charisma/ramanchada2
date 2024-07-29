from typing import Literal

import numpy as np
from pydantic import validate_arguments
from scipy import interpolate

from ramanchada2.misc.spectrum_deco import (add_spectrum_filter,
                                            add_spectrum_method)

from ..spectrum import Spectrum
from .algos import (double_spike, first_derivative, laplacian, lin_reg_extrap,
                    lr_n2o1, lr_n2o1_n2o2_and, lr_n2o1_n2o2_and_extend,
                    lr_n2o1_n2o2_extend, lr_n2o1_n2o2_mix,
                    lr_n2o1_n2o2_mix_extend, lr_n2o2, lr_n2o2_n3o1, lr_n3o1,
                    lr_n3o2, mod_z_scores, promwidth_Nspike, single_spike,
                    single_spike_double_spike_extend, single_spike_n2o2_extend)

METHODS = {
    'first_derivative': first_derivative,
    'single_spike': single_spike,
    'single_spike_double_spike_extend': single_spike_double_spike_extend,
    'single_spike_n2o2_extend': single_spike_n2o2_extend,
    'double_spike': double_spike,
    'lin_reg_extrap': lin_reg_extrap,
    'lr_n2o1': lr_n2o1,
    'lr_n2o1_n2o2_and_extend': lr_n2o1_n2o2_and_extend,
    'lr_n2o1_n2o2_and': lr_n2o1_n2o2_and,
    'lr_n2o1_n2o2_extend': lr_n2o1_n2o2_extend,
    'lr_n2o1_n2o2_mix_extend': lr_n2o1_n2o2_mix_extend,
    'lr_n2o1_n2o2_mix': lr_n2o1_n2o2_mix,
    'lr_n2o2': lr_n2o2,
    'lr_n2o2_n3o1': lr_n2o2_n3o1,
    'lr_n3o1': lr_n3o1,
    'lr_n3o2': lr_n3o2,
    'laplacian': laplacian,
    'mod_z_scores': mod_z_scores,
    'promwidth_Nspike': promwidth_Nspike,
}


# compatability names
METHODS['gg_1spike'] = METHODS['single_spike']
METHODS['gg_1spike_2spike_extend'] = METHODS['single_spike_double_spike_extend']
METHODS['gg_1spike_n2o2_extend'] = METHODS['single_spike_n2o2_extend']
METHODS['gg_2spike'] = METHODS['double_spike']
METHODS['gg_lin_reg_extrap'] = METHODS['lin_reg_extrap']
METHODS['gg_lr_n2o1'] = METHODS['lr_n2o1']
METHODS['gg_lr_n2o1_n2o2_and_extend'] = METHODS['lr_n2o1_n2o2_and_extend']
METHODS['gg_lr_n2o1_n2o2_and'] = METHODS['lr_n2o1_n2o2_and']
METHODS['gg_lr_n2o1_n2o2_extend'] = METHODS['lr_n2o1_n2o2_extend']
METHODS['gg_lr_n2o1_n2o2_mix_extend'] = METHODS['lr_n2o1_n2o2_mix_extend']
METHODS['gg_lr_n2o1_n2o2_mix'] = METHODS['lr_n2o1_n2o2_mix']
METHODS['gg_lr_n2o2'] = METHODS['lr_n2o2']
METHODS['gg_lr_n2o2_n3o1'] = METHODS['lr_n2o2_n3o1']
METHODS['gg_lr_n3o1'] = METHODS['lr_n3o1']
METHODS['gg_lr_n3o2'] = METHODS['lr_n3o2']
METHODS['ncl_promwidth_Nspike'] = METHODS['promwidth_Nspike']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def calc_spikes_metric(y, /,
                       method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                       ):
    return METHODS[method].metric(y)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def calc_spikes_fix_interp(x0, y, /,
                           method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                           kind='linear',
                           **kwargs):
    idx = calc_spikes_indices(y, method=method, **kwargs)
    x = np.delete(x0, idx)
    y = np.delete(y, idx)
    return interpolate.interp1d(x, y, kind=kind)(x0)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def calc_spikes_indices(y, /,
                        method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                        **kwargs):
    return METHODS[method].indices(y, **kwargs)


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
                   **kwargs):
    return METHODS[method].indices(spe.y, **kwargs)


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_drop(old_spe: Spectrum,
                new_spe: Spectrum, /,
                method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                **kwargs):
    idx = METHODS[method].indices(old_spe.y, **kwargs)
    new_spe.x = np.delete(old_spe.x, idx)
    new_spe.y = np.delete(old_spe.y, idx)


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_fix_interp(old_spe: Spectrum,
                      new_spe: Spectrum, /,
                      method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                      kind='linear',
                      **kwargs):
    idx = METHODS[method].indices(old_spe.y, **kwargs)
    x = np.delete(old_spe.x, idx)
    y = np.delete(old_spe.y, idx)
    new_spe.y = interpolate.interp1d(x, y, kind=kind)(old_spe.x)


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_multi_spike_fix(old_spe: Spectrum,
                           new_spe: Spectrum, /,
                           method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                           moving_window=10,
                           interp_type='linear',
                           **kwargs):
    y_out = old_spe.y.copy()
    idx = METHODS[method].indices(old_spe.y, **kwargs)
    for i, spike in enumerate(idx):
        if spike != 0:  # If we have an spike in position i
            # we select 2 * moving_window + 1 points around our spike
            window = np.arange(i - moving_window, i + moving_window + 1)
            # From such interval, we choose the ones which are not spikes
            window_exclude_spikes = window[idx[window] == 0]
            if interp_type == 'linear':
                interpolator = interpolate.interp1d(window_exclude_spikes,
                                                    y_out[window_exclude_spikes], kind='linear')
            if interp_type == 'quadratic':
                interpolator = interpolate.interp1d(window_exclude_spikes,
                                                    y_out[window_exclude_spikes], kind='quadratic')
            if interp_type == 'cubic':
                interpolator = interpolate.interp1d(window_exclude_spikes,
                                                    y_out[window_exclude_spikes], kind='cubic')
            # The corrupted point is exchanged by the interpolated value.
            y_out[i] = interpolator(i)
    new_spe.y = y_out


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_only(old_spe: Spectrum,
                new_spe: Spectrum, /,
                method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                **kwargs):
    idx = METHODS[method].indices(old_spe.y, **kwargs)
    x = np.delete(old_spe.x, idx)
    y = np.delete(old_spe.y, idx)
    new_spe.y = old_spe.y - np.interp(old_spe.x, x, y)


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def add_spike(old_spe: Spectrum,
              new_spe: Spectrum, /,
              location: float,
              values: list[float],
              ):
    idx = np.argmin(np.abs(location - old_spe.x))
    y = old_spe.y.copy()
    y[idx+(-len(values))//2:idx+(len(values))//2] += values
    new_spe.y = y
