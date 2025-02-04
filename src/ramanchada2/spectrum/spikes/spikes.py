from typing import Literal

import numpy as np
from pydantic import validate_call
from scipy import interpolate

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


@validate_call(config=dict(arbitrary_types_allowed=True))
def spikes_metric(y, /,
                  method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                  ):
    return METHODS[method].metric(y)


@validate_call(config=dict(arbitrary_types_allowed=True))
def spikes_indices(y, /,
                   method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                   **kwargs):
    return METHODS[method].indices(y, **kwargs)


@validate_call(config=dict(arbitrary_types_allowed=True))
def spikes_drop(x0, y0, /,
                method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                **kwargs):
    idx = METHODS[method].indices(y0, **kwargs)
    x = np.delete(x0, idx)
    y = np.delete(y0, idx)
    return x, y


@validate_call(config=dict(arbitrary_types_allowed=True))
def spikes_only(x0, y0, /,
                method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                **kwargs):
    idx = METHODS[method].indices(y0, **kwargs)
    x = np.delete(x0, idx)
    y = np.delete(y0, idx)
    y = y0 - np.interp(x0, x, y)
    return x0, y


@validate_call(config=dict(arbitrary_types_allowed=True))
def add_spike(x, y0, /,
              location: float,
              amplitude: float,
              ):
    idx1 = np.argmin(np.abs(x - location))
    idx2 = idx1 + int(np.sign(location - x[idx1]))
    if idx1 == idx2:
        amp1 = amplitude
        amp2 = 0
    else:
        amp1 = amplitude * np.abs((location - x[idx2])/(x[idx1]-x[idx2]))
        amp2 = amplitude * np.abs((location - x[idx1])/(x[idx1]-x[idx2]))
    y = y0.copy()
    y[idx1] += amp1
    y[idx2] += amp2
    return y


@validate_call(config=dict(arbitrary_types_allowed=True))
def spikes_fix_interp(x0, y, /,
                      method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                      kind='makima',
                      **kwargs):
    idx = spikes_indices(y, method=method, **kwargs)
    x = np.delete(x0, idx)
    y = np.delete(y, idx)
    if kind == 'pchip':
        spline = interpolate.PchipInterpolator(x, y)
    elif kind == 'akima':
        spline = interpolate.Akima1DInterpolator(x, y)
    elif kind == 'makima':
        spline = interpolate.Akima1DInterpolator(x, y, method=kind)
    else:
        spline = interpolate.interp1d(x, y, kind=kind)
    return spline(x0)
