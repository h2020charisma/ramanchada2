#!/usr/bin/env python3
from typing import Dict, List, Union
import lmfit


import numpy as np
from pydantic import validate_arguments

from ramanchada2.misc.spectrum_deco import add_spectrum_filter, add_spectrum_method
from ..spectrum import Spectrum


class DeltaSpeModel:
    def __init__(self, deltas: Dict[float, float], shift=0, scale=1):
        components = list()
        self.params = lmfit.Parameters()
        self.minx = np.min(list(deltas.keys()))
        self.maxx = np.max(list(deltas.keys()))
        self.params.add('shift', value=shift, vary=True)
        self.params.add('scale', value=scale, vary=True, min=.1, max=10)
        self.params.add('scale2', value=0, vary=False, min=-.1, max=.1)
        self.params.add('scale3', value=0, vary=False, min=-1e-3, max=1e-3)
        self.params.add('sigma', value=1, vary=True)
        self.params.add('gain', value=1, vary=True)
        for comp_i, (k, v) in enumerate(deltas.items()):
            prefix = f'comp_{comp_i}'
            components.append(lmfit.models.GaussianModel(prefix=f'comp_{comp_i}_'))
            self.params.add(prefix + '_center', expr=f'shift + {k}*scale + {k}**2*scale2 + {k}**3*scale3', vary=False)
            self.params.add(prefix + '_amplitude', expr=f'{v}*gain', vary=False)
            self.params.add(prefix + '_sigma', expr='sigma', vary=False)
        self.model = np.sum(components)

    def fit(self, spe, sigma, ax=None, no_fit=False):
        self.params['sigma'].set(value=sigma if sigma > 1 else 1)
        spe_conv = spe.convolve('gaussian', sigma=sigma)
        if no_fit:
            fit_res = self.model.fit(spe_conv.y, x=spe_conv.x, params=self.params, max_nfev=-1)
        else:
            fit_res = self.model.fit(spe_conv.y, x=spe_conv.x, params=self.params)
        self.params = fit_res.params
        if ax is not None:
            spe_conv.plot(ax=ax)
            ax.plot(spe_conv.x, fit_res.eval(x=spe_conv.x), 'r')


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def calibrate_by_deltas_model(spe: Spectrum, /,
                              deltas: Dict[float, float], convolution_steps: Union[None, List[float]] = [15, 1],
                              scale2=True, scale3=False, ax=None, **kwargs
                              ):
    spe_padded = spe.pad_zeros()  # type: ignore
    xl, *_, xr = np.nonzero(spe_padded.y)[0]
    y1 = spe_padded.x[xl]
    y2 = spe_padded.x[xr]
    x1 = np.min(list(deltas.keys()))
    x2 = np.max(list(deltas.keys()))
    xx = x2 - x1
    x1 -= xx * .15
    x2 += xx * .15
    scale = (y1-y2)/(x1-x2)
    shift = -scale * x1 + y1
    mod = DeltaSpeModel(deltas, scale=scale, shift=shift)
    if ax is not None:
        spe_padded.plot(ax=ax)

    if convolution_steps is not None:
        for sig in convolution_steps:
            mod.fit(spe=spe_padded, sigma=sig, ax=ax, **kwargs)

    if scale2:
        mod.params['scale2'].set(vary=True, value=0)
        mod.fit(spe_padded, sigma=1, ax=ax, **kwargs)
        mod.fit(spe_padded, sigma=0, ax=ax, **kwargs)
    if scale3:
        mod.params['scale2'].set(vary=True, value=0)
        mod.params['scale3'].set(vary=True, value=0)
        mod.fit(spe_padded, sigma=1, ax=ax, **kwargs)
        mod.fit(spe_padded, sigma=0, ax=ax, **kwargs)
    return mod.model, mod.params


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def calibrate_by_deltas_filter(old_spe: Spectrum,
                               new_spe: Spectrum, /,
                               deltas: Dict[float, float], convolution_steps: List[float] = [15, 1, 0],
                               **kwargs
                               ):
    mod, par = old_spe.calibrate_by_deltas_model(  # type: ignore
        deltas=deltas,
        convolution_steps=convolution_steps,
        **kwargs)
    tmp_spe = old_spe.scale_xaxis_fun(lambda x: (x - par['shift'].value)/par['scale'].value)  # type: ignore
    # par['scale2'].value*x**2 + par['scale3'].value*x**3)
    new_spe.x = tmp_spe.x
