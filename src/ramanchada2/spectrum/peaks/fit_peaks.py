#!/usr/bin/env python3

from typing import Literal
from collections import UserList
import json

import numpy as np
from pydantic import validate_arguments
from lmfit.models import lmfit_models
from lmfit import models

from ramanchada2.misc.spectrum_deco import (add_spectrum_method,
                                            add_spectrum_filter)
from ..spectrum import Spectrum


class FitPeaksResult(UserList):
    def valuesdict(self):
        ret = dict()
        for i in self:
            ret.update(i.params.valuesdict())
        return ret

    @property
    def locations(self):
        return [v for peak in self for k, v in peak.values.items() if '_center' in k]

    def to_json(self):
        mod = [peak.model.dumps() for peak in self]
        par = [peak.params.dumps() for peak in self]
        return json.dumps(dict(models=mod, params=par))


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peaks(
        spe, /,
        model: Literal['Gaussian', 'Lorentzian', 'Moffat',
                       'Voigt', 'PseudoVoigt',
                       ],
        bounds,
        amplitudes=None,
        locations=None,
        widths=None,
        intercept=True,
        slope=False,
        ):
    results = FitPeaksResult()
    if locations is None:
        locations = np.average(bounds, axis=1).astype(int)
    if amplitudes is None:
        amplitudes = spe.y[locations]
    if widths is None:
        widths = np.diff(bounds/3, axis=1).astype(int)
        widths.shape = (-1)
    locations = spe.x[locations]
    bounds = np.array(bounds).astype(int)
    for i, (b, a, x0, w) in enumerate(zip(bounds, amplitudes, locations, widths)):
        x = spe.x[b[0]:b[1]]
        y = spe.y[b[0]:b[1]]
        mod = lmfit_models[model](name=f'p{i}', prefix=f'p{i}_')
        if intercept or slope:
            mod += models.LinearModel(name=f'p{i}_pedestal', prefix=f'p{i}_pedestal_')
        params = mod.make_params()
        if intercept:
            params[f'p{i}_pedestal_intercept'].set(value=np.min(y), min=0)
        else:
            params[f'p{i}_pedestal_intercept'].set(value=0, vary=False, min=0)
        if slope:
            params[f'p{i}_pedestal_slope'].set(value=0)
        else:
            params[f'p{i}_pedestal_slope'].set(value=0, vary=False)
        params[f'p{i}_amplitude'].set(value=a*10, min=0)
        params[f'p{i}_center'].set(value=x0)
        params[f'p{i}_sigma'].set(value=w)
        if model == 'Moffat':
            params[f'p{i}_beta'].set(value=1)
        if model == 'Voigt':
            tmpres = mod.fit(y, params=params, x=x)
            params = tmpres.params
            params[f'p{i}_gamma'].set(vary=True)
        results.append(mod.fit(y, params=params, x=x))
    return results


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peaks_filter(
        old_spe: Spectrum,
        new_spe: Spectrum, /, model):
    find_peaks_res = old_spe.result
    new_spe.result = old_spe.fit_peaks(model, **find_peaks_res).to_json()  # type: ignore
