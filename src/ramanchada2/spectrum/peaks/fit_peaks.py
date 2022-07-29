#!/usr/bin/env python3

from typing import Literal, List, Union
from collections import UserList

import numpy as np
import pandas as pd
from pydantic import validate_arguments, Field
from lmfit.models import lmfit_models, LinearModel
from lmfit.model import ModelResult, Parameters, Model

from ramanchada2.misc.spectrum_deco import (add_spectrum_method,
                                            add_spectrum_filter)
from ramanchada2.misc.types import (PeakCandidatesGroupModel,
                                    ListPeakCandidateGroupsModel)
from ramanchada2.misc.plottable import Plottable
from ..spectrum import Spectrum


class FitPeaksResult(UserList, Plottable):
    def valuesdict(self):
        ret = dict()
        for i in self:
            ret.update(i.params.valuesdict())
        return ret

    @property
    def locations(self):
        return [v for peak in self for k, v in peak.values.items() if k.endswith('center')]

    def dumps(self):
        return [peak.dumps() for peak in self]

    def loads(self, json_str: List[str]):
        self.clear()
        for p in json_str:
            params = Parameters()
            modres = ModelResult(Model(lambda x: x, None), params)
            self.append(modres.loads(p))
        return self

    def _plot(self, ax, peak_candidate_groups, xarr, **kwargs):
        for i, p in enumerate(self):
            left, right = peak_candidate_groups[i].boundaries(n_sigma=3)
            x = xarr[(xarr >= left) & (xarr <= right)]
            ax.plot(x, p.eval(x=x), **kwargs)

    def to_csv(self, path_or_buf=None, sep=',', **kwargs):
        return pd.DataFrame(
            [
                dict(name=f'g{group:02d}_{key}', value=val.value, stderr=val.stderr)
                for group, res in enumerate(self)
                for key, val in res.params.items()
            ]
        ).sort_values('name').to_csv(path_or_buf=path_or_buf, sep=sep, **kwargs)


available_models_type = Literal['Gaussian', 'Lorentzian', 'Moffat', 'Voigt', 'PseudoVoigt']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def build_model_params(spe, model: Union[available_models_type, List[available_models_type]],
                       peak_candidates: PeakCandidatesGroupModel,
                       baseline_model: Literal['linear', None] = None,
                       ):
    mod_list = list()
    if baseline_model == 'linear':
        mod_list.append(LinearModel(name='baseline', prefix='bl_'))
    if isinstance(model, str):
        model = [model] * len(peak_candidates)
    else:
        if len(peak_candidates) != len(model):
            raise Exception(
                f'incompatible lengths len(peak_candidates)={len(peak_candidates)} and len(model)={len(model)}')
    for i, mod in enumerate(model):
        mod_list.append(lmfit_models[mod](name=f'p{i}', prefix=f'p{i}_'))
    fit_model = np.sum(mod_list)

    if baseline_model == 'linear':
        li_bounds = peak_candidates.left_base_idx_range(n_sigma=5, arr_len=len(spe.x))
        ri_bounds = peak_candidates.right_base_idx_range(n_sigma=5, arr_len=len(spe.x))
        li = np.argmin(spe.y[li_bounds[0]:li_bounds[1]]) + li_bounds[0]
        ri = np.argmin(spe.y[ri_bounds[0]:ri_bounds[1]]) + ri_bounds[0]
        xl = spe.x[li]
        yl = spe.y[li]
        xr = spe.x[ri]
        yr = spe.y[ri]
        slope = (yr-yl)/(xr-xl)
        intercept = -xl*slope + yl
    else:
        slope = 1
        intercept = 0

    fit_params = fit_model.make_params()
    if baseline_model == 'linear':
        fit_params['bl_slope'].set(value=slope)
        fit_params['bl_intercept'].set(value=intercept)

    pos_ampl_sigma_base = peak_candidates.pos_ampl_sigma_base_peakidx()
    for i, (mod, (x0, a, w, p, peak_i)) in enumerate(zip(model, pos_ampl_sigma_base)):
        if mod == 'Moffat':
            fwhm_factor = 2.
            height_factor = 1.
        elif mod == 'Voigt':
            fwhm_factor = 3.6013
            height_factor = 1/w/np.sqrt(2)
        elif mod == 'PseudoVoigt':
            fwhm_factor = lmfit_models[mod].fwhm_factor
            height_factor = 1/np.pi/w
        else:
            fwhm_factor = lmfit_models[mod].fwhm_factor
            height_factor = lmfit_models[mod].height_factor

        a = spe.y[peak_i] - (slope*x0 + intercept)
        if a < 0:
            a = spe.y[peak_i]
        fit_params[f'p{i}_amplitude'].set(value=a/height_factor, min=0, max=a/height_factor*20)
        fit_params[f'p{i}_center'].set(value=x0)
        fit_params[f'p{i}_sigma'].set(value=w/fwhm_factor, min=.1e-4, max=w/fwhm_factor*50)

        if mod == 'Moffat':
            fit_params[f'p{i}_beta'].set(value=1, min=1e-4, max=100)
        if mod == 'Voigt':
            fit_params[f'p{i}_gamma'].set(value=w/fwhm_factor, min=.0001, max=w/fwhm_factor*10, vary=True)
    return fit_model, fit_params


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peaks_model(spe: Spectrum, /, *,
                    model: Union[available_models_type, List[available_models_type]],
                    peak_candidates: PeakCandidatesGroupModel,
                    n_sigma_trim: float = Field(5, gt=0),
                    baseline_model: Literal['linear', None] = None,
                    kwargs_fit={}
                    ):
    fit_model, fit_params = build_model_params(spe=spe,
                                               model=model,
                                               peak_candidates=peak_candidates,
                                               baseline_model=baseline_model)
    lb, rb = peak_candidates.boundaries_idx(n_sigma=n_sigma_trim, arr_len=len(spe.x))
    fitx = spe.x[lb:rb]
    fity = spe.y[lb:rb]

    for par in fit_params:
        if par.endswith('_center'):
            fit_params[par].set(min=spe.x[lb], max=spe.x[rb], vary=True)
    fit_tmp = fit_model.fit(fity, params=fit_params, x=fitx, **kwargs_fit)
    return fit_tmp


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peak_groups(spe, /, *,
                    model: Union[available_models_type, List[available_models_type]],
                    peak_candidate_groups: ListPeakCandidateGroupsModel,
                    n_sigma_trim: float = 3,
                    kwargs_fit={}
                    ):
    fit_res = FitPeaksResult()
    for group in peak_candidate_groups.__root__:
        fit_res.append(fit_peaks_model(spe,
                                       peak_candidates=group,
                                       model=model,
                                       baseline_model='linear',
                                       n_sigma_trim=n_sigma_trim,
                                       kwargs_fit=kwargs_fit,
                                       ))
    return fit_res


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peaks(spe, /, *,
              model: Literal['Gaussian', 'Lorentzian', 'Moffat',
                             'Voigt', 'PseudoVoigt',
                             ],
              peak_candidates: PeakCandidatesGroupModel,
              n_sigma_trim: float = 3,
              ):
    fit_res = FitPeaksResult()
    fit_res.append(fit_peaks_model(spe,
                                   peak_candidates=peak_candidates,
                                   model=model,
                                   n_sigma_trim=n_sigma_trim))
    return fit_res


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peaks_filter(
        old_spe: Spectrum,
        new_spe: Spectrum, /, *args,
        kwargs_fit={},
        **kwargs,
        ):
    cand_groups = ListPeakCandidateGroupsModel.validate(old_spe.result)
    new_spe.result = old_spe.fit_peak_groups(*args,  # type: ignore
                                             peak_candidate_groups=cand_groups,
                                             kwargs_fit=kwargs_fit,
                                             **kwargs).dumps()
