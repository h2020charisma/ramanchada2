#!/usr/bin/env python3

from typing import Literal, List, Union
from collections import UserList
import logging

import numpy as np
import pandas as pd
from pydantic import validate_arguments, PositiveFloat
from lmfit.models import lmfit_models, LinearModel
from lmfit.model import ModelResult, Parameters, Model

from ramanchada2.misc.spectrum_deco import (add_spectrum_method,
                                            add_spectrum_filter)
from ramanchada2.misc.types import (PeakCandidatesGroupModel,
                                    ListPeakCandidateGroupsModel)
from ramanchada2.misc.plottable import Plottable
from ramanchada2.misc.types.peak_candidates import ListPeakCandidateMultiModel, PeakCandidateMultiModel

from ..spectrum import Spectrum

logger = logging.getLogger(__name__)


class FitPeaksResult(UserList, Plottable):
    def valuesdict(self):
        ret = dict()
        for i in self:
            ret.update(i.params.valuesdict())
        return ret

    @property
    def locations(self):
        return [v for peak in self for k, v in peak.values.items() if k.endswith('center')]

    @property
    def centers(self):
        return np.array([v for peak in self for k, v in peak.values.items() if k.endswith('center')])

    @property
    def centers_err(self):
        return np.array([
            (v.value, v.stderr)
            for peak in self
            for k, v in peak.params.items()
            if k.endswith('center')
            if hasattr(v, 'stderr') and v.stderr is not None
            ])

    @property
    def fwhms(self):
        return [v for peak in self for k, v in peak.values.items() if k.endswith('fwhm')]

    @property
    def amplitudes(self):
        return [v for peak in self for k, v in peak.values.items() if k.endswith('amplitude')]

    def dumps(self):
        return [peak.dumps() for peak in self]

    def loads(self, json_str: List[str]):
        self.clear()
        for p in json_str:
            params = Parameters()
            modres = ModelResult(Model(lambda x: x, None), params)
            self.append(modres.loads(p))
        return self

    def _plot(self, ax, peak_candidate_groups=None, individual_peaks=False, xarr=None, **kwargs):
        if isinstance(peak_candidate_groups, ListPeakCandidateMultiModel):
            for cand, fitres in zip(peak_candidate_groups, self):
                x = np.linspace(*cand.boundaries, 200)
                if individual_peaks:
                    for component in fitres.components:
                        ax.plot(x, component.eval(x=x, params=fitres.params), **kwargs)
                else:
                    ax.plot(x, fitres.eval(x=x), **kwargs)

        else:
            for i, p in enumerate(self):
                if peak_candidate_groups is None:
                    p0_cent = p.params['p0_center']
                    left, right = p0_cent.min, p0_cent.max
                else:
                    left, right = peak_candidate_groups[i].boundaries(n_sigma=3)
                if xarr is None:
                    x = np.linspace(left, right, 100)
                else:
                    x = xarr[(xarr >= left) & (xarr <= right)]
                if individual_peaks:
                    for component in p.components:
                        ax.plot(x, component.eval(x=x, params=p.params), **kwargs)
                else:
                    ax.plot(x, p.eval(x=x), **kwargs)

    def to_dataframe(self):
        return pd.DataFrame(
            [
                dict(name=f'g{group:02d}_{key}', value=val.value, stderr=val.stderr)
                for group, res in enumerate(self)
                for key, val in res.params.items()
            ]
        ).sort_values('name')

    def to_csv(self, path_or_buf=None, sep=',', **kwargs):
        return self.to_dataframe().to_csv(path_or_buf=path_or_buf, sep=sep, **kwargs)


available_models_type = Literal['Gaussian', 'Lorentzian', 'Moffat', 'Voigt', 'PseudoVoigt', 'Pearson4', 'Pearson7']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def build_multipeak_model_params(profile: Union[available_models_type, List[available_models_type]],
                                 candidates: PeakCandidateMultiModel,
                                 baseline_model: Literal['linear', None] = 'linear',
                                 ):
    mod_list = list()
    if baseline_model == 'linear':
        mod_list.append(LinearModel(name='baseline', prefix='bl_'))
    for peak_i, peak in enumerate(candidates.peaks):
        mod_list.append(lmfit_models[profile](name=f'p{peak_i}', prefix=f'p{peak_i}_'))
    fit_model = np.sum(mod_list)
    fit_params = fit_model.make_params()
    if baseline_model == 'linear':
        fit_params['bl_slope'].set(value=candidates.base_slope, vary=False)
        fit_params['bl_intercept'].set(value=candidates.base_intercept, vary=False)

    for peak_i, peak in enumerate(candidates.peaks):
        #fit_params[f'p{peak_i}_center'].set(value=peak.position, vary=True)
        #fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma)


        if profile == 'Moffat':
            fwhm_factor = 2.
            height_factor = 2./peak.sigma**.5
            fit_params[f'p{peak_i}_amplitude'].set(value=peak.amplitude/height_factor)
            fit_params[f'p{peak_i}_beta'].set(value=1, min=1e-4, max=10)
            fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma)

        elif profile == 'Voigt':
            fwhm_factor = 3.6013
            height_factor = 1/peak.sigma/2
            fit_params[f'p{peak_i}_amplitude'].set(value=peak.amplitude/height_factor)
            fit_params[f'p{peak_i}_gamma'].set(value=peak.sigma/fwhm_factor, vary=True)
            fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma/fwhm_factor)

        elif profile == 'PseudoVoigt':
            fwhm_factor = lmfit_models[profile].fwhm_factor
            height_factor = 1/np.pi/np.sqrt(peak.sigma)/2
            fit_params[f'p{peak_i}_amplitude'].set(value=peak.amplitude/height_factor)
            fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma/fwhm_factor)

        elif profile == 'Pearson4':
            fwhm_factor = 1
            fit_params[f'p{peak_i}_height'].set(value=peak.amplitude)
            fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma/fwhm_factor)

        elif profile == 'Pearson7':
            fwhm_factor = 1
            height_factor = 1/2/peak.sigma
            fit_params[f'p{peak_i}_amplitude'].set(value=peak.amplitude/height_factor)
            fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma/fwhm_factor)

        else:
            fwhm_factor = lmfit_models[profile].fwhm_factor
            height_factor = lmfit_models[profile].height_factor/peak.sigma/2
            fit_params[f'p{peak_i}_amplitude'].set(value=peak.amplitude/height_factor)
            fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma)

        fit_params[f'p{peak_i}_amplitude'].set(min=0)
        fit_params[f'p{peak_i}_fwhm'].set(min=peak.fwhm*.4, max=peak.fwhm*2)
        fit_params[f'p{peak_i}_height'].set(min=peak.amplitude*.1, max=peak.amplitude*20)
        fit_params[f'p{peak_i}_center'].set(value=peak.position)

    return fit_model, fit_params


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def build_model_params(spe, model: Union[available_models_type, List[available_models_type]],
                       peak_candidates: ListPeakCandidateGroupsModel,
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

    n_sigma = 4
    if baseline_model == 'linear':
        li, ri = peak_candidates.boundaries_idx(n_sigma, x_arr=spe.x)
        xl = spe.x[li]
        yl = spe.y[li]
        xr = spe.x[ri]
        yr = spe.y[ri]
        slope = (yr-yl)/(xr-xl)
        intercept = -xl*slope + yl
    else:
        slope = 0
        intercept = 0

    fit_params = fit_model.make_params()
    if baseline_model == 'linear':
        fit_params['bl_slope'].set(value=slope)
        fit_params['bl_intercept'].set(value=intercept)

    pos_ampl_sigma_base = peak_candidates.pos_ampl_sigma_base()
    for i, (mod, (x0, a, w, p)) in enumerate(zip(model, pos_ampl_sigma_base)):
        #a = a - (slope*x0 + intercept)
        if a < 0:
            a = -a
        if mod == 'Moffat':
            fwhm_factor = 2.
            height_factor = 1.
            fit_params[f'p{i}_amplitude'].set(value=a/height_factor, min=0, max=a/height_factor*20)
            fit_params[f'p{i}_beta'].set(value=1, min=1e-4, max=100)
            fit_params[f'p{i}_center'].set(value=x0)
            fit_params[f'p{i}_sigma'].set(value=w/fwhm_factor, min=.1e-4, max=w/fwhm_factor*50)
        elif mod == 'Voigt':
            fwhm_factor = 3.6013
            height_factor = 1/w/np.sqrt(2)
            fit_params[f'p{i}_amplitude'].set(value=a/height_factor, min=0, max=a/height_factor*20)
            fit_params[f'p{i}_gamma'].set(value=w/fwhm_factor, min=.0001, max=w/fwhm_factor*10, vary=True)
            fit_params[f'p{i}_center'].set(value=x0)
            fit_params[f'p{i}_sigma'].set(value=w/fwhm_factor, min=.1e-4, max=w/fwhm_factor*50)
        elif mod == 'PseudoVoigt':
            fwhm_factor = lmfit_models[mod].fwhm_factor
            height_factor = 1/np.pi/np.sqrt(w)/2
            fit_params[f'p{i}_amplitude'].set(value=a/height_factor, min=0, max=a/height_factor*20)
            fit_params[f'p{i}_center'].set(value=x0)
            fit_params[f'p{i}_sigma'].set(value=w/fwhm_factor, min=.1e-4, max=w/fwhm_factor*50)
        elif mod == 'Pearson4':
            fwhm_factor = 1
            fit_params[f'p{i}_height'].set(value=a, max=a*20)
            fit_params[f'p{i}_center'].set(value=x0)
            fit_params[f'p{i}_sigma'].set(value=w/fwhm_factor, min=.1e-4, max=w/fwhm_factor*50)
        elif mod == 'Pearson7':
            fwhm_factor = 1
            height_factor = 1/2/w
            fit_params[f'p{i}_amplitude'].set(value=a/height_factor, min=0, max=a/height_factor*20)
            fit_params[f'p{i}_center'].set(value=x0)
            fit_params[f'p{i}_sigma'].set(value=w/fwhm_factor, min=.1e-4, max=w/fwhm_factor*50)
        else:
            fwhm_factor = lmfit_models[mod].fwhm_factor
            height_factor = lmfit_models[mod].height_factor/np.sqrt(w)/2
            fit_params[f'p{i}_amplitude'].set(value=a/height_factor, min=0, max=a/height_factor*20)
            fit_params[f'p{i}_center'].set(value=x0)
            fit_params[f'p{i}_sigma'].set(value=w/fwhm_factor, min=.1e-4, max=w/fwhm_factor*50)
    return fit_model, fit_params


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peaks_model(spe: Spectrum, /, *,
                    model: Union[available_models_type, List[available_models_type]],
                    peak_candidates: PeakCandidatesGroupModel,
                    n_sigma_trim: PositiveFloat = 5,
                    baseline_model: Literal['linear', None] = None,
                    no_fit=False,
                    kwargs_fit={}
                    ):
    fit_model, fit_params = build_model_params(spe=spe,
                                               model=model,
                                               peak_candidates=peak_candidates,
                                               baseline_model=baseline_model)
    lb, rb = peak_candidates.boundaries_idx(n_sigma=n_sigma_trim, x_arr=spe.x)
    if len(spe.x) < len(fit_model.param_names):
        logger.warning('Not enought number of points in the spectrum')
    if rb-lb < len(fit_model.param_names):
        logger.warning(
            'Number of data points is smaller than number of model parameters. Use bigger value for `n_sigma_trim`')
    lb -= len(fit_model.param_names)//2 - 1
    rb += len(fit_model.param_names)//2
    if lb < 0:
        rb += -lb
        lb = 0
    if rb >= len(spe.x):
        lb -= rb - len(spe.x) + 1
        rb = len(spe.x) - 1

    fitx = spe.x[lb:rb]
    fity = spe.y[lb:rb]

    for par in fit_params:
        if par.endswith('_center'):
            fit_params[par].set(min=spe.x[lb], max=spe.x[rb], vary=True)

    if no_fit:
        fit_tmp = fit_model.fit(fity, params=fit_params, x=fitx, **kwargs_fit, max_nfev=-1)
    else:
        fit_tmp = fit_model.fit(fity, params=fit_params, x=fitx, **kwargs_fit)
    return fit_tmp


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peak_groups(spe, /, *,
                    model: Union[available_models_type, List[available_models_type]],
                    peak_candidate_groups: ListPeakCandidateGroupsModel,
                    n_sigma_trim: float = 3,
                    no_fit=False,
                    kwargs_fit={}
                    ):
    fit_res = FitPeaksResult()
    for group_i, group in enumerate(peak_candidate_groups.__root__):
        fit_res.append(fit_peaks_model(spe,
                                       peak_candidates=group,
                                       model=model,
                                       baseline_model='linear',
                                       n_sigma_trim=n_sigma_trim,
                                       kwargs_fit=kwargs_fit,
                                       no_fit=no_fit,
                                       ))
    return fit_res


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peak_multimodel(spe, /, *,
                        profile: Union[available_models_type, List[available_models_type]],
                        candidates: ListPeakCandidateMultiModel,
                        no_fit=False,
                        kwargs_fit={}
                        ):
    if no_fit:
        kwargs_fit = dict(kwargs_fit)
        kwargs_fit['max_nfev'] = 1
    fit_res = FitPeaksResult()
    for group in candidates.__root__:
        mod, par = build_multipeak_model_params(profile=profile, candidates=group)
        idx = (group.boundaries[0] < spe.x) & (spe.x < group.boundaries[1])
        x = spe.x[idx]
        y = spe.y[idx]
        for i in range(len(group.peaks)):
            par[f'p{i}_center'].set(vary=False)
        fr = mod.fit(y, x=x, params=par, **kwargs_fit)
        for i in range(len(group.peaks)):
            par[f'p{i}_center'].set(vary=True)
        #par['bl_slope'].set(vary=True)
        #par['bl_intercept'].set(vary=True)
        fr = mod.fit(y, x=x, params=par, **kwargs_fit)
        fit_res.append(fr)
    return fit_res


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peaks(spe, /, *,
              model: available_models_type,
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
    """
    Write fit result as metadata.
    """
    cand_groups = ListPeakCandidateGroupsModel.validate(old_spe.result)
    new_spe.result = old_spe.fit_peak_groups(*args,  # type: ignore
                                             peak_candidate_groups=cand_groups,
                                             kwargs_fit=kwargs_fit,
                                             **kwargs).dumps()
