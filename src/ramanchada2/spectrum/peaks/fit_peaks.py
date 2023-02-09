#!/usr/bin/env python3

import re
from typing import Literal, List, Union
from collections import UserList, defaultdict
import logging

import numpy as np
import pandas as pd
from pydantic import validate_arguments
from lmfit.models import lmfit_models, LinearModel
from lmfit.model import ModelResult, Parameters, Model

from ramanchada2.misc.spectrum_deco import (add_spectrum_method,
                                            add_spectrum_filter)
from ramanchada2.misc.plottable import Plottable
from ramanchada2.misc.types.peak_candidates import ListPeakCandidateMultiModel, PeakCandidateMultiModel

from ..spectrum import Spectrum

logger = logging.getLogger(__name__)
available_models = ['Gaussian', 'Lorentzian', 'Moffat', 'Voigt', 'PseudoVoigt', 'Pearson4', 'Pearson7']
available_models_type = Literal['Gaussian', 'Lorentzian', 'Moffat', 'Voigt', 'PseudoVoigt', 'Pearson4', 'Pearson7']


class FitPeaksResult(UserList, Plottable):
    def valuesdict(self):
        ret = dict()
        for i in self:
            ret.update(i.params.valuesdict())
        return ret

    @property
    def locations(self):
        return np.array([v for peak in self for k, v in peak.values.items() if k.endswith('center')])

    @property
    def centers(self):
        return np.array([v for peak in self for k, v in peak.values.items() if k.endswith('center')])

    @property
    def fwhm(self):
        return np.array([v for peak in self for k, v in peak.values.items() if k.endswith('fwhm')])

    def boundaries(self, n_sigma=5):
        bounds = list()
        for group in self:
            pos = np.array([v for k, v in group.values.items() if k.endswith('center')])
            sig = np.array([v for k, v in group.values.items() if k.endswith('fwhm')])
            sig /= 2.35
            sig *= n_sigma
            bounds.append((np.min(pos - sig), np.max(pos+sig)))
        return bounds

    def center_amplitude(self, threshold):
        return np.array([
            (v.value, peak.params[k[:-6] + 'amplitude'].value)
            for peak in self
            for k, v in peak.params.items()
            if k.endswith('center')
            if hasattr(v, 'stderr') and v.stderr is not None and v.stderr < threshold
        ]).T

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
        return np.array([v for peak in self for k, v in peak.values.items() if k.endswith('fwhm')])

    @property
    def amplitudes(self):
        return np.array([v for peak in self for k, v in peak.values.items() if k.endswith('amplitude')])

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
        def group_plot(x, fitres):
            if individual_peaks:
                color = None
                for component in fitres.components:
                    line, = ax.plot(x, component.eval(x=x, params=fitres.params), color=color, **kwargs)
                    color = line.get_c()
            else:
                ax.plot(x, fitres.eval(x=x), **kwargs)

        if peak_candidate_groups is None:
            for bound, fitres in zip(self.boundaries(), self):
                x = np.linspace(*bound, 200)
                group_plot(x, fitres)
        elif isinstance(peak_candidate_groups, ListPeakCandidateMultiModel):
            for cand, fitres in zip(peak_candidate_groups, self):
                x = np.linspace(*cand.boundaries, 2000)
                group_plot(x, fitres)
        else:
            for i, fitres in enumerate(self):
                left, right = peak_candidate_groups[i].boundaries(n_sigma=3)
                x = np.linspace(left, right, 100)
                group_plot(x, fitres)

    def to_dataframe(self):
        return pd.DataFrame(
            [
                dict(name=f'g{group:02d}_{key}', value=val.value, stderr=val.stderr)
                for group, res in enumerate(self)
                for key, val in res.params.items()
            ]
        ).sort_values('name')

    def to_dataframe_peaks(self):
        regex = re.compile(r'p([0-1]+)_(.*)')
        ret = defaultdict(dict)
        for group_i, group in enumerate(self):
            for par in group.params:
                m = regex.match(par)
                if m is None:
                    continue
                peak_i, par_name = m.groups()
                ret[f'g{group_i:02d}_p{peak_i}'][par_name] = group.params[par].value
                ret[f'g{group_i:02d}_p{peak_i}'][f'{par_name}_stderr'] = group.params[par].stderr
        return pd.DataFrame.from_dict(ret, orient='index')

    def to_csv(self, path_or_buf=None, sep=',', **kwargs):
        return self.to_dataframe_peaks().to_csv(path_or_buf=path_or_buf, sep=sep, **kwargs)


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


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peak_multimodel(spe, /, *,
                        profile: Union[available_models_type, List[available_models_type]],
                        candidates: ListPeakCandidateMultiModel,
                        no_fit=False,
                        kwargs_fit={},
                        vary_baseline: bool = False,
                        ) -> FitPeaksResult:
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
        if vary_baseline:
            par['bl_slope'].set(vary=True)
            par['bl_intercept'].set(vary=True)
        fr = mod.fit(y, x=x, params=par, **kwargs_fit)
        fit_res.append(fr)
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
    cand_groups = ListPeakCandidateMultiModel.validate(old_spe.result)
    new_spe.result = old_spe.fit_peak_multimodel(*args,  # type: ignore
                                                 candidates=cand_groups,
                                                 kwargs_fit=kwargs_fit,
                                                 **kwargs).dumps()
