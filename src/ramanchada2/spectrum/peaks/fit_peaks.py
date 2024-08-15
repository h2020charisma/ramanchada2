import logging
from typing import List, Literal, Union

import numpy as np
from lmfit.models import LinearModel, lmfit_models
from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import (add_spectrum_filter,
                                            add_spectrum_method)
from ramanchada2.misc.types.fit_peaks_result import FitPeaksResult
from ramanchada2.misc.types.peak_candidates import (
    ListPeakCandidateMultiModel, PeakCandidateMultiModel)

from ..spectrum import Spectrum

logger = logging.getLogger(__name__)
available_models = ['Gaussian', 'Lorentzian', 'Moffat', 'Voigt', 'PseudoVoigt', 'Pearson4', 'Pearson7']
available_models_type = Literal['Gaussian', 'Lorentzian', 'Moffat', 'Voigt', 'PseudoVoigt', 'Pearson4', 'Pearson7']


@validate_call(config=dict(arbitrary_types_allowed=True))
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
            # p{peak_i}_amplitude or p{peak_i}_height
            fit_params[f'p{peak_i}_amplitude'].set(value=peak.amplitude)
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
@validate_call(config=dict(arbitrary_types_allowed=True))
def fit_peak_multimodel(spe, /, *,
                        profile: Union[available_models_type, List[available_models_type]],
                        candidates: ListPeakCandidateMultiModel,
                        no_fit=False,
                        should_break=[False],
                        kwargs_fit={},
                        vary_baseline: bool = False,
                        ) -> FitPeaksResult:
    def iter_cb(params, iter, resid, *args, **kws):
        return should_break[0]
    if no_fit:
        kwargs_fit = dict(kwargs_fit)
        kwargs_fit['max_nfev'] = 1
    fit_res = FitPeaksResult()
    for group in candidates.root:
        mod, par = build_multipeak_model_params(profile=profile, candidates=group)
        idx = (group.boundaries[0] < spe.x) & (spe.x < group.boundaries[1])
        x = spe.x[idx]
        y = spe.y[idx]
        for i in range(len(group.peaks)):
            par[f'p{i}_center'].set(vary=False)
        fr = mod.fit(y, x=x, params=par, iter_cb=iter_cb,  **kwargs_fit)
        for i in range(len(group.peaks)):
            par[f'p{i}_center'].set(vary=True)
        if vary_baseline:
            par['bl_slope'].set(vary=True)
            par['bl_intercept'].set(vary=True)
        fr = mod.fit(y, x=x, params=par, iter_cb=iter_cb, **kwargs_fit)
        fit_res.append(fr)
    return fit_res


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def fit_peaks_filter(
        old_spe: Spectrum,
        new_spe: Spectrum, /, *args,
        should_break=[False],
        kwargs_fit={},
        **kwargs,
        ):
    """
    Write fit result as metadata.
    """
    cand_groups = ListPeakCandidateMultiModel.model_validate(old_spe.result)
    new_spe.result = old_spe.fit_peak_multimodel(*args,  # type: ignore
                                                 candidates=cand_groups,
                                                 should_break=should_break,
                                                 kwargs_fit=kwargs_fit,
                                                 **kwargs).dumps()
