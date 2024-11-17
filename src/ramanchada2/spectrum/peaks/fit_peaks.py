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
available_models = ['Gaussian', 'Skewed Gaussian', 'Lorentzian', 'Moffat',
                    'Voigt', 'Skewed Voigt', 'PseudoVoigt', 'Pearson4', 'Pearson7']
available_models_type = Literal['Gaussian', 'Skewed Gaussian', 'Lorentzian', 'Moffat',
                                'Voigt', 'Skewed Voigt', 'PseudoVoigt', 'Pearson4', 'Pearson7']


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
            fit_params[f'p{peak_i}_fwhm'].set(min=peak.fwhm*.4, max=peak.fwhm*2)

        elif profile == 'Voigt':
            fwhm_factor = 3.6013
            height_factor = 1/peak.sigma/2
            fit_params[f'p{peak_i}_amplitude'].set(value=peak.amplitude/height_factor)
            fit_params[f'p{peak_i}_gamma'].set(value=peak.sigma/fwhm_factor, vary=True)
            fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma/fwhm_factor)
            fit_params[f'p{peak_i}_fwhm'].set(min=peak.fwhm*.4, max=peak.fwhm*2)

        elif profile == 'Skewed Voigt':
            fwhm_factor = 3.6013
            height_factor = 1/peak.sigma/2
            fit_params[f'p{peak_i}_amplitude'].set(value=peak.amplitude/height_factor)
            fit_params[f'p{peak_i}_gamma'].set(value=peak.sigma/fwhm_factor, vary=True)
            fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma/fwhm_factor)
            fit_params[f'p{peak_i}_skew'].set(value=0)

        elif profile == 'Skewed Gaussian':
            fwhm_factor = lmfit_models['Gaussian'].fwhm_factor
            height_factor = lmfit_models['Gaussian'].height_factor/peak.sigma
            fit_params[f'p{peak_i}_amplitude'].set(value=peak.amplitude/height_factor)
            fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma)
            fit_params[f'p{peak_i}_gamma'].set(value=peak.sigma)

        elif profile == 'PseudoVoigt':
            fwhm_factor = lmfit_models[profile].fwhm_factor
            height_factor = 1/np.pi/np.sqrt(peak.sigma)/2
            fit_params[f'p{peak_i}_amplitude'].set(value=peak.amplitude/height_factor)
            fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma/fwhm_factor)

        elif profile == 'Pearson4':
            fwhm_factor = 1
            height_factor = 1/peak.sigma/3
            # p{peak_i}_amplitude or p{peak_i}_height
            fit_params[f'p{peak_i}_amplitude'].set(value=peak.amplitude/height_factor)
            fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma/fwhm_factor)

        elif profile == 'Pearson7':
            fwhm_factor = 1
            height_factor = 1/2/peak.sigma
            fit_params[f'p{peak_i}_amplitude'].set(value=peak.amplitude/height_factor)
            fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma/fwhm_factor)

        else:
            fwhm_factor = lmfit_models[profile].fwhm_factor
            height_factor = lmfit_models[profile].height_factor/peak.sigma
            fit_params[f'p{peak_i}_amplitude'].set(value=peak.amplitude/height_factor)
            fit_params[f'p{peak_i}_sigma'].set(value=peak.sigma)
            fit_params[f'p{peak_i}_fwhm'].set(min=peak.fwhm*.4, max=peak.fwhm*2)
            fit_params[f'p{peak_i}_height'].set(min=peak.amplitude*.1, max=peak.amplitude*20)

        fit_params[f'p{peak_i}_amplitude'].set(min=0)
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
                        bound_centers_to_group: bool = False
                        ) -> FitPeaksResult:
    """
    Fit a model based on candidates to the spectrum.

    Args:
        spe: internal use only
        profile: str or List[str]
            possible values are: ["""+str(available_models)+"""]
        candidates: as provided from find_peak_multipeak
        no_fit: optional. Defaults to False.
            If true, do not perform a fit. Result will be the inital guess,
            based on the data from peak candidates.
        should_break: optional. Defaults to [False].
            Use mutability of the python list type to be able to externaly
            break the minimization procedure.
        kwargs_fit: optional
            kwargs for fit function
        vary_baseline: optional. Defaults to False.
            If False baseline will not be a free parameter and its amplitude
            will be taken from the peak candidates.
        bound_centers_to_group: optional. Defaults to False.
            Perform a bounded fit. Request all peak centers to be within the group
            interval.

    Returns:
        FitPeaksResult: groups of fitted peaks
    """

    def iter_cb(params, iter, resid, *args, **kws):
        return should_break[0]
    if no_fit:
        kwargs_fit = dict(kwargs_fit)
        kwargs_fit['max_nfev'] = 1
    fit_res = FitPeaksResult()
    for group in candidates.root:
        mod, par = build_multipeak_model_params(profile=profile, candidates=group)
        if bound_centers_to_group:
            for p in par:
                if p.endswith('_center'):
                    par[p].set(min=group.boundaries[0], max=group.boundaries[1])
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
    Same as `fit_peak_multipeak` but the result is stored as metadata in the returned spectrum.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        should_break: same as in fit_peaks_multipeak
        *args, **kwargs: same as `fit_peaks_multipeak`
    """
    cand_groups = ListPeakCandidateMultiModel.model_validate(old_spe.result)
    new_spe.result = old_spe.fit_peak_multimodel(*args,  # type: ignore
                                                 candidates=cand_groups,
                                                 should_break=should_break,
                                                 kwargs_fit=kwargs_fit,
                                                 **kwargs).dumps()
