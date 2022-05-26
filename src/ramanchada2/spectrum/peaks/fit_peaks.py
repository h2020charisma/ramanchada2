#!/usr/bin/env python3

from typing import Literal, List, Union
from collections import UserList
import json

import numpy as np
from pydantic import validate_arguments, Field
from lmfit.models import lmfit_models

from ramanchada2.misc.spectrum_deco import (add_spectrum_method,
                                            add_spectrum_filter)
from ramanchada2.misc.types import (PeakCandidatesListModel,
                                    PeakCandidateModel)
from ..spectrum import Spectrum


class FitPeaksResult(UserList):
    def valuesdict(self):
        ret = dict()
        for i in self:
            ret.update(i.params.valuesdict())
        return ret

    @property
    def locations(self):
        return [v for peak in self for k, v in peak.values.items() if k.endswith('center')]

    def to_json(self):
        mod = [peak.model.dumps() for peak in self]
        par = [peak.params.dumps() for peak in self]
        return json.dumps(dict(models=mod, params=par))


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peaks_model(spe: Spectrum, /, *,
                    model: Literal['Gaussian', 'Lorentzian', 'Moffat',
                                   'Voigt', 'PseudoVoigt'],
                    peak_candidates: Union[PeakCandidateModel, PeakCandidatesListModel],
                    n_sigma_trim: float = Field(5, gt=0),
                    ):
    lb, rb = peak_candidates.boundaries_idx(n_sigma=n_sigma_trim, x_arr=spe.x)
    fitx = spe.x[lb:rb]
    fity = spe.y[lb:rb]

    if isinstance(peak_candidates, PeakCandidateModel):
        x0, a, w, p = peak_candidates.pos_ampl_sigma_base(x_arr=spe.x)
        fit_model = lmfit_models[model]()
        fit_params = fit_model.make_params()
        fit_params['amplitude'].set(value=a*3, min=0, max=a*50)
        fit_params['center'].set(value=x0, min=spe.x[lb], max=spe.x[rb])
        fit_params['sigma'].set(value=w, min=.01, max=w*5)
        if model == 'Moffat':
            fit_params['beta'].set(value=1)
    else:
        mod_list = list()
        pos_ampl_sigma_base = peak_candidates.pos_ampl_sigma_base(x_arr=spe.x)
        for i, (x0, a, w, p) in enumerate(pos_ampl_sigma_base):
            mod = lmfit_models[model](name=f'p{i}', prefix=f'p{i}_')
            mod_list.append(mod)
        fit_model = np.sum(mod_list)

        fit_params = fit_model.make_params()
        for i, (x0, a, w, p) in enumerate(pos_ampl_sigma_base):
            fit_params[f'p{i}_amplitude'].set(value=a*3, min=0, max=a*50)
            fit_params[f'p{i}_center'].set(value=x0, min=spe.x[lb], max=spe.x[rb])
            fit_params[f'p{i}_sigma'].set(value=w, min=.01, max=w*5)
            if model == 'Moffat':
                fit_params[f'p{i}_beta'].set(value=1)

    if model == 'Voigt':
        tmpres = fit_model.fit(fity, params=fit_params, x=fitx)
        fit_params = tmpres.params
        if isinstance(peak_candidates, PeakCandidateModel):
            fit_params['gamma'].set(vary=True)
        else:
            for i in range(len(peak_candidates)):
                fit_params[f'p{i}_gamma'].set(vary=True)
    return fit_model.fit(fity, params=fit_params, x=fitx)


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peak_groups(spe, /, *,
                    model: Literal['Gaussian', 'Lorentzian', 'Moffat',
                                   'Voigt', 'PseudoVoigt',
                                   ],
                    peak_candidate_groups: List[PeakCandidatesListModel],
                    n_sigma_trim: float = 3,
                    ):
    fit_res = FitPeaksResult()
    for group in peak_candidate_groups:
        fit_res.append(fit_peaks_model(spe,
                                       peak_candidates=group,
                                       model=model,
                                       n_sigma_trim=n_sigma_trim))
    return fit_res


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peaks(spe, /, *,
              model: Literal['Gaussian', 'Lorentzian', 'Moffat',
                             'Voigt', 'PseudoVoigt',
                             ],
              peak_candidates: PeakCandidatesListModel,
              n_sigma_trim: float = 3,
              ):
    fit_res = FitPeaksResult()
    for peak in peak_candidates:
        fit_res.append(fit_peaks_model(spe,
                                       peak_candidates=peak,
                                       model=model,
                                       n_sigma_trim=n_sigma_trim))
    return fit_res


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peaks_filter(
        old_spe: Spectrum,
        new_spe: Spectrum, /, model):
    find_peaks_res = old_spe.result
    new_spe.result = old_spe.fit_peaks(model, **find_peaks_res).to_json()  # type: ignore
