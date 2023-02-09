#!/usr/bin/env python3

from pydantic import validate_arguments
import logging

from ramanchada2.misc.spectrum_deco import add_spectrum_method
from ..spectrum import Spectrum

logger = logging.getLogger(__name__)


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def fit_peak_positions(spe: Spectrum, /, *,
                       mov_min=40,
                       center_err_threshold=.5,
                       kw_args_find_peaks={},
                       kw_args_fit_peaks={},
                       ):
    ss = spe.subtract_moving_minimum(mov_min)  # type: ignore
    kw_find = dict(sharpening=None)
    kw_find.update(kw_args_find_peaks)
    cand = ss.find_peak_multipeak(**kw_find)

    kw_fit = dict(profile='Gaussian')
    kw_fit.update(kw_args_fit_peaks)
    fit_res = spe.fit_peak_multimodel(candidates=cand, **kw_fit)  # type: ignore

    pos, amp = fit_res.center_amplitude(threshold=center_err_threshold)

    return dict(zip(pos, amp))
