import logging

from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_method

from ..spectrum import Spectrum

logger = logging.getLogger(__name__)


@add_spectrum_method
@validate_call(config=dict(arbitrary_types_allowed=True))
def fit_peak_positions(spe: Spectrum, /, *,
                       mov_min=40,
                       center_err_threshold=.5,
                       find_peaks_kw={},
                       fit_peaks_kw={},
                       ):
    ss = spe.subtract_moving_minimum(mov_min)  # type: ignore
    find_kw = dict(sharpening=None)
    find_kw.update(find_peaks_kw)
    cand = ss.find_peak_multipeak(**find_kw)

    fit_kw = dict(profile='Gaussian')
    fit_kw.update(fit_peaks_kw)
    fit_res = spe.fit_peak_multimodel(candidates=cand, **fit_kw)  # type: ignore

    pos, amp = fit_res.center_amplitude(threshold=center_err_threshold)

    return dict(zip(pos, amp))
