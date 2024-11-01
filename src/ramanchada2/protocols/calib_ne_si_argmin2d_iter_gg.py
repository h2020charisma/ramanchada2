from typing import Literal

import numpy as np
from scipy import interpolate

from ..misc import constants as rc2const
from ..spectrum.spectrum import Spectrum


def neon_calibration(ne_cm_1: Spectrum,
                     wl: Literal[514, 532, 633, 785]):
    ref = rc2const.neon_wl_dict[wl]
    ne_nm = ne_cm_1.subtract_moving_minimum(200).shift_cm_1_to_abs_nm_filter(wl).normalize()  # type: ignore

    ne_cal = ne_nm.xcal_argmin2d_iter_lowpass(ref=ref)
    spline = interpolate.Akima1DInterpolator(ne_cm_1.x, ne_cal.x, method='makima')
    return spline


def silicon_calibration(si_nm: Spectrum,
                        wl: Literal[514, 532, 633, 785]):
    si_nm = si_nm.dropna().normalize()  # type: ignore
    peaks = si_nm.find_peak_multipeak(prominence=.05, width=2, wlen=50)  # type: ignore
    fitres = si_nm.fit_peak_multimodel(candidates=peaks, profile='Pearson4')  # type: ignore
    si_wl = fitres.centers
    laser_wl = 1/(520.45/1e7 + 1/si_wl)

    laser_wl = laser_wl[np.argmin(np.abs(laser_wl-wl))]
    x_cm_1 = 1e7*(1/laser_wl-1/si_nm.x)

    spline = interpolate.Akima1DInterpolator(si_nm.x, x_cm_1, method='makima')
    return spline, laser_wl


def neon_silicon_calibration(ne_cm_1: Spectrum,
                             si_cm_1: Spectrum,
                             wl: Literal[514, 532, 633, 785]):
    ne_spline = neon_calibration(ne_cm_1, wl)
    si_nm = si_cm_1.scale_xaxis_fun(ne_spline)  # type: ignore
    si_spline, wl = silicon_calibration(si_nm, wl)
    ne_nm = ne_cm_1.scale_xaxis_fun(ne_spline)  # type: ignore
    ne_cal_cm_1 = ne_nm.abs_nm_to_shift_cm_1_filter(wl)
    spline = interpolate.Akima1DInterpolator(ne_cm_1.x, ne_cal_cm_1.x, method='makima')
    return spline
