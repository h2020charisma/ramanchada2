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
                        wl: Literal[514, 532, 633, 785],
                        find_peaks_kw={},
                        fit_peaks_kw={}):
    """
    Calculate calibration function for lazer zeroing

    Args:
        si_nm: Spectrum
            Silicon spectrum
        wl: Literal[514, 532, 633, 785]
            Laser wavelength
        find_peaks_kw: dict, optional
            keywords for find_peak. Default values are
            `{'prominence': min(.8, si_nm.y_noise_MAD()*50), 'width': 2, 'wlen': 100}`
        fit_peaks_kw: dict, optional
            keywords for fit_peaks. Default values are
            `{'profile': 'Pearson4', 'vary_baseline': False}`

    Returns:
        spline, esitmated_wavelength: int
    """
    si_nm_orig = si_nm
    fnd_kw = {'prominence': min(.8, si_nm.y_noise_MAD()*50),
              'width': 2,
              'wlen': 100,
              }
    fnd_kw.update(find_peaks_kw)
    ll = wl/(1-wl*(520-50)*1e-7)
    rr = wl/(1-wl*(520+50)*1e-7)
    si_nm = si_nm.dropna().trim_axes(method='x-axis', boundaries=(ll, rr)).normalize()  # type: ignore
    peaks = si_nm.find_peak_multipeak(**fnd_kw)  # type: ignore
    fp_kw = {'profile': 'Pearson4',
             'vary_baseline': False
             }
    fp_kw.update(fit_peaks_kw)
    fitres = si_nm.fit_peak_multimodel(candidates=peaks, **fp_kw)  # type: ignore
    si_wl = fitres.centers
    if len(si_wl) < 1:
        raise ValueError('No peaks were found. Please refind find_peaks parameters.')
    laser_wl = 1/(520.45/1e7 + 1/si_wl)

    laser_wl = laser_wl[np.argmin(np.abs(laser_wl-wl))]
    x_cm_1 = 1e7*(1/laser_wl-1/si_nm_orig.x)

    spline = interpolate.Akima1DInterpolator(si_nm_orig.x, x_cm_1, method='makima')
    return spline, laser_wl


def neon_silicon_calibration(ne_cm_1: Spectrum,
                             si_cm_1: Spectrum,
                             wl: Literal[514, 532, 633, 785],
                             sil_fit_kw={},
                             sil_find_kw={}
                             ):
    ne_spline = neon_calibration(ne_cm_1, wl)
    si_nm = si_cm_1.scale_xaxis_fun(ne_spline)  # type: ignore
    si_spline, wl = silicon_calibration(si_nm, wl,
                                        find_peaks_kw=sil_find_kw,
                                        fit_peaks_kw=sil_fit_kw)
    ne_nm = ne_cm_1.scale_xaxis_fun(ne_spline)  # type: ignore
    ne_cal_cm_1 = ne_nm.abs_nm_to_shift_cm_1_filter(wl)
    spline = interpolate.Akima1DInterpolator(ne_cm_1.x, ne_cal_cm_1.x, method='makima')
    return spline
