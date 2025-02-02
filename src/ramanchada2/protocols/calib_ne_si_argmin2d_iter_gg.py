from typing import Literal

import numpy as np
from scipy import interpolate

from ..misc import constants as rc2const
from ..spectrum.spectrum import Spectrum


def neon_calibration(ne_cm_1: Spectrum,
                     wl: Literal[514, 532, 633, 785],
                     neon_rough_kw={},
                     neon_fine_kw={}):
    """
    Neon calibration

    The approximate laser wavelengt `wl` is used to translate the neon spectrum to [nm].
    Then using :func:`~ramanchada2.spectrum.calibration.by_deltas.xcal_argmin2d_iter_lowpass`
    the approximate neon spectrum in [nm] is scaled to match the reference lines.
    This way it is calibrated to absolute wavelengths. A Makima spline is calculated so that
    it takes wavenumbers [1/cm] and return wavelength-calibrated x-axis in wavelengths [nm].

    Args:
        ne_cm_1 (Spectrum): neon spectrum used for the calibration. Should be in [1/cm]
        wl (Literal[514, 532, 633, 785]): Approximate laser wavelength in [nm]

    Returns:
        Callable(ArrayLike[float]): callable (spline) that applies the calibration
    """
    ref = rc2const.neon_wl_dict[wl]
    ne_nm = ne_cm_1.subtract_moving_minimum(200).shift_cm_1_to_abs_nm_filter(wl).normalize()  # type: ignore
    ne_nm.x = np.linspace(0, 1, len(ne_nm.x))

    ne_rcal = ne_nm.xcal_rough_poly2_all_pairs(ref=ref, **neon_rough_kw)
    ne_cal = ne_rcal.xcal_fine(ref=ref, **neon_fine_kw)
    spline = interpolate.Akima1DInterpolator(ne_cm_1.x, ne_cal.x, method='makima')
    return spline


def silicon_calibration(si_nm: Spectrum,
                        wl: Literal[514, 532, 633, 785],
                        find_peaks_kw={},
                        fit_peaks_kw={}):
    """
    Calculate calibration function for lazer zeroing

    Takes wavelength-calibrated Silicon spectrum in wavelengths [nm] and using
    the Silicon peak position it calculates the real laser wavelength and a Makima
    spline that translates the wavelengt-calibrated x-axis wavelength [nm] values to
    lazer-zeroed Raman shift in wavenumbers [1/cm].

    Args:
        si_nm: Spectrum
            Wavelength-calibrated Silicon spectrum in wavelengths [nm]
        wl: Literal[514, 532, 633, 785]
            Approximate Laser wavelength
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
        raise ValueError('No peaks were found. Please refine find_peaks parameters.')
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
    """
    Perform neon and silicon calibration together

    Combines :func:`~ramanchada2.protocols.calib_ne_si_argmin2d_iter_gg.neon_calibration`
    and :func:`~ramanchada2.protocols.calib_ne_si_argmin2d_iter_gg.silicon_calibration`.
    Returned spline is calculated using the wavlength-calibrated x-axis values translated
    to Raman shift wavenumbers using the calculated laser wavelength in `silicon_calibration`

    Args:
        ne_cm_1 (Spectrum): neon spectrum used for the calibration. Should be in [1/cm]
        si_cm_1 (Spectrum): silicon spectrum to estimate laser wavelength. Should be in [1/cm].
        wl (Literal[514, 532, 633, 785]): Approximate laser wavelength in [nm]
        sil_fit_kw (dict, optional): kwargs sent as `find_peaks_kw` in `silicon_calibration`. Defaults to {}.
        sil_find_kw (dict, optional): kwargs sent as `fit_peaks_kw` in `silicon_calibration`. Defaults to {}.

    Returns:
        Callable(ArrayLike[float]): callable (spline) that applies the calibration
    """
    ne_spline = neon_calibration(ne_cm_1, wl)
    si_nm = si_cm_1.scale_xaxis_fun(ne_spline)  # type: ignore
    si_nm = si_nm.dropna()
    si_spline, wl = silicon_calibration(si_nm, wl,
                                        find_peaks_kw=sil_find_kw,
                                        fit_peaks_kw=sil_fit_kw)
    ne_nm = ne_cm_1.scale_xaxis_fun(ne_spline)  # type: ignore
    ne_cal_cm_1 = ne_nm.abs_nm_to_shift_cm_1_filter(wl)
    ne_cal_cm_1 = ne_cal_cm_1.dropna()
    spline = interpolate.Akima1DInterpolator(ne_cm_1.x, ne_cal_cm_1.x, method='makima')
    return spline
