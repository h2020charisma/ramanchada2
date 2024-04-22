#!/usr/bin/env python3
import lmfit
import numpy as np
import numpy.typing as npt
from ...misc import utils as rc2utils
from ..spectrum import Spectrum
from pydantic import NonNegativeInt, validate_arguments
from ramanchada2.misc.spectrum_deco import add_spectrum_filter, add_spectrum_method
from scipy import interpolate
from typing import Dict, List, Literal, Union


class DeltaSpeModel:
    def __init__(self, deltas: Dict[float, float], shift=0, scale=1):
        components = list()
        self.params = lmfit.Parameters()
        self.minx = np.min(list(deltas.keys()))
        self.maxx = np.max(list(deltas.keys()))
        self.params.add('shift', value=shift, vary=True)
        self.params.add('scale', value=scale, vary=True, min=.1, max=10)
        self.params.add('scale2', value=0, vary=False, min=-.1, max=.1)
        self.params.add('scale3', value=0, vary=False, min=-1e-3, max=1e-3)
        self.params.add('sigma', value=1, vary=True)
        self.params.add('gain', value=1, vary=True)
        for comp_i, (k, v) in enumerate(deltas.items()):
            prefix = f'comp_{comp_i}'
            components.append(lmfit.models.GaussianModel(prefix=f'comp_{comp_i}_'))
            self.params.add(prefix + '_center', expr=f'shift + {k}*scale + {k}**2*scale2 + {k}**3*scale3', vary=False)
            self.params.add(prefix + '_amplitude', expr=f'{v}*gain', vary=False)
            self.params.add(prefix + '_sigma', expr='sigma', vary=False)
        self.model = np.sum(components)

    def fit(self, spe, sigma, ax=None, no_fit=False):
        self.params['sigma'].set(value=sigma if sigma > 1 else 1)
        spe_conv = spe.convolve('gaussian', sigma=sigma)
        if no_fit:
            fit_res = self.model.fit(spe_conv.y, x=spe_conv.x, params=self.params, max_nfev=-1)
        else:
            fit_res = self.model.fit(spe_conv.y, x=spe_conv.x, params=self.params)
        self.params = fit_res.params
        if ax is not None:
            spe_conv.plot(ax=ax)
            ax.plot(spe_conv.x, fit_res.eval(x=spe_conv.x), 'r')


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def calibrate_by_deltas_model(spe: Spectrum, /,
                              deltas: Dict[float, float],
                              convolution_steps: Union[None, List[float]] = [15, 1],
                              scale2=True, scale3=False,
                              init_guess: Literal[None, 'cumulative'] = None,
                              ax=None, **kwargs
                              ):
    """
    - Builds a composite model based on a set of user specified delta lines.
    - Initial guess is calculated based on 10-th and 90-th percentiles of
      the distributions.

    The phasespace of the model is flat with big amount of narrow minima.
    In order to find the best fit, the experimental data are successively
    convolved with gaussians with different widths startign from wide to
    narrow. The model for the calibration is 3-th order polynomial, which
    potentialy can be changed for higher order polynomial. In order to avoid
    solving the inverse of the calibration function, the result is tabulated
    and interpolated linarly for each bin of the spectrum.
    This alogrithm is useful for corse calibration.
    """
    mod = DeltaSpeModel(deltas)
    spe_padded = spe

    if init_guess == 'cumulative':
        deltasx = np.array(list(deltas.keys()))

        deltas_cs = np.cumsum(list(deltas.values()))
        deltas_cs /= deltas_cs[-1]

        deltas_idx10 = np.argmin(np.abs(deltas_cs-.1))
        deltas_idx90 = np.argmin(np.abs(deltas_cs-.9))
        x1, x2 = deltasx[[deltas_idx10, deltas_idx90]]

        spe_cs = np.cumsum(
            spe_padded.moving_average(50).subtract_moving_minimum(10).moving_average(5).y)  # type: ignore

        spe_cs /= spe_cs[-1]
        spe_idx10 = np.argmin(np.abs(spe_cs-.1))
        spe_idx90 = np.argmin(np.abs(spe_cs-.9))
        y1, y2 = spe_padded.x[[spe_idx10, spe_idx90]]

        scale = (y1-y2)/(x1-x2)
        shift = -scale * x1 + y1
    else:
        scale = 1
        shift = 0
    gain = np.sum(spe.y)/np.sum(list(deltas.values()))
    mod.params['scale'].set(value=scale)
    mod.params['shift'].set(value=shift)
    mod.params['gain'].set(value=gain)
    mod.params['sigma'].set(value=2.5)

    if ax is not None:
        spe_padded.plot(ax=ax)

    if convolution_steps is not None:
        for sig in convolution_steps:
            mod.fit(spe=spe_padded, sigma=sig, ax=ax, **kwargs)

    if scale2:
        mod.params['scale2'].set(vary=True, value=0)
        # mod.fit(spe_padded, sigma=1, ax=ax, **kwargs)
        mod.fit(spe_padded, sigma=0, ax=ax, **kwargs)
    if scale3:
        mod.params['scale2'].set(vary=True, value=0)
        mod.params['scale3'].set(vary=True, value=0)
        # mod.fit(spe_padded, sigma=1, ax=ax, **kwargs)
        mod.fit(spe_padded, sigma=0, ax=ax, **kwargs)
    return mod.model, mod.params


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def calibrate_by_deltas_filter(old_spe: Spectrum,
                               new_spe: Spectrum, /,
                               deltas: Dict[float, float],
                               convolution_steps,
                               init_guess=None,
                               **kwargs
                               ):
    mod, par = old_spe.calibrate_by_deltas_model(  # type: ignore
        deltas=deltas,
        convolution_steps=convolution_steps,
        init_guess=init_guess,
        **kwargs)

    deltasx = np.array(list(deltas.keys()))
    dxl, dxr = deltasx[[0, -1]]
    xl = dxl - (dxr - dxl)
    xr = dxl + (dxr - dxl)
    true_x = np.linspace(xl, xr, len(old_spe.x)*6)
    meas_x = (par['shift'].value + true_x * par['scale'] +
              true_x**2 * par['scale2'] + true_x**3 * par['scale3'])
    x_cal = np.zeros_like(old_spe.x)
    for i in range(len(old_spe.x)):
        idx = np.argmax(meas_x > old_spe.x[i])
        pt_rto = (old_spe.x[i] - meas_x[idx-1])/(meas_x[idx] - meas_x[idx-1])
        x_cal[i] = (true_x[idx] - true_x[idx-1])*pt_rto + true_x[idx-1]
    new_spe.x = x_cal


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def xcal_fine(old_spe: Spectrum,
              new_spe: Spectrum, /, *,
              ref: Union[Dict[float, float], List[float]],
              should_fit=False,
              poly_order: NonNegativeInt,
              find_peaks_kw={},
              ):

    if isinstance(ref, dict):
        ref_pos = np.array(list(ref.keys()))
    else:
        ref_pos = np.array(ref)

    if should_fit:
        spe_pos_dict = old_spe.fit_peak_positions(center_err_threshold=1, find_peaks_kw=find_peaks_kw)  # type: ignore
    else:
        find_kw = dict(sharpening=None)
        find_kw.update(find_peaks_kw)
        spe_pos_dict = old_spe.find_peak_multipeak(**find_kw).get_pos_ampl_dict()  # type: ignore
    spe_cent = np.array(list(spe_pos_dict.keys()))

    if poly_order == 0:
        p = rc2utils.align_shift(spe_cent, ref_pos)
        spe_cal = old_spe.scale_xaxis_fun(lambda x: x + p)  # type: ignore
    else:
        def cal_func(x, *a):
            return [par*(x/1000)**power for power, par in enumerate(a)]

        p0 = np.resize([0, 1000, 0], poly_order + 1)
        p = rc2utils.align(spe_cent, ref_pos, p0=p0, func=cal_func)
        spe_cal = old_spe.scale_xaxis_fun(  # type: ignore
            (lambda x, *args: np.sum(cal_func(x, *args), axis=0)), args=p)
    new_spe.x = spe_cal.x


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def xcal_fine_RBF(old_spe: Spectrum,
                  new_spe: Spectrum, /, *,
                  ref: Union[Dict[float, float], List[float], npt.NDArray],
                  should_fit=False,
                  kernel: Literal['thin_plate_spline', 'cubic', 'quintic',
                                  'multiquadric', 'inverse_multiquadric',
                                  'inverse_quadratic', 'gaussian',
                                  ] = 'thin_plate_spline',
                  find_peaks_kw={},
                  **kwargs,
                  ):
    """Wavelength calibration using Radial basis fuction interpolation

    Please be cautious! Interpolation might not be the most appropriate
    approach for this type of calibration.

    **kwargs are passed to RBFInterpolator
    """

    if isinstance(ref, dict):
        ref_pos = np.array(list(ref.keys()))
    else:
        ref_pos = np.array(ref)

    if should_fit:
        spe_pos_dict = old_spe.fit_peak_positions(center_err_threshold=1, find_peaks_kw=find_peaks_kw)  # type: ignore
    else:
        find_kw = dict(sharpening=None)
        find_kw.update(find_peaks_kw)
        spe_pos_dict = old_spe.find_peak_multipeak(**find_kw).get_pos_ampl_dict()  # type: ignore
    spe_cent = np.array(list(spe_pos_dict.keys()))

    spe_idx, ref_idx = rc2utils.find_closest_pairs_idx(spe_cent, ref_pos)
    if len(ref_idx) == 1:
        _offset = (ref_pos[ref_idx][0] - spe_cent[spe_idx][0])
        new_spe.x = old_spe.x + _offset
    else:
        kwargs["kernel"] = kernel
        interp = interpolate.RBFInterpolator(spe_cent[spe_idx].reshape(-1, 1), ref_pos[ref_idx], **kwargs)
        new_spe.x = interp(old_spe.x.reshape(-1, 1))
