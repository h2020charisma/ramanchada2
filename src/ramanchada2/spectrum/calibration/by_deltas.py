from typing import Dict, List, Literal, Optional, Union

import lmfit
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, NonNegativeInt, PositiveInt, validate_call
from scipy import fft, interpolate, signal

from ramanchada2.misc.spectrum_deco import (add_spectrum_filter,
                                            add_spectrum_method)

from ...misc import utils as rc2utils
from ...misc.utils.rough_poly2_calibration import rough_poly2_calibration
from ..spectrum import Spectrum


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
        spe_conv = spe.convolve('gaussian', sigma=sigma/np.mean(np.diff(spe.x)))
        if no_fit:
            fit_res = self.model.fit(spe_conv.y, x=spe_conv.x, params=self.params, max_nfev=-1)
        else:
            fit_res = self.model.fit(spe_conv.y, x=spe_conv.x, params=self.params)
        self.params = fit_res.params
        if ax is not None:
            spe_conv.plot(ax=ax)
            ax.plot(spe_conv.x, fit_res.eval(x=spe_conv.x), 'r')


class ParamBounds(BaseModel):
    min: float = -np.inf
    max: float = np.inf


class FitBounds(BaseModel):
    shift: ParamBounds = ParamBounds(min=-np.inf, max=np.inf)
    scale: ParamBounds = ParamBounds(min=-np.inf, max=np.inf)
    scale2: ParamBounds = ParamBounds(min=-np.inf, max=np.inf)
    scale3: ParamBounds = ParamBounds(min=-np.inf, max=np.inf)


@add_spectrum_method
@validate_call(config=dict(arbitrary_types_allowed=True))
def calibrate_by_deltas_model(spe: Spectrum, /,
                              deltas: Dict[float, float],
                              bounds: Optional[FitBounds] = None,
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
    if bounds is not None:
        mod.params['scale'].set(value=scale, min=bounds.scale.min, max=bounds.scale.max)
        mod.params['shift'].set(value=shift, min=bounds.shift.min, max=bounds.shift.max)
    else:
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
        if bounds is not None:
            mod.params['scale2'].set(vary=True, value=0, min=bounds.scale2.min, max=bounds.scale2.max)
        else:
            mod.params['scale2'].set(vary=True, value=0)
        mod.fit(spe_padded, sigma=0.05, ax=ax, **kwargs)
    if scale3:
        if bounds is not None:
            mod.params['scale2'].set(vary=True, value=0, min=bounds.scale2.min, max=bounds.scale2.max)
            mod.params['scale3'].set(vary=True, value=0, min=bounds.scale3.min, max=bounds.scale3.max)
        else:
            mod.params['scale2'].set(vary=True, value=0)
            mod.params['scale3'].set(vary=True, value=0)
        mod.fit(spe_padded, sigma=0.05, ax=ax, **kwargs)
    return mod.model, mod.params


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
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
@validate_call(config=dict(arbitrary_types_allowed=True))
def xcal_fine(old_spe: Spectrum,
              new_spe: Spectrum, /, *,
              ref: Union[Dict[float, float], List[float]],
              should_fit=False,
              poly_order: NonNegativeInt = 2,
              max_iter: NonNegativeInt = 1000,
              find_peaks_kw={},
              ):
    """
    Iterative calibration with provided reference based on :func:`~ramanchada2.misc.utils.argmin2d.align`

    Iteratively apply polynomial of `poly_order` degree to match
    the found peaks to the reference locations. The pairs are created
    using :func:`~ramanchada2.misc.utils.argmin2d.align` algorithm.

    Args:
        old_spe (Spectrum): internal use only
        new_spe (Spectrum): internal use only
        ref (Union[Dict[float, float], List[float]]): _description_
        ref (Dict[float, float]):
            If a dict is provided - wavenumber - amplitude pairs.
            If a list is provided - wavenumbers only.
        poly_order (NonNegativeInt): polynomial degree to be used usualy 2 or 3
        max_iter (NonNegativeInt): max number of iterations for the iterative
            polynomial alignment
        should_fit (bool, optional): Whether the peaks should be fit or to
            associate the positions with the maxima. Defaults to False.
        find_peaks_kw (dict, optional): kwargs to be used in find_peaks. Defaults to {}.
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

    if poly_order == 0:
        p = rc2utils.align_shift(spe_cent, ref_pos)
        spe_cal = old_spe.scale_xaxis_fun(lambda x: x + p)  # type: ignore
    else:
        def cal_func(x, *a):
            return [par*(x/1000)**power for power, par in enumerate(a)]

        p0 = np.resize([0, 1000, 0], poly_order + 1)
        p = rc2utils.align(spe_cent, ref_pos, p0=p0, func=cal_func, max_iter=max_iter)
        spe_cal = old_spe.scale_xaxis_fun(  # type: ignore
            (lambda x, *args: np.sum(cal_func(x, *args), axis=0)), args=p)
    new_spe.x = spe_cal.x


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
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


def semi_spe_from_dict(deltas: dict, xaxis):
    y = np.zeros_like(xaxis)
    for pos, ampl in deltas.items():
        idx = np.argmin(np.abs(xaxis - pos))
        y[idx] += ampl
    # remove overflows and underflows
    y[0] = 0
    y[-1] = 0
    return y


def low_pass(x, nbin, window=signal.windows.blackmanharris):
    h = window(nbin*2-1)[nbin-1:]
    X = fft.rfft(x)
    X[:nbin] *= h  # apply the window
    X[nbin:] = 0  # clear upper frequencies
    return fft.irfft(X, n=len(x))


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def xcal_argmin2d_iter_lowpass(old_spe: Spectrum,
                               new_spe: Spectrum, /, *,
                               ref: Dict[float, float],
                               low_pass_nfreqs: List[int] = [100, 500]):
    """
    Calibrate spectrum

    The calibration is done in multiple steps. Both the spectrum and the reference
    are passed through a low-pass filter to preserve only general structure of the
    spectrum. `low_pass_nfreqs` defines the number of frequencies to be preserved in
    each step. Once all steps with low-pass filter a final step without a low-pass
    filter is performed. Each calibration step is performed using
    :func:`~ramanchada2.spectrum.calibration.by_deltas.xcal_fine` algorithm.

    Args:
        old_spe (Spectrum): internal use only
        new_spe (Spectrum): internal use only
        ref (Dict[float, float]): wavenumber - amplitude pairs
        low_pass_nfreqs (List[int], optional): The number of elements defines the
            number of low-pass steps and their values define the amount of frequencies
            to keep. Defaults to [100, 500].
    """

    spe = old_spe.__copy__()
    for low_pass_i in low_pass_nfreqs:
        xaxis = spe.x
        y_ref_semi_spe = semi_spe_from_dict(ref, spe.x)
        y_ref_semi_spe = low_pass(y_ref_semi_spe, low_pass_i)

        r = xaxis[signal.find_peaks(y_ref_semi_spe,
                                    prominence=np.max(y_ref_semi_spe)*.001
                                    )[0]]

        spe_low = spe.__copy__()
        spe_low.y = low_pass(spe.y, low_pass_i)

        spe_cal = spe_low.xcal_fine(ref=r, should_fit=False, poly_order=2,
                                    # disables wlen.
                                    # low passed peaks are much broader and
                                    # will be affected by `wlen` so disable.
                                    find_peaks_kw={'wlen': int(xaxis[-1]-xaxis[0])},
                                    )
        spe.x = spe_cal.x
    spe_cal_fin = spe.xcal_fine(ref=ref, should_fit=False, poly_order=2)
    new_spe.x = spe_cal_fin.x


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def xcal_rough_poly2_all_pairs(old_spe: Spectrum,
                               new_spe: Spectrum, /, *,
                               ref: Dict[float, float],
                               prominence: Optional[float] = None,
                               npeaks: PositiveInt = 10,
                               **kwargs,
                               ):
    if prominence is None:
        prominence = old_spe.y_noise_MAD()*5
    cand = old_spe.find_peak_multipeak(prominence=prominence)  # type: ignore
    spe_dict = cand.get_pos_ampl_dict()

    a, b, c = rough_poly2_calibration(spe_dict, ref, npeaks=npeaks, **kwargs)
    new_spe.x = a*old_spe.x**2 + b*old_spe.x + c
