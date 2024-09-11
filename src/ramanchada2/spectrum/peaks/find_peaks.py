from typing import List, Literal, Tuple, Union

import numpy as np
from pydantic import (NonNegativeFloat, NonNegativeInt, PositiveInt,
                      validate_call)
from scipy import signal
from scipy.signal import find_peaks_cwt

from ramanchada2.misc.spectrum_deco import (add_spectrum_filter,
                                            add_spectrum_method)
from ramanchada2.misc.types.peak_candidates import ListPeakCandidateMultiModel

from ..spectrum import Spectrum


def peak_boundaries(spe, wlen, width, prominence):
    peaks = signal.find_peaks(spe.y, prominence=prominence, width=width, wlen=wlen)
    larr = peaks[1]['left_bases'][:]
    rarr = peaks[1]['right_bases'][:]
    lb = 0
    lbounds = list()
    rbounds = list()
    while len(larr):
        lbargmin = np.argmin(larr)
        lb = larr[lbargmin]
        rb = rarr[lbargmin]
        while True:
            group_bool = larr < rb
            if group_bool.any():
                rb = np.max(rarr[group_bool])
                rarr = rarr[~group_bool]
                larr = larr[~group_bool]
                continue
            break
        lbounds.append(lb)
        rbounds.append(rb)
    return np.array(list(zip(lbounds, rbounds)))


@add_spectrum_method
@validate_call(config=dict(arbitrary_types_allowed=True))
def find_peak_multipeak(
        spe: Spectrum, /,
        prominence: Union[NonNegativeFloat, None] = None,
        wlen: Union[NonNegativeInt, None] = None,
        width: Union[int, Tuple[int, int], None] = None,
        hht_chain: Union[List[PositiveInt], None] = None,
        bgm_kwargs={},
        sharpening: Union[Literal['hht'], None] = None,
        strategy: Literal['topo', 'bayesian_gaussian_mixture', 'bgm', 'cwt'] = 'topo'
        ) -> ListPeakCandidateMultiModel:
    """
    Find groups of peaks in spectrum.

    Args:
        spe: internal use only
        prominence: Optional. Defaults to None
            If None the prominence value will be `spe.y_nose`. Reasonable value for
            promience is `const * spe.y_noise_MAD`.
        wlen: optional. Defaults to None.
            wlen value used in `scipy.signal.find_peaks`. If wlen is None, 200 will be used.
        width: optional. Defaults to None.
            width value used in `scipy.signal.find_peaks`. If width is None, 2 will be used.
        hht_chain: optional. Defaults to None.
            List of hht_chain window sizes. If None, no hht sharpening is performed.
        bgm_kwargs: kwargs for bayesian_gaussian_mixture
        sharpening 'hht' or None. Defaults to None.
            If 'hht' hht sharpening will be performed before finding peaks.
        strategy: optional. Defauts to 'topo'.
            Peakfinding method

    Returns:
        ListPeakCandidateMultiModel: Located peak groups
    """

    if prominence is None:
        prominence = spe.y_noise
    if not wlen:
        wlen = 200
    if width is None:
        width = 2

    if sharpening == 'hht':
        if hht_chain is None:
            hht_chain = [20]
        sharp_spe = spe.hht_sharpening_chain(movmin=hht_chain)  # type: ignore
    else:
        sharp_spe = spe

    x_arr = sharp_spe.x
    y_arr = sharp_spe.y

    def interpolate(x):
        x1 = int(x)
        x2 = x1 + 1
        y1 = x_arr[x1]
        y2 = x_arr[x2]
        return (y2-y1)/(x2-x1)*(x-x1)+y1

    boundaries = peak_boundaries(spe, prominence=prominence, width=width, wlen=wlen)
    boundaries = [(li, ri) for li, ri in boundaries if (ri-li) > 4]

    peaks, props = signal.find_peaks(y_arr,
                                     prominence=prominence,
                                     width=width,
                                     wlen=wlen)
    peak_groups = list()

    if strategy in {'bgm', 'bayesian_gaussian_mixture'}:
        bgm = sharp_spe.bayesian_gaussian_mixture(**bgm_kwargs)

        bgm_peaks = [[mean[0], np.sqrt(cov[0][0]), weight]
                     for mean, cov, weight in
                     zip(bgm.means_, bgm.covariances_, bgm.weights_)]
        bgm_peaks = sorted(bgm_peaks, key=lambda x: x[2], reverse=True)
        integral = np.sum(y_arr)
        n_peaks = (np.round(bgm.weights_, 2) > 0).sum()
        bgm_peaks = bgm_peaks[:n_peaks]

        peak_list = list()
        for mean, sigma, weight in bgm_peaks:
            peak_list.append(dict(amplitude=weight*integral*2/sigma,
                                  position=mean,
                                  sigma=sigma,
                                  ))
        for li, ri in boundaries:
            peak_group = list()
            for peak in peak_list:
                if li < peak['position'] < ri:
                    peak_group.append(dict(position=peak['position'],
                                           amplitude=peak['amplitude'],
                                           sigma=peak['sigma'])
                                      )
            if peak_group:
                peak_groups.append(dict(boundaries=(x_arr[li], x_arr[ri]),
                                        peaks=peak_group))
    elif strategy == 'cwt':
        # TODO: cwt_args tbd
        peaks = find_peaks_cwt(spe.y, **bgm_kwargs)
        peak_list = list()
        for peak_index in peaks:
            half_max = spe.y[peak_index] / 2.0
            left_index = np.where(spe.y[:peak_index] <= half_max)[0][-1]
            right_index = np.where(spe.y[peak_index:] <= half_max)[0][0] + peak_index
            fwhm = spe.x[right_index] - spe.x[left_index]
            # rough sigma estimation based on fwhm
            sqrt2ln2 = 2 * np.sqrt(2 * np.log(2))
            # print(spe.x[peak_index], spe.y[peak_index], fwhm / sqrt2ln2 )
            peak_list.append(dict(amplitude=spe.y[peak_index],
                                  position=spe.x[peak_index],
                                  sigma=fwhm / sqrt2ln2,
                                  fwhm=fwhm))
        for li, ri in boundaries:
            peak_group = list()
            for peak in peak_list:
                if li < peak['position'] < ri:
                    peak_group.append(dict(position=peak['position'],
                                           amplitude=peak['amplitude'],
                                           sigma=peak['sigma']))
            if peak_group:
                peak_groups.append(dict(boundaries=(x_arr[li], x_arr[ri]),
                                        peaks=peak_group))
    elif strategy == 'topo':
        for li, ri in boundaries:
            peak_group = list()
            x1 = spe.x[li]
            x2 = spe.x[ri]
            y1 = spe.y[li]
            y2 = spe.y[ri]
            slope = (y2-y1)/(x2-x1)
            intercept = -slope*x1+y1
            for peak_i, peak_pos in enumerate(peaks):
                if li < peak_pos < ri:
                    pos_maximum = x_arr[peak_pos]
                    amplitude = props['prominences'][peak_i]
                    lwhm = pos_maximum - interpolate(props['left_ips'][peak_i])
                    rwhm = interpolate(props['right_ips'][peak_i]) - pos_maximum
                    fwhm = lwhm + rwhm
                    sigma = fwhm/2.355
                    skew = (rwhm-lwhm)/(rwhm+lwhm)
                    peak_group.append(dict(position=pos_maximum,
                                           amplitude=amplitude,
                                           sigma=sigma,
                                           skew=skew)
                                      )
            if peak_group:
                peak_groups.append(dict(base_intercept=intercept,
                                        base_slope=slope,
                                        boundaries=(x_arr[li], x_arr[ri]),
                                        peaks=peak_group))

    candidates = ListPeakCandidateMultiModel.model_validate(peak_groups)
    return candidates


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def find_peak_multipeak_filter(
        old_spe: Spectrum,
        new_spe: Spectrum, /,
        *args, **kwargs):
    """
    Same as `find_peak_multipeak` but the result is stored as metadata in the returned spectrum.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        *args, **kwargs: same as `find_peak_multipeak`
    """
    res = old_spe.find_peak_multipeak(*args, **kwargs)  # type: ignore
    new_spe.result = res.model_dump()
