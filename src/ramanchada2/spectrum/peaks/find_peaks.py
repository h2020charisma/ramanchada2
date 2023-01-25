#!/usr/bin/env python3

from typing import Union, Tuple, List, Literal
from scipy import signal
import numpy as np
from pydantic import validate_arguments, PositiveFloat, PositiveInt

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import (add_spectrum_method,
                                            add_spectrum_filter)

from ramanchada2.misc.types.peak_candidates import ListPeakCandidateMultiModel


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
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_peak_multipeak(
        spe: Spectrum, /,
        prominence: Union[PositiveFloat, None] = None,
        wlen: Union[PositiveInt, None] = None,
        width: Union[int, Tuple[int, int], None] = None,
        hht_chain: Union[List[PositiveInt], None] = None,
        bgm_kwargs={},
        sharpening: Union[Literal['hht'], None] = 'hht',
        strategy: Literal['topo', 'bayesian_gaussian_mixture', 'bgm'] = 'topo'
        ) -> ListPeakCandidateMultiModel:

    if prominence is None:
        prominence = spe.y_noise*15
    if wlen is None:
        wlen = 50
    if width is None:
        width = 1

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
                                     width=1,
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
                    amplitude = y_arr[peak_pos]
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

    candidates = ListPeakCandidateMultiModel.validate(peak_groups)
    return candidates


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_peak_multipeak_filter(
        old_spe: Spectrum,
        new_spe: Spectrum, /,
        *args, **kwargs):
    res = old_spe.find_peak_multipeak(*args, **kwargs)  # type: ignore
    new_spe.result = res.dict()['__root__']
