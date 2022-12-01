#!/usr/bin/env python3

from typing import Union, Tuple, List, Dict
from scipy import signal
import numpy as np
from pydantic import validate_arguments, PositiveFloat, PositiveInt

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import (add_spectrum_method,
                                            add_spectrum_filter)
from ramanchada2.misc.types import PeakCandidatesGroupModel
from ramanchada2.misc.types.peak_candidates import ListPeakCandidateMultiModel


def peak_boundaries(spe, wlen=50, width=1):
    peaks = signal.find_peaks(spe.y, prominence=spe.y_noise*10, width=width, wlen=wlen)
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
def find_peak_hht_groups(
        spe: Spectrum, /,
        prominence: float = None,
        wlen=None,
        width: Union[int, Tuple[int, int]] = 1,
        n_sigma_group: PositiveFloat = 5.,
        hht_chain=None
        ) -> List[PeakCandidatesGroupModel]:
    spe = spe.normalize()  # type: ignore
    if hht_chain is not None:
        sharp_spe = spe.hht_sharpening_chain(movmin=hht_chain)  # type: ignore
    else:
        sharp_spe = spe.hht_sharpening_chain()  # type: ignore
    boundaries = peak_boundaries(spe, wlen=wlen)
    sharp_peaks = signal.find_peaks(
        sharp_spe.y,
        prominence=(prominence if prominence is not None else spe.y_noise*10),
        width=1,
        wlen=wlen)
    peak_groups = list()
    for li, ri in boundaries:
        group = list()
        x1 = spe.x[li]
        x2 = spe.x[ri]
        y1 = spe.y[li]
        y2 = spe.y[ri]
        slope = (y2-y1)/(x2-x1)
        intercept = -slope*x1+y1
        for peak_i, peak_pos in enumerate(sharp_peaks[0]):
            if li < peak_pos < ri:
                group.append(dict(position=sharp_spe.x[peak_pos],
                                  amplitude=sharp_spe.y[peak_pos],
                                  sigma=sharp_peaks[1]['widths'][peak_i]*2.355)
                             )
        if group:
            peak_groups.append(dict(base_intercept=intercept,
                                    base_slope=slope,
                                    boundaries=(spe.x[li], spe.x[ri]),
                                    peaks=group))

    candidates = ListPeakCandidateMultiModel.validate(peak_groups)
    return candidates


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_peaks_hht_filter(
        old_spe: Spectrum,
        new_spe: Spectrum, /,
        *args, **kwargs):
    res = old_spe.find_peak_hht_groups(*args, **kwargs)  # type: ignore
    new_spe.result = res.dict()['__root__']
