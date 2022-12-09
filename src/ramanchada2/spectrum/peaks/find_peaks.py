#!/usr/bin/env python3

from typing import Union, Tuple, List, Dict
from scipy import signal
from pydantic import validate_arguments, PositiveFloat, PositiveInt

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import (add_spectrum_method,
                                            add_spectrum_filter)
from ramanchada2.misc.types import PeakCandidatesGroupModel


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_peaks(
        spe: Spectrum, /,
        prominence: float = 1e-2,
        wlen=None,
        width: Union[int, Tuple[int, int]] = 1
        ) -> PeakCandidatesGroupModel:
    """
    Find peaks in spectrum.

    Parameters
    ----------
    prominence : float, optional
        the minimal net amplitude for a peak to be considered, by default 1e-2
    width : int, optional
        the minimal width of the peaks, by default 1

    Returns
    -------
    _type_
        _description_
    """
    res = signal.find_peaks(spe.y, prominence=prominence, width=width, wlen=wlen)
    return PeakCandidatesGroupModel.from_find_peaks(res, x_arr=spe.x, y_arr=spe.y)


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_peak_groups(
        spe: Spectrum, /,
        prominence: float = 1e-2,
        wlen=None,
        width: Union[int, Tuple[int, int]] = 1,
        n_sigma_group: PositiveFloat = 5.,
        moving_minimum_window: Union[PositiveInt, None] = None,
        kw_derivative_sharpening: Union[Dict, None] = None,
        ) -> List[PeakCandidatesGroupModel]:
    if moving_minimum_window is not None:
        spe = spe.subtract_moving_minimum(moving_minimum_window)  # type: ignore
    spe = spe.normalize()  # type: ignore
    if kw_derivative_sharpening is not None:
        spe = spe.derivative_sharpening(**kw_derivative_sharpening)  # type: ignore
    res = signal.find_peaks(spe.y, prominence=prominence, width=width, wlen=wlen)
    return PeakCandidatesGroupModel.from_find_peaks(res, x_arr=spe.x, y_arr=spe.y
                                                    ).group_neighbours(n_sigma=n_sigma_group)


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_peaks_filter(
        old_spe: Spectrum,
        new_spe: Spectrum, /,
        *args, **kwargs):
    res = old_spe.find_peak_groups(*args, **kwargs)  # type: ignore
    new_spe.result = res.dict()['__root__']
