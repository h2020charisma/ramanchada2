#!/usr/bin/env python3

from typing import Union, Tuple, List
from scipy import signal
from pydantic import validate_arguments, PositiveFloat

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import (add_spectrum_method,
                                            add_spectrum_filter)
from ramanchada2.misc.types import PeakCandidatesListModel


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_peaks(
        spe: Spectrum, /,
        prominence: float = 1e-2,
        wlen=None,
        width: Union[int, Tuple[int, int]] = 1
        ) -> PeakCandidatesListModel:
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
    return PeakCandidatesListModel.from_find_peaks(res)


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_peak_groups(
        spe: Spectrum, /,
        prominence: float = 1e-2,
        wlen=None,
        width: Union[int, Tuple[int, int]] = 1,
        n_sigma_group: PositiveFloat = 5.
        ) -> List[PeakCandidatesListModel]:
    res = signal.find_peaks(spe.y, prominence=prominence, width=width, wlen=wlen)
    return PeakCandidatesListModel.from_find_peaks(res).group_neighbours(n_sigma=n_sigma_group)


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_peaks_filter(
        old_spe: Spectrum,
        new_spe: Spectrum, /,
        *args, **kwargs):
    res = old_spe.find_peaks(*args, **kwargs)  # type: ignore
    new_spe.result = {k: v.tolist() for k, v in res.items()}
