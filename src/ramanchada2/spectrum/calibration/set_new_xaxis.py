#!/usr/bin/env python3

import numpy.typing as npt
from pydantic import validate_arguments

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def set_new_xaxis(old_spe: Spectrum,
                  new_spe: Spectrum, /,
                  xaxis: npt.NDArray):
    if old_spe.x.shape != xaxis.shape:
        raise ValueError('Shape of xaxis should match the shape of xaxis of the spectrum')
    new_spe.x = xaxis
