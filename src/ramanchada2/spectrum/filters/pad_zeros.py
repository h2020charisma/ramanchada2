#!/usr/bin/env python3

import numpy as np
from pydantic import validate_arguments

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pad_zeros(old_spe: Spectrum,
              new_spe: Spectrum, /):
    lenx = len(old_spe.x)
    minx = np.min(old_spe.x)
    maxx = np.max(old_spe.x)
    xl = np.linspace(minx-(maxx-minx), minx, lenx, endpoint=True)[:-1]
    xr = np.linspace(maxx, maxx+(maxx-minx), lenx, endpoint=True)[1:]

    new_spe.y = np.concatenate((np.zeros(lenx-1), old_spe.y, np.zeros(lenx-1)))
    new_spe.x = np.concatenate((xl, old_spe.x, xr))
