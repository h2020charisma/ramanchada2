#!/usr/bin/env python3

import numpy as np

from pydantic import validate_arguments
from ramanchada2.misc.spectrum_deco import add_spectrum_method
from ..spectrum import Spectrum


def abs_nm_to_shift_cm_1(deltas, laser_wave_length_nm):
    arr = np.array(list(deltas.items()), dtype=float)
    arr[:, 0] = 1e-2*(1/laser_wave_length_nm - 1/arr[:, 0])
    return dict(arr)


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def x_absolute_nm(spe: Spectrum, /,
                  laser_wave_length_nm: float):
    shift_nm = spe.x * 1e-7
    absolute_nm = 1/(1/laser_wave_length_nm - shift_nm)
    return absolute_nm
