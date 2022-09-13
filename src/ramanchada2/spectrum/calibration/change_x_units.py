#!/usr/bin/env python3

from pydantic import validate_arguments
from ramanchada2.misc.spectrum_deco import add_spectrum_method
from ramanchada2.misc.utils.ramanshift_to_wavelength import shift_cm_1_to_abs_nm
from ..spectrum import Spectrum


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def x_absolute_nm(spe: Spectrum, /,
                  laser_wave_length_nm: float):
    return shift_cm_1_to_abs_nm(spe.x, laser_wave_length_nm=laser_wave_length_nm)
