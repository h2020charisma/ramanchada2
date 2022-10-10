#!/usr/bin/env python3

from pydantic import validate_arguments
from ramanchada2.misc.spectrum_deco import add_spectrum_method, add_spectrum_filter
from ramanchada2.misc.utils.ramanshift_to_wavelength import shift_cm_1_to_abs_nm as util_shift_cm_1_to_abs_nm
from ramanchada2.misc.utils.ramanshift_to_wavelength import abs_nm_to_shift_cm_1 as util_abs_nm_to_shift_cm_1

from ..spectrum import Spectrum


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def abs_nm_to_shift_cm_1(spe: Spectrum, /,
                         laser_wave_length_nm: float):
    return util_abs_nm_to_shift_cm_1(spe.x, laser_wave_length_nm=laser_wave_length_nm)


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def shift_cm_1_to_abs_nm(spe: Spectrum, /,
                         laser_wave_length_nm: float):
    return util_shift_cm_1_to_abs_nm(spe.x, laser_wave_length_nm=laser_wave_length_nm)


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def abs_nm_to_shift_cm_1_filter(old_spe: Spectrum,
                                new_spe: Spectrum, /,
                                laser_wave_length_nm: float):
    new_spe.x = util_abs_nm_to_shift_cm_1(old_spe.x, laser_wave_length_nm=laser_wave_length_nm)


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def shift_cm_1_to_abs_nm_filter(old_spe: Spectrum,
                                new_spe: Spectrum, /,
                                laser_wave_length_nm: float):
    new_spe.x = util_shift_cm_1_to_abs_nm(old_spe.x, laser_wave_length_nm=laser_wave_length_nm)
