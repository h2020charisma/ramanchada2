#!/usr/bin/env python3

from typing import Callable, Union

import numpy as np
import numpy.typing as npt
from pydantic import validate_arguments

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def scale_xaxis_linear(old_spe: Spectrum,
                       new_spe: Spectrum, /,
                       factor: float = 1,
                       preserve_integral: bool = False):
    new_spe.x = old_spe.x * factor
    if preserve_integral:
        new_spe.y = old_spe.y / factor


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def scale_xaxis_fun(old_spe: Spectrum,
                    new_spe: Spectrum, /,
                    fun: Callable[[Union[int, npt.NDArray]], float],
                    args=[]):
    new_spe.x = fun(old_spe.x, *args)
    if (np.diff(new_spe.x) < 0).any():
        raise ValueError('The provided function is not a monoton increasing funciton.')
