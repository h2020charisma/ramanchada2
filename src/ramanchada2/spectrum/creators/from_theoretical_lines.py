#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
from lmfit import Model
from lmfit import Parameters
from typing import Union
from pydantic import validate_arguments

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor


@add_spectrum_constructor()
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def from_theoretical_lines(
        model: Model,
        params: Parameters,
        x: Union[int, npt.NDArray[np.float64]] = 2000):
    """
    Generate spectrum from `lmfit` model.

    Parameters
    ----------
    model : lmfit.Model
        the model to be used for spectrum generation
    params : lmfit.Parameters
        the parameters to be applied to the model
    x : Union[int, npt.NDArray[np.float64]], optional
        array with x values, by default np.array(2000)
    """
    spe = Spectrum(x=x)
    spe.y = model.eval(params=params, x=spe.x)
    return spe
